"""Mandatory leak-safe point-in-time baselines for Model v2."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd


TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
ROLLING_WINDOWS = (5, 10, 20)


def add_lagged_baselines(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...] = TARGETS,
    player_column: str = "PLAYER_ID",
    date_column: str = "GAME_DATE",
    game_column: str = "GAME_ID",
    minutes_column: str = "MIN",
) -> pd.DataFrame:
    """Add every declared baseline using outcomes strictly before each row.

    Same-day role/position priors are calculated in date buckets, so one game
    on a slate cannot leak into another game on that slate. Player baselines
    use a one-row lag after rejecting duplicate player-game rows.
    """

    required = {player_column, date_column, *targets}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Baseline frame missing columns: {sorted(missing)}")
    if game_column in frame.columns and frame.duplicated(
        [game_column, player_column]
    ).any():
        raise ValueError("Baseline frame contains duplicate player-game rows")

    result = frame.copy()
    result["__BASELINE_ORDER"] = np.arange(len(result))
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    sort_columns = [player_column, date_column]
    if game_column in result.columns:
        sort_columns.append(game_column)
    result = result.sort_values(sort_columns, kind="stable").copy()
    season = _season_key(result, date_column)

    role_column = _first_present(result, ("ROLE", "ROLE_BUCKET", "ROTATION_ROLE"))
    position_column = _first_present(result, ("POSITION", "POS"))
    if role_column is None and "STARTER" in result.columns:
        result["__BASELINE_ROLE"] = np.where(
            _as_bool(result["STARTER"]), "starter", "bench"
        )
        role_column = "__BASELINE_ROLE"

    for target in targets:
        values = pd.to_numeric(result[target], errors="coerce")
        player_groups = result[player_column]
        result[f"BASELINE_LAST_{target}"] = values.groupby(
            player_groups, sort=False
        ).shift(1)
        for window in ROLLING_WINDOWS:
            result[f"BASELINE_ROLL_{window}_{target}"] = values.groupby(
                player_groups, sort=False
            ).transform(
                lambda series, size=window: series.shift(1).rolling(
                    size, min_periods=1
                ).mean()
            )
        result[f"BASELINE_EMA_10_{target}"] = values.groupby(
            player_groups, sort=False
        ).transform(
            lambda series: series.shift(1).ewm(
                span=10, adjust=False, min_periods=1
            ).mean()
        )
        result[f"BASELINE_SEASON_{target}"] = values.groupby(
            [player_groups, season], sort=False
        ).transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )

        league_prior = _prior_bucket_mean(
            result.assign(__VALUE=values),
            value_column="__VALUE",
            group_columns=(),
            date_column=date_column,
        )
        role_prior = (
            _prior_bucket_mean(
                result.assign(__VALUE=values),
                value_column="__VALUE",
                group_columns=(role_column,),
                date_column=date_column,
            )
            if role_column
            else pd.Series(np.nan, index=result.index)
        )
        position_prior = (
            _prior_bucket_mean(
                result.assign(__VALUE=values),
                value_column="__VALUE",
                group_columns=(position_column,),
                date_column=date_column,
            )
            if position_column
            else pd.Series(np.nan, index=result.index)
        )
        result[f"BASELINE_ROLE_{target}"] = role_prior
        result[f"BASELINE_POSITION_{target}"] = position_prior
        result[f"BASELINE_ROLE_POSITION_{target}"] = (
            role_prior.combine_first(position_prior).combine_first(league_prior)
        )

        if minutes_column in result.columns:
            minutes = pd.to_numeric(result[minutes_column], errors="coerce")
            rate = values.div(minutes.where(minutes > 0))
            for window in ROLLING_WINDOWS:
                prior_minutes = minutes.groupby(
                    player_groups, sort=False
                ).transform(
                    lambda series, size=window: series.shift(1).rolling(
                        size, min_periods=1
                    ).mean()
                )
                prior_rate = rate.groupby(player_groups, sort=False).transform(
                    lambda series, size=window: series.shift(1).rolling(
                        size, min_periods=1
                    ).mean()
                )
                result[f"BASELINE_MINUTES_X_RATE_{window}_{target}"] = (
                    prior_minutes * prior_rate
                ).clip(lower=0)
        else:
            for window in ROLLING_WINDOWS:
                result[f"BASELINE_MINUTES_X_RATE_{window}_{target}"] = np.nan

    result = result.sort_values("__BASELINE_ORDER", kind="stable")
    result = result.drop(
        columns=["__BASELINE_ORDER", "__BASELINE_ROLE"], errors="ignore"
    )
    return result


def build_point_in_time_baselines(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    forecast_time: date | datetime | str,
    targets: tuple[str, ...] = TARGETS,
) -> pd.DataFrame:
    """Forecast mandatory baselines for a scheduled candidate universe."""

    required = {"PLAYER_ID"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"Candidate frame missing columns: {sorted(missing)}")
    history_frame = history.copy()
    if "GAME_DATE" not in history_frame.columns:
        raise ValueError("History must contain GAME_DATE")
    cutoff = pd.Timestamp(forecast_time)
    history_frame["GAME_DATE"] = pd.to_datetime(
        history_frame["GAME_DATE"], errors="raise"
    )
    history_frame = history_frame.loc[history_frame["GAME_DATE"] < cutoff].copy()
    for target in targets:
        if target not in history_frame.columns:
            raise ValueError(f"History must contain target {target}")

    scheduled = candidates.copy()
    scheduled["GAME_DATE"] = cutoff
    if "GAME_ID" not in scheduled.columns:
        scheduled["GAME_ID"] = "__scheduled__"
    scheduled["__IS_BASELINE_CANDIDATE"] = True
    history_frame["__IS_BASELINE_CANDIDATE"] = False
    for target in targets:
        scheduled[target] = np.nan
    scheduled["MIN"] = np.nan

    combined = pd.concat([history_frame, scheduled], ignore_index=True, sort=False)
    enriched = add_lagged_baselines(combined, targets=targets)
    output = enriched.loc[enriched["__IS_BASELINE_CANDIDATE"].fillna(False)].copy()
    return output.drop(columns="__IS_BASELINE_CANDIDATE").reset_index(drop=True)


def baseline_long_frame(
    frame: pd.DataFrame,
    *,
    targets: Iterable[str] = TARGETS,
    identity_columns: Iterable[str] = (
        "REQUEST_ID",
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "TEAM_ID",
        "OPPONENT_ID",
        "HORIZON",
        "FORECAST_CUTOFF",
        "MODEL_BUNDLE_ID",
        "SOURCE_SNAPSHOT_ID",
    ),
) -> pd.DataFrame:
    """Convert scheduled wide baseline columns to auditable long rows."""

    identities = [column for column in identity_columns if column in frame.columns]
    rows: list[pd.DataFrame] = []
    for target in tuple(targets):
        suffix = f"_{target}"
        columns = [
            column
            for column in frame.columns
            if column.startswith("BASELINE_") and column.endswith(suffix)
        ]
        for column in columns:
            block = frame[identities].copy()
            block["STAT"] = target
            block["BASELINE"] = column[: -len(suffix)].removeprefix("BASELINE_")
            block["PREDICTION"] = pd.to_numeric(frame[column], errors="coerce")
            rows.append(block)
    if not rows:
        return pd.DataFrame(
            columns=[*identities, "STAT", "BASELINE", "PREDICTION"]
        )
    return pd.concat(rows, ignore_index=True, sort=False)


def _prior_bucket_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: tuple[str, ...],
    date_column: str,
) -> pd.Series:
    """Mean before the current time bucket, mapped back to source rows."""

    work = frame[[*group_columns, date_column, value_column]].copy()
    work["__ROW_INDEX"] = frame.index
    keys = [*group_columns, date_column]
    bucket = work.groupby(keys, dropna=False, sort=True)[value_column].agg(
        __SUM="sum", __COUNT="count"
    ).reset_index()
    bucket = bucket.sort_values(keys, kind="stable")
    if group_columns:
        groups = [bucket[column] for column in group_columns]
        bucket["__PRIOR_SUM"] = bucket["__SUM"].groupby(
            groups, dropna=False, sort=False
        ).cumsum() - bucket["__SUM"]
        bucket["__PRIOR_COUNT"] = bucket["__COUNT"].groupby(
            groups, dropna=False, sort=False
        ).cumsum() - bucket["__COUNT"]
    else:
        bucket["__PRIOR_SUM"] = bucket["__SUM"].cumsum() - bucket["__SUM"]
        bucket["__PRIOR_COUNT"] = bucket["__COUNT"].cumsum() - bucket["__COUNT"]
    bucket["__PRIOR_MEAN"] = bucket["__PRIOR_SUM"].div(
        bucket["__PRIOR_COUNT"].replace(0, np.nan)
    )
    mapped = work.merge(bucket[keys + ["__PRIOR_MEAN"]], on=keys, how="left")
    return mapped.set_index("__ROW_INDEX")["__PRIOR_MEAN"].reindex(frame.index)


def _season_key(frame: pd.DataFrame, date_column: str) -> pd.Series:
    for column in ("SEASON_ID", "SEASON"):
        if column in frame.columns:
            return frame[column].astype("string").fillna("unknown")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    start_year = np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)
    return pd.Series(start_year, index=frame.index, dtype="int64")


def _first_present(frame: pd.DataFrame, columns: Iterable[str]) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().isin({"1", "true", "yes", "starter"})
