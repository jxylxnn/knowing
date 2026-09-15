"""Scoring for immutable Model v2 forecast rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.contracts.forecast import validate_forecast_frame


@dataclass(frozen=True)
class ReplayScore:
    rows: int
    games: int
    reconciled_fraction: float
    targets: dict[str, dict[str, float | int | None]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_replay(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> ReplayScore:
    """Score canonical long forecasts against final wide player box scores."""

    validate_forecast_frame(predictions)
    required_actuals = {"GAME_ID", "PLAYER_ID"}
    if missing := required_actuals - set(actuals.columns):
        raise ValueError(f"Actuals missing columns: {sorted(missing)}")
    key = ["REQUEST_ID", "GAME_ID", "PLAYER_ID", "STAT"]
    if predictions.duplicated(key).any():
        raise ValueError("Replay predictions contain duplicate request/player/stat rows")

    pred = predictions.copy()
    truth = actuals.copy()
    for frame in (pred, truth):
        frame["GAME_ID"] = frame["GAME_ID"].astype("string")
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype("string")
    long_actuals = []
    for stat in sorted(set(pred["STAT"].astype(str).str.upper())):
        if stat not in truth:
            continue
        numeric = pd.to_numeric(truth[stat], errors="coerce")
        observed = truth[stat].notna()
        observed_values = numeric.loc[observed]
        if (
            observed_values.isna().any()
            or not np.isfinite(observed_values.to_numpy(dtype=float)).all()
            or (observed_values < 0).any()
        ):
            raise ValueError(
                f"Actual {stat} counts must be finite and nonnegative or missing"
            )
        block = truth[["GAME_ID", "PLAYER_ID", stat]].copy()
        block["STAT"] = stat
        block["ACTUAL"] = numeric
        block = block.drop(columns=[stat])
        long_actuals.append(block)
    if not long_actuals:
        raise ValueError("Actuals contain none of the forecast STAT targets")
    actual_long = pd.concat(long_actuals, ignore_index=True)
    if actual_long.duplicated(["GAME_ID", "PLAYER_ID", "STAT"]).any():
        raise ValueError("Actuals contain duplicate player-game rows")
    joined = pred.merge(
        actual_long,
        on=["GAME_ID", "PLAYER_ID", "STAT"],
        how="left",
        validate="many_to_one",
    )
    reconciled = joined["ACTUAL"].notna()
    target_scores: dict[str, dict[str, float | int | None]] = {}
    for stat, group in joined.groupby("STAT", sort=True):
        valid = group.dropna(subset=["ACTUAL"]).copy()
        if valid.empty:
            target_scores[str(stat)] = {
                "rows": 0,
                "mae": None,
                "rmse": None,
                "coverage_80": None,
            }
            continue
        actual = pd.to_numeric(valid["ACTUAL"], errors="raise")
        mean = pd.to_numeric(valid["MEAN"], errors="raise")
        errors = mean - actual
        target_scores[str(stat)] = {
            "rows": len(valid),
            "mae": float(errors.abs().mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "coverage_80": float(
                (
                    actual.ge(pd.to_numeric(valid["P10"], errors="raise"))
                    & actual.le(pd.to_numeric(valid["P90"], errors="raise"))
                ).mean()
            ),
        }
    return ReplayScore(
        rows=len(joined),
        games=int(joined["GAME_ID"].nunique()),
        reconciled_fraction=float(reconciled.mean()) if len(joined) else 0.0,
        targets=target_scores,
    )


__all__ = ["ReplayScore", "score_replay"]
