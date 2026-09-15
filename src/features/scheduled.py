"""Synthetic pregame player rows for scheduled Model v2 forecasts."""

from __future__ import annotations

import pandas as pd

from src.contracts.forecast import ForecastRequest


def materialize_scheduled_rows(
    request: ForecastRequest, roster: pd.DataFrame, history: pd.DataFrame,
    *, statuses: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct one scheduled-game row per roster candidate at the cutoff.

    Only previous completed games from the same player are used to calculate
    rest and lagged context.  The scheduled fixture itself is never looked up
    in player-game outcomes.
    """
    required = {"PLAYER_ID", "TEAM_ID"}
    if missing := required - set(roster.columns):
        raise ValueError(f"Roster candidates missing columns: {sorted(missing)}")
    candidates = roster.loc[
        roster["TEAM_ID"].astype(str).isin(
            {str(request.home_team_id), str(request.away_team_id)}
        ), ["PLAYER_ID", "TEAM_ID"],
    ].copy()
    if candidates["PLAYER_ID"].isna().any() or candidates["PLAYER_ID"].duplicated().any():
        raise ValueError("Scheduled roster requires unique known player identities")
    candidates["GAME_ID"] = request.game_id
    candidates["SCHEDULE_VERSION"] = request.schedule_version
    candidates["GAME_DATE"] = request.game_date.isoformat()
    candidates["SCHEDULED_TIP"] = request.scheduled_tip.isoformat()
    candidates["FORECAST_CUTOFF"] = request.forecast_cutoff.isoformat()
    candidates["HORIZON"] = request.horizon
    candidates["HOME_FLAG"] = candidates["TEAM_ID"].astype(str).eq(str(request.home_team_id)).astype(int)
    candidates["OPPONENT_ID"] = candidates["TEAM_ID"].astype(str).map(
        {str(request.home_team_id): request.away_team_id, str(request.away_team_id): request.home_team_id}
    )
    prior = history.copy()
    if not prior.empty:
        prior["GAME_DATE"] = pd.to_datetime(prior["GAME_DATE"], errors="coerce")
        prior = prior.loc[prior["GAME_DATE"].dt.date < request.game_date]
    last_dates = prior.groupby("PLAYER_ID")["GAME_DATE"].max() if not prior.empty else pd.Series(dtype="datetime64[ns]")
    candidates["LAST_GAME_DATE"] = candidates["PLAYER_ID"].map(last_dates)
    candidates["REST_DAYS"] = (pd.Timestamp(request.game_date) - pd.to_datetime(candidates["LAST_GAME_DATE"])).dt.days.sub(1).clip(lower=0)
    candidates["REST_DAYS"] = candidates["REST_DAYS"].fillna(7).clip(upper=7)
    candidates["ROSTER_ELIGIBLE"] = 1
    candidates["PRIOR_APPEARANCES"] = candidates["PLAYER_ID"].map(
        prior.groupby("PLAYER_ID").size() if not prior.empty else pd.Series(dtype=float)
    ).fillna(0)
    if not prior.empty:
        prior = prior.sort_values("GAME_DATE", kind="stable")
        for column in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"):
            if column not in prior:
                continue
            prior[column] = pd.to_numeric(prior[column], errors="raise")
            for window in (5, 10, 20):
                means = prior.groupby("PLAYER_ID", sort=False)[column].apply(
                    lambda group: group.tail(window).mean()
                )
                candidates[f"PRIOR_{column}_{window}"] = candidates.PLAYER_ID.map(means)
        if "TEAM_ID" in prior:
            for column in ("PTS", "REB", "AST", "TOV"):
                if column not in prior:
                    continue
                team_means = prior.groupby("TEAM_ID")[column].mean()
                candidates[f"PRIOR_TEAM_{column}"] = candidates.TEAM_ID.map(team_means)
                # Match string and integer identifiers without approximate names.
                lookup = {str(key): value for key, value in team_means.items()}
                candidates[f"PRIOR_OPPONENT_{column}"] = candidates.OPPONENT_ID.astype(str).map(lookup)
    if statuses is not None:
        candidates = candidates.merge(statuses[["PLAYER_ID", "TEAM_ID", "STATUS"]],
                                      on=["PLAYER_ID", "TEAM_ID"], how="left",
                                      validate="one_to_one")
    else:
        candidates["STATUS"] = "UNKNOWN"
    candidates["STATUS"] = candidates.STATUS.fillna("UNKNOWN")
    candidates["PRIOR_TEAM_OUT_COUNT"] = candidates.STATUS.eq("OUT").groupby(candidates.TEAM_ID).transform("sum")
    complete_status = candidates.STATUS.ne("UNKNOWN").groupby(candidates.TEAM_ID).transform("all")
    candidates["PRIOR_TEAM_OUT_COUNT"] = candidates.PRIOR_TEAM_OUT_COUNT.where(complete_status)
    return candidates.reset_index(drop=True)
