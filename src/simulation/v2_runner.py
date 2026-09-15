"""Strict Model v2 scheduled-game forecasting and joint simulation.

The champion path is the default.  Callers may instead name one explicit
sealed v2 bundle for non-published candidate execution; such a run never
touches the official ledger.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.snapshots import validate_source_snapshot
from src.forecasting.baseline_backend import V2BaselineBackend
from src.operations.ledger import OfficialForecastLedger
from src.pipeline.forecast_service import ForecastService
from src.pipeline.v2_execution import execute_request


EASTERN = ZoneInfo("America/New_York")


def run_scheduled_game(
    game: pd.Series,
    *,
    source_snapshot_id: str,
    data_dir: str | Path = "data",
    models_dir: str | Path = "models",
    horizon: str = "morning",
    simulations: int = 100,
    seed: int = 42,
    strict: bool = True,
    persist: bool = True,
    candidate_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast and simulate one schedule row through only Model v2 APIs.

    ``candidate_dir`` names one explicit sealed v2 bundle for non-published
    candidate execution.  It is loaded through ``V2BaselineBackend`` (which
    validates the bundle with ``promotion=False``) and injected into
    ``ForecastService``.  Candidate runs may persist recovery evidence in the
    ``ForecastRunStore`` but never write the official ledger, because the
    bundle is not the configured champion.  When ``candidate_dir`` is omitted
    the champion is loaded exactly as before.
    """

    required = {
        "GAME_ID", "GAME_DATE", "SCHEDULED_TIP", "HOME_TEAM_ID", "AWAY_TEAM_ID"
    }
    if missing := required - set(game.index):
        raise ValueError(f"Schedule row missing Model v2 fields: {sorted(missing)}")
    tip = pd.Timestamp(game["SCHEDULED_TIP"])
    if tip.tzinfo is None:
        raise ValueError("SCHEDULED_TIP must be timezone-aware")
    cutoff = horizon_cutoff(tip.to_pydatetime(), horizon)
    validate_source_snapshot(
        data_dir, source_snapshot_id, forecast_cutoff=cutoff
    )

    if candidate_dir is None:
        service = ForecastService(
            models_dir=str(models_dir), data_dir=str(data_dir)
        )
    else:
        backend = V2BaselineBackend(candidate_dir)
        service = ForecastService(
            model_backend=backend, models_dir=str(models_dir),
            data_dir=str(data_dir),
        )
    bundle_id = str(service.backend.model_version)
    request = ForecastRequest(
        game_id=str(game["GAME_ID"]),
        schedule_version=str(game.get("SCHEDULE_VERSION", "schedule_v1")),
        game_date=pd.Timestamp(game["GAME_DATE"]).date(),
        scheduled_tip=tip.to_pydatetime(),
        home_team_id=int(game["HOME_TEAM_ID"]),
        away_team_id=int(game["AWAY_TEAM_ID"]),
        forecast_cutoff=cutoff,
        horizon=horizon,
        source_snapshot_id=source_snapshot_id,
        model_bundle_id=bundle_id,
        scenario="official" if strict else "degraded_observed_roster",
    )
    distribution = execute_request(
        service, request, data_dir=data_dir, simulations=simulations,
        seed=seed, persist=persist,
    )
    forecast, simulations_frame = distribution.forecasts, distribution.samples
    # Only the champion publishes official forecasts; an explicit candidate
    # bundle stays non-published even when the run is strict.
    if persist and strict and candidate_dir is None:
        OfficialForecastLedger(Path(data_dir) / "ledger").write_forecast(forecast)
    return forecast, simulations_frame


def latest_observed_roster(
    history: pd.DataFrame,
    game_date: date,
) -> pd.DataFrame:
    """Infer the latest prior team for each player in a degraded run."""

    required = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE"}
    if missing := required - set(history.columns):
        raise ValueError(f"Player history missing roster fields: {sorted(missing)}")
    frame = history.copy()
    target_date = pd.Timestamp(game_date).date()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"], errors="raise")
    frame = frame.loc[frame["GAME_DATE"].dt.date < target_date]
    if frame.empty:
        raise ValueError("No prior player appearances exist before the scheduled game")
    frame = frame.sort_values(["GAME_DATE", "GAME_ID"], kind="stable")
    return frame.drop_duplicates("PLAYER_ID", keep="last")


def horizon_cutoff(scheduled_tip: datetime, horizon: str) -> datetime:
    local_tip = scheduled_tip.astimezone(EASTERN)
    if horizon == "previous_night":
        return datetime.combine(
            local_tip.date() - timedelta(days=1), time(23, 59), EASTERN
        )
    if horizon == "morning":
        return datetime.combine(local_tip.date(), time(9), EASTERN)
    if horizon == "pregame_90m":
        return local_tip - timedelta(minutes=90)
    if horizon == "pregame_30m":
        return local_tip - timedelta(minutes=30)
    raise ValueError(f"Unknown forecast horizon: {horizon}")


__all__ = ["horizon_cutoff", "latest_observed_roster", "run_scheduled_game"]
