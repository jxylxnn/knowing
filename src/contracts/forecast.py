"""Contracts for point-in-time forecast requests and canonical outputs.

The legacy query/report code uses a wide CSV representation.  These contracts
define the immutable request and the long-form representation used by the
forecasting core so that a forecast can be replayed and audited later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.contracts.errors import ContractError

FORECAST_HORIZONS = ("previous_night", "morning", "pregame_90m", "pregame_30m")

CANONICAL_FORECAST_COLUMNS = (
    "REQUEST_ID",
    "MODEL_BUNDLE_ID",
    "SOURCE_SNAPSHOT_ID",
    "GENERATED_AT",
    "FORECAST_CUTOFF",
    "GAME_ID",
    "GAME_DATE",
    "SCHEDULE_VERSION",
    "PLAYER_ID",
    "TEAM_ID",
    "OPPONENT_ID",
    "HORIZON",
    "P_ACTIVE",
    "P_PLAY_GIVEN_ACTIVE",
    "PLAY_PROB",
    "EXPECTED_MINUTES",
    "MIN_P10",
    "MIN_P50",
    "MIN_P90",
    "STAT",
    "MEAN",
    "P10",
    "P25",
    "P50",
    "P75",
    "P90",
    "ZERO_PROB",
    "DATA_QUALITY",
    "CALIBRATION_VERSION",
    "SCENARIO",
)


def _iso(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class ForecastRequest:
    """The complete immutable identity of one pregame forecast."""

    game_id: str
    game_date: date
    scheduled_tip: datetime
    home_team_id: int
    away_team_id: int
    forecast_cutoff: datetime
    horizon: str
    source_snapshot_id: str
    model_bundle_id: str
    schedule_version: str = "schedule_v1"
    scenario: str = "official"

    def __post_init__(self) -> None:
        if not str(self.game_id).strip():
            raise ContractError("ForecastRequest.game_id is required")
        if not str(self.source_snapshot_id).strip():
            raise ContractError("ForecastRequest.source_snapshot_id is required")
        if not str(self.model_bundle_id).strip():
            raise ContractError("ForecastRequest.model_bundle_id is required")
        if not str(self.schedule_version).strip():
            raise ContractError("ForecastRequest.schedule_version is required")
        if self.horizon not in FORECAST_HORIZONS:
            raise ContractError(
                f"Unknown forecast horizon {self.horizon!r}; "
                f"expected one of {FORECAST_HORIZONS}"
            )
        if not str(self.scenario).strip():
            raise ContractError("ForecastRequest.scenario is required")

        game_date = self.game_date
        if isinstance(game_date, datetime):
            game_date = game_date.date()
        elif not isinstance(game_date, date):
            try:
                game_date = pd.Timestamp(game_date).date()
            except Exception as exc:
                raise ContractError("game_date must be date-like") from exc
        object.__setattr__(self, "game_date", game_date)

        scheduled_tip = self.scheduled_tip
        if not isinstance(scheduled_tip, datetime):
            try:
                scheduled_tip = pd.Timestamp(scheduled_tip).to_pydatetime()
            except Exception as exc:
                raise ContractError("scheduled_tip must be datetime-like") from exc
        if scheduled_tip.tzinfo is None:
            raise ContractError("scheduled_tip must be timezone-aware")
        object.__setattr__(self, "scheduled_tip", scheduled_tip)

        cutoff = self.forecast_cutoff
        if not isinstance(cutoff, datetime):
            try:
                cutoff = pd.Timestamp(cutoff).to_pydatetime()
            except Exception as exc:
                raise ContractError("forecast_cutoff must be datetime-like") from exc
        if cutoff.tzinfo is None:
            raise ContractError("forecast_cutoff must be timezone-aware")
        object.__setattr__(self, "forecast_cutoff", cutoff)

        if cutoff >= scheduled_tip:
            raise ContractError("forecast_cutoff must be before scheduled_tip")

        if int(self.home_team_id) == int(self.away_team_id):
            raise ContractError("home_team_id and away_team_id must differ")

    @property
    def request_id(self) -> str:
        """Stable identifier for ledgering and replaying this request."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["game_date"] = _iso(self.game_date)
        payload["scheduled_tip"] = _iso(self.scheduled_tip)
        payload["forecast_cutoff"] = _iso(self.forecast_cutoff)
        payload["home_team_id"] = int(self.home_team_id)
        payload["away_team_id"] = int(self.away_team_id)
        return payload


def validate_forecast_frame(frame: pd.DataFrame) -> None:
    """Validate the canonical long-form forecast output."""
    if not isinstance(frame, pd.DataFrame):
        raise ContractError("Forecast output must be a pandas DataFrame")

    missing = [column for column in CANONICAL_FORECAST_COLUMNS if column not in frame]
    if missing:
        raise ContractError(
            "Canonical forecast is missing required columns: " + ", ".join(missing)
        )
    if frame.empty:
        return

    try:
        game_dates = pd.to_datetime(frame["GAME_DATE"], errors="coerce", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(
            "GAME_DATE must contain date-like, non-null values"
        ) from exc
    if game_dates.isna().any():
        raise ContractError("GAME_DATE must contain date-like, non-null values")

    for column in ("P_ACTIVE", "P_PLAY_GIVEN_ACTIVE", "PLAY_PROB", "ZERO_PROB"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ContractError(f"{column} must contain probabilities in [0, 1]")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ContractError(f"{column} must contain finite values")

    numeric = (
        "EXPECTED_MINUTES", "MIN_P10", "MIN_P50", "MIN_P90",
        "MEAN", "P10", "P25", "P50", "P75", "P90",
    )
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ContractError(f"{column} must be numeric and non-null")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ContractError(f"{column} must contain finite values")
        if (values < 0).any():
            raise ContractError(f"{column} must be nonnegative")

    expected_play = (
        pd.to_numeric(frame["P_ACTIVE"])
        * pd.to_numeric(frame["P_PLAY_GIVEN_ACTIVE"])
    )
    if not expected_play.sub(pd.to_numeric(frame["PLAY_PROB"])).abs().le(1e-9).all():
        raise ContractError("PLAY_PROB must equal P_ACTIVE * P_PLAY_GIVEN_ACTIVE")

    for lower, upper in (("MIN_P10", "MIN_P50"), ("MIN_P50", "MIN_P90"),
                         ("P10", "P25"), ("P25", "P50"),
                         ("P50", "P75"), ("P75", "P90")):
        if (pd.to_numeric(frame[lower]) > pd.to_numeric(frame[upper])).any():
            raise ContractError(f"{lower} must not exceed {upper}")

    if frame["STAT"].isna().any() or (frame["STAT"].astype(str).str.strip() == "").any():
        raise ContractError("STAT must be populated for every forecast row")


def forecast_request_from_mapping(payload: Mapping[str, Any]) -> ForecastRequest:
    """Build a request from JSON/YAML-friendly field names."""
    return ForecastRequest(**dict(payload))
