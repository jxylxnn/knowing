from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.contracts.errors import ContractError
from src.contracts.forecast import ForecastRequest, validate_forecast_frame
from src.features.materializer import FeatureMaterializer, FeatureRegistry, FeatureSpec


def _valid_forecast_frame():
    return pd.DataFrame([{
        **{name: 0.0 for name in (
            "REQUEST_ID", "MODEL_BUNDLE_ID", "SOURCE_SNAPSHOT_ID", "GENERATED_AT",
            "FORECAST_CUTOFF", "GAME_ID", "SCHEDULE_VERSION", "PLAYER_ID", "TEAM_ID",
            "OPPONENT_ID", "HORIZON", "P_ACTIVE", "P_PLAY_GIVEN_ACTIVE", "PLAY_PROB",
            "EXPECTED_MINUTES", "MIN_P10", "MIN_P50", "MIN_P90", "MEAN",
            "P10", "P25", "P50", "P75", "P90", "ZERO_PROB", "DATA_QUALITY",
            "CALIBRATION_VERSION", "SCENARIO",
        )},
        "GAME_DATE": "2026-10-20",
        "STAT": "PTS",
    }])


def _request():
    return ForecastRequest(
        game_id="g-1",
        game_date=date(2026, 10, 20),
        scheduled_tip=datetime(2026, 10, 20, 19, tzinfo=timezone.utc),
        home_team_id=1,
        away_team_id=2,
        forecast_cutoff=datetime(2026, 10, 20, 12, tzinfo=timezone.utc),
        horizon="morning",
        source_snapshot_id="snapshot-1",
        model_bundle_id="bundle-1",
    )


def test_forecast_request_is_immutable_and_has_stable_id():
    request = _request()
    assert request.request_id == _request().request_id
    with pytest.raises((AttributeError, TypeError)):
        request.game_id = "changed"


def test_forecast_request_rejects_unknown_horizon():
    with pytest.raises(ContractError):
        ForecastRequest(
            game_id="g-1",
            game_date=date(2026, 10, 20),
            scheduled_tip=datetime(2026, 10, 20, 19, tzinfo=timezone.utc),
            home_team_id=1,
            away_team_id=2,
            forecast_cutoff=datetime(2026, 10, 20, 12),
            horizon="after_game",
            source_snapshot_id="snapshot-1",
            model_bundle_id="bundle-1",
        )


def test_materializer_filters_unavailable_and_future_events():
    registry = FeatureRegistry()
    registry.register(
        FeatureSpec(
            name="ROLL_PTS_5",
            group="rolling",
            dtype="float64",
            entity_keys=("PLAYER_ID",),
            event_time_column="event_time",
            max_lookback_days=30,
            required_sources=("nba_game_logs",),
            allowed_horizons=("morning",),
            missing_policy="error",
            version="1",
        )
    )
    source = pd.DataFrame(
        {
            "PLAYER_ID": [10, 10, 10],
            "ROLL_PTS_5": [12.0, 99.0, 88.0],
            "event_time": ["2026-10-10", "2026-10-19", "2026-10-21"],
            "available_at": [
                "2026-10-10T10:00:00Z",
                "2026-10-20T13:00:00Z",
                "2026-10-21T10:00:00Z",
            ],
        }
    )
    frame = FeatureMaterializer(registry).materialize_forecast(_request(), source)
    assert list(frame["ROLL_PTS_5"]) == [12.0]
    assert "99.0" not in frame.astype(str).to_string()


def test_forecast_contract_requires_game_date():
    frame = _valid_forecast_frame().drop(columns=["GAME_DATE"])

    with pytest.raises(ContractError, match="GAME_DATE"):
        validate_forecast_frame(frame)


@pytest.mark.parametrize("value", [None, pd.NaT, "not-a-date"])
def test_forecast_contract_rejects_invalid_game_date(value):
    frame = _valid_forecast_frame()
    frame.loc[0, "GAME_DATE"] = value

    with pytest.raises(ContractError, match="GAME_DATE.*date-like.*non-null"):
        validate_forecast_frame(frame)


@pytest.mark.parametrize("column", ["MEAN", "EXPECTED_MINUTES", "P50"])
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_forecast_contract_rejects_nonfinite_numeric_outputs(column, value):
    frame = _valid_forecast_frame()
    frame.loc[0, column] = value
    with pytest.raises(ContractError, match=column):
        validate_forecast_frame(frame)
