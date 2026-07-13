from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.contracts.errors import ContractError
from src.contracts.forecast import ForecastRequest, validate_forecast_frame
from src.features.materializer import FeatureMaterializer, FeatureRegistry, FeatureSpec


def _request():
    return ForecastRequest(
        game_id="g-1",
        game_date=date(2026, 10, 20),
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

