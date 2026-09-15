from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.contracts.forecast import ForecastRequest
from src.evaluation.replay import score_replay
from src.pipeline.forecast_service import ForecastService


class _Backend:
    targets = ["PTS"]
    models = {"PTS": object()}
    model_version = "bundle-test"

    def predict_player_stats(self, context, history_df=None, include_confidence=False):
        return {"PTS": 20.0, "PTS_STD": 2.0}


def _forecast():
    request = ForecastRequest(
        game_id="g1",
        game_date="2026-10-20",
        scheduled_tip=datetime(2026, 10, 20, 23, tzinfo=timezone.utc),
        home_team_id=1,
        away_team_id=2,
        forecast_cutoff=datetime(2026, 10, 20, 12, tzinfo=timezone.utc),
        horizon="morning",
        source_snapshot_id="snapshot",
        model_bundle_id="bundle",
    )
    context = pd.DataFrame(
        [{"PLAYER_ID": 10, "TEAM_ID": 1, "OPPONENT_ID": 2}]
    )
    return ForecastService(manager=_Backend()).predict_forecast_frame(
        request, context
    )


def test_replay_scores_canonical_forecasts():
    result = score_replay(
        _forecast(),
        pd.DataFrame([{"GAME_ID": "g1", "PLAYER_ID": 10, "PTS": 19}]),
    )
    assert result.rows == 1
    assert result.reconciled_fraction == 1.0
    assert result.targets["PTS"]["mae"] == 1.0


@pytest.mark.parametrize("value", [np.inf, -1, "NaN", "inf"])
def test_replay_rejects_invalid_observed_counts(value):
    with pytest.raises(ValueError, match="ACTUAL|finite|nonnegative"):
        score_replay(
            _forecast(),
            pd.DataFrame([{"GAME_ID": "g1", "PLAYER_ID": 10, "PTS": value}]),
        )


def test_replay_preserves_missing_actuals_as_unreconciled():
    result = score_replay(
        _forecast(),
        pd.DataFrame([{"GAME_ID": "g1", "PLAYER_ID": 10, "PTS": np.nan}]),
    )

    assert result.rows == 1
    assert result.reconciled_fraction == 0.0
    assert result.targets["PTS"]["rows"] == 0
    assert result.targets["PTS"]["mae"] is None
