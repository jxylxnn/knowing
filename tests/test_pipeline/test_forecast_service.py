import numpy as np
import pandas as pd
import pytest

from src.contracts.forecast import ForecastRequest
from src.pipeline.forecast_service import ForecastService


class FakeBackend:
    targets = ["PTS", "REB"]
    models = {"PTS": object(), "REB": object()}
    model_version = "bundle-test"

    def predict_player_stats(self, context, history_df=None, include_confidence=False):
        return {"PTS": 20.0, "PTS_STD": 4.0, "REB": 8.0, "REB_STD": 2.0}


class FailingBackend(FakeBackend):
    def predict_player_stats(self, context, history_df=None, include_confidence=False):
        raise ValueError("model output unavailable")


class MissingMethodBackend:
    targets = ["PTS"]
    models = {"PTS": object()}


class StaticPredictionBackend(FakeBackend):
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_player_stats(self, context, history_df=None, include_confidence=False):
        return self.predictions


def _request():
    return ForecastRequest(
        game_id="g-1",
        game_date="2026-10-20",
        scheduled_tip="2026-10-20T19:00:00+00:00",
        home_team_id=1,
        away_team_id=2,
        forecast_cutoff="2026-10-20T12:00:00+00:00",
        horizon="morning",
        source_snapshot_id="snapshot-1",
        model_bundle_id="bundle-1",
    )


def _contexts():
    return pd.DataFrame(
        [{
            "PLAYER_ID": 10,
            "TEAM_ID": 1,
            "OPPONENT_ID": 2,
            "DATA_QUALITY": "FULL",
        }]
    )


def test_forecast_service_emits_canonical_long_form_rows():
    service = ForecastService(manager=FakeBackend())
    frame = service.predict_forecast_frame(_request(), _contexts())
    assert set(frame["STAT"]) == {"PTS", "REB"}
    assert frame["MODEL_BUNDLE_ID"].eq("bundle-1").all()
    assert frame["P10"].le(frame["P50"]).all()


def test_batch_prediction_errors_are_not_replaced_by_heuristic_fallback():
    service = ForecastService(manager=FailingBackend())
    contexts = pd.DataFrame([{"PLAYER_ID": 10}])

    with pytest.raises(RuntimeError, match="Prediction failed for player 10"):
        service.predict_player_stats_batch(contexts)


def test_empty_single_player_context_is_rejected():
    service = ForecastService(manager=FakeBackend())

    with pytest.raises(ValueError, match="must contain one player row"):
        service.predict_player_stats(pd.DataFrame())


def test_backend_without_prediction_method_is_rejected():
    service = ForecastService(manager=MissingMethodBackend())

    with pytest.raises(TypeError, match="must implement predict_player_stats"):
        service.predict_player_stats(_contexts())


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            {"PTS": 20.0, "PTS_STD": 4.0},
            "missing required target REB",
        ),
        (
            {"PTS": np.inf, "PTS_STD": 4.0, "REB": 8.0, "REB_STD": 2.0},
            "required target PTS must be numeric and finite",
        ),
    ],
)
def test_forecast_frame_rejects_missing_or_nonfinite_target_predictions(
    predictions,
    message,
):
    service = ForecastService(manager=StaticPredictionBackend(predictions))

    with pytest.raises(RuntimeError, match=message):
        service.predict_forecast_frame(_request(), _contexts())


def test_empty_batch_and_game_contexts_remain_no_work_results():
    service = ForecastService(manager=FakeBackend())

    assert service.predict_player_stats_batch(pd.DataFrame()).empty
    assert service.predict_forecast_frame(_request(), pd.DataFrame()).empty
