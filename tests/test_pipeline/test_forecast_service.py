import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.pipeline.forecast_service import ForecastService


class FakeBackend:
    targets = ["PTS", "REB"]
    models = {"PTS": object(), "REB": object()}
    model_version = "bundle-test"

    def predict_player_stats(self, context, history_df=None, include_confidence=False):
        return {"PTS": 20.0, "PTS_STD": 4.0, "REB": 8.0, "REB_STD": 2.0}


def test_forecast_service_emits_canonical_long_form_rows():
    service = ForecastService(manager=FakeBackend())
    request = ForecastRequest(
        game_id="g-1",
        game_date="2026-10-20",
        home_team_id=1,
        away_team_id=2,
        forecast_cutoff="2026-10-20T12:00:00+00:00",
        horizon="morning",
        source_snapshot_id="snapshot-1",
        model_bundle_id="bundle-1",
    )
    contexts = pd.DataFrame(
        [{
            "PLAYER_ID": 10,
            "TEAM_ID": 1,
            "OPPONENT_ID": 2,
            "DATA_QUALITY": "FULL",
        }]
    )
    frame = service.predict_forecast_frame(request, contexts)
    assert set(frame["STAT"]) == {"PTS", "REB"}
    assert frame["MODEL_BUNDLE_ID"].eq("bundle-1").all()
    assert frame["P10"].le(frame["P50"]).all()

