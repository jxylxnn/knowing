import pandas as pd

from src.utils.prediction_utils import FeatureSelector


def test_current_game_team_outcomes_never_enter_feature_schema():
    frame = pd.DataFrame(
        {
            "PTS": [10.0, 12.0],
            "PTS_TEAM": [110.0, 120.0],
            "REB_TEAM": [45.0, 48.0],
            "TEAM_PACE_10": [99.0, 101.0],
            "PTS_PLAYER_TE": [9.0, 10.0],
            "UNSAFE_TELEMETRY": [1.0, 2.0],
        }
    )
    schema = FeatureSelector(targets=["PTS"]).fit(frame)
    assert "PTS_TEAM" not in schema.feature_cols
    assert "REB_TEAM" not in schema.feature_cols
    assert "TEAM_PACE_10" in schema.feature_cols
    assert "PTS_PLAYER_TE" in schema.feature_cols
    assert "UNSAFE_TELEMETRY" not in schema.feature_cols
