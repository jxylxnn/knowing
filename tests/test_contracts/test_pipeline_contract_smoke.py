import pandas as pd
import pytest

from src.contracts.features import validate_feature_frame
from src.contracts.projections import validate_projection_frame
from src.contracts.schedule import normalize_schedule_frame


def test_schedule_contract_accepts_valid_schedule():
    df = pd.DataFrame(
        [
            {
                "GAME_ID": "001",
                "GAME_DATE": "2026-01-01",
                "HOME_TEAM": "BOS",
                "AWAY_TEAM": "MIA",
            }
        ]
    )

    out = normalize_schedule_frame(df)

    assert out.loc[0, "HOME_TEAM"] == "BOS"
    assert out.loc[0, "AWAY_TEAM"] == "MIA"


def test_projection_contract_requires_all_six_stats():
    row = {
        "PLAYER_NAME": "Test Player",
        "TEAM": "BOS",
        "OPPONENT": "MIA",
        "DATA_QUALITY": "FULL",
    }

    for stat in ("PTS", "REB", "AST", "STL", "BLK", "TOV"):
        row[stat] = 10.0
        row[f"{stat}_P10"] = 5.0
        row[f"{stat}_P50"] = 10.0
        row[f"{stat}_P90"] = 15.0
        row[f"{stat}_STD"] = 3.0
        row[f"{stat}_SKEW"] = 0.0
        row[f"{stat}_ZERO_PROB"] = 0.0
        row[f"{stat}_LAMBDA"] = 10.0
        row[f"{stat}_INTERVAL_80_LOW"] = 7.0
        row[f"{stat}_INTERVAL_80_HIGH"] = 13.0
        row[f"{stat}_INTERVAL_90_LOW"] = 6.0
        row[f"{stat}_INTERVAL_90_HIGH"] = 14.0
        row[f"{stat}_CONFIDENCE"] = "MEDIUM"
        row[f"{stat}_CONFIDENCE_SCORE"] = 60.0

    df = pd.DataFrame([row])

    validate_projection_frame(df)


def test_feature_contract_rejects_missing_expected_feature():
    df = pd.DataFrame({"A": [1.0], "B": [2.0]})

    with pytest.raises(Exception):
        validate_feature_frame(df, ["A", "B", "C"])
