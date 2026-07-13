"""Tests for ProjectionLoader with correction/calibration columns."""

import numpy as np
import pandas as pd
import pytest

from src.contracts.projections import validate_projection_frame, REQUIRED_PROJECTION_COLUMNS
from src.query.projection_loader import ProjectionLoader, PlayerProjection


TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


def _make_core_projection_df(num_rows=1):
    """Build a DataFrame with only core (pre-Ticket 4) columns."""
    rows = []
    for i in range(num_rows):
        row = {
            "PLAYER_NAME": f"Player {i}",
            "TEAM": "BOS",
            "OPPONENT": "LAL",
            "DATA_QUALITY": "FULL",
            "DATE": "2025-01-15",
            "IS_HOME": True,
            "GAME_ID": f"G{i:03d}",
            "PLAY_PROBABILITY": 1.0,
        }
        for stat in TARGETS:
            row[stat] = 15.0 + i
            row[f"{stat}_P10"] = 8.0 + i
            row[f"{stat}_P50"] = 15.0 + i
            row[f"{stat}_P90"] = 22.0 + i
            row[f"{stat}_STD"] = 4.5
            row[f"{stat}_SKEW"] = 0.2
            row[f"{stat}_ZERO_PROB"] = 0.01
            row[f"{stat}_LAMBDA"] = 15.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def _make_full_projection_df(num_rows=1):
    """Build a DataFrame with all columns (core + optional)."""
    df = _make_core_projection_df(num_rows)
    for stat in TARGETS:
        df[f"{stat}_INTERVAL_80_LOW"] = df[stat] - 4.0
        df[f"{stat}_INTERVAL_80_HIGH"] = df[stat] + 4.0
        df[f"{stat}_INTERVAL_90_LOW"] = df[stat] - 6.0
        df[f"{stat}_INTERVAL_90_HIGH"] = df[stat] + 6.0
        df[f"{stat}_CONFIDENCE_SCORE"] = 70.0
        df[f"{stat}_CONFIDENCE"] = "MEDIUM"
        df[f"{stat}_CORRECTED"] = df[stat] + 1.5
        df[f"{stat}_BASE"] = df[stat]
        df[f"{stat}_RESIDUAL_CORRECTION"] = 1.5
    return df


def test_core_projection_validates(tmp_path):
    """Core-only CSV should pass validation."""
    df = _make_core_projection_df()
    validate_projection_frame(df)


def test_full_projection_validates(tmp_path):
    df = _make_full_projection_df()
    validate_projection_frame(df)


def test_loader_loads_core_csv(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_core_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    df = loader.load_projections()
    assert not df.empty
    assert "PTS" in df.columns


def test_loader_loads_full_csv(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_full_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    df = loader.load_projections()
    assert not df.empty
    assert "PTS_CORRECTED" in df.columns


def test_row_to_projection_includes_corrected(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_full_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    proj = loader.find_player("Player 0")
    assert proj is not None
    assert proj.pts_corrected is not None
    assert abs(proj.pts_corrected - proj.pts_mean) > 0.1


def test_row_to_projection_core_missing_optional(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_core_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    df = loader.load_projections()
    proj = loader.find_player("Player 0")
    assert proj is not None
    assert proj.pts_corrected is None
    assert proj.pts_confidence == ""


def test_row_to_projection_interval_fields(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_full_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    proj = loader.find_player("Player 0")
    assert proj.pts_interval_90_low is not None
    assert proj.pts_interval_90_high is not None
    assert proj.pts_confidence == "MEDIUM"


def test_get_stat_corrected(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_full_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    proj = loader.find_player("Player 0")
    assert proj.get_stat_corrected("pts") is not None
    assert proj.get_stat_corrected("reb") is not None


def test_get_stat_confidence(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_full_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    proj = loader.find_player("Player 0")
    assert proj.get_stat_confidence("pts") == "MEDIUM"


def test_get_stat_interval(tmp_path):
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    _make_full_projection_df().to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    proj = loader.find_player("Player 0")
    iv90 = proj.get_stat_interval("pts", 0.9)
    assert iv90 is not None
    low, high = iv90
    assert low < high
    iv80 = proj.get_stat_interval("pts", 0.8)
    assert iv80 is not None
    assert iv80[0] > iv90[0]
    assert iv80[1] < iv90[1]


def test_interval_low_greater_than_high_fails():
    df = _make_full_projection_df()
    df["PTS_INTERVAL_90_LOW"] = 30.0
    df["PTS_INTERVAL_90_HIGH"] = 10.0
    with pytest.raises(Exception, match="must not exceed"):
        validate_projection_frame(df)


def test_correction_column_non_numeric_fails():
    df = _make_full_projection_df()
    df["PTS_CORRECTED"] = "not_a_number"
    with pytest.raises(Exception, match="is not numeric"):
        validate_projection_frame(df)


def test_base_column_non_numeric_fails():
    df = _make_full_projection_df()
    df["PTS_BASE"] = "bad_value"
    with pytest.raises(Exception, match="is not numeric"):
        validate_projection_frame(df)


def test_interval_validation_handles_mismatched_nan_rows():
    df = _make_full_projection_df(num_rows=3)
    df.loc[0, "PTS_INTERVAL_90_HIGH"] = None
    df.loc[1, "PTS_INTERVAL_90_LOW"] = None
    df.loc[2, "PTS_INTERVAL_90_LOW"] = None
    df.loc[2, "PTS_INTERVAL_90_HIGH"] = None
    validate_projection_frame(df)  # should not crash


def test_all_null_optional_correction_column_passes():
    df = _make_core_projection_df()
    df["PTS_CORRECTED"] = None
    df["PTS_BASE"] = None
    df["PTS_RESIDUAL_CORRECTION"] = None
    validate_projection_frame(df)


def test_blank_confidence_in_csv_does_not_become_nan_string(tmp_path):
    df = _make_full_projection_df()
    df["PTS_CONFIDENCE"] = None
    csv_path = tmp_path / "player_projections_20250115_120000.csv"
    df.to_csv(csv_path, index=False)
    loader = ProjectionLoader(data_dir=str(tmp_path))
    proj = loader.find_player("Player 0")
    assert proj is not None
    assert proj.pts_confidence == ""
    assert proj.get_stat_confidence("pts") == ""
