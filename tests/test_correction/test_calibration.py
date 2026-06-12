"""Tests for residual interval calibration and confidence scoring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.correction.calibration import ResidualIntervalCalibrator
from src.correction.confidence_scorer import ConfidenceScorer
from src.correction.interval_store import CalibrationIntervalStore
from src.simulation.report_generator import ReportGenerator


TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


def sample_residual_df(rows_per_stat: int = 12) -> pd.DataFrame:
    rows = []
    for stat_idx, stat in enumerate(TARGETS):
        for i in range(rows_per_stat):
            base = 10.0 + stat_idx + (i % 4)
            error = ((i % 5) - 2) * 0.6
            actual = base + error
            rows.append(
                {
                    "GAME_ID": f"G{i}",
                    "GAME_DATE": f"2025-01-{(i % 28) + 1:02d}",
                    "PLAYER_ID": f"P{i % 4}",
                    "PLAYER_NAME": f"Player {i % 4}",
                    "TEAM_ID": "BOS",
                    "OPPONENT": "LAL",
                    "STAT": stat,
                    "BASE_PREDICTION": base,
                    "ACTUAL": actual,
                    "ERROR": error,
                    "CORRECTED_PREDICTION": base + (error * 0.25),
                    "MODEL_FOLD": "fold_1",
                    "FEATURE_CUTOFF_DATE": "2025-01-01",
                    "MODEL_VERSION": "test",
                    "DATA_QUALITY": "FULL" if i % 2 == 0 else "DEGRADED_FALLBACK",
                    "MINUTES_CONFIDENCE": 0.85 if i % 2 == 0 else 0.35,
                }
            )
    return pd.DataFrame(rows)


def test_calibration_creates_interval_files_for_all_six_stats(tmp_path):
    input_path = tmp_path / "residual_training.parquet"
    output_dir = tmp_path / "calibration"
    sample_residual_df().to_parquet(input_path, index=False)

    calibrator = ResidualIntervalCalibrator(min_bucket_rows=2)
    metadata = calibrator.calibrate_file(str(input_path), str(output_dir))

    assert (output_dir / "calibration_metadata.json").exists()
    assert set(metadata["targets"]) == set(TARGETS)
    for stat in TARGETS:
        assert (output_dir / f"{stat.lower()}_intervals.json").exists()


def test_interval_widths_are_non_negative(tmp_path):
    output_dir = tmp_path / "calibration"
    ResidualIntervalCalibrator(min_bucket_rows=2).calibrate(
        sample_residual_df(),
        str(output_dir),
    )

    payload = json.loads((output_dir / "pts_intervals.json").read_text())
    widths = payload["buckets"]["GLOBAL"]["widths"]
    assert all(value >= 0.0 for value in widths.values())


def test_interval_lower_bound_is_clipped():
    interval = ResidualIntervalCalibrator.make_interval(prediction=2.0, width=5.0)
    assert interval.low == 0.0
    assert interval.high == 7.0


def test_missing_bucket_falls_back_to_global(tmp_path):
    output_dir = tmp_path / "calibration"
    ResidualIntervalCalibrator(min_bucket_rows=999).calibrate(
        sample_residual_df(),
        str(output_dir),
    )

    store = CalibrationIntervalStore(str(output_dir)).load()
    global_width = store.get_interval_width("PTS", confidence=0.9, bucket="GLOBAL")
    missing_width = store.get_interval_width("PTS", confidence=0.9, bucket="MISSING")

    assert missing_width == global_width


def test_missing_calibration_files_disable_intervals_safely(tmp_path):
    store = CalibrationIntervalStore(str(tmp_path / "missing")).load()

    assert store.enabled is False
    assert store.has_stat("PTS") is False
    assert store.get_interval_width("PTS", confidence=0.9) is None


def test_confidence_scorer_lowers_confidence_for_degraded_data():
    scorer = ConfidenceScorer()
    full = scorer.score("PTS", interval_width=3.0, data_quality="FULL")
    degraded = scorer.score("PTS", interval_width=3.0, data_quality="DEGRADED_MISSING")

    assert degraded.score < full.score
    assert ConfidenceScorer.LABELS.index(degraded.label) >= ConfidenceScorer.LABELS.index(full.label)


def test_calibration_metadata_includes_row_counts_and_coverage_levels(tmp_path):
    output_dir = tmp_path / "calibration"
    metadata = ResidualIntervalCalibrator(min_bucket_rows=2).calibrate(
        sample_residual_df(),
        str(output_dir),
    )

    assert metadata["confidence_levels"] == [0.8, 0.9, 0.95]
    assert metadata["targets"]["PTS"]["rows"] == 12
    assert "q90" in metadata["targets"]["PTS"]["buckets"]["GLOBAL"]["coverage"]


def test_projection_export_includes_interval_and_confidence_columns(tmp_path):
    generator = ReportGenerator(output_dir=str(tmp_path))
    results = [
        {
            "game_id": "TEST_001",
            "date": "2025-01-15",
            "team_a": "BOS",
            "team_b": "LAL",
            "player_averages": [
                {
                    "name": "Test Player",
                    "team": "BOS",
                    "pts": 25.0,
                    "pts_mode": 24.5,
                    "pts_95_ci": [18.0, 32.0],
                    "pts_99_ci": [15.0, 35.0],
                    "pts_interval_80_low": 20.0,
                    "pts_interval_80_high": 30.0,
                    "pts_interval_90_low": 18.5,
                    "pts_interval_90_high": 31.5,
                    "pts_confidence": "MEDIUM",
                    "reb": 8.0,
                    "reb_mode": 8.0,
                    "reb_95_ci": [5.0, 11.0],
                    "reb_99_ci": [3.0, 13.0],
                    "ast": 6.0,
                    "ast_mode": 6.0,
                    "ast_95_ci": [3.0, 9.0],
                    "ast_99_ci": [2.0, 10.0],
                    "stl": 1.2,
                    "stl_mode": 1.0,
                    "stl_95_ci": [0.0, 3.0],
                    "stl_99_ci": [0.0, 4.0],
                    "blk": 0.8,
                    "blk_mode": 1.0,
                    "blk_95_ci": [0.0, 2.0],
                    "blk_99_ci": [0.0, 3.0],
                    "tov": 2.5,
                    "tov_mode": 2.0,
                    "tov_95_ci": [1.0, 4.0],
                    "tov_99_ci": [0.0, 5.0],
                    "play_probability": 1.0,
                }
            ],
        }
    ]

    path = generator.export_player_projections(results, filename="test_projections.csv")
    df = pd.read_csv(path)

    assert df.loc[0, "PTS_INTERVAL_80_LOW"] == 20.0
    assert df.loc[0, "PTS_INTERVAL_90_HIGH"] == 31.5
    assert df.loc[0, "PTS_CONFIDENCE"] == "MEDIUM"
    assert "REB_INTERVAL_80_LOW" in df.columns
    assert np.isnan(df.loc[0, "REB_INTERVAL_80_LOW"])
    assert df.loc[0, "REB_CONFIDENCE"] == "NO_EDGE"
