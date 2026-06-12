"""Tests for walk-forward residual builder."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.correction.residual_dataset import ResidualTrainingRow, build_residual_dataframe
from src.correction.walk_forward_residuals import Fold, WalkForwardResidualBuilder


@pytest.fixture
def sample_data_three_seasons():
    """Create a DataFrame with three strictly non-overlapping seasons."""
    np.random.seed(42)
    seasons = [
        ("22020", pd.date_range("2020-10-01", periods=150, freq="D"), 700),
        ("22021", pd.date_range("2021-10-01", periods=150, freq="D"), 700),
        ("22022", pd.date_range("2022-10-01", periods=150, freq="D"), 600),
    ]
    rows = []
    for season_id, dates, n in seasons:
        rows.append(
            pd.DataFrame(
                {
                    "PLAYER_ID": np.random.randint(1000, 1050, n),
                    "PLAYER_NAME": [f"Player_{i}" for i in range(n)],
                    "TEAM_ID": np.random.randint(1610612737, 1610612767, n),
                    "TEAM_ABBREVIATION": np.random.choice(["LAL", "BOS", "GSW", "MIA"], n),
                    "GAME_ID": np.random.randint(10000000, 20000000, n),
                    "GAME_DATE": np.random.choice(dates, n),
                    "MATCHUP": ["vs. OPP"] * n,
                    "OPPONENT_ID": np.random.randint(1610612737, 1610612767, n),
                    "OPPONENT_ABBR": np.random.choice(["BOS", "LAL", "MIA", "GSW"], n),
                    "WL": np.random.choice(["W", "L"], n),
                    "MIN": np.random.uniform(10, 40, n),
                    "PTS": np.random.poisson(12, n),
                    "REB": np.random.poisson(5, n),
                    "AST": np.random.poisson(3, n),
                    "STL": np.random.poisson(1, n),
                    "BLK": np.random.poisson(0.5, n),
                    "TOV": np.random.poisson(2, n),
                    "SEASON_ID": [season_id] * n,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def test_build_folds_creates_chronological_folds(sample_data_three_seasons):
    builder = WalkForwardResidualBuilder(
        config_path="config/default.yaml",
        min_train_seasons=1,
    )
    df = sample_data_three_seasons.copy()
    folds = builder.build_folds(df)
    assert len(folds) == 2
    assert folds[0].holdout_season == "22021"
    assert folds[1].holdout_season == "22022"
    assert folds[0].train_df["SEASON_ID"].astype(str).nunique() == 1
    assert folds[0].holdout_df["SEASON_ID"].astype(str).nunique() == 1


def test_walk_forward_split_has_no_future_leakage(sample_data_three_seasons):
    builder = WalkForwardResidualBuilder(
        config_path="config/default.yaml",
        min_train_seasons=1,
    )
    df = sample_data_three_seasons.copy()
    folds = builder.build_folds(df)
    for fold in folds:
        assert fold.train_df["GAME_DATE"].max() < fold.holdout_df["GAME_DATE"].min()


def test_residual_error_calculation():
    builder = WalkForwardResidualBuilder(config_path="config/default.yaml")
    test_df = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "GAME_DATE": ["2021-01-01", "2021-01-02"],
            "PLAYER_ID": ["p1", "p2"],
            "PLAYER_NAME": ["A", "B"],
            "TEAM_ID": ["t1", "t2"],
            "OPPONENT_ABBR": ["o1", "o2"],
            "PTS": [10.0, 20.0],
            "REB": [5.0, 7.0],
        }
    )
    preds_df = pd.DataFrame(
        {
            "PTS": [8.0, 22.0],
            "REB": [4.0, 6.0],
        }
    )
    rows = builder._build_residual_rows(test_df, preds_df, "fold_1", "2021-01-01")
    assert len(rows) == 4  # 2 players x 2 stats
    for row in rows:
        assert row.error == row.actual - row.base_prediction
    pts_rows = [r for r in rows if r.stat == "PTS"]
    assert pts_rows[0].error == 10.0 - 8.0
    assert pts_rows[1].error == 20.0 - 22.0


def test_residual_dataset_has_required_columns():
    rows = [
        ResidualTrainingRow(
            game_id="g1",
            game_date="2021-01-01",
            player_id="p1",
            player_name="A",
            team_id="t1",
            opponent="o1",
            stat="PTS",
            base_prediction=8.0,
            actual=10.0,
            error=2.0,
            model_fold="fold_1",
            model_version="v1",
            data_quality="FULL",
            feature_cutoff_date="2021-01-01",
        ),
    ]
    df = build_residual_dataframe(rows)
    required = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "OPPONENT",
        "STAT",
        "BASE_PREDICTION",
        "ACTUAL",
        "ERROR",
        "MODEL_FOLD",
        "FEATURE_CUTOFF_DATE",
    ]
    for col in required:
        assert col in df.columns
    assert "MODEL_VERSION" in df.columns
    assert "DATA_QUALITY" in df.columns
    assert df["ERROR"].iloc[0] == 2.0


def test_residual_dataset_parquet_roundtrip(tmp_path):
    rows = [
        ResidualTrainingRow(
            game_id="g1",
            game_date="2021-01-01",
            player_id="p1",
            player_name="A",
            team_id="t1",
            opponent="o1",
            stat="PTS",
            base_prediction=8.0,
            actual=10.0,
            error=2.0,
            model_fold="fold_1",
            model_version=None,
            data_quality=None,
            feature_cutoff_date="2021-01-01",
        ),
    ]
    df = build_residual_dataframe(rows)
    path = tmp_path / "residuals.parquet"
    df.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    assert set(loaded.columns) == set(df.columns)
    assert loaded["ERROR"].iloc[0] == 2.0


def test_residual_summary_json(tmp_path, sample_data_three_seasons):
    builder = WalkForwardResidualBuilder(
        config_path="config/default.yaml",
        output_path=str(tmp_path / "residuals.parquet"),
        summary_path=str(tmp_path / "summary.json"),
        min_train_seasons=1,
        targets=["PTS", "REB"],
    )

    # Monkeypatch data loading and _process_fold to bypass heavy I/O / training
    builder._load_and_engineer_data = lambda: sample_data_three_seasons.copy()

    def mock_process_fold(fold):
        return [
            ResidualTrainingRow(
                game_id="g1",
                game_date="2021-01-01",
                player_id="p1",
                player_name="A",
                team_id="t1",
                opponent="o1",
                stat="PTS",
                base_prediction=8.0,
                actual=10.0,
                error=2.0,
                model_fold=fold.name,
                model_version=None,
                data_quality=None,
                feature_cutoff_date="2021-01-01",
            ),
            ResidualTrainingRow(
                game_id="g1",
                game_date="2021-01-01",
                player_id="p1",
                player_name="A",
                team_id="t1",
                opponent="o1",
                stat="REB",
                base_prediction=4.0,
                actual=5.0,
                error=1.0,
                model_fold=fold.name,
                model_version=None,
                data_quality=None,
                feature_cutoff_date="2021-01-01",
            ),
        ]

    builder._process_fold = mock_process_fold
    residual_df = builder.run()
    assert len(residual_df) == 4  # 2 folds x 2 stats

    summary_path = Path(builder.summary_path)
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["num_folds"] == 2
    assert summary["num_residual_rows"] == 4
    assert summary["targets"] == ["PTS", "REB"]
    assert "mean_absolute_error_by_stat" in summary
    assert summary["mean_absolute_error_by_stat"]["PTS"] == 2.0
    assert summary["mean_absolute_error_by_stat"]["REB"] == 1.0
