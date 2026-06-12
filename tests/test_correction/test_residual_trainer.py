"""Tests for residual model trainer, correction features, and runtime loader."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.correction.correction_features import CorrectionFeatureBuilder
from src.correction.residual_model import ResidualCorrectionModel
from src.correction.correction_store import CorrectionStore


@pytest.fixture
def sample_residual_df():
    """Create a synthetic residual DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    rows = []
    for stat in ["PTS", "REB", "AST"]:
        for i in range(n):
            base_pred = np.random.uniform(5, 30)
            actual = base_pred + np.random.normal(0, 3)
            rows.append({
                "GAME_ID": f"g{i}",
                "GAME_DATE": dates[i % len(dates)],
                "PLAYER_ID": f"p{i % 20}",
                "PLAYER_NAME": f"Player_{i % 20}",
                "TEAM_ID": f"t{i % 5}",
                "OPPONENT": f"o{i % 10}",
                "STAT": stat,
                "BASE_PREDICTION": base_pred,
                "ACTUAL": actual,
                "ERROR": actual - base_pred,
                "MODEL_FOLD": "fold_1",
                "MODEL_VERSION": None,
                "DATA_QUALITY": np.random.choice(["FULL", "PARTIAL", "DEGRADED"]),
                "FEATURE_CUTOFF_DATE": "2023-01-01",
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


@pytest.fixture
def builder():
    return CorrectionFeatureBuilder()


def test_recent_error_features_do_not_leak_current_row(builder, sample_residual_df):
    """First row per player/stat group must have NaN rolling features."""
    features, cols = builder.build(sample_residual_df)

    for stat in ["PTS", "REB", "AST"]:
        stat_df = features[features["STAT"] == stat].sort_values("GAME_DATE")
        for player_id in stat_df["PLAYER_ID"].unique():
            player_rows = stat_df[stat_df["PLAYER_ID"] == player_id]
            if len(player_rows) > 0:
                first_row = player_rows.iloc[0]
                assert pd.isna(first_row["RECENT_PLAYER_ERROR_MEAN"]) or first_row["RECENT_PLAYER_ERROR_MEAN"] == 0.0


def test_correction_features_returns_expected_columns(builder, sample_residual_df):
    features, cols = builder.build(sample_residual_df)

    required = [
        "BASE_PREDICTION",
        "DATA_QUALITY_SCORE",
        "RECENT_PLAYER_ERROR_MEAN",
        "RECENT_PLAYER_ERROR_ABS_MEAN",
        "RECENT_STAT_ERROR_MEAN",
        "RECENT_STAT_ERROR_ABS_MEAN",
    ]
    for col in required:
        assert col in cols


def test_data_quality_encoding(builder, sample_residual_df):
    features, cols = builder.build(sample_residual_df)

    assert "DATA_QUALITY_SCORE" in features.columns
    assert features["DATA_QUALITY_SCORE"].between(0.0, 1.0).all()


def test_missing_optional_features_do_not_crash(builder):
    """Builder should fill safe defaults for missing optional columns."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "GAME_ID": [f"g{i}" for i in range(n)],
        "GAME_DATE": pd.date_range("2023-01-01", periods=n, freq="D"),
        "PLAYER_ID": [f"p{i % 5}" for i in range(n)],
        "STAT": ["PTS"] * n,
        "BASE_PREDICTION": np.random.uniform(5, 30, n),
        "ACTUAL": np.random.uniform(5, 30, n),
        "ERROR": np.random.normal(0, 3, n),
    })

    features, cols = builder.build(df)
    assert len(features) == n
    assert len(cols) > 0


def test_correction_features_no_nan_in_output(builder, sample_residual_df):
    """All NaN values should be filled with 0.0."""
    features, cols = builder.build(sample_residual_df)

    for col in cols:
        assert not features[col].isna().any(), f"Column {col} has NaN values"


def test_residual_trainer_creates_artifacts(tmp_path, sample_residual_df):
    """Trainer should create one .cbm file per stat plus metadata."""
    from src.correction.residual_trainer import ResidualModelTrainer

    parquet_path = tmp_path / "residual_training.parquet"
    sample_residual_df.to_parquet(parquet_path, index=False)

    output_dir = tmp_path / "models" / "residual"

    trainer = ResidualModelTrainer(min_rows=50, iterations=10, early_stopping_rounds=5)
    result = trainer.train_all(str(parquet_path), str(output_dir))

    assert (output_dir / "residual_metadata.json").exists()
    assert (output_dir / "residual_feature_schema.json").exists()

    for stat in ["PTS", "REB", "AST"]:
        assert (output_dir / f"{stat.lower()}_residual.cbm").exists()
        assert stat in result.targets


def test_corrected_mae_calculation():
    """Verify the corrected MAE formula: abs(ACTUAL - (BASE + CORRECTION))."""
    actual = np.array([10.0, 20.0, 30.0])
    base_pred = np.array([8.0, 22.0, 28.0])
    correction = np.array([1.5, -1.5, 1.0])

    base_mae = float(np.mean(np.abs(actual - base_pred)))
    corrected_mae = float(np.mean(np.abs(actual - (base_pred + correction))))

    assert base_mae == pytest.approx(2.0, abs=0.01)
    assert corrected_mae < base_mae


def test_bad_residual_model_marked_rejected(tmp_path):
    """If corrected MAE is worse, the model should be rejected."""
    from src.correction.residual_trainer import ResidualModelTrainer, ResidualTargetResult

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    rows = []
    for i in range(n):
        base_pred = 10.0
        actual = base_pred + np.random.normal(0, 0.1)
        rows.append({
            "GAME_ID": f"g{i}",
            "GAME_DATE": dates[i % len(dates)],
            "PLAYER_ID": f"p{i % 5}",
            "PLAYER_NAME": f"Player_{i % 5}",
            "TEAM_ID": "t1",
            "OPPONENT": "o1",
            "STAT": "PTS",
            "BASE_PREDICTION": base_pred,
            "ACTUAL": actual,
            "ERROR": actual - base_pred,
            "MODEL_FOLD": "fold_1",
            "MODEL_VERSION": None,
            "DATA_QUALITY": "FULL",
            "FEATURE_CUTOFF_DATE": "2023-01-01",
        })
    df = pd.DataFrame(rows).sort_values("GAME_DATE").reset_index(drop=True)

    parquet_path = tmp_path / "residual.parquet"
    df.to_parquet(parquet_path, index=False)

    output_dir = tmp_path / "models" / "residual"
    trainer = ResidualModelTrainer(min_rows=50, iterations=10, early_stopping_rounds=5)
    result = trainer.train_all(str(parquet_path), str(output_dir), targets=["PTS"])

    metadata_path = output_dir / "residual_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert "PTS" in metadata["targets"]
    assert metadata["targets"]["PTS"]["status"] in ("accepted", "rejected")


def test_residual_loader_returns_zero_for_missing_model():
    """Loader returns 0.0 when no model directory exists."""
    loader = ResidualCorrectionModel()
    loader.load(model_dir="/nonexistent/path")

    feature_row = pd.DataFrame({"BASE_PREDICTION": [10.0]})
    assert loader.predict_correction("PTS", feature_row) == 0.0


def test_residual_loader_returns_zero_for_rejected_stat(tmp_path):
    """Loader returns 0.0 for stats marked as rejected in metadata."""
    model_dir = tmp_path / "residual"
    model_dir.mkdir()

    metadata = {
        "targets": {
            "PTS": {"status": "rejected", "reason": "test"},
        },
        "feature_cols": ["BASE_PREDICTION"],
    }
    (model_dir / "residual_metadata.json").write_text(json.dumps(metadata))

    schema = {"version": 1, "feature_cols": ["BASE_PREDICTION"]}
    (model_dir / "residual_feature_schema.json").write_text(json.dumps(schema))

    loader = ResidualCorrectionModel()
    loader.load(str(model_dir))

    feature_row = pd.DataFrame({"BASE_PREDICTION": [10.0]})
    assert loader.predict_correction("PTS", feature_row) == 0.0
    assert not loader.is_enabled("PTS")


def test_residual_loader_is_enabled_for_missing_stat():
    """is_enabled returns False for stats not in metadata."""
    loader = ResidualCorrectionModel()
    assert not loader.is_enabled("PTS")


def test_correction_store_get_accepted_stats(tmp_path):
    store = CorrectionStore(str(tmp_path / "residual"))

    metadata = {
        "targets": {
            "PTS": {"status": "accepted"},
            "REB": {"status": "rejected"},
            "AST": {"status": "accepted"},
        }
    }
    store.save_metadata(metadata)

    assert sorted(store.get_accepted_stats()) == ["AST", "PTS"]
    assert store.get_rejected_stats() == ["REB"]


def test_correction_store_empty_dir(tmp_path):
    store = CorrectionStore(str(tmp_path / "empty"))
    assert store.load_metadata() is None
    assert store.get_accepted_stats() == []


def test_residual_metadata_json_structure(tmp_path, sample_residual_df):
    """Verify metadata JSON has the expected structure."""
    from src.correction.residual_trainer import ResidualModelTrainer

    parquet_path = tmp_path / "residual.parquet"
    sample_residual_df.to_parquet(parquet_path, index=False)

    output_dir = tmp_path / "models" / "residual"
    trainer = ResidualModelTrainer(min_rows=50, iterations=10, early_stopping_rounds=5)
    trainer.train_all(str(parquet_path), str(output_dir))

    metadata = json.loads((output_dir / "residual_metadata.json").read_text())

    assert "targets" in metadata
    assert "feature_cols" in metadata
    assert "trained_at" in metadata

    for stat, meta in metadata["targets"].items():
        assert "rows" in meta
        assert "base_mae" in meta
        assert "corrected_mae" in meta
        assert "mae_improvement" in meta
        assert "mae_improvement_pct" in meta
        assert "status" in meta
        assert meta["status"] in ("accepted", "rejected")


def test_feature_schema_json_structure(tmp_path, sample_residual_df):
    """Verify feature schema JSON has the expected structure."""
    from src.correction.residual_trainer import ResidualModelTrainer

    parquet_path = tmp_path / "residual.parquet"
    sample_residual_df.to_parquet(parquet_path, index=False)

    output_dir = tmp_path / "models" / "residual"
    trainer = ResidualModelTrainer(min_rows=50, iterations=10, early_stopping_rounds=5)
    trainer.train_all(str(parquet_path), str(output_dir))

    schema = json.loads((output_dir / "residual_feature_schema.json").read_text())

    assert "version" in schema
    assert "feature_cols" in schema
    assert "created_at" in schema
    assert isinstance(schema["feature_cols"], list)
    assert len(schema["feature_cols"]) > 0


def test_trainer_skips_stat_with_insufficient_rows(tmp_path):
    """Stats with fewer rows than min_rows should be rejected."""
    from src.correction.residual_trainer import ResidualModelTrainer

    np.random.seed(42)
    n = 10
    df = pd.DataFrame({
        "GAME_ID": [f"g{i}" for i in range(n)],
        "GAME_DATE": pd.date_range("2023-01-01", periods=n, freq="D"),
        "PLAYER_ID": [f"p{i}" for i in range(n)],
        "PLAYER_NAME": [f"Player_{i}" for i in range(n)],
        "TEAM_ID": ["t1"] * n,
        "OPPONENT": ["o1"] * n,
        "STAT": ["PTS"] * n,
        "BASE_PREDICTION": np.random.uniform(5, 30, n),
        "ACTUAL": np.random.uniform(5, 30, n),
        "ERROR": np.random.normal(0, 3, n),
        "MODEL_FOLD": ["fold_1"] * n,
        "MODEL_VERSION": [None] * n,
        "DATA_QUALITY": ["FULL"] * n,
        "FEATURE_CUTOFF_DATE": ["2023-01-01"] * n,
    })

    parquet_path = tmp_path / "residual.parquet"
    df.to_parquet(parquet_path, index=False)

    output_dir = tmp_path / "models" / "residual"
    trainer = ResidualModelTrainer(min_rows=1000)
    result = trainer.train_all(str(parquet_path), str(output_dir), targets=["PTS"])

    assert result.targets["PTS"].status == "rejected"
    assert "insufficient rows" in result.targets["PTS"].reason
