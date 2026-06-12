"""Tests for CorrectionApplier and runtime correction integration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.correction.correction_applier import CorrectionApplier, TARGETS
from src.correction.correction_features import CorrectionFeatureBuilder
from src.correction.residual_model import ResidualCorrectionModel


@pytest.fixture
def feature_builder():
    return CorrectionFeatureBuilder()


@pytest.fixture
def base_predictions():
    return {
        "PTS": 24.5,
        "REB": 8.0,
        "AST": 5.5,
        "STL": 1.2,
        "BLK": 0.8,
        "TOV": 2.5,
    }


class _FakeResidualModel:
    """Fake residual model that returns a fixed correction for PTS."""

    def __init__(self, correction=2.1, enabled_stats=None):
        self._correction = correction
        self._enabled = ["PTS"] if enabled_stats is None else list(enabled_stats)
        self._feature_cols = [
            "BASE_PREDICTION",
            "DATA_QUALITY_SCORE",
            "RECENT_PLAYER_ERROR_MEAN",
            "RECENT_PLAYER_ERROR_ABS_MEAN",
            "RECENT_STAT_ERROR_MEAN",
            "RECENT_STAT_ERROR_ABS_MEAN",
        ]

    def is_enabled(self, stat):
        return stat in self._enabled

    def predict_correction(self, stat, feature_row):
        if not self.is_enabled(stat):
            return 0.0
        return self._correction

    @property
    def feature_cols(self):
        return list(self._feature_cols)


def test_correction_applier_adds_accepted_correction(feature_builder, base_predictions):
    """Accepted residual model should apply correction."""
    residual_model = _FakeResidualModel(correction=2.1, enabled_stats=["PTS"])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    corrected, meta = applier.apply(base_predictions)

    assert corrected["PTS"] == pytest.approx(26.6, abs=0.01)
    assert meta["PTS"]["residual_applied"] is True
    assert meta["PTS"]["residual_correction"] == pytest.approx(2.1, abs=0.01)


def test_rejected_residual_model_returns_correction_zero(feature_builder, base_predictions):
    """Stats not in enabled set should get 0.0 correction."""
    residual_model = _FakeResidualModel(correction=2.1, enabled_stats=[])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    corrected, meta = applier.apply(base_predictions)

    assert corrected["PTS"] == pytest.approx(24.5, abs=0.01)
    assert meta["PTS"]["residual_applied"] is False


def test_missing_residual_model_does_not_crash(feature_builder, base_predictions):
    """When residual model has no enabled stats, all corrections are 0.0."""
    residual_model = _FakeResidualModel(correction=0.0, enabled_stats=[])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    corrected, meta = applier.apply(base_predictions)

    for stat in TARGETS:
        assert corrected[stat] == pytest.approx(base_predictions[stat], abs=0.01)
        assert meta[stat]["residual_applied"] is False


def test_corrected_prediction_clipped_at_zero(feature_builder):
    """Predictions should never go below clip_min (default 0.0)."""
    residual_model = _FakeResidualModel(correction=-100.0, enabled_stats=["PTS"])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    base = {"PTS": 10.0, "REB": 5.0, "AST": 3.0, "STL": 1.0, "BLK": 0.5, "TOV": 2.0}
    corrected, meta = applier.apply(base)

    assert corrected["PTS"] == 0.0
    assert meta["PTS"]["corrected_prediction"] == 0.0


def test_correction_metadata_structure(feature_builder, base_predictions):
    """Correction meta should have all required keys."""
    residual_model = _FakeResidualModel(correction=1.5, enabled_stats=["PTS", "REB"])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    _, meta = applier.apply(base_predictions)

    for stat in TARGETS:
        assert "base_prediction" in meta[stat]
        assert "residual_correction" in meta[stat]
        assert "corrected_prediction" in meta[stat]
        assert "residual_applied" in meta[stat]
        assert isinstance(meta[stat]["residual_applied"], bool)


def test_multiple_stats_get_corrections(feature_builder, base_predictions):
    """Multiple accepted stats should all get corrections."""
    residual_model = _FakeResidualModel(correction=1.0, enabled_stats=["PTS", "REB", "AST"])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    corrected, meta = applier.apply(base_predictions)

    assert corrected["PTS"] == pytest.approx(25.5, abs=0.01)
    assert corrected["REB"] == pytest.approx(9.0, abs=0.01)
    assert corrected["AST"] == pytest.approx(6.5, abs=0.01)
    assert meta["STL"]["residual_applied"] is False


def test_context_row_passed_to_feature_builder(feature_builder, base_predictions):
    """Context row features should flow through to the feature builder."""
    captured = {}

    class CapturingResidualModel(_FakeResidualModel):
        def __init__(self):
            super().__init__(enabled_stats=["PTS"])
            self._feature_cols = [
                "BASE_PREDICTION",
                "DATA_QUALITY_SCORE",
                "RECENT_PLAYER_ERROR_MEAN",
                "RECENT_PLAYER_ERROR_ABS_MEAN",
                "RECENT_STAT_ERROR_MEAN",
                "RECENT_STAT_ERROR_ABS_MEAN",
                "MINUTES_CONFIDENCE",
                "ROLLING_MINUTES",
            ]

        def predict_correction(self, stat, feature_row):
            captured["feature_row"] = feature_row
            return 1.0

    residual_model = CapturingResidualModel()
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    context = pd.DataFrame({"MINUTES_CONFIDENCE": [0.85], "ROLLING_MINUTES": [32.0]})
    applier.apply(base_predictions, context_row=context)

    assert "feature_row" in captured
    row = captured["feature_row"]
    assert row["MINUTES_CONFIDENCE"].iloc[0] == pytest.approx(0.85, abs=0.01)
    assert row["ROLLING_MINUTES"].iloc[0] == pytest.approx(32.0, abs=0.01)


def test_applier_with_empty_base_predictions(feature_builder):
    """Applier should handle missing stats gracefully."""
    residual_model = _FakeResidualModel(correction=1.0, enabled_stats=["PTS"])
    applier = CorrectionApplier(
        residual_model=residual_model,
        feature_builder=feature_builder,
    )

    corrected, meta = applier.apply({})

    assert corrected["PTS"] == pytest.approx(1.0, abs=0.01)


def test_build_runtime_row_produces_correct_shape(feature_builder):
    """build_runtime_row should return a single-row DataFrame."""
    row = feature_builder.build_runtime_row(
        stat="PTS",
        base_prediction=24.5,
    )
    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
    assert "BASE_PREDICTION" in row.columns


def test_build_runtime_row_uses_context(feature_builder):
    """build_runtime_row should pull values from context_row when available."""
    context = pd.DataFrame({"MINUTES_CONFIDENCE": [0.9], "REST_DAYS": [2.0]})
    row = feature_builder.build_runtime_row(
        stat="PTS",
        base_prediction=24.5,
        context_row=context,
        feature_cols=["BASE_PREDICTION", "MINUTES_CONFIDENCE", "REST_DAYS"],
    )
    assert row["MINUTES_CONFIDENCE"].iloc[0] == pytest.approx(0.9, abs=0.01)
    assert row["REST_DAYS"].iloc[0] == pytest.approx(2.0, abs=0.01)


def test_build_runtime_row_default_feature_cols():
    """Default feature_cols should include the standard correction features."""
    builder = CorrectionFeatureBuilder()
    row = builder.build_runtime_row(stat="PTS", base_prediction=10.0)
    assert "BASE_PREDICTION" in row.columns
    assert "DATA_QUALITY_SCORE" in row.columns
    assert "RECENT_PLAYER_ERROR_MEAN" in row.columns


def test_model_manager_loads_without_residual_models(tmp_path):
    """ModelManager should initialize cleanly when no residual dir exists."""
    from src.models.model_manager import ModelManager

    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = str(tmp_path)
    manager.residual_correction_model = None
    manager.correction_applier = None
    manager.residual_corrections_enabled = False
    manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

    manager._load_residual_corrections()

    assert manager.residual_corrections_enabled is False
    assert manager.correction_applier is None


def test_model_manager_predict_without_residual(tmp_path):
    """predict_player_stats should work without residual corrections."""
    import pickle
    import numpy as np
    from src.models.model_manager import ModelManager

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(["f1"], f)

    class FakeModel:
        feature_names_ = ["f1"]
        def predict(self, X):
            return np.array([10.0])

    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = str(models_dir)
    manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    manager.models = {t: FakeModel() for t in manager.targets}
    manager.catboost_mae_models = {}
    manager.catboost_quantile_models = {}
    manager.transformer_model = None
    manager.blend_weights = {}
    manager.ensemble_weights = None
    manager.feature_cols = ["f1"]
    manager.feature_schema = None
    manager.feature_selector = None
    manager.residual_corrections_enabled = False
    manager.correction_applier = None
    manager._FALLBACK_VALUES = ModelManager._FALLBACK_VALUES

    df = pd.DataFrame({"f1": [1.0]})
    result = manager.predict_player_stats(df)

    assert result["PTS"] == 10.0
    assert result["REB"] == 10.0


def test_batch_prediction_preserves_row_count(tmp_path):
    """predict_player_stats_batch should return one row per input row."""
    import pickle
    import numpy as np
    from src.models.model_manager import ModelManager

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(["f1"], f)

    class FakeModel:
        feature_names_ = ["f1"]
        def predict(self, X):
            return np.array([10.0])

    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = str(models_dir)
    manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    manager.models = {t: FakeModel() for t in manager.targets}
    manager.catboost_mae_models = {}
    manager.catboost_quantile_models = {}
    manager.transformer_model = None
    manager.blend_weights = {}
    manager.ensemble_weights = None
    manager.feature_cols = ["f1"]
    manager.feature_schema = None
    manager.feature_selector = None
    manager.residual_corrections_enabled = False
    manager.correction_applier = None
    manager._FALLBACK_VALUES = ModelManager._FALLBACK_VALUES

    context_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "PLAYER_ID": [101, 102, 103]})
    result = manager.predict_player_stats_batch(context_df)

    assert len(result) == 3


def test_disabled_config_returns_base_unchanged(tmp_path):
    """When residual_corrections_enabled is False, base predictions pass through."""
    import pickle
    import numpy as np
    from src.models.model_manager import ModelManager

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(["f1"], f)

    class FakeModel:
        feature_names_ = ["f1"]
        def predict(self, X):
            return np.array([25.0])

    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = str(models_dir)
    manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    manager.models = {t: FakeModel() for t in manager.targets}
    manager.catboost_mae_models = {}
    manager.catboost_quantile_models = {}
    manager.transformer_model = None
    manager.blend_weights = {}
    manager.ensemble_weights = None
    manager.feature_cols = ["f1"]
    manager.feature_schema = None
    manager.feature_selector = None
    manager.residual_corrections_enabled = False
    manager.correction_applier = None
    manager._FALLBACK_VALUES = ModelManager._FALLBACK_VALUES

    df = pd.DataFrame({"f1": [1.0]})
    result = manager.predict_player_stats(df)

    for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
        assert result[stat] == 25.0


def test_correction_applier_import():
    """CorrectionApplier can be imported from src.correction."""
    from src.correction import CorrectionApplier as CA

    assert CA is CorrectionApplier


def test_residual_correction_model_import():
    """ResidualCorrectionModel can be imported from src.correction."""
    from src.correction import ResidualCorrectionModel as RCM

    assert RCM is ResidualCorrectionModel
