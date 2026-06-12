"""Test models module."""

import pickle
import numpy as np
import pandas as pd
import pytest


def _create_feature_schema(models_dir, feature_cols):
    """Create a minimal feature_cols.pkl for tests."""
    path = models_dir / "feature_cols.pkl"
    with open(path, "wb") as f:
        pickle.dump(feature_cols, f)


class TestModelManager:
    """Tests for ModelManager class."""

    def test_model_manager_import(self):
        """Test ModelManager can be imported."""
        from src.models.model_manager import ModelManager

        assert ModelManager is not None

    def test_model_manager_initialization(self, temp_data_dir):
        """Test ModelManager initializes with correct directories."""
        from src.models.model_manager import ModelManager

        manager = ModelManager(
            data_dir=str(temp_data_dir["data_dir"]),
            models_dir=str(temp_data_dir["models_dir"]),
        )

        assert manager.data_dir == str(temp_data_dir["data_dir"])
        assert manager.models_dir == str(temp_data_dir["models_dir"])
        assert manager.targets == ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

    def test_model_manager_invalid_data_dir(self):
        """Test ModelManager raises error for invalid data_dir."""
        from src.models.model_manager import ModelManager

        with pytest.raises(ValueError):
            ModelManager(data_dir="", models_dir="models")

        with pytest.raises(ValueError):
            ModelManager(data_dir=None, models_dir="models")

    def test_model_manager_prepare_data_missing_dir(self, temp_data_dir):
        """Test ModelManager raises error when data directory is empty."""
        from src.models.model_manager import ModelManager

        manager = ModelManager(
            data_dir=str(temp_data_dir["data_dir"]),
            models_dir=str(temp_data_dir["models_dir"]),
        )

        with pytest.raises(ValueError, match="Players file not found"):
            manager.prepare_data()

    def test_residual_correction_attributes_initialized(self, temp_data_dir):
        """ModelManager should initialize residual correction attributes."""
        from src.models.model_manager import ModelManager

        manager = ModelManager(
            data_dir=str(temp_data_dir["data_dir"]),
            models_dir=str(temp_data_dir["models_dir"]),
        )

        assert manager.residual_correction_model is None
        assert manager.correction_applier is None
        assert manager.residual_corrections_enabled is False

    def test_residual_corrections_not_loaded_when_dir_missing(self, tmp_path):
        """_load_residual_corrections should be a no-op when residual dir absent."""
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
        assert manager.residual_correction_model is None


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_registry_import(self):
        """Test ModelRegistry can be imported."""
        from src.models.base import ModelRegistry

        assert ModelRegistry is not None

    def test_registry_initialization(self, temp_data_dir):
        """Test ModelRegistry initializes correctly."""
        from src.models.base import ModelRegistry

        registry = ModelRegistry(str(temp_data_dir["models_dir"]))
        assert registry.models_dir == str(temp_data_dir["models_dir"])

    def test_registry_list_models_empty(self, mock_model_registry):
        """Test ModelRegistry list_models returns empty when no models."""
        models = mock_model_registry.list_models()
        assert isinstance(models, list)
        assert len(models) == 0


class TestFallbackPredictor:
    """Tests for fallback prediction logic."""

    def test_fallback_prediction_import(self):
        """Test that fallback prediction utilities can be accessed."""
        from src.models.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

        df = pd.DataFrame({"PTS": [10], "ROLL_PTS_AVG_10": [12]})

        result = manager._get_fallback_value(df, "PTS")
        assert result == 12

    def test_fallback_prediction_league_avg(self):
        """Test fallback prediction uses league average when no history."""
        from src.models.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

        df = pd.DataFrame()

        result = manager._get_fallback_value(df, "PTS")
        assert result == 10.0

        result = manager._get_fallback_value(df, "REB")
        assert result == 4.5

    def test_transformer_prediction_uses_all_targets(self):
        """Test transformer predictions map to all six output targets."""
        from src.models.model_manager import ModelManager

        class DummyTransformer:
            seq_len = 2

            def predict(self, seq):
                assert seq.shape == (2, 2)
                return np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)

        manager = ModelManager.__new__(ModelManager)
        manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
        manager.feature_cols = ["f1", "f2"]
        manager.transformer_model = DummyTransformer()

        history_df = pd.DataFrame({"f1": [10, 20], "f2": [30, 40]})

        assert manager._predict_transformer_target("PTS", history_df) == 1.0
        assert manager._predict_transformer_target("STL", history_df) == 4.0
        assert manager._predict_transformer_target("TOV", history_df) == 6.0


class TestBlendContractEnforcement:
    """Tests for the blend-weight / Transformer artifact contract."""

    def test_missing_transformer_raises_when_blend_weights_expect_it(self):
        from src.models.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.models_dir = "/tmp/fake_models"
        manager.transformer_model = None
        manager.blend_weights = {
            "PTS": {"catboost": 0.7, "transformer": 0.3},
            "REB": {"catboost": 0.8, "transformer": 0.2},
        }

        with pytest.raises(FileNotFoundError, match="attention_transformer.pkl"):
            manager._validate_blend_contract()

    def test_missing_transformer_raises_runtime_error_when_file_exists_but_load_failed(
        self, tmp_path
    ):
        from src.models.model_manager import ModelManager

        (tmp_path / "attention_transformer.pkl").write_text("corrupt", encoding="utf-8")

        manager = ModelManager.__new__(ModelManager)
        manager.models_dir = str(tmp_path)
        manager.transformer_model = None
        manager.blend_weights = {
            "PTS": {"catboost": 0.6, "transformer": 0.4},
        }

        with pytest.raises(
            RuntimeError, match="attention_transformer.pkl.*failed to load"
        ):
            manager._validate_blend_contract()

    def test_no_error_when_blend_weights_have_zero_transformer_weight(self):
        from src.models.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.models_dir = "/tmp/fake_models"
        manager.transformer_model = None
        manager.blend_weights = {
            "PTS": {"catboost": 1.0, "transformer": 0.0},
            "REB": {"catboost": 1.0, "transformer": 0.0},
        }

        manager._validate_blend_contract()

    def test_no_error_when_transformer_loaded_and_blend_weights_expect_it(self):
        from src.models.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.models_dir = "/tmp/fake_models"
        manager.transformer_model = object()
        manager.blend_weights = {
            "PTS": {"catboost": 0.7, "transformer": 0.3},
        }

        manager._validate_blend_contract()

    def test_no_error_when_no_blend_weights(self):
        from src.models.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.models_dir = "/tmp/fake_models"
        manager.transformer_model = None
        manager.blend_weights = {}

        manager._validate_blend_contract()

    def test_predict_player_stats_does_not_apply_partial_blend(self, tmp_path):
        from src.models.model_manager import ModelManager

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _create_feature_schema(models_dir, ["f1", "f2"])

        class FakeCatBoost:
            def __init__(self, value):
                self.value = value
                self.feature_names_ = ["f1", "f2"]

            def predict(self, X):
                return np.array([self.value], dtype=np.float32)

        manager = ModelManager.__new__(ModelManager)
        manager.models_dir = str(models_dir)
        manager.targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
        manager.models = {
            t: FakeCatBoost(10.0 + i) for i, t in enumerate(manager.targets)
        }
        manager.catboost_mae_models = {}
        manager.catboost_quantile_models = {}
        manager.transformer_model = None
        manager.blend_weights = {}
        manager.feature_cols = ["f1", "f2"]
        manager.feature_schema = None
        manager.feature_selector = None
        manager.residual_corrections_enabled = False
        manager.correction_applier = None
        manager._FALLBACK_VALUES = ModelManager._FALLBACK_VALUES

        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})

        result = manager.predict_player_stats(df)

        assert result["PTS"] == 10.0
        assert result["REB"] == 11.0


class TestResidualCorrectionIntegration:
    """Tests for residual correction integration in ModelManager."""

    def _make_manager(self, tmp_path, residual_enabled=False, correction_applier=None):
        from src.models.model_manager import ModelManager

        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        _create_feature_schema(models_dir, ["f1"])

        class FakeModel:
            feature_names_ = ["f1"]
            def predict(self, X):
                return np.array([20.0])

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
        manager.residual_corrections_enabled = residual_enabled
        manager.correction_applier = correction_applier
        manager._FALLBACK_VALUES = ModelManager._FALLBACK_VALUES
        return manager

    def test_predict_without_residual_returns_base(self, tmp_path):
        """When corrections disabled, predictions equal base values."""
        manager = self._make_manager(tmp_path, residual_enabled=False)
        df = pd.DataFrame({"f1": [1.0]})
        result = manager.predict_player_stats(df)

        for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
            assert result[stat] == 20.0

    def test_predict_with_residual_applies_correction(self, tmp_path):
        """When corrections enabled, predictions should be adjusted."""
        from src.correction.correction_applier import CorrectionApplier
        from src.correction.correction_features import CorrectionFeatureBuilder

        class FakeResidualModel:
            _feature_cols = [
                "BASE_PREDICTION", "DATA_QUALITY_SCORE",
                "RECENT_PLAYER_ERROR_MEAN", "RECENT_PLAYER_ERROR_ABS_MEAN",
                "RECENT_STAT_ERROR_MEAN", "RECENT_STAT_ERROR_ABS_MEAN",
            ]
            def is_enabled(self, stat):
                return stat == "PTS"
            def predict_correction(self, stat, feature_row):
                return 3.0 if stat == "PTS" else 0.0
            @property
            def feature_cols(self):
                return list(self._feature_cols)

        applier = CorrectionApplier(
            residual_model=FakeResidualModel(),
            feature_builder=CorrectionFeatureBuilder(),
        )
        manager = self._make_manager(tmp_path, residual_enabled=True, correction_applier=applier)

        df = pd.DataFrame({"f1": [1.0]})
        result = manager.predict_player_stats(df)

        assert result["PTS"] == pytest.approx(23.0, abs=0.01)
        assert result["REB"] == 20.0

    def test_batch_prediction_preserves_row_count(self, tmp_path):
        """Batch prediction should return one row per input."""
        manager = self._make_manager(tmp_path, residual_enabled=False)
        context_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "PLAYER_ID": [101, 102, 103]})
        result = manager.predict_player_stats_batch(context_df)
        assert len(result) == 3
