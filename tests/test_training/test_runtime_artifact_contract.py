import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


class FakeCatBoostModel:
    """Minimal CatBoost-compatible test double for save/load contract tests."""

    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self.feature_names_ = []

    def save_model(self, path: str) -> None:
        payload = {
            "value": self.value,
            "feature_names_": list(self.feature_names_),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def load_model(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.value = float(payload["value"])
        self.feature_names_ = list(payload.get("feature_names_", []))

    def predict(self, X):
        size = len(X) if hasattr(X, "__len__") else 1
        return np.full(size, self.value, dtype=np.float32)


def _build_contract_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 1200
    player_ids = np.tile(np.arange(1000, 1030), rows // 30)
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")

    frame = pd.DataFrame(
        {
            "PLAYER_ID": player_ids,
            "TEAM_ID": np.tile([1610612747, 1610612738, 1610612744], rows // 3),
            "OPPONENT_ID": np.tile([1610612741, 1610612761, 1610612752], rows // 3),
            "GAME_DATE": dates,
            "PTS": np.linspace(10, 30, rows),
            "REB": np.linspace(4, 14, rows),
            "AST": np.linspace(3, 11, rows),
            "STL": np.linspace(0.5, 2.5, rows),
            "BLK": np.linspace(0.2, 1.8, rows),
            "TOV": np.linspace(1.0, 4.5, rows),
            "ROLL_PTS_AVG_10": np.linspace(9, 29, rows),
            "ROLL_REB_AVG_10": np.linspace(3, 13, rows),
            "ROLL_AST_AVG_10": np.linspace(2, 10, rows),
            "ROLL_STL_AVG_10": np.linspace(0.4, 2.0, rows),
            "ROLL_BLK_AVG_10": np.linspace(0.1, 1.5, rows),
            "ROLL_TOV_AVG_10": np.linspace(0.8, 4.0, rows),
            "PACE_FACTOR": np.linspace(96, 104, rows),
            "TEAM_PACE_10": np.linspace(95, 103, rows),
            "REST_DAYS": np.tile([0, 1, 2, 3], rows // 4),
            "IS_HOME": np.tile([0, 1], rows // 2),
        }
    )

    return frame.iloc[:1050].copy(), frame.iloc[1050:].copy()


def _make_fake_catboost_results(pipeline, *, break_target: str | None = None):
    from src.training.catboost_trainer import CatBoostTrainer
    from src.training.trainer import TrainResult

    results = {}
    for idx, target in enumerate(pipeline.TARGETS, start=1):
        trainer = CatBoostTrainer(
            model_name=f"catboost_{target}",
            target=target,
            config={"iterations": 5},
            use_multi_loss=True,
            use_quantile=True,
        )
        trainer.feature_cols = list(pipeline.feature_cols or [])
        trainer.cat_features = list(pipeline.cat_features)

        primary = FakeCatBoostModel(idx)
        primary.feature_names_ = list(trainer.feature_cols)
        trainer.primary_model = primary

        mae = FakeCatBoostModel(idx + 0.1)
        mae.feature_names_ = list(trainer.feature_cols)
        trainer.mae_model = mae

        qlow = FakeCatBoostModel(idx - 0.2)
        qlow.feature_names_ = list(trainer.feature_cols)
        trainer.quantile_low_model = qlow

        qhigh = FakeCatBoostModel(idx + 0.2)
        qhigh.feature_names_ = list(trainer.feature_cols)
        trainer.quantile_high_model = qhigh
        trainer.is_trained = True

        if target == break_target:
            trainer.save = lambda path: None

        pipeline.trainers[f"catboost_{target}"] = trainer
        pipeline.models[target] = trainer.primary_model
        results[target] = TrainResult(
            model=trainer,
            metrics={"mae": float(idx)},
            training_time=0.0,
        )

    return results


def _build_pipeline(tmp_path, monkeypatch):
    from src.training import catboost_trainer as catboost_module
    from src.training.pipeline import TrainingPipeline

    monkeypatch.setattr(catboost_module, "CatBoostRegressor", FakeCatBoostModel)
    monkeypatch.setattr(catboost_module, "CATBOOST_AVAILABLE", True)

    pipeline = TrainingPipeline(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        parallel=False,
        use_gpu=False,
    )
    pipeline.model_config["transformer"]["enabled"] = False
    return pipeline


def test_training_pipeline_persists_runtime_artifact_contract(tmp_path, monkeypatch):
    from src.models.model_manager import ModelManager

    pipeline = _build_pipeline(tmp_path, monkeypatch)
    pipeline.training_preset = "small"
    pipeline.feature_group_selection = [
        "rolling",
        "efficiency",
        "momentum",
        "pace",
        "opponent_strength",
        "archetype",
    ]
    fit_df, val_df = _build_contract_frames()

    monkeypatch.setattr(
        pipeline,
        "_train_catboost_parallel",
        lambda fit, val: _make_fake_catboost_results(pipeline),
    )

    pipeline.train(fit_df, val_df)

    for target in pipeline.TARGETS:
        stem = target.lower()
        assert (pipeline.models_dir / f"{stem}_catboost.cbm").exists()
        assert (pipeline.models_dir / f"{stem}_catboost_mae.cbm").exists()
        assert (pipeline.models_dir / f"{stem}_catboost_qlow.cbm").exists()
        assert (pipeline.models_dir / f"{stem}_catboost_qhigh.cbm").exists()
        assert (pipeline.models_dir / f"{stem}_metadata.joblib").exists()

    assert (pipeline.models_dir / "feature_schema.pkl").exists()
    assert (pipeline.models_dir / "feature_cols.pkl").exists()
    assert (pipeline.models_dir / "blend_weights.pkl").exists()
    assert (pipeline.models_dir / "model_stack_metadata.pkl").exists()
    assert not (pipeline.models_dir / "attention_transformer.pkl").exists()

    metadata = joblib.load(pipeline.models_dir / "model_stack_metadata.pkl")
    assert metadata["transformer_enabled"] is False
    assert metadata["model_count"] == 1
    assert metadata["training_preset"] == "small"
    assert metadata["feature_groups"] == [
        "rolling",
        "efficiency",
        "momentum",
        "pace",
        "opponent_strength",
        "archetype",
    ]

    manager = ModelManager(
        data_dir=str(tmp_path / "data"),
        models_dir=str(pipeline.models_dir),
    )
    counts = manager.load_models()

    assert counts["catboost"] == len(pipeline.TARGETS)
    assert set(manager.models) == set(manager.targets)
    assert manager.feature_cols == pipeline.feature_cols
    assert set(manager.blend_weights) == set(pipeline.TARGETS)


def test_training_pipeline_fails_loudly_when_runtime_artifacts_are_missing(
    tmp_path, monkeypatch
):
    pipeline = _build_pipeline(tmp_path, monkeypatch)
    fit_df, val_df = _build_contract_frames()

    monkeypatch.setattr(
        pipeline,
        "_train_catboost_parallel",
        lambda fit, val: _make_fake_catboost_results(pipeline, break_target="PTS"),
    )

    with pytest.raises(RuntimeError, match="required runtime artifacts"):
        pipeline.train(fit_df, val_df)


def test_transformer_validation_batch_prediction_delegates_to_wrapper(
    tmp_path, monkeypatch
):
    from src.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        parallel=False,
        use_gpu=False,
    )

    class DummyTransformer:
        def __init__(self):
            self.calls = 0

        def predict_batch(self, sequences):
            self.calls += 1
            return np.full(
                (len(sequences), len(pipeline.TARGETS)), 2.0, dtype=np.float32
            )

    transformer = DummyTransformer()
    sequences = np.zeros((3, 4, 5), dtype=np.float32)

    preds = pipeline._predict_transformer_batch(transformer, sequences)

    assert transformer.calls == 1
    assert preds.shape == (3, len(pipeline.TARGETS))
    assert np.allclose(preds, 2.0)


def test_feature_schema_import_contract_survives_clean_process(tmp_path):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "torch.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "\n"
        "    @staticmethod\n"
        "    def get_device_capability(index=0):\n"
        "        return (0, 0)\n"
        "\n"
        "    @staticmethod\n"
        "    def get_device_name(index=0):\n"
        "        return 'cpu'\n"
        "\n"
        "    @staticmethod\n"
        "    def get_device_properties(index=0):\n"
        "        class _Props:\n"
        "            total_memory = 0\n"
        "        return _Props()\n"
        "\n"
        "    @staticmethod\n"
        "    def empty_cache():\n"
        "        return None\n"
        "\n"
        "cuda = _Cuda()\n"
        "__version__ = 'stub'\n"
        "compile = lambda *args, **kwargs: None\n"
        "device = lambda value: value\n"
        "Tensor = object\n"
        "backends = type('B', (), {'cuda': type('C', (), {'enable_flash_sdp': staticmethod(lambda *args, **kwargs: None), 'enable_mem_efficient_sdp': staticmethod(lambda *args, **kwargs: None), 'enable_math_sdp': staticmethod(lambda *args, **kwargs: None)})()})()\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(stub_dir) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "from src.utils.prediction_utils import FeatureSchema\n"
        "from src.utils import FeatureSchema as PackageFeatureSchema\n"
        "from src.training.pipeline import TrainingPipeline\n"
        "print(FeatureSchema.__name__)\n"
        "print(PackageFeatureSchema.__name__)\n"
        "print(TrainingPipeline.__name__)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "FeatureSchema" in result.stdout
    assert "TrainingPipeline" in result.stdout


def test_model_manager_rejects_missing_transformer_when_blend_weights_expect_it(
    tmp_path, monkeypatch
):
    from src.models.model_manager import ModelManager
    from src.training import catboost_trainer as catboost_module

    monkeypatch.setattr(catboost_module, "CatBoostRegressor", FakeCatBoostModel)

    pipeline = _build_pipeline(tmp_path, monkeypatch)
    fit_df, val_df = _build_contract_frames()

    monkeypatch.setattr(
        pipeline,
        "_train_catboost_parallel",
        lambda fit, val: _make_fake_catboost_results(pipeline),
    )

    pipeline.train(fit_df, val_df)

    import joblib

    blend_weights = joblib.load(pipeline.models_dir / "blend_weights.pkl")
    blend_weights["PTS"]["transformer"] = 0.3
    blend_weights["PTS"]["catboost"] = 0.7
    joblib.dump(blend_weights, pipeline.models_dir / "blend_weights.pkl")

    # Remove versioned WeightStore so the legacy blend_weights.pkl edit is honored.
    import shutil
    shutil.rmtree(pipeline.models_dir / "blend_weights", ignore_errors=True)

    manager = ModelManager(
        data_dir=str(tmp_path / "data"),
        models_dir=str(pipeline.models_dir),
    )

    with pytest.raises(FileNotFoundError, match="attention_transformer.pkl"):
        manager.load_models()
