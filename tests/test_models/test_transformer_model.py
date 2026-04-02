"""Tests for the Transformer sequence model."""

import subprocess
import sys

import joblib
import numpy as np
import pytest


def _torch_import_works() -> bool:
    """Return True when importing torch in a subprocess succeeds."""
    result = subprocess.run(
        [sys.executable, '-c', 'import torch'],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _cpu_transformer_config():
    return {
        'd_model': 16,
        'nhead': 4,
        'num_layers': 1,
        'dim_feedforward': 32,
        'dropout': 0.0,
        'batch_size': 4,
        'epochs': 1,
        'lr': 1e-3,
        'warmup_ratio': 0.0,
        'grad_checkpoint': False,
        'use_compile': False,
    }


class TestTransformerWrapper:
    """Tests for TransformerWrapper."""

    def test_transformer_compile_is_disabled_by_default(self, monkeypatch):
        if not _torch_import_works():
            pytest.skip("torch import is not safe in this environment")

        from src.models import transformer_model as tm
        import torch

        monkeypatch.setattr(tm, 'get_device', lambda: torch.device('cpu'))

        compile_calls = []

        def fake_apply_compile(model, use_compile, model_name):
            compile_calls.append((use_compile, model_name))
            return model

        monkeypatch.setattr(tm, 'apply_compile', fake_apply_compile)

        wrapper = tm.TransformerWrapper(
            input_dim=4,
            seq_len=3,
            config=_cpu_transformer_config(),
            output_dim=6,
        )

        assert wrapper.compile_enabled is False
        assert compile_calls == []
        assert wrapper.model is wrapper.eager_model

    def test_transformer_wrapper_uses_six_output_targets(self, monkeypatch):
        if not _torch_import_works():
            pytest.skip("torch import is not safe in this environment")

        from src.models import transformer_model as tm
        import torch

        monkeypatch.setattr(tm, 'get_device', lambda: torch.device('cpu'))

        wrapper = tm.TransformerWrapper(
            input_dim=4,
            seq_len=3,
            config=_cpu_transformer_config(),
            output_dim=6,
        )

        assert wrapper.output_dim == 6
        assert wrapper.model.fc.out_features == 6

        sample = np.zeros((2, 3, 4), dtype=np.float32)
        preds = wrapper.model(torch.from_numpy(sample))

        assert preds.shape == (2, 6)

    def test_transformer_validation_prediction_uses_eager_model(self, monkeypatch):
        if not _torch_import_works():
            pytest.skip("torch import is not safe in this environment")

        from src.models import transformer_model as tm
        import torch

        monkeypatch.setattr(tm, 'get_device', lambda: torch.device('cpu'))

        wrapper = tm.TransformerWrapper(
            input_dim=4,
            seq_len=3,
            config=_cpu_transformer_config(),
            output_dim=6,
        )

        class EagerModule(torch.nn.Module):
            def forward(self, x):
                return torch.ones((x.shape[0], 6), dtype=torch.float32)

        class CompiledModule(torch.nn.Module):
            def forward(self, x):
                raise AssertionError("compiled model should not be used for validation prediction")

        wrapper.eager_model = EagerModule()
        wrapper.model = CompiledModule()
        wrapper.validation_model = wrapper.eager_model
        wrapper.is_trained = True
        wrapper.feat_mean = np.zeros(4, dtype=np.float32)
        wrapper.feat_std = np.ones(4, dtype=np.float32)

        sample = np.zeros((2, 3, 4), dtype=np.float32)
        preds = wrapper.predict_batch(sample)

        assert preds.shape == (2, 6)
        assert np.allclose(preds, 1.0)

    def test_transformer_wrapper_loads_legacy_three_output_checkpoint(self, tmp_path, monkeypatch):
        if not _torch_import_works():
            pytest.skip("torch import is not safe in this environment")

        from src.models import transformer_model as tm
        import torch

        monkeypatch.setattr(tm, 'get_device', lambda: torch.device('cpu'))

        legacy_wrapper = tm.TransformerWrapper(
            input_dim=4,
            seq_len=3,
            config=_cpu_transformer_config(),
            output_dim=3,
        )

        legacy_state = {
            'model_state': legacy_wrapper.model.state_dict(),
            'feat_mean': np.zeros(4, dtype=np.float32),
            'feat_std': np.ones(4, dtype=np.float32),
            'input_dim': 4,
            'seq_len': 3,
            'config': {},
        }

        path = tmp_path / 'legacy_transformer.pkl'
        joblib.dump(legacy_state, path)

        loaded = tm.TransformerWrapper.load(str(path))

        assert loaded.output_dim == 3
        assert loaded.model.fc.out_features == 3
