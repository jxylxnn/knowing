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


def test_m_tier_seq_len_is_20():
    """Verify M tier transformer config has seq_len=20 and max_seq_length=20."""
    from src.config.model_config import SIZE_TIER_SPECS

    m_transformer = SIZE_TIER_SPECS['M']['transformer']
    assert m_transformer['seq_len'] == 20, (
        f"Expected M tier seq_len=20, got {m_transformer['seq_len']}"
    )
    assert m_transformer['max_seq_length'] == 20, (
        f"Expected M tier max_seq_length=20, got {m_transformer['max_seq_length']}"
    )


def test_sequence_creation_with_zero_padding(monkeypatch):
    """Players with fewer than seq_len games produce zero-padded sequences."""
    if not _torch_import_works():
        pytest.skip("torch import is not safe in this environment")

    from src.models import transformer_model as tm
    import torch
    import pandas as pd

    monkeypatch.setattr(tm, 'get_device', lambda: torch.device('cpu'))

    wrapper = tm.TransformerWrapper(
        input_dim=3,
        seq_len=5,
        config=_cpu_transformer_config(),
        output_dim=2,
    )

    # Create a player with only 3 games (fewer than seq_len=5)
    df = pd.DataFrame({
        'PLAYER_ID': ['P1'] * 3,
        'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05']),
        'f1': [1.0, 2.0, 3.0],
        'f2': [4.0, 5.0, 6.0],
        'f3': [7.0, 8.0, 9.0],
        't1': [10.0, 20.0, 30.0],
        't2': [11.0, 21.0, 31.0],
    })

    feature_cols = ['f1', 'f2', 'f3']
    target_cols = ['t1', 't2']

    sequences, targets = wrapper._create_sequences(df, feature_cols, target_cols)

    # Should produce 3 sequences (one per game), not 0
    assert len(sequences) == 3, f"Expected 3 sequences, got {len(sequences)}"
    assert len(targets) == 3, f"Expected 3 targets, got {len(targets)}"

    # Each sequence should have shape (seq_len, n_features) = (5, 3)
    assert sequences.shape == (3, 5, 3), f"Expected shape (3, 5, 3), got {sequences.shape}"

    # First game (idx=0): full zero-padding, target is first game's targets
    assert np.all(sequences[0] == 0.0), "First sequence should be all zeros (no context)"
    np.testing.assert_array_almost_equal(targets[0], [10.0, 11.0])

    # Second game (idx=1): 4 zeros + 1 context row
    assert np.all(sequences[1, :4] == 0.0), "First 4 positions should be zero-padded"
    np.testing.assert_array_almost_equal(sequences[1, 4], [1.0, 4.0, 7.0])
    np.testing.assert_array_almost_equal(targets[1], [20.0, 21.0])

    # Third game (idx=2): 3 zeros + 2 context rows
    assert np.all(sequences[2, :3] == 0.0), "First 3 positions should be zero-padded"
    np.testing.assert_array_almost_equal(sequences[2, 3], [1.0, 4.0, 7.0])
    np.testing.assert_array_almost_equal(sequences[2, 4], [2.0, 5.0, 8.0])
    np.testing.assert_array_almost_equal(targets[2], [30.0, 31.0])


def test_sequence_batch_with_zero_padding():
    """_build_sequence_batch produces zero-padded sequences for short players."""
    from src.training.pipeline import TrainingPipeline
    import pandas as pd

    # Create a player with only 3 games (fewer than seq_len=5)
    df = pd.DataFrame({
        'PLAYER_ID': ['P1'] * 3,
        'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05']),
        'f1': [1.0, 2.0, 3.0],
        'f2': [4.0, 5.0, 6.0],
        'f3': [7.0, 8.0, 9.0],
        't1': [10.0, 20.0, 30.0],
        't2': [11.0, 21.0, 31.0],
    })

    feature_cols = ['f1', 'f2', 'f3']
    target_cols = ['t1', 't2']
    seq_len = 5

    # _build_sequence_batch does not reference self, so a minimal mock suffices
    class _MockPipeline:
        pass

    mock = _MockPipeline()
    sequences, targets, indices = TrainingPipeline._build_sequence_batch(
        mock, df, feature_cols, target_cols, seq_len
    )

    # Should produce 3 sequences (one per game), not empty
    assert len(sequences) == 3, f"Expected 3 sequences, got {len(sequences)}"
    assert len(targets) == 3, f"Expected 3 targets, got {len(targets)}"
    assert len(indices) == 3, f"Expected 3 indices, got {len(indices)}"

    # Each sequence should have shape (seq_len, n_features) = (5, 3)
    assert sequences.shape == (3, 5, 3), f"Expected shape (3, 5, 3), got {sequences.shape}"

    # First game (idx=0): full zero-padding
    assert np.all(sequences[0] == 0.0), "First sequence should be all zeros (no context)"
    np.testing.assert_array_almost_equal(targets[0], [10.0, 11.0])

    # Second game (idx=1): 4 zeros + 1 context row
    assert np.all(sequences[1, :4] == 0.0), "First 4 positions should be zero-padded"
    np.testing.assert_array_almost_equal(sequences[1, 4], [1.0, 4.0, 7.0])
    np.testing.assert_array_almost_equal(targets[1], [20.0, 21.0])

    # Third game (idx=2): 3 zeros + 2 context rows
    assert np.all(sequences[2, :3] == 0.0), "First 3 positions should be zero-padded"
    np.testing.assert_array_almost_equal(sequences[2, 3], [1.0, 4.0, 7.0])
    np.testing.assert_array_almost_equal(sequences[2, 4], [2.0, 5.0, 8.0])
    np.testing.assert_array_almost_equal(targets[2], [30.0, 31.0])


def test_sequence_creation_with_enough_games_unchanged(monkeypatch):
    """Players with more than seq_len + 1 games still produce standard sliding windows."""
    if not _torch_import_works():
        pytest.skip("torch import is not safe in this environment")

    from src.models import transformer_model as tm
    import torch
    import pandas as pd

    monkeypatch.setattr(tm, 'get_device', lambda: torch.device('cpu'))

    seq_len = 3
    wrapper = tm.TransformerWrapper(
        input_dim=2,
        seq_len=seq_len,
        config=_cpu_transformer_config(),
        output_dim=2,
    )

    # Create a player with 7 games (more than seq_len + 1 = 4)
    dates = pd.date_range('2024-01-01', periods=7, freq='D')
    df = pd.DataFrame({
        'PLAYER_ID': ['P1'] * 7,
        'GAME_DATE': dates,
        'f1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'f2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        't1': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
        't2': [110.0, 210.0, 310.0, 410.0, 510.0, 610.0, 710.0],
    })

    feature_cols = ['f1', 'f2']
    target_cols = ['t1', 't2']

    sequences, targets = wrapper._create_sequences(df, feature_cols, target_cols)

    # With 7 games and seq_len=3, we now get 7 samples (idx 0-6).
    # Indices >= seq_len (3,4,5,6) should produce standard sliding windows
    # identical to the old behavior.
    assert len(sequences) == 7, f"Expected 7 sequences, got {len(sequences)}"

    # idx=3: context is games [0,1,2] -> features [[1,10],[2,20],[3,30]]
    # target is game at idx=3 -> [400, 410]
    np.testing.assert_array_almost_equal(
        sequences[3], [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    )
    np.testing.assert_array_almost_equal(targets[3], [400.0, 410.0])

    # idx=4: context is games [1,2,3] -> features [[2,20],[3,30],[4,40]]
    # target is game at idx=4 -> [500, 510]
    np.testing.assert_array_almost_equal(
        sequences[4], [[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    )
    np.testing.assert_array_almost_equal(targets[4], [500.0, 510.0])

    # idx=6: context is games [3,4,5] -> features [[4,40],[5,50],[6,60]]
    # target is game at idx=6 -> [700, 710]
    np.testing.assert_array_almost_equal(
        sequences[6], [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]
    )
    np.testing.assert_array_almost_equal(targets[6], [700.0, 710.0])
