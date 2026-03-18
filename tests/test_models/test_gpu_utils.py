import sys
from types import ModuleType, SimpleNamespace

from src.models import gpu_utils


def test_optimize_for_training_uses_stable_tf32_helper(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.backends = SimpleNamespace(cudnn=SimpleNamespace())
    fake_torch.use_deterministic_algorithms = lambda enabled: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gpu_utils, "_ENABLE_TF32_HELPER", lambda enabled=True: enabled)

    settings = gpu_utils.optimize_for_training(
        enable_tf32=True,
        enable_cudnn_benchmark=False,
        enable_cudnn_deterministic=False,
        set_sync_point=False,
    )

    assert settings["tf32_enabled"] is True
    assert settings["cudnn_benchmark"] is False
