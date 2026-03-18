import contextlib
import importlib.util
import pathlib
import sys
import types

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _install_fake_torch(monkeypatch):
    """Install a minimal fake Torch package for trainer import tests."""
    torch = types.ModuleType("torch")
    torch.__path__ = []

    class Device(str):
        @property
        def type(self):
            return str(self).split(":", 1)[0]

    torch.device = Device
    torch.Tensor = np.ndarray
    torch.float32 = np.float32
    torch.float16 = np.float16
    torch.bfloat16 = np.float32
    torch.manual_seed = lambda seed=0: None
    torch.from_numpy = lambda array: np.asarray(array)
    torch.no_grad = contextlib.nullcontext
    torch.exp = np.exp
    torch.mean = np.mean
    torch.clamp = np.clip
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False, deterministic=False)
    )
    torch.amp = types.SimpleNamespace(
        autocast=lambda *args, **kwargs: contextlib.nullcontext(),
        GradScaler=lambda *args, **kwargs: None,
    )

    nn_mod = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    class MSELoss:
        def __call__(self, input, target):
            return float(np.mean((np.asarray(input) - np.asarray(target)) ** 2))

    nn_mod.Module = Module
    nn_mod.MSELoss = MSELoss
    torch.nn = nn_mod

    optim_mod = types.ModuleType("torch.optim")

    class Optimizer:
        pass

    optim_mod.Optimizer = Optimizer
    torch.optim = optim_mod

    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class TensorDataset:
        def __init__(self, *args, **kwargs):
            self.args = args

    class DataLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    data_mod.TensorDataset = TensorDataset
    data_mod.DataLoader = DataLoader
    utils_mod.data = data_mod
    torch.utils = utils_mod

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.optim", optim_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)


def _load_module(module_name: str, file_path: pathlib.Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_nn_trainer_handles_tuple_outputs(monkeypatch):
    _install_fake_torch(monkeypatch)

    # Provide lightweight package placeholders so we can load the modules
    # directly without executing src.training.__init__.
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(ROOT / "src")]
    training_pkg = types.ModuleType("src.training")
    training_pkg.__path__ = [str(ROOT / "src" / "training")]
    models_pkg = types.ModuleType("src.models")
    models_pkg.__path__ = [str(ROOT / "src" / "models")]
    monkeypatch.setitem(sys.modules, "src", src_pkg)
    monkeypatch.setitem(sys.modules, "src.training", training_pkg)
    monkeypatch.setitem(sys.modules, "src.models", models_pkg)

    _load_module("src.models.gpu_utils", ROOT / "src" / "models" / "gpu_utils.py", monkeypatch)
    _load_module("src.training.trainer", ROOT / "src" / "training" / "trainer.py", monkeypatch)
    nn_trainer = _load_module(
        "src.training.nn_trainer",
        ROOT / "src" / "training" / "nn_trainer.py",
        monkeypatch,
    )

    trainer = nn_trainer.NeuralNetworkTrainer.__new__(nn_trainer.NeuralNetworkTrainer)

    means = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    logvars = np.zeros_like(means)
    target = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    assert trainer._extract_primary_output((means, logvars)) is means
    assert np.isclose(trainer._compute_loss((means, logvars), target), 0.5)
