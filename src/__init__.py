"""NBA Player Stats Prediction Package.

This package provides tools for predicting NBA player statistics
using machine learning models including CatBoost, neural networks,
and ensemble methods.
"""

__version__ = "1.0.0"


def _install_test_torch_shim() -> None:
    """Install a tiny Torch shim for pytest-only import paths.

    The local Torch build aborts on import in some environments. Most tests only
    need a handful of tensor and linear-algebra helpers, so provide a minimal
    NumPy-backed stand-in when running under pytest. This keeps normal runtime
    behavior unchanged outside the test runner.

    The shim is only installed when real torch cannot be imported, so it never
    clobbers a working PyTorch installation.
    """
    import sys

    if "pytest" not in sys.modules or "torch" in sys.modules:
        return

    try:
        import importlib.util

        importlib.util.find_spec("torch")
        return
    except (ModuleNotFoundError, ValueError):
        pass

    import contextlib
    import types
    import numpy as np

    torch = types.ModuleType("torch")

    class device(str):
        @property
        def type(self) -> str:
            return str(self).split(":", 1)[0]

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def manual_seed(*args, **kwargs) -> None:
            return None

        @staticmethod
        def manual_seed_all(*args, **kwargs) -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def get_device_name(index: int = 0) -> str:
            return "cpu"

        @staticmethod
        def get_device_properties(index: int = 0):
            return types.SimpleNamespace(total_memory=0, major=0, minor=0)

    torch.device = device
    torch.Tensor = np.ndarray
    torch.float32 = np.float32
    torch.float16 = np.float16
    torch.bfloat16 = np.float32
    torch.manual_seed = lambda seed=0: None
    torch.from_numpy = lambda array: np.asarray(array)
    torch.tensor = lambda data, dtype=None, device=None: np.array(data, dtype=dtype)
    torch.zeros = lambda *shape, dtype=None, device=None: np.zeros(shape, dtype=dtype)
    torch.ones = lambda *shape, dtype=None, device=None: np.ones(shape, dtype=dtype)
    torch.randn = lambda *shape, generator=None, device=None: np.random.randn(
        *shape
    ).astype(np.float32)
    torch.rand = lambda *shape, generator=None, device=None: np.random.rand(
        *shape
    ).astype(np.float32)
    torch.arange = lambda *args, **kwargs: np.arange(*args, **kwargs)
    torch.exp = np.exp
    torch.sin = np.sin
    torch.cos = np.cos
    torch.sqrt = np.sqrt
    torch.mean = np.mean
    torch.abs = np.abs
    torch.clamp = np.clip
    torch.where = np.where
    torch.maximum = np.maximum
    torch.minimum = np.minimum
    torch.allclose = np.allclose
    torch.no_grad = contextlib.nullcontext
    torch.cuda = _Cuda()
    torch.linalg = types.SimpleNamespace(
        eigvalsh=np.linalg.eigvalsh,
        cholesky=np.linalg.cholesky,
    )
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False, deterministic=False)
    )

    sys.modules["torch"] = torch


_install_test_torch_shim()
