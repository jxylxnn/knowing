"""
GPU Compatibility Utilities for CUDA-enabled training.
Provides GPU detection, memory management, and tensor conversion helpers.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)

_GPU_COMPATIBLE: bool | None = None
_DEVICE = None


def check_gpu_compatibility() -> bool:
    """
    Check if the GPU is compatible with the installed PyTorch CUDA kernels.
    
    RTX 50-series (Blackwell, compute capability 12.0) requires PyTorch builds
    that include kernels for this architecture.
    
    Returns:
        bool: True if GPU can be used, False if should fallback to CPU.
    """
    global _GPU_COMPATIBLE
    
    if _GPU_COMPATIBLE is not None:
        return _GPU_COMPATIBLE
    
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed. Using CPU.")
        _GPU_COMPATIBLE = False
        return False
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available. Using CPU.")
        _GPU_COMPATIBLE = False
        return False
    
    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"Detected GPU: {gpu_name} (Compute {major}.{minor}, {vram:.1f}GB VRAM)")
        
        # RTX 50-series (Blackwell) has compute capability 12.0+
        if major >= 12:
            logger.warning(
                f"GPU '{gpu_name}' uses Blackwell architecture (compute {major}.{minor}) "
                f"which may not be fully supported by PyTorch {torch.__version__}. "
                f"If you encounter errors, try PyTorch nightly: "
                f"pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124"
            )
        
        # Test if CUDA actually works with a simple operation
        try:
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
            logger.info("CUDA kernel test passed. GPU acceleration enabled.")
            _GPU_COMPATIBLE = True
            return True
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                logger.warning(
                    f"CUDA kernel not available for GPU '{gpu_name}'. "
                    f"Install PyTorch with CUDA support for your architecture. "
                    f"Falling back to CPU."
                )
                _GPU_COMPATIBLE = False
                return False
            raise
            
    except Exception as e:
        logger.warning(f"GPU compatibility check failed: {e}. Using CPU.")
        _GPU_COMPATIBLE = False
        return False


def get_device() -> 'torch.device':
    """Get the appropriate PyTorch device based on GPU compatibility."""
    global _DEVICE
    
    if _DEVICE is not None:
        return _DEVICE
    
    try:
        import torch
    except ImportError:
        return None
    
    if check_gpu_compatibility():
        _DEVICE = torch.device('cuda')
    else:
        _DEVICE = torch.device('cpu')
    
    return _DEVICE


def is_gpu_available() -> bool:
    """Alias for check_gpu_compatibility for clearer code."""
    return check_gpu_compatibility()


def clear_gpu_memory():
    """Clear GPU cache to free memory between model training phases."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared")
    except ImportError:
        pass


def get_gpu_memory_usage() -> Tuple[float, float]:
    """
    Get current GPU memory usage in GB.
    
    Returns:
        Tuple of (allocated_gb, reserved_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    except ImportError:
        return 0.0, 0.0


def log_gpu_memory(stage: str = ""):
    """Log current GPU memory usage with optional stage label."""
    allocated, reserved = get_gpu_memory_usage()
    if allocated > 0:
        msg = f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        if stage:
            msg = f"[{stage}] {msg}"
        logger.info(msg)


def move_to_device(data: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'], 
                   device: Optional['torch.device'] = None) -> 'torch.Tensor':
    """
    Move data to GPU efficiently.
    
    Args:
        data: DataFrame, numpy array, or tensor to move
        device: Target device (defaults to get_device())
    
    Returns:
        PyTorch tensor on the target device
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for GPU operations")
    
    if device is None:
        device = get_device()
    
    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values, dtype=torch.float32, device=device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def to_numpy(tensor: 'torch.Tensor') -> np.ndarray:
    """
    Move tensor to CPU and convert to numpy array.
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        NumPy array
    """
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def get_optimal_batch_size(model_memory_gb: float = 1.0, 
                            input_size: int = 1000,
                            safety_factor: float = 0.7) -> int:
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model_memory_gb: Estimated model size in GB
        input_size: Size of single input sample (features)
        safety_factor: Fraction of GPU memory to use (default 0.7)
    
    Returns:
        Recommended batch size
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 32  # Default CPU batch size
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        available = (total_memory - model_memory_gb) * safety_factor
        
        # Estimate: each sample needs ~input_size * 4 bytes (float32) * 3 (forward/backward/grad)
        bytes_per_sample = input_size * 4 * 3
        batch_size = int((available * 1e9) / bytes_per_sample)
        
        # Clamp to reasonable range
        return max(16, min(batch_size, 8192))
    except (RuntimeError, ValueError, ImportError) as e:
        logger.debug("Could not determine optimal batch size: %s", e)
        return 32


def set_cudnn_benchmark(enabled: bool = True):
    """
    Enable/disable cuDNN auto-tuner for optimal convolution algorithms.
    
    Enable for fixed input sizes (faster training).
    Disable for variable input sizes (less memory overhead).
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = enabled
            logger.debug(f"cuDNN benchmark: {enabled}")
    except ImportError:
        pass


def set_deterministic_mode(enabled: bool = True):
    """
    Set PyTorch to deterministic mode for reproducibility.
    
    Note: This may slow down training slightly.
    """
    try:
        import torch
        torch.use_deterministic_algorithms(enabled)
        if enabled:
            torch.backends.cudnn.deterministic = True
            logger.info("Deterministic mode enabled (may be slower)")
    except ImportError:
        pass


class GPUMemoryContext:
    """Context manager to track GPU memory usage for a code block."""
    
    def __init__(self, label: str = ""):
        self.label = label
        self.start_allocated = 0
        self.start_reserved = 0
        self.end_allocated = 0
        self.end_reserved = 0
    
    def __enter__(self):
        self.start_allocated, self.start_reserved = get_gpu_memory_usage()
        return self
    
    def __exit__(self, *args):
        self.end_allocated, self.end_reserved = get_gpu_memory_usage()
        delta = self.end_allocated - self.start_allocated
        label = f"[{self.label}] " if self.label else ""
        logger.info(f"{label}GPU memory delta: {delta:+.2f}GB "
                   f"(now at {self.end_allocated:.2f}GB)")