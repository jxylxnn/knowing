"""
GPU Compatibility Utilities for CUDA-enabled training.
Provides GPU detection, memory management, tensor conversion helpers,
and performance optimization utilities.

This module centralizes all GPU-related functionality for the training pipeline,
including:
- Hardware detection and capability assessment
- Tensor operations and device management
- Mixed precision training utilities (FP16/BF16)
- Performance optimization (TF32, cuDNN, CUDA graphs)
- Memory management and profiling
"""

import logging
import math
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Module-level state for caching
_GPU_COMPATIBLE: bool | None = None
_DEVICE = None
_GPU_CAPABILITIES: dict = {}
_TF32_ENABLED: bool = False
_BF16_SUPPORTED: bool = False


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
    except Exception:
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


class WarmupCosineScheduler:
    """
    Shared learning rate scheduler with warmup + cosine decay.
    
    Used by all PyTorch models for consistent training behavior.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def apply_compile(model, use_compile: bool, model_name: str):
    """
    Apply torch.compile() for PyTorch 2.0+ speedup.
    
    Args:
        model: PyTorch model
        use_compile: Whether to compile
        model_name: Name for logging
    
    Returns:
        Compiled model or original model
    """
    import math
    
    if not use_compile:
        return model
    
    try:
        import torch
        if not hasattr(torch, 'compile'):
            logger.debug("torch.compile() not available (PyTorch < 2.0)")
            return model
        
        logger.info(f"Compiling {model_name} with torch.compile()...")
        compiled_model = torch.compile(model, mode='reduce-overhead')
        logger.info(f"{model_name} compiled successfully")
        return compiled_model
    except Exception as e:
        logger.warning(f"torch.compile() failed for {model_name}: {e}")
        return model


def get_compile_status() -> bool:
    """
    Check if torch.compile is available and should be used.
    
    Returns:
        True if torch.compile is available and GPU is detected
    """
    try:
        import torch
        if not hasattr(torch, 'compile'):
            return False
        return check_gpu_compatibility()
    except ImportError:
        return False


# =============================================================================
# PERFORMANCE OPTIMIZATION FUNCTIONS
# =============================================================================

def get_gpu_capabilities() -> dict:
    """
    Get detailed GPU capabilities for optimization decisions.
    
    Returns:
        Dictionary with GPU capability information including:
        - tf32_supported: Whether TensorFloat-32 is supported
        - bf16_supported: Whether BFloat16 is supported
        - compute_capability: (major, minor) tuple
        - vram_gb: Total VRAM in GB
        - gpu_name: GPU device name
        - multi_processor_count: Number of streaming multiprocessors
    """
    global _GPU_CAPABILITIES
    
    if _GPU_CAPABILITIES:
        return _GPU_CAPABILITIES
    
    capabilities = {
        'tf32_supported': False,
        'bf16_supported': False,
        'compute_capability': (0, 0),
        'vram_gb': 0.0,
        'gpu_name': 'Unknown',
        'multi_processor_count': 0,
        'is_available': False,
    }
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            return capabilities
        
        props = torch.cuda.get_device_properties(0)
        major, minor = props.major, props.minor
        
        # TF32 supported on Ampere+ (compute 8.0+)
        tf32_supported = major >= 8
        
        # BF16 supported on Ampere+ (compute 8.0+)
        # Note: Some older GPUs may have compute 8.0 but limited BF16
        bf16_supported = major >= 8
        
        capabilities = {
            'tf32_supported': tf32_supported,
            'bf16_supported': bf16_supported,
            'compute_capability': (major, minor),
            'vram_gb': props.total_memory / 1e9,
            'gpu_name': torch.cuda.get_device_name(0),
            'multi_processor_count': props.multi_processor_count,
            'is_available': True,
        }
        
        _GPU_CAPABILITIES = capabilities
        
    except ImportError:
        pass
    
    return capabilities


def enable_tf32(enabled: bool = True) -> bool:
    """
    Enable or disable TensorFloat-32 for faster matmul operations.
    
    TF32 provides 2-8x speedup on Ampere+ GPUs (RTX 30xx, A100, etc.)
    with minimal accuracy loss (about 1 part in 10^5).
    
    Args:
        enabled: Whether to enable TF32 (default True)
    
    Returns:
        True if TF32 was successfully set, False otherwise
    
    Safety:
        TF32 has ~10 bits of mantissa vs 23 in FP32. This is sufficient
        for most deep learning workloads but may not be appropriate
        for numerical precision-critical applications.
    """
    global _TF32_ENABLED
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.debug("TF32 not available: CUDA not available")
            return False
        
        caps = get_gpu_capabilities()
        if not caps['tf32_supported']:
            logger.debug(f"TF32 not supported on GPU with compute {caps['compute_capability']}")
            return False
        
        # Set TF32 matmul precision
        if enabled:
            # 'high' = TF32 enabled (recommended)
            # 'medium' = TF32 enabled for matmul only
            # 'highest' = FP32 only (slowest but most precise)
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _TF32_ENABLED = True
            logger.info("TF32 enabled for matmul and cuDNN (Ampere+ optimization)")
        else:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            _TF32_ENABLED = False
            logger.info("TF32 disabled (using full FP32 precision)")
        
        return True
        
    except ImportError:
        return False


def is_bf16_supported() -> bool:
    """
    Check if BFloat16 is supported on this GPU.
    
    BF16 is supported on:
    - Ampere+ (RTX 30xx, A100, etc.) via native hardware
    - Turing (RTX 20xx, T4) via emulation
    
    Returns:
        True if BF16 is available
    """
    global _BF16_SUPPORTED
    
    if _BF16_SUPPORTED:
        return True
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False
        
        # Check via PyTorch's helper
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_bf16_supported'):
            _BF16_SUPPORTED = torch.cuda.is_bf16_supported()
            return _BF16_SUPPORTED
        
        # Fallback: Check compute capability (Ampere+ = 8.0+)
        caps = get_gpu_capabilities()
        _BF16_SUPPORTED = caps['compute_capability'][0] >= 8
        return _BF16_SUPPORTED
        
    except ImportError:
        return False


def get_autocast_dtype() -> 'torch.dtype':
    """
    Get the optimal autocast dtype for mixed precision training.
    
    Returns:
        torch.bfloat16 if supported, torch.float16 otherwise
    """
    import torch
    
    if is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@contextmanager
def autocast_context(device: Optional[str] = None, enabled: bool = True):
    """
    Context manager for automatic mixed precision with optimal dtype.
    
    Automatically selects BF16 on Ampere+ GPUs, FP16 on older GPUs.
    
    Args:
        device: Device to use ('cuda' or 'cpu'), auto-detected if None
        enabled: Whether to enable autocast
    
    Yields:
        Nothing (context manager)
    
    Example:
        with autocast_context():
            output = model(input)
            loss = criterion(output, target)
    """
    import torch
    
    if not enabled:
        yield
        return
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dtype = get_autocast_dtype()
    
    # Use the string-based autocast for newer PyTorch
    device_type = 'cuda' if device == 'cuda' or str(device).startswith('cuda') else 'cpu'
    
    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
        yield


def check_flash_attention_available() -> bool:
    """
    Check if Flash Attention is available.
    
    Flash Attention 2 provides significant speedups for attention-based models.
    It requires:
    - PyTorch 2.0+
    - CUDA 11.6+
    - Ampere+ GPU (compute 8.0+)
    - flash-attn package or PyTorch 2.1+ built-in
    
    Returns:
        True if Flash Attention is available
    """
    try:
        import torch
        
        # Check GPU compatibility
        if not torch.cuda.is_available():
            return False
        
        caps = get_gpu_capabilities()
        if caps['compute_capability'][0] < 8:
            # Flash Attention requires Ampere+
            return False
        
        # Check if flash_attn package is installed
        try:
            import flash_attn
            return True
        except ImportError:
            pass
        
        # Check if PyTorch has built-in scaled_dot_product_attention (Flash)
        # This is available in PyTorch 2.0+
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ has SDPA which uses Flash/Memory-Efficient attention
            return True
        
        return False
        
    except ImportError:
        return False


def enable_flash_attention_optimizations(model) -> None:
    """
    Enable Flash Attention optimizations for a model in-place.
    
    This replaces standard attention implementations with Flash Attention
    when available. Currently supports Transformer models with standard
    MultiheadAttention.
    
    Args:
        model: PyTorch model to optimize
    
    Note:
        This is a no-op if Flash Attention is not available.
        The model must be on CUDA for Flash Attention to work.
    """
    if not check_flash_attention_available():
        return
    
    import torch
    import torch.nn as nn
    
    # Check if model is on CUDA
    try:
        next(model.parameters()).device
        if not next(model.parameters()).device.type == 'cuda':
            return
    except StopIteration:
        return
    
    # Replace attention implementations where possible
    # This is model-specific and requires careful handling
    # For now, we just set a flag for models to use at forward time
    if hasattr(model, '_use_flash_attention'):
        model._use_flash_attention = True
        logger.info("Flash Attention enabled for model")


def create_grad_scaler(device_type: str = 'cuda', enabled: bool = True):
    """
    Create an appropriate gradient scaler for mixed precision training.
    
    For BF16, gradient scaling is typically not needed as the dynamic range
    is sufficient. For FP16, GradScaler is essential.
    
    Args:
        device_type: 'cuda' or 'cpu'
        enabled: Whether to enable gradient scaling
    
    Returns:
        torch.amp.GradScaler instance (or None for CPU/BF16)
    """
    import torch
    
    if device_type == 'cpu':
        return None
    
    if not enabled:
        return torch.amp.GradScaler(device_type, enabled=False)
    
    # BF16 doesn't need gradient scaling
    if is_bf16_supported():
        # Return a disabled scaler for consistency
        # Some training loops expect a scaler object
        return torch.amp.GradScaler(device_type, enabled=False)
    
    # FP16 needs gradient scaling
    return torch.amp.GradScaler(device_type, enabled=True)


def optimize_for_training(
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
    enable_cudnn_deterministic: bool = False,
    set_sync_point: bool = False,
) -> dict:
    """
    Apply optimal PyTorch settings for training.
    
    This should be called once at the start of training to configure
    PyTorch for maximum performance.
    
    Args:
        enable_tf32: Enable TF32 for faster matmul on Ampere+ GPUs
        enable_cudnn_benchmark: Enable cuDNN auto-tuner for conv operations
        enable_cudnn_deterministic: Enable deterministic cuDNN (slower but reproducible)
        set_sync_point: Enable CUDA launch blocking for debugging
    
    Returns:
        Dictionary of applied settings
    """
    import torch
    
    settings = {
        'tf32_enabled': False,
        'cudnn_benchmark': False,
        'cudnn_deterministic': False,
        'cuda_launch_blocking': False,
    }
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping GPU optimizations")
        return settings
    
    # TF32 for Ampere+ GPUs
    if enable_tf32:
        settings['tf32_enabled'] = enable_tf32(enabled=True)
    
    # cuDNN benchmark for faster conv operations
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        settings['cudnn_benchmark'] = True
        logger.debug("cuDNN benchmark enabled")
    
    # Deterministic mode for reproducibility (slower)
    if enable_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        settings['cudnn_deterministic'] = True
        logger.info("Deterministic mode enabled (training may be slower)")
    
    # CUDA launch blocking for debugging (very slow, only use for debugging)
    if set_sync_point:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        settings['cuda_launch_blocking'] = True
        logger.warning("CUDA launch blocking enabled (for debugging only)")
    
    return settings


def get_optimal_dataloader_workers() -> int:
    """
    Get optimal number of DataLoader workers based on system resources.
    
    Returns:
        Recommended number of workers (typically 8 for GPU training)
    """
    cpu_count = os.cpu_count() or 4
    
    # For GPU training, use more workers to keep GPU fed
    # For CPU training, use fewer to avoid oversubscription
    try:
        import torch
        if torch.cuda.is_available():
            # GPU training: 8 workers is typically optimal
            # More workers don't help much and increase memory overhead
            return min(8, cpu_count)
    except ImportError:
        pass
    
    # CPU training: use number of physical cores
    # Leave 2 cores for system/other work
    return max(2, cpu_count - 2)


def estimate_model_memory(
    model: 'torch.nn.Module',
    input_size: tuple,
    batch_size: int = 1,
    include_gradients: bool = True,
    include_optimizer: bool = True,
) -> float:
    """
    Estimate memory required for a model during training.
    
    Args:
        model: PyTorch model
        input_size: Tuple of input dimensions (excluding batch)
        batch_size: Batch size for estimation
        include_gradients: Include gradient memory
        include_optimizer: Include optimizer state memory
    
    Returns:
        Estimated memory in GB
    """
    import torch
    
    # Parameter memory
    param_memory = 0
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    # Gradient memory
    grad_memory = param_memory if include_gradients else 0
    
    # Optimizer state (Adam uses 2 additional buffers per param)
    optimizer_memory = 0
    if include_optimizer:
        optimizer_memory = 2 * param_memory  # Adam: m and v
    
    # Activation memory (rough estimate based on input size)
    # This is model-dependent and varies significantly
    input_memory = batch_size * input_size[0] * 4  # Assume float32
    activation_memory = input_memory * 3  # Rough estimate
    
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
    return total_memory / 1e9  # Convert to GB


def print_gpu_summary():
    """Print a summary of GPU capabilities and current settings."""
    import torch
    
    caps = get_gpu_capabilities()
    
    print("\n" + "=" * 60)
    print("GPU Summary")
    print("=" * 60)
    
    if not caps['is_available']:
        print("GPU: Not available")
        print("=" * 60)
        return
    
    print(f"Device: {caps['gpu_name']}")
    print(f"VRAM: {caps['vram_gb']:.1f} GB")
    print(f"Compute Capability: {caps['compute_capability'][0]}.{caps['compute_capability'][1]}")
    print(f"Multiprocessors: {caps['multi_processor_count']}")
    print(f"TF32 Supported: {'Yes' if caps['tf32_supported'] else 'No'}")
    print(f"BF16 Supported: {'Yes' if caps['bf16_supported'] else 'No'}")
    print(f"Flash Attention: {'Yes' if check_flash_attention_available() else 'No'}")
    
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory_usage()
        print(f"Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("=" * 60 + "\n")


# =============================================================================
# INITIALIZATION HELPER
# =============================================================================

def initialize_gpu_optimizations(
    enable_tf32: bool = True,
    enable_benchmark: bool = True,
    log_summary: bool = True,
) -> dict:
    """
    One-stop initialization for GPU training optimizations.
    
    Call this once at the start of training for optimal performance.
    
    Args:
        enable_tf32: Enable TF32 for faster matmul
        enable_benchmark: Enable cuDNN benchmark
        log_summary: Print GPU summary
    
    Returns:
        Dictionary with optimization status
    """
    import torch
    
    result = {
        'gpu_available': False,
        'tf32_enabled': False,
        'bf16_available': False,
        'flash_attention_available': False,
        'cudnn_benchmark': False,
        'optimal_workers': get_optimal_dataloader_workers(),
    }
    
    if not check_gpu_compatibility():
        if log_summary:
            print_gpu_summary()
        return result
    
    result['gpu_available'] = True
    
    # Apply optimizations
    settings = optimize_for_training(
        enable_tf32=enable_tf32,
        enable_cudnn_benchmark=enable_benchmark,
    )
    
    result['tf32_enabled'] = settings['tf32_enabled']
    result['bf16_available'] = is_bf16_supported()
    result['flash_attention_available'] = check_flash_attention_available()
    result['cudnn_benchmark'] = settings['cudnn_benchmark']
    
    if log_summary:
        print_gpu_summary()
    
    return result
