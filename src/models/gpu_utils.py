"""
GPU Compatibility Utilities for RTX 50-series (Blackwell) and other GPUs.
Provides a centralized check for GPU availability and CUDA kernel compatibility.
"""

import logging
import torch

logger = logging.getLogger(__name__)

_GPU_COMPATIBLE: bool | None = None  # Cached result


def check_gpu_compatibility() -> bool:
    """
    Check if the GPU is compatible with the installed PyTorch CUDA kernels.
    
    RTX 50-series (Blackwell, compute capability 12.0) requires PyTorch builds
    that include kernels for this architecture. As of PyTorch 2.6.0, this is
    not yet supported.
    
    Returns:
        bool: True if GPU can be used, False if should fallback to CPU.
    """
    global _GPU_COMPATIBLE
    
    # Return cached result if already checked
    if _GPU_COMPATIBLE is not None:
        return _GPU_COMPATIBLE
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available. Using CPU.")
        _GPU_COMPATIBLE = False
        return False
    
    try:
        # Get compute capability
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        
        logger.info(f"Detected GPU: {gpu_name} (Compute Capability: {major}.{minor})")
        
        # RTX 50-series (Blackwell) has compute capability 12.0
        # PyTorch 2.6.0+cu124 doesn't have kernels for this yet
        if major >= 12:
            logger.warning(
                f"GPU '{gpu_name}' uses Blackwell architecture (compute {major}.{minor}) "
                f"which is not yet supported by PyTorch {torch.__version__}. "
                f"Falling back to CPU. Update PyTorch when Blackwell support is released."
            )
            _GPU_COMPATIBLE = False
            return False
        
        # Test if CUDA actually works with a simple operation
        try:
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
            logger.info("CUDA kernel test passed. GPU is compatible.")
            _GPU_COMPATIBLE = True
            return True
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                logger.warning(
                    f"CUDA kernel not available for GPU '{gpu_name}'. "
                    f"This typically means PyTorch wasn't compiled for your GPU architecture. "
                    f"Falling back to CPU."
                )
                _GPU_COMPATIBLE = False
                return False
            raise
            
    except Exception as e:
        logger.warning(f"GPU compatibility check failed: {e}. Using CPU.")
        _GPU_COMPATIBLE = False
        return False


def get_device() -> torch.device:
    """Get the appropriate PyTorch device based on GPU compatibility."""
    if check_gpu_compatibility():
        return torch.device('cuda')
    return torch.device('cpu')


def is_gpu_available() -> bool:
    """Alias for check_gpu_compatibility for clearer code."""
    return check_gpu_compatibility()
