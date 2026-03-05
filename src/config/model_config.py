"""
Auto-detect hardware and generate optimal model configuration.

This module detects GPU/CPU capabilities and dynamically sizes models
to maximize training efficiency on the available hardware.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# GPU registry with compute multipliers
# Higher multiplier = more aggressive scaling for that GPU
GPU_REGISTRY = {
    'H100': {'mult': 3.0, 'vram_default': 80},
    'A100': {'mult': 2.5, 'vram_default': 40},  # A100-40GB default
    'V100': {'mult': 2.0, 'vram_default': 16},
    'L4':   {'mult': 1.5, 'vram_default': 24},
    'T4':   {'mult': 1.0, 'vram_default': 16},
    'RTX 4090': {'mult': 2.2, 'vram_default': 24},
    'RTX 4080': {'mult': 1.8, 'vram_default': 16},
    'RTX 3090': {'mult': 1.6, 'vram_default': 24},
    'RTX 3080': {'mult': 1.4, 'vram_default': 10},
    'P100': {'mult': 1.2, 'vram_default': 16},
    'K80':  {'mult': 0.8, 'vram_default': 12},
}


def detect_hardware() -> Dict[str, Any]:
    """
    Detect GPU or CPU hardware and calculate compute score.
    
    Returns a dict with:
        - type: 'gpu' or 'cpu'
        - name: Hardware name
        - score: Compute score for scaling decisions
        - vram: GPU memory in GB (0 for CPU)
        - cores: CPU cores
        - ram: System RAM in GB
    """
    result = {
        'type': 'cpu',
        'name': 'Unknown',
        'score': 1.0,
        'vram': 0.0,
        'cores': 4,
        'ram': 8.0,
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)  # Convert to GB
            
            # Compute capability
            compute_cap = props.major + props.minor / 10
            
            # Find matching GPU in registry
            mult = 1.0
            for key, val in GPU_REGISTRY.items():
                if key.upper() in gpu_name.upper():
                    mult = val['mult']
                    break
            
            # Adjust for 80GB A100 variant
            if 'A100' in gpu_name.upper() and vram > 70:
                mult = 2.8
            
            # Adjust for HBM memory (H100, A100 with HBM)
            if 'HBM' in gpu_name.upper():
                mult *= 1.1
            
            score = vram * mult
            
            result = {
                'type': 'gpu',
                'name': gpu_name,
                'score': score,
                'vram': round(vram, 2),
                'cores': 0,
                'ram': 0.0,
                'compute_cap': compute_cap,
                'mult': mult,
            }
            
            logger.info(f"GPU detected: {gpu_name} ({vram:.1f}GB, score={score:.1f})")
            return result
    except ImportError:
        logger.warning("PyTorch not available, falling back to CPU detection")
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}, falling back to CPU")
    
    # CPU detection
    try:
        import psutil
        cores = os.cpu_count() or 4
        ram = psutil.virtual_memory().total / (1024 ** 3)
        
        # CPU score: scales with cores and RAM
        # Typical 8-core/32GB machine scores ~5
        # High-end 64-core/256GB machine scores ~20
        score = cores * 0.05 + ram * 0.02
        
        result = {
            'type': 'cpu',
            'name': f"CPU ({cores} cores, {ram:.1f}GB RAM)",
            'score': max(1.0, score),
            'vram': 0.0,
            'cores': cores,
            'ram': round(ram, 1),
        }
        
        logger.info(f"CPU detected: {cores} cores, {ram:.1f}GB RAM (score={score:.1f})")
    except ImportError:
        logger.warning("psutil not available, using minimal CPU config")
    
    return result


def generate_model_config(score: float, vram: float = 0.0) -> Dict[str, Any]:
    """
    Generate optimal model configuration based on compute score.
    
    Args:
        score: Compute score from detect_hardware()
        vram: GPU memory in GB (0 for CPU)
    
    Returns:
        Dict with model configurations for all models
    """
    # Scale factor normalized to T4 baseline (score=16)
    scale = max(1.0, score / 16.0)
    
    # Pre-calculate values that need divisibility checks
    _tx_nhead = _clamp(4 + int(scale / 4), 4, 16)
    _tx_d_model = _clamp(int(128 * (scale ** 0.5)), 64, 768)
    _tx_d_model = (_tx_d_model // _tx_nhead) * _tx_nhead
    _tx_d_model = max(64, _tx_d_model)
    
    _temp_num_heads = _clamp(4 + int(scale / 5), 4, 12)
    _temp_hidden = _clamp(int(128 * (scale ** 0.5)), 64, 512)
    _temp_hidden = (_temp_hidden // _temp_num_heads) * _temp_num_heads
    _temp_hidden = max(64, _temp_hidden)
    
    config = {
        # ===== LSTM CONFIG =====
        'lstm': {
            'hidden_dim': _clamp(int(128 * (scale ** 0.6)), 64, 1024),
            'num_layers': _clamp(2 + int(scale / 5), 2, 6),
            'bidirectional': score >= 16,
            'dropout': min(0.4, 0.2 + scale * 0.01),
            'batch_size': _clamp(int(32 * scale), 32, 2048),
            'epochs': _clamp(50 + int(scale * 2), 50, 200),
            'lr': max(1e-5, 1e-3 / (scale ** 0.1)),
            'warmup_ratio': 0.1,
            'seq_len': 10,
            'grad_checkpoint': score > 20,  # Enable gradient checkpointing for larger models
            'use_compile': score > 10,  # Enable torch.compile for PyTorch 2.0+
        },
        
        # ===== TRANSFORMER CONFIG =====
        'transformer': {
            'd_model': _tx_d_model,
            'nhead': _tx_nhead,
            'num_layers': _clamp(4 + int(scale / 6), 2, 12),
            'dim_feedforward': _clamp(int(512 * (scale ** 0.4)), 256, 2048),
            'dropout': min(0.25, 0.1 + scale * 0.005),
            'batch_size': _clamp(int(32 * scale), 32, 1024),
            'epochs': _clamp(50 + int(scale * 2), 50, 200),
            'lr': max(2e-5, 5e-4 / (scale ** 0.1)),
            'warmup_ratio': 0.1,
            'grad_checkpoint': score > 30,
            'seq_len': 50,
            'use_compile': score > 10,  # Enable torch.compile for PyTorch 2.0+
        },
        
        # ===== MULTI-OUTPUT NN CONFIG =====
        'nn': {
            'hidden_dim': _clamp(int(512 * (scale ** 0.5)), 256, 2048),
            'num_blocks': _clamp(6 + int(scale / 4), 4, 16),
            'dropout': min(0.4, 0.2 + scale * 0.01),
            'batch_size': _clamp(int(512 * scale), 256, 8192),
            'epochs': _clamp(100 + int(scale * 3), 100, 300),
            'lr': max(5e-5, 1e-3 / (scale ** 0.15)),
            'warmup_ratio': 0.05,
            'label_smoothing': 0.05 if score > 20 else 0,
            'use_compile': score > 10,  # Enable torch.compile for PyTorch 2.0+
        },
        
        # ===== GNN CONFIG =====
        'gnn': {
            'hidden_dim': _clamp(int(64 * (scale ** 0.5)), 32, 256),
            'num_layers': _clamp(2 + int(scale / 10), 2, 4),
            'dropout': 0.2,
            'batch_size': _clamp(int(64 * scale), 32, 512),
            'epochs': _clamp(50 + int(scale), 50, 100),
            'lr': max(5e-4, 1e-2 / (scale ** 0.1)),
            'use_attention': True,
            'use_compile': score > 10,  # Enable torch.compile for PyTorch 2.0+
        },
        
        # ===== TEMPORAL ATTENTION CONFIG =====
        'temporal': {
            'hidden_dim': _temp_hidden,
            'num_heads': _temp_num_heads,
            'dropout': min(0.25, 0.1 + scale * 0.005),
            'batch_size': _clamp(int(32 * scale), 32, 256),
            'epochs': _clamp(50 + int(scale * 1.5), 50, 150),
            'lr': max(2e-5, 5e-4 / (scale ** 0.1)),
            'warmup_ratio': 0.1,
            'seq_len': 20,
            'use_compile': score > 10,  # Enable torch.compile for PyTorch 2.0+
        },
        
        # ===== CATBOOST CONFIG =====
        'catboost': {
            'iterations': _clamp(int(5000 * (scale ** 0.4)), 3000, 10000),
            'depth': _clamp(9 + int(scale / 8), 6, 12),
            'learning_rate': max(0.005, 0.02 / (scale ** 0.15)),
            'l2_leaf_reg': 4.0 + scale * 0.3,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'border_count': 254,
            'grow_policy': 'Depthwise',
            'min_data_in_leaf': max(5, 10 - int(scale / 5)),
            'score_function': 'Cosine',
            'rsm': max(0.6, 0.8 - scale * 0.005),
            'langevin': score >= 10,
            'diffusion_temperature': 10000.0,
            'early_stopping_rounds': _clamp(30 + int(scale * 2), 20, 100),  # Reduced from 100-300 for faster training
            'use_multi_loss': True,
            'multi_loss_rmse_weight': 0.6,
            'multi_loss_mae_weight': 0.4,
            'use_quantile_models': True,  # Always train quantile models for uncertainty
            'n_temporal_folds': _clamp(3 + int(scale / 10), 2, 5),
            'use_per_target_tuning': True,
        },
        
        # ===== XGBOOST CONFIG =====
        'xgboost': {
            'n_estimators': _clamp(int(1000 * (scale ** 0.5)), 500, 3000),
            'max_depth': _clamp(6 + int(scale / 10), 4, 12),
            'learning_rate': max(0.005, 0.03 / (scale ** 0.15)),
            'subsample': max(0.6, 0.8 - scale * 0.01),
            'colsample_bytree': max(0.6, 0.8 - scale * 0.01),
            'colsample_bylevel': max(0.6, 0.8 - scale * 0.01),
            'reg_alpha': max(0.1, scale * 0.05),
            'reg_lambda': max(1.0, 2.0 + scale * 0.2),
            'min_child_weight': max(1, int(5 - scale * 0.2)),
            'gamma': max(0, scale * 0.01),
            'early_stopping_rounds': _clamp(50 + int(scale * 5), 30, 150),
            'random_state': 42,
            'n_jobs': -1,
            'use_gpu': score >= 10,
        },
        
        # ===== LIGHTGBM CONFIG =====
        'lightgbm': {
            'n_estimators': _clamp(int(1500 * (scale ** 0.6)), 500, 5000),
            'max_depth': _clamp(6 + int(scale / 12), -1, 15),
            'learning_rate': max(0.005, 0.05 / (scale ** 0.2)),
            'num_leaves': _clamp(31 + int(scale * 3), 16, 128),
            'feature_fraction': max(0.6, 0.8 - scale * 0.01),
            'bagging_fraction': max(0.6, 0.8 - scale * 0.01),
            'bagging_freq': 1,
            'min_child_samples': max(5, int(20 - scale * 0.5)),
            'reg_alpha': max(0.1, scale * 0.05),
            'reg_lambda': max(0.1, scale * 0.1),
            'early_stopping_rounds': _clamp(50 + int(scale * 5), 30, 150),
            'random_state': 42,
            'n_jobs': -1,
            'use_gpu': score >= 10,
            'verbose': -1,
        },
        
        # ===== TRAINING CONFIG =====
        'training': {
            'test_split_date': '2024-03-01',
            'warmup_steps': _clamp(int(scale * 20), 0, 1000),
            'early_stop_patience': _clamp(15 + int(scale / 2), 15, 40),
            'label_smoothing': 0.05 if score > 20 else 0,
            'use_compile': score > 10,  # Enable torch.compile for PyTorch 2.0+ on capable hardware
            'amp': True,  # Always use mixed precision
            'gradient_accumulation_steps': 1 if score > 10 else 2,
        },
        
        # ===== METADATA =====
        'metadata': {
            'score': round(score, 2),
            'vram_gb': round(vram, 2),
            'scale_factor': round(scale, 2),
            'generated_at': datetime.now().isoformat(),
        }
    }
    
    # Memory safety check for GPU
    if vram > 0:
        config = _validate_memory(config, vram)
    
    return config


def _clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def _validate_memory(config: Dict[str, Any], vram_gb: float) -> Dict[str, Any]:
    """
    Validate config fits in VRAM and scale down if necessary.
    
    Estimates memory usage and scales down hidden dims if needed.
    """
    # Estimate parameters
    lstm_h = config['lstm']['hidden_dim']
    lstm_l = config['lstm']['num_layers']
    lstm_bi = 2 if config['lstm']['bidirectional'] else 1
    
    tx_d = config['transformer']['d_model']
    tx_l = config['transformer']['num_layers']
    tx_ff = config['transformer']['dim_feedforward']
    
    nn_h = config['nn']['hidden_dim']
    nn_b = config['nn']['num_blocks']
    
    # Rough parameter counts (params * 4 bytes for float32)
    # LSTM: 4 * (input*hidden + hidden*hidden + bias) * layers * direction
    lstm_params = 4 * (lstm_h * lstm_h + lstm_h) * lstm_l * lstm_bi
    
    # Transformer: embedding + attention + ff per layer
    # Attention: 4 * d^2 per layer (Q,K,V,O projections)
    # FF: 2 * d * ff_dim per layer
    tx_params = tx_l * (4 * tx_d * tx_d + 2 * tx_d * tx_ff)
    
    # NN: hidden * hidden per block * 2 (fc layers) + some overhead
    nn_params = nn_b * (nn_h * nn_h * 2 + nn_h * 4)  # rough estimate
    
    # Total model memory (params * 4 bytes * 2 for gradients + activations overhead)
    model_mem_gb = (lstm_params + tx_params + nn_params) * 4 * 3 / (1024 ** 3)
    
    # Batch memory (activations)
    batch_size = config['nn']['batch_size']
    batch_mem_gb = batch_size * nn_h * 4 / (1024 ** 3)  # rough estimate
    
    total_estimated = model_mem_gb + batch_mem_gb
    budget = vram_gb * 0.7  # 70% safety margin
    
    if total_estimated > budget:
        scale_down = (budget / total_estimated) ** 0.5  # Scale dims by sqrt
        logger.warning(
            f"Memory budget exceeded ({total_estimated:.1f}GB > {budget:.1f}GB), "
            f"scaling hidden dims by {scale_down:.2f}"
        )
        
        # Scale down hidden dimensions
        config['lstm']['hidden_dim'] = int(config['lstm']['hidden_dim'] * scale_down)
        # Ensure d_model remains divisible by nhead
        _scaled_d_model = int(config['transformer']['d_model'] * scale_down)
        config['transformer']['d_model'] = (_scaled_d_model // config['transformer']['nhead']) * config['transformer']['nhead']
        config['transformer']['dim_feedforward'] = int(config['transformer']['dim_feedforward'] * scale_down)
        config['nn']['hidden_dim'] = int(config['nn']['hidden_dim'] * scale_down)
        config['gnn']['hidden_dim'] = int(config['gnn']['hidden_dim'] * scale_down)
        # Ensure temporal hidden_dim remains divisible by num_heads
        _scaled_temp_hidden = int(config['temporal']['hidden_dim'] * scale_down)
        config['temporal']['hidden_dim'] = (_scaled_temp_hidden // config['temporal']['num_heads']) * config['temporal']['num_heads']
        
        # Scale batch sizes
        config['lstm']['batch_size'] = int(config['lstm']['batch_size'] * scale_down)
        config['transformer']['batch_size'] = int(config['transformer']['batch_size'] * scale_down)
        config['nn']['batch_size'] = int(config['nn']['batch_size'] * scale_down)
        config['gnn']['batch_size'] = int(config['gnn']['batch_size'] * scale_down)
        config['temporal']['batch_size'] = int(config['temporal']['batch_size'] * scale_down)
        
        # Add flag that we scaled
        config['metadata']['memory_scaled'] = True
        config['metadata']['memory_scale_factor'] = round(scale_down, 3)
    
    return config


def get_model_config(
    force_size: Optional[str] = None,
    custom_score: Optional[float] = None,
    custom_vram: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get model configuration based on detected or specified hardware.
    
    Args:
        force_size: Override auto-detection. One of 'small', 'medium', 'large', 'pro', 'ultra'
        custom_score: Override detected score
        custom_vram: Override detected VRAM
    
    Returns:
        Tuple of (model_config, hardware_info)
    """
    # Hardcoded size presets
    SIZE_PRESETS = {
        'small': (5.0, 0.0),    # CPU-like
        'medium': (16.0, 16.0),  # T4-like
        'large': (50.0, 40.0),   # A100-40GB-like
        'pro': (140.0, 80.0),    # A100-80GB-like
        'ultra': (240.0, 80.0),  # H100-like
    }
    
    if force_size and force_size.lower() in SIZE_PRESETS:
        score, vram = SIZE_PRESETS[force_size.lower()]
        hw_info = {
            'type': 'preset',
            'name': f"preset:{force_size}",
            'score': score,
            'vram': vram,
        }
    else:
        # Auto-detect
        hw_info = detect_hardware()
        score = hw_info['score']
        vram = hw_info['vram']
        
        # Override if specified
        if custom_score is not None:
            score = custom_score
            hw_info['score'] = score
        if custom_vram is not None:
            vram = custom_vram
            hw_info['vram'] = vram
    
    config = generate_model_config(score, vram)
    
    return config, hw_info


def save_model_config(config: Dict[str, Any], path: str = 'models/training_config.json') -> str:
    """
    Save model configuration to JSON for reproducibility.
    
    Args:
        config: Model configuration dict
        path: Path to save JSON file
    
    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Model config saved to {path}")
    return str(path)


def load_model_config(path: str = 'models/training_config.json') -> Optional[Dict[str, Any]]:
    """
    Load model configuration from JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Config dict or None if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


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
    if not use_compile:
        return model
    
    try:
        import torch
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile() not available (PyTorch < 2.0)")
            return model
        
        logger.info(f"Compiling {model_name} with torch.compile()...")
        compiled_model = torch.compile(model, mode='reduce-overhead')
        logger.info(f"{model_name} compiled successfully")
        return compiled_model
    except Exception as e:
        logger.warning(f"torch.compile() failed for {model_name}: {e}")
        return model


def print_config_summary(config: Dict[str, Any], hw_info: Dict[str, Any]) -> None:
    """Print a formatted summary of the detected hardware and generated config."""
    lines = [
        "=" * 50,
        "NBA MODEL TRAINER - Auto Configuration",
        "=" * 50,
        f"Detected Hardware:",
        f"  Type: {hw_info.get('type', 'unknown').upper()}",
        f"  Name: {hw_info.get('name', 'unknown')}",
        f"  Compute Score: {hw_info.get('score', 0):.1f}",
    ]
    
    if hw_info.get('vram', 0) > 0:
        lines.append(f"  VRAM: {hw_info.get('vram', 0):.1f} GB")
    if hw_info.get('cores', 0) > 0:
        lines.append(f"  CPU Cores: {hw_info.get('cores', 0)}")
    if hw_info.get('ram', 0) > 0:
        lines.append(f"  RAM: {hw_info.get('ram', 0):.1f} GB")
    
    scale = config.get('metadata', {}).get('scale_factor', 1.0)
    lines.extend([
        "",
        f"Generated Config (scale={scale:.2f}):",
        f"  LSTM:",
        f"    hidden={config['lstm']['hidden_dim']}, layers={config['lstm']['num_layers']}, "
        f"bidirectional={config['lstm']['bidirectional']}",
        f"  Transformer:",
        f"    d_model={config['transformer']['d_model']}, heads={config['transformer']['nhead']}, "
        f"layers={config['transformer']['num_layers']}",
        f"  MultiOutputNN:",
        f"    hidden={config['nn']['hidden_dim']}, blocks={config['nn']['num_blocks']}",
        f"  GNN:",
        f"    hidden={config['gnn']['hidden_dim']}, layers={config['gnn']['num_layers']}",
        f"  TemporalAttention:",
        f"    hidden={config['temporal']['hidden_dim']}, heads={config['temporal']['num_heads']}",
        f"  CatBoost:",
        f"    iterations={config['catboost']['iterations']}, depth={config['catboost']['depth']}, "
        f"grow={config['catboost']['grow_policy']}, multi_loss={config['catboost']['use_multi_loss']}, "
        f"quantile={config['catboost']['use_quantile_models']}",
        "",
        f"Training Settings:",
        f"  Warmup Steps: {config['training']['warmup_steps']}",
        f"  Early Stop Patience: {config['training']['early_stop_patience']}",
        f"  torch.compile: {'Yes' if config['training']['use_compile'] else 'No'}",
        f"  Gradient Checkpointing: {'Yes' if config['transformer']['grad_checkpoint'] else 'No'}",
        f"  Mixed Precision (AMP): {'Yes' if config['training']['amp'] else 'No'}",
        "=" * 50,
    ])
    
    print("\n".join(lines))