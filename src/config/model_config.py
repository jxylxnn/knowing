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

SIZE_TIER_ALIASES = {
    'small': 'S',
    'medium': 'M',
    'large': 'L',
    'pro': 'XL',
    'ultra': 'XL',
}

SIZE_TIER_ORDER = ['S', 'M', 'L', 'XL']

SIZE_TIER_SPECS = {
    'S': {
        'catboost': {
            'iterations': 200,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 8.0,
            'min_data_in_leaf': 20,
            'early_stopping_rounds': 25,
            'rsm': 1.0,
            'langevin': False,
            'use_multi_loss': False,
            'use_quantile_models': True,
        },
        'transformer': {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 1,
            'dim_feedforward': 256,
            'dropout': 0.12,
            'batch_size': 256,
            'epochs': 25,
            'lr': 1e-3,
            'seq_len': 5,
            'max_seq_length': 5,
        },
        'training': {
            'warmup_steps': 0,
            'early_stop_patience': 8,
            'use_compile': False,
            'compile_mode': 'reduce-overhead',
            'amp': False,
            'use_bf16': False,
            'gradient_accumulation_steps': 2,
            'tf32_enabled': False,
        },
        'simulation': {
            'default_num_sims': 1000,
            'fast_path_threshold': 250,
            'detailed_path_threshold': 100,
        },
    },
    'M': {
        'catboost': {
            'iterations': 1500,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 5.0,
            'min_data_in_leaf': 10,
            'early_stopping_rounds': 80,
            'rsm': 0.8,
            'langevin': True,
            'use_multi_loss': False,
            'use_quantile_models': True,
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.15,
            'batch_size': 256,
            'epochs': 60,
            'lr': 8e-4,
            'seq_len': 10,
            'max_seq_length': 10,
        },
        'training': {
            'warmup_steps': 10,
            'early_stop_patience': 12,
            'use_compile': True,
            'compile_mode': 'reduce-overhead',
            'amp': True,
            'use_bf16': False,
            'gradient_accumulation_steps': 1,
            'tf32_enabled': True,
        },
        'simulation': {
            'default_num_sims': 5000,
            'fast_path_threshold': 500,
            'detailed_path_threshold': 250,
        },
    },
    'L': {
        'catboost': {
            'iterations': 5000,
            'depth': 8,
            'learning_rate': 0.015,
            'l2_leaf_reg': 4.5,
            'min_data_in_leaf': 8,
            'early_stopping_rounds': 120,
            'rsm': 0.7,
            'langevin': True,
            'use_multi_loss': False,
            'use_quantile_models': True,
        },
        'transformer': {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.15,
            'batch_size': 192,
            'epochs': 100,
            'lr': 5e-4,
            'seq_len': 20,
            'max_seq_length': 20,
        },
        'training': {
            'warmup_steps': 20,
            'early_stop_patience': 16,
            'use_compile': True,
            'compile_mode': 'reduce-overhead',
            'amp': True,
            'use_bf16': True,
            'gradient_accumulation_steps': 1,
            'tf32_enabled': True,
        },
        'simulation': {
            'default_num_sims': 10000,
            'fast_path_threshold': 1000,
            'detailed_path_threshold': 500,
        },
    },
    'XL': {
        'catboost': {
            'iterations': 10000,
            'depth': 10,
            'learning_rate': 0.01,
            'l2_leaf_reg': 4.0,
            'min_data_in_leaf': 6,
            'early_stopping_rounds': 160,
            'rsm': 0.65,
            'langevin': True,
            'use_multi_loss': False,
            'use_quantile_models': True,
        },
        'transformer': {
            'd_model': 512,
            'nhead': 16,
            'num_encoder_layers': 12,
            'dim_feedforward': 2048,
            'dropout': 0.18,
            'batch_size': 128,
            'epochs': 150,
            'lr': 3e-4,
            'seq_len': 50,
            'max_seq_length': 50,
        },
        'training': {
            'warmup_steps': 40,
            'early_stop_patience': 20,
            'use_compile': True,
            'compile_mode': 'max-autotune',
            'amp': True,
            'use_bf16': True,
            'gradient_accumulation_steps': 1,
            'tf32_enabled': True,
        },
        'simulation': {
            'default_num_sims': 25000,
            'fast_path_threshold': 2000,
            'detailed_path_threshold': 1000,
        },
    },
}


def normalize_model_size(force_size: Optional[str]) -> Optional[str]:
    """Normalize a requested model size to one of S/M/L/XL or ``auto``."""
    if force_size is None:
        return None

    raw = str(force_size).strip()
    if not raw:
        return None
    if raw.lower() == 'auto':
        return 'auto'

    alias = SIZE_TIER_ALIASES.get(raw.lower())
    if alias is not None:
        return alias

    tier = raw.upper()
    if tier in SIZE_TIER_ORDER:
        return tier

    raise ValueError(f"Unsupported model size '{force_size}'. Expected auto or one of {SIZE_TIER_ORDER}.")


def _tier_from_score(score: float) -> str:
    """Map a hardware score to a default size tier."""
    if score < 8:
        return 'S'
    if score < 24:
        return 'M'
    if score < 80:
        return 'L'
    return 'XL'


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
    """Generate a size-tiered model config from a hardware score."""
    tier = _tier_from_score(score)
    spec = SIZE_TIER_SPECS[tier]

    # Keep the legacy shape of the config object, but make the active path
    # explicit: CatBoost + Transformer.
    transformer_cfg = {
        'enabled': True,
        'd_model': spec['transformer']['d_model'],
        'nhead': spec['transformer']['nhead'],
        'num_layers': spec['transformer']['num_encoder_layers'],
        'num_encoder_layers': spec['transformer']['num_encoder_layers'],
        'dim_feedforward': spec['transformer']['dim_feedforward'],
        'dropout': spec['transformer']['dropout'],
        'batch_size': spec['transformer']['batch_size'],
        'epochs': spec['transformer']['epochs'],
        'lr': spec['transformer']['lr'],
        'warmup_ratio': 0.1,
        'grad_checkpoint': tier in {'L', 'XL'},
        'use_compile': spec['training']['use_compile'],
        'seq_len': spec['transformer']['seq_len'],
        'max_seq_length': spec['transformer']['max_seq_length'],
    }

    config = {
        'lstm': {
            'enabled': False,
            'hidden_dim': 64 if tier == 'S' else 128,
            'num_layers': 1 if tier == 'S' else 2,
            'bidirectional': tier != 'S',
            'dropout': 0.2,
            'batch_size': 128,
            'epochs': 10,
            'lr': 1e-3,
            'warmup_ratio': 0.1,
            'seq_len': spec['transformer']['seq_len'],
            'grad_checkpoint': False,
            'use_compile': False,
        },
        'transformer': transformer_cfg,
        'temporal': {
            'hidden_dim': transformer_cfg['d_model'],
            'num_heads': transformer_cfg['nhead'],
            'dropout': transformer_cfg['dropout'],
            'batch_size': transformer_cfg['batch_size'],
            'epochs': transformer_cfg['epochs'],
            'lr': transformer_cfg['lr'],
            'warmup_ratio': transformer_cfg['warmup_ratio'],
            'seq_len': transformer_cfg['seq_len'],
            'use_compile': transformer_cfg['use_compile'],
        },
        'nn': {
            'enabled': False,
            'hidden_dim': 128,
            'num_blocks': 2,
            'dropout': 0.2,
            'batch_size': 256,
            'epochs': 20,
            'lr': 1e-3,
            'warmup_ratio': 0.05,
            'label_smoothing': 0.0,
            'use_compile': False,
        },
        'gnn': {
            'enabled': False,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 64,
            'epochs': 20,
            'lr': 1e-3,
            'use_attention': False,
            'use_compile': False,
        },
        'catboost': {
            'enabled': True,
            'iterations': spec['catboost']['iterations'],
            'learning_rate': spec['catboost']['learning_rate'],
            'depth': spec['catboost']['depth'],
            'l2_leaf_reg': spec['catboost']['l2_leaf_reg'],
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'border_count': 254,
            'thread_count': -1,
            'random_seed': 42,
            'early_stopping_rounds': spec['catboost']['early_stopping_rounds'],
            'grow_policy': 'Depthwise',
            'min_data_in_leaf': spec['catboost']['min_data_in_leaf'],
            'score_function': 'Cosine',
            'rsm': spec['catboost']['rsm'],
            'langevin': spec['catboost']['langevin'],
            'diffusion_temperature': 10000.0,
            'use_multi_loss': spec['catboost']['use_multi_loss'],
            'multi_loss_rmse_weight': 0.6,
            'multi_loss_mae_weight': 0.4,
            'use_quantile_models': spec['catboost']['use_quantile_models'],
            'quantile_alpha_low': 0.1,
            'quantile_alpha_high': 0.9,
            'n_temporal_folds': 3 if tier in {'S', 'M'} else 5,
            'use_per_target_tuning': True,
        },
        'xgboost': {
            'enabled': False,
            'n_estimators': 500,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1,
            'gamma': 0.0,
            'early_stopping_rounds': 50,
            'random_state': 42,
            'n_jobs': -1,
            'use_gpu': False,
        },
        'lightgbm': {
            'enabled': False,
            'n_estimators': 500,
            'max_depth': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'early_stopping_rounds': 50,
            'random_state': 42,
            'n_jobs': -1,
            'use_gpu': False,
            'verbose': -1,
        },
        'training': {
            'test_split_date': '2024-03-01',
            'warmup_steps': spec['training']['warmup_steps'],
            'early_stop_patience': spec['training']['early_stop_patience'],
            'label_smoothing': 0.0,
            'use_compile': spec['training']['use_compile'],
            'compile_mode': spec['training']['compile_mode'],
            'amp': spec['training']['amp'],
            'use_bf16': spec['training']['use_bf16'],
            'gradient_accumulation_steps': spec['training']['gradient_accumulation_steps'],
            'tf32_enabled': spec['training']['tf32_enabled'],
            'dataloader_workers': min(8, os.cpu_count() or 4),
        },
        'simulation': {
            'default_num_sims': spec['simulation']['default_num_sims'],
            'fast_path_threshold': spec['simulation']['fast_path_threshold'],
            'detailed_path_threshold': spec['simulation']['detailed_path_threshold'],
        },
        'metadata': {
            'score': round(score, 2),
            'vram_gb': round(vram, 2),
            'scale_factor': round(max(1.0, score / 16.0), 2),
            'tier': tier,
            'generated_at': datetime.now().isoformat(),
        },
    }

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
        
        # Scale down hidden dimensions.
        config['lstm']['hidden_dim'] = max(32, int(config['lstm']['hidden_dim'] * scale_down))

        _scaled_d_model = max(32, int(config['transformer']['d_model'] * scale_down))
        nhead = max(1, int(config['transformer']['nhead']))
        config['transformer']['d_model'] = max(nhead, (_scaled_d_model // nhead) * nhead)
        config['transformer']['hidden_dim'] = config['transformer']['d_model']
        config['transformer']['dim_feedforward'] = max(64, int(config['transformer']['dim_feedforward'] * scale_down))
        config['transformer']['nhead'] = nhead
        config['transformer']['num_encoder_layers'] = config['transformer']['num_layers']

        # Keep the legacy temporal alias synchronized with the transformer.
        config['temporal']['hidden_dim'] = config['transformer']['d_model']
        config['temporal']['num_heads'] = nhead
        
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
    normalized_size = normalize_model_size(force_size) if force_size is not None else None

    # Hardware presets used when the user explicitly requests a tier.
    SIZE_PRESETS = {
        'S': (5.0, 0.0),
        'M': (16.0, 16.0),
        'L': (50.0, 40.0),
        'XL': (140.0, 80.0),
    }

    if normalized_size and normalized_size != 'auto':
        score, vram = SIZE_PRESETS[normalized_size]
        hw_info = {
            'type': 'preset',
            'name': f"preset:{normalized_size}",
            'score': score,
            'vram': vram,
            'tier': normalized_size,
        }
    else:
        # Auto-detect
        hw_info = detect_hardware()
        score = hw_info['score']
        vram = hw_info['vram']
        hw_info['tier'] = _tier_from_score(score)
        
        # Override if specified
        if custom_score is not None:
            score = custom_score
            hw_info['score'] = score
            hw_info['tier'] = _tier_from_score(score)
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
