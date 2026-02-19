"""Configuration management for NBA prediction system."""

from .config import (
    Config,
    load_config,
    save_config,
    get_config,
    set_config,
    DataConfig,
    TrainingConfig,
    FeatureConfig,
    SimulationConfig,
    LoggingConfig,
    CatBoostConfig,
    LSTMConfig,
    TransformerConfig,
    GNNConfig,
    EnsembleConfig,
)

from .model_config import (
    detect_hardware,
    generate_model_config,
    get_model_config,
    save_model_config,
    load_model_config,
    apply_compile,
    print_config_summary,
    GPU_REGISTRY,
)

__all__ = [
    'Config',
    'load_config',
    'save_config',
    'get_config',
    'set_config',
    'DataConfig',
    'TrainingConfig',
    'FeatureConfig',
    'SimulationConfig',
    'LoggingConfig',
    'CatBoostConfig',
    'LSTMConfig',
    'TransformerConfig',
    'GNNConfig',
    'EnsembleConfig',
    # Auto-sizing model config
    'detect_hardware',
    'generate_model_config',
    'get_model_config',
    'save_model_config',
    'load_model_config',
    'apply_compile',
    'print_config_summary',
    'GPU_REGISTRY',
]
