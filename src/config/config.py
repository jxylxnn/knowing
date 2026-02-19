"""Configuration management for NBA prediction system."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class DataConfig:
    """Data loading and storage configuration."""
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cache_dir: Path = Path("cache")
    raw_data_file: str = "nba_games.csv"
    use_parquet: bool = False
    parquet_compression: str = "zstd"
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        self.cache_dir = Path(self.cache_dir)


@dataclass
class CatBoostConfig:
    """CatBoost model configuration."""
    enabled: bool = True
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 0.5
    border_count: int = 128
    thread_count: int = -1  # Use all cores
    random_seed: int = 42
    early_stopping_rounds: int = 100
    

@dataclass
class NeuralNetConfig:
    """Neural network base configuration."""
    enabled: bool = True
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 512
    epochs: int = 100
    patience: int = 10
    random_seed: int = 42


@dataclass
class LSTMConfig(NeuralNetConfig):
    """LSTM-specific configuration."""
    sequence_length: int = 5
    bidirectional: bool = True


@dataclass
class TransformerConfig(NeuralNetConfig):
    """Transformer-specific configuration."""
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    max_seq_length: int = 10


@dataclass
class GNNConfig(NeuralNetConfig):
    """GNN-specific configuration."""
    num_node_features: int = 64
    num_edge_features: int = 16
    num_graph_layers: int = 3


@dataclass
class EnsembleConfig:
    """Ensemble and blending configuration."""
    enabled: bool = True
    method: str = "ridge"  # ridge, weighted_average, stacking
    cv_folds: int = 5
    use_temporal_refinement: bool = True
    use_advanced_temporal: bool = True


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    targets: List[str] = field(default_factory=lambda: ["PTS", "REB", "AST", "STL", "BLK", "TOV"])
    test_split_date: str = "2024-03-01"
    temporal_decay_lambda: float = 0.023
    outlier_percentile: float = 0.99
    use_sample_weights: bool = True
    use_adversarial_validation: bool = True
    adversarial_threshold: float = 0.7
    use_feature_selection: bool = True
    min_samples_per_player: int = 10
    

@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    use_matchup_features: bool = True
    use_fatigue_features: bool = True
    use_momentum_features: bool = True
    use_contextual_features: bool = True
    max_lag_days: int = 7


@dataclass
class SimulationConfig:
    """Game simulation configuration."""
    default_num_sims: int = 1000
    max_workers: int = 4
    use_gpu: bool = True
    cache_rosters: bool = True
    injury_probability_threshold: float = 0.1
    minutes_distribution: str = "dirichlet"  # dirichlet, beta, uniform
    correlation_injection: bool = True
    clutch_adjustment: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: Path = Path("logs")
    max_bytes: int = 10_000_000  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        self.log_dir = Path(self.log_dir)


@dataclass
class Config:
    """Root configuration for the NBA prediction system."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Helper to create Config from nested dictionary."""
        config = cls()
        
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'features' in data:
            config.features = FeatureConfig(**data['features'])
        if 'simulation' in data:
            config.simulation = SimulationConfig(**data['simulation'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        if 'catboost' in data:
            config.catboost = CatBoostConfig(**data['catboost'])
        if 'lstm' in data:
            config.lstm = LSTMConfig(**data['lstm'])
        if 'transformer' in data:
            config.transformer = TransformerConfig(**data['transformer'])
        if 'gnn' in data:
            config.gnn = GNNConfig(**data['gnn'])
        if 'ensemble' in data:
            config.ensemble = EnsembleConfig(**data['ensemble'])
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        def convert_paths(obj):
            """Helper to convert Path objects to strings."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert_paths(self.to_dict()), f, indent=2)
    
    def get_model_config(self, model_name: str) -> Any:
        """Get configuration for a specific model."""
        config_map = {
            'catboost': self.catboost,
            'lstm': self.lstm,
            'transformer': self.transformer,
            'gnn': self.gnn,
        }
        return config_map.get(model_name.lower())


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from file or return default.
    
    Args:
        path: Path to config file. If None, returns default config.
        
    Returns:
        Config object
    """
    if path is None:
        return Config()
    return Config.from_yaml(path)


def save_config(config: Config, path: Path) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration object to save
        path: Path to save to
    """
    config.to_yaml(path)


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Configuration to set as global
    """
    global _config
    _config = config
