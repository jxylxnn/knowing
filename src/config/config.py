"""Configuration management for NBA prediction system."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
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
    """CatBoost model configuration.

    Supports multi-loss training (RMSE + MAE blend), per-target hyperparameter
    profiles, quantile regression for uncertainty estimation, and temporal
    K-fold cross-validation.
    """
    enabled: bool = True
    iterations: int = 1500
    learning_rate: float = 0.02
    depth: int = 6
    l2_leaf_reg: float = 5.0
    random_strength: float = 1.0
    bagging_temperature: float = 0.5
    border_count: int = 254
    thread_count: int = -1
    random_seed: int = 42
    early_stopping_rounds: int = 80

    # Advanced tree structure
    grow_policy: str = "Depthwise"
    min_data_in_leaf: int = 10
    score_function: str = "Cosine"
    rsm: float = 0.8

    # Langevin stochastic gradient boosting
    langevin: bool = True
    diffusion_temperature: float = 10000.0

    # Multi-loss training: blend RMSE + MAE models for robustness
    use_multi_loss: bool = False
    multi_loss_rmse_weight: float = 0.6
    multi_loss_mae_weight: float = 0.4

    # Quantile regression for uncertainty estimation
    use_quantile_models: bool = True
    quantile_alpha_low: float = 0.1
    quantile_alpha_high: float = 0.9

    # Temporal K-fold cross-validation
    n_temporal_folds: int = 3

    # Per-target hyperparameter overrides (populated at runtime)
    use_per_target_tuning: bool = True
    

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
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    max_seq_length: int = 10


@dataclass
class GNNConfig(NeuralNetConfig):
    """GNN-specific configuration."""
    num_node_features: int = 64
    num_edge_features: int = 16
    num_graph_layers: int = 3
    use_attention: bool = True


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
    test_split_date: str = "2025-01-01"
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
    seed: int = 42
    use_context_engine: bool = True
    use_player_correlations: bool = True
    use_betting_calibration: bool = True
    use_minutes_model: bool = True
    use_error_calibration: bool = True
    four_factors_weight: float = 0.25
    detailed_path_threshold: int = 250
    fast_path_threshold: int = 1000


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
class HTTPConfig:
    """HTTP request configuration."""
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class CacheConfig:
    """Cache TTL configuration (in hours unless specified)."""
    schedule_ttl_hours: float = 1.0
    season_schedule_ttl_days: float = 1.0
    betting_lines_ttl_hours: float = 4.0
    defense_stats_ttl_hours: float = 12.0
    lineup_ttl_hours: float = 6.0
    injury_ttl_minutes: float = 30.0
    rotowire_ttl_minutes: float = 30.0
    basketball_ref_ttl_hours: float = 6.0
    directory: Path = Path("cache")
    
    def __post_init__(self):
        self.directory = Path(self.directory)


@dataclass
class APIConfig:
    """API endpoint configuration."""
    # NBA Stats API
    nba_stats_base_url: str = "https://stats.nba.com/stats"
    nba_player_logs_url: str = "https://stats.nba.com/stats/playergamelogs"
    nba_matchups_url: str = "https://stats.nba.com/stats/leagueplayerposstats"
    bref_ratings_url_base: str = "https://www.basketball-reference.com/leagues/NBA_{year}_ratings.html"
    
    # Basketball Reference
    basketball_reference_base_url: str = "https://www.basketball-reference.com"
    
    # ESPN
    espn_injuries_url: str = "https://www.espn.com/nba/injuries"
    espn_health_api_url: str = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/health"
    
    # Action Network (Betting)
    action_network_base_url: str = "https://www.actionnetwork.com/nba"
    
    # RotoWire
    rotowire_lineups_url: str = "https://www.rotowire.com/basketball/nba-lineups.php"
    rotowire_lineups_daily_url: str = "https://www.rotowire.com/basketball/nba-lineups-daily.php"
    rotowire_projections_url: str = "https://www.rotowire.com/basketball/projections-daily.php"
    
    # Rate limiting
    rate_limit_delay: float = 0.6
    team_fetch_delay: float = 0.5
    team_fetch_delay_bref: float = 1.5


@dataclass
class TeamsConfig:
    """Team configuration."""
    mappings_file: Path = Path("data/team_mappings.json")
    switch_profiles_file: Path = Path("data/team_switch_profiles.json")
    interior_defense_file: Path = Path("data/team_interior_defense.json")
    default_opponent: str = "LAL"


@dataclass
class LeagueAveragesConfig:
    """League average statistics configuration (fallback defaults)."""
    points_per_100: float = 114.0
    offensive_rating: float = 114.0
    defensive_rating: float = 114.0
    pace: float = 100.0
    effective_fg_pct: float = 0.54
    turnover_pct: float = 0.135
    offensive_rebound_pct: float = 0.25
    free_throw_rate: float = 0.23
    
    # Default game totals
    default_total: float = 225.0
    implied_home_pts: float = 112.5
    default_home_rating: float = 114.0
    default_away_rating: float = 114.0
    
    # Player position defaults
    fg_pct: float = 0.470
    fg3_pct: float = 0.360
    ft_pct: float = 0.75
    
    # Historical spread std
    historical_std_total: float = 11.0
    historical_std_spread: float = 10.5
    
    # Vegas calibration
    vegas_weight: float = 0.30


@dataclass
class SimulationParamsConfig:
    """Detailed game simulation parameters."""
    # Fatigue multipliers (days of rest)
    fatigue_3in4: float = 0.96
    fatigue_back_to_back: float = 0.975
    fatigue_2_days: float = 1.0
    fatigue_3_days: float = 1.01
    fatigue_4plus_days: float = 1.005
    
    # Defense adjustment bounds
    defense_pts_range: tuple = field(default_factory=lambda: (0.85, 1.15))
    defense_reb_range: tuple = field(default_factory=lambda: (0.90, 1.10))
    defense_ast_range: tuple = field(default_factory=lambda: (0.92, 1.08))
    defense_tov_range: tuple = field(default_factory=lambda: (0.90, 1.10))
    
    # Pace distribution
    pace_multiplier_mean: float = 1.0
    pace_multiplier_std: float = 0.04
    pace_clip_range: tuple = field(default_factory=lambda: (0.88, 1.12))
    
    # Environment factor
    env_factor_mean: float = 1.0
    env_factor_std: float = 0.05
    env_factor_clip_range: tuple = field(default_factory=lambda: (0.88, 1.15))
    
    # Minutes distribution
    minutes_std_high: float = 2.0
    minutes_std_low: float = 3.5
    minutes_threshold: float = 32.0
    
    # Team totals bounds
    team_totals_min: List[float] = field(default_factory=lambda: [70, 30, 15, 3, 2, 5])
    team_totals_max: List[float] = field(default_factory=lambda: [160, 70, 45, 20, 15, 28])
    
    # Dirichlet concentrations
    dirichlet_concentrations: List[float] = field(default_factory=lambda: [70.0, 90.0, 85.0, 95.0, 95.0, 90.0])
    
    # Correlation noise
    noise_intensity: float = 0.25
    
    # Clutch settings
    clutch_score_threshold: float = 118.0
    clutch_boost_multiplier: float = 1.3
    blowout_bench_multiplier: float = 1.5
    
    # FG2/FG3 adjustments
    fg2_pct_boost: float = 1.05
    
    # Three point frequency modifiers
    three_pt_freq_high_shooter: float = 1.5
    three_pt_freq_low_shooter: float = 0.8
    three_pt_clutch_modifier: float = 1.2
    three_pt_blowout_modifier: float = 0.8
    
    # FT rate modifiers
    ft_rate_fg3_modifier: float = 0.8
    ft_rate_fg2_modifier: float = 1.2
    
    # Overtime settings
    overtime_margin_threshold: float = 3.0
    overtime_points_mean: float = 5.0
    overtime_points_std: float = 2.5
    overtime_points_clip: tuple = field(default_factory=lambda: (0, 15))
    overtime_home_win_prob: float = 0.48
    
    # Home court advantage
    home_edge: float = 2.5
    
    # Stat correlation matrix (6x6 for PTS, REB, AST, STL, BLK, TOV)
    stat_correlation_matrix: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.35, 0.45, 0.15, 0.08, 0.20],
        [0.35, 1.0, 0.25, 0.12, 0.18, 0.10],
        [0.45, 0.25, 1.0, 0.18, 0.06, 0.28],
        [0.15, 0.12, 0.18, 1.0, 0.22, 0.35],
        [0.08, 0.18, 0.06, 0.22, 1.0, 0.12],
        [0.20, 0.10, 0.28, 0.35, 0.12, 1.0],
    ])


@dataclass
class FeatureConfigExt:
    """Extended feature engineering configuration."""
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    ewma_spans: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    trend_window_pairs: List[tuple] = field(default_factory=lambda: [(3, 10), (5, 20)])
    hot_streak_threshold: float = 1.15
    cold_streak_threshold: float = 0.85
    rest_days_max: int = 7
    matchup_history_window: int = 5
    bayesian_shrinkage_factor: int = 10
    league_percentile_window: int = 2000
    league_percentile_min_periods: int = 500
    target_encoding_smoothing: int = 20
    usage_windows: List[int] = field(default_factory=lambda: [5, 10])
    
    # Feature engineering params from base config
    use_matchup_features: bool = True
    use_fatigue_features: bool = True
    use_momentum_features: bool = True
    use_contextual_features: bool = True
    max_lag_days: int = 7


@dataclass
class Config:
    """Root configuration for the NBA prediction system."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    training_presets: Dict[str, Any] = field(default_factory=dict)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    features_ext: FeatureConfigExt = field(default_factory=FeatureConfigExt)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    simulation_params: SimulationParamsConfig = field(default_factory=SimulationParamsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    teams: TeamsConfig = field(default_factory=TeamsConfig)
    league_averages: LeagueAveragesConfig = field(default_factory=LeagueAveragesConfig)
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
        if 'training_presets' in data:
            config.training_presets = data['training_presets']
        if 'features' in data:
            config.features = FeatureConfig(**data['features'])
        if 'features_ext' in data:
            config.features_ext = FeatureConfigExt(**data['features_ext'])
        if 'simulation' in data:
            config.simulation = SimulationConfig(**data['simulation'])
        if 'simulation_params' in data:
            config.simulation_params = SimulationParamsConfig(**data['simulation_params'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        if 'http' in data:
            config.http = HTTPConfig(**data['http'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'api' in data:
            config.api = APIConfig(**data['api'])
        if 'teams' in data:
            config.teams = TeamsConfig(**data['teams'])
        if 'league_averages' in data:
            config.league_averages = LeagueAveragesConfig(**data['league_averages'])
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
