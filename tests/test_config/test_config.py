"""Test configuration module."""

import pytest
from pathlib import Path


class TestConfig:
    """Tests for Config dataclass."""
    
    def test_config_import(self):
        """Test that Config can be imported."""
        from src.config import Config
        assert Config is not None
    
    def test_config_default_values(self):
        """Test Config creates with default values."""
        from src.config import Config
        
        config = Config()
        assert config.data is not None
        assert config.training is not None
        assert config.catboost is not None
    
    def test_config_custom_paths(self, temp_data_dir):
        """Test Config with custom paths."""
        from src.config import Config, DataConfig
        
        config = Config(
            data=DataConfig(
                data_dir=str(temp_data_dir['data_dir']),
                models_dir=str(temp_data_dir['models_dir']),
            )
        )
        
        assert str(config.data.data_dir) == str(temp_data_dir['data_dir'])
        assert str(config.data.models_dir) == str(temp_data_dir['models_dir'])

    def test_model_v2_config_loads_through_canonical_loader(self):
        from src.config import Config

        config = Config.from_yaml(Path("config/model_v2.yaml"))

        assert config.architecture == "v2"
        assert config.data.strict_core is True
        assert "rosters" in config.data.required_sources
        assert config.features.enabled_families
        assert config.simulation.regulation_team_minutes == 240


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        from src.config import TrainingConfig
        
        config = TrainingConfig()
        assert config.targets == ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        assert config.test_split_date is not None
        assert config.temporal_decay_lambda > 0
    
    def test_training_config_custom_targets(self):
        """Test TrainingConfig with custom targets."""
        from src.config import TrainingConfig
        
        config = TrainingConfig(targets=['PTS', 'REB'])
        assert config.targets == ['PTS', 'REB']


class TestCatBoostConfig:
    """Tests for CatBoostConfig."""
    
    def test_catboost_config_defaults(self):
        """Test CatBoostConfig default values."""
        from src.config import CatBoostConfig
        
        config = CatBoostConfig()
        assert config.iterations > 0
        assert 0 < config.learning_rate < 1
        assert config.depth > 0
