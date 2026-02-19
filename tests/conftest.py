"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_player_data():
    """Generate sample player statistics data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2015-10-01', periods=100, freq='D')
    player_ids = np.random.randint(1000, 1050, n_samples)
    
    data = pd.DataFrame({
        'PLAYER_ID': player_ids,
        'PLAYER_NAME': [f'Player_{pid}' for pid in player_ids],
        'TEAM_ID': np.random.randint(1610612737, 1610612767, n_samples),
        'TEAM_ABBREVIATION': np.random.choice(['LAL', 'BOS', 'GSW', 'MIA'], n_samples),
        'TEAM_NAME': ['Team'] * n_samples,
        'GAME_ID': np.random.randint(10000000, 20000000, n_samples),
        'GAME_DATE': np.random.choice(dates, n_samples),
        'MATCHUP': ['vs. OPP'] * n_samples,
        'OPPONENT_ID': np.random.randint(1610612737, 1610612767, n_samples),
        'OPPONENT_ABBR': np.random.choice(['BOS', 'LAL', 'MIA', 'GSW'], n_samples),
        'WL': np.random.choice(['W', 'L'], n_samples),
        'MIN': np.random.uniform(10, 40, n_samples),
        'PTS': np.random.poisson(12, n_samples),
        'REB': np.random.poisson(5, n_samples),
        'AST': np.random.poisson(3, n_samples),
        'STL': np.random.poisson(1, n_samples),
        'BLK': np.random.poisson(0.5, n_samples),
        'TOV': np.random.poisson(2, n_samples),
        'FGA': np.random.poisson(10, n_samples),
        'FGM': np.random.poisson(5, n_samples),
        'FG3A': np.random.poisson(4, n_samples),
        'FG3M': np.random.poisson(1.5, n_samples),
        'FTA': np.random.poisson(3, n_samples),
        'FTM': np.random.poisson(2, n_samples),
        'SEASON_ID': ['22015'] * n_samples,
        'VIDEO_AVAILABLE': [1] * n_samples,
    })
    
    data['FGA_TEAM'] = data['FGA'] * 5
    data['FTA_TEAM'] = data['FTA'] * 5
    data['TOV_TEAM'] = data['TOV'] * 5
    data['PTS_TEAM'] = data['PTS'] * 5
    data['OREB_TEAM'] = np.random.poisson(10, n_samples)
    data['OPP_DREB'] = np.random.poisson(30, n_samples)
    
    return data


@pytest.fixture
def sample_games_data():
    """Generate sample games data for testing."""
    np.random.seed(42)
    n_games = 500
    
    dates = pd.date_range('2015-10-01', periods=50, freq='D')
    
    data = pd.DataFrame({
        'GAME_ID': np.arange(10000000, 10000000 + n_games),
        'GAME_DATE': np.random.choice(dates, n_games),
        'HOME_TEAM_ID': np.random.randint(1610612737, 1610612767, n_games),
        'AWAY_TEAM_ID': np.random.randint(1610612737, 1610612767, n_games),
        'HOME_PTS': np.random.poisson(105, n_games),
        'AWAY_PTS': np.random.poisson(102, n_games),
    })
    
    return data


@pytest.fixture
def sample_feature_data(sample_player_data):
    """Generate sample feature-engineered data for testing."""
    from src.preprocessing.feature_engineer import FeatureEngineer
    
    fe = FeatureEngineer()
    featured_df = fe.create_features(sample_player_data, is_training=True)
    return featured_df


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with sample files."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    
    return {
        'data_dir': data_dir,
        'models_dir': models_dir,
        'root': tmp_path
    }


@pytest.fixture
def sample_config(temp_data_dir):
    """Create a sample configuration for testing."""
    from src.config import Config, DataConfig, TrainingConfig, CatBoostConfig
    
    config = Config(
        data=DataConfig(
            data_dir=str(temp_data_dir['data_dir']),
            models_dir=str(temp_data_dir['models_dir']),
        ),
        training=TrainingConfig(
            test_split_date='2016-02-01',
            targets=['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'],
        ),
        catboost=CatBoostConfig(
            iterations=100,
            learning_rate=0.1,
            depth=4,
        ),
    )
    return config


@pytest.fixture
def mock_model_registry(temp_data_dir):
    """Create a mock model registry for testing."""
    from src.models.base import ModelRegistry
    
    registry = ModelRegistry(str(temp_data_dir['models_dir']))
    return registry


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )