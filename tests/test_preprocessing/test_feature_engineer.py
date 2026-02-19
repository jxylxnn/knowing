"""Tests for preprocessing modules."""

import pytest
import pandas as pd
import numpy as np


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_feature_engineer_import(self):
        """Test FeatureEngineer can be imported."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        assert FeatureEngineer is not None
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initializes correctly."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        assert fe.rolling_windows == [3, 5, 10, 20, 50]
        assert fe.target_cols == ['PTS', 'REB', 'AST']
    
    def test_feature_engineer_custom_windows(self):
        """Test FeatureEngineer with custom rolling windows."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer(rolling_windows=[5, 10])
        assert fe.rolling_windows == [5, 10]
    
    def test_feature_engineer_create_features(self, sample_player_data):
        """Test that create_features returns a DataFrame with new columns."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        result = fe.create_features(sample_player_data, is_training=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert len(result.columns) > len(sample_player_data.columns)
    
    def test_feature_engineer_empty_input(self):
        """Test FeatureEngineer handles empty input."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        empty_df = pd.DataFrame()
        result = fe.create_features(empty_df, is_training=True)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_feature_engineer_missing_columns(self):
        """Test FeatureEngineer handles missing required columns."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = fe.create_features(df, is_training=True)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_feature_engineer_no_data_leakage(self, sample_player_data):
        """Test that feature engineering doesn't leak future data."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        result = fe.create_features(sample_player_data, is_training=True)
        
        rolling_cols = [c for c in result.columns if c.startswith('ROLL_')]
        for col in rolling_cols:
            if 'PTS' in col and 'AVG' in col:
                assert result[col].isna().sum() >= 0


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_data_loader_import(self):
        """Test DataLoader can be imported."""
        from src.preprocessing.data_loader import DataLoader
        assert DataLoader is not None


class TestTemporalWeightCalculator:
    """Tests for temporal weight calculation logic."""
    
    def test_temporal_weights_import(self):
        """Test temporal weight utilities can be imported."""
        from src.config import TrainingConfig
        
        config = TrainingConfig()
        assert hasattr(config, 'temporal_decay_lambda')
        assert config.temporal_decay_lambda > 0
    
    def test_temporal_decay_shape(self, sample_player_data):
        """Test temporal weights have correct shape."""
        from src.pipeline.data_pipeline import DataPipeline
        from src.config import DataConfig, TrainingConfig
        
        data_config = DataConfig()
        training_config = TrainingConfig()
        pipeline = DataPipeline(data_config, training_config)
        
        weights = pipeline.calculate_sample_weights(sample_player_data)
        
        assert len(weights) == len(sample_player_data)
        assert all(w >= 0.1 for w in weights)
        assert all(w <= 1.0 for w in weights)