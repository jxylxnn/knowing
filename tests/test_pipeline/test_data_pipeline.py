"""Tests for pipeline modules."""

import pytest
import pandas as pd
import numpy as np


class TestDataPipeline:
    """Tests for DataPipeline class."""
    
    def test_data_pipeline_import(self):
        """Test DataPipeline can be imported."""
        from src.pipeline.data_pipeline import DataPipeline
        assert DataPipeline is not None
    
    def test_data_pipeline_initialization(self):
        """Test DataPipeline initializes correctly."""
        from src.pipeline.data_pipeline import DataPipeline
        from src.config import DataConfig, TrainingConfig
        
        data_config = DataConfig()
        training_config = TrainingConfig()
        
        pipeline = DataPipeline(data_config, training_config)
        
        assert pipeline.data_config is not None
        assert pipeline.training_config is not None
        assert pipeline.feature_engineer is not None
    
    def test_data_pipeline_calculate_weights(self, sample_player_data):
        """Test temporal weight calculation."""
        from src.pipeline.data_pipeline import DataPipeline
        from src.config import DataConfig, TrainingConfig
        
        data_config = DataConfig()
        training_config = TrainingConfig()
        pipeline = DataPipeline(data_config, training_config)
        
        weights = pipeline.calculate_sample_weights(sample_player_data)
        
        assert len(weights) == len(sample_player_data)
        assert all(w >= 0.1 for w in weights)
        assert all(w <= 1.0 for w in weights)
    
    def test_data_pipeline_preprocess_targets(self, sample_player_data):
        """Test target preprocessing."""
        from src.pipeline.data_pipeline import DataPipeline
        from src.config import DataConfig, TrainingConfig
        
        data_config = DataConfig()
        training_config = TrainingConfig()
        pipeline = DataPipeline(data_config, training_config)
        
        result = pipeline.preprocess_targets(sample_player_data)
        
        assert isinstance(result, pd.DataFrame)
        for target in training_config.targets:
            if target in sample_player_data.columns:
                assert f'{target}_CLEAN' in result.columns
    
    def test_data_pipeline_select_features(self, sample_feature_data):
        """Test feature selection."""
        from src.pipeline.data_pipeline import DataPipeline
        from src.config import DataConfig, TrainingConfig
        
        data_config = DataConfig()
        training_config = TrainingConfig()
        pipeline = DataPipeline(data_config, training_config)
        
        features = pipeline.select_features(sample_feature_data)
        
        assert isinstance(features, list)
        assert pipeline.feature_cols is not None
        assert len(features) > 0


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""
    
    def test_training_pipeline_import(self):
        """Test TrainingPipeline can be imported."""
        from src.pipeline.training_pipeline import TrainingPipeline
        assert TrainingPipeline is not None
    
    def test_training_pipeline_initialization(self, sample_config):
        """Test TrainingPipeline initializes correctly."""
        from src.pipeline.training_pipeline import TrainingPipeline
        
        pipeline = TrainingPipeline(sample_config)
        
        assert pipeline.config is not None
        assert pipeline.training_config is not None
        assert pipeline.models == {}
    
    def test_training_pipeline_load_models(self, sample_config):
        """Test model loading (with empty models dir)."""
        from src.pipeline.training_pipeline import TrainingPipeline
        
        pipeline = TrainingPipeline(sample_config)
        pipeline.load_models()
        
        assert pipeline.feature_cols is None or isinstance(pipeline.feature_cols, list)


class TestPredictionService:
    """Tests for PredictionService class."""
    
    def test_prediction_service_import(self):
        """Test PredictionService can be imported."""
        from src.pipeline.prediction_service import PredictionService
        assert PredictionService is not None