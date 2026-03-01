"""Test models module."""

import pytest


class TestModelManager:
    """Tests for ModelManager class."""
    
    def test_model_manager_import(self):
        """Test ModelManager can be imported."""
        from src.models.model_manager import ModelManager
        assert ModelManager is not None
    
    def test_model_manager_initialization(self, temp_data_dir):
        """Test ModelManager initializes with correct directories."""
        from src.models.model_manager import ModelManager
        
        manager = ModelManager(
            data_dir=str(temp_data_dir['data_dir']),
            models_dir=str(temp_data_dir['models_dir'])
        )
        
        assert manager.data_dir == str(temp_data_dir['data_dir'])
        assert manager.models_dir == str(temp_data_dir['models_dir'])
        assert manager.targets == ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    
    def test_model_manager_invalid_data_dir(self):
        """Test ModelManager raises error for invalid data_dir."""
        from src.models.model_manager import ModelManager
        
        with pytest.raises(ValueError):
            ModelManager(data_dir='', models_dir='models')
        
        with pytest.raises(ValueError):
            ModelManager(data_dir=None, models_dir='models')
    
    def test_model_manager_prepare_data_missing_dir(self, temp_data_dir):
        """Test ModelManager raises error when data directory is empty."""
        from src.models.model_manager import ModelManager
        
        manager = ModelManager(
            data_dir=str(temp_data_dir['data_dir']),
            models_dir=str(temp_data_dir['models_dir'])
        )
        
        with pytest.raises(ValueError, match="Players file not found"):
            manager.prepare_data()


class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    def test_registry_import(self):
        """Test ModelRegistry can be imported."""
        from src.models.base import ModelRegistry
        assert ModelRegistry is not None
    
    def test_registry_initialization(self, temp_data_dir):
        """Test ModelRegistry initializes correctly."""
        from pathlib import Path
        from src.models.base import ModelRegistry
        
        registry = ModelRegistry(str(temp_data_dir['models_dir']))
        assert registry.models_dir == Path(temp_data_dir['models_dir'])
    
    def test_registry_list_models_empty(self, mock_model_registry):
        """Test ModelRegistry list_models returns empty when no models."""
        models = mock_model_registry.list_models()
        assert isinstance(models, list)
        assert len(models) == 0


class TestFallbackPredictor:
    """Tests for fallback prediction logic."""
    
    def test_fallback_prediction_import(self):
        """Test that fallback prediction utilities can be accessed."""
        from src.models.model_manager import ModelManager
        
        manager = ModelManager.__new__(ModelManager)
        manager.targets = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        
        import pandas as pd
        df = pd.DataFrame({'PTS': [10], 'ROLL_PTS_AVG_10': [12]})
        
        result = manager._get_fallback_value(df, 'PTS')
        assert result == 12
    
    def test_fallback_prediction_league_avg(self):
        """Test fallback prediction uses league average when no history."""
        from src.models.model_manager import ModelManager
        
        manager = ModelManager.__new__(ModelManager)
        manager.targets = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        
        import pandas as pd
        df = pd.DataFrame()
        
        result = manager._get_fallback_value(df, 'PTS')
        assert result == 10.0
        
        result = manager._get_fallback_value(df, 'REB')
        assert result == 4.5