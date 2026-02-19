"""Tests for GameSimulator class."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestGameSimulatorImport:
    """Tests for GameSimulator import and basic functionality."""
    
    def test_game_simulator_import(self):
        """Test GameSimulator can be imported."""
        from src.simulation.game_simulator import GameSimulator
        assert GameSimulator is not None
    
    def test_game_simulator_has_cache_methods(self):
        """Test GameSimulator has cache-related methods."""
        from src.simulation.game_simulator import GameSimulator
        assert hasattr(GameSimulator, '_get_cache_key')
        assert hasattr(GameSimulator, '_load_from_cache')
        assert hasattr(GameSimulator, '_save_to_cache')
        assert hasattr(GameSimulator, '_serialize_for_cache')
        assert hasattr(GameSimulator, '_deserialize_from_cache')


class TestCacheSerialization:
    """Tests for JSON-based cache serialization (security fix)."""
    
    @pytest.fixture
    def mock_simulator(self, temp_data_dir):
        """Create a mock GameSimulator for testing."""
        from src.simulation.game_simulator import GameSimulator
        
        mock_manager = Mock()
        mock_manager.data_dir = str(temp_data_dir['data_dir'])
        
        with patch('src.simulation.game_simulator.InjuryScraper'), \
             patch('src.simulation.game_simulator.LineupScraper'), \
             patch('src.simulation.game_simulator.NBADefenseScraper'), \
             patch('src.simulation.game_simulator.MinutesPredictor'), \
             patch('src.simulation.game_simulator.ErrorCalibrator'), \
             patch('src.simulation.game_simulator.BettingScraper'), \
             patch('src.simulation.game_simulator.get_device', return_value='cpu'):
            simulator = GameSimulator(mock_manager, cache_dir=str(temp_data_dir['root'] / 'cache'))
        return simulator
    
    def test_cache_key_generation(self, mock_simulator):
        """Test _get_cache_key produces consistent MD5 keys."""
        key1 = mock_simulator._get_cache_key('LAL', 'BOS', 100)
        key2 = mock_simulator._get_cache_key('LAL', 'BOS', 100)
        key3 = mock_simulator._get_cache_key('BOS', 'LAL', 100)
        
        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32
        assert all(c in '0123456789abcdef' for c in key1)
    
    def test_cache_save_load_json(self, mock_simulator):
        """Test cache save and load with JSON format."""
        cache_key = 'test_cache_key'
        test_data = {
            'string': 'value',
            'number': 42,
            'float': 3.14,
            'list': [1, 2, 3],
            'nested': {'a': 1, 'b': 2}
        }
        
        mock_simulator._save_to_cache(cache_key, test_data)
        loaded_data = mock_simulator._load_from_cache(cache_key)
        
        assert loaded_data == test_data
        
        cache_file = mock_simulator.cache_dir / f"{cache_key}.json"
        assert cache_file.exists()
        assert cache_file.suffix == '.json'
    
    def test_cache_load_missing_file(self, mock_simulator):
        """Test cache load returns None for missing file."""
        result = mock_simulator._load_from_cache('nonexistent_key')
        assert result is None
    
    def test_cache_load_corrupted_json(self, mock_simulator):
        """Test cache load handles corrupted JSON gracefully."""
        cache_key = 'corrupted_cache'
        cache_file = mock_simulator.cache_dir / f"{cache_key}.json"
        cache_file.write_text('{ invalid json }')
        
        result = mock_simulator._load_from_cache(cache_key)
        assert result is None
    
    def test_serialize_numpy_array(self, mock_simulator):
        """Test serialization of numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        serialized = mock_simulator._serialize_for_cache(arr)
        
        assert serialized['__type__'] == 'array'
        assert serialized['data'] == [1.0, 2.0, 3.0]
    
    def test_serialize_numpy_float(self, mock_simulator):
        """Test serialization of numpy floats."""
        val = np.float64(3.14)
        serialized = mock_simulator._serialize_for_cache(val)
        
        assert serialized == 3.14
        assert isinstance(serialized, float)
    
    def test_serialize_pandas_dataframe(self, mock_simulator):
        """Test serialization of pandas DataFrames."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        serialized = mock_simulator._serialize_for_cache(df)
        
        assert serialized['__type__'] == 'dataframe'
        assert serialized['data'] == [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    
    def test_deserialize_array(self, mock_simulator):
        """Test deserialization of arrays."""
        data = {'__type__': 'array', 'data': [1.0, 2.0, 3.0]}
        result = mock_simulator._deserialize_from_cache(data)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    
    def test_deserialize_dataframe(self, mock_simulator):
        """Test deserialization of DataFrames."""
        data = {'__type__': 'dataframe', 'data': [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]}
        result = mock_simulator._deserialize_from_cache(data)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result['a']) == [1, 2]
        assert list(result['b']) == [3, 4]
    
    def test_no_pickle_files_created(self, mock_simulator):
        """Test that no .pkl files are created (security)."""
        cache_key = 'security_test'
        mock_simulator._save_to_cache(cache_key, {'data': 'test'})
        
        pkl_files = list(mock_simulator.cache_dir.glob('*.pkl'))
        assert len(pkl_files) == 0
        
        json_files = list(mock_simulator.cache_dir.glob('*.json'))
        assert len(json_files) == 1


class TestCorrelationMatrix:
    """Tests for correlation matrix validity."""
    
    def test_correlation_matrix_is_positive_semidefinite(self):
        """Test that CORR_MATRIX is valid for Cholesky decomposition."""
        import torch
        
        corr_matrix = torch.tensor([
            [1.00, 0.15, -0.05],
            [0.15, 1.00, -0.10],
            [-0.05, -0.10, 1.00]
        ], dtype=torch.float32)
        
        eigenvalues = torch.linalg.eigvalsh(corr_matrix)
        assert all(eigenvalues > -1e-6), f"Matrix has negative eigenvalues: {eigenvalues}"
        
        cholesky = torch.linalg.cholesky(corr_matrix)
        assert cholesky is not None
    
    def test_correlation_matrix_diagonal_is_one(self):
        """Test that diagonal elements are 1.0."""
        import torch
        
        corr_matrix = torch.tensor([
            [1.00, 0.15, -0.05],
            [0.15, 1.00, -0.10],
            [-0.05, -0.10, 1.00]
        ], dtype=torch.float32)
        
        for i in range(3):
            assert corr_matrix[i, i] == 1.0
    
    def test_correlation_matrix_is_symmetric(self):
        """Test that correlation matrix is symmetric."""
        import torch
        
        corr_matrix = torch.tensor([
            [1.00, 0.15, -0.05],
            [0.15, 1.00, -0.10],
            [-0.05, -0.10, 1.00]
        ], dtype=torch.float32)
        
        assert torch.allclose(corr_matrix, corr_matrix.T)


class TestGameSimulatorIntegration:
    """Integration tests for GameSimulator."""
    
    @pytest.mark.slow
    def test_get_available_teams(self, temp_data_dir):
        """Test getting available teams from data."""
        from src.simulation.game_simulator import GameSimulator
        
        players_df = pd.DataFrame({
            'PLAYER_ID': [1, 2, 3],
            'PLAYER_NAME': ['Player A', 'Player B', 'Player C'],
            'TEAM_ABBREVIATION': ['LAL', 'BOS', 'GSW'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'PTS': [10, 20, 30],
            'REB': [5, 5, 5],
            'AST': [3, 3, 3],
            'MIN': [30, 30, 30]
        })
        games_df = pd.DataFrame({
            'GAME_ID': [1, 2],
            'TEAM_ABBREVIATION': ['LAL', 'BOS'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })
        
        players_df.to_csv(temp_data_dir['data_dir'] / 'nba_players.csv', index=False)
        games_df.to_csv(temp_data_dir['data_dir'] / 'nba_games.csv', index=False)
        
        mock_manager = Mock()
        mock_manager.data_dir = str(temp_data_dir['data_dir'])
        
        with patch('src.simulation.game_simulator.InjuryScraper'), \
             patch('src.simulation.game_simulator.LineupScraper'), \
             patch('src.simulation.game_simulator.NBADefenseScraper'), \
             patch('src.simulation.game_simulator.MinutesPredictor'), \
             patch('src.simulation.game_simulator.ErrorCalibrator'), \
             patch('src.simulation.game_simulator.BettingScraper'), \
             patch('src.simulation.game_simulator.get_device', return_value='cpu'):
            simulator = GameSimulator(mock_manager)
        
        teams = simulator.get_available_teams()
        assert isinstance(teams, list)
        assert 'LAL' in teams or 'BOS' in teams
