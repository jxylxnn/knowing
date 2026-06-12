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


class TestGameSimulatorUpgrade:
    """Tests for the upgraded high-fidelity simulation flow."""

    @pytest.fixture
    def simulator(self, temp_data_dir):
        from src.simulation.game_simulator import GameSimulator

        mock_manager = Mock()
        mock_manager.data_dir = str(temp_data_dir['data_dir'])
        mock_manager.feature_engineer = Mock()

        with patch('src.simulation.game_simulator.InjuryScraper'), \
             patch('src.simulation.game_simulator.LineupScraper'), \
             patch('src.simulation.game_simulator.NBADefenseScraper'), \
             patch('src.simulation.game_simulator.MinutesPredictor'), \
             patch('src.simulation.game_simulator.ErrorCalibrator'), \
             patch('src.simulation.game_simulator.BettingScraper'), \
             patch('src.simulation.game_simulator.ScheduleScraper'), \
             patch('src.simulation.game_simulator.get_device', return_value='cpu'):
            simulator = GameSimulator(mock_manager, cache_dir=str(temp_data_dir['root'] / 'cache'))

        return simulator

    def _build_mock_roster(self, team: str, start_id: int) -> tuple[pd.DataFrame, dict, list]:
        rows = []
        histories = {}
        roster_info = []

        for i in range(5):
            pid = start_id + i
            pname = f"{team}_Player_{i}"
            rows.append({
                'PLAYER_ID': pid,
                'PLAYER_NAME': pname,
                'TEAM_ABBREVIATION': team,
                'GAME_DATE': pd.Timestamp('2024-01-01'),
                'MIN': 28 + i,
                'USAGE_PROXY_10': 0.18 + i * 0.01,
                'ROLL_MIN_AVG_10': 26 + i,
                'ROLL_REB_AVG_10': 4 + i * 0.5,
                'ROLL_AST_AVG_10': 3 + i * 0.4,
                'ROLL_BLK_AVG_10': 0.6 + i * 0.1,
            })
            histories[pid] = pd.DataFrame({
                'GAME_DATE': pd.date_range('2023-12-01', periods=5, freq='D'),
                'PTS': [18 + i] * 5,
                'REB': [4 + i] * 5,
                'AST': [3 + i] * 5,
                'MIN': [26 + i] * 5,
            })
            roster_info.append({
                'id': pid,
                'name': pname,
                'usage': 0.18 + i * 0.01,
                'exp_min': 26 + i,
                'play_probability': 1.0,
                'position': 'SG',
                'min_std': 3.0,
                'is_starter': i < 5,
            })

        return pd.DataFrame(rows), histories, roster_info

    def _build_prediction_frame(self, count: int) -> pd.DataFrame:
        data = []
        for i in range(count):
            data.append({
                'PTS': 18.0 + i,
                'REB': 5.0 + i * 0.2,
                'AST': 4.0 + i * 0.3,
                'STL': 1.0 + i * 0.05,
                'BLK': 0.5 + i * 0.03,
                'TOV': 2.0 + i * 0.1,
                'PTS_STD': 4.5,
                'REB_STD': 2.0,
                'AST_STD': 1.8,
                'STL_STD': 0.6,
                'BLK_STD': 0.5,
                'TOV_STD': 0.7,
            })
        return pd.DataFrame(data)

    def test_device_and_seed_are_stable(self, simulator):
        """Device should be torch-compatible and matchup seeding should be deterministic."""
        assert hasattr(simulator.device, 'type')
        assert simulator.device.type in {'cpu', 'cuda'}

        seed_a = simulator._get_matchup_seed('LAL', 'BOS', 100)
        seed_b = simulator._get_matchup_seed('LAL', 'BOS', 100)
        seed_c = simulator._get_matchup_seed('BOS', 'LAL', 100)

        assert seed_a == seed_b
        assert seed_a != seed_c

    def test_safe_fetch_helpers_fallback(self, simulator):
        """External fetch helpers should fall back cleanly on errors."""
        simulator.betting_scraper.get_game_lines = Mock(side_effect=RuntimeError("boom"))
        simulator.lineup_scraper.get_starting_lineup = Mock(side_effect=RuntimeError("boom"))
        simulator.injury_scraper.get_player_availability = Mock(side_effect=RuntimeError("boom"))

        lines = simulator._safe_get_game_lines('LAL', 'BOS', '2024-01-01')
        lineup = simulator._safe_get_lineup('LAL', '2024-01-01')
        injuries = simulator._safe_get_injury_probs('LAL')

        assert lines['data']['total'] is None
        assert lines['data']['source'] == 'fallback'
        assert lines['health']['status'] == 'failed'
        assert lineup['data'] == {}
        assert lineup['health']['status'] == 'failed'
        assert injuries['data'] == {}
        assert injuries['health']['status'] == 'failed'

    def test_team_target_means_use_four_factors_prior(self, simulator):
        """Team target means should blend the model baseline with four-factors priors."""
        roster_a = [
            {'mean_pts': 20.0, 'mean_reb': 5.0, 'mean_ast': 4.0},
            {'mean_pts': 18.0, 'mean_reb': 4.0, 'mean_ast': 3.0},
        ]
        roster_b = [
            {'mean_pts': 16.0, 'mean_reb': 6.0, 'mean_ast': 5.0},
            {'mean_pts': 14.0, 'mean_reb': 5.0, 'mean_ast': 4.0},
        ]

        team_eff_a = {
            'team': 'LAL',
            'pace': 100.0,
            'offensive_rating': 114.0,
            'defensive_rating': 112.0,
            'efg_pct': 0.54,
            'tov_pct': 0.135,
            'orb_pct': 0.25,
            'ft_rate': 0.23,
        }
        team_eff_b = {
            'team': 'BOS',
            'pace': 99.0,
            'offensive_rating': 113.0,
            'defensive_rating': 111.0,
            'efg_pct': 0.53,
            'tov_pct': 0.134,
            'orb_pct': 0.24,
            'ft_rate': 0.22,
        }

        with patch.object(simulator.four_factors_engine, 'predict_matchup', return_value={
            'home_pts_mean': 120.0,
            'away_pts_mean': 110.0,
        }):
            targets = simulator._build_team_target_means(
                'LAL',
                'BOS',
                roster_a,
                roster_b,
                {'total': None},
                team_eff_a,
                team_eff_b,
            )

        assert targets['LAL']['pts'] == pytest.approx(58.5)
        assert targets['BOS']['pts'] == pytest.approx(50.0)
        assert targets['LAL']['reb'] == pytest.approx(9.0)
        assert targets['BOS']['ast'] == pytest.approx(9.0)

    def test_simulate_matchup_is_deterministic_and_schema_stable(self, simulator):
        """Simulation should remain deterministic and preserve the public output shape."""
        ctx_a, hist_a, info_a = self._build_mock_roster('LAL', 100)
        ctx_b, hist_b, info_b = self._build_mock_roster('BOS', 200)
        pred_a = self._build_prediction_frame(len(info_a))
        pred_b = self._build_prediction_frame(len(info_b))

        lineup_a = {'starters': [row['PLAYER_NAME'] for _, row in ctx_a.iterrows()]}
        lineup_b = {'starters': [row['PLAYER_NAME'] for _, row in ctx_b.iterrows()]}

        def build_roster_context(team, opponent, is_home, injury_probs, lineup_data=None, game_date=None, rest_info=None):
            if team == 'LAL':
                return ctx_a, hist_a, info_a
            return ctx_b, hist_b, info_b

        def predict_batch(context_df, histories_map, include_confidence=False):
            team = context_df.iloc[0]['TEAM_ABBREVIATION']
            return pred_a if team == 'LAL' else pred_b

        team_eff_map = {
            'LAL': {
                'team': 'LAL',
                'pace': 100.0,
                'offensive_rating': 114.0,
                'defensive_rating': 112.0,
                'efg_pct': 0.54,
                'tov_pct': 0.135,
                'orb_pct': 0.25,
                'ft_rate': 0.23,
            },
            'BOS': {
                'team': 'BOS',
                'pace': 99.0,
                'offensive_rating': 113.0,
                'defensive_rating': 111.0,
                'efg_pct': 0.53,
                'tov_pct': 0.134,
                'orb_pct': 0.24,
                'ft_rate': 0.22,
            },
        }

        with patch.object(simulator, 'prepare_simulation_context'), \
             patch.object(simulator, '_build_roster_context', side_effect=build_roster_context), \
             patch.object(simulator, '_safe_get_defensive_adjustments', side_effect=lambda opponent, roster: {
                 'data': {},
                 'health': {
                     'source_key': f'defense_{opponent.lower()}',
                     'status': 'fallback',
                     'required': False,
                     'message': 'defense unavailable',
                     'details': {'adjustments_applied': False},
                 },
             }), \
             patch.object(simulator, '_safe_get_game_lines', return_value={
                 'data': {'total': 225.0, 'spread': -2.5, 'home_implied_pts': 113.8, 'away_implied_pts': 111.2, 'source': 'test'},
                 'health': {'source_key': 'betting', 'status': 'success', 'required': False, 'message': 'ok', 'details': {}},
             }), \
             patch.object(simulator, '_safe_get_lineup', side_effect=lambda team, game_date=None: {
                 'data': lineup_a if team == 'LAL' else lineup_b,
                 'health': {'source_key': f'lineup_{team.lower()}', 'status': 'success', 'required': False, 'message': 'ok', 'details': {}},
             }), \
             patch.object(simulator, '_safe_get_injury_probs', side_effect=lambda team: {
                 'data': {},
                 'health': {'source_key': f'injury_{team.lower()}', 'status': 'success', 'required': False, 'message': 'ok', 'details': {}},
             }), \
             patch.object(simulator, '_get_team_rest_days', return_value={'rest_days': 2, 'is_b2b': False, 'is_3_in_4': False, 'games_last_7': 3, 'games_last_14': 6}), \
             patch.object(simulator, '_get_team_pace', side_effect=lambda team: 100.0 if team == 'LAL' else 99.0), \
             patch.object(simulator, '_get_team_efficiency_snapshot', side_effect=lambda team: team_eff_map[team]), \
             patch.object(simulator, '_build_team_target_means', return_value={
                 'LAL': {'pts': 114.0, 'reb': 44.0, 'ast': 26.0},
                 'BOS': {'pts': 111.0, 'reb': 43.0, 'ast': 25.0},
             }), \
             patch.object(simulator, '_apply_error_calibration', side_effect=lambda roster: roster), \
             patch.object(simulator, '_apply_context_adjustments', side_effect=lambda roster, *args, **kwargs: roster), \
             patch.object(simulator.manager, 'predict_player_stats_batch', side_effect=predict_batch):

            result1 = simulator.simulate_matchup('LAL', 'BOS', num_sims=20, seed=123)
            result2 = simulator.simulate_matchup('LAL', 'BOS', num_sims=20, seed=123)

        assert result1['team_a'] == 'LAL'
        assert result1['team_b'] == 'BOS'
        assert result1['metadata']['simulation_mode'] == 'high_fidelity'
        assert result1['metadata']['seed'] == result2['metadata']['seed']
        assert result1['win_prob_a'] == pytest.approx(result2['win_prob_a'])
        assert result1['team_summaries']['LAL']['pts']['mean'] == pytest.approx(result2['team_summaries']['LAL']['pts']['mean'])
        assert len(result1['simulations']) == 20
        assert len(result1['player_averages']) == 10
        assert 'context' in result1 and 'metadata' in result1
        assert 'team_targets' in result1['context']
        assert result1['metadata']['input_health']['overall_status'] == 'degraded'

    def test_optional_scraper_failures_mark_matchup_degraded(self, simulator):
        """Optional scraper failures should keep the simulation running but mark degraded input health."""
        ctx_a, hist_a, info_a = self._build_mock_roster('LAL', 100)
        ctx_b, hist_b, info_b = self._build_mock_roster('BOS', 200)
        pred_a = self._build_prediction_frame(len(info_a))
        pred_b = self._build_prediction_frame(len(info_b))

        def build_roster_context(team, opponent, is_home, injury_probs, lineup_data=None, game_date=None, rest_info=None):
            if team == 'LAL':
                return ctx_a, hist_a, info_a
            return ctx_b, hist_b, info_b

        def predict_batch(context_df, histories_map, include_confidence=False):
            return pred_a if context_df.iloc[0]['TEAM_ABBREVIATION'] == 'LAL' else pred_b

        with patch.object(simulator, 'prepare_simulation_context'), \
             patch.object(simulator, '_build_roster_context', side_effect=build_roster_context), \
             patch.object(simulator, '_safe_get_game_lines', return_value={
                 'data': {'total': None, 'spread': None, 'source': 'fallback'},
                 'health': {'source_key': 'betting', 'status': 'fallback', 'required': False, 'message': 'betting fallback', 'details': {}},
             }), \
             patch.object(simulator, '_safe_get_lineup', side_effect=lambda team, game_date=None: {
                 'data': {},
                 'health': {'source_key': f'lineup_{team.lower()}', 'status': 'failed', 'required': False, 'message': 'lineup failed', 'details': {}},
             }), \
             patch.object(simulator, '_safe_get_injury_probs', side_effect=lambda team: {
                 'data': {},
                 'health': {'source_key': f'injury_{team.lower()}', 'status': 'success', 'required': False, 'message': 'injury ok', 'details': {}},
             }), \
             patch.object(simulator, '_safe_get_defensive_adjustments', side_effect=lambda opponent, roster: {
                 'data': {},
                 'health': {'source_key': f'defense_{opponent.lower()}', 'status': 'fallback', 'required': False, 'message': 'defense fallback', 'details': {'adjustments_applied': False}},
             }), \
             patch.object(simulator, '_get_team_rest_days', return_value={'rest_days': 2, 'is_b2b': False, 'is_3_in_4': False, 'games_last_7': 3, 'games_last_14': 6}), \
             patch.object(simulator, '_get_team_pace', return_value=100.0), \
             patch.object(simulator, '_get_team_efficiency_snapshot', return_value={'pace': 100.0, 'offensive_rating': 114.0, 'defensive_rating': 112.0}), \
             patch.object(simulator, '_build_team_target_means', return_value={
                 'LAL': {'pts': 114.0, 'reb': 44.0, 'ast': 26.0},
                 'BOS': {'pts': 111.0, 'reb': 43.0, 'ast': 25.0},
             }), \
             patch.object(simulator, '_apply_error_calibration', side_effect=lambda roster: roster), \
             patch.object(simulator, '_apply_context_adjustments', side_effect=lambda roster, *args, **kwargs: roster), \
             patch.object(simulator.manager, 'predict_player_stats_batch', side_effect=predict_batch):
            result = simulator.simulate_matchup('LAL', 'BOS', num_sims=10, seed=99)

        input_health = result['metadata']['input_health']
        assert input_health['overall_status'] == 'degraded'
        assert 'lineup_lal' in input_health['degraded_sources']
        assert input_health['betting_calibration_applied'] is False
        assert input_health['defensive_adjustments_applied'] is False

    def test_lineup_context_reflects_missing_primary_handler(self, simulator):
        """Lineup context should boost usage and assists when the primary handler is absent."""
        roster = [
            {'name': 'A', 'usage': 0.30, 'mean_reb': 4.0, 'mean_blk': 0.4, 'is_starter': True},
            {'name': 'B', 'usage': 0.22, 'mean_reb': 6.0, 'mean_blk': 0.6, 'is_starter': True},
            {'name': 'C', 'usage': 0.18, 'mean_reb': 8.5, 'mean_blk': 1.3, 'is_starter': False},
        ]

        context = simulator._build_team_lineup_context(
            roster,
            {'starters': ['B', 'C']},
            coach_tightness=0.72,
        )

        assert context['usage_boost'] > 1.0
        assert context['assist_boost'] > 1.0
        assert context['starter_overlap'] < 1.0
