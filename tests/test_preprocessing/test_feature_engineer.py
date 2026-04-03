"""Tests for preprocessing modules."""

import pytest
import pandas as pd
import numpy as np
import warnings


def _build_single_player_history(rows: int = 8, start_date: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=rows, freq="D")
    return pd.DataFrame(
        {
            'PLAYER_ID': [101] * rows,
            'PLAYER_NAME': ['Tester'] * rows,
            'TEAM_ID': [1] * rows,
            'TEAM_ABBREVIATION': ['TST'] * rows,
            'TEAM_NAME': ['Testers'] * rows,
            'GAME_ID': np.arange(1000, 1000 + rows),
            'GAME_DATE': dates,
            'MATCHUP': ['vs. OPP'] * rows,
            'OPPONENT_ID': [2] * rows,
            'OPPONENT_ABBR': ['OPP'] * rows,
            'WL': ['W'] * rows,
            'MIN': np.linspace(18, 30, rows),
            'PTS': np.arange(10, 10 + rows),
            'REB': np.arange(4, 4 + rows),
            'AST': np.arange(3, 3 + rows),
            'STL': np.linspace(0.5, 1.2, rows),
            'BLK': np.linspace(0.2, 0.8, rows),
            'TOV': np.linspace(1.0, 2.5, rows),
            'FGA': np.arange(8, 8 + rows),
            'FGM': np.arange(4, 4 + rows),
            'FG3A': np.arange(3, 3 + rows),
            'FG3M': np.arange(1, 1 + rows),
            'FTA': np.arange(2, 2 + rows),
            'FTM': np.arange(1, 1 + rows),
            'SEASON_ID': ['22024'] * rows,
            'VIDEO_AVAILABLE': [1] * rows,
        }
    )


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

    def test_feature_engineer_accepts_disable_groups(self):
        """FeatureEngineer should accept disable_groups directly."""
        from src.preprocessing.feature_engineer import FeatureEngineer

        fe = FeatureEngineer(disable_groups=['matchup', 'opponent_strength'])

        assert fe.disable_groups == {'matchup', 'opponent_strength'}

    def test_build_feature_engineer_backfills_disable_groups_for_legacy_ctor(self, monkeypatch):
        """The training helper should adapt to older constructors without disable_groups."""
        import src.preprocessing.feature_engineer as feature_engineer_module

        class LegacyFeatureEngineer:
            def __init__(
                self,
                rolling_windows=None,
                use_gpu=None,
                enable_groups=None,
                disable_columns=None,
                max_missing_rate=0.35,
                max_imputed_rate=0.40,
            ):
                self.rolling_windows = rolling_windows or [3, 5, 10, 20, 50]
                self.use_gpu = use_gpu
                self.enable_groups = set(enable_groups) if enable_groups else None
                self.disable_columns = set(disable_columns) if disable_columns else set()
                self.max_missing_rate = max_missing_rate
                self.max_imputed_rate = max_imputed_rate
                self.disable_groups = set()

        monkeypatch.setattr(feature_engineer_module, "FeatureEngineer", LegacyFeatureEngineer)

        fe = feature_engineer_module.build_feature_engineer(
            disable_groups=['matchup', 'opponent_strength'],
            disable_columns=['PACE_ADJ_USAGE'],
        )

        assert fe.disable_groups == {'matchup', 'opponent_strength'}
        assert fe.disable_columns == {'PACE_ADJ_USAGE'}

    def test_feature_engineer_create_features(self, sample_player_data):
        """Test that create_features returns a DataFrame with new columns."""
        from src.preprocessing.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        result = fe.create_features(sample_player_data, is_training=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert len(result.columns) > len(sample_player_data.columns)

    def test_feature_engineer_avoids_fragmentation_warnings(self, sample_player_data):
        """Batch column assembly should not emit pandas fragmentation warnings."""
        from src.preprocessing.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.PerformanceWarning)
            result = fe.create_features(sample_player_data, is_training=True)

        assert isinstance(result, pd.DataFrame)
        assert not any(issubclass(w.category, pd.errors.PerformanceWarning) for w in caught)

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
        """Future rows should not affect earlier rolling features."""
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        df = _build_single_player_history()
        first = fe.create_features(df, is_training=True)
        df_future = df.copy()
        df_future.loc[df_future.index[-1], 'PTS'] = 999
        second = fe.create_features(df_future, is_training=True)

        compare_cols = [c for c in first.columns if c.startswith('ROLL_PTS_AVG_')]
        assert compare_cols
        early_rows = first.index[:-1]
        for col in compare_cols:
            pd.testing.assert_series_equal(
                first.loc[early_rows, col].reset_index(drop=True),
                second.loc[early_rows, col].reset_index(drop=True),
                check_names=False,
            )

    def test_feature_engineer_cold_start_preserves_rows(self):
        """Players without enough history should still produce usable features."""
        from src.preprocessing.feature_engineer import FeatureEngineer

        df = _build_single_player_history(rows=3)
        fe = FeatureEngineer()
        result = fe.create_features(df, is_training=True)

        assert len(result) == len(df)
        assert 'ROLL_PTS_AVG_5' in result.columns
        assert result['ROLL_PTS_AVG_5'].isna().sum() == 0
        assert 'ROLL_PTS_COLD_START_5' in result.columns
        assert result['ROLL_PTS_COLD_START_5'].iloc[0] == 1

    def test_feature_engineer_chunked_matches_full(self, sample_player_data):
        """Chunked feature generation should match the full in-memory path."""
        from src.preprocessing.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        full = fe.create_features(sample_player_data, is_training=True).sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        chunked = fe.create_features_chunked(sample_player_data, chunk_size=200).sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        assert list(full.columns) == list(chunked.columns)
        common_cols = [c for c in full.columns if c in chunked.columns and pd.api.types.is_numeric_dtype(full[c])]
        for col in common_cols[:20]:
            pd.testing.assert_series_equal(
                pd.to_numeric(full[col], errors='coerce').fillna(0).reset_index(drop=True),
                pd.to_numeric(chunked[col], errors='coerce').fillna(0).reset_index(drop=True),
                check_names=False,
            )

    def test_feature_selector_schema_alignment(self, sample_feature_data):
        """Train and inference should align to the same feature schema."""
        from src.utils.prediction_utils import FeatureSelector

        selector = FeatureSelector(targets=['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'])
        schema = selector.fit(sample_feature_data)
        aligned = selector.transform(sample_feature_data.sample(frac=1.0, random_state=7), schema, strict=True)

        assert list(aligned.columns) == list(schema.feature_cols)
        assert selector.feature_schema.schema_hash == schema.schema_hash


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
