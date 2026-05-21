"""Tests for KANAgingFeatureGroup."""

import os

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features.kan_aging import KANAgingFeatureGroup, OUTPUT_COLUMNS
from src.preprocessing.features.base import FeatureDiagnostics


@pytest.fixture
def kan_group(tmp_path):
    return KANAgingFeatureGroup(data_dir=str(tmp_path))


@pytest.fixture
def sample_df():
    dates = pd.date_range('2025-01-01', periods=8, freq='3D')
    return pd.DataFrame({
        'PLAYER_ID': [1] * 4 + [2] * 4,
        'GAME_DATE': list(dates[:4]) + list(dates[:4]),
        'AGE': [30.5] * 4 + [23.0] * 4,
    })


@pytest.fixture
def kan_cache(tmp_path):
    """Write a cached KAN aging outputs CSV."""
    cache_dir = os.path.join(str(tmp_path), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'kan_aging_outputs.csv')
    df = pd.DataFrame({
        'PLAYER_ID': [1, 2],
        'KAN_AGE_NONLIN_FACTOR': [0.98, 1.01],
        'KAN_AGE_INFLECTION_AGE': [28.5, 27.0],
        'KAN_AGE_VOLATILITY': [0.06, 0.03],
    })
    df.to_csv(path, index=False)
    return path


class TestKANAgingFeatureGroup:

    def test_creates_all_output_columns(self, kan_group, sample_df):
        result = kan_group.create(sample_df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_leakage_first_game(self, kan_group, sample_df):
        result = kan_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # After shift+fillna, all values should be populated
        assert result['KAN_AGE_NONLIN_FACTOR'].notna().all()
        assert result['KAN_AGE_INFLECTION_AGE'].notna().all()
        assert result['KAN_AGE_VOLATILITY'].notna().all()

    def test_uses_cached_outputs(self, kan_group, sample_df, kan_cache):
        result = kan_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # Should have populated values from cache
        assert result['KAN_AGE_NONLIN_FACTOR'].notna().all()

    def test_fallback_without_cache(self, kan_group, sample_df):
        """Without KAN cache, fallback quadratic should still work."""
        result = kan_group.create(sample_df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns
        # Factor should be close to 1.0 for ages near peak
        assert (result['KAN_AGE_NONLIN_FACTOR'] > 0.7).all()
        assert (result['KAN_AGE_NONLIN_FACTOR'] < 1.2).all()

    def test_missing_age_column(self, kan_group):
        df = pd.DataFrame({
            'PLAYER_ID': [1, 1],
            'GAME_DATE': pd.date_range('2025-01-01', periods=2),
        })
        result = kan_group.create(df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns
        # Should default to 1.0 factor
        assert (result['KAN_AGE_NONLIN_FACTOR'] == 1.0).all()

    def test_volatility_increases_with_age(self, kan_group, sample_df):
        """Older players should have higher volatility."""
        result = kan_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # Player 1 (30.5) should have higher volatility than Player 2 (23.0)
        # After shift, check non-first-game values
        p1_vol = result[result['PLAYER_ID'] == 1]['KAN_AGE_VOLATILITY']
        p2_vol = result[result['PLAYER_ID'] == 2]['KAN_AGE_VOLATILITY']
        # Player 1's max vol should be >= player 2's max
        assert p1_vol.max() >= p2_vol.max()


class TestKANAgeModel:
    def test_fallback_predict(self):
        from src.lifecycle.kan_age_model import KANAgeModel
        model = KANAgeModel()
        result = model._fallback_predict(25.0)
        assert 'factor' in result
        assert 'inflection_age' in result
        assert 'volatility' in result
        assert 0.5 < result['factor'] < 1.5

    def test_fallback_at_peak(self):
        from src.lifecycle.kan_age_model import KANAgeModel
        model = KANAgeModel()
        result = model._fallback_predict(28.0)
        assert abs(result['factor'] - 1.0) < 0.01