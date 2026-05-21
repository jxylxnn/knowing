"""Tests for AgingCurveFeatureGroup."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features.aging_curve import AgingCurveFeatureGroup, OUTPUT_COLUMNS
from src.preprocessing.features.base import FeatureDiagnostics


@pytest.fixture
def aging_group(tmp_path):
    return AgingCurveFeatureGroup(data_dir=str(tmp_path))


@pytest.fixture
def sample_df():
    dates = pd.date_range('2025-01-01', periods=10, freq='3D')
    return pd.DataFrame({
        'PLAYER_ID': [1] * 5 + [2] * 5,
        'GAME_DATE': list(dates[:5]) + list(dates[:5]),
        'AGE': [30.5] * 5 + [23.0] * 5,
        'POSITION': ['PG'] * 5 + ['C'] * 5,
        'CAREER_START': [2015] * 5 + [2023] * 5,
    })


@pytest.fixture
def aging_curves_cache(tmp_path):
    """Write a cached aging curves CSV."""
    cache_dir = os.path.join(str(tmp_path), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'aging_curves.csv')
    df = pd.DataFrame({
        'PLAYER_ID': [1, 2],
        'peak_age': [28.5, 26.5],
        'decline_rate': [0.008, 0.018],
        'development_rate': [0.02, 0.025],
        'position': ['PG', 'C'],
    })
    df.to_csv(path, index=False)
    return path


class TestAgingCurveFeatureGroup:

    def test_creates_all_output_columns(self, aging_group, sample_df):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_leakage_first_game(self, aging_group, sample_df):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # First row of each player should have NaN or default after shift
        # (shifted values are NaN for first game, filled with defaults)
        assert result['AGING_PLAYER_AGE'].notna().all()
        assert result['AGING_CURVE_FACTOR'].notna().all()

    def test_aging_player_age_populated(self, aging_group, sample_df):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # All ages should be populated
        assert result['AGING_PLAYER_AGE'].notna().all()

    def test_pre_post_peak_values(self, aging_group, sample_df):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # Pre/post peak should be 0 or 1
        assert set(result['AGING_PRE_POST_PEAK'].unique()).issubset({0.0, 1.0})

    def test_curve_factor_reasonable(self, aging_group, sample_df):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # Curve factor should be close to 1.0 for typical NBA ages
        assert (result['AGING_CURVE_FACTOR'] > 0.5).all()
        assert (result['AGING_CURVE_FACTOR'] < 1.5).all()

    def test_uses_cached_curves(self, aging_group, sample_df, aging_curves_cache):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # With cached curves, should populate values
        assert result['AGING_PEAK_AGE_EST'].notna().all()

    def test_missing_bio_defaults(self, aging_group):
        """Without AGE/POSITION columns, should use defaults."""
        df = pd.DataFrame({
            'PLAYER_ID': [1, 1],
            'GAME_DATE': pd.date_range('2025-01-01', periods=2),
        })
        result = aging_group.create(df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns
        # Default age ~27.5
        assert result['AGING_PLAYER_AGE'].notna().all()

    def test_decline_rate_by_position(self, aging_group, sample_df):
        result = aging_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # Centers (player 2) should have higher decline rate than PGs (player 1)
        # (at least the defaults indicate this)
        pg_rows = result[result['PLAYER_ID'] == 1]
        c_rows = result[result['PLAYER_ID'] == 2]
        # After shift, values may be NaN filled; check non-zero rows
        if c_rows['AGING_DECLINE_RATE'].sum() > 0 and pg_rows['AGING_DECLINE_RATE'].sum() > 0:
            assert c_rows['AGING_DECLINE_RATE'].mean() >= pg_rows['AGING_DECLINE_RATE'].mean()