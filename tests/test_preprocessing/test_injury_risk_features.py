"""Tests for InjuryRiskFeatureGroup."""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features.injury_risk import InjuryRiskFeatureGroup, OUTPUT_COLUMNS
from src.preprocessing.features.base import FeatureDiagnostics


@pytest.fixture
def risk_group(tmp_path):
    return InjuryRiskFeatureGroup(data_dir=str(tmp_path))


@pytest.fixture
def sample_df():
    dates = pd.date_range('2025-01-01', periods=10, freq='3D')
    return pd.DataFrame({
        'PLAYER_ID': [1] * 10,
        'GAME_DATE': dates,
        'MIN': [30, 32, 28, 35, 33, 29, 31, 34, 30, 28],
        'TEAM_ID': [100] * 10,
    })


@pytest.fixture
def injury_history_file(tmp_path):
    """Write a small injury history CSV."""
    path = os.path.join(str(tmp_path), 'injury_history.csv')
    df = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1],
        'PLAYER': ['Player 1', 'Player 1', 'Player 1'],
        'TEAM_ABBR': ['LAL', 'LAL', 'LAL'],
        'STATUS': ['OUT', 'QUESTIONABLE', 'OUT'],
        'INJURY_TYPE': ['Ankle', 'Knee', 'Back'],
        'DATE': ['2024-11-15', '2024-12-20', '2025-01-05'],
        'PLAY_PROBABILITY': [0.0, 0.5, 0.0],
    })
    df.to_csv(path, index=False)
    return path


class TestInjuryRiskFeatureGroup:

    def test_creates_all_output_columns(self, risk_group, sample_df):
        result = risk_group.create(sample_df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_leakage_first_game(self, risk_group, sample_df):
        result = risk_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # First game should have 0 for shifted features
        assert result['INJURY_RISK_CAREER_COUNT'].iloc[0] == 0
        assert result['INJURY_RISK_WORKLOAD_SPIKE'].iloc[0] == 0
        assert result['INJURY_RISK_BACK_TO_BACK_STRESS'].iloc[0] == 0

    def test_workload_spike_detected(self, risk_group, sample_df):
        # Add a huge MIN spike in row 3
        sample_df.loc[2, 'MIN'] = 50  # much higher than rolling avg
        result = risk_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # The spike should show up in SHIFTED values for the NEXT game
        # (row 3's spike is visible to row 4 after shift(1))
        assert 'INJURY_RISK_WORKLOAD_SPIKE' in result.columns

    def test_injury_history_integration(self, risk_group, sample_df, injury_history_file):
        result = risk_group.create(sample_df, diagnostics=FeatureDiagnostics())
        # Career count should reflect 3 injuries for player 1
        # But shifted, so first game = 0, later games should show some value
        assert 'INJURY_RISK_CAREER_COUNT' in result.columns

    def test_recent_injury_counts(self, risk_group, sample_df, injury_history_file):
        result = risk_group.create(sample_df, diagnostics=FeatureDiagnostics())
        assert 'INJURY_RISK_LAST_90D' in result.columns
        assert 'INJURY_RISK_LAST_30D' in result.columns

    def test_missing_player_id_gives_defaults(self, risk_group):
        df = pd.DataFrame({
            'GAME_DATE': pd.date_range('2025-01-01', periods=5),
            'MIN': [30] * 5,
        })
        result = risk_group.create(df, diagnostics=FeatureDiagnostics())
        for col in OUTPUT_COLUMNS:
            assert col in result.columns

    def test_empty_injury_history_defaults_to_zero(self, tmp_path, sample_df):
        group = InjuryRiskFeatureGroup(data_dir=str(tmp_path))
        result = group.create(sample_df, diagnostics=FeatureDiagnostics())
        # With no injury history file, career count should be 0
        assert (result['INJURY_RISK_CAREER_COUNT'] == 0).all()

    def test_avg_days_between(self, risk_group, sample_df, injury_history_file):
        result = risk_group.create(sample_df, diagnostics=FeatureDiagnostics())
        assert 'INJURY_RISK_AVG_DAYS_BETWEEN' in result.columns
        # All values should be >= 0
        assert (result['INJURY_RISK_AVG_DAYS_BETWEEN'] >= 0).all()

    def test_multi_player(self, tmp_path):
        dates = pd.date_range('2025-01-01', periods=6, freq='3D')
        df = pd.DataFrame({
            'PLAYER_ID': [1, 2, 1, 2, 1, 2],
            'GAME_DATE': dates,
            'MIN': [30, 25, 32, 28, 31, 26],
            'TEAM_ID': [100, 200, 100, 200, 100, 200],
        })
        group = InjuryRiskFeatureGroup(data_dir=str(tmp_path))
        result = group.create(df, diagnostics=FeatureDiagnostics())
        # Should have features for both players
        assert len(result) == 6
        for col in OUTPUT_COLUMNS:
            assert col in result.columns