"""Integration tests for the player lifecycle & bio-mechanical feature pipeline.

Verifies that:
1. All 4 lifecycle feature groups can run end-to-end with synthesized data.
2. Config lifecycle section loads and propagates correctly.
3. FeatureEngineer produces the expected lifecycle columns when enabled.
4. Precomputation of aging models produces cache files.
5. Missing bio data degrades gracefully (no crash, neutral defaults).
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player_data(n=500, seed=42):
    """Synthetic player-game rows with AGE/POSITION/PLAYER_ID for lifecycle features."""
    rng = np.random.RandomState(seed)
    player_ids = rng.randint(2000, 2020, n)
    ages = rng.randint(19, 38, n)
    positions = rng.choice(['PG', 'SG', 'SF', 'PF', 'C'], n)

    df = pd.DataFrame({
        'PLAYER_ID': player_ids,
        'PLAYER_NAME': [f'Player_{pid}' for pid in player_ids],
        'TEAM_ID': rng.randint(1610612737, 1610612767, n),
        'TEAM_ABBREVIATION': rng.choice(['LAL', 'BOS', 'GSW', 'MIA'], n),
        'TEAM_NAME': ['Team'] * n,
        'GAME_ID': rng.randint(10000000, 20000000, n),
        'GAME_DATE': pd.date_range('2023-10-01', periods=n, freq='D'),
        'MATCHUP': ['vs. OPP'] * n,
        'OPPONENT_ID': rng.randint(1610612737, 1610612767, n),
        'OPPONENT_ABBR': rng.choice(['BOS', 'LAL', 'MIA', 'GSW'], n),
        'WL': rng.choice(['W', 'L'], n),
        'MIN': rng.uniform(10, 40, n),
        'PTS': rng.poisson(12, n),
        'REB': rng.poisson(5, n),
        'AST': rng.poisson(3, n),
        'STL': rng.poisson(1, n),
        'BLK': rng.poisson(0.5, n),
        'TOV': rng.poisson(2, n),
        'FGA': rng.poisson(10, n),
        'FGM': rng.poisson(5, n),
        'FG3A': rng.poisson(4, n),
        'FG3M': rng.poisson(1.5, n),
        'FTA': rng.poisson(3, n),
        'FTM': rng.poisson(2, n),
        'SEASON_ID': ['22023'] * n,
        'VIDEO_AVAILABLE': [1] * n,
        # Lifecycle columns
        'AGE': ages,
        'POSITION': positions,
    })

    df['FGA_TEAM'] = df['FGA'] * 5
    df['FTA_TEAM'] = df['FTA'] * 5
    df['TOV_TEAM'] = df['TOV'] * 5
    df['PTS_TEAM'] = df['PTS'] * 5
    df['OREB_TEAM'] = rng.poisson(10, n)
    df['OPP_DREB'] = rng.poisson(30, n)

    return df


def _make_bios_df(player_data):
    """Extract unique player bios from player_data."""
    bios = player_data[['PLAYER_ID', 'AGE', 'POSITION']].drop_duplicates('PLAYER_ID')
    bios = bios.copy()
    bios['HEIGHT'] = '6-6'
    bios['WEIGHT'] = 210
    return bios


def _make_injury_history(player_data, tmp_dir):
    """Write a small injury_history.csv for testing injury risk features."""
    pids = player_data['PLAYER_ID'].unique()[:5]
    rows = []
    for pid in pids:
        for injury_type in ['Knee', 'Ankle', 'Back']:
            rows.append({
                'PLAYER_ID': pid,
                'DATE': '2023-11-15',
                'INJURY_TYPE': injury_type,
                'STATUS': 'Out',
                'TEAM_ABBR': 'LAL',
            })
    inj_df = pd.DataFrame(rows)
    path = Path(tmp_dir) / 'injury_history.csv'
    inj_df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_all_lifecycle_feature_groups_produce_columns():
    """All 4 lifecycle feature groups produce their expected output columns."""
    from src.preprocessing.feature_engineer import build_feature_engineer

    df = _make_player_data()
    fe = build_feature_engineer(
        enable_groups=['injury_risk', 'aging_curve', 'kan_aging', 'skill_development'],
        disable_groups=['rolling', 'efficiency', 'momentum', 'context', 'fatigue',
                        'minutes_confidence', 'rest_density', 'matchup',
                        'opponent_strength', 'pace', 'team_role', 'lineup_stability',
                        'injury_opportunity', 'teammate_usage', 'recency_form',
                        'archetype', 'defense_position', 'target_encoding', 'league_rank'],
    )
    result = fe.create_features(df)

    # Injury risk columns
    for col in ['INJURY_RISK_CAREER_COUNT', 'INJURY_RISK_LAST_90D',
                 'INJURY_RISK_LAST_30D', 'INJURY_RISK_WORKLOAD_SPIKE',
                 'INJURY_RISK_BACK_TO_BACK_STRESS', 'INJURY_RISK_AVG_DAYS_BETWEEN']:
        assert col in result.columns, f"Missing injury risk column: {col}"

    # Aging curve columns
    for col in ['AGING_PLAYER_AGE', 'AGING_YEARS_IN_LEAGUE',
                 'AGING_PEAK_AGE_EST', 'AGING_PRE_POST_PEAK',
                 'AGING_CURVE_FACTOR', 'AGING_DECLINE_RATE']:
        assert col in result.columns, f"Missing aging curve column: {col}"

    # KAN aging columns
    for col in ['KAN_AGE_NONLIN_FACTOR', 'KAN_AGE_INFLECTION_AGE', 'KAN_AGE_VOLATILITY']:
        assert col in result.columns, f"Missing KAN aging column: {col}"

    # Skill development columns
    for col in ['SKILL_DEV_PTS_VELOCITY', 'SKILL_DEV_EFF_VELOCITY',
                 'SKILL_DEV_REB_VELOCITY', 'SKILL_DEV_AST_TOV_TREND',
                 'SKILL_DEV_YOUTH_BOOST', 'SKILL_DEV_VETERAN_STEADY']:
        assert col in result.columns, f"Missing skill dev column: {col}"


@pytest.mark.integration
def test_lifecycle_config_loads_from_yaml():
    """Lifecycle config section loads from default.yaml with correct keys."""
    from src.config import load_config

    cfg = load_config(Path('config/default.yaml'))
    assert hasattr(cfg, 'lifecycle'), "Config missing 'lifecycle' attribute"
    lc = cfg.lifecycle
    assert lc.get('injury_risk_enabled') is True
    assert lc.get('aging_curve_enabled') is True
    assert lc.get('kan_aging_enabled') is True
    assert lc.get('skill_development_enabled') is True
    assert 'PG' in lc.get('aging_peak_priors', {})
    assert lc.get('kan_grid_size') == 5
    assert lc.get('youth_age_threshold') == 25


@pytest.mark.integration
def test_aging_model_precompute_creates_cache(tmp_path):
    """BIanusAgingModel.precompute_all writes a cache CSV file."""
    from src.lifecycle.aging_model import BIanusAgingModel

    df = _make_player_data()
    bios = _make_bios_df(df)
    cache_dir = str(tmp_path / 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    model = BIanusAgingModel()
    result = model.precompute_all(bios, df, cache_dir=cache_dir)

    # Should produce a non-empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # Cache file should exist
    cache_path = Path(cache_dir) / 'aging_curves.csv'
    assert cache_path.exists(), "Aging curves cache file not created"


@pytest.mark.integration
def test_kan_model_precompute_creates_cache(tmp_path):
    """KANAgeModel.precompute_all writes a cache CSV file."""
    from src.lifecycle.kan_age_model import KANAgeModel

    df = _make_player_data()
    bios = _make_bios_df(df)
    cache_dir = str(tmp_path / 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    model = KANAgeModel(device='cpu')
    result = model.precompute_all(bios, df, cache_dir=cache_dir)

    # Should produce a non-empty DataFrame or at least not crash
    assert isinstance(result, pd.DataFrame)
    cache_path = Path(cache_dir) / 'kan_aging_outputs.csv'
    assert cache_path.exists(), "KAN aging cache file not created"


@pytest.mark.integration
def test_missing_bio_data_graceful_degradation():
    """FeatureEngineer doesn't crash when AGE/POSITION columns are missing."""
    from src.preprocessing.feature_engineer import build_feature_engineer

    # Player data WITHOUT AGE/POSITION
    df = _make_player_data()
    df = df.drop(columns=['AGE', 'POSITION'])

    fe = build_feature_engineer(
        enable_groups=['injury_risk', 'aging_curve', 'kan_aging', 'skill_development'],
        disable_groups=['rolling', 'efficiency', 'momentum', 'context', 'fatigue',
                        'minutes_confidence', 'rest_density', 'matchup',
                        'opponent_strength', 'pace', 'team_role', 'lineup_stability',
                        'injury_opportunity', 'teammate_usage', 'recency_form',
                        'archetype', 'defense_position', 'target_encoding', 'league_rank'],
    )
    # Should not raise
    result = fe.create_features(df)
    # Lifecycle columns should still exist (with neutral defaults)
    assert 'AGING_PLAYER_AGE' in result.columns
    assert 'INJURY_RISK_CAREER_COUNT' in result.columns


@pytest.mark.integration
def test_injury_risk_uses_history_file(tmp_path):
    """Injury risk features read from injury_history.csv when present."""
    from src.preprocessing.features.injury_risk import InjuryRiskFeatureGroup

    df = _make_player_data(n=200)
    hist_path = _make_injury_history(df, str(tmp_path))

    group = InjuryRiskFeatureGroup(data_dir=str(tmp_path))
    # InjuryHistoryLogger writes to data_dir/injury_history.csv, so set data_dir to tmp_path
    result = group.create(df)

    # Players with injury history should have non-zero career counts
    # (Note: shift(1) means the first row per player is always 0, so check max)
    players_with_history = df['PLAYER_ID'].unique()[:5]
    for pid in players_with_history:
        pid_rows = result[result['PLAYER_ID'] == pid]
        if len(pid_rows) > 1:
            assert pid_rows['INJURY_RISK_CAREER_COUNT'].max() > 0, \
                f"Player {pid} should have non-zero injury history after shift"