"""Skill development feature group — growth velocity metrics.

Proxies skill development using year-over-year stat improvements as
"growth velocity" metrics. Full CV-based skill assessment is out of scope;
we use stat trajectory as a proxy.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    normalize_output_columns,
)

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    'SKILL_DEV_PTS_VELOCITY',
    'SKILL_DEV_EFF_VELOCITY',
    'SKILL_DEV_REB_VELOCITY',
    'SKILL_DEV_AST_TOV_TREND',
    'SKILL_DEV_YOUTH_BOOST',
    'SKILL_DEV_VETERAN_STEADY',
]


class SkillDevelopmentFeatureGroup(FeatureGroup):
    """Skill development metrics from stat growth velocity.

    Computes per-player YoY changes in key efficiency stats and flags
    young improvers (prospect signal) and stable veterans (floor signal).
    All features use shift(1) within player groups.
    """

    @property
    def name(self) -> str:
        return 'skill_development'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'MIN', 'PTS']

    @property
    def optional_columns(self) -> List[str]:
        return ['REB', 'AST', 'TOV', 'FGA', 'FGM', 'FTA', 'FTM', 'AGE']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()

        if 'PLAYER_ID' not in df.columns or 'GAME_DATE' not in df.columns:
            df = normalize_output_columns(df, OUTPUT_COLUMNS)
            return df

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        # Extract season from GAME_DATE for YoY computation
        df['_season'] = df['GAME_DATE'].dt.year - (
            df['GAME_DATE'].dt.month < 10
        ).astype(int)  # NBA season: Oct start

        # Per-player per-season aggregates
        has_reb = 'REB' in df.columns
        has_ast = 'AST' in df.columns
        has_tov = 'TOV' in df.columns
        has_age = 'AGE' in df.columns

        # Compute per-minute stats
        df['_pts_per_min'] = df['PTS'] / df['MIN'].clip(lower=1)

        if has_reb:
            df['_reb_per_min'] = df['REB'] / df['MIN'].clip(lower=1)
        else:
            df['_reb_per_min'] = 0.0

        if has_ast and has_tov:
            df['_ast_tov_ratio'] = df['AST'] / df['TOV'].clip(lower=1)
        elif has_ast:
            df['_ast_tov_ratio'] = df['AST'].clip(lower=0)
        else:
            df['_ast_tov_ratio'] = 1.4  # league average

        # True Shooting % proxy
        if all(c in df.columns for c in ['FGA', 'FGM', 'FTA', 'FTM']):
            df['_ts_pct'] = (
                df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']).clip(lower=1))
            )
        else:
            df['_ts_pct'] = 0.56  # league average

        # Season averages per player
        season_avgs = df.groupby(['PLAYER_ID', '_season']).agg(
            _pts_per_min_avg=('_pts_per_min', 'mean'),
            _ts_pct_avg=('_ts_pct', 'mean'),
            _reb_per_min_avg=('_reb_per_min', 'mean'),
            _ast_tov_avg=('_ast_tov_ratio', 'mean'),
        ).reset_index()

        # Compute YoY velocity (change from previous season)
        season_avgs = season_avgs.sort_values(['PLAYER_ID', '_season'])
        for stat_col in ['_pts_per_min_avg', '_ts_pct_avg', '_reb_per_min_avg', '_ast_tov_avg']:
            vel_col = stat_col + '_vel'
            season_avgs[vel_col] = season_avgs.groupby('PLAYER_ID')[stat_col].diff()

        # Merge velocity back onto game-level df
        velocity_cols = [
            '_pts_per_min_avg_vel', '_ts_pct_avg_vel',
            '_reb_per_min_avg_vel', '_ast_tov_avg_vel',
        ]
        velocity_map = season_avgs[['PLAYER_ID', '_season'] + velocity_cols]
        df = df.merge(velocity_map, on=['PLAYER_ID', '_season'], how='left')

        # Assign output columns
        df['SKILL_DEV_PTS_VELOCITY'] = df['_pts_per_min_avg_vel'].fillna(0)
        df['SKILL_DEV_EFF_VELOCITY'] = df['_ts_pct_avg_vel'].fillna(0)
        df['SKILL_DEV_REB_VELOCITY'] = df['_reb_per_min_avg_vel'].fillna(0)
        df['SKILL_DEV_AST_TOV_TREND'] = df['_ast_tov_avg_vel'].fillna(0)

        # Youth boost flag: age < 25 AND improving
        if has_age:
            age_series = pd.to_numeric(df['AGE'], errors='coerce').fillna(30.0)
            improving = (df['SKILL_DEV_PTS_VELOCITY'] > 0) | (df['SKILL_DEV_EFF_VELOCITY'] > 0)
            df['SKILL_DEV_YOUTH_BOOST'] = ((age_series < 25) & improving).astype(float)
        else:
            df['SKILL_DEV_YOUTH_BOOST'] = 0.0

        # Veteran steady flag: age > 30 AND efficiency change within ±2%
        if has_age:
            eff_stable = df['SKILL_DEV_EFF_VELOCITY'].abs() < 0.02
            df['SKILL_DEV_VETERAN_STEADY'] = ((age_series > 30) & eff_stable).astype(float)
        else:
            df['SKILL_DEV_VETERAN_STEADY'] = 0.0

        # Shift all features within player groups
        for col in OUTPUT_COLUMNS:
            if col in df.columns:
                df[col] = df.groupby('PLAYER_ID')[col].shift(1)

        # Fill first-game NaN
        df['SKILL_DEV_PTS_VELOCITY'] = df['SKILL_DEV_PTS_VELOCITY'].fillna(0)
        df['SKILL_DEV_EFF_VELOCITY'] = df['SKILL_DEV_EFF_VELOCITY'].fillna(0)
        df['SKILL_DEV_REB_VELOCITY'] = df['SKILL_DEV_REB_VELOCITY'].fillna(0)
        df['SKILL_DEV_AST_TOV_TREND'] = df['SKILL_DEV_AST_TOV_TREND'].fillna(0)
        df['SKILL_DEV_YOUTH_BOOST'] = df['SKILL_DEV_YOUTH_BOOST'].fillna(0)
        df['SKILL_DEV_VETERAN_STEADY'] = df['SKILL_DEV_VETERAN_STEADY'].fillna(0)

        # Cleanup temp columns
        temp_cols = [c for c in df.columns if c.startswith('_')]
        df = df.drop(columns=temp_cols)

        df = normalize_output_columns(df, OUTPUT_COLUMNS)
        return df