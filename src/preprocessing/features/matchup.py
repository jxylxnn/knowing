"""Matchup-related feature groups."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    fill_series_with_prior,
)


class MatchupFeatureGroup(FeatureGroup):
    """Historical player-vs-opponent performance."""

    def __init__(self, target_cols: Optional[List[str]] = None, recent_window: int = 5):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
        self.recent_window = recent_window

    @property
    def name(self) -> str:
        return 'matchup'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'OPPONENT_ID']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()
        context = context or FeatureContext()

        if 'OPPONENT_ID' not in df.columns:
            for stat in self.target_cols:
                df[f'VS_OPP_{stat}_AVG'] = context.league_priors.get(stat, 0.0)
                df[f'VS_OPP_{stat}_RECENT'] = context.league_priors.get(stat, 0.0)
            return df

        grouped = df.groupby(['PLAYER_ID', 'OPPONENT_ID'], sort=False)
        for stat in [c for c in self.target_cols if c in df.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            shifted = grouped[stat].shift(1)

            career_mean = shifted.groupby([df['PLAYER_ID'], df['OPPONENT_ID']]).transform(lambda x: x.expanding().mean())
            recent_mean = shifted.groupby([df['PLAYER_ID'], df['OPPONENT_ID']]).transform(
                lambda x: x.rolling(self.recent_window, min_periods=1).mean()
            )
            matchup_count = shifted.groupby([df['PLAYER_ID'], df['OPPONENT_ID']]).transform(lambda x: x.rolling(self.recent_window, min_periods=1).count())

            df[f'VS_OPP_{stat}_AVG'] = fill_series_with_prior(career_mean, prior, diagnostics, f'VS_OPP_{stat}_AVG')
            df[f'VS_OPP_{stat}_RECENT'] = fill_series_with_prior(recent_mean, prior, diagnostics, f'VS_OPP_{stat}_RECENT')
            df[f'VS_OPP_{stat}_COUNT'] = matchup_count.fillna(0.0)

        return df


class OpponentStrengthFeatureGroup(FeatureGroup):
    """Safe opponent defensive strength using historical opponent team context."""

    def __init__(self, target_cols: Optional[List[str]] = None):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']

    @property
    def name(self) -> str:
        return 'opponent_strength'

    @property
    def required_columns(self) -> List[str]:
        return ['OPPONENT_ID']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()
        context = context or FeatureContext()

        league_avgs = {'PTS': 105.0, 'REB': 42.0, 'AST': 22.0}
        safe_prefix = 'OPP_TEAM_DEF_'
        for stat in self.target_cols:
            safe_col = f'{safe_prefix}{stat}_ALLOWED_ROLL_10'
            if safe_col in df.columns:
                def_norm = df[safe_col] / league_avgs.get(stat, 105.0)
                df[f'RELATIVE_OPP_DEF_{stat}'] = def_norm.fillna(1.0)
                df[f'ROLL_OPP_DEF_{stat}_10'] = df[safe_col].fillna(league_avgs.get(stat, 105.0)) / league_avgs.get(stat, 105.0)
            else:
                df[f'RELATIVE_OPP_DEF_{stat}'] = 1.0
                df[f'ROLL_OPP_DEF_{stat}_10'] = 1.0
            if diagnostics is not None:
                diagnostics.record_imputation(f'RELATIVE_OPP_DEF_{stat}', int(df[f'RELATIVE_OPP_DEF_{stat}'].isna().sum()))

        df['OPP_DEF_RATING'] = (
            df[[f'RELATIVE_OPP_DEF_{stat}' for stat in self.target_cols if f'RELATIVE_OPP_DEF_{stat}' in df.columns]]
            .mean(axis=1)
            .fillna(1.0)
        )
        df['DEF_DIFFICULTY'] = (2.0 - df['OPP_DEF_RATING']).clip(0.5, 1.5)
        df['OPP_DEF_RANK'] = 0.5
        df['QUALITY_DEF_AVOIDANCE'] = 0.5

        for stat in self.target_cols:
            df[f'DEF_MATCHUP_TREND_{stat}'] = (
                df.get(f'VS_OPP_{stat}_AVG', context.league_priors.get(stat, 0.0))
                / (df['OPP_DEF_RATING'] + 1e-6)
            ).fillna(0.0)

        if 'IS_HOME' in df.columns:
            home_advantage = 1.05
            tough_def_penalty = df['OPP_DEF_RATING'] * 0.05
            df['DEF_MATCHUP_HOME_ADJ'] = (home_advantage - (1 - df['IS_HOME']) * tough_def_penalty).clip(0.8, 1.2)
            df['DEF_MATCHUP_AWAY_ADJ'] = (1.0 - df['IS_HOME'] * tough_def_penalty).clip(0.8, 1.2)
        else:
            df['DEF_MATCHUP_HOME_ADJ'] = 1.0
            df['DEF_MATCHUP_AWAY_ADJ'] = 1.0

        return df
