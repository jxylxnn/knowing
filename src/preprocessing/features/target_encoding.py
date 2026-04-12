"""Target encoding and league-rank feature groups."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    fill_series_with_prior,
)


class TargetEncodingFeatureGroup(FeatureGroup):
    """Past-only player and team target encodings."""

    def __init__(self, target_cols: Optional[List[str]] = None, smoothing: int = 20):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
        self.smoothing = smoothing

    @property
    def name(self) -> str:
        return 'target_encoding'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID']

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

        for stat in [c for c in self.target_cols if c in df.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            player_expanding = df.groupby('PLAYER_ID')[stat].transform(lambda x: x.shift(1).expanding().mean())
            player_counts = df.groupby('PLAYER_ID').cumcount()
            player_weight = player_counts / (player_counts + self.smoothing)

            df[f'{stat}_PLAYER_TE'] = (
                player_weight * player_expanding + (1 - player_weight) * prior
            ).fillna(prior)

            if 'TEAM_ID' in df.columns:
                team_expanding = df.groupby('TEAM_ID')[stat].transform(lambda x: x.shift(1).expanding().mean())
                team_counts = df.groupby('TEAM_ID').cumcount()
                team_weight = team_counts / (team_counts + self.smoothing)
                df[f'{stat}_TEAM_TE'] = (
                    team_weight * team_expanding + (1 - team_weight) * prior
                ).fillna(prior)
            if diagnostics is not None:
                diagnostics.record_imputation(f'{stat}_PLAYER_TE', int(pd.isna(df[f'{stat}_PLAYER_TE']).sum()))

        return df


class LeagueRankingFeatureGroup(FeatureGroup):
    """Global percentile ranks over time."""

    def __init__(self, target_cols: Optional[List[str]] = None, window: int = 2000, min_periods: int = 500):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
        self.window = window
        self.min_periods = min_periods

    @property
    def name(self) -> str:
        return 'league_rank'

    @property
    def required_columns(self) -> List[str]:
        return ['GAME_DATE']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()

        if 'GAME_DATE' not in df.columns:
            for stat in self.target_cols:
                df[f'LEAGUE_PCT_{stat}'] = 0.5
            return df

        df_sorted = df.sort_values('GAME_DATE')
        for stat in [c for c in self.target_cols if c in df_sorted.columns]:
            past_values = df_sorted[stat].shift(1)
            pct_rank = past_values.rolling(window=self.window, min_periods=self.min_periods).apply(
                lambda x: float((x.iloc[-1] >= x).mean()), raw=False
            )
            df[f'LEAGUE_PCT_{stat}'] = pct_rank.reindex(df_sorted.index).reindex(df.index).fillna(0.5)

        return df
