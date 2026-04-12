"""Context and fatigue feature groups."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    fill_series_with_prior,
)


class ContextualFeatureGroup(FeatureGroup):
    """Home/away and rest-day context."""

    @property
    def name(self) -> str:
        return 'context'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE']

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

        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = df['MATCHUP'].astype(str).str.contains('vs.').astype(int)
        else:
            df['IS_HOME'] = 0

        days_since_last = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
        missing = days_since_last.isna()
        df['DAYS_SINCE_LAST'] = fill_series_with_prior(days_since_last, 4.0, diagnostics, 'DAYS_SINCE_LAST')
        df['REST_DAYS'] = df['DAYS_SINCE_LAST'].clip(0, 7)
        df['IS_B2B'] = (df['DAYS_SINCE_LAST'] == 1).astype(int)
        df['CONTEXT_COLD_START'] = missing.astype(int)

        if diagnostics is not None:
            diagnostics.record_imputation('DAYS_SINCE_LAST', int(missing.sum()))
            diagnostics.record_imputation('CONTEXT_COLD_START', int(missing.sum()))

        return df


class FatigueFeatureGroup(FeatureGroup):
    """Historical load and fatigue indicators."""

    @property
    def name(self) -> str:
        return 'fatigue'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'MIN']

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

        if 'MIN' not in df.columns:
            df['MIN'] = context.league_priors.get('MIN', 24.0)

        mins_lag = df.groupby('PLAYER_ID')['MIN'].shift(1).fillna(0.0)
        df['MINS_LAST_3'] = mins_lag.groupby(df['PLAYER_ID']).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
        df['MINS_LAST_7'] = mins_lag.groupby(df['PLAYER_ID']).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)

        df['FATIGUE_SCORE'] = (
            (df['MINS_LAST_3'] / 100.0) * 0.4
            + (df.get('IS_B2B', 0) * 0.3)
            + ((4 - df.get('REST_DAYS', 4).clip(0, 4)) * 0.3)
        ).clip(lower=0.0, upper=2.0)

        if diagnostics is not None:
            diagnostics.record_imputation('MINS_LAST_3', int(pd.isna(df['MINS_LAST_3']).sum()))
            diagnostics.record_imputation('MINS_LAST_7', int(pd.isna(df['MINS_LAST_7']).sum()))
            diagnostics.record_imputation('FATIGUE_SCORE', int(pd.isna(df['FATIGUE_SCORE']).sum()))

        return df
