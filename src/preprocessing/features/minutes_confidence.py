"""Minutes confidence feature group — rolling variance, trend, and starter-rate signals."""

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


def _concat_new_columns(df: pd.DataFrame, new_columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Append a batch of aligned columns in a single concat."""
    if not new_columns:
        return df
    new_df = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df, new_df], axis=1)


class MinutesConfidenceFeatureGroup(FeatureGroup):
    """Quantifies confidence in a player's minutes projection.

    Output columns:
        MIN_CONF_VAR_5 / MIN_CONF_VAR_10 — rolling variance of MIN
        MIN_CONF_TREND_3_10 / MIN_CONF_TREND_5_20 — short-vs-long trend ratios
        MIN_CONF_ABOVE_NORMAL_10 — fraction of recent games above rolling avg
        MIN_CONF_STARTER_RATE_10 — fraction of recent games with MIN >= 25
        MIN_CONF_COLD_START — binary flag when rolling count < 5
    """

    def __init__(self, target_cols: Optional[List[str]] = None, windows: Optional[List[int]] = None):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
        self.windows = windows or [3, 5, 10, 20, 50]

    @property
    def name(self) -> str:
        return 'minutes_confidence'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'MIN']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()
        ctx = context or FeatureContext()

        if 'MIN' not in df.columns:
            return df

        # Sort by player and date for correct rolling order
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # --- Rolling variance of MIN ---
        for w in [5, 10]:
            var = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=3).var()
            )
            col_name = f'MIN_CONF_VAR_{w}'
            prior = ctx.league_priors.get('MIN', 24.0)
            new_columns[col_name] = fill_series_with_prior(var, prior, diagnostics, col_name)

        # --- Minutes trend: short vs long window ---
        for short_w, long_w in [(3, 10), (5, 20)]:
            short_avg = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.shift(1).rolling(short_w, min_periods=2).mean()
            )
            long_avg = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.shift(1).rolling(long_w, min_periods=3).mean()
            )
            trend = ((short_avg - long_avg) / long_avg.replace(0, np.nan)).clip(-1, 1)
            col_name = f'MIN_CONF_TREND_{short_w}_{long_w}'
            new_columns[col_name] = fill_series_with_prior(trend, 0.0, diagnostics, col_name)

        # --- Fraction of games above normal minutes ---
        min_avg_10 = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        above_normal = (df['MIN'].shift(1) > min_avg_10).astype(float)
        above_normal_rate = above_normal.groupby(df['PLAYER_ID']).transform(
            lambda x: x.rolling(10, min_periods=3).mean()
        )
        col_name = 'MIN_CONF_ABOVE_NORMAL_10'
        new_columns[col_name] = fill_series_with_prior(above_normal_rate, 0.5, diagnostics, col_name)

        # --- Starter rate (MIN >= 25) ---
        is_starter = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).apply(lambda s: (s >= 25).mean(), raw=False)
        )
        col_name = 'MIN_CONF_STARTER_RATE_10'
        new_columns[col_name] = fill_series_with_prior(is_starter, 0.5, diagnostics, col_name)

        # --- Cold start flag ---
        min_count_10 = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).count()
        )
        new_columns['MIN_CONF_COLD_START'] = (min_count_10 < 5).astype(float).fillna(0.0)

        return _concat_new_columns(df, new_columns)