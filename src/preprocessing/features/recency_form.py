"""Recency form feature group — recent-vs-season deltas, form ratios, and volatility."""

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


class RecencyFormFeatureGroup(FeatureGroup):
    """Quantifies how a player's recent form compares to their season baseline.

    Output columns (per stat in target_cols):
        RECENCY_{STAT}_VS_SEASON   — (last 5 avg - expanding avg) / expanding avg, clipped [-2, 2]
        RECENCY_{STAT}_FORM_RATIO  — last 3 avg / last 10 avg, clipped [0.2, 5.0]
        RECENCY_{STAT}_VOLATILITY_5 — coefficient of variation over last 5 games, clipped [0, 2]

    Additional output columns:
        RECENCY_USAGE_DELTA_5  — change in usage rate over last 5 games
        RECENCY_EFF_DELTA_5    — change in efficiency over last 5 games
        RECENCY_MIN_DELTA_5    — change in minutes over last 5 games
    """

    def __init__(self, target_cols: Optional[List[str]] = None):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']

    @property
    def name(self) -> str:
        return 'recency_form'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE']

    @property
    def optional_columns(self) -> List[str]:
        return ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']

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

        # Sort by player and date for correct rolling order
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # Determine which target cols are available
        available_stats = [c for c in self.target_cols if c in df.columns]

        for stat in available_stats:
            prior = float(ctx.league_priors.get(stat, 0.0))
            shifted = df.groupby('PLAYER_ID')[stat].shift(1)

            # --- RECENCY_{STAT}_VS_SEASON ---
            # (last 5 avg - expanding season avg) / expanding season avg, clipped [-2, 2]
            roll_5 = shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(5, min_periods=2).mean()
            )
            expanding_avg = shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.expanding(min_periods=2).mean()
            )
            vs_season = ((roll_5 - expanding_avg) / expanding_avg.replace(0, np.nan)).clip(-2, 2)
            col_name = f'RECENCY_{stat}_VS_SEASON'
            new_columns[col_name] = fill_series_with_prior(vs_season, 0.0, diagnostics, col_name)

            # --- RECENCY_{STAT}_FORM_RATIO ---
            # last 3 avg / last 10 avg, clipped [0.2, 5.0], fill NaN with 1.0
            roll_3 = shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(3, min_periods=2).mean()
            )
            roll_10 = shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(10, min_periods=3).mean()
            )
            form_ratio = (roll_3 / roll_10.replace(0, np.nan)).clip(0.2, 5.0)
            col_name = f'RECENCY_{stat}_FORM_RATIO'
            new_columns[col_name] = fill_series_with_prior(form_ratio, 1.0, diagnostics, col_name)

            # --- RECENCY_{STAT}_VOLATILITY_5 ---
            # coefficient of variation (std/mean) over last 5 games, clipped [0, 2]
            roll_std_5 = shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(5, min_periods=3).std()
            )
            roll_mean_5 = shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
            volatility = (roll_std_5 / roll_mean_5.replace(0, np.nan)).clip(0, 2)
            col_name = f'RECENCY_{stat}_VOLATILITY_5'
            new_columns[col_name] = fill_series_with_prior(volatility, 0.0, diagnostics, col_name)

        # --- RECENCY_USAGE_DELTA_5 ---
        # Change in usage rate over last 5 games
        if 'ROLL_USG_PCT_5' in df.columns:
            # Use pre-computed usage if available
            usage_delta = df.groupby('PLAYER_ID')['ROLL_USG_PCT_5'].diff()
            col_name = 'RECENCY_USAGE_DELTA_5'
            new_columns[col_name] = fill_series_with_prior(usage_delta, 0.0, diagnostics, col_name)
        elif 'FGA' in df.columns:
            # Compute from FGA as a simple proxy
            shifted_fga = df.groupby('PLAYER_ID')['FGA'].shift(1)
            roll_fga_5 = shifted_fga.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(5, min_periods=2).mean()
            )
            roll_fga_1 = shifted_fga.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(1, min_periods=1).mean()
            )
            # Delta: recent avg minus 5-game avg
            usage_delta = roll_fga_1 - roll_fga_5
            col_name = 'RECENCY_USAGE_DELTA_5'
            new_columns[col_name] = fill_series_with_prior(usage_delta, 0.0, diagnostics, col_name)

        # --- RECENCY_EFF_DELTA_5 ---
        # Change in efficiency over last 5 games
        if 'ROLL_TS_PCT_5' in df.columns:
            eff_delta = df.groupby('PLAYER_ID')['ROLL_TS_PCT_5'].diff()
            col_name = 'RECENCY_EFF_DELTA_5'
            new_columns[col_name] = fill_series_with_prior(eff_delta, 0.0, diagnostics, col_name)
        elif all(c in df.columns for c in ['PTS', 'FGA', 'FTA']):
            # Compute TS% proxy delta
            shifted_pts = df.groupby('PLAYER_ID')['PTS'].shift(1)
            shifted_fga = df.groupby('PLAYER_ID')['FGA'].shift(1)
            shifted_fta = df.groupby('PLAYER_ID')['FTA'].shift(1)
            ts_proxy = shifted_pts / (2 * (shifted_fga + 0.44 * shifted_fta + 1e-7))
            roll_ts_5 = ts_proxy.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(5, min_periods=2).mean()
            )
            roll_ts_1 = ts_proxy.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(1, min_periods=1).mean()
            )
            eff_delta = roll_ts_1 - roll_ts_5
            col_name = 'RECENCY_EFF_DELTA_5'
            new_columns[col_name] = fill_series_with_prior(eff_delta, 0.0, diagnostics, col_name)

        # --- RECENCY_MIN_DELTA_5 ---
        # Change in minutes over last 5 games
        if 'MIN' in df.columns:
            shifted_min = df.groupby('PLAYER_ID')['MIN'].shift(1)
            roll_min_5 = shifted_min.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(5, min_periods=2).mean()
            )
            roll_min_1 = shifted_min.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(1, min_periods=1).mean()
            )
            min_delta = roll_min_1 - roll_min_5
            col_name = 'RECENCY_MIN_DELTA_5'
            new_columns[col_name] = fill_series_with_prior(min_delta, 0.0, diagnostics, col_name)

        return _concat_new_columns(df, new_columns)