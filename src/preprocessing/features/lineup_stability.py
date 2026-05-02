"""Lineup stability feature group — starter rate, teammate continuity, rotation size variance, and minutes rank."""

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
from src.preprocessing.features._teammate_utils import TeammateContext


def _concat_new_columns(df: pd.DataFrame, new_columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Append a batch of aligned columns in a single concat."""
    if not new_columns:
        return df
    new_df = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df, new_df], axis=1)


class LineupStabilityFeatureGroup(FeatureGroup):
    """Quantifies lineup stability — how settled a player's role and teammates are.

    Output columns:
        LINEUP_STARTER_RATE_10       — fraction of last 10 games where MIN >= 25
        LINEUP_TEAM_STABILITY_5      — teammate Jaccard similarity over last 5 transitions
        LINEUP_TEAM_STABILITY_10     — teammate Jaccard similarity over last 10 transitions
        LINEUP_ROTATION_SIZE_VAR_5   — variance of team roster size over last 5 games
        LINEUP_MIN_RANK_AVG_5        — player's avg minutes rank on team over last 5 games
    """

    @property
    def name(self) -> str:
        return 'lineup_stability'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'TEAM_ID', 'MIN']

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

        if 'MIN' not in df.columns or 'TEAM_ID' not in df.columns:
            return df

        # Sort by player and date for correct rolling order
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # --- LINEUP_STARTER_RATE_10 ---
        # Fraction of last 10 games where MIN >= 25 (starter proxy)
        starter_rate = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).apply(
                lambda s: (s >= 25).mean(), raw=False
            )
        )
        col_name = 'LINEUP_STARTER_RATE_10'
        new_columns[col_name] = fill_series_with_prior(starter_rate, 0.5, diagnostics, col_name)

        # --- Shared teammate context ---
        tctx = TeammateContext(df)
        teammate_map = tctx.game_roster_map

        # --- LINEUP_TEAM_STABILITY_5 and LINEUP_TEAM_STABILITY_10 ---
        # Jaccard similarity of teammate sets across consecutive game transitions
        # Vectorized: shift the (TEAM_ID, GAME_DATE) key within each player group,
        # then compute Jaccard in a single pass over all rows.
        df['_curr_key'] = list(zip(df['TEAM_ID'], df['GAME_DATE']))
        df['_prev_key'] = df.groupby('PLAYER_ID')['_curr_key'].shift(1)

        def _jaccard_for_keys(curr_key, prev_key):
            if pd.isna(prev_key):
                return np.nan
            # prev_key comes from a tuple, but after shift it may be a list or tuple
            prev_key = tuple(prev_key) if not isinstance(prev_key, tuple) else prev_key
            curr_set = teammate_map.get(curr_key, set())
            prev_set = teammate_map.get(prev_key, set())
            if not curr_set and not prev_set:
                return 1.0
            if not curr_set or not prev_set:
                return 0.0
            inter = len(curr_set & prev_set)
            union = len(curr_set | prev_set)
            return inter / union if union > 0 else 0.0

        jaccard_vals = [
            _jaccard_for_keys(c, p)
            for c, p in zip(df['_curr_key'], df['_prev_key'])
        ]
        jaccard_series = pd.Series(jaccard_vals, index=df.index, dtype=float)

        # Shift Jaccard values to prevent leakage (we want past transitions only)
        jaccard_shifted = jaccard_series.groupby(df['PLAYER_ID']).shift(1)

        # Rolling average of Jaccard stability over windows
        for window, col_name in [(5, 'LINEUP_TEAM_STABILITY_5'), (10, 'LINEUP_TEAM_STABILITY_10')]:
            stability = jaccard_shifted.groupby(df['PLAYER_ID']).transform(
                lambda x: x.rolling(window, min_periods=2).mean()
            )
            new_columns[col_name] = fill_series_with_prior(stability, 0.5, diagnostics, col_name)

        # --- LINEUP_ROTATION_SIZE_VAR_5 ---
        # Variance of team roster size (players with MIN > 0) over last 5 games
        # Precompute roster size per (TEAM_ID, GAME_DATE) via the shared map
        roster_size_map = {k: len(v) for k, v in teammate_map.items()}
        df['_roster_size'] = df['_curr_key'].map(roster_size_map).astype(float)

        # Rolling variance of roster size per player (shifted)
        roster_var = df.groupby('PLAYER_ID')['_roster_size'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=3).var()
        )
        col_name = 'LINEUP_ROTATION_SIZE_VAR_5'
        new_columns[col_name] = fill_series_with_prior(roster_var, 0.0, diagnostics, col_name)

        # --- LINEUP_MIN_RANK_AVG_5 ---
        # Player's average minutes rank on team over last 5 games (1 = highest minutes)
        # Precompute MIN rank per (TEAM_ID, GAME_DATE) — rank descending (highest MIN = rank 1)
        df['_min_rank'] = df.groupby(['TEAM_ID', 'GAME_DATE'])['MIN'].transform(
            lambda x: x.rank(ascending=False, method='min')
        )

        min_rank_avg = df.groupby('PLAYER_ID')['_min_rank'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        col_name = 'LINEUP_MIN_RANK_AVG_5'
        new_columns[col_name] = fill_series_with_prior(min_rank_avg, 5.0, diagnostics, col_name)

        # Clean up temporary columns
        df.drop(columns=['_curr_key', '_prev_key', '_roster_size', '_min_rank'], inplace=True, errors='ignore')

        return _concat_new_columns(df, new_columns)
