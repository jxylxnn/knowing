"""Rest and game density feature group — schedule load, back-to-backs, and rest advantage."""

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


class RestGameDensityFeatureGroup(FeatureGroup):
    """Quantifies schedule density, rest patterns, and back-to-back fatigue.

    Output columns:
        SCHED_GAMES_3D         — number of games in last 3 days (shifted)
        SCHED_GAMES_5D         — number of games in last 5 days
        SCHED_GAMES_7D         — number of games in last 7 days
        SCHED_MIN_PER_DAY_5    — avg minutes per day over last 5 games
        SCHED_IS_B2B_SECOND    — binary: second night of a back-to-back
        SCHED_REST_ADVANTAGE   — player rest days minus opponent rest days
        SCHED_DENSITY_SCORE    — composite density score, clipped [0, 2]
    """

    @property
    def name(self) -> str:
        return 'rest_density'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'MIN']

    @property
    def optional_columns(self) -> List[str]:
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
        ctx = context or FeatureContext()

        if 'MIN' not in df.columns:
            return df

        # Sort by player and date for correct rolling order
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # --- SCHED_GAMES_XD: count of games in last X days (shifted) ---
        # Vectorized using time-based rolling windows on a unit series.
        # For each player we count previous games whose date falls in
        # [current - delta, current).  `closed='left'` excludes the
        # current game and `min_periods=0` yields 0 for empty windows.
        tmp = df[['PLAYER_ID', 'GAME_DATE']].copy()
        tmp['_count'] = 1.0

        for days, col_name in [(3, 'SCHED_GAMES_3D'), (5, 'SCHED_GAMES_5D'), (7, 'SCHED_GAMES_7D')]:
            rolling = (
                tmp.set_index('GAME_DATE')
                .groupby('PLAYER_ID')['_count']
                .rolling(f'{days}D', closed='left', min_periods=0)
                .sum()
            )
            # `rolling` aligns 1-to-1 with `tmp` rows because both are sorted
            # by PLAYER_ID and GAME_DATE.
            counts = pd.Series(rolling.values, index=df.index, dtype=float)
            counts_shifted = counts.groupby(df['PLAYER_ID']).shift(1)
            new_columns[col_name] = fill_series_with_prior(counts_shifted, 0.0, diagnostics, col_name)

        # --- SCHED_MIN_PER_DAY_5: average minutes per day over last 5 games ---
        # Total minutes in last 5 games / max(1, days spanned by those 5 games)
        shifted_min = df.groupby('PLAYER_ID')['MIN'].shift(1)

        # Convert dates to numeric (days since epoch) for rolling operations
        date_numeric = (df['GAME_DATE'] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 86400.0
        shifted_date_numeric = date_numeric.groupby(df['PLAYER_ID']).shift(1)

        min_sum_5 = shifted_min.groupby(df['PLAYER_ID']).transform(
            lambda x: x.rolling(5, min_periods=2).sum()
        )
        date_first_5 = shifted_date_numeric.groupby(df['PLAYER_ID']).transform(
            lambda x: x.rolling(5, min_periods=2).min()
        )
        date_last_5 = shifted_date_numeric.groupby(df['PLAYER_ID']).transform(
            lambda x: x.rolling(5, min_periods=2).max()
        )
        days_spanned = (date_last_5 - date_first_5).fillna(1.0).replace(0.0, 1.0).abs()
        days_spanned = days_spanned.clip(lower=1.0)

        min_per_day = min_sum_5 / days_spanned
        col_name = 'SCHED_MIN_PER_DAY_5'
        new_columns[col_name] = fill_series_with_prior(min_per_day, 24.0, diagnostics, col_name)

        # --- SCHED_IS_B2B_SECOND: binary flag for second night of back-to-back ---
        # A B2B second night means the previous game was within 1 day
        days_since_last = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.total_seconds() / 86400.0
        is_b2b_second = (days_since_last == 1.0).astype(float)
        # Shift by 1 to prevent leakage: we want to know if the PREVIOUS game was a B2B second
        is_b2b_shifted = is_b2b_second.groupby(df['PLAYER_ID']).shift(1)
        new_columns['SCHED_IS_B2B_SECOND'] = is_b2b_shifted.fillna(0.0)

        # --- SCHED_REST_ADVANTAGE: player's rest days minus opponent's rest days ---
        if 'OPPONENT_ID' in df.columns:
            # Compute player's rest days (days since last game)
            player_rest = days_since_last.fillna(4.0).clip(0, 7)

            # Precompute sorted unique game dates per team for fast binary search
            team_dates = df.groupby('TEAM_ID')['GAME_DATE'].apply(
                lambda x: np.array(sorted(x.unique()), dtype='datetime64[ns]')
            )

            # Vectorized opponent rest using np.searchsorted per opponent group
            opponent_rest = pd.Series(4.0, index=df.index, dtype=float)
            for opp_id, group in df.groupby('OPPONENT_ID'):
                if pd.isna(opp_id):
                    continue
                dates = team_dates.get(opp_id, np.array([], dtype='datetime64[ns]'))
                if len(dates) == 0:
                    continue
                game_dates = group['GAME_DATE'].values.astype('datetime64[ns]')
                idxs = np.searchsorted(dates, game_dates, side='left')
                valid = idxs > 0
                if valid.any():
                    prev_dates = dates[idxs[valid] - 1]
                    rest_days = (game_dates[valid] - prev_dates).astype('timedelta64[s]') / 86400.0
                    opponent_rest.loc[group.index[valid]] = rest_days.astype(float)

            rest_advantage = (player_rest - opponent_rest).fillna(0.0).clip(-7, 7)
            new_columns['SCHED_REST_ADVANTAGE'] = rest_advantage
        else:
            new_columns['SCHED_REST_ADVANTAGE'] = pd.Series(0.0, index=df.index)

        # --- SCHED_DENSITY_SCORE: composite density metric ---
        # (games_5d_norm * 0.3 + min_per_day_norm * 0.4 + is_b2b * 0.3), clipped to [0, 2]
        games_5d = new_columns.get('SCHED_GAMES_5D', pd.Series(0.0, index=df.index)).fillna(0.0)
        min_per_day_col = new_columns.get('SCHED_MIN_PER_DAY_5', pd.Series(24.0, index=df.index)).fillna(24.0)
        is_b2b = new_columns.get('SCHED_IS_B2B_SECOND', pd.Series(0.0, index=df.index)).fillna(0.0)

        # Normalize components to similar scales
        games_norm = games_5d / 5.0  # 0-1ish scale (5 games in 5 days = 1.0)
        min_norm = min_per_day_col / 48.0  # normalize by max possible min/game
        density_score = (games_norm * 0.3 + min_norm * 0.4 + is_b2b * 0.3).clip(0, 2)
        new_columns['SCHED_DENSITY_SCORE'] = density_score

        return _concat_new_columns(df, new_columns)
