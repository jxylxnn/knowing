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
        # For each player, for each game, count how many previous games (shifted)
        # fall within the X-day window before the current game date.
        for days, col_name in [(3, 'SCHED_GAMES_3D'), (5, 'SCHED_GAMES_5D'), (7, 'SCHED_GAMES_7D')]:
            delta = pd.Timedelta(days=days)
            counts = pd.Series(0.0, index=df.index, dtype=float)

            for player_id, group in df.groupby('PLAYER_ID'):
                dates = group['GAME_DATE'].values
                idx = group.index.values
                for i in range(len(dates)):
                    current_date = dates[i]
                    window_start = current_date - delta
                    # Count previous games (not including current) within window
                    count = 0
                    for j in range(i):
                        if window_start <= dates[j] < current_date:
                            count += 1
                    counts.loc[idx[i]] = float(count)

            # Shift to prevent leakage: we want past games only
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

            # Build team game date lookup for opponent rest calculation
            team_game_dates: dict[str, list] = {}
            for tid, group in df.groupby('TEAM_ID'):
                team_game_dates[tid] = sorted(group['GAME_DATE'].unique())

            # For each row, compute opponent's rest days
            opponent_rest = pd.Series(4.0, index=df.index, dtype=float)

            for idx, row in df.iterrows():
                opp_id = row.get('OPPONENT_ID')
                game_date = row['GAME_DATE']
                if pd.isna(opp_id) or opp_id not in team_game_dates:
                    continue
                opp_dates = team_game_dates[opp_id]
                # Find the most recent game date before current for the opponent
                prev_dates = [d for d in opp_dates if d < game_date]
                if prev_dates:
                    opp_rest_days = (game_date - prev_dates[-1]).total_seconds() / 86400.0
                    opponent_rest.iloc[idx] = opp_rest_days

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