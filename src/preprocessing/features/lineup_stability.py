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
        is_starter = (df['MIN'] >= 25).astype(float)
        starter_rate = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).apply(
                lambda s: (s >= 25).mean(), raw=False
            )
        )
        col_name = 'LINEUP_STARTER_RATE_10'
        new_columns[col_name] = fill_series_with_prior(starter_rate, 0.5, diagnostics, col_name)

        # --- Precompute teammate sets per (TEAM_ID, GAME_DATE) ---
        # Build mapping: (TEAM_ID, GAME_DATE) -> set of PLAYER_IDs who played (MIN > 0)
        played_mask = df['MIN'] > 0
        teammate_map: dict[tuple, set] = {}
        for (tid, gdate), group in df[played_mask].groupby(['TEAM_ID', 'GAME_DATE']):
            teammate_map[(tid, gdate)] = set(group['PLAYER_ID'].values)

        # --- LINEUP_TEAM_STABILITY_5 and LINEUP_TEAM_STABILITY_10 ---
        # Jaccard similarity of teammate sets across consecutive game transitions
        def _compute_jaccard_stability(player_games: pd.DataFrame, window: int) -> pd.Series:
            """For each row, compute avg Jaccard similarity of teammate sets over last `window` transitions."""
            results = []
            for _idx, row in player_games.iterrows():
                tid = row['TEAM_ID']
                gdate = row['GAME_DATE']
                # Get this player's game dates for this team, sorted
                player_team_games = player_games[
                    (player_games['TEAM_ID'] == tid)
                ].sort_values('GAME_DATE')
                # Find games before current date
                prior_games = player_team_games[player_team_games['GAME_DATE'] < gdate]['GAME_DATE'].values[-window:]
                if len(prior_games) < 1:
                    results.append(np.nan)
                    continue
                jaccard_vals = []
                for i in range(len(prior_games)):
                    g = prior_games[i]
                    prev_games = player_team_games[player_team_games['GAME_DATE'] < g]['GAME_DATE'].values
                    if len(prev_games) == 0:
                        continue
                    prev_g = prev_games[-1]
                    set_curr = teammate_map.get((tid, g), set())
                    set_prev = teammate_map.get((tid, prev_g), set())
                    if len(set_curr) == 0 and len(set_prev) == 0:
                        jaccard_vals.append(1.0)
                    elif len(set_curr) == 0 or len(set_prev) == 0:
                        jaccard_vals.append(0.0)
                    else:
                        intersection = len(set_curr & set_prev)
                        union = len(set_curr | set_prev)
                        jaccard_vals.append(intersection / union if union > 0 else 0.0)
                results.append(np.mean(jaccard_vals) if jaccard_vals else np.nan)
            return pd.Series(results, index=player_games.index)

        # Vectorized approach: compute per-player Jaccard stability
        # For efficiency, we compute a shifted version using precomputed teammate sets
        # Map each row to its team+date, then look up prior game's teammate set
        df['_team_date_key'] = list(zip(df['TEAM_ID'], df['GAME_DATE']))

        # For each player, get their sorted game dates and compute Jaccard for consecutive pairs
        jaccard_series = pd.Series(np.nan, index=df.index, dtype=float)

        for player_id, player_df in df.groupby('PLAYER_ID'):
            player_idx = player_df.index
            player_dates = player_df['GAME_DATE'].values
            player_teams = player_df['TEAM_ID'].values

            # For each game, compute Jaccard with previous game's teammate set
            jaccard_vals = []
            for i in range(len(player_dates)):
                curr_key = (player_teams[i], player_dates[i])
                if i == 0:
                    jaccard_vals.append(np.nan)
                    continue
                prev_key = (player_teams[i - 1], player_dates[i - 1])
                set_curr = teammate_map.get(curr_key, set())
                set_prev = teammate_map.get(prev_key, set())
                if len(set_curr) == 0 and len(set_prev) == 0:
                    jaccard_vals.append(1.0)
                elif len(set_curr) == 0 or len(set_prev) == 0:
                    jaccard_vals.append(0.0)
                else:
                    intersection = len(set_curr & set_prev)
                    union = len(set_curr | set_prev)
                    jaccard_vals.append(intersection / union if union > 0 else 0.0)

            jaccard_series.loc[player_idx] = jaccard_vals

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
        # Precompute roster size per (TEAM_ID, GAME_DATE)
        roster_size_map: dict[tuple, int] = {}
        for (tid, gdate), group in df[played_mask].groupby(['TEAM_ID', 'GAME_DATE']):
            roster_size_map[(tid, gdate)] = len(group)

        # Map roster size to each row
        df['_roster_size'] = df.apply(
            lambda row: roster_size_map.get((row['TEAM_ID'], row['GAME_DATE']), np.nan), axis=1
        )

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
        df.drop(columns=['_team_date_key', '_roster_size', '_min_rank'], inplace=True, errors='ignore')

        return _concat_new_columns(df, new_columns)