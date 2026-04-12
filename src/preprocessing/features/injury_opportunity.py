"""Injury-adjusted opportunity feature group — detects missing high-usage teammates and estimates opportunity boost."""

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


class InjuryAdjustedOpportunityFeatureGroup(FeatureGroup):
    """Quantifies the opportunity created by missing teammates due to injury.

    Output columns:
        INJURY_OPP_MISSING_HIGH_USAGE  — binary: is a high-usage teammate (top 3) missing?
        INJURY_OPP_MISSING_SAME_POS   — count of missing regular teammates with similar minutes
        INJURY_OPP_MIN_BOOST          — estimated minutes boost when high-usage teammate missing (shifted)
        INJURY_OPP_USAGE_BOOST        — estimated usage boost when high-usage teammate missing (shifted)
        INJURY_OPP_TEAM_ABSENCES_5    — rolling 5-game count of teammate absences (shifted)
    """

    @property
    def name(self) -> str:
        return 'injury_opportunity'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'TEAM_ID', 'MIN']

    @property
    def optional_columns(self) -> List[str]:
        return ['PTS', 'FGA', 'AST', 'REB']

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

        # ----------------------------------------------------------------
        # Step 1: Build (TEAM_ID, GAME_DATE) → set of PLAYER_IDs mapping
        # ----------------------------------------------------------------
        played_mask = df['MIN'] > 0
        game_roster: dict[tuple, set] = {}
        for (tid, gdate), group in df[played_mask].groupby(['TEAM_ID', 'GAME_DATE']):
            game_roster[(tid, gdate)] = set(group['PLAYER_ID'].values)

        # ----------------------------------------------------------------
        # Step 2: Identify "regular" teammates per team
        #   A player is "regular" if they appeared in >50% of the team's
        #   last 20 games (using expanding window for early games).
        # ----------------------------------------------------------------
        # Get sorted unique game dates per team
        team_games: dict = {}
        for tid, group in df[played_mask].groupby('TEAM_ID'):
            team_games[tid] = sorted(group['GAME_DATE'].unique())

        # For each team, compute game appearance counts per player using rolling windows
        # Build a lookup: (TEAM_ID, GAME_DATE) → set of regular PLAYER_IDs
        regular_teammates: dict[tuple, set] = {}

        for tid, dates in team_games.items():
            # For each date, look at the last 20 games before that date
            for i, gdate in enumerate(dates):
                # Get the last 20 game dates before (or including) this one
                window_dates = dates[max(0, i - 19): i + 1]
                # Count appearances of each player in those games
                player_counts: dict = {}
                for wd in window_dates:
                    roster = game_roster.get((tid, wd), set())
                    for pid in roster:
                        player_counts[pid] = player_counts.get(pid, 0) + 1

                # Regular = appeared in >50% of those games
                threshold = len(window_dates) * 0.5
                regulars = {pid for pid, cnt in player_counts.items() if cnt > threshold}
                regular_teammates[(tid, gdate)] = regulars

        # ----------------------------------------------------------------
        # Step 3: Identify "high usage" teammates per team per game
        #   Usage = FGA / team_FGA over last 20 games. Top 3 are "high usage".
        # ----------------------------------------------------------------
        fga_available = 'FGA' in df.columns
        if not fga_available:
            # Fallback: use PTS as a proxy for usage
            usage_col = 'PTS' if 'PTS' in df.columns else 'MIN'
        else:
            usage_col = 'FGA'

        # Compute per-player rolling usage per team
        # For each (TEAM_ID, GAME_DATE), compute each player's usage share
        high_usage_teammates: dict[tuple, set] = {}

        for tid, dates in team_games.items():
            for i, gdate in enumerate(dates):
                window_dates = dates[max(0, i - 19): i + 1]
                # Sum usage per player over window
                player_usage: dict = {}
                for wd in window_dates:
                    roster = game_roster.get((tid, wd), set())
                    # Get usage values for players in this game
                    mask = (df['TEAM_ID'] == tid) & (df['GAME_DATE'] == wd) & (df['PLAYER_ID'].isin(roster))
                    game_rows = df.loc[mask]
                    for _, row in game_rows.iterrows():
                        pid = row['PLAYER_ID']
                        val = row.get(usage_col, 0)
                        if pd.isna(val):
                            val = 0
                        player_usage[pid] = player_usage.get(pid, 0) + val

                # Top 3 by usage
                sorted_players = sorted(player_usage.items(), key=lambda x: x[1], reverse=True)
                top3 = {pid for pid, _ in sorted_players[:3]}
                high_usage_teammates[(tid, gdate)] = top3

        # ----------------------------------------------------------------
        # Step 4: Compute per-row features
        # ----------------------------------------------------------------
        # For each row, determine which regular teammates are missing
        missing_high_usage = pd.Series(0.0, index=df.index, dtype=float)
        missing_same_pos = pd.Series(0.0, index=df.index, dtype=float)
        team_absences = pd.Series(0.0, index=df.index, dtype=float)

        # Compute per-player average MIN for "same position" check
        player_avg_min = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

        for idx, row in df.iterrows():
            tid = row['TEAM_ID']
            gdate = row['GAME_DATE']
            pid = row['PLAYER_ID']

            current_roster = game_roster.get((tid, gdate), set())
            regulars = regular_teammates.get((tid, gdate), set())
            high_usage = high_usage_teammates.get((tid, gdate), set())

            # Missing regular teammates
            missing_regulars = regulars - current_roster
            # Don't count the player themselves as missing
            missing_regulars = missing_regulars - {pid}

            # INJURY_OPP_MISSING_HIGH_USAGE: is any high-usage regular missing?
            missing_high_usage_regulars = missing_regulars & high_usage
            missing_high_usage.iloc[idx] = 1.0 if len(missing_high_usage_regulars) > 0 else 0.0

            # INJURY_OPP_MISSING_SAME_POS: count missing regulars with similar minutes
            my_avg_min = player_avg_min.iloc[idx] if not pd.isna(player_avg_min.iloc[idx]) else ctx.league_priors.get('MIN', 24.0)
            similar_min_count = 0
            for missing_pid in missing_regulars:
                # Get this missing player's average minutes
                missing_player_rows = df[df['PLAYER_ID'] == missing_pid]
                if len(missing_player_rows) > 0:
                    missing_avg = missing_player_rows['MIN'].mean()
                else:
                    missing_avg = ctx.league_priors.get('MIN', 24.0)
                if abs(missing_avg - my_avg_min) <= 5.0:
                    similar_min_count += 1
            missing_same_pos.iloc[idx] = float(similar_min_count)

            # Count total missing regulars for TEAM_ABSENCES
            team_absences.iloc[idx] = float(len(missing_regulars))

        # Shift to prevent leakage
        missing_high_usage_shifted = missing_high_usage.groupby(df['PLAYER_ID']).shift(1).fillna(0.0)
        missing_same_pos_shifted = missing_same_pos.groupby(df['PLAYER_ID']).shift(1).fillna(0.0)

        new_columns['INJURY_OPP_MISSING_HIGH_USAGE'] = missing_high_usage_shifted
        new_columns['INJURY_OPP_MISSING_SAME_POS'] = missing_same_pos_shifted

        # ----------------------------------------------------------------
        # Step 5: INJURY_OPP_MIN_BOOST and INJURY_OPP_USAGE_BOOST
        #   When high-usage teammate is missing, compute the difference between
        #   this player's MIN in this game and their season avg MIN. Shift by 1.
        # ----------------------------------------------------------------
        # Season average MIN per player (rolling expanding mean, shifted)
        season_avg_min = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).expanding(min_periods=3).mean()
        )

        # MIN boost = current MIN - season avg, only when high-usage teammate missing
        min_boost = (df['MIN'] - season_avg_min).fillna(0.0).clip(-15, 15)
        min_boost = min_boost * missing_high_usage  # Only when flag is 1
        min_boost_shifted = min_boost.groupby(df['PLAYER_ID']).shift(1)
        new_columns['INJURY_OPP_MIN_BOOST'] = fill_series_with_prior(
            min_boost_shifted, 0.0, diagnostics, 'INJURY_OPP_MIN_BOOST'
        )

        # Usage boost: similar but for usage (FGA/game)
        if fga_available:
            season_avg_fga = df.groupby('PLAYER_ID')['FGA'].transform(
                lambda x: x.shift(1).expanding(min_periods=3).mean()
            )
            usage_boost = (df['FGA'] - season_avg_fga).fillna(0.0).clip(-10, 10)
            usage_boost = usage_boost * missing_high_usage
            usage_boost_shifted = usage_boost.groupby(df['PLAYER_ID']).shift(1)
            new_columns['INJURY_OPP_USAGE_BOOST'] = fill_series_with_prior(
                usage_boost_shifted, 0.0, diagnostics, 'INJURY_OPP_USAGE_BOOST'
            )
        else:
            new_columns['INJURY_OPP_USAGE_BOOST'] = 0.0

        # ----------------------------------------------------------------
        # Step 6: INJURY_OPP_TEAM_ABSENCES_5
        #   Rolling 5-game count of teammate absences per game, shifted.
        # ----------------------------------------------------------------
        team_absences_shifted = team_absences.groupby(df['PLAYER_ID']).shift(1)
        team_absences_rolling = team_absences_shifted.groupby(df['PLAYER_ID']).transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        new_columns['INJURY_OPP_TEAM_ABSENCES_5'] = fill_series_with_prior(
            team_absences_rolling, 0.0, diagnostics, 'INJURY_OPP_TEAM_ABSENCES_5'
        )

        return _concat_new_columns(df, new_columns)