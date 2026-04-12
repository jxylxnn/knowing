"""Teammate usage feature group — active/missing teammate usage shares and scoring depth."""

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


class TeammateUsageFeatureGroup(FeatureGroup):
    """Quantifies the impact of teammate availability on usage and scoring opportunity.

    Output columns:
        TEAMMATE_TOP_USAGE_ACTIVE      — binary: is the highest-usage teammate active?
        TEAMMATE_MISSING_USAGE_SHARE   — total FGA share of missing regular teammates (shifted)
        TEAMMATE_MISSING_AST_SHARE     — total AST share of missing regular teammates (shifted)
        TEAMMATE_MISSING_REB_SHARE     — total REB share of missing regular teammates (shifted)
        TEAMMATE_MISSING_SHOT_VOLUME   — total FGA/game of missing regular teammates (shifted)
        TEAMMATE_ACTIVE_SCORING_DEPTH  — number of active teammates averaging >10 PTS/game (shifted)
    """

    @property
    def name(self) -> str:
        return 'teammate_usage'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'TEAM_ID', 'MIN']

    @property
    def optional_columns(self) -> List[str]:
        return ['FGA', 'AST', 'REB', 'FG3A', 'PTS']

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
        # Step 2: Identify "regular" teammates per team per game
        #   A player is "regular" if they appeared in >50% of the team's
        #   last 20 games.
        # ----------------------------------------------------------------
        team_games: dict = {}
        for tid, group in df[played_mask].groupby('TEAM_ID'):
            team_games[tid] = sorted(group['GAME_DATE'].unique())

        regular_teammates: dict[tuple, set] = {}

        for tid, dates in team_games.items():
            for i, gdate in enumerate(dates):
                window_dates = dates[max(0, i - 19): i + 1]
                player_counts: dict = {}
                for wd in window_dates:
                    roster = game_roster.get((tid, wd), set())
                    for pid in roster:
                        player_counts[pid] = player_counts.get(pid, 0) + 1

                threshold = len(window_dates) * 0.5
                regulars = {pid for pid, cnt in player_counts.items() if cnt > threshold}
                regular_teammates[(tid, gdate)] = regulars

        # ----------------------------------------------------------------
        # Step 3: Compute per-player usage and shares per game
        #   Usage = FGA / team_FGA. Shares = player stat / team total.
        # ----------------------------------------------------------------
        fga_available = 'FGA' in df.columns
        ast_available = 'AST' in df.columns
        reb_available = 'REB' in df.columns
        pts_available = 'PTS' in df.columns

        # Compute team totals per game
        team_totals: dict[tuple, dict] = {}
        for (tid, gdate), group in df[played_mask].groupby(['TEAM_ID', 'GAME_DATE']):
            totals = {}
            totals['FGA'] = group['FGA'].sum() if fga_available else 1.0
            totals['AST'] = group['AST'].sum() if ast_available else 1.0
            totals['REB'] = group['REB'].sum() if reb_available else 1.0
            totals['PTS'] = group['PTS'].sum() if pts_available else 1.0
            # Avoid zero denominators
            for key in totals:
                totals[key] = max(totals[key], 1.0)
            team_totals[(tid, gdate)] = totals

        # Compute per-player shares per game
        df['_fga_share'] = 0.0
        df['_ast_share'] = 0.0
        df['_reb_share'] = 0.0
        df['_fga_per_game'] = 0.0
        df['_pts_per_game'] = 0.0

        for idx, row in df.iterrows():
            tid = row['TEAM_ID']
            gdate = row['GAME_DATE']
            totals = team_totals.get((tid, gdate), {'FGA': 1.0, 'AST': 1.0, 'REB': 1.0, 'PTS': 1.0})

            if fga_available and pd.notna(row.get('FGA', np.nan)):
                df.loc[idx, '_fga_share'] = row['FGA'] / totals['FGA']
                df.loc[idx, '_fga_per_game'] = row['FGA']

            if ast_available and pd.notna(row.get('AST', np.nan)):
                df.loc[idx, '_ast_share'] = row['AST'] / totals['AST']

            if reb_available and pd.notna(row.get('REB', np.nan)):
                df.loc[idx, '_reb_share'] = row['REB'] / totals['REB']

            if pts_available and pd.notna(row.get('PTS', np.nan)):
                df.loc[idx, '_pts_per_game'] = row['PTS']

        # ----------------------------------------------------------------
        # Step 4: Determine highest-usage regular teammate per player
        #   For each player, find the regular teammate with the highest
        #   rolling FGA share (or PTS if FGA unavailable).
        # ----------------------------------------------------------------
        # Compute rolling average usage per player (shifted)
        if fga_available:
            usage_col = '_fga_per_game'
        elif pts_available:
            usage_col = '_pts_per_game'
        else:
            usage_col = 'MIN'

        player_avg_usage = df.groupby('PLAYER_ID')[usage_col].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

        # For each player, find the top-usage regular teammate
        # Build a lookup: PLAYER_ID → (highest usage regular teammate ID)
        # We compute this per (TEAM_ID, GAME_DATE) context

        # ----------------------------------------------------------------
        # Step 5: Compute per-row features
        # ----------------------------------------------------------------
        top_usage_active = pd.Series(0.0, index=df.index, dtype=float)
        missing_usage_share = pd.Series(0.0, index=df.index, dtype=float)
        missing_ast_share = pd.Series(0.0, index=df.index, dtype=float)
        missing_reb_share = pd.Series(0.0, index=df.index, dtype=float)
        missing_shot_volume = pd.Series(0.0, index=df.index, dtype=float)
        active_scoring_depth = pd.Series(0.0, index=df.index, dtype=float)

        # Precompute per-player rolling averages for PTS (for scoring depth)
        if pts_available:
            player_avg_pts = df.groupby('PLAYER_ID')['PTS'].transform(
                lambda x: x.shift(1).rolling(20, min_periods=3).mean()
            )
        else:
            player_avg_pts = pd.Series(ctx.league_priors.get('PTS', 10.0), index=df.index, dtype=float)

        # Precompute per-player rolling averages for shares
        player_avg_fga_share = df.groupby('PLAYER_ID')['_fga_share'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )
        player_avg_ast_share = df.groupby('PLAYER_ID')['_ast_share'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )
        player_avg_reb_share = df.groupby('PLAYER_ID')['_reb_share'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )
        player_avg_fga = df.groupby('PLAYER_ID')[usage_col].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

        # Build per-player average usage lookup
        avg_usage_lookup: dict = {}
        for pid, group in df.groupby('PLAYER_ID'):
            avg_usage_lookup[pid] = player_avg_usage.loc[group.index].mean()

        # Build per-player average PTS lookup
        avg_pts_lookup: dict = {}
        if pts_available:
            for pid, group in df.groupby('PLAYER_ID'):
                avg_pts_lookup[pid] = player_avg_pts.loc[group.index].mean()

        for idx, row in df.iterrows():
            tid = row['TEAM_ID']
            gdate = row['GAME_DATE']
            pid = row['PLAYER_ID']

            current_roster = game_roster.get((tid, gdate), set())
            regulars = regular_teammates.get((tid, gdate), set())

            # Missing regular teammates (not in current game, excluding self)
            missing_regulars = (regulars - current_roster) - {pid}
            active_regulars = (regulars & current_roster) - {pid}

            # --- TEAMMATE_TOP_USAGE_ACTIVE ---
            # Find the highest-usage regular teammate (by avg usage)
            if regulars:
                # Exclude self from regulars for top-usage check
                other_regulars = regulars - {pid}
                if other_regulars:
                    top_usage_pid = max(other_regulars, key=lambda p: avg_usage_lookup.get(p, 0.0))
                    top_usage_active.iloc[idx] = 1.0 if top_usage_pid in current_roster else 0.0
                else:
                    top_usage_active.iloc[idx] = 0.0
            else:
                top_usage_active.iloc[idx] = 0.0

            # --- Missing teammate shares ---
            for missing_pid in missing_regulars:
                missing_usage_share.iloc[idx] += player_avg_fga_share.loc[
                    df['PLAYER_ID'] == missing_pid
                ].mean() if fga_available else 0.0
                missing_ast_share.iloc[idx] += player_avg_ast_share.loc[
                    df['PLAYER_ID'] == missing_pid
                ].mean() if ast_available else 0.0
                missing_reb_share.iloc[idx] += player_avg_reb_share.loc[
                    df['PLAYER_ID'] == missing_pid
                ].mean() if reb_available else 0.0
                missing_shot_volume.iloc[idx] += player_avg_fga.loc[
                    df['PLAYER_ID'] == missing_pid
                ].mean() if fga_available else 0.0

            # --- TEAMMATE_ACTIVE_SCORING_DEPTH ---
            # Count active regular teammates with avg PTS > 10
            if pts_available:
                for active_pid in active_regulars:
                    if avg_pts_lookup.get(active_pid, 0.0) > 10.0:
                        active_scoring_depth.iloc[idx] += 1.0

        # Shift all features to prevent leakage
        top_usage_active_shifted = top_usage_active.groupby(df['PLAYER_ID']).shift(1).fillna(0.0)
        missing_usage_share_shifted = missing_usage_share.groupby(df['PLAYER_ID']).shift(1)
        missing_ast_share_shifted = missing_ast_share.groupby(df['PLAYER_ID']).shift(1)
        missing_reb_share_shifted = missing_reb_share.groupby(df['PLAYER_ID']).shift(1)
        missing_shot_volume_shifted = missing_shot_volume.groupby(df['PLAYER_ID']).shift(1)
        active_scoring_depth_shifted = active_scoring_depth.groupby(df['PLAYER_ID']).shift(1)

        new_columns['TEAMMATE_TOP_USAGE_ACTIVE'] = top_usage_active_shifted
        new_columns['TEAMMATE_MISSING_USAGE_SHARE'] = fill_series_with_prior(
            missing_usage_share_shifted, 0.0, diagnostics, 'TEAMMATE_MISSING_USAGE_SHARE'
        )
        new_columns['TEAMMATE_MISSING_AST_SHARE'] = fill_series_with_prior(
            missing_ast_share_shifted, 0.0, diagnostics, 'TEAMMATE_MISSING_AST_SHARE'
        )
        new_columns['TEAMMATE_MISSING_REB_SHARE'] = fill_series_with_prior(
            missing_reb_share_shifted, 0.0, diagnostics, 'TEAMMATE_MISSING_REB_SHARE'
        )
        new_columns['TEAMMATE_MISSING_SHOT_VOLUME'] = fill_series_with_prior(
            missing_shot_volume_shifted, 0.0, diagnostics, 'TEAMMATE_MISSING_SHOT_VOLUME'
        )
        new_columns['TEAMMATE_ACTIVE_SCORING_DEPTH'] = fill_series_with_prior(
            active_scoring_depth_shifted, 0.0, diagnostics, 'TEAMMATE_ACTIVE_SCORING_DEPTH'
        )

        # Clean up temporary columns
        df.drop(columns=['_fga_share', '_ast_share', '_reb_share', '_fga_per_game', '_pts_per_game'],
                inplace=True, errors='ignore')

        return _concat_new_columns(df, new_columns)