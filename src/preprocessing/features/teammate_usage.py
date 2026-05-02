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
from src.preprocessing.features._teammate_utils import TeammateContext, build_team_totals_map


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
        # Shared teammate context
        # ----------------------------------------------------------------
        tctx = TeammateContext(df)
        game_roster = tctx.game_roster_map
        regular_teammates = tctx.regular_teammates_map

        # ----------------------------------------------------------------
        # Step 1: Compute per-player usage and shares per game (vectorized)
        # ----------------------------------------------------------------
        fga_available = 'FGA' in df.columns
        ast_available = 'AST' in df.columns
        reb_available = 'REB' in df.columns
        pts_available = 'PTS' in df.columns

        # Team totals per game via shared utility
        team_totals = build_team_totals_map(df, ['FGA', 'AST', 'REB', 'PTS'])

        # Vectorized share computation using map + groupby transform
        df['_fga_share'] = 0.0
        df['_ast_share'] = 0.0
        df['_reb_share'] = 0.0
        df['_fga_per_game'] = 0.0
        df['_pts_per_game'] = 0.0

        if fga_available:
            df['_team_fga'] = pd.Series(
                [team_totals.get((tid, gd), {}).get('FGA', 1.0) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
                index=df.index,
            )
            df['_fga_share'] = (df['FGA'] / df['_team_fga']).fillna(0.0)
            df['_fga_per_game'] = df['FGA'].fillna(0.0)

        if ast_available:
            df['_team_ast'] = pd.Series(
                [team_totals.get((tid, gd), {}).get('AST', 1.0) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
                index=df.index,
            )
            df['_ast_share'] = (df['AST'] / df['_team_ast']).fillna(0.0)

        if reb_available:
            df['_team_reb'] = pd.Series(
                [team_totals.get((tid, gd), {}).get('REB', 1.0) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
                index=df.index,
            )
            df['_reb_share'] = (df['REB'] / df['_team_reb']).fillna(0.0)

        if pts_available:
            df['_team_pts'] = pd.Series(
                [team_totals.get((tid, gd), {}).get('PTS', 1.0) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
                index=df.index,
            )
            df['_pts_per_game'] = df['PTS'].fillna(0.0)

        # ----------------------------------------------------------------
        # Step 2: Rolling averages per player (shifted)
        # ----------------------------------------------------------------
        if fga_available:
            usage_col = '_fga_per_game'
        elif pts_available:
            usage_col = '_pts_per_game'
        else:
            usage_col = 'MIN'

        player_avg_usage = df.groupby('PLAYER_ID')[usage_col].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

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

        if pts_available:
            player_avg_pts = df.groupby('PLAYER_ID')['PTS'].transform(
                lambda x: x.shift(1).rolling(20, min_periods=3).mean()
            )
        else:
            player_avg_pts = pd.Series(ctx.league_priors.get('PTS', 10.0), index=df.index, dtype=float)

        # Build per-player average usage / pts lookups (mean over all rows for that player)
        avg_usage_lookup = player_avg_usage.groupby(df['PLAYER_ID']).mean().to_dict()
        avg_pts_lookup = player_avg_pts.groupby(df['PLAYER_ID']).mean().to_dict()

        # Build per-player average share lookups
        avg_fga_share_lookup = player_avg_fga_share.groupby(df['PLAYER_ID']).mean().to_dict()
        avg_ast_share_lookup = player_avg_ast_share.groupby(df['PLAYER_ID']).mean().to_dict()
        avg_reb_share_lookup = player_avg_reb_share.groupby(df['PLAYER_ID']).mean().to_dict()
        avg_fga_lookup = player_avg_fga.groupby(df['PLAYER_ID']).mean().to_dict()

        # ----------------------------------------------------------------
        # Step 3: Compute per-row features
        # ----------------------------------------------------------------
        top_usage_active = pd.Series(0.0, index=df.index, dtype=float)
        missing_usage_share = pd.Series(0.0, index=df.index, dtype=float)
        missing_ast_share = pd.Series(0.0, index=df.index, dtype=float)
        missing_reb_share = pd.Series(0.0, index=df.index, dtype=float)
        missing_shot_volume = pd.Series(0.0, index=df.index, dtype=float)
        active_scoring_depth = pd.Series(0.0, index=df.index, dtype=float)

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
            if regulars:
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
                if fga_available:
                    missing_usage_share.iloc[idx] += avg_fga_share_lookup.get(missing_pid, 0.0)
                    missing_shot_volume.iloc[idx] += avg_fga_lookup.get(missing_pid, 0.0)
                if ast_available:
                    missing_ast_share.iloc[idx] += avg_ast_share_lookup.get(missing_pid, 0.0)
                if reb_available:
                    missing_reb_share.iloc[idx] += avg_reb_share_lookup.get(missing_pid, 0.0)

            # --- TEAMMATE_ACTIVE_SCORING_DEPTH ---
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
        drop_cols = [
            '_fga_share', '_ast_share', '_reb_share', '_fga_per_game', '_pts_per_game',
            '_team_fga', '_team_ast', '_team_reb', '_team_pts',
        ]
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return _concat_new_columns(df, new_columns)
