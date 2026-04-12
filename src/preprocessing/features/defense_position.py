"""Defense position feature group — opponent defensive stats allowed by position group (guard/wing/big)."""

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


# Map ARCHETYPE_ID to position group
# 0,1 → "guard", 2,4 → "wing", 3,5 → "big"
_ARCHETYPE_TO_POSITION = {
    0: 'guard',
    1: 'guard',
    2: 'wing',
    3: 'big',
    4: 'wing',
    5: 'big',
}


class DefensePositionFeatureGroup(FeatureGroup):
    """Quantifies how opponents defend against players of each position group.

    Depends on PlayerArchetypeFeatureGroup for ARCHETYPE_ID to determine
    position group (guard/wing/big). If ARCHETYPE_ID is not available,
    infers position from player's MIN and REB patterns, defaulting to "wing".

    Output columns:
        DEF_POS_PTS_ALLOWED          — opponent's avg PTS allowed to this position group
        DEF_POS_REB_ALLOWED          — opponent's avg REB allowed to this position group
        DEF_POS_AST_ALLOWED          — opponent's avg AST allowed to this position group
        DEF_POS_STL_ALLOWED          — opponent's avg STL allowed to this position group
        DEF_POS_BLK_ALLOWED          — opponent's avg BLK allowed to this position group
        DEF_POS_TOV_ALLOWED          — opponent's avg TOV allowed to this position group
        DEF_POS_RANK                 — opponent's defensive rank against this position group (1=best)
        DEF_POS_RECENT_PTS_ALLOWED   — opponent's recent (last 200 matchups) PTS allowed to this position group
    """

    @property
    def name(self) -> str:
        return 'defense_position'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'OPPONENT_ID']

    @property
    def optional_columns(self) -> List[str]:
        return ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'ARCHETYPE_ID']

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

        if 'OPPONENT_ID' not in df.columns:
            # No opponent data — fill all outputs with league priors
            for stat, prior in [('PTS', ctx.league_priors.get('PTS', 10.0)),
                                ('REB', ctx.league_priors.get('REB', 4.5)),
                                ('AST', ctx.league_priors.get('AST', 2.5)),
                                ('STL', ctx.league_priors.get('STL', 0.8)),
                                ('BLK', ctx.league_priors.get('BLK', 0.6)),
                                ('TOV', ctx.league_priors.get('TOV', 1.5))]:
                df[f'DEF_POS_{stat}_ALLOWED'] = prior
            df['DEF_POS_RANK'] = 15.0
            df['DEF_POS_RECENT_PTS_ALLOWED'] = ctx.league_priors.get('PTS', 10.0)
            return df

        # Sort by player and date for correct rolling order
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # ----------------------------------------------------------------
        # Step 1: Determine position group for each player
        # ----------------------------------------------------------------
        df['_pos_group'] = self._assign_position_groups(df)

        # ----------------------------------------------------------------
        # Step 2: Compute opponent defense by position group
        #   For each (OPPONENT_ID, position_group), compute rolling averages
        #   of stats allowed. Use shift(1) for past-only data.
        # ----------------------------------------------------------------
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        stat_defaults = {
            'PTS': ctx.league_priors.get('PTS', 10.0),
            'REB': ctx.league_priors.get('REB', 4.5),
            'AST': ctx.league_priors.get('AST', 2.5),
            'STL': ctx.league_priors.get('STL', 0.8),
            'BLK': ctx.league_priors.get('BLK', 0.6),
            'TOV': ctx.league_priors.get('TOV', 1.5),
        }

        # Ensure stat columns exist with defaults
        for stat in stats:
            if stat not in df.columns:
                df[stat] = stat_defaults[stat]
            else:
                df[stat] = df[stat].fillna(stat_defaults[stat])

        # Sort by opponent and date for correct rolling computation
        df_sorted_opp = df.sort_values(['OPPONENT_ID', '_pos_group', 'GAME_DATE']).reset_index(drop=False)
        original_index = df_sorted_opp['index'].values

        # Group by (OPPONENT_ID, _pos_group) and compute rolling averages
        opp_pos_group = df_sorted_opp.groupby(['OPPONENT_ID', '_pos_group'], sort=False)

        for stat in stats:
            prior = stat_defaults[stat]
            shifted = opp_pos_group[stat].shift(1)
            # Expanding mean of past values
            expanding_avg = shifted.groupby([df_sorted_opp['OPPONENT_ID'], df_sorted_opp['_pos_group']]).transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
            # Map back to original index
            expanding_avg.index = original_index
            col_name = f'DEF_POS_{stat}_ALLOWED'
            new_columns[col_name] = fill_series_with_prior(
                expanding_avg, prior, diagnostics, col_name
            )

        # ----------------------------------------------------------------
        # Step 3: DEF_POS_RECENT_PTS_ALLOWED
        #   Rolling average of PTS allowed over last 200 matchups per
        #   (OPPONENT_ID, position_group), shifted by 1.
        # ----------------------------------------------------------------
        pts_shifted = opp_pos_group['PTS'].shift(1)
        recent_pts = pts_shifted.groupby([df_sorted_opp['OPPONENT_ID'], df_sorted_opp['_pos_group']]).transform(
            lambda x: x.rolling(200, min_periods=5).mean()
        )
        recent_pts.index = original_index
        new_columns['DEF_POS_RECENT_PTS_ALLOWED'] = fill_series_with_prior(
            recent_pts, stat_defaults['PTS'], diagnostics, 'DEF_POS_RECENT_PTS_ALLOWED'
        )

        # ----------------------------------------------------------------
        # Step 4: DEF_POS_RANK
        #   Rank opponents by PTS allowed to this position group.
        #   Lower PTS allowed = better defense = rank 1.
        #   Use rolling rank over last 2000 games.
        # ----------------------------------------------------------------
        # Compute per-opponent average PTS allowed per position group (shifted)
        # Then rank within each position group
        pts_allowed_col = new_columns.get('DEF_POS_PTS_ALLOWED')
        if pts_allowed_col is not None:
            # Rank opponents within each position group by their avg PTS allowed
            # Lower PTS allowed → better defense → rank 1
            rank_series = pd.Series(np.nan, index=df.index, dtype=float)
            for pos_group in df['_pos_group'].unique():
                if pd.isna(pos_group):
                    continue
                mask = df['_pos_group'] == pos_group
                if mask.sum() == 0:
                    continue
                # Get unique opponents and their average PTS allowed
                opp_avg_pts = df.loc[mask].groupby('OPPONENT_ID')['DEF_POS_PTS_ALLOWED'] if 'DEF_POS_PTS_ALLOWED' in df.columns else None
                # Use the computed column for ranking
                pos_data = df.loc[mask].copy()
                pos_data['_pts_allowed'] = pts_allowed_col.loc[mask] if hasattr(pts_allowed_col, 'loc') else stat_defaults['PTS']

                # Compute per-opponent average for this position group
                opp_means = pos_data.groupby('OPPONENT_ID')['_pts_allowed'].mean()
                # Rank: lower PTS allowed = better defense = rank 1
                opp_ranks = opp_means.rank(method='min', ascending=True)

                # Map rank back to each row
                for idx in pos_data.index:
                    opp_id = pos_data.loc[idx, 'OPPONENT_ID']
                    if pd.notna(opp_id) and opp_id in opp_ranks.index:
                        rank_series.iloc[idx] = opp_ranks[opp_id]

            new_columns['DEF_POS_RANK'] = fill_series_with_prior(
                rank_series, 15.0, diagnostics, 'DEF_POS_RANK'
            )
        else:
            new_columns['DEF_POS_RANK'] = 15.0

        # Clean up temporary columns
        df.drop(columns=['_pos_group'], inplace=True, errors='ignore')

        return _concat_new_columns(df, new_columns)

    def _assign_position_groups(self, df: pd.DataFrame) -> pd.Series:
        """Assign position group (guard/wing/big) to each row.

        Uses ARCHETYPE_ID if available. Falls back to inferring from
        MIN and REB patterns. Defaults to 'wing'.
        """
        pos_groups = pd.Series('wing', index=df.index, dtype='object')

        if 'ARCHETYPE_ID' in df.columns:
            # Map archetype IDs to position groups
            for arch_id, pos in _ARCHETYPE_TO_POSITION.items():
                mask = df['ARCHETYPE_ID'] == arch_id
                pos_groups.loc[mask] = pos

            # Handle any unmapped or NaN archetype IDs
            na_mask = df['ARCHETYPE_ID'].isna() | ~df['ARCHETYPE_ID'].isin(_ARCHETYPE_TO_POSITION.keys())
            if na_mask.any():
                pos_groups.loc[na_mask] = self._infer_position_from_stats(df, na_mask)
        else:
            # No ARCHETYPE_ID — infer from stats
            pos_groups = self._infer_position_from_stats(df, pd.Series(True, index=df.index))

        return pos_groups

    def _infer_position_from_stats(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Infer position group from MIN and REB patterns.

        Heuristic:
        - High REB + high MIN → "big"
        - Low REB + high MIN → "guard"
        - Everything else → "wing"
        """
        result = pd.Series('wing', index=df.index, dtype='object')

        if 'REB' in df.columns and 'MIN' in df.columns:
            reb_vals = df['REB'].fillna(4.5)
            min_vals = df['MIN'].fillna(24.0)

            # Use rolling averages if available for more stable inference
            reb_avg = df.groupby('PLAYER_ID')['REB'].transform(
                lambda x: x.shift(1).rolling(20, min_periods=3).mean()
            ).fillna(4.5)
            min_avg = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.shift(1).rolling(20, min_periods=3).mean()
            ).fillna(24.0)

            # Big: high rebounders (REB > 7 per game)
            big_mask = mask & (reb_avg > 7.0)
            # Guard: low rebounders with significant minutes (REB < 4.5, MIN > 20)
            guard_mask = mask & (reb_avg < 4.5) & (min_avg > 20.0)
            # Wing: everything else

            result.loc[big_mask] = 'big'
            result.loc[guard_mask] = 'guard'
            # Default is already 'wing'
        else:
            # No stats available — default all to wing
            pass

        return result