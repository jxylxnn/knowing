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
from src.preprocessing.features._teammate_utils import TeammateContext


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
        # Shared teammate context (replaces duplicated roster maps)
        # ----------------------------------------------------------------
        tctx = TeammateContext(df)
        game_roster = tctx.game_roster_map
        regular_teammates = tctx.regular_teammates_map
        high_usage_teammates = tctx.high_usage_teammates_map

        # ----------------------------------------------------------------
        # Compute per-row features
        # ----------------------------------------------------------------
        # Vectorized lookups using Series map
        df['_current_roster'] = pd.Series(
            [game_roster.get((tid, gd), set()) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
            index=df.index,
        )
        df['_regulars'] = pd.Series(
            [regular_teammates.get((tid, gd), set()) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
            index=df.index,
        )
        df['_high_usage'] = pd.Series(
            [high_usage_teammates.get((tid, gd), set()) for tid, gd in zip(df['TEAM_ID'], df['GAME_DATE'])],
            index=df.index,
        )

        # Missing regular teammates (excluding self)
        df['_missing_regulars'] = [
            regs - curr - {pid}
            for regs, curr, pid in zip(df['_regulars'], df['_current_roster'], df['PLAYER_ID'])
        ]

        # INJURY_OPP_MISSING_HIGH_USAGE
        missing_high_usage = pd.Series(0.0, index=df.index, dtype=float)
        for idx in df.index:
            missing = df.loc[idx, '_missing_regulars']
            high = df.loc[idx, '_high_usage']
            missing_high_usage.iloc[idx] = 1.0 if (missing & high) else 0.0

        # INJURY_OPP_MISSING_SAME_POS
        # Compute per-player average MIN for "same position" check
        player_avg_min = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

        missing_same_pos = pd.Series(0.0, index=df.index, dtype=float)
        # Precompute global average MIN per player for fast lookup
        global_avg_min = df.groupby('PLAYER_ID')['MIN'].mean().to_dict()

        for idx in df.index:
            missing_regulars = df.loc[idx, '_missing_regulars']
            my_avg_min = player_avg_min.iloc[idx] if not pd.isna(player_avg_min.iloc[idx]) else ctx.league_priors.get('MIN', 24.0)
            similar_min_count = 0
            for missing_pid in missing_regulars:
                missing_avg = global_avg_min.get(missing_pid, ctx.league_priors.get('MIN', 24.0))
                if abs(missing_avg - my_avg_min) <= 5.0:
                    similar_min_count += 1
            missing_same_pos.iloc[idx] = float(similar_min_count)

        # Shift to prevent leakage
        missing_high_usage_shifted = missing_high_usage.groupby(df['PLAYER_ID']).shift(1).fillna(0.0)
        missing_same_pos_shifted = missing_same_pos.groupby(df['PLAYER_ID']).shift(1).fillna(0.0)

        new_columns['INJURY_OPP_MISSING_HIGH_USAGE'] = missing_high_usage_shifted
        new_columns['INJURY_OPP_MISSING_SAME_POS'] = missing_same_pos_shifted

        # ----------------------------------------------------------------
        # Step 5: INJURY_OPP_MIN_BOOST and INJURY_OPP_USAGE_BOOST
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
        fga_available = 'FGA' in df.columns
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
        # ----------------------------------------------------------------
        team_absences = pd.Series(
            [float(len(m)) for m in df['_missing_regulars']],
            index=df.index,
            dtype=float,
        )
        team_absences_shifted = team_absences.groupby(df['PLAYER_ID']).shift(1)
        team_absences_rolling = team_absences_shifted.groupby(df['PLAYER_ID']).transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        new_columns['INJURY_OPP_TEAM_ABSENCES_5'] = fill_series_with_prior(
            team_absences_rolling, 0.0, diagnostics, 'INJURY_OPP_TEAM_ABSENCES_5'
        )

        # Clean up temporary columns
        df.drop(
            columns=['_current_roster', '_regulars', '_high_usage', '_missing_regulars'],
            inplace=True,
            errors='ignore',
        )

        return _concat_new_columns(df, new_columns)
