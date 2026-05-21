"""Team motivation feature group — late-season tanking, load management, playoff clinching."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
)

OUTPUT_COLUMNS = [
    'TEAM_CUMULATIVE_WIN_PCT',
    'IS_LATE_SEASON',
    'IS_TANKING_PROXY',
    'IS_PLAYOFF_LOCK_PROXY',
]


def _concat_new_columns(df: pd.DataFrame, new_columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Append a batch of aligned columns in a single concat."""
    if not new_columns:
        return df
    new_df = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df, new_df], axis=1)


class TeamMotivationFeatureGroup(FeatureGroup):
    """Captures late-season motivation shifts (tanking, load management, playoff locks).

    Output columns:
        TEAM_CUMULATIVE_WIN_PCT  — rolling win % for the team (up to and including prior game)
        IS_LATE_SEASON            — binary: game date is March or later
        IS_TANKING_PROXY          — binary: late season + team win % < 0.35
        IS_PLAYOFF_LOCK_PROXY     — binary: late season + team win % > 0.65
    """

    @property
    def name(self) -> str:
        return 'team_motivation'

    @property
    def required_columns(self) -> List[str]:
        return ['TEAM_ID', 'WL', 'GAME_DATE']

    @property
    def optional_columns(self) -> List[str]:
        return []

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # --- TEAM_CUMULATIVE_WIN_PCT ---
        is_win = (df['WL'] == 'W').astype(float)
        cum_wins = df.groupby('TEAM_ID')['WL'].transform(
            lambda x: (x == 'W').cumsum()
        )
        cum_games = df.groupby('TEAM_ID').cumcount() + 1
        win_pct = cum_wins / cum_games

        # Shift by 1 to prevent leakage: the model sees win % BEFORE the current game
        win_pct_shifted = win_pct.groupby(df['TEAM_ID']).shift(1)
        new_columns['TEAM_CUMULATIVE_WIN_PCT'] = win_pct_shifted.fillna(0.5)

        # --- IS_LATE_SEASON: March onwards ---
        month = df['GAME_DATE'].dt.month
        is_late = (month >= 3).astype(float)
        new_columns['IS_LATE_SEASON'] = is_late

        # --- IS_TANKING_PROXY: late season + poor record ---
        new_columns['IS_TANKING_PROXY'] = (
            (is_late == 1) & (new_columns['TEAM_CUMULATIVE_WIN_PCT'] < 0.35)
        ).astype(float)

        # --- IS_PLAYOFF_LOCK_PROXY: late season + strong record ---
        new_columns['IS_PLAYOFF_LOCK_PROXY'] = (
            (is_late == 1) & (new_columns['TEAM_CUMULATIVE_WIN_PCT'] > 0.65)
        ).astype(float)

        return _concat_new_columns(df, new_columns)
