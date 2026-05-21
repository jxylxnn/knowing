"""Season phase feature group — early-season ramp-up, trade deadline resets."""

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

OUTPUT_COLUMNS = [
    'DAYS_SINCE_SEASON_START',
    'IS_SEASON_OPENER',
    'GAMES_WITH_CURRENT_TEAM',
    'IS_RECENT_TRADE',
]


def _concat_new_columns(df: pd.DataFrame, new_columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Append a batch of aligned columns in a single concat."""
    if not new_columns:
        return df
    new_df = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df, new_df], axis=1)


class SeasonPhaseFeatureGroup(FeatureGroup):
    """Captures early-season ramp-up and trade-deadline context resets.

    Output columns:
        DAYS_SINCE_SEASON_START  — days since Oct 20 of the season year, capped at 30
        IS_SEASON_OPENER         — binary: first 2 days of the season
        GAMES_WITH_CURRENT_TEAM  — consecutive games with current team (resets on trade)
        IS_RECENT_TRADE          — binary: <= 5 games with current team
    """

    @property
    def name(self) -> str:
        return 'season_phase'

    @property
    def required_columns(self) -> List[str]:
        return ['GAME_DATE', 'PLAYER_ID']

    @property
    def optional_columns(self) -> List[str]:
        return ['SEASON_ID', 'TEAM_ID']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        new_columns: dict[str, pd.Series] = {}

        # --- DAYS_SINCE_SEASON_START: capped at 30 to isolate early-season ramp-up ---
        if 'SEASON_ID' in df.columns:
            # e.g., "2024-25" -> 2024
            start_year = df['SEASON_ID'].astype(str).str[:4].astype(int)
        else:
            # If month is before September, the season started the previous calendar year
            start_year = df['GAME_DATE'].dt.year - (df['GAME_DATE'].dt.month < 9).astype(int)

        season_start_date = pd.to_datetime(start_year.astype(str) + '-10-20')
        days_since_start = (df['GAME_DATE'] - season_start_date).dt.days
        days_capped = np.clip(days_since_start, 0, 30)
        new_columns['DAYS_SINCE_SEASON_START'] = days_capped.astype(float)

        new_columns['IS_SEASON_OPENER'] = (days_capped <= 2).astype(float)

        # --- GAMES_WITH_CURRENT_TEAM / IS_RECENT_TRADE ---
        if 'TEAM_ID' in df.columns:
            df_sorted = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
            prev_team = df_sorted.groupby('PLAYER_ID')['TEAM_ID'].shift(1)
            team_changed = (df_sorted['TEAM_ID'] != prev_team).fillna(False).astype(int)

            # Count games with current team (resets on trade)
            team_change_group = team_changed.cumsum()
            games_with_team = df_sorted.groupby(['PLAYER_ID', team_change_group]).cumcount() + 1

            new_columns['GAMES_WITH_CURRENT_TEAM'] = games_with_team.astype(float)
            new_columns['IS_RECENT_TRADE'] = (games_with_team <= 5).astype(float)
        else:
            new_columns['GAMES_WITH_CURRENT_TEAM'] = pd.Series(10.0, index=df.index)
            new_columns['IS_RECENT_TRADE'] = pd.Series(0.0, index=df.index)

        return _concat_new_columns(df, new_columns)
