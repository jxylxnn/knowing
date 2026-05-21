"""Postseason context feature group — playoff pace drops and intensity spikes."""

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
    'IS_PLAYOFF_GAME',
    'PLAYOFF_PACE_PRIOR',
]


def _concat_new_columns(df: pd.DataFrame, new_columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Append a batch of aligned columns in a single concat."""
    if not new_columns:
        return df
    new_df = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df, new_df], axis=1)


class PostseasonContextFeatureGroup(FeatureGroup):
    """Flags playoff games and provides a pace-adjustment prior.

    Output columns:
        IS_PLAYOFF_GAME    — binary: 1 if game is a playoff/postseason game
        PLAYOFF_PACE_PRIOR — historical prior: 0.95 for playoff games, 1.0 otherwise
    """

    @property
    def name(self) -> str:
        return 'postseason_context'

    @property
    def required_columns(self) -> List[str]:
        return ['GAME_DATE']

    @property
    def optional_columns(self) -> List[str]:
        return ['SEASON_TYPE', 'GAME_TYPE']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        new_columns: dict[str, pd.Series] = {}

        game_type_col: Optional[str] = None
        if 'SEASON_TYPE' in df.columns:
            game_type_col = 'SEASON_TYPE'
        elif 'GAME_TYPE' in df.columns:
            game_type_col = 'GAME_TYPE'

        if game_type_col:
            val = df[game_type_col].astype(str).str.lower()
            # nba_api uses "Playoffs", "Postseason", or numeric "4"
            is_playoff = val.str.contains('playoff|postseason|4', na=False).astype(float)
        else:
            is_playoff = pd.Series(0.0, index=df.index)

        new_columns['IS_PLAYOFF_GAME'] = is_playoff

        # Historical prior: playoff pace drops by ~5%. The model learns the exact coefficient.
        new_columns['PLAYOFF_PACE_PRIOR'] = np.where(is_playoff == 1, 0.95, 1.0)

        return _concat_new_columns(df, new_columns)
