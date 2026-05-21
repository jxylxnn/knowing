"""KAN aging feature group — Kolmogorov-Arnold Network nonlinear age curves.

Uses pre-computed KAN outputs (cached to data/cache/kan_aging_outputs.csv)
to inject nonlinear aging features into the feature set.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    normalize_output_columns,
)

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    'KAN_AGE_NONLIN_FACTOR',
    'KAN_AGE_INFLECTION_AGE',
    'KAN_AGE_VOLATILITY',
]


class KANAgingFeatureGroup(FeatureGroup):
    """KAN-based nonlinear aging features.

    Loads pre-computed KAN outputs from cache and joins onto the player DF.
    Falls back to quadratic approximation when no cache exists.
    All features are shift(1) to prevent leakage.
    """

    @property
    def name(self) -> str:
        return 'kan_aging'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE']

    @property
    def optional_columns(self) -> List[str]:
        return ['AGE']

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self._kan_cache: Optional[pd.DataFrame] = None

    def _load_kan_outputs(self) -> pd.DataFrame:
        if self._kan_cache is not None:
            return self._kan_cache
        cache_path = os.path.join(self.data_dir, 'cache', 'kan_aging_outputs.csv')
        if os.path.exists(cache_path):
            try:
                self._kan_cache = pd.read_csv(cache_path)
                return self._kan_cache
            except Exception:
                pass
        self._kan_cache = pd.DataFrame()
        return self._kan_cache

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()

        if 'PLAYER_ID' not in df.columns:
            df = normalize_output_columns(df, OUTPUT_COLUMNS)
            return df

        df['GAME_DATE'] = pd.to_datetime(df.get('GAME_DATE', pd.Timestamp.now()), errors='coerce')
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        kan_outputs = self._load_kan_outputs()

        if not kan_outputs.empty and 'PLAYER_ID' in kan_outputs.columns:
            # Merge KAN outputs onto player data
            kan_subset = kan_outputs[OUTPUT_COLUMNS + ['PLAYER_ID']].drop_duplicates('PLAYER_ID')
            df = df.merge(kan_subset, on='PLAYER_ID', how='left')

            # Fill missing players with neutral defaults
            df['KAN_AGE_NONLIN_FACTOR'] = df['KAN_AGE_NONLIN_FACTOR'].fillna(1.0)
            df['KAN_AGE_INFLECTION_AGE'] = df['KAN_AGE_INFLECTION_AGE'].fillna(28.0)
            df['KAN_AGE_VOLATILITY'] = df['KAN_AGE_VOLATILITY'].fillna(0.05)
        else:
            # No KAN cache available — compute fallback from AGE column
            df = self._compute_fallback(df)

        # Shift all features within player groups
        for col in OUTPUT_COLUMNS:
            if col in df.columns:
                df[col] = df.groupby('PLAYER_ID')[col].shift(1)

        # Fill first-game NaN
        df['KAN_AGE_NONLIN_FACTOR'] = df['KAN_AGE_NONLIN_FACTOR'].fillna(1.0)
        df['KAN_AGE_INFLECTION_AGE'] = df['KAN_AGE_INFLECTION_AGE'].fillna(28.0)
        df['KAN_AGE_VOLATILITY'] = df['KAN_AGE_VOLATILITY'].fillna(0.05)

        df = normalize_output_columns(df, OUTPUT_COLUMNS)
        return df

    def _compute_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quadratic approximation when KAN model cache is unavailable."""
        if 'AGE' not in df.columns:
            df['KAN_AGE_NONLIN_FACTOR'] = 1.0
            df['KAN_AGE_INFLECTION_AGE'] = 28.0
            df['KAN_AGE_VOLATILITY'] = 0.05
            return df

        # Simple quadratic: peak at 28, symmetric rise/fall
        peak_age = 28.0
        ages = pd.to_numeric(df['AGE'], errors='coerce').fillna(peak_age)

        # Quadratic factor: 1.0 at peak, decreasing quadratically
        df['KAN_AGE_NONLIN_FACTOR'] = 1.0 - 0.0008 * (ages - peak_age) ** 2

        # Inflection age is always the estimated peak
        df['KAN_AGE_INFLECTION_AGE'] = peak_age

        # Volatility increases with distance from peak
        df['KAN_AGE_VOLATILITY'] = 0.02 + 0.003 * (ages - peak_age).abs()

        return df