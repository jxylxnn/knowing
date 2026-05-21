"""Aging curve feature group — B-Ianus Bayesian model features.

Uses position-specific priors and MAP-estimated peak ages to generate
aging adjustment features for each player-game row.
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
    'AGING_PLAYER_AGE',
    'AGING_YEARS_IN_LEAGUE',
    'AGING_PEAK_AGE_EST',
    'AGING_PRE_POST_PEAK',
    'AGING_CURVE_FACTOR',
    'AGING_DECLINE_RATE',
]


class AgingCurveFeatureGroup(FeatureGroup):
    """B-Ianus Bayesian aging curve features.

    All features use shift(1) within player groups to prevent leakage.
    When bio data (AGE, POSITION) is missing, features default to neutral values.
    """

    @property
    def name(self) -> str:
        return 'aging_curve'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE']

    @property
    def optional_columns(self) -> List[str]:
        return ['AGE', 'POSITION', 'CAREER_START']

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()

        if 'PLAYER_ID' not in df.columns or 'GAME_DATE' not in df.columns:
            df = normalize_output_columns(df, OUTPUT_COLUMNS)
            return df

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        # Load cached aging curves
        aging_curves = self._load_aging_curves()

        bio_lookup = self._build_bio_lookup(aging_curves)

        # Compute features per row
        ages = []
        years_in_league = []
        peak_age_ests = []
        pre_post_peaks = []
        curve_factors = []
        decline_rates = []

        for _, row in df.iterrows():
            pid = row['PLAYER_ID']
            bio = bio_lookup.get(pid, {})

            # Player age at game date
            if 'AGE' in row and pd.notna(row.get('AGE')):
                age = float(row['AGE'])
            elif 'BIRTHDATE' in bio and pd.notna(bio.get('BIRTHDATE')):
                gd = row['GAME_DATE']
                bd = pd.Timestamp(bio['BIRTHDATE'])
                age = (gd - bd).days / 365.25
            else:
                age = 27.5  # league average

            # Years in league
            if 'CAREER_START' in row and pd.notna(row.get('CAREER_START')):
                career_start = int(row['CAREER_START'])
            elif 'career_start' in bio and pd.notna(bio.get('career_start')):
                career_start = int(bio['career_start'])
            else:
                career_start = row['GAME_DATE'].year - 4  # assume 4yr vet

            season_year = row['GAME_DATE'].year
            yil = float(max(0, season_year - career_start))

            # Position and aging curve params
            position = str(row.get('POSITION', bio.get('position', 'SF')))
            params = bio.get('curve_params')

            if params:
                peak_age = params['peak_age']
                decline_rate = params['decline_rate']
                dev_rate = params['development_rate']
            else:
                # Use position defaults
                peak_age, decline_rate, dev_rate = self._position_defaults(position)

            # Pre/post peak indicator
            pre_post = 0.0 if age <= peak_age else 1.0

            # Curve factor
            if age <= peak_age:
                cf = 1.0 + dev_rate * (age - peak_age)
            else:
                cf = 1.0 - decline_rate * (age - peak_age)

            ages.append(age)
            years_in_league.append(yil)
            peak_age_ests.append(peak_age)
            pre_post_peaks.append(pre_post)
            curve_factors.append(cf)
            decline_rates.append(decline_rate)

        df['AGING_PLAYER_AGE'] = ages
        df['AGING_YEARS_IN_LEAGUE'] = years_in_league
        df['AGING_PEAK_AGE_EST'] = peak_age_ests
        df['AGING_PRE_POST_PEAK'] = pre_post_peaks
        df['AGING_CURVE_FACTOR'] = curve_factors
        df['AGING_DECLINE_RATE'] = decline_rates

        # Shift all features within player groups (no leakage)
        for col in OUTPUT_COLUMNS:
            if col in df.columns:
                df[col] = df.groupby('PLAYER_ID')[col].shift(1)

        # Fill first-game NaN with neutral defaults
        df['AGING_PLAYER_AGE'] = df['AGING_PLAYER_AGE'].fillna(27.5)
        df['AGING_YEARS_IN_LEAGUE'] = df['AGING_YEARS_IN_LEAGUE'].fillna(4.0)
        df['AGING_PEAK_AGE_EST'] = df['AGING_PEAK_AGE_EST'].fillna(27.5)
        df['AGING_PRE_POST_PEAK'] = df['AGING_PRE_POST_PEAK'].fillna(0.0)
        df['AGING_CURVE_FACTOR'] = df['AGING_CURVE_FACTOR'].fillna(1.0)
        df['AGING_DECLINE_RATE'] = df['AGING_DECLINE_RATE'].fillna(0.012)

        df = normalize_output_columns(df, OUTPUT_COLUMNS)
        return df

    def _load_aging_curves(self) -> pd.DataFrame:
        """Load precomputed aging curves from cache."""
        cache_path = os.path.join(self.data_dir, 'cache', 'aging_curves.csv')
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception:
                pass
        return pd.DataFrame()

    def _build_bio_lookup(self, aging_curves: pd.DataFrame) -> dict:
        """Build a player_id -> bio dict from aging curves CSV + bios CSV."""
        lookup = {}

        # Load standalone bios file if it exists
        bio_path = os.path.join(self.data_dir, 'player_bios.csv')
        if os.path.exists(bio_path):
            try:
                bios = pd.read_csv(bio_path)
                if 'PLAYER_ID' in bios.columns:
                    for _, row in bios.iterrows():
                        pid = row['PLAYER_ID']
                        lookup[pid] = {
                            'BIRTHDATE': row.get('BIRTHDATE'),
                            'position': row.get('POSITION', 'SF'),
                            'career_start': row.get('CAREER_START'),
                        }
            except Exception:
                pass

        # Merge in aging curve params
        if not aging_curves.empty and 'PLAYER_ID' in aging_curves.columns:
            for _, row in aging_curves.iterrows():
                pid = row['PLAYER_ID']
                if pid not in lookup:
                    lookup[pid] = {}
                lookup[pid]['curve_params'] = {
                    'peak_age': row.get('peak_age', 27.5),
                    'decline_rate': row.get('decline_rate', 0.012),
                    'development_rate': row.get('development_rate', 0.02),
                }
                if 'position' in row and 'position' not in lookup[pid]:
                    lookup[pid]['position'] = row['position']

        return lookup

    @staticmethod
    def _position_defaults(position: str) -> tuple:
        """Return (peak_age, decline_rate, dev_rate) defaults for a position."""
        from src.lifecycle.aging_model import POSITION_PEAK_PRIORS, POSITION_DECLINE_PRIORS, normalize_position
        pos = normalize_position(str(position))
        peak = POSITION_PEAK_PRIORS.get(pos, (27.5, 1.5))[0]
        decline = POSITION_DECLINE_PRIORS.get(pos, (0.012, 0.004))[0]
        return (peak, decline, 0.02)