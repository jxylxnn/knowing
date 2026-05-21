"""Injury risk feature group — METIC-style workload + injury history signals.

Reads from the persistent injury history log (data/injury_history.csv)
produced by InjuryHistoryLogger, and combines with workload metrics
from the game log itself.
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

# Columns this feature group always emits
OUTPUT_COLUMNS = [
    'INJURY_RISK_CAREER_COUNT',
    'INJURY_RISK_LAST_90D',
    'INJURY_RISK_LAST_30D',
    'INJURY_RISK_WORKLOAD_SPIKE',
    'INJURY_RISK_BACK_TO_BACK_STRESS',
    'INJURY_RISK_AVG_DAYS_BETWEEN',
]


class InjuryRiskFeatureGroup(FeatureGroup):
    """METIC-inspired injury risk features from workload + injury history.

    All features are shifted by 1 within each player group so the current
    game never leaks future information.
    """

    @property
    def name(self) -> str:
        return 'injury_risk'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'MIN', 'TEAM_ID']

    @property
    def optional_columns(self) -> List[str]:
        return []

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self._injury_history: Optional[pd.DataFrame] = None

    def _load_injury_history(self) -> pd.DataFrame:
        """Load injury history CSV, cache in memory for the session."""
        if self._injury_history is not None:
            return self._injury_history

        path = os.path.join(self.data_dir, 'injury_history.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                if 'PLAYER_ID' in df.columns:
                    df['PLAYER_ID'] = pd.to_numeric(df['PLAYER_ID'], errors='coerce')
                self._injury_history = df
                return df
            except Exception as e:
                logger.warning(f"Failed to load injury history: {e}")
        self._injury_history = pd.DataFrame()
        return self._injury_history

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

        # Load injury history
        injury_hist = self._load_injury_history()

        # Build career injury count per player (cumulative, shifted)
        if not injury_hist.empty and 'PLAYER_ID' in injury_hist.columns:
            injury_counts = (
                injury_hist[injury_hist['PLAYER_ID'].notna()]
                .groupby('PLAYER_ID').size()
                .rename('INJURY_RISK_CAREER_COUNT_raw')
            )
            df = df.merge(
                injury_counts,
                left_on='PLAYER_ID',
                right_index=True,
                how='left',
            )
            df['INJURY_RISK_CAREER_COUNT_raw'] = df['INJURY_RISK_CAREER_COUNT_raw'].fillna(0)
            # Shift within player group so we never see today's injury
            df['INJURY_RISK_CAREER_COUNT'] = df.groupby('PLAYER_ID')[
                'INJURY_RISK_CAREER_COUNT_raw'
            ].shift(1).fillna(0)
            df = df.drop(columns=['INJURY_RISK_CAREER_COUNT_raw'])
        else:
            df['INJURY_RISK_CAREER_COUNT'] = 0.0

        # Recent injury counts (90d, 30d)
        df['INJURY_RISK_LAST_90D'] = 0.0
        df['INJURY_RISK_LAST_30D'] = 0.0

        if not injury_hist.empty and 'PLAYER_ID' in injury_hist.columns and 'DATE' in injury_hist.columns:
            # Create a dict for fast lookup: player_id -> sorted dates
            valid_hist = injury_hist[injury_hist['PLAYER_ID'].notna()].copy()
            player_injury_dates = valid_hist.groupby('PLAYER_ID')['DATE'].apply(
                lambda x: sorted(x.dropna().tolist())
            ).to_dict()

            counts_90 = []
            counts_30 = []
            for _, row in df.iterrows():
                pid = row['PLAYER_ID']
                gd = row['GAME_DATE']
                dates = player_injury_dates.get(pid, [])
                # Only count injuries BEFORE this game date (shifted)
                prior_dates = [d for d in dates if d < gd]
                count_90 = sum(1 for d in prior_dates if (gd - d).days <= 90)
                count_30 = sum(1 for d in prior_dates if (gd - d).days <= 30)
                counts_90.append(count_90)
                counts_30.append(count_30)

            df['INJURY_RISK_LAST_90D'] = counts_90
            df['INJURY_RISK_LAST_30D'] = counts_30

        # Workload spike: MIN > 1.3 * rolling 10-game avg (shifted)
        if 'MIN' in df.columns:
            rolling_min_avg = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df['INJURY_RISK_WORKLOAD_SPIKE'] = (
                df['MIN'] > 1.3 * rolling_min_avg
            ).astype(float)
            df['INJURY_RISK_WORKLOAD_SPIKE'] = df.groupby('PLAYER_ID')[
                'INJURY_RISK_WORKLOAD_SPIKE'
            ].shift(1).fillna(0)
        else:
            df['INJURY_RISK_WORKLOAD_SPIKE'] = 0.0

        # Back-to-back stress: count B2B games in last 14 days (shifted)
        df['INJURY_RISK_BACK_TO_BACK_STRESS'] = 0.0
        if 'GAME_DATE' in df.columns:
            b2b_counts = []
            for pid, group in df.groupby('PLAYER_ID'):
                dates = group['GAME_DATE'].sort_values().values
                counts = np.zeros(len(dates), dtype=float)
                for i in range(len(dates)):
                    # Look at games in prior 14 days
                    window_start = dates[i] - pd.Timedelta(days=14)
                    prior_mask = (dates[:i] >= window_start) & (dates[:i] < dates[i])
                    prior_dates_in_window = dates[:i][prior_mask]
                    # Count B2B: consecutive-day games
                    if len(prior_dates_in_window) >= 2:
                        sorted_days = np.sort(prior_dates_in_window)
                        diffs = np.diff(sorted_days)
                        counts[i] = float(np.sum(diffs == pd.Timedelta(days=1)))
                b2b_counts.append(pd.Series(counts, index=group.index))
            if b2b_counts:
                df['INJURY_RISK_BACK_TO_BACK_STRESS'] = pd.concat(b2b_counts).sort_index()
            # Shift
            df['INJURY_RISK_BACK_TO_BACK_STRESS'] = df.groupby('PLAYER_ID')[
                'INJURY_RISK_BACK_TO_BACK_STRESS'
            ].shift(1).fillna(0)

        # Average days between injury events
        df['INJURY_RISK_AVG_DAYS_BETWEEN'] = 0.0
        if not injury_hist.empty and 'PLAYER_ID' in injury_hist.columns and 'DATE' in injury_hist.columns:
            valid_hist = injury_hist[injury_hist['PLAYER_ID'].notna()].copy()
            player_injury_dates_map = valid_hist.groupby('PLAYER_ID')['DATE'].apply(
                lambda x: sorted(x.dropna().tolist())
            ).to_dict()

            avg_days = []
            for _, row in df.iterrows():
                pid = row['PLAYER_ID']
                gd = row['GAME_DATE']
                dates = player_injury_dates_map.get(pid, [])
                prior_dates = sorted([d for d in dates if d < gd])
                if len(prior_dates) >= 2:
                    diffs = np.diff([d.timestamp() for d in prior_dates]) / 86400.0
                    avg_days.append(float(np.mean(diffs)))
                else:
                    avg_days.append(0.0)
            df['INJURY_RISK_AVG_DAYS_BETWEEN'] = avg_days

        df = normalize_output_columns(df, OUTPUT_COLUMNS)
        return df