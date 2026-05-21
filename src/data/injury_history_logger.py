"""Accumulates injury events into a persistent longitudinal log.

Used by the METIC-style injury forecasting feature group to build
per-player injury histories across update_data.py runs.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class InjuryHistoryLogger:
    """Persists injury events across update_data runs into a CSV log.

    Each call to log_injuries() appends new events and deduplicates by
    (PLAYER_ID or PLAYER, DATE, INJURY_TYPE) so the same injury isn't
    recorded twice.
    """

    DEFAULT_COLUMNS = [
        'PLAYER_ID', 'PLAYER', 'TEAM_ABBR', 'STATUS',
        'INJURY_TYPE', 'DATE', 'PLAY_PROBABILITY',
    ]

    def __init__(
        self,
        history_dir: str = 'data',
        filename: str = 'injury_history.csv',
    ):
        self.path = os.path.join(history_dir, filename)
        os.makedirs(history_dir, exist_ok=True)

    def log_injuries(self, events: List[Dict]) -> None:
        """Append injury events to the persistent log (deduplicates)."""
        if not events:
            return

        new_df = pd.DataFrame(events)

        # Validate minimum columns
        if 'PLAYER' not in new_df.columns and 'PLAYER_ID' not in new_df.columns:
            logger.warning(
                "Injury events must have PLAYER or PLAYER_ID; skipping log"
            )
            return

        # Fill missing optional columns
        for col in self.DEFAULT_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = None

        # Keep only known columns + any extras
        keep_cols = [c for c in self.DEFAULT_COLUMNS if c in new_df.columns]
        extra_cols = [c for c in new_df.columns if c not in self.DEFAULT_COLUMNS]
        new_df = new_df[keep_cols + extra_cols]

        # Normalize DATE
        if 'DATE' in new_df.columns:
            new_df['DATE'] = pd.to_datetime(new_df['DATE'], errors='coerce')

        existing = self.load_history()
        if existing.empty:
            combined = new_df
        else:
            # Normalize existing DATE too
            if 'DATE' in existing.columns:
                existing['DATE'] = pd.to_datetime(existing['DATE'], errors='coerce')
            combined = pd.concat([existing, new_df], ignore_index=True)

            # Deduplicate: prefer PLAYER_ID if available, fall back to PLAYER name
            dedup_cols = []
            if 'PLAYER_ID' in combined.columns and combined['PLAYER_ID'].notna().any():
                dedup_cols.append('PLAYER_ID')
            elif 'PLAYER' in combined.columns:
                dedup_cols.append('PLAYER')
            else:
                dedup_cols = []  # no dedup possible

            dedup_cols += ['DATE']
            if 'INJURY_TYPE' in combined.columns:
                dedup_cols.append('INJURY_TYPE')

            # Only dedup if we have enough keys
            valid_dedup = [c for c in dedup_cols if c in combined.columns]
            if len(valid_dedup) >= 2 and not combined[valid_dedup].isna().all(axis=1).any():
                combined = combined.drop_duplicates(subset=valid_dedup, keep='last')

        combined.to_csv(self.path, index=False)
        logger.info(f"Injury history: {len(combined)} events saved to {self.path}")

    def load_history(self) -> pd.DataFrame:
        """Load the persistent injury history CSV."""
        if os.path.exists(self.path):
            try:
                df = pd.read_csv(self.path)
                return df
            except Exception as e:
                logger.warning(f"Failed to load injury history: {e}")
        return pd.DataFrame()

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Get all injury events for a specific player."""
        df = self.load_history()
        if df.empty:
            return pd.DataFrame()
        if 'PLAYER_ID' not in df.columns:
            return pd.DataFrame()
        # PLAYER_ID may be float due to CSV round-trip
        df['PLAYER_ID'] = pd.to_numeric(df['PLAYER_ID'], errors='coerce')
        return df[df['PLAYER_ID'] == player_id].sort_values('DATE')

    def get_player_history_by_name(self, player_name: str) -> pd.DataFrame:
        """Get all injury events for a player by name."""
        df = self.load_history()
        if df.empty or 'PLAYER' not in df.columns:
            return pd.DataFrame()
        return df[
            df['PLAYER'].str.lower().str.contains(player_name.lower(), na=False)
        ].sort_values('DATE')

    def count_events_since(
        self,
        player_id: int,
        since_date: str,
    ) -> int:
        """Count injury events for a player since a given date."""
        history = self.get_player_history(player_id)
        if history.empty or 'DATE' not in history.columns:
            return 0
        history['DATE'] = pd.to_datetime(history['DATE'], errors='coerce')
        since = pd.Timestamp(since_date)
        return int((history['DATE'] >= since).sum())