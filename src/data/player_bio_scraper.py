"""Fetches player biographical data (birthdate, position, height, weight) from NBA API.

Uses nba_api.stats.endpoints.commonplayerinfo to enrich game logs with
AGE, POSITION, HEIGHT, WEIGHT, DRAFT_YEAR etc.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from nba_api.stats.endpoints import commonplayerinfo
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False


class PlayerBioScraper:
    """Scrapes player bio data from nba_api commonplayerinfo endpoint.

    Caches results to a CSV so repeated runs only fetch missing players.
    """

    RENAME_MAP = {
        'PERSON_ID': 'PLAYER_ID',
        'BIRTHDATE': 'BIRTHDATE',
        'POSITION': 'POSITION',
        'HEIGHT': 'HEIGHT',
        'WEIGHT': 'WEIGHT',
        'COUNTRY': 'COUNTRY',
        'DRAFT_YEAR': 'DRAFT_YEAR',
        'DRAFT_ROUND': 'DRAFT_ROUND',
        'DRAFT_NUMBER': 'DRAFT_NUMBER',
        'FROM_YEAR': 'CAREER_START',
        'TO_YEAR': 'CAREER_END',
        'SEASON_EXP': 'YEARS_EXPERIENCE',
        'DISPLAY_FIRST_LAST': 'PLAYER_NAME',
        'TEAM_ABBREVIATION': 'TEAM_ABBR',
    }

    OUTPUT_COLUMNS = [
        'PLAYER_ID', 'PLAYER_NAME', 'BIRTHDATE', 'AGE', 'POSITION',
        'HEIGHT', 'WEIGHT', 'COUNTRY', 'DRAFT_YEAR', 'DRAFT_ROUND',
        'DRAFT_NUMBER', 'CAREER_START', 'CAREER_END', 'YEARS_EXPERIENCE',
        'TEAM_ABBR',
    ]

    def __init__(
        self,
        cache_dir: str = 'data/cache',
        rate_delay: float = 0.6,
        config: Optional[Any] = None,
    ):
        self.cache_dir = cache_dir
        self.rate_delay = rate_delay
        self._config = config
        self._cache_path = os.path.join(cache_dir, 'player_bios.csv')
        os.makedirs(cache_dir, exist_ok=True)
        self.last_fetch_status: Dict[str, Any] = {}

    def _set_last_fetch_status(
        self,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_fetch_status = {
            'source_key': 'player_bio',
            'status': status,
            'required': False,
            'message': message,
            'details': details or {},
        }

    def get_last_fetch_status(self) -> Dict[str, Any]:
        return dict(self.last_fetch_status)

    def fetch_player_bio(self, player_id: int) -> pd.DataFrame:
        """Fetch bio for a single player by NBA player ID."""
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api not installed; cannot fetch player bio")
            return pd.DataFrame()

        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            df = info.common_player_info.get_data_frame()
            time.sleep(self.rate_delay)
            return self._normalize_bio(df)
        except Exception as e:
            logger.warning(f"Failed to fetch bio for player {player_id}: {e}")
            return pd.DataFrame()

    def _normalize_bio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and compute AGE from BIRTHDATE."""
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Rename known columns
        rename = {k: v for k, v in self.RENAME_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)

        # Parse BIRTHDATE and compute AGE
        if 'BIRTHDATE' in df.columns:
            df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce')
            now = pd.Timestamp.now()
            df['AGE'] = (now - df['BIRTHDATE']).dt.days / 365.25
            df['AGE'] = df['AGE'].round(2)
        else:
            df['BIRTHDATE'] = pd.NaT
            df['AGE'] = float('nan')

        # Parse career start/end to int
        for col in ['CAREER_START', 'CAREER_END', 'DRAFT_YEAR']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Parse years experience
        if 'YEARS_EXPERIENCE' in df.columns:
            df['YEARS_EXPERIENCE'] = pd.to_numeric(df['YEARS_EXPERIENCE'], errors='coerce')

        # Parse weight to numeric (API sometimes returns string like "250")
        if 'WEIGHT' in df.columns:
            df['WEIGHT'] = pd.to_numeric(df['WEIGHT'], errors='coerce')

        # Ensure all output columns exist
        for col in self.OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[self.OUTPUT_COLUMNS]

    def fetch_all_bios(
        self,
        player_ids: List[int],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch bio for multiple players with disk caching.

        Only fetches players not already in cache (unless force_refresh=True).
        """
        cached = self._load_cache()

        if cached is not None and not force_refresh:
            cached_ids = set(cached['PLAYER_ID'].unique())
            missing = [pid for pid in player_ids if pid not in cached_ids]
            if not missing:
                self._set_last_fetch_status(
                    'success',
                    'All player bios loaded from cache',
                    {'source': 'disk_cache', 'count': len(player_ids)},
                )
                return cached[cached['PLAYER_ID'].isin(player_ids)]
        else:
            missing = list(player_ids) if force_refresh else \
                [pid for pid in player_ids
                 if cached is None or pid not in set(cached['PLAYER_ID'].unique())]

        results = []
        fetched = 0
        failed = 0

        for pid in missing:
            bio_df = self.fetch_player_bio(pid)
            if not bio_df.empty:
                results.append(bio_df)
                fetched += 1
            else:
                failed += 1

        new_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        # Merge with cache
        if cached is not None and not new_df.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset='PLAYER_ID', keep='last')
        elif not new_df.empty:
            combined = new_df
        else:
            combined = cached if cached is not None else pd.DataFrame()

        if not combined.empty:
            self._save_cache(combined)

        self._set_last_fetch_status(
            'success' if fetched > 0 else 'fallback',
            f"Fetched {fetched} new player bios, {failed} failed",
            {
                'source': 'nba_api',
                'fetched': fetched,
                'failed': failed,
                'cached': len(cached) if cached is not None else 0,
            },
        )

        if combined.empty:
            return combined
        return combined[combined['PLAYER_ID'].isin(player_ids)]

    def resolve_name_to_id(self, player_name: str) -> Optional[int]:
        """Look up a player ID by name using nba_api static player list.

        Returns the first match (case-insensitive).
        """
        if not NBA_API_AVAILABLE:
            return None

        try:
            from nba_api.stats.static import players as nba_players
            all_players = nba_players.get_players()
            name_lower = player_name.lower().strip()
            for p in all_players:
                if p['full_name'].lower() == name_lower:
                    return p['id']
            # Fuzzy: check if search name is contained
            for p in all_players:
                if name_lower in p['full_name'].lower():
                    return p['id']
        except Exception:
            pass
        return None

    def _load_cache(self) -> Optional[pd.DataFrame]:
        if os.path.exists(self._cache_path):
            try:
                df = pd.read_csv(self._cache_path)
                if 'PLAYER_ID' in df.columns:
                    logger.info(f"Loaded {len(df)} cached player bios from {self._cache_path}")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load player bio cache: {e}")
        return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._cache_path, index=False)
            logger.info(f"Saved {len(df)} player bios to {self._cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save player bio cache: {e}")