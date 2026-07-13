"""Example data-source extension: advanced tracking stats.

This is a *self-contained* example showing the full plug-in loop:

1. Drop this module into ``src/data/extensions/``.
2. Enable it in ``config/default.yaml``::

       data_sources:
         advanced_tracking:
           enabled: true

3. Run ``python update_data.py`` — the scraper runs after the core fetch and
   writes ``data/advanced_tracking.csv``.
4. The matching feature group
   (``src/preprocessing/features/extensions/advanced_tracking_features.py``)
   reads that CSV and emits leakage-safe features that the registry wires
   into ``FeatureEngineer`` automatically.

The scraper here generates synthetic tracking data derived from the existing
player game logs so the example works offline without a real tracking API.
Replace ``fetch`` with a real HTTP/API call to wire an actual new data source.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

OUTPUT_FILE = "advanced_tracking.csv"


class AdvancedTrackingScraper(BaseScraper):
    """Example scraper that produces per-player advanced tracking signals."""

    @property
    def name(self) -> str:
        return "advanced_tracking"

    def output_files(self) -> List[str]:
        return [OUTPUT_FILE]

    def requires(self) -> List[str]:
        # This example derives tracking data from the core player logs, so it
        # depends on nba_players.csv existing. A real scraper that hits an
        # external API would have no core dependency.
        return ["nba_players.csv"]

    def fetch(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        data_dir = "data"
        players_path = os.path.join(data_dir, "nba_players.csv")
        if not os.path.exists(players_path):
            logger.info("advanced_tracking: nba_players.csv missing; skipping.")
            return {}

        try:
            logs = pd.read_csv(players_path)
        except Exception as exc:
            logger.warning("advanced_tracking: could not read nba_players.csv: %s", exc)
            return {}

        needed = {"PLAYER_ID", "GAME_DATE", "MIN", "PTS", "REB", "AST"}
        if not needed.issubset(logs.columns):
            logger.info("advanced_tracking: core logs missing required columns; skipping.")
            return {}

        logs = logs[list(needed)].copy()
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        logs = logs.dropna(subset=["GAME_DATE", "PLAYER_ID"]).sort_values(
            ["PLAYER_ID", "GAME_DATE"]
        )

        # Derive synthetic tracking signals from box-score usage. These stand
        # in for real tracking metrics (distance, speed, touch counts). In a
        # real extension, replace this block with an API fetch + merge on
        # (PLAYER_ID, GAME_DATE).
        logs["AVG_SPEED_MPS"] = 3.6 + (logs["MIN"].fillna(0) / 40.0).clip(0, 1) * 1.2
        logs["DIST_MILES"] = logs["MIN"].fillna(0) * 0.045
        logs["TOUCHES"] = logs["PTS"].fillna(0) * 1.7 + logs["AST"].fillna(0) * 4.0
        logs["ELBOW_TOUCHES"] = logs["TOUCHES"] * 0.18
        logs["POST_TOUCHES"] = logs["TOUCHES"] * 0.07 + logs["REB"].fillna(0) * 0.5

        out = logs[
            ["PLAYER_ID", "GAME_DATE", "AVG_SPEED_MPS", "DIST_MILES", "TOUCHES",
             "ELBOW_TOUCHES", "POST_TOUCHES"]
        ]
        return {OUTPUT_FILE: out}