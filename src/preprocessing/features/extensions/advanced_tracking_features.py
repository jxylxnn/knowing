"""Example feature-group extension: advanced tracking features.

Paired with ``advanced_tracking_scraper.py``. Demonstrates the complete
plug-in loop for the *feature* side:

* Subclass :class:`FeatureGroup`.
* Declare ``feature_prefixes`` so the ``FeatureSelector`` keeps these columns
  (without this, columns whose names don't match a safe prefix are dropped —
  the registry folds declared prefixes into the selector automatically).
* Declare ``external_files()`` so the feature cache is invalidated whenever
  ``data/advanced_tracking.csv`` changes (the cache key folds each file's
  size + mtime).
* Implement ``create`` with a 1-step shift within each player so the current
  game never leaks future tracking data.

This module is auto-discovered by ``FeatureGroupRegistry``; no edits to
``features/__init__.py``, ``FeatureEngineer._build_groups``, or presets are
required. Enable it via a preset's ``enable_groups`` (or the ``"all"``
sentinel used by the ``full`` preset).
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
)

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "ADVTRACK_AVG_SPEED_10",
    "ADVTRACK_DIST_10",
    "ADVTRACK_TOUCHES_10",
    "ADVTRACK_ELBOW_SHARE",
    "ADVTRACK_POST_SHARE",
]

SOURCE_FILE = "advanced_tracking.csv"


class AdvancedTrackingFeatureGroup(FeatureGroup):
    """Leakage-safe rolling features from advanced tracking data."""

    # Declared so FeatureSelector keeps these columns automatically.
    feature_prefixes = ("ADVTRACK_",)
    feature_keywords = ()

    @property
    def name(self) -> str:
        return "advanced_tracking"

    @property
    def required_columns(self) -> List[str]:
        return ["PLAYER_ID", "GAME_DATE"]

    @property
    def optional_columns(self) -> List[str]:
        return []

    def __init__(self, data_dir: str = "data", window: int = 10):
        self.data_dir = data_dir
        self.window = window
        self._tracking: Optional[pd.DataFrame] = None

    def external_files(self) -> List[str]:
        return [os.path.join(self.data_dir, SOURCE_FILE)]

    def _load_tracking(self) -> pd.DataFrame:
        if self._tracking is not None:
            return self._tracking
        path = os.path.join(self.data_dir, SOURCE_FILE)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "GAME_DATE" in df.columns:
                    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
                if "PLAYER_ID" in df.columns:
                    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
                self._tracking = df
                return df
            except Exception as exc:
                logger.warning("advanced_tracking: failed to load %s: %s", path, exc)
        self._tracking = pd.DataFrame()
        return self._tracking

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()

        for col in OUTPUT_COLUMNS:
            df[col] = 0.0

        tracking = self._load_tracking()
        if tracking.empty:
            if diagnostics is not None:
                diagnostics.warn("advanced_tracking: no tracking data; using zeros.")
            return df
        if "PLAYER_ID" not in tracking.columns or "GAME_DATE" not in tracking.columns:
            if diagnostics is not None:
                diagnostics.warn("advanced_tracking: tracking file missing join keys.")
            return df

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

        merged = df[["PLAYER_ID", "GAME_DATE"]].merge(
            tracking[["PLAYER_ID", "GAME_DATE", "AVG_SPEED_MPS", "DIST_MILES",
                      "TOUCHES", "ELBOW_TOUCHES", "POST_TOUCHES"]],
            on=["PLAYER_ID", "GAME_DATE"],
            how="left",
        )

        merged["AVG_SPEED_MPS"] = merged["AVG_SPEED_MPS"].fillna(0.0)
        merged["DIST_MILES"] = merged["DIST_MILES"].fillna(0.0)
        merged["TOUCHES"] = merged["TOUCHES"].fillna(0.0)
        merged["ELBOW_TOUCHES"] = merged["ELBOW_TOUCHES"].fillna(0.0)
        merged["POST_TOUCHES"] = merged["POST_TOUCHES"].fillna(0.0)

        # Past-only rolling means (shift(1) so the current game never leaks).
        g = merged.groupby("PLAYER_ID")
        merged["ADVTRACK_AVG_SPEED_10"] = g["AVG_SPEED_MPS"].transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=1).mean()
        )
        merged["ADVTRACK_DIST_10"] = g["DIST_MILES"].transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=1).mean()
        )
        merged["ADVTRACK_TOUCHES_10"] = g["TOUCHES"].transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=1).mean()
        )
        # Touch-type shares (also shifted to avoid leakage).
        touches_shifted = g["TOUCHES"].transform(lambda s: s.shift(1))
        elbow_shifted = g["ELBOW_TOUCHES"].transform(lambda s: s.shift(1))
        post_shifted = g["POST_TOUCHES"].transform(lambda s: s.shift(1))
        merged["ADVTRACK_ELBOW_SHARE"] = np.where(
            touches_shifted > 0, elbow_shifted / touches_shifted, 0.0
        )
        merged["ADVTRACK_POST_SHARE"] = np.where(
            touches_shifted > 0, post_shifted / touches_shifted, 0.0
        )

        for col in OUTPUT_COLUMNS:
            df[col] = merged[col].fillna(0.0).values

        return df