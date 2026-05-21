"""Archetype-conditioned Empirical Copula engine.

Computes and caches 6x6 correlation matrices of residuals grouped by
player Archetype.  These matrices let the Monte Carlo engine generate
realistically correlated multi-stat draws without a neural Copula head.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 6 canonical target stats in order
STAT_ORDER: List[str] = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]


class CovarianceCache:
    """Computes and caches 6x6 correlation matrices grouped by Archetype.

    The core idea: residuals (actual - projected) for the same archetype
    share a correlation structure.  By caching one 6x6 matrix per
    archetype we can inject realistic stat correlations during Monte
    Carlo simulation without training a neural copula.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache: Dict[str, np.ndarray] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_and_save(self, df: pd.DataFrame) -> None:
        """Compute archetype-conditioned correlation matrices from historical data.

        Args:
            df: DataFrame with columns:
                - ``ARCHETYPE_LABEL`` (str) – archetype bucket for each row.
                - ``{stat}_ACTUAL`` (float) – actual box-score value.
                - ``{stat}_MEAN_PROJ`` (float) – model projection for that game.
        """
        if "ARCHETYPE_LABEL" not in df.columns:
            logger.warning("CovarianceCache: no ARCHETYPE_LABEL column, using single global matrix")
            df = df.copy()
            df["ARCHETYPE_LABEL"] = "GLOBAL"

        for archetype, group in df.groupby("ARCHETYPE_LABEL"):
            actual_cols = [f"{s}_ACTUAL" for s in STAT_ORDER]
            proj_cols = [f"{s}_MEAN_PROJ" for s in STAT_ORDER]

            missing = [c for c in actual_cols + proj_cols if c not in group.columns]
            if missing:
                logger.warning(
                    "CovarianceCache: missing columns for archetype '%s': %s — skipping",
                    archetype, missing,
                )
                continue

            residuals = group[actual_cols].values - group[proj_cols].values

            if len(residuals) < 10:
                logger.debug("CovarianceCache: too few samples for '%s' (%d), using identity", archetype, len(residuals))
                self._cache[archetype] = np.eye(6)
                continue

            corr = np.corrcoef(residuals, rowvar=False)

            # Ensure Positive Semi-Definite (PSD) via eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.maximum(eigvals, 1e-6)
            corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            np.fill_diagonal(corr, 1.0)

            self._cache[archetype] = corr
            logger.info(
                "CovarianceCache: cached %dx%d matrix for archetype '%s' (%d samples)",
                6, 6, archetype, len(residuals),
            )

        # Save
        save_path = os.path.join(self.cache_dir, "archetype_covariances.npz")
        np.savez(save_path, **{k: v for k, v in self._cache.items()})
        logger.info("CovarianceCache: saved %d matrices to %s", len(self._cache), save_path)
        self._loaded = True

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load cached matrices from disk.

        Returns:
            True if at least one matrix was loaded successfully.
        """
        path = os.path.join(self.cache_dir, "archetype_covariances.npz")
        if not os.path.exists(path):
            logger.info("CovarianceCache: no cache file at %s", path)
            return False
        try:
            data = np.load(path, allow_pickle=True)
            self._cache = {str(k): data[k] for k in data.files}
            self._loaded = True
            logger.info("CovarianceCache: loaded %d matrices", len(self._cache))
            return True
        except Exception as exc:
            logger.warning("CovarianceCache: failed to load %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_correlation(self, archetype: str) -> np.ndarray:
        """Return the 6x6 correlation matrix for an archetype.

        Falls back to identity matrix when the archetype is unknown or
        no cache has been loaded.
        """
        if not self._loaded and not self.load():
            return np.eye(6)
        return self._cache.get(archetype, self._cache.get("GLOBAL", np.eye(6)))

    @property
    def available_archetypes(self) -> List[str]:
        return list(self._cache.keys())
