"""Runtime loader for residual correction models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

logger = logging.getLogger(__name__)

CATBOOST_AVAILABLE = CatBoostRegressor is not None


class ResidualCorrectionModel:
    """Load and apply trained residual correction models at inference time.

    Returns ``0.0`` for missing or rejected stat models so the correction
    layer is always safe to call.
    """

    METADATA_FILE = "residual_metadata.json"
    SCHEMA_FILE = "residual_feature_schema.json"

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._feature_cols: List[str] = []
        self._loaded = False

    def load(self, model_dir: str = "models/residual") -> None:
        """Load all accepted residual models from *model_dir*."""
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning("Residual model directory does not exist: %s", model_path)
            return

        metadata_file = model_path / self.METADATA_FILE
        if metadata_file.exists():
            self._metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        else:
            logger.warning("No residual metadata found at %s", metadata_file)
            return

        schema_file = model_path / self.SCHEMA_FILE
        if schema_file.exists():
            schema = json.loads(schema_file.read_text(encoding="utf-8"))
            self._feature_cols = schema.get("feature_cols", [])
        else:
            logger.warning("No residual feature schema found at %s", schema_file)

        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available — residual corrections disabled")
            return

        targets_meta = self._metadata.get("targets", {})
        for stat, meta in targets_meta.items():
            if meta.get("status") != "accepted":
                continue
            cbm_file = model_path / f"{stat.lower()}_residual.cbm"
            if not cbm_file.exists():
                logger.warning("Accepted model file missing for %s: %s", stat, cbm_file)
                continue
            try:
                model = CatBoostRegressor()
                model.load_model(str(cbm_file))
                self._models[stat] = model
                logger.info("Loaded residual model for %s", stat)
            except Exception as exc:
                logger.error("Failed to load residual model for %s: %s", stat, exc)

        self._loaded = True
        logger.info(
            "Residual correction models loaded: %s",
            list(self._models.keys()) if self._models else "none",
        )

    def predict_correction(self, stat: str, feature_row: pd.DataFrame) -> float:
        """Predict the residual correction for a single row.

        Args:
            stat: Target stat (e.g. ``"PTS"``).
            feature_row: Single-row DataFrame with the correction features.

        Returns:
            Predicted correction value, or ``0.0`` if the model is
            missing or rejected.
        """
        if not self.is_enabled(stat):
            return 0.0

        model = self._models.get(stat)
        if model is None:
            return 0.0

        try:
            cols = self._feature_cols if self._feature_cols else list(feature_row.columns)
            X = feature_row[cols].values if isinstance(feature_row, pd.DataFrame) else feature_row
            pred = model.predict(X)
            return float(np.asarray(pred).ravel()[0])
        except Exception as exc:
            logger.warning("Residual prediction failed for %s: %s", stat, exc)
            return 0.0

    def is_enabled(self, stat: str) -> bool:
        """Return True if a correction model is loaded and accepted for *stat*."""
        if not self._loaded:
            return False
        targets_meta = self._metadata.get("targets", {})
        meta = targets_meta.get(stat, {})
        return meta.get("status") == "accepted" and stat in self._models

    @property
    def feature_cols(self) -> List[str]:
        """Return the feature columns expected by the residual models."""
        return list(self._feature_cols)

    @property
    def loaded_stats(self) -> List[str]:
        """Return the list of stats with loaded models."""
        return sorted(self._models.keys())
