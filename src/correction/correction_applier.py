"""Correction applier — integrates residual models into the prediction path."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.correction.correction_features import CorrectionFeatureBuilder
from src.correction.residual_model import ResidualCorrectionModel

logger = logging.getLogger(__name__)

TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


class CorrectionApplier:
    """Apply accepted residual corrections to base predictions.

    Wraps :class:`ResidualCorrectionModel` and
    :class:`CorrectionFeatureBuilder` so callers only need to call
    :meth:`apply`.
    """

    def __init__(
        self,
        residual_model: ResidualCorrectionModel,
        feature_builder: CorrectionFeatureBuilder,
        clip_min: float = 0.0,
    ):
        self.residual_model = residual_model
        self.feature_builder = feature_builder
        self.clip_min = clip_min

    def apply(
        self,
        base_predictions: Dict[str, float],
        context_row: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """Apply residual corrections to a set of base predictions.

        Args:
            base_predictions: ``{stat: base_value}`` from the base model.
            context_row: Optional single-row DataFrame with player/game
                context features.

        Returns:
            Tuple of ``(corrected_predictions, correction_meta)`` where
            *corrected_predictions* is ``{stat: final_value}`` and
            *correction_meta* is ``{stat: {base_prediction, residual_correction,
            corrected_prediction, residual_applied}}``.
        """
        corrected: Dict[str, float] = {}
        correction_meta: Dict[str, Dict[str, Any]] = {}

        feature_cols = self.residual_model.feature_cols

        for stat in TARGETS:
            base_value = float(base_predictions.get(stat, 0.0))

            if self.residual_model.is_enabled(stat):
                feature_row = self.feature_builder.build_runtime_row(
                    stat=stat,
                    base_prediction=base_value,
                    context_row=context_row,
                    feature_cols=feature_cols or None,
                )
                try:
                    correction = float(
                        self.residual_model.predict_correction(stat, feature_row)
                    )
                except Exception as exc:
                    logger.warning(
                        "Correction failed for %s, falling back to 0.0: %s",
                        stat,
                        exc,
                    )
                    correction = 0.0
            else:
                correction = 0.0

            final_value = max(self.clip_min, base_value + correction)

            corrected[stat] = final_value
            correction_meta[stat] = {
                "base_prediction": base_value,
                "residual_correction": correction,
                "corrected_prediction": final_value,
                "residual_applied": correction != 0.0,
            }

        return corrected, correction_meta
