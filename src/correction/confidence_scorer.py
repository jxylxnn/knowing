"""Confidence labels for residual-calibrated stat projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ConfidenceResult:
    label: str
    score: float


class ConfidenceScorer:
    """Convert calibration/context signals into a coarse confidence label."""

    LABELS = ("HIGH", "MEDIUM", "LOW", "NO_EDGE")

    STAT_WIDTH_THRESHOLDS = {
        "PTS": 7.5,
        "REB": 4.0,
        "AST": 3.5,
        "STL": 1.6,
        "BLK": 1.5,
        "TOV": 2.2,
    }

    def score(
        self,
        stat: str,
        interval_width: Optional[float],
        data_quality: str = "FULL",
        minutes_confidence: Optional[float] = None,
        residual_applied: bool = False,
        residual_model_enabled: bool = True,
    ) -> ConfidenceResult:
        """Return a confidence label and numeric score."""
        if interval_width is None:
            return ConfidenceResult("NO_EDGE", 0.0)

        stat_upper = str(stat).upper()
        score = 70.0

        threshold = self.STAT_WIDTH_THRESHOLDS.get(stat_upper, 4.0)
        width = float(interval_width)
        if width <= threshold * 0.6:
            score += 15.0
        elif width > threshold:
            score -= 25.0
        elif width > threshold * 0.8:
            score -= 10.0

        if str(data_quality or "FULL").upper() != "FULL":
            score -= 20.0

        if minutes_confidence is not None:
            minutes = float(minutes_confidence)
            if minutes < 0.5:
                score -= 20.0
            elif minutes >= 0.75:
                score += 8.0

        if residual_applied:
            score += 10.0
        elif not residual_model_enabled:
            score -= 10.0

        score = max(0.0, min(100.0, score))
        if score >= 78.0:
            label = "HIGH"
        elif score >= 52.0:
            label = "MEDIUM"
        elif score >= 25.0:
            label = "LOW"
        else:
            label = "NO_EDGE"

        return ConfidenceResult(label, score)

    @staticmethod
    def minutes_confidence_from_context(context: Any) -> Optional[float]:
        """Extract a minutes confidence value from a row-like object."""
        if context is None:
            return None

        candidates = ("MINUTES_CONFIDENCE", "MIN_CONFIDENCE", "MINUTES_CONFIDENCE_SCORE")

        if hasattr(context, "columns"):
            for col in candidates:
                if col in context.columns and len(context) > 0:
                    return ConfidenceScorer._safe_float(context[col].iloc[0])

        if isinstance(context, Mapping):
            for col in candidates:
                if col in context:
                    return ConfidenceScorer._safe_float(context[col])

        if hasattr(context, "index"):
            for col in candidates:
                if col in context.index:
                    return ConfidenceScorer._safe_float(context.get(col))

        return None

    @staticmethod
    def data_quality_from_context(context: Any) -> str:
        """Extract DATA_QUALITY from a row-like object."""
        if context is None:
            return "FULL"

        if hasattr(context, "columns") and "DATA_QUALITY" in context.columns and len(context) > 0:
            return str(context["DATA_QUALITY"].iloc[0] or "FULL").upper()
        if isinstance(context, Mapping) and "DATA_QUALITY" in context:
            return str(context["DATA_QUALITY"] or "FULL").upper()
        if hasattr(context, "index") and "DATA_QUALITY" in context.index:
            return str(context.get("DATA_QUALITY") or "FULL").upper()
        return "FULL"

    @staticmethod
    def bucket_from_context(context: Any) -> str:
        """Select the calibration bucket for runtime prediction."""
        quality = ConfidenceScorer.data_quality_from_context(context)
        if quality != "FULL":
            return "DATA_QUALITY_DEGRADED"

        minutes = ConfidenceScorer.minutes_confidence_from_context(context)
        if minutes is not None:
            if minutes >= 0.7:
                return "HIGH_MINUTES_CONFIDENCE"
            if minutes < 0.5:
                return "LOW_MINUTES_CONFIDENCE"

        return "DATA_QUALITY_FULL"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
