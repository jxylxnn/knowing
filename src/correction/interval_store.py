"""Load and query residual interval calibration artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.correction.calibration import PredictionInterval, ResidualIntervalCalibrator

logger = logging.getLogger(__name__)


class CalibrationIntervalStore:
    """Best-effort loader for per-stat interval calibration files."""

    def __init__(self, calibration_dir: str = "models/calibration"):
        self.calibration_dir = Path(calibration_dir)
        self.metadata: Dict[str, Any] = {}
        self.intervals: Dict[str, Dict[str, Any]] = {}
        self.enabled = False

    def load(self, calibration_dir: Optional[str] = None) -> "CalibrationIntervalStore":
        """Load calibration artifacts if present.

        Missing or malformed artifacts disable the store without raising so
        runtime prediction can continue.
        """
        if calibration_dir is not None:
            self.calibration_dir = Path(calibration_dir)

        self.metadata = {}
        self.intervals = {}
        self.enabled = False

        metadata_path = self.calibration_dir / "calibration_metadata.json"
        if not metadata_path.exists():
            logger.debug("No interval calibration metadata at %s", metadata_path)
            return self

        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for path in sorted(self.calibration_dir.glob("*_intervals.json")):
                artifact = json.loads(path.read_text(encoding="utf-8"))
                stat = str(artifact.get("stat") or path.stem.split("_")[0]).upper()
                if artifact.get("buckets"):
                    self.intervals[stat] = artifact
            self.enabled = bool(self.intervals)
        except Exception as exc:
            logger.warning("Residual interval calibration disabled: %s", exc)
            self.metadata = {}
            self.intervals = {}
            self.enabled = False

        return self

    def has_stat(self, stat: str) -> bool:
        return self.enabled and str(stat).upper() in self.intervals

    def get_interval_width(
        self,
        stat: str,
        confidence: float = 0.9,
        bucket: str = "GLOBAL",
    ) -> Optional[float]:
        """Return interval half-width for a stat/confidence/bucket."""
        artifact = self.intervals.get(str(stat).upper())
        if not artifact:
            return None

        buckets = artifact.get("buckets", {})
        bucket_payload = buckets.get(bucket) or buckets.get("GLOBAL")
        if not bucket_payload:
            return None

        widths = bucket_payload.get("widths", {})
        key = self._confidence_key(confidence)
        value = widths.get(key)
        if value is None:
            return None
        return max(0.0, float(value))

    def make_interval(
        self,
        stat: str,
        prediction: float,
        confidence: float = 0.9,
        bucket: str = "GLOBAL",
    ) -> Optional[PredictionInterval]:
        """Return clipped prediction interval bounds when available."""
        width = self.get_interval_width(stat, confidence=confidence, bucket=bucket)
        if width is None:
            return None
        return ResidualIntervalCalibrator.make_interval(prediction, width)

    @staticmethod
    def _confidence_key(confidence: float) -> str:
        return f"q{int(round(float(confidence) * 100))}"
