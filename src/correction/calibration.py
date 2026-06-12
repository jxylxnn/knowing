"""Conformal interval calibration from residual prediction errors."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
DEFAULT_CONFIDENCE_LEVELS = (0.8, 0.9, 0.95)


@dataclass(frozen=True)
class PredictionInterval:
    """Lower/upper prediction interval bounds."""

    low: float
    high: float


class ResidualIntervalCalibrator:
    """Build per-stat conformal interval artifacts from residual errors."""

    REQUIRED_COLUMNS = {
        "STAT",
        "BASE_PREDICTION",
        "ACTUAL",
        "ERROR",
        "GAME_DATE",
        "DATA_QUALITY",
        "PLAYER_ID",
        "MODEL_FOLD",
    }

    def __init__(
        self,
        confidence_levels: Iterable[float] = DEFAULT_CONFIDENCE_LEVELS,
        min_bucket_rows: int = 500,
        targets: Iterable[str] = TARGETS,
    ):
        self.confidence_levels = tuple(sorted(float(v) for v in confidence_levels))
        self.min_bucket_rows = int(min_bucket_rows)
        self.targets = tuple(str(t).upper() for t in targets)

    @staticmethod
    def make_interval(prediction: float, width: float) -> PredictionInterval:
        """Create a clipped lower/upper interval around a prediction."""
        pred = float(prediction)
        w = max(0.0, float(width))
        return PredictionInterval(low=max(0.0, pred - w), high=pred + w)

    def calibrate_file(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Read residual parquet and write interval artifacts."""
        residual_df = pd.read_parquet(input_path)
        return self.calibrate(residual_df, output_dir)

    def calibrate(self, residual_df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """Build interval artifacts for all configured targets."""
        self._validate_input(residual_df)
        df = self._with_calibration_error(residual_df)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metadata: Dict[str, Any] = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "input_rows": int(len(df)),
            "confidence_levels": list(self.confidence_levels),
            "min_bucket_rows": self.min_bucket_rows,
            "targets": {},
        }

        for stat in self.targets:
            stat_df = df[df["STAT"].astype(str).str.upper() == stat].copy()
            if stat_df.empty:
                metadata["targets"][stat] = {
                    "rows": 0,
                    "status": "skipped",
                    "reason": "no rows for target",
                }
                continue

            artifact = self._build_stat_artifact(stat, stat_df)
            (out / f"{stat.lower()}_intervals.json").write_text(
                json.dumps(artifact, indent=2),
                encoding="utf-8",
            )
            metadata["targets"][stat] = {
                "rows": int(len(stat_df)),
                "status": "written",
                "buckets": {
                    name: {
                        "rows": bucket["rows"],
                        "coverage": bucket.get("coverage", {}),
                    }
                    for name, bucket in artifact["buckets"].items()
                },
            }

        (out / "calibration_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote residual interval calibration artifacts to %s", out)
        return metadata

    def _validate_input(self, residual_df: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS - set(residual_df.columns))
        if missing:
            raise ValueError(
                "Residual calibration input is missing required columns: "
                + ", ".join(missing)
            )
        if not self.confidence_levels:
            raise ValueError("At least one confidence level is required")
        for level in self.confidence_levels:
            if level <= 0.0 or level >= 1.0:
                raise ValueError(f"Confidence level must be in (0, 1): {level}")

    def _with_calibration_error(self, residual_df: pd.DataFrame) -> pd.DataFrame:
        df = residual_df.copy()
        if "CORRECTED_PREDICTION" in df.columns:
            df["CALIBRATION_ERROR"] = (
                pd.to_numeric(df["ACTUAL"], errors="coerce")
                - pd.to_numeric(df["CORRECTED_PREDICTION"], errors="coerce")
            ).abs()
        elif "CORRECTED_ERROR" in df.columns:
            df["CALIBRATION_ERROR"] = pd.to_numeric(
                df["CORRECTED_ERROR"], errors="coerce"
            ).abs()
        else:
            df["CALIBRATION_ERROR"] = pd.to_numeric(df["ERROR"], errors="coerce").abs()

        df["CALIBRATION_ERROR"] = df["CALIBRATION_ERROR"].replace(
            [np.inf, -np.inf], np.nan
        )
        return df.dropna(subset=["CALIBRATION_ERROR"])

    def _build_stat_artifact(self, stat: str, stat_df: pd.DataFrame) -> Dict[str, Any]:
        buckets: Dict[str, Dict[str, Any]] = {}
        for name, bucket_df, min_rows in self._bucket_frames(stat_df):
            if len(bucket_df) < min_rows:
                continue
            buckets[name] = self._summarize_bucket(bucket_df)

        if "GLOBAL" not in buckets:
            buckets["GLOBAL"] = self._summarize_bucket(stat_df)

        return {
            "version": 1,
            "stat": stat,
            "created_at": datetime.now().isoformat(),
            "confidence_levels": list(self.confidence_levels),
            "buckets": buckets,
        }

    def _bucket_frames(
        self, stat_df: pd.DataFrame
    ) -> Iterable[Tuple[str, pd.DataFrame, int]]:
        yield "GLOBAL", stat_df, 1

        quality = stat_df["DATA_QUALITY"].astype(str).str.upper()
        yield "DATA_QUALITY_FULL", stat_df[quality == "FULL"], self.min_bucket_rows
        yield (
            "DATA_QUALITY_DEGRADED",
            stat_df[(quality != "FULL") & (quality != "")],
            self.min_bucket_rows,
        )

        minutes_col = self._find_minutes_confidence_col(stat_df)
        if minutes_col:
            mins = pd.to_numeric(stat_df[minutes_col], errors="coerce")
            yield (
                "HIGH_MINUTES_CONFIDENCE",
                stat_df[mins >= 0.7],
                self.min_bucket_rows,
            )
            yield (
                "LOW_MINUTES_CONFIDENCE",
                stat_df[mins < 0.5],
                self.min_bucket_rows,
            )

        player_volatility = stat_df.groupby("PLAYER_ID")["CALIBRATION_ERROR"].mean()
        if len(player_volatility) >= 2:
            median_vol = float(player_volatility.median())
            high_ids = set(player_volatility[player_volatility >= median_vol].index)
            low_ids = set(player_volatility[player_volatility < median_vol].index)
            yield (
                "PLAYER_HIGH_VOLATILITY",
                stat_df[stat_df["PLAYER_ID"].isin(high_ids)],
                self.min_bucket_rows,
            )
            yield (
                "PLAYER_LOW_VOLATILITY",
                stat_df[stat_df["PLAYER_ID"].isin(low_ids)],
                self.min_bucket_rows,
            )

    @staticmethod
    def _find_minutes_confidence_col(df: pd.DataFrame) -> Optional[str]:
        for col in ("MINUTES_CONFIDENCE", "MIN_CONFIDENCE", "MINUTES_CONFIDENCE_SCORE"):
            if col in df.columns:
                return col
        return None

    def _summarize_bucket(self, bucket_df: pd.DataFrame) -> Dict[str, Any]:
        errors = pd.to_numeric(bucket_df["CALIBRATION_ERROR"], errors="coerce")
        errors = errors.replace([np.inf, -np.inf], np.nan).dropna()
        widths: Dict[str, float] = {}
        coverage: Dict[str, float] = {}

        for level in self.confidence_levels:
            key = self._level_key(level)
            width = float(np.quantile(errors.to_numpy(dtype=float), level))
            widths[key] = max(0.0, width)
            coverage[key] = float((errors <= width).mean())

        return {
            "rows": int(len(errors)),
            "widths": widths,
            "coverage": coverage,
        }

    @staticmethod
    def _level_key(level: float) -> str:
        return f"q{int(round(level * 100))}"
