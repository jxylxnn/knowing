"""Backtest metrics and result types for NBA prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TargetMetrics:
    """Per-target evaluation metrics from a backtest run."""

    target: str
    mae: float
    rmse: float
    r2: float
    mape: float = 0.0
    num_samples: int = 0

    # Quantile calibration: what % of actuals fall below P10 / P90
    calibration_p10: Optional[float] = None
    calibration_p90: Optional[float] = None

    # Bias metrics
    mean_error: float = 0.0
    median_error: float = 0.0

    # Prediction interval coverage
    std_mean: float = 0.0      # average predicted STD
    std_actual: float = 0.0    # actual residual std

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "target": self.target,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "num_samples": self.num_samples,
            "calibration_p10": self.calibration_p10,
            "calibration_p90": self.calibration_p90,
            "mean_error": self.mean_error,
            "median_error": self.median_error,
            "std_mean": self.std_mean,
            "std_actual": self.std_actual,
        }


@dataclass
class BacktestResult:
    """Complete backtest result for a date range."""

    date_start: str
    date_end: str
    num_games: int
    num_players: int
    per_target: Dict[str, TargetMetrics] = field(default_factory=dict)

    # Aggregate metrics
    overall_mae: float = 0.0
    overall_rmse: float = 0.0
    overall_r2: float = 0.0

    # Weighted by target importance (higher weight for core targets)
    weighted_score: float = 0.0

    # Metadata
    model_version: str = ""
    blend_version: int = 0
    timestamp: str = ""
    data_hash: str = ""

    @property
    def core_mae(self) -> float:
        """Average MAE across core targets (PTS, REB, AST)."""
        core = [m for t, m in self.per_target.items() if t in ("PTS", "REB", "AST")]
        if not core:
            return float("inf")
        return float(np.mean([m.mae for m in core]))

    @property
    def secondary_mae(self) -> float:
        """Average MAE across secondary targets (STL, BLK, TOV)."""
        secondary = [m for t, m in self.per_target.items() if t in ("STL", "BLK", "TOV")]
        if not secondary:
            return float("inf")
        return float(np.mean([m.mae for m in secondary]))

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "date_start": self.date_start,
            "date_end": self.date_end,
            "num_games": self.num_games,
            "num_players": self.num_players,
            "overall_mae": self.overall_mae,
            "overall_rmse": self.overall_rmse,
            "overall_r2": self.overall_r2,
            "weighted_score": self.weighted_score,
            "core_mae": self.core_mae,
            "secondary_mae": self.secondary_mae,
            "model_version": self.model_version,
            "blend_version": self.blend_version,
            "per_target": {t: m.to_dict() for t, m in self.per_target.items()},
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Backtest: {self.date_start} → {self.date_end}",
            f"  Games: {self.num_games} | Players: {self.num_players}",
            f"  Overall MAE: {self.overall_mae:.3f} | RMSE: {self.overall_rmse:.3f} | R²: {self.overall_r2:.3f}",
            f"  Core MAE (PTS/REB/AST): {self.core_mae:.3f}",
            f"  Secondary MAE (STL/BLK/TOV): {self.secondary_mae:.3f}",
            "",
        ]
        for target in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
            if target in self.per_target:
                m = self.per_target[target]
                cal = ""
                if m.calibration_p10 is not None and m.calibration_p90 is not None:
                    cal = f" | P10:{m.calibration_p10:.1%} P90:{m.calibration_p90:.1%}"
                lines.append(
                    f"  {target:4s}: MAE={m.mae:.3f} RMSE={m.rmse:.3f} "
                    f"R²={m.r2:.3f} MAPE={m.mape:.1%}{cal} (n={m.num_samples})"
                )
        return "\n".join(lines)


def compute_target_metrics(
    target: str,
    actuals: np.ndarray,
    predictions: np.ndarray,
    stds: Optional[np.ndarray] = None,
) -> TargetMetrics:
    """Compute per-target evaluation metrics.

    Args:
        target: Target stat name (e.g., 'PTS')
        actuals: Array of actual values
        predictions: Array of predicted values
        stds: Optional array of predicted standard deviations for calibration

    Returns:
        TargetMetrics dataclass with computed metrics
    """
    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    # Filter out NaN/inf
    mask = np.isfinite(actuals) & np.isfinite(predictions)
    actuals = actuals[mask]
    predictions = predictions[mask]

    if len(actuals) < 2:
        return TargetMetrics(
            target=target,
            mae=float("nan"),
            rmse=float("nan"),
            r2=float("nan"),
            num_samples=len(actuals),
        )

    residuals = actuals - predictions
    abs_residuals = np.abs(residuals)

    mae = float(np.mean(abs_residuals))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # R²
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # MAPE (with epsilon guard)
    eps = 1e-8
    mape = float(np.mean(abs_residuals / (np.abs(actuals) + eps)))

    # Bias
    mean_error = float(np.mean(residuals))
    median_error = float(np.median(residuals))

    # Calibration
    calibration_p10 = None
    calibration_p90 = None
    if stds is not None:
        stds_arr = np.asarray(stds, dtype=float)[mask]
        if len(stds_arr) > 0 and np.all(np.isfinite(stds_arr)):
            # P10: z = (actual - pred) / std, expect 10% below -1.28
            z_scores = residuals / (stds_arr + eps)
            calibration_p10 = float(np.mean(z_scores < -1.28))
            calibration_p90 = float(np.mean(z_scores < 1.28))

    metrics = TargetMetrics(
        target=target,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        num_samples=len(actuals),
        calibration_p10=calibration_p10,
        calibration_p90=calibration_p90,
        mean_error=mean_error,
        median_error=median_error,
        std_mean=float(np.mean(stds_arr)) if stds is not None and len(stds_arr) > 0 else 0.0,
        std_actual=float(np.std(residuals)),
    )

    return metrics
