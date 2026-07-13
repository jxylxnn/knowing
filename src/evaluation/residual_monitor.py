"""Residual correction monitoring — proves whether corrections help or hurt.

Compares base prediction error against corrected prediction error for every
target stat (PTS, REB, AST, STL, BLK, TOV) and produces structured per-target
diagnostics:

    * Base vs corrected MAE
    * Improvement percentage
    * Hit rate / harm rate
    * Bias shifts
    * Data-quality breakdown
    * Confidence-label breakdown
    * Rolling-window status
    * Status labels (HELPING / NEUTRAL / HURTING / INSUFFICIENT_DATA)
    * Recommendations (KEEP_ENABLED / DISABLE_CORRECTION / NEUTRAL_REVIEW)

The monitor is intentionally pure: it takes a DataFrame of prediction
history and returns a structured report.  No file I/O happens here.
Persistence is the responsibility of :mod:`src.evaluation.residual_report`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Canonical six stats the system predicts
DEFAULT_TARGETS: Tuple[str, ...] = ("PTS", "REB", "AST", "STL", "BLK", "TOV")

# Data quality buckets the residual system understands
DEFAULT_DATA_QUALITIES: Tuple[str, ...] = (
    "FULL",
    "DEGRADED_FALLBACK",
    "DEGRADED_MISSING",
)

# Confidence labels the scorer emits
DEFAULT_CONFIDENCE_LABELS: Tuple[str, ...] = (
    "HIGH",
    "MEDIUM",
    "LOW",
    "NO_EDGE",
)

# Status labels
STATUS_HELPING = "HELPING"
STATUS_NEUTRAL = "NEUTRAL"
STATUS_HURTING = "HURTING"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Recommendation labels
RECOMMENDATION_KEEP = "KEEP_ENABLED"
RECOMMENDATION_DISABLE = "DISABLE_CORRECTION"
RECOMMENDATION_REVIEW = "NEUTRAL_REVIEW"
RECOMMENDATION_INSUFFICIENT = "INSUFFICIENT_DATA"


@dataclass
class MonitoringThresholds:
    """Configurable thresholds for status / recommendation rules."""

    min_rows: int = 500
    helping_threshold_pct: float = 1.0      # corrected_mae must improve by ≥ this
    hurting_threshold_pct: float = -1.0     # corrected_mae must be worse by ≥ this
    neutral_band_pct: float = 1.0           # ± this percent counts as NEUTRAL
    min_window_rows: int = 50               # per-window minimum for status flag

    def status_for_pct(self, mae_improvement_pct: float, rows: int) -> str:
        """Return the status label implied by the percent improvement.

        Uses the global ``min_rows`` threshold.  For rolling-window status,
        use :meth:`window_status_for_pct` instead.
        """
        if rows < self.min_rows:
            return STATUS_INSUFFICIENT_DATA
        return self._classify_pct(mae_improvement_pct)

    def window_status_for_pct(self, mae_improvement_pct: float, rows: int) -> str:
        """Return the status label for a rolling window.

        Uses the (much smaller) ``min_window_rows`` threshold so that short
        windows are not silently demoted to ``INSUFFICIENT_DATA`` whenever
        ``min_rows`` is set high.
        """
        if rows < self.min_window_rows:
            return STATUS_INSUFFICIENT_DATA
        return self._classify_pct(mae_improvement_pct)

    def _classify_pct(self, mae_improvement_pct: float) -> str:
        """Apply HELPING / HURTING / NEUTRAL rules to a percent improvement.

        ``neutral_band_pct`` defines the symmetric dead-zone around 0%.
        Outside that band the strict helping / hurting thresholds apply.
        NaN inputs always fall through to NEUTRAL.
        """
        pct = mae_improvement_pct
        if pct is None or not np.isfinite(float(pct)):
            return STATUS_NEUTRAL
        pct = float(pct)
        if abs(pct) <= self.neutral_band_pct:
            return STATUS_NEUTRAL
        if pct >= self.helping_threshold_pct:
            return STATUS_HELPING
        if pct <= self.hurting_threshold_pct:
            return STATUS_HURTING
        return STATUS_NEUTRAL

    def recommendation_for_status(
        self,
        status: str,
        mae_improvement_pct: float,
    ) -> str:
        """Map a status + percent improvement to a recommendation."""
        if status == STATUS_INSUFFICIENT_DATA:
            return RECOMMENDATION_INSUFFICIENT
        if status == STATUS_HURTING:
            return RECOMMENDATION_DISABLE
        if status == STATUS_HELPING:
            return RECOMMENDATION_KEEP
        # NEUTRAL — small positive improvements still keep, larger
        # middling ranges warrant review.
        if mae_improvement_pct >= 0.0:
            return RECOMMENDATION_KEEP
        return RECOMMENDATION_REVIEW


@dataclass
class StatMetrics:
    """Per-target metric block (overall or a slice)."""

    rows: int = 0
    base_mae: float = math.nan
    corrected_mae: float = math.nan
    mae_improvement: float = math.nan
    mae_improvement_pct: float = math.nan
    base_bias: float = math.nan
    corrected_bias: float = math.nan
    base_rmse: float = math.nan
    corrected_rmse: float = math.nan
    correction_hit_rate: float = math.nan
    harm_rate: float = math.nan
    neutral_rate: float = math.nan

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict (NaN → None)."""
        return _scrub_nans({
            "rows": self.rows,
            "base_mae": self.base_mae,
            "corrected_mae": self.corrected_mae,
            "mae_improvement": self.mae_improvement,
            "mae_improvement_pct": self.mae_improvement_pct,
            "base_bias": self.base_bias,
            "corrected_bias": self.corrected_bias,
            "base_rmse": self.base_rmse,
            "corrected_rmse": self.corrected_rmse,
            "correction_hit_rate": self.correction_hit_rate,
            "harm_rate": self.harm_rate,
            "neutral_rate": self.neutral_rate,
        })


@dataclass
class StatReport:
    """Full per-target report block."""

    target: str
    overall: StatMetrics = field(default_factory=StatMetrics)
    status: str = STATUS_INSUFFICIENT_DATA
    recommendation: str = RECOMMENDATION_INSUFFICIENT

    by_data_quality: Dict[str, StatMetrics] = field(default_factory=dict)
    by_confidence: Dict[str, StatMetrics] = field(default_factory=dict)
    rolling_windows: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "rows": self.overall.rows,
            "status": self.status,
            "recommendation": self.recommendation,
            "overall": self.overall.to_dict(),
            "by_data_quality": {
                k: v.to_dict() for k, v in self.by_data_quality.items()
            },
            "by_confidence": {
                k: v.to_dict() for k, v in self.by_confidence.items()
            },
            "rolling_windows": self.rolling_windows,
        }


@dataclass
class MonitoringReport:
    """Aggregate monitoring report across all targets."""

    timestamp: str
    input_path: str
    targets: Dict[str, StatReport] = field(default_factory=dict)
    windows_days: Tuple[int, ...] = (7, 14, 30)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    recommendations: Dict[str, str] = field(default_factory=dict)
    overall_status: str = STATUS_INSUFFICIENT_DATA
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "input_path": self.input_path,
            "windows_days": list(self.windows_days),
            "thresholds": self.thresholds,
            "overall_status": self.overall_status,
            "recommendations": self.recommendations,
            "targets": {t: r.to_dict() for t, r in self.targets.items()},
            "summary": self.summary,
        }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _scrub_nans(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN/inf floats with None for JSON serialization."""
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            if not np.isfinite(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned


def _safe_float(value: Any) -> float:
    """Convert a value to float, returning NaN for non-numeric inputs."""
    if value is None:
        return math.nan
    try:
        f = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(f):
        return math.nan
    return f


def _series_float(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a float Series for *column*, with non-numeric values dropped."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _resolve_corrected_prediction(df: pd.DataFrame) -> pd.Series:
    """Return the corrected prediction series.

    Prefers ``CORRECTED_PREDICTION`` when present.  Otherwise derives it from
    ``BASE_PREDICTION`` + ``RESIDUAL_CORRECTION`` (or, failing that, leaves
    NaN so the row is filtered downstream).

    If both ``CORRECTED_PREDICTION`` and ``RESIDUAL_CORRECTION`` are present,
    rows with a NaN ``CORRECTED_PREDICTION`` fall back row-by-row to
    ``BASE_PREDICTION + RESIDUAL_CORRECTION`` so partial gaps in the
    corrected column don't drop otherwise-valid rows.
    """
    if "CORRECTED_PREDICTION" in df.columns:
        corrected = pd.to_numeric(df["CORRECTED_PREDICTION"], errors="coerce")
        if "RESIDUAL_CORRECTION" in df.columns:
            base = _series_float(df, "BASE_PREDICTION")
            residual = pd.to_numeric(df["RESIDUAL_CORRECTION"], errors="coerce")
            derived = base.add(residual, fill_value=math.nan)
            corrected = corrected.fillna(derived)
        return corrected

    base = _series_float(df, "BASE_PREDICTION")
    if "RESIDUAL_CORRECTION" in df.columns:
        residual = pd.to_numeric(df["RESIDUAL_CORRECTION"], errors="coerce")
        return base.add(residual, fill_value=math.nan)

    return pd.Series(dtype=float, index=df.index)


def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing ACTUAL / BASE_PREDICTION / corrected prediction."""
    actual = _series_float(df, "ACTUAL")
    base = _series_float(df, "BASE_PREDICTION")
    corrected = _resolve_corrected_prediction(df)

    valid = (
        actual.notna()
        & base.notna()
        & corrected.notna()
        & np.isfinite(actual.values)
        & np.isfinite(base.values)
        & np.isfinite(corrected.values)
    )
    return df.loc[valid].copy()


def _compute_stat_metrics(
    df: pd.DataFrame,
    actual_col: str = "ACTUAL",
    base_col: str = "BASE_PREDICTION",
    corrected_col: str = "_CORRECTED",
) -> StatMetrics:
    """Compute the core per-target metric block for *df*."""
    if df.empty:
        return StatMetrics()

    actual = pd.to_numeric(df[actual_col], errors="coerce").to_numpy(dtype=float)
    base = pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=float)
    corrected = pd.to_numeric(df[corrected_col], errors="coerce").to_numpy(dtype=float)

    base_err = np.abs(actual - base)
    corrected_err = np.abs(actual - corrected)
    improvement = base_err - corrected_err

    base_mae = float(np.mean(base_err))
    corrected_mae = float(np.mean(corrected_err))
    mae_improvement = base_mae - corrected_mae
    mae_improvement_pct = (
        (mae_improvement / base_mae) * 100.0 if base_mae > 0 else math.nan
    )

    base_bias = float(np.mean(actual - base))
    corrected_bias = float(np.mean(actual - corrected))
    base_rmse = float(np.sqrt(np.mean(base_err ** 2)))
    corrected_rmse = float(np.sqrt(np.mean(corrected_err ** 2)))

    hit_rate = float(np.mean(improvement > 0))
    harm_rate = float(np.mean(improvement < 0))
    neutral_rate = float(np.mean(improvement == 0))

    return StatMetrics(
        rows=int(len(df)),
        base_mae=base_mae,
        corrected_mae=corrected_mae,
        mae_improvement=mae_improvement,
        mae_improvement_pct=mae_improvement_pct,
        base_bias=base_bias,
        corrected_bias=corrected_bias,
        base_rmse=base_rmse,
        corrected_rmse=corrected_rmse,
        correction_hit_rate=hit_rate,
        harm_rate=harm_rate,
        neutral_rate=neutral_rate,
    )


def _ensure_corrected_column(df: pd.DataFrame) -> pd.DataFrame:
    """Materialise a CORRECTED_PREDICTION column when absent.

    When both ``CORRECTED_PREDICTION`` and ``RESIDUAL_CORRECTION`` are
    present, NaN entries in ``CORRECTED_PREDICTION`` are filled in
    row-by-row from ``BASE_PREDICTION + RESIDUAL_CORRECTION``.
    """
    if "CORRECTED_PREDICTION" in df.columns:
        out = df.copy()
        corrected = pd.to_numeric(out["CORRECTED_PREDICTION"], errors="coerce")
        if "RESIDUAL_CORRECTION" in out.columns:
            derived = (
                pd.to_numeric(out["BASE_PREDICTION"], errors="coerce")
                + pd.to_numeric(out["RESIDUAL_CORRECTION"], errors="coerce")
            )
            corrected = corrected.fillna(derived)
        out["_CORRECTED"] = corrected
        return out

    if "RESIDUAL_CORRECTION" in df.columns:
        out = df.copy()
        out["_CORRECTED"] = (
            pd.to_numeric(out["BASE_PREDICTION"], errors="coerce")
            + pd.to_numeric(out["RESIDUAL_CORRECTION"], errors="coerce")
        )
        return out

    raise ValueError(
        "Residual monitor input must contain CORRECTED_PREDICTION or "
        "RESIDUAL_CORRECTION + BASE_PREDICTION."
    )


# ---------------------------------------------------------------------
# ResidualMonitor
# ---------------------------------------------------------------------


class ResidualMonitor:
    """Compute the structured monitoring report for residual corrections.

    Usage::

        monitor = ResidualMonitor()
        report = monitor.evaluate(history_df)
        payload = report.to_dict()
    """

    REQUIRED_BASE_COLUMNS: Tuple[str, ...] = (
        "GAME_DATE",
        "PLAYER_ID",
        "STAT",
        "BASE_PREDICTION",
        "ACTUAL",
    )

    def __init__(
        self,
        targets: Iterable[str] = DEFAULT_TARGETS,
        windows: Iterable[int] = (7, 14, 30),
        thresholds: Optional[MonitoringThresholds] = None,
        data_qualities: Iterable[str] = DEFAULT_DATA_QUALITIES,
        confidence_labels: Iterable[str] = DEFAULT_CONFIDENCE_LABELS,
    ) -> None:
        self.targets: Tuple[str, ...] = tuple(str(t).upper() for t in targets)
        self.windows: Tuple[int, ...] = tuple(int(w) for w in windows)
        self.thresholds = thresholds or MonitoringThresholds()
        self.data_qualities: Tuple[str, ...] = tuple(
            str(q).upper() for q in data_qualities
        )
        self.confidence_labels: Tuple[str, ...] = tuple(
            str(c).upper() for c in confidence_labels
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_BASE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "Residual monitor input is missing required columns: "
                + ", ".join(missing)
            )
        if "CORRECTED_PREDICTION" not in df.columns and "RESIDUAL_CORRECTION" not in df.columns:
            raise ValueError(
                "Residual monitor input must contain CORRECTED_PREDICTION or "
                "RESIDUAL_CORRECTION."
            )

    # ------------------------------------------------------------------
    # Slice helpers
    # ------------------------------------------------------------------

    def _slice_for_target(self, df: pd.DataFrame, stat: str) -> pd.DataFrame:
        """Filter to rows for the given target stat."""
        return df.loc[df["STAT"].astype(str).str.upper() == stat.upper()].copy()

    def _slice_by_quality(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Group rows by DATA_QUALITY bucket.

        NaN values are filled to ``"UNKNOWN"`` (not "FULL") so they don't
        masquerade as a real quality bucket.  Values outside the
        configured quality list are also routed to ``UNKNOWN``.
        """
        if "DATA_QUALITY" not in df.columns:
            return {}

        quality = (
            df["DATA_QUALITY"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
        )
        quality = quality.where(
            quality.isin(self.data_qualities), "UNKNOWN"
        )

        out: Dict[str, pd.DataFrame] = {}
        for q in self.data_qualities:
            sub = df.loc[quality == q]
            if not sub.empty:
                out[q] = sub
        # Include any unexpected quality buckets so the report doesn't hide data
        for q in quality.unique():
            if q not in out:
                sub = df.loc[quality == q]
                if not sub.empty:
                    out[q] = sub
        return out

    def _slice_by_confidence(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Group rows by CONFIDENCE label.

        NaN values are filled to ``"NO_EDGE"`` and unknown labels are
        routed to ``"NO_EDGE"`` so the report does not invent fake
        confidence buckets.
        """
        if "CONFIDENCE" not in df.columns:
            return {}

        conf = (
            df["CONFIDENCE"]
            .fillna("NO_EDGE")
            .astype(str)
            .str.upper()
        )
        conf = conf.where(
            conf.isin(self.confidence_labels), "NO_EDGE"
        )

        out: Dict[str, pd.DataFrame] = {}
        for c in self.confidence_labels:
            sub = df.loc[conf == c]
            if not sub.empty:
                out[c] = sub
        for c in conf.unique():
            if c not in out:
                sub = df.loc[conf == c]
                if not sub.empty:
                    out[c] = sub
        return out

    def _rolling_windows(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute status / metrics for each rolling window.

        "season_to_date" uses the full df as the window.
        """
        if "GAME_DATE" not in df.columns or df.empty:
            return {}

        dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.assign(_GAME_DATE=dates)
        df = df.dropna(subset=["_GAME_DATE"])
        if df.empty:
            return {}

        most_recent = df["_GAME_DATE"].max()
        out: Dict[str, Dict[str, Any]] = {}

        for window in self.windows:
            cutoff = most_recent - pd.Timedelta(days=window)
            sub = df.loc[df["_GAME_DATE"] >= cutoff]
            metrics = _compute_stat_metrics(sub)
            status = self.thresholds.window_status_for_pct(
                metrics.mae_improvement_pct if metrics.mae_improvement_pct is not None else math.nan,
                metrics.rows,
            )
            out[f"last_{window}_days"] = {
                "rows": metrics.rows,
                "base_mae": metrics.base_mae,
                "corrected_mae": metrics.corrected_mae,
                "mae_improvement_pct": metrics.mae_improvement_pct,
                "status": status,
            }

        metrics = _compute_stat_metrics(df)
        status = self.thresholds.status_for_pct(
            metrics.mae_improvement_pct if metrics.mae_improvement_pct is not None else math.nan,
            metrics.rows,
        )
        out["season_to_date"] = {
            "rows": metrics.rows,
            "base_mae": metrics.base_mae,
            "corrected_mae": metrics.corrected_mae,
            "mae_improvement_pct": metrics.mae_improvement_pct,
            "status": status,
        }

        return out

    # ------------------------------------------------------------------
    # Top-level evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        input_path: str = "",
        timestamp: Optional[str] = None,
    ) -> MonitoringReport:
        """Run the full monitoring pipeline and return a :class:`MonitoringReport`."""
        self._validate_input(df)
        df = _filter_valid_rows(df)
        df = _ensure_corrected_column(df)

        timestamp = timestamp or datetime.now().isoformat()

        report = MonitoringReport(
            timestamp=timestamp,
            input_path=str(input_path),
            targets={},
            windows_days=self.windows,
            thresholds={
                "min_rows": self.thresholds.min_rows,
                "helping_threshold_pct": self.thresholds.helping_threshold_pct,
                "hurting_threshold_pct": self.thresholds.hurting_threshold_pct,
                "neutral_band_pct": self.thresholds.neutral_band_pct,
                "min_window_rows": self.thresholds.min_window_rows,
            },
        )

        for stat in self.targets:
            sub = self._slice_for_target(df, stat)
            stat_report = self._evaluate_stat(stat, sub)
            report.targets[stat] = stat_report
            report.recommendations[stat] = stat_report.recommendation

        report.overall_status = self._aggregate_status(report)
        report.summary = self._build_summary(report)
        return report

    # ------------------------------------------------------------------
    # Per-target pipeline
    # ------------------------------------------------------------------

    def _evaluate_stat(self, stat: str, df: pd.DataFrame) -> StatReport:
        """Compute the per-target report block."""
        overall = _compute_stat_metrics(df)
        pct = overall.mae_improvement_pct
        status = self.thresholds.status_for_pct(pct, overall.rows)
        recommendation = self.thresholds.recommendation_for_status(status, pct)

        stat_report = StatReport(
            target=stat,
            overall=overall,
            status=status,
            recommendation=recommendation,
        )

        # Data-quality breakdown
        for quality, sub in self._slice_by_quality(df).items():
            stat_report.by_data_quality[quality] = _compute_stat_metrics(sub)

        # Confidence breakdown
        for label, sub in self._slice_by_confidence(df).items():
            stat_report.by_confidence[label] = _compute_stat_metrics(sub)

        # Rolling windows
        stat_report.rolling_windows = self._rolling_windows(df)

        return stat_report

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_status(self, report: MonitoringReport) -> str:
        """Pick the worst status across all targets.

        HURTING dominates — if any correction model is hurting, the
        report must surface that fact and not hide it behind
        ``INSUFFICIENT_DATA`` from another stat.
        """
        # Explicit priority order: HURTING wins over INSUFFICIENT_DATA,
        # INSUFFICIENT_DATA wins over NEUTRAL, NEUTRAL wins over HELPING.
        if any(r.status == STATUS_HURTING for r in report.targets.values()):
            return STATUS_HURTING
        if any(r.status == STATUS_INSUFFICIENT_DATA for r in report.targets.values()):
            return STATUS_INSUFFICIENT_DATA
        if any(r.status == STATUS_NEUTRAL for r in report.targets.values()):
            return STATUS_NEUTRAL
        return STATUS_HELPING

    def _build_summary(self, report: MonitoringReport) -> Dict[str, Any]:
        """Generate high-level summary counts."""
        counts = {
            STATUS_HELPING: 0,
            STATUS_NEUTRAL: 0,
            STATUS_HURTING: 0,
            STATUS_INSUFFICIENT_DATA: 0,
        }
        for stat_report in report.targets.values():
            counts[stat_report.status] = counts.get(stat_report.status, 0) + 1
        return {
            "status_counts": counts,
            "total_targets": len(report.targets),
            "kept": sum(
                1 for r in report.targets.values()
                if r.recommendation == RECOMMENDATION_KEEP
            ),
            "disabled": sum(
                1 for r in report.targets.values()
                if r.recommendation == RECOMMENDATION_DISABLE
            ),
            "review": sum(
                1 for r in report.targets.values()
                if r.recommendation == RECOMMENDATION_REVIEW
            ),
            "insufficient": sum(
                1 for r in report.targets.values()
                if r.recommendation == RECOMMENDATION_INSUFFICIENT
            ),
        }

    # ------------------------------------------------------------------
    # Helpers used by the report writer / CLI
    # ------------------------------------------------------------------

    def rows_for(self, df: pd.DataFrame, stat: str) -> pd.DataFrame:
        """Filter *df* to the given target with a CORRECTED column guaranteed."""
        df = _ensure_corrected_column(df)
        return self._slice_for_target(df, stat)

    @staticmethod
    def load_parquet(path: str) -> pd.DataFrame:
        """Read a parquet file and validate it has the required shape."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return pd.read_parquet(p)

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """Read a CSV file and validate it has the required shape."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return pd.read_csv(p)

    @staticmethod
    def load_input(
        path: str,
        required_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Load parquet or CSV based on extension.

        ``required_columns`` allows callers (e.g. tests) to enforce extra
        columns beyond the always-required monitor columns.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = p.suffix.lower()
        if suffix in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    "Residual monitor input is missing required columns: "
                    + ", ".join(missing)
                )
        return df


__all__ = [
    "DEFAULT_CONFIDENCE_LABELS",
    "DEFAULT_DATA_QUALITIES",
    "DEFAULT_TARGETS",
    "MonitoringReport",
    "MonitoringThresholds",
    "RECOMMENDATION_DISABLE",
    "RECOMMENDATION_INSUFFICIENT",
    "RECOMMENDATION_KEEP",
    "RECOMMENDATION_REVIEW",
    "ResidualMonitor",
    "STAT_REPORT_FIELDS",
    "STATUS_HELPING",
    "STATUS_HURTING",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_NEUTRAL",
    "StatMetrics",
    "StatReport",
]


# A tiny safety net for any consumer that imports the full tuple of fields
STAT_REPORT_FIELDS: Tuple[str, ...] = tuple(StatMetrics().__dict__.keys())
