"""Performance drift detector for NBA prediction models.

Tracks per-stat accuracy over rolling windows and detects when model
performance has degraded meaningfully, indicating that either:
  (a) blend weights need retuning (minor drift), or
  (b) models need full retraining (major drift).

Uses statistical process control: flags when rolling MAE exceeds 2σ
above the historical baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftStatus:
    """Status report for a single target stat."""

    target: str
    current_mae: float
    baseline_mae: float
    rolling_mae: float
    sigma_multiple: float  # how many σ above baseline
    status: str  # "ok", "warning", "critical"
    recommendation: str
    num_samples: int


@dataclass
class DriftReport:
    """Complete drift detection report across all targets."""

    timestamp: str
    overall_status: str  # "ok", "warning", "critical"
    per_target: Dict[str, DriftStatus] = field(default_factory=dict)
    window_days: int = 30
    baseline_window_days: int = 90

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Drift Report — {self.timestamp[:19]}",
            f"  Overall: {self.overall_status.upper()}",
            f"  Window: {self.window_days}d rolling | Baseline: {self.baseline_window_days}d",
            "",
        ]
        for target in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
            if target in self.per_target:
                ds = self.per_target[target]
                lines.append(
                    f"  {target:4s}: MAE={ds.current_mae:.3f} "
                    f"(baseline={ds.baseline_mae:.3f}, +{ds.sigma_multiple:.1f}σ) "
                    f"[{ds.status}] — {ds.recommendation}"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "window_days": self.window_days,
            "baseline_window_days": self.baseline_window_days,
            "per_target": {
                t: {
                    "target": ds.target,
                    "current_mae": ds.current_mae,
                    "baseline_mae": ds.baseline_mae,
                    "rolling_mae": ds.rolling_mae,
                    "sigma_multiple": ds.sigma_multiple,
                    "status": ds.status,
                    "recommendation": ds.recommendation,
                    "num_samples": ds.num_samples,
                }
                for t, ds in self.per_target.items()
            },
        }


class DriftDetector:
    """Monitor prediction accuracy and detect performance degradation.

    Tracks per-stat MAE over rolling windows and compares to a historical
    baseline. Uses σ-based thresholds:
      - < 1.5σ: OK (within normal variance)
      - 1.5–2.5σ: WARNING (minor drift — retune weights)
      - > 2.5σ: CRITICAL (major drift — retrain models)
    """

    WARNING_SIGMA = 1.5
    CRITICAL_SIGMA = 2.5

    def __init__(
        self,
        store_path: str = "models/drift_state.json",
        window_days: int = 30,
        baseline_window_days: int = 90,
    ):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.window_days = window_days
        self.baseline_window_days = baseline_window_days

        self._history: List[Dict] = self._load_history()

    # ------------------------------------------------------------------
    # Phase-aware baseline helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_phase_from_date(date_str: str) -> str:
        """Infer the NBA phase (REGULAR or PLAYOFF) from a date string.

        Rough heuristic: NBA regular season typically runs Oct 20 – Apr 15.
        Playoffs run Apr 15 – Jun 20. Anything outside this range defaults
        to REGULAR.
        """
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return "REGULAR"
        month_day = (dt.month, dt.day)
        # Playoff window: April 15 through June 20
        if (month_day >= (4, 15) and month_day <= (6, 20)):
            return "PLAYOFF"
        return "REGULAR"

    def _phase_key(self, phase: str, date: Optional[str] = None) -> str:
        """Return the effective phase key, inferring from date if not provided."""
        if phase and phase.upper() in ("REGULAR", "PLAYOFF"):
            return phase.upper()
        if date:
            return self._infer_phase_from_date(date)
        return "REGULAR"

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _load_history(self) -> List[Dict]:
        """Load drift history from disk."""
        if not self.store_path.exists():
            return []
        try:
            with open(self.store_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupted drift history, starting fresh")
            return []

    def _save_history(self) -> None:
        """Save drift history to disk (atomic write)."""
        import os
        import tempfile

        json_text = json.dumps(self._history, indent=2)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=".tmp_drift_", dir=str(self.store_path.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json_text)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(self.store_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Record an observation
    # ------------------------------------------------------------------

    def record(
        self,
        per_target_mae: Dict[str, float],
        per_target_samples: Optional[Dict[str, int]] = None,
        date: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Record per-target MAE values for a specific date.

        Args:
            per_target_mae: Dict mapping target name → MAE.
            per_target_samples: Optional dict mapping target name → sample count.
            date: Date string (YYYY-MM-DD). Defaults to today.
            phase: Game phase ('REGULAR' or 'PLAYOFF'). Auto-inferred from date if omitted.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        effective_phase = self._phase_key(phase or "", date)

        entry = {
            "date": date,
            "mae": per_target_mae,
            "samples": per_target_samples or {},
            "phase": effective_phase,
        }
        self._history.append(entry)

        # Prune history older than baseline window
        cutoff = datetime.now() - timedelta(days=self.baseline_window_days + 30)
        self._history = [
            e for e in self._history
            if datetime.strptime(e["date"], "%Y-%m-%d") >= cutoff
        ]

        self._save_history()
        logger.debug("Recorded drift observation for %s (%d targets)", date, len(per_target_mae))

    # ------------------------------------------------------------------
    # Detect drift
    # ------------------------------------------------------------------

    def detect(self, targets: Optional[List[str]] = None, phase: Optional[str] = None) -> DriftReport:
        """Run drift detection on the current history.

        Args:
            targets: List of target stats to check. Defaults to all in history.
            phase: Game phase ('REGULAR' or 'PLAYOFF'). When provided, only history
                   from that phase is used for baseline comparison, preventing false
                   drift alerts when the playoffs naturally lower scoring/pace.

        Returns:
            DriftReport with per-target status and recommendations.
        """
        if not self._history:
            return DriftReport(
                timestamp=datetime.now().isoformat(),
                overall_status="unknown",
                window_days=self.window_days,
                baseline_window_days=self.baseline_window_days,
            )

        now = datetime.now()
        rolling_cutoff = now - timedelta(days=self.window_days)
        baseline_cutoff = now - timedelta(days=self.baseline_window_days)

        # Filter by phase if specified
        history_pool = self._history
        if phase:
            effective_phase = self._phase_key(phase)
            history_pool = [e for e in self._history if e.get("phase") == effective_phase]
            if not history_pool:
                # Fall back to full history if no phase-specific data exists
                history_pool = self._history

        # Determine available targets
        all_targets = set()
        for entry in history_pool:
            all_targets.update(entry["mae"].keys())
        check_targets = targets or sorted(all_targets)

        per_target: Dict[str, DriftStatus] = {}
        overall_max_sigma = 0.0

        for target in check_targets:
            # Collect MAE values
            rolling_values = []
            baseline_values = []

            for entry in history_pool:
                if target not in entry["mae"]:
                    continue
                mae = entry["mae"][target]
                if not np.isfinite(mae):
                    continue

                entry_date = datetime.strptime(entry["date"], "%Y-%m-%d")
                if entry_date >= rolling_cutoff:
                    rolling_values.append(mae)
                elif entry_date >= baseline_cutoff:
                    baseline_values.append(mae)
                else:
                    baseline_values.append(mae)

            if len(rolling_values) < 3 or len(baseline_values) < 5:
                per_target[target] = DriftStatus(
                    target=target,
                    current_mae=float(np.mean(rolling_values)) if rolling_values else 0.0,
                    baseline_mae=float(np.mean(baseline_values)) if baseline_values else 0.0,
                    rolling_mae=float(np.mean(rolling_values)) if rolling_values else 0.0,
                    sigma_multiple=0.0,
                    status="unknown",
                    recommendation="Insufficient data for drift detection",
                    num_samples=len(rolling_values),
                )
                continue

            rolling_mean = float(np.mean(rolling_values))
            baseline_mean = float(np.mean(baseline_values))
            baseline_std = float(np.std(baseline_values))

            if baseline_std < 1e-8:
                # No variance in baseline — can't compute sigma
                sigma = 0.0
                status = "ok"
                recommendation = "No variance in baseline"
            else:
                sigma = (rolling_mean - baseline_mean) / baseline_std

                if sigma < self.WARNING_SIGMA:
                    status = "ok"
                    recommendation = "Within normal range"
                elif sigma < self.CRITICAL_SIGMA:
                    status = "warning"
                    recommendation = "Minor drift — consider retuning blend weights"
                else:
                    status = "critical"
                    recommendation = "Major drift — full model retraining recommended"

            overall_max_sigma = max(overall_max_sigma, sigma)
            per_target[target] = DriftStatus(
                target=target,
                current_mae=rolling_mean,
                baseline_mae=baseline_mean,
                rolling_mae=rolling_mean,
                sigma_multiple=round(sigma, 2),
                status=status,
                recommendation=recommendation,
                num_samples=len(rolling_values),
            )

        # Overall status
        if overall_max_sigma >= self.CRITICAL_SIGMA:
            overall_status = "critical"
        elif overall_max_sigma >= self.WARNING_SIGMA:
            overall_status = "warning"
        else:
            overall_status = "ok"

        return DriftReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            per_target=per_target,
            window_days=self.window_days,
            baseline_window_days=self.baseline_window_days,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def record_and_detect(
        self,
        per_target_mae: Dict[str, float],
        targets: Optional[List[str]] = None,
        date: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> DriftReport:
        """Record a new observation and immediately run drift detection.

        Args:
            per_target_mae: Dict mapping target → MAE.
            targets: Targets to check.
            date: Date of observation.
            phase: Game phase ('REGULAR' or 'PLAYOFF').

        Returns:
            DriftReport with current status.
        """
        self.record(per_target_mae, date=date, phase=phase)
        return self.detect(targets=targets, phase=phase)

    def status_summary(self) -> str:
        """Quick status string for use in logs/dashboards."""
        report = self.detect()
        if report.overall_status == "ok":
            return "DRIFT: OK — all targets within normal range"
        elif report.overall_status == "warning":
            drifted = [
                t for t, ds in report.per_target.items()
                if ds.status == "warning"
            ]
            return f"DRIFT: WARNING — {', '.join(drifted)} drifting"
        else:
            critical = [
                t for t, ds in report.per_target.items()
                if ds.status == "critical"
            ]
            return f"DRIFT: CRITICAL — {', '.join(critical)} severely degraded"
