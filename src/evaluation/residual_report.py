"""Residual correction monitoring — JSON / CSV report writing.

Writes the :class:`~src.evaluation.residual_monitor.MonitoringReport` to disk:

    reports/residual_monitoring/latest_summary.json
    reports/residual_monitoring/residual_report_<timestamp>.json
    reports/residual_monitoring/residual_report_<timestamp>.csv

Reports use atomic temp-file → rename to avoid corruption on crash.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.residual_monitor import (
    MonitoringReport,
    StatReport,
)

logger = logging.getLogger(__name__)


def _timestamp_slug(timestamp: Optional[str] = None) -> str:
    """Return a filesystem-safe timestamp slug."""
    raw = timestamp or datetime.now().isoformat()
    return raw.replace(":", "-").replace(".", "-")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_residual_report_",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``json.dumps(..., allow_nan=True)`` is the Python default, but it
    emits ``NaN`` / ``Infinity`` tokens that are not valid strict JSON
    and will fail in JavaScript dashboards, jq, and many other
    downstream consumers.  This helper walks any nested dict / list
    structure (the shape produced by :meth:`MonitoringReport.to_dict`)
    and converts non-finite floats to ``None`` so that
    ``json.dumps(..., allow_nan=False)`` can be used safely.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------


def write_json_report(
    report: MonitoringReport,
    output_path: str,
) -> Path:
    """Write the full monitoring report to a JSON file.

    Returns the path that was written.

    All non-finite floats are converted to ``null`` via :func:`_json_safe`
    and ``json.dumps`` is called with ``allow_nan=False`` so the output
    is strict RFC-8259 JSON.
    """
    path = Path(output_path)
    payload = _json_safe(report.to_dict())
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    _atomic_write_text(path, text)
    logger.info("Wrote residual monitoring JSON to %s", path)
    return path


def write_latest_summary(
    report: MonitoringReport,
    output_dir: str,
) -> Path:
    """Write a copy of the report to ``latest_summary.json``.

    The contents are identical to the timestamped report, but the
    filename is stable so downstream tools (dashboards, alerts) can
    always read the most recent run.
    """
    out = Path(output_dir) / "latest_summary.json"
    return write_json_report(report, str(out))


# ---------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------


def _flatten_stat_row(stat_report: StatReport) -> Dict[str, Any]:
    """Flatten a :class:`StatReport` into a single CSV row."""
    overall = stat_report.overall
    row: Dict[str, Any] = {
        "target": stat_report.target,
        "status": stat_report.status,
        "recommendation": stat_report.recommendation,
        "rows": overall.rows,
        "base_mae": overall.base_mae,
        "corrected_mae": overall.corrected_mae,
        "mae_improvement": overall.mae_improvement,
        "mae_improvement_pct": overall.mae_improvement_pct,
        "base_bias": overall.base_bias,
        "corrected_bias": overall.corrected_bias,
        "base_rmse": overall.base_rmse,
        "corrected_rmse": overall.corrected_rmse,
        "correction_hit_rate": overall.correction_hit_rate,
        "harm_rate": overall.harm_rate,
        "neutral_rate": overall.neutral_rate,
    }

    for q, m in stat_report.by_data_quality.items():
        prefix = f"dq_{q.lower()}_"
        row[f"{prefix}rows"] = m.rows
        row[f"{prefix}base_mae"] = m.base_mae
        row[f"{prefix}corrected_mae"] = m.corrected_mae
        row[f"{prefix}mae_improvement_pct"] = m.mae_improvement_pct
        row[f"{prefix}hit_rate"] = m.correction_hit_rate
        row[f"{prefix}harm_rate"] = m.harm_rate

    for c, m in stat_report.by_confidence.items():
        prefix = f"conf_{c.lower()}_"
        row[f"{prefix}rows"] = m.rows
        row[f"{prefix}base_mae"] = m.base_mae
        row[f"{prefix}corrected_mae"] = m.corrected_mae
        row[f"{prefix}hit_rate"] = m.correction_hit_rate

    for window, payload in stat_report.rolling_windows.items():
        prefix = f"win_{window}_"
        row[f"{prefix}rows"] = payload.get("rows")
        row[f"{prefix}mae_improvement_pct"] = payload.get("mae_improvement_pct")
        row[f"{prefix}status"] = payload.get("status")

    return _scrub_nan(row)


def _scrub_nan(row: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN/inf floats with empty strings for CSV friendliness."""
    cleaned: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float):
            cleaned[k] = "" if not np.isfinite(v) else float(v)
        else:
            cleaned[k] = v
    return cleaned


def report_to_dataframe(report: MonitoringReport) -> pd.DataFrame:
    """Convert the monitoring report to a flat per-target DataFrame."""
    rows = [_flatten_stat_row(r) for r in report.targets.values()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_csv_report(
    report: MonitoringReport,
    output_path: str,
) -> Path:
    """Write a per-target CSV summary to *output_path*.

    Returns the path that was written.
    """
    df = report_to_dataframe(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote residual monitoring CSV to %s (%d rows)", path, len(df))
    return path


# ---------------------------------------------------------------------
# Top-level writer
# ---------------------------------------------------------------------


def write_report(
    report: MonitoringReport,
    output_dir: str,
    *,
    timestamp: Optional[str] = None,
    write_csv: bool = True,
) -> Dict[str, Path]:
    """Write the full set of report artifacts to *output_dir*.

    Always writes:
        * ``latest_summary.json``
        * ``residual_report_<timestamp>.json``

    Optionally writes:
        * ``residual_report_<timestamp>.csv``
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = _timestamp_slug(timestamp or report.timestamp)
    written: Dict[str, Path] = {}

    latest = write_latest_summary(report, str(out))
    written["latest_summary"] = latest

    timestamped_json = write_json_report(
        report,
        str(out / f"residual_report_{slug}.json"),
    )
    written["timestamped_json"] = timestamped_json

    if write_csv:
        timestamped_csv = write_csv_report(
            report,
            str(out / f"residual_report_{slug}.csv"),
        )
        written["timestamped_csv"] = timestamped_csv

    return written


# ---------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------


def render_console_summary(report: MonitoringReport) -> str:
    """Return a human-readable summary string for stdout."""
    lines: List[str] = []
    header = "Residual Correction Monitoring Report"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"Generated: {report.timestamp}")
    if report.input_path:
        lines.append(f"Input:     {report.input_path}")
    lines.append(f"Overall:   {report.overall_status}")
    lines.append("")

    for stat in ("PTS", "REB", "AST", "STL", "BLK", "TOV"):
        if stat not in report.targets:
            continue
        sr = report.targets[stat]
        overall = sr.overall
        lines.append(f"{stat}")
        lines.append(
            f"  Base MAE:       {overall.base_mae:.3f}"
            if np.isfinite(overall.base_mae)
            else "  Base MAE:       N/A"
        )
        lines.append(
            f"  Corrected MAE:  {overall.corrected_mae:.3f}"
            if np.isfinite(overall.corrected_mae)
            else "  Corrected MAE:  N/A"
        )
        if np.isfinite(overall.mae_improvement_pct):
            lines.append(
                f"  Improvement:    {overall.mae_improvement_pct:+.2f}%"
            )
        if np.isfinite(overall.correction_hit_rate):
            lines.append(
                f"  Hit Rate:       {overall.correction_hit_rate:.0%}"
            )
        lines.append(f"  Status:         {sr.status}")
        lines.append(f"  Recommendation: {sr.recommendation}")
        lines.append("")

    lines.append("Recommendation:")
    for stat, rec in report.recommendations.items():
        lines.append(f"  - {stat}: {rec}")

    return "\n".join(lines)


__all__ = [
    "render_console_summary",
    "report_to_dataframe",
    "write_csv_report",
    "write_json_report",
    "write_latest_summary",
    "write_report",
]
