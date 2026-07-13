"""Confidence-aware probability adjustment helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

CONFIDENCE_STRENGTH = {
    "HIGH": 1.00,
    "MEDIUM": 0.85,
    "LOW": 0.65,
    "NO_EDGE": 0.50,
}

Z_BY_CONFIDENCE = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def _to_float_or_none(v: Any) -> Optional[float]:
    """Try to convert a value to float; return None for None/NaN/unconvertible."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) or np.isnan(f) else f


def _extract(row: Any, col: str) -> Optional[float]:
    """Safely extract a float column value from dict/Series/DataFrame."""
    if isinstance(row, (pd.Series, pd.DataFrame)):
        if col not in row:
            return None
        v = row[col]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        return _to_float_or_none(v)
    if isinstance(row, Mapping):
        return _to_float_or_none(row.get(col))
    return None


def get_projection_value(row: Any, stat: str) -> Optional[float]:
    """Return corrected projection if available, else raw projection.

    Returns None when neither exists — never 0.0 as a fallback sentinel.
    """
    stat_upper = str(stat).upper()
    corrected = _extract(row, f"{stat_upper}_CORRECTED")
    if corrected is not None:
        return corrected
    raw = _extract(row, stat_upper)
    if raw is not None:
        return raw
    return None


def get_correction_delta(row: Any, stat: str) -> Optional[float]:
    """Return correction delta if available (CORRECTED - BASE or CORRECTED - raw)."""
    stat_upper = str(stat).upper()
    corrected = _extract(row, f"{stat_upper}_CORRECTED")
    if corrected is None:
        return None
    base = _extract(row, f"{stat_upper}_BASE")
    if base is not None:
        return corrected - base
    raw = _extract(row, stat_upper)
    if raw is not None:
        return corrected - raw
    return None


VALID_CONFIDENCE_LABELS = {"HIGH", "MEDIUM", "LOW", "NO_EDGE"}


def _normalize_label(v: Any) -> str:
    """Convert a raw value to a valid confidence label or empty string.

    Returns '' for None, NaN, blank, or anything not in VALID_CONFIDENCE_LABELS.
    """
    if v is None:
        return ""
    if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
        return ""
    label = str(v).upper().strip()
    return label if label in VALID_CONFIDENCE_LABELS else ""


def get_confidence_label(row: Any, stat: str) -> str:
    """Return confidence label if available, else empty string.

    Never returns "nan" or unknown labels — only HIGH, MEDIUM, LOW, NO_EDGE, or "".
    """
    stat_upper = str(stat).upper()
    col = f"{stat_upper}_CONFIDENCE"
    if isinstance(row, (pd.Series, pd.DataFrame)):
        if col in row:
            v = row[col]
            val = v.iloc[0] if isinstance(v, pd.Series) else v
            return _normalize_label(val)
        return ""
    if isinstance(row, Mapping):
        v = row.get(col)
        return _normalize_label(v)
    return ""


def get_confidence_score(row: Any, stat: str) -> float:
    """Return confidence score if available, else 0.0.

    Returns 0.0 for None, NaN, or non-numeric values.
    """
    raw = _extract(row, f"{str(stat).upper()}_CONFIDENCE_SCORE")
    if raw is not None and not (isinstance(raw, float) and (pd.isna(raw) or np.isnan(raw))):
        return max(0.0, float(raw))
    return 0.0


def get_data_quality_label(row: Any) -> str:
    """Return DATA_QUALITY if available, else FULL."""
    if isinstance(row, (pd.Series, pd.DataFrame)):
        if "DATA_QUALITY" in row:
            v = row["DATA_QUALITY"]
            return str(v.iloc[0]) if isinstance(v, pd.Series) else str(v)
        return "FULL"
    if isinstance(row, Mapping):
        return str(row.get("DATA_QUALITY", "FULL"))
    return "FULL"


def std_from_interval(
    row: Any,
    stat: str,
    confidence: float = 0.90,
) -> Optional[float]:
    """Derive standard deviation from calibrated interval width.

    90% interval ≈ mean ± 1.645σ
    80% interval ≈ mean ± 1.282σ
    """
    stat_upper = str(stat).upper()
    z = Z_BY_CONFIDENCE.get(confidence, 1.645)

    if abs(confidence - 0.90) < 0.01:
        low_col = f"{stat_upper}_INTERVAL_90_LOW"
        high_col = f"{stat_upper}_INTERVAL_90_HIGH"
    elif abs(confidence - 0.80) < 0.01:
        low_col = f"{stat_upper}_INTERVAL_80_LOW"
        high_col = f"{stat_upper}_INTERVAL_80_HIGH"
    else:
        low_col = f"{stat_upper}_INTERVAL_90_LOW"
        high_col = f"{stat_upper}_INTERVAL_90_HIGH"

    low = _extract(row, low_col)
    high = _extract(row, high_col)
    if low is None or high is None:
        return None

    width = high - low
    if width <= 0:
        return None

    return width / (2.0 * z)


def damp_probability(prob: float, confidence_label: str) -> float:
    """Pull probability toward 0.5 when confidence is low."""
    prob = float(prob)
    strength = CONFIDENCE_STRENGTH.get(
        str(confidence_label).upper().strip(),
        0.50,
    )
    return 0.5 + (prob - 0.5) * strength
