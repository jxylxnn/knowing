"""Conditional rate forecasts converted to total-stat distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def totals_from_minutes_and_rates(frame: pd.DataFrame, *, targets: tuple[str, ...]) -> pd.DataFrame:
    """Propagate participation, predicted minutes, and conditional rates."""
    required = {"PLAY_PROB", "EXPECTED_MINUTES"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Rate frame missing columns: {sorted(missing)}")
    result = frame.copy()
    for target in targets:
        rate = pd.to_numeric(result.get(f"{target}_RATE", 0), errors="coerce").fillna(0).clip(lower=0)
        rate_std = pd.to_numeric(result.get(f"{target}_RATE_STD", 0), errors="coerce").fillna(0).clip(lower=0)
        mean = result["PLAY_PROB"] * result["EXPECTED_MINUTES"] * rate
        spread = result["PLAY_PROB"] * result["EXPECTED_MINUTES"] * rate_std
        result[f"{target}_MEAN"] = mean
        result[f"{target}_P10"] = (mean - 1.28155 * spread).clip(lower=0)
        result[f"{target}_P50"] = mean
        result[f"{target}_P90"] = mean + 1.28155 * spread
        result[f"{target}_ZERO_PROB"] = (1 - result["PLAY_PROB"]).clip(0, 1)
    return result
