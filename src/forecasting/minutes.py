"""Opportunity-first minutes allocation for Model v2."""

from __future__ import annotations

import numpy as np
import pandas as pd


def allocate_team_minutes(
    frame: pd.DataFrame,
    *,
    team_minutes: float = 240.0,
    raw_column: str = "EXPECTED_MINUTES_RAW",
    probability_column: str = "PLAY_PROB",
) -> pd.DataFrame:
    """Return conditional minutes and capped unconditional team expectations.

    ``UNCONDITIONAL_MINUTES`` sums to the regulation budget; public
    ``EXPECTED_MINUTES`` is conditional on appearing (zero when p=0).
    """

    required = {"TEAM_ID", raw_column, probability_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Minutes frame missing columns: {missing}")
    if not np.isfinite(team_minutes) or team_minutes <= 0:
        raise ValueError("team_minutes must be a positive finite number")

    result = frame.copy()
    raw = pd.to_numeric(result[raw_column], errors="coerce")
    play = pd.to_numeric(result[probability_column], errors="coerce")
    if raw.isna().any() or play.isna().any():
        raise ValueError("Minutes and participation values must be numeric")
    if not np.isfinite(raw.to_numpy()).all() or not np.isfinite(play.to_numpy()).all():
        raise ValueError("Minutes and participation values must be finite")
    if ((play < 0) | (play > 1)).any():
        raise ValueError(f"{probability_column} must be in [0, 1]")

    if (raw < 0).any():
        raise ValueError("Raw minutes must be nonnegative")
    if ((play > 0) & (raw <= 0)).any():
        raise ValueError("Positive participation requires positive conditional minutes")
    if not result.index.is_unique:
        raise ValueError("Minutes frame index must be unique")
    result["UNCONDITIONAL_MINUTES"] = 0.0
    weights = raw * play
    grouped = result.groupby("TEAM_ID", sort=False, dropna=False).groups
    for team_id, indices in grouped.items():
        index = list(indices)
        caps = 48.0 * play.loc[index].to_numpy(dtype=float)
        if caps.sum() < team_minutes - 1e-10:
            raise ValueError(f"Infeasible regulation roster for team {team_id!r}")
        allocated = capped_allocation(
            weights.loc[index].to_numpy(dtype=float), caps, team_minutes
        )
        result.loc[index, "UNCONDITIONAL_MINUTES"] = allocated
    result["EXPECTED_MINUTES"] = np.divide(
        result["UNCONDITIONAL_MINUTES"], play,
        out=np.zeros(len(result)), where=play.to_numpy() > 0,
    )

    return result


def capped_allocation(weights, caps, total):
    """Proportionally allocate a budget with individual capacity constraints."""
    weights = np.asarray(weights, dtype=float)
    caps = np.asarray(caps, dtype=float)
    allocated = np.zeros_like(weights)
    remaining = float(total)
    available = caps > 0
    while remaining > 1e-10 and available.any():
        local = weights[available]
        if local.sum() <= 0:
            raise ValueError("Cannot replace zero opportunity weights with a heuristic")
        proposal = remaining * local / local.sum()
        positions = np.flatnonzero(available)
        saturated = proposal >= caps[available]
        if not saturated.any():
            allocated[positions] = proposal
            remaining = 0.0
            break
        fixed = positions[saturated]
        allocated[fixed] = caps[fixed]
        remaining -= float(caps[fixed].sum())
        available[fixed] = False
    residual = float(total) - float(allocated.sum())
    if abs(residual) > 1e-8:
        raise ValueError("Infeasible capped minutes allocation")
    if residual:
        room = caps - allocated if residual > 0 else allocated
        allocated[int(np.argmax(room))] += residual
    return allocated


__all__ = ["allocate_team_minutes"]
