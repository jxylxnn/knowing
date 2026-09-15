"""Proper scores and explicit missingness from the stored discrete law."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.eligible_panel import TARGETS


def score_distributions(forecasts, actuals, *, target_scales):
    keys = ["GAME_ID", "PLAYER_ID"]
    truth = actuals.copy()
    predicted = forecasts.copy()
    for frame in (truth, predicted):
        for key in keys:
            frame[key] = frame[key].astype(str)
    if truth.duplicated(keys).any():
        raise ValueError("Outcome rows must be unique per player-game")
    rows = []
    for stat in TARGETS:
        group = predicted.loc[predicted.STAT.eq(stat)].copy()
        labels = truth[keys + [stat]].rename(columns={stat: "ACTUAL"}) if stat in truth else truth[keys].assign(ACTUAL=np.nan)
        joined = group.merge(labels, on=keys, how="left", validate="many_to_one")
        for _, row in joined.iterrows():
            actual = pd.to_numeric(pd.Series([row.ACTUAL]), errors="raise").iloc[0]
            result = {"REQUEST_ID": row.REQUEST_ID, "GAME_ID": row.GAME_ID,
                      "PLAYER_ID": row.PLAYER_ID, "STAT": stat,
                      "HORIZON": row.HORIZON, "ACTUAL": actual,
                      "MEAN": row.MEAN, "KNOWN": pd.notna(actual),
                      "STATUS": row.get("STATUS", "UNKNOWN"),
                      "COLD_START": row.get("COLD_START", False),
                      "MINUTES_BAND": ("low" if row.EXPECTED_MINUTES < 15 else
                                       "mid" if row.EXPECTED_MINUTES < 30 else "high")}
            if pd.notna(actual):
                if not np.isfinite(actual) or actual < 0 or actual % 1:
                    raise ValueError("Count outcomes must be finite nonnegative integers")
                values = np.asarray(json.loads(row.DISTRIBUTION_SAMPLES), dtype=float)
                if not len(values) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Distribution samples must be finite nonnegative counts")
                values.sort()
                n = len(values)
                # E|X-y| - 0.5 E|X-X'|, computed without a quadratic matrix.
                pair_term = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * values) / n**2)
                support, counts = np.unique(values, return_counts=True)
                masses = counts / n
                p_actual = float(masses[support == actual].sum())
                result.update({
                    "ABS_ERROR": abs(float(row.MEAN) - actual),
                    "CRPS": float(np.abs(values - actual).mean() - pair_term),
                    "DISCRETE_BRIER": float(np.square(masses).sum() - 2 * p_actual + 1),
                    "ZERO_BRIER": float((np.mean(values == 0) - (actual == 0)) ** 2),
                    "COVERAGE_80": float(row.P10 <= actual <= row.P90),
                    "COVERAGE_90": float(row.P05 <= actual <= row.P95),
                    "WIDTH_80": float(row.P90 - row.P10),
                    "WIDTH_90": float(row.P95 - row.P05),
                    "MINUTES_BAND": ("low" if row.EXPECTED_MINUTES < 15 else
                                     "mid" if row.EXPECTED_MINUTES < 30 else "high"),
                })
                for probability, column in ((.1, "P10"), (.5, "P50"), (.9, "P90")):
                    residual = actual - float(row[column])
                    result[f"PINBALL_{int(probability * 100)}"] = max(
                        probability * residual, (probability - 1) * residual
                    )
            rows.append(result)
    detail = pd.DataFrame(rows)
    targets = {}
    for stat in TARGETS:
        group = detail.loc[detail.STAT.eq(stat)]
        valid = group.loc[group.KNOWN]
        metrics = {"expected_rows": len(group), "rows": len(valid),
                   "missing_rows": len(group) - len(valid)}
        for field in ("ABS_ERROR", "CRPS", "DISCRETE_BRIER", "ZERO_BRIER",
                      "COVERAGE_80", "COVERAGE_90", "WIDTH_80", "WIDTH_90",
                      "PINBALL_10", "PINBALL_50", "PINBALL_90"):
            metrics[field.lower()] = float(valid[field].mean()) if len(valid) else None
        targets[stat] = metrics
    complete = all(targets[stat]["rows"] > 0 for stat in TARGETS)
    aggregate = (float(np.mean([targets[stat]["abs_error"] / target_scales[stat]
                               for stat in TARGETS])) if complete else None)
    return {"targets": targets, "normalized_mae": aggregate,
            "expected_rows": len(detail), "known_rows": int(detail.KNOWN.sum()),
            "reconciled_fraction": float(detail.KNOWN.mean()) if len(detail) else 0.0}, detail


def slice_scorecards(detail, *, minimum_rows):
    """Keep ineligible/small slices visible with their predeclared denominators."""
    records = {}
    for column in ("STAT", "HORIZON", "MINUTES_BAND", "STATUS", "COLD_START"):
        if column not in detail:
            continue
        for value, group in detail.groupby(column, dropna=False):
            known = group.loc[group.KNOWN]
            records[f"{column}={value}"] = {
                "expected_rows": len(group), "rows": len(known),
                "missing_rows": len(group) - len(known),
                "eligible": len(known) >= minimum_rows,
                "reconciliation": float(group.KNOWN.mean()),
                "coverage_80": float(known.COVERAGE_80.mean()) if len(known) else None,
                "coverage_90": float(known.COVERAGE_90.mean()) if len(known) else None,
                "mae": float(known.ABS_ERROR.mean()) if len(known) else None,
                "crps": float(known.CRPS.mean()) if len(known) else None,
            }
    return {"minimum_rows": minimum_rows, "major_slices": records}
