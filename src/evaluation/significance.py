"""Game-clustered statistical comparisons for forecast candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd


MultiplicityMethod = Literal["none", "bonferroni", "holm"]


def paired_game_bootstrap(
    rows: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    actual_column: str = "ACTUAL",
    game_column: str = "GAME_ID",
    samples: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
    comparison_count: int = 1,
    multiplicity_method: MultiplicityMethod = "none",
) -> dict[str, Any]:
    """Compare paired absolute errors using a game-cluster bootstrap.

    A negative delta favors the candidate. Games, rather than independent
    player rows, are resampled; the statistic remains row-weighted inside a
    draw. Bonferroni intervals are available for simultaneous inference.
    ``holm`` applies a Holm correction in :func:`paired_game_bootstrap_many`
    and uses a conservative Bonferroni interval here.
    """

    needed = {candidate_column, baseline_column, actual_column, game_column}
    if missing := needed - set(rows.columns):
        raise ValueError(f"Bootstrap rows missing columns: {sorted(missing)}")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if comparison_count < 1:
        raise ValueError("comparison_count must be positive")
    if multiplicity_method not in {"none", "bonferroni", "holm"}:
        raise ValueError("Unknown multiplicity method")

    frame = rows.dropna(subset=list(needed)).copy()
    for column in (candidate_column, baseline_column, actual_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[[candidate_column, baseline_column, actual_column]])
        .all(axis=1)
    ]
    if frame.empty:
        return {
            "delta_mae": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "adjusted_p_value": float("nan"),
            "games": 0.0,
            "rows": 0.0,
            "confidence_level": confidence_level,
            "multiplicity_method": multiplicity_method,
            "comparison_count": comparison_count,
            "interpretation": "negative_delta_favors_candidate",
        }

    delta = (frame[candidate_column] - frame[actual_column]).abs() - (
        frame[baseline_column] - frame[actual_column]
    ).abs()
    clusters = (
        pd.DataFrame({game_column: frame[game_column], "delta": delta})
        .groupby(game_column, dropna=False)["delta"]
        .agg(["sum", "count"])
    )
    sums = clusters["sum"].to_numpy(dtype=float)
    counts = clusters["count"].to_numpy(dtype=float)
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(samples, len(clusters)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)

    alpha = 1.0 - confidence_level
    if multiplicity_method in {"bonferroni", "holm"}:
        alpha /= comparison_count
    ci_low, ci_high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])

    # Centering the bootstrap distribution estimates the null distribution of
    # the paired statistic. This avoids the common error of measuring the
    # proportion of an uncentered bootstrap on either side of zero.
    null_estimates = estimates - observed
    p_value = float(
        min(1.0, (np.count_nonzero(np.abs(null_estimates) >= abs(observed)) + 1)
            / (samples + 1))
    )
    adjusted = (
        min(1.0, p_value * comparison_count)
        if multiplicity_method == "bonferroni"
        else p_value
    )
    return {
        "delta_mae": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "adjusted_p_value": adjusted,
        "games": float(len(clusters)),
        "rows": float(len(frame)),
        "confidence_level": confidence_level,
        "multiplicity_method": multiplicity_method,
        "comparison_count": comparison_count,
        "interpretation": "negative_delta_favors_candidate",
    }


def paired_game_bootstrap_many(
    rows: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_columns: Mapping[str, str],
    actual_column: str = "ACTUAL",
    game_column: str = "GAME_ID",
    samples: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
    multiplicity_method: MultiplicityMethod = "holm",
) -> pd.DataFrame:
    """Run several paired comparisons with explicit multiplicity handling."""

    if not baseline_columns:
        return pd.DataFrame()
    count = len(baseline_columns)
    records: list[dict[str, Any]] = []
    for offset, (name, column) in enumerate(baseline_columns.items()):
        result = paired_game_bootstrap(
            rows,
            candidate_column=candidate_column,
            baseline_column=column,
            actual_column=actual_column,
            game_column=game_column,
            samples=samples,
            seed=seed + offset,
            confidence_level=confidence_level,
            comparison_count=count,
            multiplicity_method=multiplicity_method,
        )
        result["baseline"] = name
        records.append(result)

    if multiplicity_method == "holm":
        raw = np.asarray([record["p_value"] for record in records], dtype=float)
        finite = np.flatnonzero(np.isfinite(raw))
        order = finite[np.argsort(raw[finite])]
        running = 0.0
        for rank, index in enumerate(order):
            adjusted = min(1.0, raw[index] * (len(order) - rank))
            running = max(running, adjusted)
            records[index]["adjusted_p_value"] = running
    elif multiplicity_method == "none":
        for record in records:
            record["adjusted_p_value"] = record["p_value"]

    return pd.DataFrame(records).sort_values("baseline", kind="stable").reset_index(
        drop=True
    )


def paired_macro_bootstrap(rows, *, baseline_column, target_scales, samples=2000, seed=42):
    """Game bootstrap of the predeclared equal-target normalized MAE delta."""
    required = {"GAME_ID", "STAT", "MEAN", "ACTUAL", baseline_column}
    if required - set(rows) or samples < 100:
        raise ValueError("Macro bootstrap requires paired rows and at least 100 draws")
    frame = rows.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("Macro bootstrap accepts only the frozen paired population")
    frame["DELTA"] = ((frame.MEAN - frame.ACTUAL).abs()
                       - (frame[baseline_column] - frame.ACTUAL).abs())
    frame["DELTA"] /= frame.STAT.map(target_scales)
    if not np.isfinite(frame.DELTA).all():
        raise ValueError("Macro bootstrap needs finite normalized errors")
    targets = list(target_scales)
    grouped = frame.groupby(["GAME_ID", "STAT"]).DELTA.agg(["sum", "count"])
    sums = grouped["sum"].unstack("STAT").reindex(columns=targets).fillna(0).to_numpy()
    counts = grouped["count"].unstack("STAT").reindex(columns=targets).fillna(0).to_numpy()
    if len(sums) < 2 or (counts.sum(axis=0) == 0).any():
        return {"status": "insufficient_games_or_targets", "ci": None}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        draw = rng.integers(0, len(sums), size=len(sums))
        denominators = counts[draw].sum(axis=0)
        if (denominators == 0).any():
            return {"status": "resampled_target_missing", "ci": None}
        estimates.append(float(np.mean(sums[draw].sum(axis=0) / denominators)))
    return {"status": "computed", "seed": seed, "samples": samples,
            "unit": "game", "normalization": "equal_target_frozen_scales",
            "delta": float(np.mean(sums.sum(axis=0) / counts.sum(axis=0))),
            "ci": np.quantile(estimates, [.025, .975]).tolist(), "games": len(sums)}
