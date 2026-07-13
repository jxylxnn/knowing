"""Leak-safe promotion gates for manual continual model improvement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


@dataclass
class PromotionPolicy:
    """Default champion/challenger safety thresholds."""

    min_rows_per_target: int = 1000
    min_weighted_improvement_pct: float = 1.0
    max_target_regression_pct: float = 2.0
    max_coverage_regression_pp: float = 2.0
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    random_seed: int = 42
    core_weight: float = 2.0
    secondary_weight: float = 1.0


@dataclass
class PromotionDecision:
    """Machine-readable result of all promotion gates."""

    eligible: bool
    weighted_improvement_pct: float
    bootstrap_low_pct: Optional[float]
    bootstrap_high_pct: Optional[float]
    per_target_regression_pct: Dict[str, float] = field(default_factory=dict)
    coverage_regression_pp: Dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _weighted_improvement(
    champion_mae: Mapping[str, float],
    candidate_mae: Mapping[str, float],
    policy: PromotionPolicy,
) -> float:
    improvements = []
    weights = []
    for stat in TARGETS:
        base = float(champion_mae.get(stat, np.nan))
        challenger = float(candidate_mae.get(stat, np.nan))
        if not np.isfinite(base) or base <= 0 or not np.isfinite(challenger):
            continue
        weight = policy.core_weight if stat in ("PTS", "REB", "AST") else policy.secondary_weight
        improvements.append((base - challenger) / base * weight)
        weights.append(weight)
    return 100.0 * float(np.sum(improvements) / np.sum(weights)) if weights else float("nan")


def paired_bootstrap_improvement(
    champion_errors: Sequence[float],
    candidate_errors: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap champion absolute error minus candidate absolute error."""
    champion = np.asarray(champion_errors, dtype=float)
    candidate = np.asarray(candidate_errors, dtype=float)
    n = min(len(champion), len(candidate))
    if n < 2:
        raise ValueError("At least two paired errors are required for bootstrap evaluation")
    champion = champion[:n]
    candidate = candidate[:n]
    delta = np.abs(champion) - np.abs(candidate)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(max(1, int(samples)), n))
    estimates = delta[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def evaluate_promotion(
    champion_mae: Mapping[str, float],
    candidate_mae: Mapping[str, float],
    *,
    champion_errors: Optional[Mapping[str, Sequence[float]]] = None,
    candidate_errors: Optional[Mapping[str, Sequence[float]]] = None,
    champion_rows: Optional[Mapping[str, int]] = None,
    candidate_rows: Optional[Mapping[str, int]] = None,
    champion_coverage: Optional[Mapping[str, float]] = None,
    candidate_coverage: Optional[Mapping[str, float]] = None,
    policy: Optional[PromotionPolicy] = None,
) -> PromotionDecision:
    """Apply every configured gate and return a decision without side effects."""
    policy = policy or PromotionPolicy()
    reasons: list[str] = []
    weighted = _weighted_improvement(champion_mae, candidate_mae, policy)
    if not np.isfinite(weighted):
        reasons.append("no finite MAE metrics for the required targets")
    elif weighted < policy.min_weighted_improvement_pct:
        reasons.append(
            f"weighted improvement {weighted:.2f}% is below {policy.min_weighted_improvement_pct:.2f}%"
        )

    target_regression: Dict[str, float] = {}
    for stat in TARGETS:
        base = float(champion_mae.get(stat, np.nan))
        challenger = float(candidate_mae.get(stat, np.nan))
        if np.isfinite(base) and base > 0 and np.isfinite(challenger):
            regression = (challenger - base) / base * 100.0
            target_regression[stat] = regression
            if regression > policy.max_target_regression_pct:
                reasons.append(
                    f"{stat} regresses {regression:.2f}% (limit {policy.max_target_regression_pct:.2f}%)"
                )

    for stat in TARGETS:
        rows = min(
            int((champion_rows or {}).get(stat, 0)),
            int((candidate_rows or {}).get(stat, 0)),
        )
        if rows < policy.min_rows_per_target:
            reasons.append(
                f"{stat} has only {rows} paired rows (minimum {policy.min_rows_per_target})"
            )

    low = high = None
    if champion_errors is None or candidate_errors is None:
        reasons.append("paired holdout errors are required for bootstrap confidence")
    else:
        deltas = []
        for stat in TARGETS:
            if stat not in champion_errors or stat not in candidate_errors:
                continue
            try:
                stat_low, stat_high = paired_bootstrap_improvement(
                    champion_errors[stat],
                    candidate_errors[stat],
                    samples=policy.bootstrap_samples,
                    confidence=policy.confidence,
                    seed=policy.random_seed,
                )
                deltas.append((stat_low, stat_high))
            except ValueError:
                continue
        if deltas:
            low = 100.0 * min(delta[0] for delta in deltas)
            high = 100.0 * max(delta[1] for delta in deltas)
            if low <= 0:
                reasons.append(f"bootstrap improvement lower bound {low:.2f}% is not positive")
        else:
            reasons.append("no valid paired holdout errors for bootstrap confidence")

    coverage_regression: Dict[str, float] = {}
    if champion_coverage is not None and candidate_coverage is not None:
        for stat in TARGETS:
            if stat in champion_coverage and stat in candidate_coverage:
                regression = 100.0 * (
                    float(champion_coverage[stat]) - float(candidate_coverage[stat])
                )
                coverage_regression[stat] = regression
                if regression > policy.max_coverage_regression_pp:
                    reasons.append(
                        f"{stat} interval coverage drops {regression:.2f} percentage points"
                    )

    return PromotionDecision(
        eligible=not reasons,
        weighted_improvement_pct=float(weighted),
        bootstrap_low_pct=low,
        bootstrap_high_pct=high,
        per_target_regression_pct=target_regression,
        coverage_regression_pp=coverage_regression,
        reasons=reasons,
    )
