"""Transparent, versioned Model v2 promotion evidence gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


PROMOTION_EVIDENCE_SCHEMA_VERSION = "promotion_evidence_v1"
REQUIRED_TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
REQUIRED_CONTRACT_FLAGS = ("artifact_contract", "calibration")


@dataclass(frozen=True)
class PromotionThresholds:
    normalized_mae_improvement: float = 0.01
    core_target_max_regression: float = 0.01
    secondary_target_max_regression: float = 0.02
    coverage_tolerance_overall: float = 0.03
    coverage_tolerance_major_slice: float = 0.05
    max_fallback_rate: float = 0.05
    max_degraded_rate: float = 0.10
    min_major_slice_reconciliation: float = 0.95
    min_core_reconciliation: float = 0.999


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    checks: Mapping[str, bool] = field(default_factory=dict)


def evaluate_v2_promotion(
    *,
    normalized_mae_improvement: float,
    target_improvements: Mapping[str, float],
    participation_beats_baseline: bool,
    minutes_beats_baseline: bool,
    coverage_80: float,
    coverage_90: float,
    point_in_time_valid: bool,
    bootstrap_ci: tuple[float, float] | None = None,
    contract_flags: Mapping[str, bool] | None = None,
    live_replay_parity: bool | None = None,
    core_reconciliation: float | None = None,
    fallback_rate: float | None = None,
    degraded_rate: float | None = None,
    major_slice_reconciliation: Mapping[str, float] | None = None,
    major_slice_coverage_80: Mapping[str, float] | None = None,
    major_slice_coverage_90: Mapping[str, float] | None = None,
    thresholds: PromotionThresholds | None = None,
    require_complete_evidence: bool = False,
) -> PromotionDecision:
    """Apply accuracy, uncertainty, contract, parity, and operations gates.

    ``bootstrap_ci`` is candidate-minus-baseline MAE, so its upper endpoint
    must be below zero. ``require_complete_evidence`` is intended for official
    promotion; the default preserves the historical unit-level API while
    still evaluating every supplied evidence field.
    """

    policy = thresholds or PromotionThresholds()
    reasons: list[str] = []
    checks: dict[str, bool] = {}

    aggregate_improvement = _finite_metric(
        "normalized_mae_improvement", normalized_mae_improvement, reasons
    )

    checks["aggregate_improvement"] = aggregate_improvement is not None and (
        aggregate_improvement >= policy.normalized_mae_improvement
    )
    if not checks["aggregate_improvement"]:
        if aggregate_improvement is not None:
            reasons.append(
                "aggregate normalized MAE improvement is below "
                f"{policy.normalized_mae_improvement:.1%}"
            )

    core = {"PTS", "REB", "AST"}
    target_gate = True
    if not isinstance(target_improvements, Mapping):
        target_gate = False
        reasons.append("target improvement evidence must be a mapping")
        target_improvements = {}
    target_keys = {str(target).upper() for target in target_improvements}
    if require_complete_evidence:
        missing_targets = sorted(set(REQUIRED_TARGETS) - target_keys)
        if missing_targets:
            target_gate = False
            reasons.append("target improvement evidence is missing: " + ", ".join(missing_targets))
    for target, raw_improvement in target_improvements.items():
        target_name = str(target).upper()
        improvement = _finite_metric(
            f"target improvement {target_name}", raw_improvement, reasons
        )
        if improvement is None:
            target_gate = False
            continue
        allowed_regression = (
            policy.core_target_max_regression
            if target_name in core
            else policy.secondary_target_max_regression
        )
        if improvement < -allowed_regression:
            target_gate = False
            reasons.append(
                f"{target_name} regression {improvement:.2%} exceeds "
                f"{-allowed_regression:.0%}"
            )
    checks["target_regressions"] = target_gate

    checks["participation"] = _boolean_metric(
        "participation", participation_beats_baseline, reasons,
        required=require_complete_evidence,
    )
    if not checks["participation"]:
        reasons.append("participation does not beat both rule baselines")
    checks["minutes"] = _boolean_metric(
        "minutes", minutes_beats_baseline, reasons,
        required=require_complete_evidence,
    )
    if not checks["minutes"]:
        reasons.append("minutes does not beat rolling-10 and role-median baselines")

    coverage_80_value = _bounded_metric("coverage_80", coverage_80, reasons)
    coverage_90_value = _bounded_metric("coverage_90", coverage_90, reasons)
    checks["overall_interval_coverage"] = (
        coverage_80_value is not None
        and coverage_90_value is not None
        and abs(coverage_80_value - 0.80) <= policy.coverage_tolerance_overall + 1e-12
        and abs(coverage_90_value - 0.90) <= policy.coverage_tolerance_overall + 1e-12
    )
    if not checks["overall_interval_coverage"]:
        reasons.append("overall interval coverage is outside the configured tolerance")

    checks["point_in_time"] = _boolean_metric(
        "point_in_time", point_in_time_valid, reasons,
        required=require_complete_evidence,
    )
    if not point_in_time_valid:
        reasons.append("point-in-time validation failed")

    if bootstrap_ci is not None:
        if not isinstance(bootstrap_ci, (tuple, list)) or len(bootstrap_ci) != 2:
            checks["bootstrap_ci"] = False
            reasons.append("paired game-bootstrap evidence must contain two finite endpoints")
        else:
            low = _finite_metric("bootstrap_ci lower endpoint", bootstrap_ci[0], reasons)
            high = _finite_metric("bootstrap_ci upper endpoint", bootstrap_ci[1], reasons)
            checks["bootstrap_ci"] = (
                low is not None and high is not None and low <= high and high < 0
            )
        if not checks["bootstrap_ci"]:
            reasons.append(
                "paired game-bootstrap interval does not exclude baseline superiority"
            )
    elif require_complete_evidence:
        checks["bootstrap_ci"] = False
        reasons.append("paired game-bootstrap evidence is missing")

    if contract_flags is not None:
        if not isinstance(contract_flags, Mapping):
            checks["contracts"] = False
            reasons.append("contract evidence must be a mapping")
            contract_flags = {}
        missing_contracts = (
            sorted(set(REQUIRED_CONTRACT_FLAGS) - set(contract_flags))
            if require_complete_evidence else []
        )
        invalid_contracts = sorted(
            str(name) for name, passed in contract_flags.items()
            if type(passed) is not bool
        )
        failed = sorted(
            str(name) for name, passed in contract_flags.items()
            if type(passed) is bool and not passed
        )
        checks["contracts"] = not (missing_contracts or invalid_contracts or failed)
        for name, passed in contract_flags.items():
            if type(passed) is bool:
                checks[str(name)] = passed
        if missing_contracts:
            reasons.append("contract evidence is missing: " + ", ".join(missing_contracts))
        if invalid_contracts:
            reasons.append("contract checks must be booleans: " + ", ".join(invalid_contracts))
        if failed:
            reasons.append("contract checks failed: " + ", ".join(failed))
    elif require_complete_evidence:
        checks["contracts"] = False
        reasons.append("contract evidence is missing")

    if live_replay_parity is not None:
        checks["live_replay_parity"] = _boolean_metric(
            "live_replay_parity", live_replay_parity, reasons,
            required=require_complete_evidence,
        )
        if not live_replay_parity:
            reasons.append("live/replay parity failed")
    elif require_complete_evidence:
        checks["live_replay_parity"] = False
        reasons.append("live/replay parity evidence is missing")

    if core_reconciliation is not None or require_complete_evidence:
        core_fraction = _bounded_metric("core_reconciliation", core_reconciliation, reasons)
        checks["core_reconciliation"] = (
            core_fraction is not None and core_fraction >= policy.min_core_reconciliation
        )
        if not checks["core_reconciliation"]:
            reasons.append("core reconciliation is below the configured minimum")

    _rate_gate(
        "fallback_rate", fallback_rate, policy.max_fallback_rate,
        checks, reasons, require_complete_evidence,
    )
    _rate_gate(
        "degraded_rate", degraded_rate, policy.max_degraded_rate,
        checks, reasons, require_complete_evidence,
    )

    _minimum_mapping_gate(
        "major_slice_reconciliation",
        major_slice_reconciliation,
        policy.min_major_slice_reconciliation,
        checks,
        reasons,
        require_complete_evidence,
    )
    _coverage_mapping_gate(
        "major_slice_coverage_80",
        major_slice_coverage_80,
        expected=0.80,
        tolerance=policy.coverage_tolerance_major_slice,
        checks=checks,
        reasons=reasons,
        required=require_complete_evidence,
    )
    _coverage_mapping_gate(
        "major_slice_coverage_90",
        major_slice_coverage_90,
        expected=0.90,
        tolerance=policy.coverage_tolerance_major_slice,
        checks=checks,
        reasons=reasons,
        required=require_complete_evidence,
    )
    _consistent_slice_keys(
        major_slice_reconciliation,
        major_slice_coverage_80,
        major_slice_coverage_90,
        checks,
        reasons,
        required=require_complete_evidence,
    )
    return PromotionDecision(not reasons, tuple(reasons), checks)


def _rate_gate(
    name: str,
    value: float | None,
    maximum: float,
    checks: dict[str, bool],
    reasons: list[str],
    required: bool,
) -> None:
    if value is None:
        if required:
            checks[name] = False
            reasons.append(f"{name} evidence is missing")
        return
    checked = _bounded_metric(name, value, reasons, maximum=maximum)
    checks[name] = checked is not None
    if not checks[name]:
        if checked is not None:
            reasons.append(f"{name} {checked:.2%} exceeds {maximum:.2%}")


def _minimum_mapping_gate(
    name: str,
    values: Mapping[str, float] | None,
    minimum: float,
    checks: dict[str, bool],
    reasons: list[str],
    required: bool,
) -> None:
    if values is None:
        if required:
            checks[name] = False
            reasons.append(f"{name} evidence is missing")
        return
    if not isinstance(values, Mapping):
        checks[name] = False
        reasons.append(f"{name} evidence must be a mapping")
        return
    failed: list[str] = []
    invalid: list[str] = []
    for key, value in values.items():
        checked = _bounded_metric(f"{name}[{key}]", value, reasons)
        if checked is None:
            invalid.append(str(key))
        elif checked < minimum:
            failed.append(str(key))
    checks[name] = bool(values) and not failed
    if not values:
        reasons.append(f"{name} has no evaluated slices")
    if invalid:
        checks[name] = False
        reasons.append(f"{name} contains invalid values for: " + ", ".join(sorted(invalid)))
    elif failed:
        reasons.append(f"{name} below threshold for: " + ", ".join(failed))


def _coverage_mapping_gate(
    name: str,
    values: Mapping[str, float] | None,
    *,
    expected: float,
    tolerance: float,
    checks: dict[str, bool],
    reasons: list[str],
    required: bool,
) -> None:
    if values is None:
        if required:
            checks[name] = False
            reasons.append(f"{name} evidence is missing")
        return
    if not isinstance(values, Mapping):
        checks[name] = False
        reasons.append(f"{name} evidence must be a mapping")
        return
    failed: list[str] = []
    invalid: list[str] = []
    for key, value in values.items():
        checked = _bounded_metric(f"{name}[{key}]", value, reasons)
        if checked is None:
            invalid.append(str(key))
        elif abs(checked - expected) > tolerance + 1e-12:
            failed.append(str(key))
    checks[name] = bool(values) and not failed
    if not values:
        reasons.append(f"{name} has no evaluated slices")
    if invalid:
        checks[name] = False
        reasons.append(f"{name} contains invalid values for: " + ", ".join(sorted(invalid)))
    elif failed:
        reasons.append(f"{name} outside tolerance for: " + ", ".join(failed))


def _finite_metric(name: str, value: object, reasons: list[str]) -> float | None:
    if isinstance(value, bool):
        reasons.append(f"{name} must be a finite number")
        return None
    try:
        checked = float(value)
    except (TypeError, ValueError):
        reasons.append(f"{name} must be a finite number")
        return None
    if not math.isfinite(checked):
        reasons.append(f"{name} must be finite")
        return None
    return checked


def _bounded_metric(
    name: str,
    value: object,
    reasons: list[str],
    *,
    maximum: float = 1.0,
) -> float | None:
    checked = _finite_metric(name, value, reasons)
    if checked is None:
        return None
    if checked < 0 or checked > maximum:
        reasons.append(f"{name} must be in [0, {maximum:g}]")
        return None
    return checked


def _boolean_metric(
    name: str,
    value: object,
    reasons: list[str],
    *,
    required: bool,
) -> bool:
    if type(value) is not bool:
        if required:
            reasons.append(f"{name} evidence must be a boolean")
            return False
        return bool(value)
    return value


def _consistent_slice_keys(
    reconciliation: Mapping[str, float] | None,
    coverage_80: Mapping[str, float] | None,
    coverage_90: Mapping[str, float] | None,
    checks: dict[str, bool],
    reasons: list[str],
    *,
    required: bool,
) -> None:
    mappings = [
        mapping for mapping in (reconciliation, coverage_80, coverage_90)
        if mapping is not None
    ]
    if not required or not mappings:
        return
    if any(not isinstance(mapping, Mapping) for mapping in mappings):
        return
    key_sets = [set(mapping) for mapping in mappings]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        for name in (
            "major_slice_reconciliation",
            "major_slice_coverage_80",
            "major_slice_coverage_90",
        ):
            if name in checks:
                checks[name] = False
        reasons.append("major-slice evidence mappings must cover the same slices")


__all__ = [
    "PROMOTION_EVIDENCE_SCHEMA_VERSION",
    "PromotionDecision",
    "PromotionThresholds",
    "REQUIRED_CONTRACT_FLAGS",
    "REQUIRED_TARGETS",
    "evaluate_v2_promotion",
]
