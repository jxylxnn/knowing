"""Immutable Model-v2 bundle construction and promotion validation.

The legacy runtime contract validates a flat CatBoost layout.  Model v2 uses
an evidence bundle instead: predictors, fold manifests, calibration evidence,
source provenance, and the resolved environment travel together and are
content-addressed before a champion pointer can reference them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from src.contracts.forecast import CANONICAL_FORECAST_COLUMNS
from src.utils.file_lock import exclusive_file_lock


V2_BUNDLE_SCHEMA_VERSION = "model_bundle_v2"

V2_REQUIRED_FILES = (
    "config_resolved.yaml",
    "source_snapshot_manifest.json",
    "feature_schema.json",
    "training_folds.json",
    "baseline_scorecard.json",
    "candidate_scorecard.json",
    "slice_scorecard.json",
    "calibration_scorecard.json",
    "simulation_parameters.json",
    "environment.lock",
    "prediction_contract.json",
    "checksums.json",
    "promotion_decision.json",
)

V2_REQUIRED_MODEL_DIRS = (
    "availability_model",
    "minutes_models",
    "stat_rate_models",
    "calibrators",
)


@dataclass(frozen=True)
class V2BundleValidation:
    """Machine-readable result of strict v2 bundle validation."""

    bundle_id: str
    files_checked: int
    promotion_eligible: bool
    code_version: str
    source_snapshot_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def runtime_environment() -> dict[str, Any]:
    """Return a deterministic, fully resolved local runtime inventory."""

    packages = {
        str(distribution.metadata.get("Name")): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def git_state(project_root: str | Path = ".") -> tuple[str, bool]:
    """Return the exact commit and dirty flag without mutating the checkout."""

    root = Path(project_root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Could not resolve Git state for {root}") from exc
    if not revision:
        raise ValueError("Git revision is empty")
    return revision, bool(porcelain.strip())


def write_v2_support_files(
    directory: str | Path,
    *,
    resolved_config: Mapping[str, Any],
    source_snapshot_manifest: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    training_folds: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    baseline_scorecard: Mapping[str, Any],
    candidate_scorecard: Mapping[str, Any],
    slice_scorecard: Mapping[str, Any],
    calibration_scorecard: Mapping[str, Any],
    simulation_parameters: Mapping[str, Any],
    promotion_decision: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
) -> None:
    """Write the non-model evidence files needed by a v2 candidate.

    Callers must create the four required model/calibrator directories with
    actual trained or declared-baseline artifacts.  This function never
    fabricates a component merely to satisfy validation.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _write_text_once(
        root / "config_resolved.yaml",
        yaml.safe_dump(dict(resolved_config), sort_keys=True),
    )
    payloads: dict[str, Any] = {
        "source_snapshot_manifest.json": dict(source_snapshot_manifest),
        "feature_schema.json": dict(feature_schema),
        "training_folds.json": training_folds,
        "baseline_scorecard.json": dict(baseline_scorecard),
        "candidate_scorecard.json": dict(candidate_scorecard),
        "slice_scorecard.json": dict(slice_scorecard),
        "calibration_scorecard.json": dict(calibration_scorecard),
        "simulation_parameters.json": dict(simulation_parameters),
        "promotion_decision.json": dict(promotion_decision),
        "environment.lock": dict(environment or runtime_environment()),
        "prediction_contract.json": {
            "schema_version": "canonical_forecast_v1",
            "columns": list(CANONICAL_FORECAST_COLUMNS),
        },
    }
    for name, payload in payloads.items():
        _write_json_once(root / name, payload)


def write_checksums(directory: str | Path) -> dict[str, str]:
    """Write content hashes for all bundle payloads except cyclic manifests."""

    root = Path(directory)
    checksums = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"checksums.json", "bundle_manifest.json"}
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    }
    _write_json_once(root / "checksums.json", checksums)
    return checksums


def finalize_v2_bundle(
    directory: str | Path,
    *,
    data_cutoff: str,
    config: Mapping[str, Any],
    source_snapshot_id: str,
    cutoffs: Mapping[str, str | None],
    metrics: Mapping[str, Any],
    project_root: str | Path = ".",
    code_version: str | None = None,
    dirty_worktree: bool | None = None,
    component_versions: Mapping[str, str] | None = None,
) -> "ModelBundleManifest":
    """Seal a populated candidate directory with checksums and a manifest."""

    from src.models.versioning import ModelBundleManifest

    root = Path(directory)
    if code_version is None or dirty_worktree is None:
        detected_version, detected_dirty = git_state(project_root)
        code_version = code_version or detected_version
        dirty_worktree = detected_dirty if dirty_worktree is None else dirty_worktree

    from src.utils.code_provenance import source_tree_manifest

    _write_json_once(root / "source_tree_manifest.json", source_tree_manifest(project_root))
    write_checksums(root)
    source_manifest = _load_json(root / "source_snapshot_manifest.json")
    manifest = ModelBundleManifest.from_directory(
        root,
        data_cutoff=data_cutoff,
        config=config,
        metrics=dict(metrics),
        code_version=code_version,
        dirty_worktree=bool(dirty_worktree),
        source_snapshot_id=source_snapshot_id,
        schema_version=V2_BUNDLE_SCHEMA_VERSION,
        architecture="v2",
        cutoffs=dict(cutoffs),
        source_snapshots=(source_manifest,),
        component_versions=dict(component_versions or {}),
        fold_metrics=dict(metrics.get("folds", {})),
        aggregate_metrics=dict(metrics.get("aggregate", {})),
        slice_metrics=dict(metrics.get("slices", {})),
        calibration_metrics=dict(metrics.get("calibration", {})),
        baseline_comparison=dict(metrics.get("baseline_comparison", {})),
        promotion_decision=dict(metrics.get("promotion_decision", {})),
        runtime_metadata=runtime_environment(),
    )
    manifest.write(root)
    validate_v2_bundle(root, promotion=False)
    return manifest


def validate_v2_bundle(
    directory: str | Path,
    *,
    promotion: bool = False,
) -> V2BundleValidation:
    """Validate v2 contents, provenance, checksums, and optional promotion gates."""

    from src.models.versioning import ModelBundleManifest

    root = Path(directory)
    manifest_path = root / ModelBundleManifest.FILE_NAME
    if not manifest_path.is_file():
        raise ValueError("Model v2 bundle is missing bundle_manifest.json")
    manifest = ModelBundleManifest.load(manifest_path)
    if manifest.schema_version != V2_BUNDLE_SCHEMA_VERSION or manifest.architecture != "v2":
        raise ValueError("Bundle is not a model_bundle_v2 artifact")
    manifest.validate(root)

    missing_files = [name for name in V2_REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        raise ValueError("Model v2 bundle is missing required files: " + ", ".join(missing_files))
    empty_model_dirs = [
        name
        for name in V2_REQUIRED_MODEL_DIRS
        if not (root / name).is_dir()
        or not any(path.is_file() for path in (root / name).rglob("*"))
    ]
    if empty_model_dirs:
        raise ValueError(
            "Model v2 bundle is missing populated component directories: "
            + ", ".join(empty_model_dirs)
        )

    checksums = _load_json(root / "checksums.json")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("checksums.json must be a non-empty mapping")
    for relative, expected in checksums.items():
        path = root / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {relative}")

    source_manifest = _load_json(root / "source_snapshot_manifest.json")
    snapshot_id = str(source_manifest.get("snapshot_id", "")).strip()
    if not snapshot_id or snapshot_id != str(manifest.source_snapshot_id or ""):
        raise ValueError("Bundle source snapshot identity is missing or inconsistent")
    if not manifest.code_version or manifest.code_version == "unknown":
        raise ValueError("Model v2 bundle requires an exact code version")
    required_cutoffs = {"training", "validation", "calibration", "outer_test"}
    if required_cutoffs - set(manifest.cutoffs):
        raise ValueError("Model v2 bundle is missing training/validation/calibration/outer-test cutoffs")
    if not manifest.schema_hash or not manifest.config_hash:
        raise ValueError("Model v2 bundle requires feature-schema and config hashes")

    decision = _load_json(root / "promotion_decision.json")
    if type(decision.get("eligible")) is not bool:
        raise ValueError("promotion_decision.json eligible must be a boolean")
    eligible = decision["eligible"]
    if promotion:
        from src.evaluation.promotion import PROMOTION_EVIDENCE_SCHEMA_VERSION

        if manifest.dirty_worktree:
            raise ValueError("Promotion requires a bundle built from a clean worktree")
        if not eligible:
            reasons = decision.get("reasons") or ["promotion gates did not pass"]
            raise ValueError("Candidate is not promotion eligible: " + "; ".join(map(str, reasons)))
        if decision.get("evidence_schema_version") != PROMOTION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "Promotion decision must declare the versioned evidence schema"
            )
        if not manifest.promotion_decision.get("eligible", eligible):
            raise ValueError("Manifest promotion decision disagrees with recorded decision")
        checks = decision.get("checks") or {}
        required_checks = {
            "point_in_time",
            "live_replay_parity",
            "artifact_contract",
            "calibration",
        }
        failed_checks = sorted(
            key for key in required_checks if checks.get(key) is not True
        )
        if failed_checks:
            raise ValueError(
                "Candidate lacks passing promotion evidence: "
                + ", ".join(failed_checks)
            )
        shadow = decision.get("shadow_slates") or {}
        if int(shadow.get("consecutive_successes", 0)) < 7:
            raise ValueError("Candidate requires seven consecutive successful shadow slates")

        folds_payload = _load_json(root / "training_folds.json")
        folds = folds_payload.get("folds", [])
        if not isinstance(folds, list) or len(folds) < 2:
            raise ValueError("Promotion requires repeated rolling-origin folds")
        for scorecard_name in (
            "baseline_scorecard.json",
            "candidate_scorecard.json",
            "slice_scorecard.json",
            "calibration_scorecard.json",
        ):
            if not _load_json(root / scorecard_name):
                raise ValueError(f"Promotion requires non-empty {scorecard_name}")

        recomputed = _validate_and_evaluate_promotion_evidence(root)
        stored_checks = decision.get("checks")
        if not isinstance(stored_checks, dict):
            raise ValueError("Promotion decision must contain a checks mapping")
        if any(type(value) is not bool for value in stored_checks.values()):
            raise ValueError("Promotion decision checks must be booleans")
        if dict(stored_checks) != dict(recomputed.checks):
            raise ValueError(
                "Stored promotion checks disagree with recomputed evidence"
            )
        stored_reasons = decision.get("reasons")
        if not isinstance(stored_reasons, list) or tuple(stored_reasons) != recomputed.reasons:
            raise ValueError(
                "Stored promotion reasons disagree with recomputed evidence"
            )
        if bool(decision.get("eligible")) != recomputed.eligible:
            raise ValueError(
                "Stored promotion eligibility disagrees with recomputed evidence"
            )
        if manifest.promotion_decision != decision:
            raise ValueError(
                "Manifest promotion decision disagrees with recorded decision"
            )

    return V2BundleValidation(
        bundle_id=manifest.bundle_id,
        files_checked=len(checksums),
        promotion_eligible=eligible and not manifest.dirty_worktree,
        code_version=manifest.code_version,
        source_snapshot_id=snapshot_id,
    )


def _validate_and_evaluate_promotion_evidence(root: Path):
    """Validate the versioned evidence files and recompute their policy result."""

    from src.evaluation.promotion import (
        PROMOTION_EVIDENCE_SCHEMA_VERSION,
        REQUIRED_CONTRACT_FLAGS,
        REQUIRED_TARGETS,
        evaluate_v2_promotion,
    )

    baseline = _load_json(root / "baseline_scorecard.json")
    candidate = _load_json(root / "candidate_scorecard.json")
    slices = _load_json(root / "slice_scorecard.json")
    calibration = _load_json(root / "calibration_scorecard.json")
    folds_payload = _load_json(root / "training_folds.json")

    for name, payload in (
        ("baseline_scorecard.json", baseline),
        ("candidate_scorecard.json", candidate),
        ("slice_scorecard.json", slices),
        ("calibration_scorecard.json", calibration),
    ):
        if payload.get("evidence_schema_version") != PROMOTION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"{name} must declare evidence schema {PROMOTION_EVIDENCE_SCHEMA_VERSION}"
            )

    _validate_scorecard_targets(
        baseline, REQUIRED_TARGETS, "baseline_scorecard.json", require_improvement=False
    )
    _validate_scorecard_targets(
        candidate, REQUIRED_TARGETS, "candidate_scorecard.json", require_improvement=True
    )
    _validate_aggregate(baseline, "baseline_scorecard.json", require_improvement=False)
    _validate_aggregate(candidate, "candidate_scorecard.json", require_improvement=True)

    comparison = candidate.get("baseline_comparison")
    _validate_target_mapping(comparison, REQUIRED_TARGETS, "candidate baseline_comparison")
    _validate_numeric_payload(comparison, "candidate baseline_comparison")
    _validate_recorded_comparisons(baseline, candidate, comparison, REQUIRED_TARGETS)

    _validate_folds(folds_payload, REQUIRED_TARGETS)

    major_slices = slices.get("major_slices")
    if not isinstance(major_slices, dict) or not major_slices:
        raise ValueError("slice_scorecard.json must contain non-empty major_slices")
    reconciliation: dict[str, float] = {}
    coverage_80: dict[str, float] = {}
    coverage_90: dict[str, float] = {}
    for name, metrics in major_slices.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"major slice {name!r} must contain a metric mapping")
        for field in ("reconciliation", "coverage_80", "coverage_90"):
            if field not in metrics:
                raise ValueError(f"major slice {name!r} is missing {field}")
            _finite_domain(metrics[field], f"major_slices[{name}].{field}")
        reconciliation[str(name)] = float(metrics["reconciliation"])
        coverage_80[str(name)] = float(metrics["coverage_80"])
        coverage_90[str(name)] = float(metrics["coverage_90"])

    aggregate = candidate["aggregate"]
    target_improvements = {
        target: candidate["targets"][target]["improvement"]
        for target in REQUIRED_TARGETS
    }
    participation = _require_mapping(candidate, "participation")
    minutes = _require_mapping(candidate, "minutes")
    contracts = _require_mapping(candidate, "contracts")
    operations = _require_mapping(candidate, "operations")
    for field in ("beats_baseline",):
        _require_bool(participation.get(field), f"participation.{field}")
        _require_bool(minutes.get(field), f"minutes.{field}")
    for field in (*REQUIRED_CONTRACT_FLAGS, "point_in_time", "live_replay_parity"):
        _require_bool(contracts.get(field), f"contracts.{field}")
    _finite_domain(operations.get("fallback_rate"), "operations.fallback_rate")
    _finite_domain(operations.get("degraded_rate"), "operations.degraded_rate")

    bootstrap = candidate.get("bootstrap_ci")
    if bootstrap is None:
        bootstrap_payload = _require_mapping(candidate, "bootstrap")
        bootstrap = bootstrap_payload.get("ci")
    if not isinstance(bootstrap, (list, tuple)) or len(bootstrap) != 2:
        raise ValueError("candidate bootstrap evidence must contain a two-value ci")
    _finite_metric(bootstrap[0], "candidate bootstrap_ci[0]")
    _finite_metric(bootstrap[1], "candidate bootstrap_ci[1]")

    thresholds = _promotion_thresholds(root)

    return evaluate_v2_promotion(
        normalized_mae_improvement=aggregate["normalized_mae_improvement"],
        target_improvements=target_improvements,
        participation_beats_baseline=participation["beats_baseline"],
        minutes_beats_baseline=minutes["beats_baseline"],
        coverage_80=calibration.get("coverage_80"),
        coverage_90=calibration.get("coverage_90"),
        point_in_time_valid=contracts["point_in_time"],
        bootstrap_ci=(float(bootstrap[0]), float(bootstrap[1])),
        contract_flags={
            name: contracts[name] for name in REQUIRED_CONTRACT_FLAGS
        },
        live_replay_parity=contracts["live_replay_parity"],
        core_reconciliation=operations.get("core_reconciliation"),
        fallback_rate=operations["fallback_rate"],
        degraded_rate=operations["degraded_rate"],
        major_slice_reconciliation=reconciliation,
        major_slice_coverage_80=coverage_80,
        major_slice_coverage_90=coverage_90,
        thresholds=thresholds,
        require_complete_evidence=True,
    )


def _promotion_thresholds(root: Path):
    """Load the versioned promotion policy captured in resolved config."""

    from src.evaluation.promotion import PromotionThresholds

    try:
        config = yaml.safe_load((root / "config_resolved.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Could not load resolved promotion policy") from exc
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("Resolved evaluation config must be a mapping")
    configured = evaluation.get("promotion_thresholds", {})
    if not isinstance(configured, dict):
        raise ValueError("Resolved promotion_thresholds must be a mapping")
    aliases = {
        "coverage_tolerance": "coverage_tolerance_overall",
    }
    fields = {
        field: configured.get(aliases.get(field, field), getattr(PromotionThresholds(), field))
        for field in PromotionThresholds.__dataclass_fields__
    }
    return PromotionThresholds(**fields)


def _validate_scorecard_targets(
    payload: dict[str, Any],
    required_targets: Sequence[str],
    name: str,
    *,
    require_improvement: bool,
) -> None:
    targets = payload.get("targets")
    _validate_target_mapping(targets, required_targets, f"{name} targets")
    for target in required_targets:
        metrics = targets[target]
        if not isinstance(metrics, dict):
            raise ValueError(f"{name} target {target} must contain a metric mapping")
        for field in ("rows", "mae"):
            if field not in metrics:
                raise ValueError(f"{name} target {target} is missing {field}")
        rows = metrics["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"{name} target {target}.rows must be a positive integer")
        _nonnegative_metric(metrics["mae"], f"{name} target {target}.mae")
        if require_improvement:
            if "improvement" not in metrics:
                raise ValueError(f"{name} target {target} is missing improvement")
            _finite_metric(metrics["improvement"], f"{name} target {target}.improvement")


def _validate_aggregate(
    payload: dict[str, Any], name: str, *, require_improvement: bool
) -> None:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{name} must contain an aggregate mapping")
    for field in ("mean_mae", "normalized_mae"):
        if field not in aggregate:
            raise ValueError(f"{name} aggregate is missing {field}")
        _nonnegative_metric(aggregate[field], f"{name} aggregate.{field}")
    if require_improvement:
        if "normalized_mae_improvement" not in aggregate:
            raise ValueError(f"{name} aggregate is missing normalized_mae_improvement")
        _finite_metric(
            aggregate["normalized_mae_improvement"],
            f"{name} aggregate.normalized_mae_improvement",
        )


def _validate_recorded_comparisons(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    comparison: dict[str, Any],
    required_targets: Sequence[str],
) -> None:
    baseline_normalized = float(baseline["aggregate"]["normalized_mae"])
    candidate_normalized = float(candidate["aggregate"]["normalized_mae"])
    if baseline_normalized <= 0:
        raise ValueError("baseline aggregate normalized_mae must be positive")
    expected_aggregate = 1.0 - candidate_normalized / baseline_normalized
    recorded_aggregate = float(
        candidate["aggregate"]["normalized_mae_improvement"]
    )
    if not math.isclose(expected_aggregate, recorded_aggregate, abs_tol=1e-9):
        raise ValueError(
            "candidate aggregate improvement disagrees with baseline comparison"
        )

    for target in required_targets:
        baseline_mae = float(baseline["targets"][target]["mae"])
        candidate_mae = float(candidate["targets"][target]["mae"])
        recorded = comparison[target]
        if not isinstance(recorded, dict):
            raise ValueError(f"candidate baseline_comparison.{target} must be a mapping")
        for field in ("baseline_mae", "candidate_mae"):
            if field not in recorded:
                raise ValueError(
                    f"candidate baseline_comparison.{target} is missing {field}"
                )
        if not math.isclose(float(recorded["baseline_mae"]), baseline_mae, abs_tol=1e-9):
            raise ValueError(
                f"candidate baseline_comparison.{target}.baseline_mae disagrees with baseline"
            )
        if not math.isclose(float(recorded["candidate_mae"]), candidate_mae, abs_tol=1e-9):
            raise ValueError(
                f"candidate baseline_comparison.{target}.candidate_mae disagrees with candidate"
            )
        expected = 0.0 if baseline_mae == 0 else 1.0 - candidate_mae / baseline_mae
        if baseline_mae == 0 and candidate_mae != 0:
            raise ValueError(
                f"candidate target {target} cannot compare against zero baseline MAE"
            )
        if not math.isclose(
            expected, float(candidate["targets"][target]["improvement"]), abs_tol=1e-9
        ):
            raise ValueError(
                f"candidate target {target} improvement disagrees with baseline comparison"
            )


def _validate_target_mapping(
    payload: object, required_targets: Sequence[str], name: str
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    missing = sorted(set(required_targets) - set(payload))
    if missing:
        raise ValueError(f"{name} is missing required targets: " + ", ".join(missing))


def _validate_folds(payload: dict[str, Any], required_targets: Sequence[str]) -> None:
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("Promotion requires at least two evaluated rolling-origin folds")
    from src.evaluation.chronology import validate_four_role_membership

    validate_four_role_membership(folds)
    for index, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict):
            raise ValueError(f"training fold {index} must be an object")
        for field in (
            "fold_id", "roles", "evaluation",
        ):
            if not fold.get(field):
                raise ValueError(f"training fold {index} is missing {field}")
        evaluation = fold["evaluation"]
        if not isinstance(evaluation, dict):
            raise ValueError(f"training fold {index}.evaluation must be a mapping")
        for field in ("rows", "games", "targets"):
            if field not in evaluation:
                raise ValueError(f"training fold {index}.evaluation is missing {field}")
        for field in ("rows", "games"):
            value = evaluation[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"training fold {index}.evaluation.{field} must be a positive integer"
                )
        _validate_scorecard_targets(
            {"targets": evaluation["targets"]},
            required_targets,
            f"training fold {index}.evaluation",
            require_improvement=False,
        )


def _require_mapping(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"candidate evidence requires a non-empty {name} mapping")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_metric(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        checked = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(checked):
        raise ValueError(f"{name} must be finite")
    return checked


def _finite_domain(value: object, name: str) -> float:
    checked = _finite_metric(value, name)
    if checked < 0 or checked > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return checked


def _nonnegative_metric(value: object, name: str) -> float:
    checked = _finite_metric(value, name)
    if checked < 0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _validate_numeric_payload(value: object, name: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _validate_numeric_payload(nested, f"{name}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_numeric_payload(nested, f"{name}[{index}]")
    else:
        _finite_metric(value, name)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load bundle JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Bundle JSON must be an object: {path.name}")
    return payload


def _write_json_once(path: Path, payload: Any) -> None:
    _write_text_once(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with exclusive_file_lock(lock_path):
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"Immutable bundle file already exists: {path}")
            return
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
