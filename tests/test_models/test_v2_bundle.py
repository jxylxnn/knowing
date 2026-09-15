import json

import pytest

from src.models.bundle import (
    finalize_v2_bundle,
    validate_v2_bundle,
    write_v2_support_files,
)
from src.evaluation.promotion import (
    PROMOTION_EVIDENCE_SCHEMA_VERSION,
    REQUIRED_TARGETS,
    evaluate_v2_promotion,
)
from src.models.versioning import ModelVersionRegistry
from promote_model import promote_with_rollback


def _bundle(tmp_path, *, eligible=True, dirty=False):
    candidate = tmp_path / "models" / "versions" / "candidate-a"
    for directory in (
        "availability_model", "minutes_models", "stat_rate_models", "calibrators"
    ):
        path = candidate / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "baseline.json").write_text(
            json.dumps({"kind": "declared_baseline", "version": 1}),
            encoding="utf-8",
        )

    baseline_targets = {
        target: {"rows": 100, "mae": 20.0} for target in REQUIRED_TARGETS
    }
    candidate_targets = {
        target: {"rows": 100, "mae": 18.0, "improvement": 0.10}
        for target in REQUIRED_TARGETS
    }
    baseline_scorecard = {
        "evidence_schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "targets": baseline_targets,
        "aggregate": {"mean_mae": 20.0, "normalized_mae": 1.0},
    }
    candidate_scorecard = {
        "evidence_schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "targets": candidate_targets,
        "aggregate": {
            "mean_mae": 18.0,
            "normalized_mae": 0.90,
            "normalized_mae_improvement": 0.10,
        },
        "baseline_comparison": {
            target: {"baseline_mae": 20.0, "candidate_mae": 18.0}
            for target in REQUIRED_TARGETS
        },
        "participation": {"beats_baseline": True},
        "minutes": {"beats_baseline": True},
        "bootstrap_ci": [-0.20, -0.05],
        "contracts": {
            "point_in_time": True,
            "live_replay_parity": True,
            "artifact_contract": True,
            "calibration": True,
        },
        "operations": {"fallback_rate": 0.01, "degraded_rate": 0.02, "core_reconciliation": 1.0},
    }
    fold_evaluation = {
        "rows": 100,
        "games": 20,
        "targets": baseline_targets,
    }
    from src.evaluation.chronology import digest_payload

    training_folds = {"folds": []}
    for index in (1, 2):
        roles = {}
        for role, first, last in (("fit", 1, 5), ("tune", 6, 10),
                                   ("calibrate", 11, 15), ("outer_test", 16, 20)):
            start = f"2026-0{index + 5}-{first:02d}"
            end = f"2026-0{index + 5}-{last:02d}"
            records = [{"GAME_ID": f"{index}-{role}-{game}",
                        "REQUEST_ID": f"r-{index}-{role}-{game}",
                        "GAME_DATE": start, "SOURCE_SNAPSHOT_ID": "snapshot-a"}
                       for game in range(20)]
            roles[role] = {"start": start, "end": end,
                           "game_ids": [row["GAME_ID"] for row in records],
                           "request_ids": [row["REQUEST_ID"] for row in records],
                           "request_records": records, "input_hash": digest_payload(records)}
        training_folds["folds"].append({"fold_id": f"fold-{index}", "roles": roles,
                                       "evaluation": fold_evaluation})
    slice_scorecard = {
        "evidence_schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "major_slices": {
            "all_players": {
                "reconciliation": 0.99,
                "coverage_80": 0.80,
                "coverage_90": 0.90,
            }
        },
    }
    calibration_scorecard = {
        "evidence_schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "coverage_80": 0.80,
        "coverage_90": 0.90,
    }
    recomputed = evaluate_v2_promotion(
        normalized_mae_improvement=0.10,
        target_improvements={target: 0.10 for target in REQUIRED_TARGETS},
        participation_beats_baseline=True,
        minutes_beats_baseline=True,
        coverage_80=0.80,
        coverage_90=0.90,
        point_in_time_valid=True,
        bootstrap_ci=(-0.20, -0.05),
        contract_flags={"artifact_contract": True, "calibration": True},
        live_replay_parity=True,
        core_reconciliation=1.0,
        fallback_rate=0.01,
        degraded_rate=0.02,
        major_slice_reconciliation={"all_players": 0.99},
        major_slice_coverage_80={"all_players": 0.80},
        major_slice_coverage_90={"all_players": 0.90},
        require_complete_evidence=True,
    )
    decision = {
        "evidence_schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "eligible": recomputed.eligible if eligible else False,
        "reasons": list(recomputed.reasons) if eligible else ["candidate failed baseline gate"],
        "checks": dict(recomputed.checks) if eligible else {},
        "shadow_slates": {"consecutive_successes": 7},
    }
    write_v2_support_files(
        candidate,
        resolved_config={"architecture": "v2"},
        source_snapshot_manifest={
            "snapshot_id": "snapshot-a",
            "created_at": "2026-08-01T00:00:00+00:00",
            "schema_version": "source_snapshot_v1",
            "files": [],
        },
        feature_schema={"version": "feature_schema_v4", "features": ["ROLL_PTS_5"]},
        training_folds=training_folds,
        baseline_scorecard=baseline_scorecard,
        candidate_scorecard=candidate_scorecard,
        slice_scorecard=slice_scorecard,
        calibration_scorecard=calibration_scorecard,
        simulation_parameters={"regulation_team_minutes": 240},
        promotion_decision=decision,
        environment={"python": "test", "packages": {}},
    )
    manifest = finalize_v2_bundle(
        candidate,
        data_cutoff="2026-08-01",
        config={"architecture": "v2"},
        source_snapshot_id="snapshot-a",
        cutoffs={
            "training": "2026-06-01",
            "validation": "2026-06-30",
            "calibration": "2026-07-15",
            "outer_test": "2026-08-01",
        },
        metrics={"promotion_decision": decision},
        code_version="test-commit",
        dirty_worktree=dirty,
    )
    return candidate, manifest


def test_v2_bundle_is_content_addressed_and_promotable(tmp_path):
    candidate, manifest = _bundle(tmp_path)

    report = validate_v2_bundle(candidate, promotion=True)
    assert report.bundle_id == manifest.bundle_id
    assert report.promotion_eligible

    registry = ModelVersionRegistry(tmp_path / "models")
    champion = registry.promote(candidate)
    assert champion.bundle_id == manifest.bundle_id
    assert champion.manifest_checksum


def test_v2_bundle_rejects_failed_gates_dirty_tree_and_tampering(tmp_path):
    failed_dir, _ = _bundle(tmp_path / "failed", eligible=False)
    with pytest.raises(ValueError, match="not promotion eligible"):
        validate_v2_bundle(failed_dir, promotion=True)

    dirty_dir, _ = _bundle(tmp_path / "dirty", dirty=True)
    with pytest.raises(ValueError, match="clean worktree"):
        validate_v2_bundle(dirty_dir, promotion=True)

    clean_dir, _ = _bundle(tmp_path / "tampered")
    (clean_dir / "candidate_scorecard.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|checksum mismatch"):
        validate_v2_bundle(clean_dir)


def test_post_promotion_failure_restores_previous_pointer(tmp_path):
    first, _ = _bundle(tmp_path / "first")
    second, _ = _bundle(tmp_path / "second")
    registry = ModelVersionRegistry(tmp_path / "deployment")
    registry.promote(first, version="first")
    previous = registry.manifest_path.read_bytes()

    def fail_post_check(_candidate):
        raise RuntimeError("golden forecast failed")

    with pytest.raises(RuntimeError, match="golden forecast"):
        promote_with_rollback(registry, second, post_check=fail_post_check)

    assert registry.manifest_path.read_bytes() == previous
