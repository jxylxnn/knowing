"""Fit and seal a diagnostic candidate from a fold's verified fit inputs only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.snapshots import validate_source_snapshot
from src.evaluation.chronology import digest_payload
from src.evaluation.baselines import TARGETS
from src.features.snapshot_inputs import load_request_history
from src.models.bundle import finalize_v2_bundle, write_v2_support_files, validate_v2_bundle
from src.models.versioning import ModelVersionRegistry, ModelBundleManifest
from src.training.v2_baseline import _component_payloads
from src.utils.code_provenance import source_tree_manifest


def fit_fold_candidate(*, boundary_request, fit_game_ids, models_dir, data_dir,
                       settings, policy, fold_id, calibration=None):
    """The first tune request bounds all labels used to fit component weights.

    Outer labels are not accepted by this interface. Calibration scales can
    only be attached with independent request membership and prediction hashes.
    This baseline stays ineligible; the payload makes that absence explicit.
    """
    allowed = {"window", "shrinkage_minutes", "exposure_weighted"}
    if set(settings) - allowed:
        raise ValueError("Unregistered baseline experiment settings")
    source = validate_source_snapshot(
        data_dir, boundary_request.source_snapshot_id,
        forecast_cutoff=boundary_request.forecast_cutoff,
    )
    history = load_request_history(data_dir, boundary_request)
    fit = history.loc[history.GAME_ID.isin(set(map(str, fit_game_ids)))].copy()
    if fit.empty or set(fit.GAME_ID) != set(map(str, fit_game_ids)):
        raise ValueError("Fit snapshot lacks declared fit-game observations")
    required = {"MIN", *TARGETS}
    if required - set(fit):
        raise ValueError("Fit snapshot lacks candidate count/minutes targets")
    fit = fit.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"], kind="stable").reset_index(drop=True)
    if fit.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Fit inputs duplicate a player-game")
    for column in required:
        values = pd.to_numeric(fit[column], errors="raise")
        if values.isna().any() or values.lt(0).any():
            raise ValueError("Baseline fitting requires known nonnegative target labels")
    window = int(settings.get("window", 20))
    if window < 1:
        raise ValueError("Rolling window must be positive")
    # Truncate per player before fitting; no tune/calibrate/outer rows enter.
    recent = fit.groupby("PLAYER_ID", sort=False).tail(window).reset_index(drop=True)
    components = _component_payloads(recent)
    components["minutes_models"]["kind"] = f"rolling_{window}_conditional_minutes"
    components["stat_rate_models"]["kind"] = f"rolling_{window}_per_minute_rates"
    shrinkage = float(settings.get("shrinkage_minutes", 0.0))
    if shrinkage < 0:
        raise ValueError("Shrinkage strength must be nonnegative")
    exposure_weighted = bool(settings.get("exposure_weighted", False))
    if shrinkage or exposure_weighted:
        for player, group in recent.groupby("PLAYER_ID", sort=False):
            minutes = pd.to_numeric(group.MIN)
            exposure = float(minutes.sum())
            for target in TARGETS:
                prior = components["stat_rate_models"]["global_rates"][target]["mean"]
                rate = (float(pd.to_numeric(group[target]).sum()) + shrinkage * prior)
                rate /= exposure + shrinkage if exposure + shrinkage else 1.0
                components["stat_rate_models"]["players"][str(player)][target]["mean"] = rate
    training_end = str(fit.GAME_DATE.max())
    evidence = {
        "kind": "fold_fit_only", "fold_id": fold_id, "policy_id": policy.policy_id,
        "source_tree_sha256": source_tree_manifest()["source_tree_sha256"],
        "settings": dict(settings), "fit_game_ids": sorted(set(fit.GAME_ID)),
        "fit_input_hash": digest_payload(fit.to_dict("records")),
        "source_snapshot_id": source.snapshot_id, "training_end": training_end,
        "calibration": calibration or {"status": "not_fitted"},
    }
    if calibration:
        uncalibrated_evidence = {**evidence, "calibration": {"status": "not_fitted"}}
        base_directory = (Path(models_dir) / "versions"
                          / ("fold-" + digest_payload(uncalibrated_evidence)[:24]))
        validate_v2_bundle(base_directory)
        expected_base = ModelBundleManifest.load(base_directory / ModelBundleManifest.FILE_NAME).bundle_id
        if calibration.get("base_bundle_id") != expected_base:
            raise ValueError("Calibration predictions belong to another fitted candidate")
        if (calibration.get("status") != "independent_mean_scale"
                or set(calibration.get("scales", {})) != set(TARGETS)
                or not calibration.get("prediction_hash")
                or not calibration.get("request_ids")
                or pd.Timestamp(calibration["start"]).date() <= pd.Timestamp(training_end).date()):
            raise ValueError("Calibration requires independent bound prediction evidence")
        components["calibrators"] = calibration
    candidate_id = "fold-" + digest_payload(evidence)[:24]
    registry = ModelVersionRegistry(models_dir)
    existing = registry.versions_dir / candidate_id
    if existing.exists():
        validate_v2_bundle(existing)
        if json.loads((existing / "training_provenance.json").read_text()) != evidence:
            raise ValueError("Existing fold candidate provenance differs")
        return existing
    candidate = registry.create_candidate(candidate_id)
    for directory, payload in components.items():
        path = candidate / directory
        path.mkdir(parents=True, exist_ok=False)
        (path / "baseline.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    simulation = {"regulation_team_minutes": 240,
                  "sampling_order": ["participation", "minutes", "rates"],
                  "seed": policy.seed, "simulations": policy.simulations}
    if "OVERTIME_PERIODS" in fit:
        periods = pd.to_numeric(fit.OVERTIME_PERIODS, errors="raise")
        if (periods.isna().any() or periods.lt(0).any() or periods.mod(1).ne(0).any()
                or fit.assign(__OT=periods).groupby("GAME_ID").__OT.nunique().gt(1).any()):
            raise ValueError("Overtime labels must be complete consistent game counts")
        counts = fit.assign(__OT=periods).drop_duplicates("GAME_ID").__OT.value_counts()
        simulation["overtime"] = {
            "fitted": True, "fit_input_hash": evidence["fit_input_hash"],
            "games": int(counts.sum()),
            "probabilities": [float(counts.get(i, 0) / counts.sum())
                              for i in range(int(counts.index.max()) + 1)],
        }
    decision = {"eligible": False, "reasons": [
        "diagnostic fold baseline requires official participation, calibrated distributions, and shadow qualification"
    ], "checks": {"point_in_time": True, "artifact_contract": True,
                  "participation_evidence": False, "official_replay": False}}
    config = {"architecture": "v2", "experiment": dict(settings),
              "evaluation_policy": policy.to_dict()}
    write_v2_support_files(
        candidate, resolved_config=config, source_snapshot_manifest=source.to_dict(),
        feature_schema={"architecture": "v2", "features": ["PLAYER_ID", "TEAM_ID"]},
        training_folds={"folds": [{"fold_id": fold_id, "fit_game_ids": evidence["fit_game_ids"]}]},
        baseline_scorecard={"status": "not_evaluated"},
        candidate_scorecard={"status": "not_evaluated"}, slice_scorecard={"status": "not_evaluated"},
        calibration_scorecard=calibration or {"status": "not_fitted"},
        simulation_parameters=simulation, promotion_decision=decision,
    )
    (candidate / "training_provenance.json").write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    finalize_v2_bundle(
        candidate, data_cutoff=training_end, config=config,
        source_snapshot_id=source.snapshot_id,
        cutoffs={"training": training_end, "validation": None,
                 "calibration": calibration.get("end") if calibration else None, "outer_test": None},
        metrics={"promotion_decision": decision},
    )
    return candidate
