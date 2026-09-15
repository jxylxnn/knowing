"""Small, attributable fold trials with independent tune and calibration roles."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.eligible_panel import TARGETS
from src.evaluation.chronology import digest_payload, four_role_folds
from src.evaluation.distribution_scores import score_distributions, slice_scorecards
from src.evaluation.significance import paired_macro_bootstrap
from src.evaluation.opportunity_scores import score_opportunity
from src.evaluation.request_baselines import BASELINES, paired_baseline_scores, request_baselines
from src.evaluation.request_replay import replay_requests, request_record, write_requests
from src.features.snapshot_inputs import read_snapshot_csv
from src.models.versioning import ModelBundleManifest
from src.models.bundle import write_checksums
from src.operations.forecast_export import export_forecast_run
from src.training.fold_candidate import fit_fold_candidate


DEFAULT_TRIALS = ({"window": 5}, {"window": 10}, {"window": 20})


def run_fold_campaign(records, *, data_dir, models_dir, output_dir, actuals_snapshot_id,
                      policy, trials=DEFAULT_TRIALS, windows=None, baseline_evidence=None):
    """Persist each trial before outer labels are scored; never promote here."""
    if any(set(trial) - {"window"} for trial in trials):
        _require_complete_baseline(baseline_evidence)
    if not trials or len(trials) > 12:
        raise ValueError("Campaign requires one to twelve predeclared small trials")
    if any(record["request"]["scenario"] == "official" for record in records):
        raise ValueError("Baseline campaigns must be explicitly diagnostic")
    requests = {item["request_id"]: ForecastRequest(**item["request"]) for item in records}
    if len(requests) != len(records) or any(key != value.request_id for key, value in requests.items()):
        raise ValueError("Campaign request identities are invalid")
    frame = pd.DataFrame([{"REQUEST_ID": key, "GAME_ID": req.game_id,
                           "GAME_DATE": req.game_date, "SOURCE_SNAPSHOT_ID": req.source_snapshot_id}
                          for key, req in requests.items()])
    folds = four_role_folds(frame, policy=policy, **(windows or {}))
    if not folds["sufficient_folds"]:
        raise ValueError("Insufficient nonempty four-role folds under the frozen policy")
    # These labels are scoring-only. Component fitting reads its own earlier snapshot.
    actuals = read_snapshot_csv(data_dir, actuals_snapshot_id, ("player_game_eligibility.csv",))
    required = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "ACTIVE", "APPEARED", "MIN", *TARGETS}
    if required - set(actuals):
        raise ValueError("Campaign actuals need a complete official eligible-player panel")
    source_hash = digest_payload(actuals.to_dict("records"))
    identity = {"folds": folds, "trials": list(trials), "actuals_snapshot_id": actuals_snapshot_id,
                "actuals_hash": source_hash}
    root = Path(output_dir) / digest_payload(identity)
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "campaign.json", identity)
    summaries = []
    for fold in folds["folds"]:
        directory = root / fold["fold_id"]
        directory.mkdir()
        role_requests = {role: [requests[key] for key in membership["request_ids"]]
                         for role, membership in fold["roles"].items()}
        boundary = min(role_requests["tune"], key=lambda req: req.forecast_cutoff)
        tune_labels = _role_labels(
            min(role_requests["calibrate"], key=lambda req: req.forecast_cutoff), data_dir
        )
        calibration_labels = _role_labels(
            min(role_requests["outer_test"], key=lambda req: req.forecast_cutoff), data_dir
        )
        trials_record = []
        for settings in trials:
            candidate = fit_fold_candidate(
                boundary_request=boundary, fit_game_ids=fold["roles"]["fit"]["game_ids"],
                models_dir=models_dir, data_dir=data_dir, settings=settings,
                policy=policy, fold_id=fold["fold_id"],
            )
            tune = _run_role(role_requests["tune"], candidate, policy, data_dir)
            score, detail = score_distributions(
                tune.forecasts, tune_labels, target_scales=dict(policy.target_scales)
            )
            baselines = request_baselines(tune.requests, tune.forecasts, data_dir=data_dir)
            comparisons, _ = paired_baseline_scores(
                detail, baselines, target_scales=dict(policy.target_scales)
            )
            objective = comparisons["candidate"]["normalized_mae"]
            if objective is None or not np.isfinite(objective):
                raise ValueError("Tune window lacks comparable labels for all six targets")
            trial_dir = directory / digest_payload(settings)
            trial_dir.mkdir()
            _persist_role(trial_dir, tune)
            trial = {"settings": settings, "candidate": str(candidate), "objective": objective,
                     "scorecard": score, "baseline_comparisons": comparisons}
            _write_json(trial_dir / "tune_scorecard.json", trial)
            trials_record.append(trial)
        winner = min(trials_record, key=lambda row: (row["objective"], digest_payload(row["settings"])))
        baseline_scores = winner["baseline_comparisons"]
        comparator = min(BASELINES, key=lambda name: (baseline_scores[name]["normalized_mae"], name))
        selection = {"winner": winner["settings"], "comparator": comparator,
                     "trials": trials_record, "policy_id": policy.policy_id}
        _write_json(directory / "selection.json", selection)
        # Selection is frozen before independent calibration forecasts or outer labels are used.
        calibration_run = _run_role(role_requests["calibrate"], winner["candidate"], policy, data_dir)
        _persist_role(directory / "calibration", calibration_run)
        calibration = _fit_mean_scales(
            calibration_run, calibration_labels, minimum_rows=policy.min_slice_rows
        )
        calibrated = fit_fold_candidate(
            boundary_request=boundary, fit_game_ids=fold["roles"]["fit"]["game_ids"],
            models_dir=models_dir, data_dir=data_dir, settings=winner["settings"],
            policy=policy, fold_id=fold["fold_id"], calibration=calibration,
        )
        outer = _run_role(role_requests["outer_test"], calibrated, policy, data_dir)
        _persist_role(directory / "outer", outer)
        score, detail = score_distributions(
            outer.forecasts, actuals, target_scales=dict(policy.target_scales)
        )
        baseline_frame = request_baselines(outer.requests, outer.forecasts, data_dir=data_dir)
        comparisons, paired = paired_baseline_scores(
            detail, baseline_frame, target_scales=dict(policy.target_scales)
        )
        detail.to_parquet(directory / "outer_scores.parquet", index=False)
        paired.to_parquet(directory / "paired_errors.parquet", index=False)
        opportunity = score_opportunity(
            outer.requests, outer.forecasts, actuals, data_dir=data_dir,
            minimum_rows=policy.min_slice_rows,
        )
        slices = slice_scorecards(detail, minimum_rows=policy.min_slice_rows)
        bootstrap = paired_macro_bootstrap(
            paired, baseline_column=comparator, target_scales=dict(policy.target_scales),
            samples=policy.bootstrap_samples, seed=policy.seed,
        )
        _write_json(directory / "slice_scorecard.json", slices)
        summary = {"fold_id": fold["fold_id"], "bootstrap": bootstrap,
                   "opportunity": opportunity, "candidate": str(calibrated),
                   "comparator": comparator, "candidate_scorecard": score,
                   "baseline_comparisons": comparisons, "promotion_eligible": False}
        _write_json(directory / "outer_scorecard.json", summary)
        summaries.append(summary)
    _write_json(root / "summary.json", {
        "folds": summaries, "promotion_eligible": False,
        "reason": "Diagnostic trials do not establish release qualification",
    })
    write_checksums(root)
    return root


def _run_role(requests, candidate, policy, data_dir):
    bundle_id = ModelBundleManifest.load(Path(candidate) / ModelBundleManifest.FILE_NAME).bundle_id
    records = [request_record(replace(req, model_bundle_id=bundle_id), seed=policy.seed,
                              simulations=policy.simulations) for req in requests]
    return replay_requests(records, bundle_dir=candidate, data_dir=data_dir)


def _persist_role(directory, run):
    directory.mkdir(parents=True, exist_ok=True)
    write_requests(directory / "requests.json", run.requests)
    export_forecast_run(directory, run.forecasts, run.samples)


def _fit_mean_scales(run, actuals, *, minimum_rows):
    scales = {}
    for target in TARGETS:
        forecasts = run.forecasts.loc[run.forecasts.STAT.eq(target)].copy()
        forecasts["PLAYER_ID"] = forecasts.PLAYER_ID.astype(str)
        joined = forecasts.merge(actuals[["GAME_ID", "PLAYER_ID", target]],
                                 on=["GAME_ID", "PLAYER_ID"], validate="many_to_one")
        values = pd.to_numeric(joined[target], errors="raise")
        known = values.notna()
        predicted = float(joined.loc[known, "MEAN"].sum())
        observed = float(values[known].sum())
        if known.sum() < minimum_rows or predicted <= 0 or observed < 0:
            raise ValueError("Insufficient independent mean-calibration evidence")
        scales[target] = observed / predicted
    games = set(run.forecasts.GAME_ID)
    labels = actuals.loc[actuals.GAME_ID.isin(games)].sort_values(["GAME_ID", "PLAYER_ID"])
    return {"status": "independent_mean_scale", "scales": scales,
            "base_bundle_id": str(run.forecasts.MODEL_BUNDLE_ID.iloc[0]),
            "labels_hash": digest_payload(labels.to_dict("records")),
            "start": run.forecasts.GAME_DATE.min(), "end": run.forecasts.GAME_DATE.max(),
            "request_ids": sorted(run.forecasts.REQUEST_ID.unique()),
            "prediction_hash": digest_payload(
                run.forecasts.drop(columns="GENERATED_AT").to_dict("records")
            ),
            "calibration_claim": "mean_only_not_interval_or_zero_calibration"}


def _role_labels(boundary_request, data_dir):
    """Selection/calibration labels must themselves exist before the next role."""
    return read_snapshot_csv(
        data_dir, boundary_request.source_snapshot_id, ("player_game_eligibility.csv",),
        cutoff=boundary_request.forecast_cutoff,
    )


def _write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, default=str, allow_nan=False)
        handle.write("\n")


def _require_complete_baseline(directory):
    """Modeling experiments cannot precede an honest complete baseline campaign."""
    import hashlib

    if directory is None:
        raise ValueError("Component experiments require completed baseline evidence")
    root = Path(directory)
    checksums = json.loads((root / "checksums.json").read_text())
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path.name != "checksums.json"}
    if (not isinstance(checksums, dict)
            or not {"campaign.json", "summary.json"}.issubset(checksums)
            or set(checksums) != actual_files):
        raise ValueError("Baseline campaign requires complete immutable checksums")
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Unsafe baseline evidence path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Baseline campaign checksum mismatch")
    summary = json.loads((root / "summary.json").read_text())
    campaign = json.loads((root / "campaign.json").read_text())
    minimum = campaign["folds"]["policy"]["min_folds"]
    folds = summary.get("folds", [])
    if len(folds) < minimum:
        raise ValueError("Baseline campaign lacks the frozen minimum fold count")
    for fold in folds:
        score = fold["candidate_scorecard"]
        opportunity = fold["opportunity"]
        if (score["expected_rows"] != score["known_rows"] or not score["known_rows"]
                or opportunity["participation"]["active"]["missing"]
                or opportunity["participation"]["appeared"]["missing"]):
            raise ValueError("Baseline campaign has incomplete eligible-population evidence")
