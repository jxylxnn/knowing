"""Lightweight, evidence-first Model v2 baseline training.

This is the safe first champion candidate described by the architecture plan:
player rolling opportunity and per-minute rates, no neural blend, and no claim
of promotion eligibility when appearance-only history lacks inactive players.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.data.canonicalize import canonicalize_snapshot, load_canonical_table
from src.data.snapshots import (
    create_source_snapshot,
    load_source_snapshot_manifest,
)
from src.evaluation.baselines import TARGETS, add_lagged_baselines
from src.evaluation.folds import fold_manifest, rolling_origin_folds
from src.models.bundle import (
    finalize_v2_bundle,
    write_v2_support_files,
)
from src.models.versioning import ModelVersionRegistry


def train_v2_baseline(
    *,
    data_dir: str | Path,
    models_dir: str | Path,
    config_path: str | Path,
    snapshot_id: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Create one immutable v2 baseline candidate and return its directory."""

    data_root = Path(data_dir)
    model_root = Path(models_dir)
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if config.get("architecture") != "v2":
        raise ValueError("Model v2 training requires architecture: v2")

    resolved_snapshot = snapshot_id or _snapshot_id()
    manifest_path = (
        data_root / "manifests" / f"source_snapshot_{resolved_snapshot}.json"
    )
    if not manifest_path.is_file():
        if snapshot_id is not None:
            raise FileNotFoundError(f"Requested source snapshot does not exist: {snapshot_id}")
        create_source_snapshot(data_root, snapshot_id=resolved_snapshot)
    canonical_dir = data_root / "canonical" / f"snapshot={resolved_snapshot}"
    if not canonical_dir.is_dir():
        canonicalize_snapshot(data_root, resolved_snapshot)

    player_games = load_canonical_table(
        data_root, resolved_snapshot, "player_games"
    )
    if player_games.empty:
        raise ValueError("Cannot train Model v2 from an empty player-game table")
    required = {"PLAYER_ID", "GAME_ID", "GAME_DATE", "MIN", *TARGETS}
    if missing := required - set(player_games.columns):
        raise ValueError(f"Player-game data missing v2 targets: {sorted(missing)}")

    player_games = player_games.sort_values(
        ["GAME_DATE", "GAME_ID", "PLAYER_ID"], kind="stable"
    ).reset_index(drop=True)
    folds = _folds(player_games, config)
    enriched = add_lagged_baselines(player_games)
    scorecard = _baseline_scorecard(enriched)
    components = _component_payloads(player_games)

    registry = ModelVersionRegistry(model_root)
    candidate = registry.create_candidate(run_id or _snapshot_id(prefix="v2-baseline"))
    try:
        for directory, payload in components.items():
            target = candidate / directory
            target.mkdir(parents=True, exist_ok=False)
            (target / "baseline.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        source_manifest = load_source_snapshot_manifest(
            data_root, resolved_snapshot
        ).to_dict()
        promotion_decision = {
            "eligible": False,
            "reasons": [
                "appearance-only history cannot validate inactive-player participation",
                "rolling baseline requires official multi-fold replay evidence before promotion",
            ],
            "checks": {
                "point_in_time": False,
                "artifact_contract": True,
                "participation_evidence": False,
                "official_replay": False,
            },
        }
        write_v2_support_files(
            candidate,
            resolved_config=config,
            source_snapshot_manifest=source_manifest,
            feature_schema={
                "version": "feature_schema_v4",
                "architecture": "v2",
                "features": [
                    "PLAYER_ID", "TEAM_ID", "OPPONENT_ID", "PLAY_PROB",
                    "EXPECTED_MINUTES", "DATA_QUALITY",
                ],
            },
            training_folds=fold_manifest(
                folds,
                source_hash=hashlib.sha256(
                    (canonical_dir / "player_games.parquet").read_bytes()
                ).hexdigest(),
            ),
            baseline_scorecard=scorecard,
            candidate_scorecard={
                "status": "not_evaluated",
                "targets": {}, "aggregate": {},
                "reason": "Final refit has no candidate-specific out-of-fold replay",
            },
            slice_scorecard={"slices": {}, "status": "not_yet_available"},
            calibration_scorecard={
                "status": "empirical_training_residuals_only",
                "coverage_80": None,
                "coverage_90": None,
            },
            simulation_parameters={
                "regulation_team_minutes": 240,
                "sampling_order": ["participation", "minutes", "rates"],
            },
            promotion_decision=promotion_decision,
        )
        cutoff = pd.to_datetime(player_games["GAME_DATE"]).max().date().isoformat()
        cutoffs = {
            "training": cutoff, "validation": None,
            "calibration": None, "outer_test": None,
        }
        provenance = {
            "kind": "final_refit_diagnostic_only",
            "training_start": str(player_games["GAME_DATE"].min()),
            "training_end": cutoff,
            "rows": len(player_games),
            "games": int(player_games["GAME_ID"].nunique()),
            "source_snapshot_id": resolved_snapshot,
            "canonical_content_id": json.loads(
                (canonical_dir / "manifest.json").read_text()
            )["content_id"],
            "selection": "fixed_rolling_20_no_tuning",
            "calibration": "not_fitted",
            "candidate_evaluation": "not_performed",
        }
        (candidate / "training_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        finalize_v2_bundle(
            candidate,
            data_cutoff=cutoff,
            config=config,
            source_snapshot_id=resolved_snapshot,
            cutoffs=cutoffs,
            metrics={
                "diagnostic_baseline": scorecard,
                "candidate_evaluation": "not_performed",
                "promotion_decision": promotion_decision,
            },
        )
        return candidate
    except Exception:
        # An unsealed failed candidate is not evidence. Removing only the
        # directory created by this invocation keeps the registry clean.
        if candidate.exists() and not (candidate / "bundle_manifest.json").exists():
            shutil.rmtree(candidate)
        raise


def _component_payloads(player_games: pd.DataFrame) -> dict[str, dict[str, Any]]:
    frame = player_games.copy()
    minutes = pd.to_numeric(frame["MIN"], errors="coerce").fillna(0).clip(lower=0)
    players: dict[str, dict[str, float]] = {}
    for player_id, group in frame.groupby("PLAYER_ID", sort=False):
        index = group.index
        recent = index[-20:]
        player_minutes = minutes.loc[recent]
        record: dict[str, float] = {
            "minutes_mean": float(player_minutes.mean()),
            "minutes_std": float(player_minutes.std(ddof=0)),
        }
        for target in TARGETS:
            totals = pd.to_numeric(frame.loc[recent, target], errors="coerce").fillna(0)
            valid_minutes = player_minutes.where(player_minutes > 0)
            rates = totals.div(valid_minutes).replace([np.inf, -np.inf], np.nan).dropna()
            record[f"{target}_rate"] = float(rates.mean()) if len(rates) else 0.0
            record[f"{target}_rate_std"] = float(rates.std(ddof=0)) if len(rates) else 0.0
        players[str(player_id)] = record

    global_minutes = float(minutes[minutes > 0].median()) if (minutes > 0).any() else 24.0
    global_rates = {}
    for target in TARGETS:
        totals = pd.to_numeric(frame[target], errors="coerce").fillna(0)
        rates = totals.div(minutes.where(minutes > 0)).replace([np.inf, -np.inf], np.nan).dropna()
        global_rates[target] = {
            "mean": float(rates.median()) if len(rates) else 0.0,
            "std": float(rates.std(ddof=0)) if len(rates) else 0.0,
        }
    return {
        "availability_model": {
            "kind": "declared_status_and_recent_appearance_baseline",
            "default_p_active": 0.90,
            "default_p_play_given_active": 0.95,
            "training_label_coverage": "appearances_only",
        },
        "minutes_models": {
            "kind": "rolling_20_conditional_minutes",
            "global_minutes": global_minutes,
            "players": {
                key: {
                    "mean": value["minutes_mean"],
                    "std": value["minutes_std"],
                }
                for key, value in players.items()
            },
        },
        "stat_rate_models": {
            "kind": "rolling_20_per_minute_rates",
            "targets": list(TARGETS),
            "global_rates": global_rates,
            "players": {
                key: {
                    target: {
                        "mean": value[f"{target}_rate"],
                        "std": value[f"{target}_rate_std"],
                    }
                    for target in TARGETS
                }
                for key, value in players.items()
            },
        },
        "calibrators": {
            "kind": "declared_uncalibrated_baseline",
            "version": "baseline_v1",
            "promotion_eligible": False,
        },
    }


def _baseline_scorecard(frame: pd.DataFrame) -> dict[str, Any]:
    targets = {}
    for target in TARGETS:
        prediction = pd.to_numeric(
            frame[f"BASELINE_ROLL_10_{target}"], errors="coerce"
        )
        actual = pd.to_numeric(frame[target], errors="coerce")
        valid = prediction.notna() & actual.notna()
        error = (prediction[valid] - actual[valid]).abs()
        targets[target] = {
            "rows": int(valid.sum()),
            "mae": float(error.mean()) if len(error) else None,
            "rmse": float(np.sqrt(np.square(error).mean())) if len(error) else None,
        }
    valid_mae = [item["mae"] for item in targets.values() if item["mae"] is not None]
    return {
        "baseline": "rolling_10",
        "status": "diagnostic_appearance_only",
        "candidate_evidence": False,
        "targets": targets,
        "aggregate": {
            "mean_mae": float(np.mean(valid_mae)) if valid_mae else None,
            "normalized_mae": 1.0,
        },
    }


def _folds(frame: pd.DataFrame, config: Mapping[str, Any]):
    evaluation = config.get("evaluation", {})
    folds = rolling_origin_folds(
        frame,
        validation_days=int(evaluation.get("validation_days", 14)),
        test_days=int(evaluation.get("test_days", 28)),
        min_train_days=int(evaluation.get("min_train_days", 90)),
    )
    if not folds:
        raise ValueError("Insufficient chronological history for one v2 fold")
    return folds


def _snapshot_id(*, prefix: str = "snapshot") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


__all__ = ["train_v2_baseline"]
