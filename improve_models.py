#!/usr/bin/env python3
"""Manually train and safely evaluate a champion/challenger model bundle.

The command never replaces the active artifacts until every promotion gate
passes.  It can be run after ``update_data.py`` has incorporated new games.

Examples:
    python improve_models.py --dry-run
    python improve_models.py --preset laptop_quality --mode standard --no-gpu
    python improve_models.py --rollback 20260709T120000Z
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.evaluation.backtest_runner import BacktestRunner
from src.evaluation.continual_learning import PromotionPolicy, evaluate_promotion
from src.evaluation.prediction_ledger import PredictionLedger
from src.models.model_manager import ModelManager
from src.models.versioning import ModelVersionRegistry

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def _date_window(data_dir: Path, holdout_days: int) -> tuple[str, str]:
    players_path = data_dir / "nba_players.csv"
    if not players_path.exists():
        raise FileNotFoundError(f"Players data not found: {players_path}")
    df = pd.read_csv(players_path, usecols=["GAME_DATE"])
    dates = pd.to_datetime(df["GAME_DATE"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("No valid GAME_DATE values found for evaluation")
    end = dates.max().normalize()
    start = end - timedelta(days=max(1, int(holdout_days)) - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _metrics(result) -> Dict[str, float]:
    return {
        stat: float(metrics.mae)
        for stat, metrics in result.per_target.items()
        if metrics is not None
    }


def _rows(result) -> Dict[str, int]:
    return {
        stat: int(metrics.num_samples)
        for stat, metrics in result.per_target.items()
        if metrics is not None
    }


def _run_training(args: argparse.Namespace, candidate_dir: Path, cutoff: str) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--data-dir", str(Path(args.data_dir).resolve()),
        "--models-dir", str(candidate_dir),
        "--cache-dir", str(Path(args.cache_dir).resolve()),
        "--preset", args.preset,
        "--mode", args.mode,
        "--model-size", args.model_size,
        "--test-split-date", cutoff,
    ]
    if args.parallel:
        command.append("--parallel")
    if args.no_gpu:
        command.append("--no-gpu")
    if args.feature_selection:
        command.extend(["--feature-selection", args.feature_selection])
    if args.selection_profile:
        command.extend(["--selection-profile", args.selection_profile])
    logger.info("Training challenger: %s", " ".join(command))
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Challenger training failed with exit code {completed.returncode}")


def _run_residual_training(args: argparse.Namespace, candidate_dir: Path) -> None:
    input_path = Path(args.residual_input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Residual input not found: {input_path}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_residual_models.py"),
        "--input", str(input_path),
        "--output-dir", str(candidate_dir / "residual"),
        "--min-rows", str(args.residual_min_rows),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Residual training failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--cache-dir", default="cache/training")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--preset", choices=["small", "laptop_quality", "full"], default="full")
    parser.add_argument("--model-size", choices=["S", "M", "L", "XL"], default="M")
    parser.add_argument("--learning-mode", choices=["full", "corrections-only"], default="full")
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--candidate-dir", default=None)
    parser.add_argument("--residual-input", default="data/evaluation/residual_training.parquet")
    parser.add_argument("--residual-min-rows", type=int, default=1000)
    parser.add_argument("--feature-selection", choices=["off", "smart"], default=None)
    parser.add_argument("--selection-profile", choices=["fast", "balanced", "max_accuracy"], default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but do not promote")
    parser.add_argument(
        "--allow-legacy-artifacts", action="store_true",
        help="Allow quarantined legacy artifacts for diagnostics only; promotion is disabled.",
    )
    parser.add_argument("--rollback", default=None, help="Point the champion manifest at a prior version")
    parser.add_argument("--report", default="reports/continual_learning/latest.json")
    parser.add_argument("--ledger-path", default="data/evaluation/prediction_history.parquet")
    args = parser.parse_args()

    if args.allow_legacy_artifacts and not args.dry_run:
        parser.error("--allow-legacy-artifacts disables promotion; use --dry-run")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(args.models_dir)
    registry = ModelVersionRegistry(root)

    if args.rollback:
        manifest = registry.rollback(args.rollback)
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0

    data_dir = Path(args.data_dir)
    date_start, date_end = _date_window(data_dir, args.holdout_days)
    ledger_path = Path(args.ledger_path)
    if not ledger_path.is_absolute():
        ledger_path = PROJECT_ROOT / ledger_path
    ledger = PredictionLedger(ledger_path)
    ledger_frame = ledger.load()
    if not ledger_frame.empty:
        actuals_path = data_dir / "nba_players.csv"
        actuals = pd.read_csv(actuals_path, low_memory=False)
        ledger_frame = ledger.reconcile_actuals(actuals)
    candidate_dir = Path(args.candidate_dir) if args.candidate_dir else registry.create_candidate()
    candidate_dir = candidate_dir.resolve()
    champion_dir = registry.active_dir().resolve()
    candidate_created = args.candidate_dir is None

    try:
        if args.learning_mode == "full":
            _run_training(args, candidate_dir, date_start)
        else:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            if candidate_dir != champion_dir:
                shutil.copytree(champion_dir, candidate_dir, dirs_exist_ok=True)
            _run_residual_training(args, candidate_dir)

        if args.allow_legacy_artifacts:
            print("UNSAFE LEGACY ARTIFACT MODE: replay and promotion are disabled")
        champion_manager = ModelManager(
            data_dir=str(data_dir),
            models_dir=str(root),
            allow_legacy_artifacts=args.allow_legacy_artifacts,
        )
        challenger_manager = ModelManager(
            data_dir=str(data_dir),
            models_dir=str(candidate_dir),
            allow_legacy_artifacts=args.allow_legacy_artifacts,
        )
        champion_runner = BacktestRunner(champion_manager, data_dir=str(data_dir), models_dir=str(champion_dir))
        challenger_runner = BacktestRunner(challenger_manager, data_dir=str(data_dir), models_dir=str(candidate_dir))
        champion_result = champion_runner.run(date_start, date_end, progress=False)
        challenger_result = challenger_runner.run(date_start, date_end, progress=False)

        policy = PromotionPolicy(min_rows_per_target=args.min_rows)
        decision = evaluate_promotion(
            _metrics(champion_result),
            _metrics(challenger_result),
            champion_errors=champion_runner.last_errors,
            candidate_errors=challenger_runner.last_errors,
            champion_rows=_rows(champion_result),
            candidate_rows=_rows(challenger_result),
            champion_coverage=champion_runner.last_coverage,
            candidate_coverage=challenger_runner.last_coverage,
            policy=policy,
        )
        payload: Dict[str, Any] = {
            "date_start": date_start,
            "date_end": date_end,
            "champion_dir": str(champion_dir),
            "candidate_dir": str(candidate_dir),
            "champion": champion_result.to_dict(),
            "candidate": challenger_result.to_dict(),
            "decision": decision.to_dict(),
            "ledger": {
                "path": str(ledger_path),
                "rows": int(len(ledger_frame)),
                "resolved_actuals": int(ledger_frame.get("ACTUAL", pd.Series(dtype=float)).notna().sum())
                if not ledger_frame.empty else 0,
            },
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if decision.eligible and not args.dry_run:
            manifest = registry.promote(
                candidate_dir,
                version=candidate_dir.name,
                data_cutoff=date_start,
                metrics=decision.to_dict(),
            )
            payload["promotion"] = manifest.to_dict()
            report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"PROMOTED {manifest.version}")
        else:
            print("NOT PROMOTED")
            for reason in decision.reasons:
                print(f"  - {reason}")
        print(f"Report: {report_path}")
        return 0 if decision.eligible else 2
    finally:
        # Failed or rejected candidates remain available for audit/rollback;
        # no active artifacts are deleted by this workflow.
        if candidate_created:
            logger.info("Candidate artifacts retained at %s", candidate_dir)


if __name__ == "__main__":
    raise SystemExit(main())
