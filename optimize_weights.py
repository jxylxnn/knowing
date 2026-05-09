#!/usr/bin/env python3
"""Optimize ensemble blend weights using holdout backtesting.

Usage:
    python optimize_weights.py --from 2026-04-15 --to 2026-05-01
    python optimize_weights.py --recent 14
    python optimize_weights.py --dry-run --from 2026-04-15 --to 2026-05-01
    python optimize_weights.py --rollback 3
    python optimize_weights.py --list
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.config import Config
from src.evaluation.backtest_runner import BacktestRunner
from src.evaluation.ensemble_optimizer import EnsembleOptimizer
from src.evaluation.weight_store import WeightStore
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-optimizing ensemble weight tuner for NBA predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_weights.py --from 2026-04-15 --to 2026-05-01
  python optimize_weights.py --recent 14 --model-size L
  python optimize_weights.py --dry-run --from 2026-04-15 --to 2026-05-01
  python optimize_weights.py --rollback 3
  python optimize_weights.py --list
        """,
    )

    # Actions
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run optimization but do NOT save results.",
    )
    parser.add_argument(
        "--rollback", type=int, metavar="VERSION",
        help="Roll back to a specific weight version number.",
    )
    parser.add_argument(
        "--list", dest="list_versions", action="store_true",
        help="List all saved weight versions and exit.",
    )

    # Date range
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--from", dest="date_from", type=str, metavar="YYYY-MM-DD",
        help="Start of holdout period (inclusive). Requires --to.",
    )
    date_group.add_argument(
        "--recent", type=int, metavar="DAYS",
        help="Use the most recent N days as holdout period.",
    )
    parser.add_argument(
        "--to", dest="date_to", type=str, metavar="YYYY-MM-DD",
        help="End of holdout period (inclusive).",
    )

    # Verification
    parser.add_argument(
        "--verify-from", type=str, metavar="YYYY-MM-DD",
        help="Override verification period start.",
    )
    parser.add_argument(
        "--verify-to", type=str, metavar="YYYY-MM-DD",
        help="Override verification period end.",
    )

    # Optimization parameters
    parser.add_argument(
        "--accept-margin", type=float, default=None,
        help="Minimum improvement fraction to accept (default from config).",
    )
    parser.add_argument(
        "--verify-margin", type=float, default=None,
        help="Maximum degradation fraction on verification (default from config).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Maximum optimizer iterations (default from config).",
    )

    # I/O
    parser.add_argument(
        "--data-dir", default="data",
        help="Path to raw data directory (default: data).",
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Path to trained model artifacts (default: models).",
    )
    parser.add_argument(
        "--weights-dir", default="models/blend_weights",
        help="Path to versioned weight store (default: models/blend_weights).",
    )
    parser.add_argument(
        "--model-size", default="M", choices=["S", "M", "L", "XL"],
        help="Model size tier (default: M).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-row progress logging during backtests.",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Recompute feature DataFrame instead of using cache.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def cmd_list(store: WeightStore) -> None:
    """List all saved weight versions."""
    versions = store.list_versions()
    if not versions:
        print("No weight versions found.")
        return

    current = store.load_current()
    current_ver = current.version if current else None

    print(f"{'Version':>8}  {'Current':^8}  {'Score':>10}  {'Created':<20}  Description")
    print("-" * 90)
    for v in versions:
        marker = "▶" if v["version"] == current_ver else " "
        score = f"{v['backtest_score']:.4f}" if v["backtest_score"] is not None else "N/A"
        created = v["created_at"][:19] if v["created_at"] else ""
        desc = v.get("description", "")[:50]
        print(f"  v{v['version']:04d}     {marker:^6}  {score:>10}  {created:<20}  {desc}")


def cmd_rollback(store: WeightStore, version: int) -> None:
    """Roll back to a previous weight version."""
    print(f"Rolling back to v{version:04d}...")
    weights = store.rollback(version)
    if weights is None:
        print(f"ERROR: Version v{version:04d} not found.")
        sys.exit(1)
    print(f"Rolled back to v{version:04d} (saved as v{weights.version:04d})")
    print()
    print(weights.summary())


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run the optimization loop."""
    # --- Load config ---
    config_path = Path("config/default.yaml")
    if config_path.exists():
        config = Config.from_yaml(config_path)
        so_config = config.self_optimization
    else:
        from src.config.config import SelfOptimizationConfig
        so_config = SelfOptimizationConfig()

    accept_margin = args.accept_margin if args.accept_margin is not None else so_config.accept_margin
    verify_margin = args.verify_margin if args.verify_margin is not None else so_config.verification_margin
    max_iter = args.max_iterations if args.max_iterations is not None else so_config.max_iterations

    # --- Resolve date range ---
    if args.recent is not None:
        end = datetime.now()
        start = end - __import__("pandas").Timedelta(days=args.recent)
        date_from = start.strftime("%Y-%m-%d")
        date_to = end.strftime("%Y-%m-%d")
    elif args.date_from:
        if not args.date_to:
            logger.error("--to is required when using --from")
            sys.exit(1)
        date_from = args.date_from
        date_to = args.date_to
    else:
        logger.error("Must specify either --from/--to or --recent for optimization")
        sys.exit(1)

    # --- Initialize ---
    logger.info("Initializing ModelManager (size=%s)...", args.model_size)
    manager = ModelManager(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        model_size=args.model_size,
    )

    runner = BacktestRunner(
        manager,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
    )

    store = WeightStore(args.weights_dir)

    optimizer = EnsembleOptimizer(
        runner,
        store,
        accept_margin=accept_margin,
        verification_margin=verify_margin,
        max_iterations=max_iter,
    )

    # --- Run ---
    print(f"\n{'='*60}")
    print(f"  Self-Optimizing Ensemble Weight Tuner")
    print(f"  Holdout: {date_from} → {date_to}")
    if args.verify_from:
        print(f"  Verification: {args.verify_from} → {args.verify_to or date_to}")
    print(f"  Accept margin: {accept_margin:.1%} | Verify margin: {verify_margin:.1%}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    result = optimizer.optimize(
        date_from,
        date_to,
        verification_start=args.verify_from,
        verification_end=args.verify_to,
        progress=not args.no_progress,
    )

    # --- Report ---
    print()
    print(f"  Baseline score:  {result.baseline_score:.4f}")
    print(f"  Candidate score: {result.candidate_score:.4f}")
    print(f"  Improvement:     {result.improvement_pct:+.2f}%")
    print(f"  Iterations:      {result.num_iterations}")
    print(f"  Accepted:        {'YES ✅' if result.accepted else 'NO ❌'}")

    if not result.accepted:
        print(f"  Reason:          {result.rejection_reason}")
        print()
        print("  Current weights retained. No changes saved.")
        return

    if args.dry_run:
        print()
        print("  DRY RUN — weights NOT saved. Candidate would be:")
        print()
        print(result.weights.summary())
        # Restore original weights
        current = store.load_current()
        if current:
            manager.use_ensemble_weights(current)
        return

    print(f"  New version:     v{result.weights.version:04d}")
    print()
    print(result.weights.summary())
    print()
    print("  Weights saved and deployed.")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    store = WeightStore(args.weights_dir)

    # Handle list/rollback actions (no models needed)
    if args.list_versions:
        cmd_list(store)
        return

    if args.rollback is not None:
        cmd_rollback(store, args.rollback)
        return

    # Run optimization
    cmd_optimize(args)


if __name__ == "__main__":
    main()
