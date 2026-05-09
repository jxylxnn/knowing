#!/usr/bin/env python3
"""Backtest CLI — evaluate NBA prediction accuracy on historical games.

Usage:
    python backtest.py --from 2026-04-01 --to 2026-04-15
    python backtest.py --recent 14
    python backtest.py --from 2026-04-01 --to 2026-04-15 --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path for direct invocation
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.backtest_runner import BacktestRunner
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
        description="Backtest NBA prediction accuracy on historical games.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --from 2026-04-01 --to 2026-04-15
  python backtest.py --recent 14
  python backtest.py --from 2026-04-01 --to 2026-04-15 --output results.json
  python backtest.py --from 2026-04-01 --to 2026-04-15 --model-size L --no-progress
        """,
    )

    # Date range (mutually exclusive with --recent)
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--from", dest="date_from", type=str, metavar="YYYY-MM-DD",
        help="Start date (inclusive). Requires --to.",
    )
    date_group.add_argument(
        "--recent", type=int, metavar="DAYS",
        help="Backtest the most recent N days of games.",
    )
    parser.add_argument(
        "--to", dest="date_to", type=str, metavar="YYYY-MM-DD",
        help="End date (inclusive). Required when using --from.",
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
        "--output", "-o", type=str, default=None,
        help="Write JSON results to this file path.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-row progress logging.",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Recompute feature DataFrame instead of using cache.",
    )
    parser.add_argument(
        "--model-size", default="M", choices=["S", "M", "L", "XL"],
        help="Model size tier (default: M).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # --- Validate date arguments ---
    if args.recent is not None:
        if args.recent < 1:
            logger.error("--recent must be >= 1")
            sys.exit(1)
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
        logger.error("Must specify either --from/--to or --recent")
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

    # --- Run backtest ---
    logger.info("Running backtest: %s → %s", date_from, date_to)
    result = runner.run(
        date_from,
        date_to,
        force_recompute=args.force_recompute,
        progress=not args.no_progress,
    )

    # --- Output ---
    print()
    print(result.summary())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
