#!/usr/bin/env python3
"""CLI entry point to build the walk-forward residual training dataset."""

import argparse
import logging
import sys
from pathlib import Path

from src.correction.walk_forward_residuals import WalkForwardResidualBuilder
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Build a walk-forward residual training dataset for mistake learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_residual_dataset.py --preset small --min-train-seasons 3
  python build_residual_dataset.py --start-season 22020 --end-season 22024
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/residual_training.parquet",
        help="Output Parquet path (default: data/evaluation/residual_training.parquet)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="data/evaluation/residual_training_summary.json",
        help="Output JSON summary path (default: data/evaluation/residual_training_summary.json)",
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=3,
        help="Minimum number of seasons to use for training before the first holdout (default: 3)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="full",
        choices=["small", "full"],
        help="Training preset (default: full)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["quick", "standard", "full"],
        help="Training mode override",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=["auto", "S", "M", "L", "XL"],
        help="Model size override",
    )
    parser.add_argument(
        "--start-season",
        type=str,
        default=None,
        help="First holdout season to include (e.g. 22020)",
    )
    parser.add_argument(
        "--end-season",
        type=str,
        default=None,
        help="Last holdout season to include (e.g. 22024)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Target stats to include (default: PTS REB AST STL BLK TOV)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel training across targets",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU even if available",
    )

    args = parser.parse_args()

    builder = WalkForwardResidualBuilder(
        config_path=args.config,
        output_path=args.output,
        summary_path=args.summary,
        min_train_seasons=args.min_train_seasons,
        preset=args.preset,
        mode=args.mode,
        model_size=args.model_size,
        start_season=args.start_season,
        end_season=args.end_season,
        targets=args.targets,
        max_workers=args.max_workers,
        parallel=args.parallel,
        use_gpu=False if args.no_gpu else None,
    )

    try:
        residual_df = builder.run()
        if residual_df.empty:
            logger.warning("Residual dataset is empty.")
            return 1
        logger.info("Residual dataset build complete: %d rows", len(residual_df))
        return 0
    except Exception as exc:
        logger.exception("Residual dataset build failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
