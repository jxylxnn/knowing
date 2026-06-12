"""CLI entry point for residual conformal interval calibration."""

from __future__ import annotations

import argparse
import logging
import sys

from src.correction.calibration import DEFAULT_CONFIDENCE_LEVELS, TARGETS, ResidualIntervalCalibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conformal confidence intervals from residual prediction errors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python calibrate_residual_intervals.py\n"
            "  python calibrate_residual_intervals.py --confidence-levels 0.8 0.9 0.95\n"
            "  python calibrate_residual_intervals.py --targets PTS REB AST\n"
        ),
    )
    parser.add_argument(
        "--input",
        default="data/evaluation/residual_training.parquet",
        help="Residual training parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/calibration",
        help="Directory for calibration interval artifacts.",
    )
    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONFIDENCE_LEVELS),
        help="Coverage levels to calibrate, e.g. 0.8 0.9 0.95.",
    )
    parser.add_argument(
        "--min-bucket-rows",
        type=int,
        default=500,
        help="Minimum rows required before writing a context bucket.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGETS),
        help="Target stats to calibrate.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    calibrator = ResidualIntervalCalibrator(
        confidence_levels=args.confidence_levels,
        min_bucket_rows=args.min_bucket_rows,
        targets=args.targets,
    )
    metadata = calibrator.calibrate_file(args.input, args.output_dir)
    written = [
        stat
        for stat, payload in metadata.get("targets", {}).items()
        if payload.get("status") == "written"
    ]
    print(
        f"Calibration complete: wrote {len(written)} target interval files to {args.output_dir}"
    )
    if written:
        print("Targets:", ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
