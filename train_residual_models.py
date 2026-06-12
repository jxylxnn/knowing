#!/usr/bin/env python3
"""Train per-target residual correction models.

Reads the walk-forward residual dataset produced by Ticket 1 and trains
one CatBoost model per stat (PTS, REB, AST, STL, BLK, TOV) that predicts
how wrong the base model usually is.

Usage:
    python train_residual_models.py \\
        --input data/evaluation/residual_training.parquet \\
        --output-dir models/residual
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.correction.residual_trainer import ResidualModelTrainer
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent


def _print_results_table(result, console=None):
    """Print a summary table of training results."""
    if RICH_AVAILABLE and console:
        table = Table(
            title="Residual Model Training Results",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Stat", style="bold cyan")
        table.add_column("Rows", justify="right")
        table.add_column("Base MAE", justify="right")
        table.add_column("Corrected MAE", justify="right")
        table.add_column("Improvement", justify="right")
        table.add_column("Improvement %", justify="right")
        table.add_column("Status", justify="center")

        for stat, r in sorted(result.targets.items()):
            status_style = "green" if r.status == "accepted" else "red"
            table.add_row(
                stat,
                str(r.rows),
                f"{r.base_mae:.4f}",
                f"{r.corrected_mae:.4f}",
                f"{r.mae_improvement:.4f}",
                f"{r.mae_improvement_pct:.2f}%",
                f"[{status_style}]{r.status}[/{status_style}]",
            )

        console.print(table)
        if result.total_time:
            console.print(f"\nTotal training time: {result.total_time:.1f}s")
    else:
        print("\n=== Residual Model Training Results ===")
        print(f"{'Stat':<6} {'Rows':>6} {'Base MAE':>10} {'Corrected':>10} {'Improve':>10} {'%':>8} {'Status':>10}")
        print("-" * 70)
        for stat, r in sorted(result.targets.items()):
            print(
                f"{stat:<6} {r.rows:>6} {r.base_mae:>10.4f} {r.corrected_mae:>10.4f} "
                f"{r.mae_improvement:>10.4f} {r.mae_improvement_pct:>7.2f}% {r.status:>10}"
            )
        print(f"\nTotal training time: {result.total_time:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train per-target residual correction models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_residual_models.py\n"
            "  python train_residual_models.py --input data/evaluation/residual_training.parquet\n"
            "  python train_residual_models.py --targets PTS REB AST --min-rows 500\n"
        ),
    )
    parser.add_argument(
        "--input",
        default="data/evaluation/residual_training.parquet",
        help="Path to residual training parquet (default: data/evaluation/residual_training.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/residual",
        help="Output directory for model artifacts (default: models/residual)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1000,
        help="Minimum rows per stat to train (default: 1000)",
    )
    parser.add_argument(
        "--acceptance-min-improvement",
        type=float,
        default=0.0,
        help="Minimum MAE improvement to accept model (default: 0.0)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Targets to train (default: PTS REB AST STL BLK TOV)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="CatBoost iterations (default: 1000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="CatBoost learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="CatBoost tree depth (default: 6)",
    )

    args = parser.parse_args()

    console = Console() if RICH_AVAILABLE else None

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        print(f"Error: Input file not found: {input_path}")
        print("Run the walk-forward residual builder first (Ticket 1).")
        return 1

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if console:
        console.print(Panel(
            f"[bold]Residual Model Training[/bold]\n"
            f"Input: {input_path}\n"
            f"Output: {output_dir}\n"
            f"Targets: {args.targets or 'PTS REB AST STL BLK TOV'}\n"
            f"Min rows: {args.min_rows}",
            title="Configuration",
        ))

    trainer = ResidualModelTrainer(
        min_rows=args.min_rows,
        acceptance_min_improvement=args.acceptance_min_improvement,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
    )

    try:
        result = trainer.train_all(
            residual_path=str(input_path),
            output_dir=str(output_dir),
            targets=args.targets,
        )
    except Exception as exc:
        logger.error("Residual training failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1

    _print_results_table(result, console=console)

    accepted = [s for s, r in result.targets.items() if r.status == "accepted"]
    rejected = [s for s, r in result.targets.items() if r.status == "rejected"]

    if accepted:
        print(f"\nAccepted: {', '.join(accepted)}")
    if rejected:
        print(f"Rejected: {', '.join(rejected)}")

    print(f"\nArtifacts saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
