#!/usr/bin/env python3
"""CLI entry point — monitor whether residual corrections are helping.

Reads a residual / prediction history parquet (or CSV) and produces a
monitoring report that compares base prediction error against corrected
prediction error for every stat (PTS, REB, AST, STL, BLK, TOV).

Outputs go to ``reports/residual_monitoring/``:

    latest_summary.json
    residual_report_<timestamp>.json
    residual_report_<timestamp>.csv

Example::

    python monitor_residual_corrections.py \
        --input data/evaluation/prediction_history.parquet \
        --output-dir reports/residual_monitoring
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.residual_monitor import (
    DEFAULT_CONFIDENCE_LABELS,
    DEFAULT_DATA_QUALITIES,
    DEFAULT_TARGETS,
    MonitoringThresholds,
    ResidualMonitor,
)
from src.evaluation.residual_report import (
    render_console_summary,
    write_report,
)
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("monitor_residual_corrections")


DEFAULT_CONFIG_PATH = "config/default.yaml"


def _load_config_block(config_path: Optional[str]) -> Dict[str, Any]:
    """Best-effort load of the ``residual_monitoring`` config block.

    Returns ``{}`` when the config is missing or malformed so the CLI keeps
    working with built-in defaults.
    """
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        logger.debug("Config file not found at %s; using CLI defaults", path)
        return {}

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not available; skipping config load")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read config %s: %s", path, exc)
        return {}

    block = data.get("residual_monitoring", {}) if isinstance(data, dict) else {}
    return block if isinstance(block, dict) else {}


def _resolve_input_path(
    input_arg: Optional[str],
    config_block: Dict[str, Any],
) -> Optional[str]:
    """Pick the input parquet/CSV path.

    Strict mode: if the user passes ``--input`` explicitly, that exact file
    must exist — no silent fallback to config defaults.  Only when
    ``--input`` is omitted do we walk the config candidate list
    (``default_input`` → ``fallback_input``).
    """
    if input_arg:
        path = Path(input_arg)
        if not path.exists():
            raise FileNotFoundError(
                f"--input file not found: {input_arg}"
            )
        return str(path)

    candidates: List[str] = []
    default = config_block.get("default_input")
    if default:
        candidates.append(str(default))
    fallback = config_block.get("fallback_input")
    if fallback:
        candidates.append(str(fallback))

    for path_str in candidates:
        path = Path(path_str)
        if path.exists():
            return str(path)
    return candidates[0] if candidates else None


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Monitor whether residual correction models are improving "
            "predictions across PTS, REB, AST, STL, BLK, TOV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python monitor_residual_corrections.py\n"
            "  python monitor_residual_corrections.py --min-rows 200\n"
            "  python monitor_residual_corrections.py \\\n"
            "      --input data/evaluation/prediction_history.parquet \\\n"
            "      --output-dir reports/residual_monitoring\n"
        ),
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to a parquet/CSV file with prediction history. "
            "Defaults to data/evaluation/prediction_history.parquet, "
            "falling back to data/evaluation/residual_training.parquet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. When omitted, the value is taken from "
            "config residual_monitoring.output_dir, falling back to "
            "reports/residual_monitoring."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to YAML config (default: config/default.yaml). "
            "Used to pick up residual_monitoring defaults."
        ),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Minimum rows required for a stat to receive a status label.",
    )
    parser.add_argument(
        "--helping-threshold",
        type=float,
        default=None,
        help="Min MAE improvement %% for HELPING status (positive).",
    )
    parser.add_argument(
        "--hurting-threshold",
        type=float,
        default=None,
        help="MAE improvement %% below which the stat is HURTING (negative).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="Rolling window sizes in days (e.g. 7 14 30).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Target stats to evaluate (default: PTS REB AST STL BLK TOV).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Force JSON output (default: on).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write a per-target CSV report in addition to JSON.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip the CSV output even if enabled in config.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a console summary of the report.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def _build_thresholds(
    args: argparse.Namespace,
    config_block: Dict[str, Any],
) -> MonitoringThresholds:
    """Merge CLI args with config defaults into a :class:`MonitoringThresholds`."""
    min_rows = args.min_rows if args.min_rows is not None else int(
        config_block.get("min_rows", 500)
    )
    helping = args.helping_threshold if args.helping_threshold is not None else float(
        config_block.get("helping_threshold_pct", 1.0)
    )
    hurting = args.hurting_threshold if args.hurting_threshold is not None else float(
        config_block.get("hurting_threshold_pct", -1.0)
    )
    neutral = float(config_block.get("neutral_band_pct", 1.0))
    min_window_rows = int(config_block.get("min_window_rows", 50))
    return MonitoringThresholds(
        min_rows=min_rows,
        helping_threshold_pct=helping,
        hurting_threshold_pct=hurting,
        neutral_band_pct=neutral,
        min_window_rows=min_window_rows,
    )


def _resolve_targets(
    args: argparse.Namespace,
    config_block: Dict[str, Any],
) -> Tuple[str, ...]:
    """Resolve target list from CLI / config / defaults."""
    if args.targets:
        return tuple(str(t).upper() for t in args.targets)
    cfg_targets = config_block.get("targets")
    if cfg_targets:
        return tuple(str(t).upper() for t in cfg_targets)
    return DEFAULT_TARGETS


def _resolve_windows(
    args: argparse.Namespace,
    config_block: Dict[str, Any],
) -> Tuple[int, ...]:
    """Resolve rolling window sizes from CLI / config / defaults."""
    if args.windows:
        return tuple(int(w) for w in args.windows)
    cfg_windows = config_block.get("windows_days")
    if cfg_windows:
        return tuple(int(w) for w in cfg_windows)
    return (7, 14, 30)


DEFAULT_OUTPUT_DIR = "reports/residual_monitoring"


def _resolve_output_dir(
    args: argparse.Namespace,
    config_block: Dict[str, Any],
) -> str:
    """Resolve the report output directory.

    Resolution order:
        1. ``--output-dir`` CLI flag
        2. ``config_block['output_dir']``
        3. :data:`DEFAULT_OUTPUT_DIR` hardcoded fallback
    """
    if args.output_dir:
        return str(args.output_dir)
    cfg = config_block.get("output_dir")
    if cfg:
        return str(cfg)
    return DEFAULT_OUTPUT_DIR


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    config_block = _load_config_block(args.config)

    try:
        input_path = _resolve_input_path(args.input, config_block)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if not input_path or not Path(input_path).exists():
        logger.error(
            "No usable input file found. Tried config defaults: %s",
            config_block.get("default_input"),
        )
        return 2

    thresholds = _build_thresholds(args, config_block)
    targets = _resolve_targets(args, config_block)
    windows = _resolve_windows(args, config_block)
    output_dir = _resolve_output_dir(args, config_block)

    write_csv = bool(args.csv) or (
        not args.no_csv
        and bool(config_block.get("write_csv", True))
    )

    logger.info("Loading prediction history from %s", input_path)
    history = ResidualMonitor.load_input(input_path)

    monitor = ResidualMonitor(
        targets=targets,
        windows=windows,
        thresholds=thresholds,
        data_qualities=DEFAULT_DATA_QUALITIES,
        confidence_labels=DEFAULT_CONFIDENCE_LABELS,
    )

    report = monitor.evaluate(history, input_path=input_path)

    written = write_report(
        report,
        output_dir=output_dir,
        write_csv=write_csv,
    )

    logger.info("Wrote report artifacts to %s:", output_dir)
    for key, path in written.items():
        logger.info("  %s: %s", key, path)

    if args.print_summary:
        print(render_console_summary(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
