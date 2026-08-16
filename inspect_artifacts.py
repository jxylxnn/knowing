#!/usr/bin/env python3
"""Read-only report of the currently deployed runtime artifacts.

Reports what the artifact files in a models directory actually contain —
training preset, Transformer state, feature groups, blend weight version and
optimizer provenance — without loading inference models and without mutating
any files. When ``champion.json`` exists, the champion version directory is
inspected; otherwise the supplied directory is inspected directly.

Usage:
    python inspect_artifacts.py --models-dir models
    python inspect_artifacts.py --models-dir models --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.contracts.runtime_status import RuntimeStatusError, inspect_runtime_status

# (human label, status key) pairs in display order.
_ORDERED_FIELDS = [
    ("Models root", "models_root"),
    ("Active models dir", "active_models_dir"),
    ("Champion version", "champion_version"),
    ("Training preset", "training_preset"),
    ("Transformer enabled", "transformer_enabled"),
    ("Transformer artifact present", "transformer_artifact_present"),
    ("MAE companion targets", "mae_companion_targets"),
    ("Model count", "model_count"),
    ("Feature groups", "feature_groups"),
    ("Feature group count", "feature_group_count"),
    ("Training blend method", "training_blend_method"),
    ("Weight version", "weight_version"),
    ("Weight description", "weight_description"),
    ("Optimizer method", "optimizer_method"),
    ("Backtest score", "backtest_score"),
    ("Backtest date range", "backtest_date_range"),
    ("Optimizer-produced current weights", "optimizer_produced_current_weights"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report deployed runtime artifact status (read-only)."
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="models root to inspect (default: models)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit stable machine-readable JSON instead of the human report",
    )
    return parser.parse_args()


def _print_human_report(status: dict) -> None:
    width = max(len(label) for label, _ in _ORDERED_FIELDS) + 2
    for label, key in _ORDERED_FIELDS:
        print(f"{label:<{width}}{status[key]}")
    warnings = status.get("warnings") or []
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


def main() -> int:
    args = parse_args()
    try:
        status = inspect_runtime_status(Path(args.models_dir))
    except RuntimeStatusError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        _print_human_report(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
