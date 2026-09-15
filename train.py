#!/usr/bin/env python3
"""Create an immutable NBA forecasting Model v2 candidate bundle."""

from __future__ import annotations

import argparse
import json

from src.training.v2_baseline import train_v2_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("v2",),
        default="v2",
        help="Supported forecast architecture (default: v2).",
    )
    parser.add_argument(
        "--preset",
        choices=("baseline",),
        default="baseline",
        help="Evidence-first candidate family (default: baseline).",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--config", default="config/model_v2.yaml")
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="Reuse an existing source snapshot instead of creating one.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional immutable candidate directory name.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        candidate = train_v2_baseline(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            config_path=args.config,
            snapshot_id=args.snapshot_id,
            run_id=args.run_id,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        payload = {
            "status": "failed",
            "architecture": "v2",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
        return 2

    payload = {
        "status": "candidate_created",
        "architecture": "v2",
        "candidate": str(candidate),
        "promoted": False,
    }
    print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
