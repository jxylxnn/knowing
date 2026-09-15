#!/usr/bin/env python3
"""Score immutable Model v2 forecast rows against final player actuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.replay import score_replay
from src.evaluation.request_replay import read_requests, replay_requests
from src.operations.forecast_export import export_forecast_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/replay_v2.yaml")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--predictions", default=None)
    inputs.add_argument("--requests", help="Verified historical request manifest")
    parser.add_argument("--candidate", help="Exact sealed v2 bundle for request replay")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--replay-dir", default="data/replays")
    parser.add_argument("--actuals", default="data/nba_players.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if config.get("architecture") != "v2":
        raise ValueError("Replay requires architecture: v2")
    if args.requests:
        if not args.candidate:
            parser.error("--candidate is required with --requests")
        replayed = replay_requests(
            read_requests(args.requests), bundle_dir=args.candidate, data_dir=args.data_dir
        )
        export_forecast_run(args.replay_dir, replayed.forecasts, replayed.samples)
        predictions = replayed.forecasts
    else:
        prediction_path = args.predictions or config.get("predictions")
        if not prediction_path:
            parser.error("Supply --requests or --predictions")
        predictions = pd.read_parquet(prediction_path)
    actuals = pd.read_csv(args.actuals, dtype={"GAME_ID": str, "PLAYER_ID": str})
    result = score_replay(predictions, actuals).to_dict()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))
    return 0 if result["reconciled_fraction"] == 1.0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
