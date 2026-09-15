#!/usr/bin/env python3
"""Execute small sealed baseline trials on verified four-role request folds."""

import argparse

from src.evaluation.chronology import EvaluationPolicy
from src.evaluation.request_replay import read_requests
from src.training.fold_campaign import run_fold_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--actuals-snapshot-id", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="data/campaigns")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_fold_campaign(
        read_requests(args.requests), data_dir=args.data_dir, models_dir=args.models_dir,
        output_dir=args.output_dir, actuals_snapshot_id=args.actuals_snapshot_id,
        policy=EvaluationPolicy(simulations=args.simulations, seed=args.seed),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
