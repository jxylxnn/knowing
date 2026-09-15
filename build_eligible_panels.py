#!/usr/bin/env python3
"""Build request-bound official eligible panels, retaining missing/DNP labels."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.eligible_panel import build_eligible_panel
from src.evaluation.chronology import digest_payload
from src.evaluation.request_replay import read_requests
from src.features.snapshot_inputs import read_snapshot_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--actuals-snapshot-id", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/eligible_panels")
    args = parser.parse_args()
    records = read_requests(args.requests)
    actuals = read_snapshot_csv(args.data_dir, args.actuals_snapshot_id,
                                ("player_game_eligibility.csv",))
    panels, quarantines, reports = [], [], []
    for record in records:
        panel, quarantine, report = build_eligible_panel(
            args.data_dir, ForecastRequest(**record["request"]), actuals
        )
        panels.append(panel)
        quarantines.append(quarantine)
        reports.append(report)
    identity = {"requests": records, "actuals_snapshot_id": args.actuals_snapshot_id,
                "actuals_hash": digest_payload(actuals.to_dict("records")), "reports": reports}
    destination = Path(args.output_dir) / digest_payload(identity)
    destination.mkdir(parents=True, exist_ok=False)
    pd.concat(panels, ignore_index=True).to_parquet(destination / "panels.parquet", index=False)
    pd.concat(quarantines, ignore_index=True).to_parquet(destination / "quarantine.parquet", index=False)
    (destination / "manifest.json").write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
