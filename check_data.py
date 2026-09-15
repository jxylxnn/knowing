#!/usr/bin/env python3
"""Validate a source snapshot and report canonical core-data coverage."""

from __future__ import annotations

import argparse
import json

from src.data.coverage import build_coverage_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    report = build_coverage_report(args.data_dir, args.snapshot_id)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if (report.missing_player_team_game == 0
                 and report.core_reconciled_fraction >= 0.999) else 2


if __name__ == "__main__":
    raise SystemExit(main())
