#!/usr/bin/env python3
"""Build canonical point-in-time core tables from an immutable source snapshot."""

from __future__ import annotations

import argparse
import json

from src.data.canonicalize import canonicalize_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    report = canonicalize_snapshot(args.data_dir, args.snapshot_id)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.invalid_games == report.missing_player_team_game == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
