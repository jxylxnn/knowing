#!/usr/bin/env python3
"""Attach final actuals to immutable Model v2 forecast evidence."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.operations.ledger import OfficialForecastLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--actuals", default="data/nba_players.csv")
    parser.add_argument("--ledger-dir", default="data/ledger")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    actuals = pd.read_csv(args.actuals)
    path = OfficialForecastLedger(args.ledger_dir).reconcile(
        actuals, game_date=args.date
    )
    payload = {"status": "reconciled", "path": str(path)}
    print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
