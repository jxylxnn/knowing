#!/usr/bin/env python3
"""Query an immutable Model v2 player-stat forecast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.operations.ledger import OfficialForecastLedger
from src.query.v2_probability import probability_at_line, select_forecast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", "-p", required=True, help="Player ID or exact name")
    parser.add_argument(
        "--stat", "-s", required=True,
        choices=("pts", "reb", "ast", "stl", "blk", "tov"),
    )
    parser.add_argument("--line", "-l", type=float, default=None)
    parser.add_argument("--date", "-d", default=None)
    parser.add_argument("--forecast-file", default=None)
    parser.add_argument("--ledger-dir", default="data/ledger")
    parser.add_argument("--players-file", default="data/nba_players.csv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    forecasts = (
        pd.read_parquet(args.forecast_file)
        if args.forecast_file
        else OfficialForecastLedger(args.ledger_dir).load_forecasts()
    )
    names_path = Path(args.players_file)
    names = pd.read_csv(names_path) if names_path.is_file() else None
    try:
        row = select_forecast(
            forecasts,
            player=args.player,
            stat=args.stat,
            game_date=args.date,
            player_names=names,
        )
    except ValueError as exc:
        print(json.dumps({"status": "not_found", "error": str(exc)}))
        return 3

    payload = {
        "status": "ok",
        "request_id": str(row["REQUEST_ID"]),
        "bundle_id": str(row["MODEL_BUNDLE_ID"]),
        "snapshot_id": str(row["SOURCE_SNAPSHOT_ID"]),
        "player_id": str(row["PLAYER_ID"]),
        "stat": str(row["STAT"]),
        "mean": float(row["MEAN"]),
        "p10": float(row["P10"]),
        "p50": float(row["P50"]),
        "p90": float(row["P90"]),
        "calibration_version": str(row["CALIBRATION_VERSION"]),
        "data_quality": str(row["DATA_QUALITY"]),
    }
    if args.line is not None:
        try:
            probabilities = probability_at_line(row, args.line)
        except ValueError as exc:
            payload.update({"status": "invalid_query", "error": str(exc)})
            print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
            return 2
        payload["line"] = args.line
        payload.update(probabilities)
    print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
