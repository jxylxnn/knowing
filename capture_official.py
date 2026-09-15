#!/usr/bin/env python3
"""Capture an official CSV endpoint now, preserving raw bytes and receipt time."""

import argparse

from src.data.official_capture import REQUIRED, capture_official_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url")
    mode.add_argument("--native", choices=("schedule", "rosters"))
    parser.add_argument("--season")
    parser.add_argument("--team-id", type=int, action="append")
    parser.add_argument("--table", choices=tuple(REQUIRED))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    if args.native:
        if not args.season:
            parser.error("--season is required for native captures")
        from src.data.nba_capture import capture_nba_rosters, capture_nba_schedule

        if args.native == "schedule":
            result = capture_nba_schedule(args.data_dir, season=args.season)
        else:
            result = capture_nba_rosters(args.data_dir, season=args.season, team_ids=args.team_id)
    else:
        if not args.table:
            parser.error("--table is required with --url")
        result = capture_official_source(args.data_dir, url=args.url, filename=args.table)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
