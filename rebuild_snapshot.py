#!/usr/bin/env python3
"""Create a new verified snapshot without changing its base archive."""

import argparse

from src.data.rebuild_snapshot import rebuild_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-snapshot-id", required=True)
    parser.add_argument("--import-v1", action="store_true",
                        help="Explicitly verify/import a v1 archive into a new v2 snapshot")
    parser.add_argument("--capture", action="append", default=[])
    parser.add_argument("--historical-schedule", action="append", default=[],
                        help="Archive a historical schedule for outcome reconciliation")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    manifest = rebuild_snapshot(args.data_dir, base_snapshot_id=args.base_snapshot_id,
                                captures=args.capture, import_v1=args.import_v1,
                                historical_schedules=args.historical_schedule)
    print(manifest.snapshot_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
