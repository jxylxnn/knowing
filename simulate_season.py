#!/usr/bin/env python3
"""Run opportunity-first Model v2 forecasts and joint simulations."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd

from src.features.snapshot_inputs import load_snapshot_schedule
from src.simulation.v2_runner import run_scheduled_game
from src.operations.forecast_export import export_forecast_run


_BUNDLE_ID_SENTINELS = frozenset({"nan", "none", "<na>"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--today", action="store_true")
    modes.add_argument("--date")
    modes.add_argument("--week", action="store_true")
    modes.add_argument("--season", action="store_true", help="Next 30 days")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--horizon", default="morning", choices=(
        "previous_night", "morning", "pregame_90m", "pregame_30m"
    ))
    parser.add_argument("--schedule-file", default=None,
                        help="Optional exact copy of the captured snapshot schedule")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--candidate",
        default=None,
        help=(
            "Exact sealed v2 bundle directory for non-published candidate "
            "execution; the run is a shadow attempt and never publishes to "
            "the official ledger."
        ),
    )
    parser.add_argument("--output-dir", default="data/sim_results/v2")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Allow appearance-derived rosters; output is never written to the official ledger.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dates = _dates(args)
    mode = _execution_mode(args)
    official = mode == "official"
    try:
        schedule = _schedule(args, dates)
        forecasts = []
        samples = []
        for offset, (_, game) in enumerate(schedule.iterrows()):
            forecast, simulated = run_scheduled_game(
                game,
                source_snapshot_id=args.snapshot_id,
                data_dir=args.data_dir,
                models_dir=args.models_dir,
                horizon=args.horizon,
                simulations=args.sims,
                seed=args.seed + offset,
                strict=not args.allow_degraded,
                persist=not args.allow_degraded,
                candidate_dir=args.candidate,
            )
            forecasts.append(forecast)
            simulated = simulated.copy()
            simulated["GAME_ID"] = str(game["GAME_ID"])
            samples.append(simulated)
        forecast_frame = pd.concat(forecasts, ignore_index=True)
        samples_frame = pd.concat(samples, ignore_index=True)
        model_bundle_id = _single_model_bundle_id(forecast_frame)
        forecast_path, samples_path = export_forecast_run(
            args.output_dir, forecast_frame, samples_frame,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({
            "status": "failed",
            "architecture": "v2",
            "mode": mode,
            "candidate": str(args.candidate) if args.candidate else None,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }, indent=2 if args.json else None, sort_keys=True))
        return 3

    payload = {
        "status": "complete",
        "architecture": "v2",
        "mode": mode,
        "official": official,
        "published_to_official_ledger": official,
        "model_bundle_id": model_bundle_id,
        "candidate": str(args.candidate) if args.candidate else None,
        "games": len(schedule),
        "forecast_path": str(forecast_path),
        "samples_path": str(samples_path),
    }
    print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
    return 0


def _execution_mode(args: argparse.Namespace) -> str:
    """Classify one CLI run as official publication, shadow, or degraded."""

    if args.allow_degraded:
        return "degraded"
    if args.candidate:
        return "shadow"
    return "official"


def _single_model_bundle_id(forecast: pd.DataFrame) -> str:
    """Return the one nonempty bundle identity shared by every forecast row.

    Every row must carry a real bundle identity: a null, blank, or
    missing-value sentinel anywhere in the column is a contract failure even
    when the remaining rows agree on one value.
    """

    if "MODEL_BUNDLE_ID" not in forecast.columns:
        raise ValueError("Forecast output is missing MODEL_BUNDLE_ID")
    column = forecast["MODEL_BUNDLE_ID"]
    if column.isna().any():
        raise ValueError(
            "Forecast output must carry exactly one nonempty MODEL_BUNDLE_ID; "
            "found a missing value"
        )
    normalized = [value.strip() for value in column.astype(str)]
    invalid = sorted({
        value
        for value in normalized
        if not value or value.lower() in _BUNDLE_ID_SENTINELS
    })
    if invalid:
        raise ValueError(
            "Forecast output must carry exactly one nonempty MODEL_BUNDLE_ID; "
            f"found invalid value(s) {invalid}"
        )
    unique = set(normalized)
    if len(unique) != 1:
        raise ValueError(
            "Forecast output must carry exactly one nonempty MODEL_BUNDLE_ID; "
            f"found {', '.join(sorted(unique))}"
        )
    return unique.pop()


def _dates(args: argparse.Namespace) -> list[str]:
    start = date.today() if args.today or args.week or args.season else pd.Timestamp(args.date).date()
    days = 7 if args.week else 30 if args.season else 1
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days)]


def _schedule(args: argparse.Namespace, dates: list[str]) -> pd.DataFrame:
    frame = load_snapshot_schedule(args.data_dir, args.snapshot_id)
    if args.schedule_file:
        supplied = pd.read_csv(args.schedule_file, dtype=str)
        try:
            pd.testing.assert_frame_equal(supplied, frame, check_dtype=False)
        except AssertionError as exc:
            raise ValueError("--schedule-file must match the named snapshot schedule") from exc
    game_dates = pd.to_datetime(frame["GAME_DATE"], errors="raise").dt.date.astype(str)
    frame = frame.loc[game_dates.isin(dates)].copy()
    if frame.empty:
        raise ValueError("No scheduled games found")
    required = {"GAME_ID", "GAME_DATE", "SCHEDULED_TIP", "HOME_TEAM_ID", "AWAY_TEAM_ID"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Schedule missing Model v2 columns: {sorted(missing)}")
    if frame["SCHEDULED_TIP"].isna().any():
        raise ValueError("Schedule contains games without a timezone-aware tipoff")
    return frame.reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
