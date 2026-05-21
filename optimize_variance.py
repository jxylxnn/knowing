#!/usr/bin/env python3
"""Variance optimization CLI — tune volatility multipliers via CRPS.

The existing ``optimize_weights.py`` tunes the *mean* (MAE/RMSE).
This script tunes the *variance* (Std multipliers) using the
Continuous Ranked Probability Score (CRPS), which rewards forecast
distributions that are both sharp and well-calibrated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.evaluation.metrics import calculate_empirical_crps

logger = logging.getLogger(__name__)

# Context multipliers to tune: [B2B, Rookie, Blowout, Home, Away, Playoff, RestAdvantage]
CONTEXT_LABELS = ["b2b", "rookie", "blowout", "home", "away", "playoff", "rest_advantage"]
N_CONTEXT = len(CONTEXT_LABELS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize variance multipliers via CRPS",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        type=str,
        default=None,
        help="Start date for backtest window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=str,
        default=None,
        help="End date for backtest window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=30,
        help="Number of recent days to use (default: 30)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="PTS",
        help="Target stat to optimize variance for (default: PTS)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1000,
        help="Monte Carlo iterations per evaluation (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print current multipliers and exit without optimizing",
    )
    return parser.parse_args()


def load_backtest_data(
    data_dir: str,
    date_from: str,
    date_to: str,
    target: str,
) -> List[Dict]:
    """Load historical player-game data for CRPS evaluation.

    Returns a list of dicts with keys: mean, std, actual, context_label.
    """
    players_path = f"{data_dir}/nba_players.csv"
    if not __import__("os").path.exists(players_path):
        logger.error("Player data not found at %s", players_path)
        return []

    df = pd.read_csv(players_path, parse_dates=["GAME_DATE"])
    if date_from:
        df = df[df["GAME_DATE"] >= date_from]
    if date_to:
        df = df[df["GAME_DATE"] <= date_to]

    if df.empty:
        logger.warning("No data in date range %s — %s", date_from, date_to)
        return []

    records = []
    for _, row in df.iterrows():
        mean = row.get(f"ROLL_{target}_AVG_10", row.get(target, 0))
        std = max(row.get(f"ROLL_{target}_STD_10", mean * 0.3), 0.1)
        actual = row.get(target, 0)

        # Infer context label from available columns
        context = "normal"
        if row.get("SCHED_IS_B2B_SECOND", 0) == 1:
            context = "b2b"
        elif row.get("IS_PLAYOFF_GAME", 0) == 1:
            context = "playoff"
        elif row.get("IS_TANKING_PROXY", 0) == 1:
            context = "blowout"

        records.append({
            "mean": mean,
            "std": std,
            "actual": actual,
            "context": context,
        })

    logger.info(
        "Loaded %d records for CRPS evaluation (target=%s)",
        len(records), target,
    )
    return records


def _get_context_multiplier(context: str, multipliers: np.ndarray) -> float:
    """Look up the variance multiplier for a given context."""
    context_map = {
        "b2b": 0, "rookie": 1, "blowout": 2,
        "home": 3, "away": 4, "playoff": 5, "rest_advantage": 6,
    }
    idx = context_map.get(context, -1)
    if idx >= 0 and idx < len(multipliers):
        return multipliers[idx]
    return 1.0


def crps_objective(
    multipliers: np.ndarray,
    backtest_data: List[Dict],
    target: str,
    num_sims: int,
) -> float:
    """Objective function: mean CRPS across all backtest records.

    Args:
        multipliers: Array of N_CONTEXT volatility multipliers.
        backtest_data: List of dicts with mean, std, actual, context.
        target: Target stat name.
        num_sims: Monte Carlo iterations per evaluation.

    Returns:
        Mean CRPS (lower = better).
    """
    rng = np.random.default_rng(42)
    crps_values = []

    default_std_mult = np.mean(multipliers)

    for rec in backtest_data:
        ctx_mult = _get_context_multiplier(rec["context"], multipliers)
        adj_std = rec["std"] * ctx_mult

        # Generate normal draws
        sims = rng.normal(rec["mean"], max(adj_std, 0.1), size=num_sims)
        sims = np.maximum(sims, 0.0)

        crps = calculate_empirical_crps(sims, rec["actual"])
        if np.isfinite(crps):
            crps_values.append(crps)

    if not crps_values:
        return 999.0

    return float(np.mean(crps_values))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    data = load_backtest_data(args.data_dir, args.date_from, args.date_to, args.target)
    if not data:
        logger.error("No backtest data loaded. Exiting.")
        sys.exit(1)

    # Default multipliers (all 1.0 = no adjustment)
    default_mult = np.ones(N_CONTEXT)

    if args.dry_run:
        print(f"\nCurrent variance multipliers for '{args.target}':")
        for label, val in zip(CONTEXT_LABELS, default_mult):
            print(f"  {label:20s}: {val:.3f}")
        print(f"\nMean CRPS: {crps_objective(default_mult, data, args.target, args.sims):.4f}")
        return

    print(f"\nOptimizing variance multipliers for '{args.target}'...")
    print(f"  Records: {len(data)}")
    print(f"  Sims/eval: {args.sims}")
    print()

    result = minimize(
        crps_objective,
        default_mult,
        args=(data, args.target, args.sims),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 0.01, "fatol": 0.001},
        bounds=[(0.5, 2.0)] * N_CONTEXT,
    )

    print(f"\nOptimization {'converged' if result.success else 'FAILED'} "
          f"({result.message})")
    print(f"  Final CRPS: {result.fun:.4f}")
    print(f"  Iterations: {result.nit}")
    print()

    print("Optimal variance multipliers:")
    for label, val in zip(CONTEXT_LABELS, result.x):
        print(f"  {label:20s}: {val:.3f}")

    print()
    baseline_crps = crps_objective(default_mult, data, args.target, args.sims)
    print(f"  Baseline CRPS (all 1.0): {baseline_crps:.4f}")
    print(f"  Optimized CRPS:           {result.fun:.4f}")
    print(f"  Improvement:              {(baseline_crps - result.fun) / baseline_crps * 100:.1f}%")


if __name__ == "__main__":
    main()
