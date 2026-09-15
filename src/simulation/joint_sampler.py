"""Coherent Model v2 joint sampling from calibrated opportunity forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.minutes import allocate_team_minutes, capped_allocation


def sample_game(
    player_frame: pd.DataFrame, *, targets: tuple[str, ...], simulations: int,
    seed: int | None = None,
    overtime_probabilities: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Sample participation → minutes → conditional stat totals.

    The returned long frame is intentionally model-agnostic; rate means and
    standard deviations originate from calibrated component artifacts.
    """
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if player_frame["TEAM_ID"].isna().any():
        raise ValueError("TEAM_ID must be known")
    if player_frame["PLAYER_ID"].isna().any() or player_frame["PLAYER_ID"].duplicated().any():
        raise ValueError("Each player must have one known identity per game")
    # Validate probabilities and feasibility before consuming randomness.
    allocate_team_minutes(player_frame)
    for target in targets:
        for column in (f"{target}_RATE", f"{target}_RATE_STD"):
            values = pd.to_numeric(player_frame[column], errors="raise")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid count-rate parameter: {column}")
    uncertainty = pd.to_numeric(
        player_frame.get("MINUTES_STD", pd.Series(0.0, index=player_frame.index)),
        errors="raise",
    )
    if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
        raise ValueError("MINUTES_STD must be finite and nonnegative")
    active_probability = pd.to_numeric(
        player_frame.get("P_ACTIVE", pd.Series(1.0, index=player_frame.index)),
        errors="raise",
    )
    play_probability = player_frame["PLAY_PROB"].to_numpy(dtype=float)
    if (not np.isfinite(active_probability).all()
            or (active_probability > 1).any()
            or (active_probability.to_numpy() < play_probability).any()):
        raise ValueError("P_ACTIVE must be finite and between PLAY_PROB and 1")
    overtime = np.asarray(overtime_probabilities or (1.0,), dtype=float)
    if (overtime.ndim != 1 or not len(overtime) or not np.isfinite(overtime).all()
            or (overtime < 0).any() or not np.isclose(overtime.sum(), 1.0)):
        raise ValueError("Overtime probabilities must be a normalized discrete law")
    opportunity_seed, rate_seed, count_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(opportunity_seed)
    rate_rng = np.random.default_rng(rate_seed)
    count_rng = np.random.default_rng(count_seed)
    rows: list[dict] = []
    for simulation in range(simulations):
        sampled = player_frame.copy()
        for indices in sampled.groupby("TEAM_ID", sort=False).groups.values():
            sampled.loc[indices, "PLAY_PROB"] = sample_participants(
                sampled.loc[indices, "PLAY_PROB"].to_numpy(dtype=float), rng
            )
        # Complete the joint active/appearance law conditional on the feasible
        # appearance draw, preserving the supplied active marginal as well.
        inactive_play = np.divide(
            active_probability.to_numpy() - play_probability,
            1 - play_probability, out=np.zeros(len(sampled)),
            where=play_probability < 1,
        )
        sampled["ACTIVE"] = np.maximum(
            sampled["PLAY_PROB"].to_numpy(),
            rng.binomial(1, np.clip(inactive_play, 0, 1)),
        )
        raw = sampled["EXPECTED_MINUTES_RAW"].to_numpy(dtype=float)
        std = uncertainty.to_numpy(dtype=float)
        # Lognormal weights stay positive and carry minutes uncertainty into
        # the constrained allocation. Zero raw means use a tiny positive weight.
        mean = np.maximum(raw, 1e-8)
        sigma = np.sqrt(np.log1p((std / mean) ** 2))
        sampled["EXPECTED_MINUTES_RAW"] = rng.lognormal(
            np.log(mean) - sigma ** 2 / 2, sigma
        )
        sampled = allocate_team_minutes(sampled)
        periods = int(rng.choice(len(overtime), p=overtime))
        sampled["OVERTIME_MINUTES"] = 0.0
        if periods:
            for indices in sampled.groupby("TEAM_ID", sort=False).groups.values():
                group = sampled.loc[indices]
                allocation = capped_allocation(
                    group["EXPECTED_MINUTES"].to_numpy(),
                    group["PLAY_PROB"].to_numpy() * 5.0 * periods,
                    25.0 * periods,
                )
                sampled.loc[indices, "OVERTIME_MINUTES"] = allocation
        for _, player in sampled.iterrows():
            for target in targets:
                mean_rate = max(0.0, float(player.get(f"{target}_RATE", 0)))
                std_rate = max(0.0, float(player.get(f"{target}_RATE_STD", 0)))
                rate = mean_rate
                if mean_rate > 0 and std_rate > 0:
                    rate = rate_rng.gamma((mean_rate / std_rate) ** 2,
                                     std_rate ** 2 / mean_rate)
                full_minutes = player["EXPECTED_MINUTES"] + player["OVERTIME_MINUTES"]
                slope = float(player.get(f"{target}_MINUTES_RATE_SLOPE", 0.0))
                if not np.isfinite(slope):
                    raise ValueError("Minutes/rate dependence must be finite")
                center = float(player_frame.loc[player.name, "EXPECTED_MINUTES_RAW"])
                rate *= np.exp(np.clip(slope * (player["EXPECTED_MINUTES"] - center), -10, 10))
                total = int(count_rng.poisson(full_minutes * rate))
                rows.append({"SIMULATION": simulation, "PLAYER_ID": player["PLAYER_ID"],
                             "TEAM_ID": player["TEAM_ID"], "STAT": target,
                             "ACTIVE": int(player["ACTIVE"]),
                             "APPEARED": int(player["PLAY_PROB"]),
                             "MINUTES": player["EXPECTED_MINUTES"],
                             "OVERTIME_PERIODS": periods,
                             "OVERTIME_MINUTES": player["OVERTIME_MINUTES"],
                             "FULL_GAME_MINUTES": full_minutes,
                             "VALUE": max(0, total)})
    return pd.DataFrame(rows)


def sample_participants(probabilities, rng):
    """Dependent rounding preserves marginals and a feasible roster size.

    The joint participation law has floor/ceil(sum(p)) participants. This is
    explicitly dependent sampling, not rescued independent Bernoulli draws.
    """
    values = np.asarray(probabilities, dtype=float).copy()
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or values.sum() < 5):
        raise ValueError("Infeasible participation probabilities")
    while True:
        fractional = np.flatnonzero((values > 0) & (values < 1))
        if len(fractional) < 2:
            break
        i, j = fractional[:2]
        up = min(1 - values[i], values[j])
        down = min(values[i], 1 - values[j])
        if rng.random() < down / (up + down):
            values[i] += up
            values[j] -= up
        else:
            values[i] -= down
            values[j] += down
        values[[i, j]] = np.clip(values[[i, j]], 0, 1)
    if len(fractional):
        i = fractional[0]
        values[i] = float(rng.random() < values[i])
    return values.astype(int)
