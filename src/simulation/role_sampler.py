"""NBA Player Role Sampler for Monte Carlo Simulation.

Samples a role state (limited, normal, expanded, starter, bench, closer)
for each player before a simulation run, with archetype-aware adjustments.
"""
import numpy as np
from typing import Dict

from src.simulation.sim_types import RoleSample


# Probability distributions for role states
_STARTER_STATES = ['limited', 'normal', 'expanded', 'starter', 'closer']
_STARTER_PROBS = np.array([0.08, 0.32, 0.18, 0.26, 0.16], dtype=float)

_BENCH_STATES = ['limited', 'normal', 'expanded', 'bench', 'closer']
_BENCH_PROBS = np.array([0.18, 0.34, 0.18, 0.20, 0.10], dtype=float)

# Pre-defined state profiles (state_name, minute_mult, usage_mult, eff_mult, ast_mult, reb_mult, tov_mult, close_mult, blowout_mult, zero_infl, volatility)
_STATE_PROFILES: Dict[str, RoleSample] = {
    'limited': RoleSample('limited', 0.68, 0.84, 0.95, 0.82, 0.90, 0.92, 0.78, 1.10, 0.24, 1.16),
    'normal': RoleSample('normal', 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.10, 1.00),
    'expanded': RoleSample('expanded', 1.12, 1.10, 1.04, 1.10, 0.95, 1.03, 1.08, 0.96, 0.08, 1.08),
    'starter': RoleSample('starter', 1.16, 1.12, 1.03, 1.08, 0.96, 1.02, 1.12, 0.92, 0.06, 1.04),
    'bench': RoleSample('bench', 0.90, 0.96, 1.02, 0.92, 1.06, 1.00, 0.92, 1.06, 0.18, 1.22),
    'closer': RoleSample('closer', 1.10, 1.16, 1.08, 1.14, 0.92, 1.04, 1.24, 0.88, 0.05, 1.10),
    'non-closer': RoleSample('non-closer', 0.86, 0.90, 0.96, 0.82, 1.04, 0.96, 0.82, 1.08, 0.14, 1.12),
}


class RoleSampler:
    """Samples role states for players before each simulation run."""

    def sample(
        self,
        player: dict,
        np_rng: np.random.Generator,
        coach_tightness: float,
        close_game_prob: float,
        archetype: str = 'balanced',
    ) -> RoleSample:
        """Sample a role state for a player.

        Args:
            player: Dict with 'is_starter' and 'play_probability' keys.
            np_rng: NumPy random generator for this simulation.
            coach_tightness: 0-1 scale of coach rotation rigidity.
            close_game_prob: Probability the game will be close.
            archetype: Player archetype string for adjustments.

        Returns:
            RoleSample with multipliers for this run.
        """
        is_starter = bool(player.get('is_starter', False))

        if is_starter:
            state_names = _STARTER_STATES
            base_probs = _STARTER_PROBS.copy()
        else:
            state_names = _BENCH_STATES
            base_probs = _BENCH_PROBS.copy()

        self._adjust_probs_for_context(base_probs, is_starter, coach_tightness,
                                        close_game_prob, player)

        base_probs = np.clip(base_probs, 0.01, None)
        base_probs /= base_probs.sum()

        state = str(np_rng.choice(state_names, p=base_probs))
        sampled = _STATE_PROFILES[state]

        # Apply archetype-specific adjustments
        sampled = self._apply_archetype_adjustments(sampled, state, archetype)

        return sampled

    def _adjust_probs_for_context(
        self,
        base_probs: np.ndarray,
        is_starter: bool,
        coach_tightness: float,
        close_game_prob: float,
        player: dict,
    ) -> None:
        """Mutate base_probs in-place based on context."""
        tightness = float(np.clip(coach_tightness, 0.0, 1.0))
        close_prob = float(np.clip(close_game_prob, 0.0, 1.0))

        if tightness >= 0.6:
            base_probs[-2:] *= 1.20
            base_probs[0] *= 0.85
        else:
            base_probs[1:4] *= 1.08

        if close_prob >= 0.55:
            base_probs[-1] *= 1.45
            if is_starter:
                base_probs[-2] *= 1.20

        if float(player.get('play_probability', 1.0)) < 0.85:
            base_probs[0] *= 1.25
            base_probs[2] *= 0.85

    def _apply_archetype_adjustments(
        self,
        sampled: RoleSample,
        state: str,
        archetype: str,
    ) -> RoleSample:
        """Apply archetype-specific tweaks to the sampled role state."""
        if archetype == 'microwave_bench_scorer' and state in {'bench', 'expanded'}:
            return RoleSample(
                sampled.state,
                sampled.minute_multiplier * 1.03,
                sampled.usage_multiplier * 1.08,
                sampled.efficiency_multiplier * 1.04,
                sampled.assist_multiplier,
                sampled.rebound_multiplier,
                sampled.turnover_multiplier,
                sampled.close_game_multiplier * 1.04,
                sampled.blowout_multiplier,
                sampled.zero_inflation * 0.92,
                sampled.volatility * 1.10,
            )
        elif archetype == 'rebound_first_center' and state in {'starter', 'expanded'}:
            return RoleSample(
                sampled.state,
                sampled.minute_multiplier * 1.04,
                sampled.usage_multiplier * 0.96,
                sampled.efficiency_multiplier * 1.03,
                sampled.assist_multiplier,
                sampled.rebound_multiplier * 1.10,
                sampled.turnover_multiplier,
                sampled.close_game_multiplier,
                sampled.blowout_multiplier,
                sampled.zero_inflation * 0.90,
                sampled.volatility,
            )
        return sampled