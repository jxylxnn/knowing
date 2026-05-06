"""NBA Player Archetype Engine.

Infers coarse player archetypes from projection shape and position,
and provides volatility/style priors for each archetype.
"""
from typing import Dict


# Define the archetype profiles as a module-level constant
ARCHETYPE_PROFILES: Dict[str, Dict[str, float]] = {
    'heliocentric_star_guard': {
        'three_rate': 0.39, 'fg2_pct': 0.49, 'fg3_pct': 0.37, 'ft_pct': 0.87,
        'usage_bias': 1.22, 'assist_bias': 1.32, 'rebound_bias': 0.82,
        'turnover_bias': 1.18, 'shot_bias': 1.14, 'defense_bias': 0.92,
        'rim_bias': 0.88, 'paint_bias': 0.94, 'zero_inflation': 0.02,
        'volatility': 1.22, 'clutch_bonus': 1.12, 'blowout_penalty': 0.86,
    },
    'low_usage_3_and_d_wing': {
        'three_rate': 0.56, 'fg2_pct': 0.53, 'fg3_pct': 0.40, 'ft_pct': 0.80,
        'usage_bias': 0.88, 'assist_bias': 0.72, 'rebound_bias': 0.92,
        'turnover_bias': 0.72, 'shot_bias': 0.88, 'defense_bias': 1.12,
        'rim_bias': 0.82, 'paint_bias': 0.88, 'zero_inflation': 0.14,
        'volatility': 0.82, 'clutch_bonus': 1.02, 'blowout_penalty': 0.92,
    },
    'rebound_first_center': {
        'three_rate': 0.03, 'fg2_pct': 0.63, 'fg3_pct': 0.28, 'ft_pct': 0.68,
        'usage_bias': 0.96, 'assist_bias': 0.74, 'rebound_bias': 1.36,
        'turnover_bias': 0.88, 'shot_bias': 0.90, 'defense_bias': 1.15,
        'rim_bias': 1.24, 'paint_bias': 1.10, 'zero_inflation': 0.05,
        'volatility': 0.88, 'clutch_bonus': 1.05, 'blowout_penalty': 0.90,
    },
    'microwave_bench_scorer': {
        'three_rate': 0.45, 'fg2_pct': 0.47, 'fg3_pct': 0.38, 'ft_pct': 0.84,
        'usage_bias': 1.08, 'assist_bias': 0.82, 'rebound_bias': 0.78,
        'turnover_bias': 1.00, 'shot_bias': 1.16, 'defense_bias': 0.88,
        'rim_bias': 0.96, 'paint_bias': 0.90, 'zero_inflation': 0.20,
        'volatility': 1.34, 'clutch_bonus': 1.08, 'blowout_penalty': 1.02,
    },
    'secondary_creator_forward': {
        'three_rate': 0.33, 'fg2_pct': 0.52, 'fg3_pct': 0.36, 'ft_pct': 0.78,
        'usage_bias': 1.06, 'assist_bias': 1.14, 'rebound_bias': 1.04,
        'turnover_bias': 0.98, 'shot_bias': 1.04, 'defense_bias': 1.00,
        'rim_bias': 1.00, 'paint_bias': 0.98, 'zero_inflation': 0.06,
        'volatility': 1.02, 'clutch_bonus': 1.10, 'blowout_penalty': 0.94,
    },
    'balanced': {
        'three_rate': 0.37, 'fg2_pct': 0.50, 'fg3_pct': 0.36, 'ft_pct': 0.77,
        'usage_bias': 1.00, 'assist_bias': 1.00, 'rebound_bias': 1.00,
        'turnover_bias': 1.00, 'shot_bias': 1.00, 'defense_bias': 1.00,
        'rim_bias': 1.00, 'paint_bias': 1.00, 'zero_inflation': 0.08,
        'volatility': 1.00, 'clutch_bonus': 1.00, 'blowout_penalty': 1.00,
    },
}


class ArchetypeEngine:
    """Infers NBA player archetypes for simulation priors."""

    def infer(self, player: dict) -> str:
        """Infer a coarse archetype from projection shape and position.

        Args:
            player: Dict with keys 'usage', 'mean_pts', 'mean_reb', 'mean_ast',
                    'mean_stl', 'mean_blk', 'position'.

        Returns:
            One of: 'heliocentric_star_guard', 'rebound_first_center',
            'low_usage_3_and_d_wing', 'secondary_creator_forward',
            'microwave_bench_scorer', 'balanced'.
        """
        usage = float(player.get('usage', 0.15))
        pts = float(player.get('mean_pts', 0.0))
        reb = float(player.get('mean_reb', 0.0))
        ast = float(player.get('mean_ast', 0.0))
        stl_blk = float(player.get('mean_stl', 0.0)) + float(player.get('mean_blk', 0.0))
        position = str(player.get('position', 'SF')).upper()

        if usage >= 0.26 and ast >= 4.5:
            return 'heliocentric_star_guard'
        if reb >= 8.0 and position in {'C', 'PF'}:
            return 'rebound_first_center'
        if usage <= 0.17 and pts >= 8.0 and stl_blk >= 1.0:
            return 'low_usage_3_and_d_wing'
        if pts >= 10.0 and ast >= 3.8 and usage >= 0.18:
            return 'secondary_creator_forward'
        if usage <= 0.18 and pts >= 10.0:
            return 'microwave_bench_scorer'
        return 'balanced'

    def get_profile(self, archetype: str) -> Dict[str, float]:
        """Return volatility and style priors for a player archetype.

        Args:
            archetype: String archetype name (from infer()).

        Returns:
            Dict of bias/multiplier floats for simulation.
        """
        return ARCHETYPE_PROFILES.get(archetype, ARCHETYPE_PROFILES['balanced']).copy()