"""Shared simulation data types.

Typed dataclasses replace raw dicts throughout the simulation pipeline,
making contracts explicit and reducing key-typo bugs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RoleSample:
    """Sampled role state for a player in a given simulation run."""
    state: str
    minute_multiplier: float
    usage_multiplier: float
    efficiency_multiplier: float
    assist_multiplier: float
    rebound_multiplier: float
    turnover_multiplier: float
    close_game_multiplier: float
    blowout_multiplier: float
    zero_inflation: float
    volatility: float


@dataclass
class PhaseDefinition:
    """A phase of game flow used by the reactive simulator."""
    name: str
    minutes: float
    clutch_window: bool = False
    overtime: bool = False


@dataclass
class PlayerProjection:
    """Projection for one player — replaces the untyped dict."""
    id: int
    name: str
    usage: float
    exp_min: float
    play_probability: float
    position: str
    is_starter: bool
    min_std: float = 5.0

    # Stat means / stds (populated by _build_player_projection)
    mean_pts: float = 0.0
    std_pts: float = 1.0
    mean_reb: float = 0.0
    std_reb: float = 0.5
    mean_ast: float = 0.0
    std_ast: float = 0.5
    mean_stl: float = 0.0
    std_stl: float = 0.3
    mean_blk: float = 0.0
    std_blk: float = 0.3
    mean_tov: float = 0.0
    std_tov: float = 0.3

    # Populated during simulation run
    archetype: str = 'balanced'
    archetype_profile: Optional[Dict[str, float]] = None
    role_state: Optional[RoleSample] = None

    def to_dict(self) -> dict:
        """Convert back to the legacy dict format for backward compatibility."""
        return {
            'id': self.id,
            'name': self.name,
            'usage': self.usage,
            'exp_min': self.exp_min,
            'play_probability': self.play_probability,
            'position': self.position,
            'is_starter': self.is_starter,
            'min_std': self.min_std,
            'mean_pts': self.mean_pts,
            'std_pts': self.std_pts,
            'mean_reb': self.mean_reb,
            'std_reb': self.std_reb,
            'mean_ast': self.mean_ast,
            'std_ast': self.std_ast,
            'mean_stl': self.mean_stl,
            'std_stl': self.std_stl,
            'mean_blk': self.mean_blk,
            'std_blk': self.std_blk,
            'mean_tov': self.mean_tov,
            'std_tov': self.std_tov,
            'archetype': self.archetype,
            'archetype_profile': self.archetype_profile,
            'role_state': self.role_state,
        }


@dataclass
class TeamContext:
    """Team-level context assembled before simulation.

    Replaces the raw dict returned by _build_team_lineup_context.
    """
    usage_boost: float = 1.0
    assist_boost: float = 1.0
    rebound_boost: float = 1.0
    efficiency_boost: float = 1.0
    opp_efficiency_penalty: float = 1.0
    paint_boost: float = 1.0
    shot_volume: float = 1.0
    ft_rate: float = 1.0
    turnover_pressure: float = 1.0
    closing_bonus: float = 1.0
    starter_overlap: float = 0.0
    coach_tightness: float = 0.5
    # Added during reactive simulation
    pace: float = 100.0
    off_env: float = 1.0
    three_defense: float = 1.0

    def to_dict(self) -> dict:
        return {
            'usage_boost': self.usage_boost,
            'assist_boost': self.assist_boost,
            'rebound_boost': self.rebound_boost,
            'efficiency_boost': self.efficiency_boost,
            'opp_efficiency_penalty': self.opp_efficiency_penalty,
            'paint_boost': self.paint_boost,
            'shot_volume': self.shot_volume,
            'ft_rate': self.ft_rate,
            'turnover_pressure': self.turnover_pressure,
            'closing_bonus': self.closing_bonus,
            'starter_overlap': self.starter_overlap,
            'coach_tightness': self.coach_tightness,
            'pace': self.pace,
            'off_env': self.off_env,
            'three_defense': self.three_defense,
        }


@dataclass
class GameEnvironment:
    """Sampled game environment uncertainty for a single sim run."""
    pace_anchor: float
    pace_shock: float
    total_anchor: float
    total_shock: float
    margin_draw: float
    close_factor: float
    blowout_factor: float
    game_total: float

    def to_dict(self) -> dict:
        return {
            'pace_anchor': self.pace_anchor,
            'pace_shock': self.pace_shock,
            'total_anchor': self.total_anchor,
            'total_shock': self.total_shock,
            'margin_draw': self.margin_draw,
            'close_factor': self.close_factor,
            'blowout_factor': self.blowout_factor,
            'game_total': self.game_total,
        }


@dataclass
class SimResult:
    """Structured result from simulate_matchup.

    Provides typed access to the raw dict output, and a method to
    convert back to the legacy dict format.
    """
    team_a: str
    team_b: str
    win_prob_a: float
    team_summaries: Dict[str, dict]
    simulations: List[dict]
    player_averages: List[dict]
    betting_lines: dict
    lineup_a: dict
    lineup_b: dict
    metadata: dict
    context: dict

    def to_dict(self) -> dict:
        return {
            'team_a': self.team_a,
            'team_b': self.team_b,
            'win_prob_a': self.win_prob_a,
            'team_summaries': self.team_summaries,
            'simulations': self.simulations,
            'player_averages': self.player_averages,
            'betting_lines': self.betting_lines,
            'lineup_a': self.lineup_a,
            'lineup_b': self.lineup_b,
            'metadata': self.metadata,
            'context': self.context,
        }