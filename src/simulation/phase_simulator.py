"""Phase-based game simulation engine.

Extracted from GameSimulator to isolate the Monte Carlo game loop.
Handles sampling game environments, role states, and running phase-by-phase
simulation for each sim iteration.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from src.simulation.sim_types import (
    GameEnvironment,
    PhaseDefinition,
    PlayerProjection,
    RoleSample,
    TeamContext,
)
from src.simulation.archetype import ArchetypeEngine
from src.simulation.role_sampler import RoleSampler

logger = logging.getLogger(__name__)

STAT_NAMES = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']


class PhaseSimulator:
    """Runs the phase-by-phase Monte Carlo simulation for a single matchup."""

    def __init__(self, config: Any = None):
        self.archetype_engine = ArchetypeEngine()
        self.role_sampler = RoleSampler()
        self.overtime_margin_threshold = 3.0
        if config and hasattr(config, 'simulation_params'):
            self.overtime_margin_threshold = getattr(
                config.simulation_params, 'overtime_margin_threshold', 3.0
            )

    # ------------------------------------------------------------------
    # Phase definitions
    # ------------------------------------------------------------------
    @staticmethod
    def phase_definitions() -> List[PhaseDefinition]:
        """Standard phase schedule for a single game."""
        return [
            PhaseDefinition('first_half', 24.0),
            PhaseDefinition('second_half', 18.0),
            PhaseDefinition('clutch', 6.0, clutch_window=True),
            PhaseDefinition('overtime', 5.0, overtime=True),
        ]

    # ------------------------------------------------------------------
    # Game environment sampling
    # ------------------------------------------------------------------
    def sample_game_environment(
        self,
        np_rng: np.random.Generator,
        betting_lines: dict,
        team_targets: Dict[str, Dict[str, float]],
        team_a: str,
        team_b: str,
        team_a_eff: dict,
        team_b_eff: dict,
        rest_a: dict,
        rest_b: dict,
    ) -> GameEnvironment:
        """Sample shared game-environment uncertainty for a single run."""
        model_total = float(
            team_targets.get(team_a, {}).get('pts', 110.0)
            + team_targets.get(team_b, {}).get('pts', 108.0)
        )
        vegas_total = betting_lines.get('total')
        vegas_spread = betting_lines.get('spread')

        if vegas_total is not None and float(vegas_total) > 0:
            total_anchor = 0.55 * float(vegas_total) + 0.45 * model_total
        else:
            total_anchor = model_total

        if vegas_spread is not None:
            margin_anchor = -float(vegas_spread)
        else:
            margin_anchor = float(
                team_targets.get(team_a, {}).get('pts', 110.0)
                - team_targets.get(team_b, {}).get('pts', 108.0)
            )

        rest_penalty = 0.0
        if rest_a.get('is_b2b'):
            rest_penalty += 0.5
        if rest_b.get('is_b2b'):
            rest_penalty += 0.5

        pace_anchor = float(np.mean([
            float(team_a_eff.get('pace', 100.0)),
            float(team_b_eff.get('pace', 100.0)),
        ]))

        pace_shock = float(np.clip(
            np_rng.normal(
                1.0 + (total_anchor / max(model_total, 1.0) - 1.0) * 0.12
                - rest_penalty * 0.01,
                0.04,
            ),
            0.88, 1.14,
        ))
        total_shock = float(np.clip(np_rng.normal(1.0, 0.055), 0.85, 1.16))
        margin_draw = float(np_rng.normal(margin_anchor, 8.5))
        close_factor = float(np.clip(np.exp(-abs(margin_draw) / 7.0), 0.0, 1.0))
        blowout_factor = float(np.clip(
            max(0.0, (abs(margin_draw) - 10.0) / 20.0), 0.0, 1.0
        ))
        game_total = float(total_anchor * total_shock)

        return GameEnvironment(
            pace_anchor=pace_anchor,
            pace_shock=pace_shock,
            total_anchor=total_anchor,
            total_shock=total_shock,
            margin_draw=margin_draw,
            close_factor=close_factor,
            blowout_factor=blowout_factor,
            game_total=game_total,
        )

    # ------------------------------------------------------------------
    # Possession sampling
    # ------------------------------------------------------------------
    @staticmethod
    def sample_phase_possessions(
        np_rng: np.random.Generator,
        team_pace: float,
        phase: PhaseDefinition,
        game_env: GameEnvironment,
        score_diff: float,
    ) -> int:
        """Sample offensive possessions for a team in a specific phase."""
        base_possessions = team_pace * phase.minutes / 48.0 * game_env.pace_shock
        if phase.clutch_window:
            if abs(score_diff) <= 5:
                base_possessions *= 1.04 + 0.05 * game_env.close_factor
            else:
                base_possessions *= 0.94
        if phase.overtime:
            base_possessions = team_pace * 5.0 / 48.0 * 1.08
        if abs(score_diff) >= 15 and not phase.overtime:
            base_possessions *= 0.95
        if abs(score_diff) <= 8:
            base_possessions *= 1.02
        sampled = float(np_rng.normal(base_possessions, max(1.2, base_possessions * 0.05)))
        return int(np.clip(round(sampled), 1, 70))

    # ------------------------------------------------------------------
    # Pool allocation
    # ------------------------------------------------------------------
    @staticmethod
    def allocate_pool(np_rng: np.random.Generator, total: int, weights: np.ndarray) -> np.ndarray:
        """Allocate an integer pool to players using a multinomial draw."""
        total = int(max(0, total))
        if total == 0:
            return np.zeros(len(weights), dtype=int)
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 1e-8, None)
        probs = weights / weights.sum()
        return np_rng.multinomial(total, probs)

    # ------------------------------------------------------------------
    # Team building helpers
    # ------------------------------------------------------------------
    def build_team_lineup_context(
        self,
        roster: List[dict],
        lineup_data: Optional[dict],
        coach_tightness: float,
    ) -> TeamContext:
        """Build soft priors for lineup interaction and coach behavior."""
        import numpy as np  # local import avoids issues at module level

        starter_names = {
            str(name).strip()
            for name in (lineup_data or {}).get('starters', [])
            if name
        }
        if not starter_names:
            starter_names = {str(p.get('name', '')).strip() for p in roster if p.get('is_starter', False)}

        if not roster:
            return TeamContext(
                coach_tightness=float(np.clip(coach_tightness, 0.0, 1.0))
            )

        primary = max(roster, key=lambda p: float(p.get('usage', 0.0)))
        rebounder = max(roster, key=lambda p: float(p.get('mean_reb', 0.0)))
        rim_protector = max(roster, key=lambda p: float(p.get('mean_blk', 0.0)))
        bench_starters = sum(
            1
            for p in roster
            if str(p.get('name', '')).strip() in starter_names
            and not bool(p.get('is_starter', False))
        )
        starter_overlap = sum(
            1
            for p in roster
            if str(p.get('name', '')).strip() in starter_names
            and bool(p.get('is_starter', False))
        ) / max(len(starter_names), 1)

        primary_out = str(primary.get('name', '')).strip() not in starter_names
        rebounder_out = str(rebounder.get('name', '')).strip() not in starter_names
        rim_out = str(rim_protector.get('name', '')).strip() not in starter_names

        ct = float(np.clip(coach_tightness, 0.0, 1.0))
        usage_boost = 1.0 + (0.09 if primary_out else 0.0) + 0.03 * bench_starters
        assist_boost = 1.0 + (0.10 if primary_out else 0.0) + 0.03 * ct
        rebound_boost = 1.0 + (0.10 if rebounder_out else 0.0)
        efficiency_boost = 1.0 + (0.03 if primary_out else 0.0) + 0.02 * ct
        opp_efficiency_penalty = 1.0 + (0.06 if rim_out else 0.0)
        paint_boost = 1.0 + (0.05 if rim_out else 0.0)
        shot_volume = 1.0 + (0.03 if ct >= 0.6 else -0.01)
        ft_rate = 1.0 + (0.03 if ct >= 0.55 else 0.0)
        turnover_pressure = 1.0 + (0.03 if primary_out else 0.0)
        closing_bonus = 1.0 + 0.06 * ct

        return TeamContext(
            usage_boost=float(np.clip(usage_boost, 0.9, 1.25)),
            assist_boost=float(np.clip(assist_boost, 0.9, 1.25)),
            rebound_boost=float(np.clip(rebound_boost, 0.9, 1.25)),
            efficiency_boost=float(np.clip(efficiency_boost, 0.9, 1.18)),
            opp_efficiency_penalty=float(np.clip(opp_efficiency_penalty, 0.9, 1.18)),
            paint_boost=float(np.clip(paint_boost, 0.9, 1.20)),
            shot_volume=float(np.clip(shot_volume, 0.92, 1.10)),
            ft_rate=float(np.clip(ft_rate, 0.92, 1.10)),
            turnover_pressure=float(np.clip(turnover_pressure, 0.92, 1.12)),
            closing_bonus=float(np.clip(closing_bonus, 1.0, 1.18)),
            starter_overlap=float(np.clip(starter_overlap, 0.0, 1.0)),
            coach_tightness=float(np.clip(ct, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Single-team phase simulation
    # ------------------------------------------------------------------
    def simulate_team_phase(
        self,
        np_rng: np.random.Generator,
        roster: List[dict],
        team_name: str,
        is_home: bool,
        phase: PhaseDefinition,
        current_score_diff: float,
        team_context: TeamContext,
        opponent_context: TeamContext,
        game_env: GameEnvironment,
        player_totals: Dict[str, Dict[str, Any]],
    ) -> int:
        """Simulate one team across one game phase.

        Returns the team's points scored in this phase.
        """
        if not roster:
            return 0

        team_ctx = team_context.to_dict() if isinstance(team_context, TeamContext) else team_context
        opp_ctx = opponent_context.to_dict() if isinstance(opponent_context, TeamContext) else opponent_context

        phase_possessions = self.sample_phase_possessions(
            np_rng,
            float(team_ctx.get('pace', 100.0)),
            phase,
            game_env,
            current_score_diff,
        )
        phase_minutes_total = phase.minutes * 5.0
        close_game = phase.clutch_window and abs(current_score_diff) <= 5
        blowout = abs(current_score_diff) >= (20 if phase.minutes <= 24 else 15)

        minute_weights = []
        usage_weights = []
        shot_weights = []
        assist_weights = []
        rebound_weights = []
        tov_weights = []
        stl_weights = []
        blk_weights = []
        profiles = []
        role_states_list = []

        for player in roster:
            totals = player_totals[player['name']]
            archetype = player.get('archetype', 'balanced')
            profile = player.get('archetype_profile') or self.archetype_engine.get_profile(archetype)
            role_state = player.get('role_state')
            if role_state is None:
                role_state = RoleSample('normal', 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.10, 1.0)

            base_minutes = float(player.get('exp_min', 20.0)) * role_state.minute_multiplier

            if close_game:
                if player.get('is_starter', False) or role_state.state == 'closer':
                    base_minutes *= role_state.close_game_multiplier * team_ctx.get('closing_bonus', 1.0)
                else:
                    base_minutes *= 0.88
            elif blowout:
                if player.get('is_starter', False):
                    base_minutes *= role_state.blowout_multiplier * 0.82
                else:
                    base_minutes *= 1.12

            if totals.get('fouls', 0) >= 5:
                base_minutes *= 0.50
            elif totals.get('fouls', 0) >= 4:
                base_minutes *= 0.82

            if float(player.get('play_probability', 1.0)) < 0.85:
                base_minutes *= 0.88

            minutes_weight = max(0.0, base_minutes) * role_state.volatility
            usage_weight = (
                float(player.get('usage', 0.15))
                * role_state.usage_multiplier
                * profile['usage_bias']
                * team_ctx['usage_boost']
            )
            if close_game:
                usage_weight *= role_state.close_game_multiplier
            elif blowout:
                usage_weight *= role_state.blowout_multiplier

            shot_weight = minutes_weight * usage_weight * profile['shot_bias'] * team_ctx['shot_volume']
            assist_weight = minutes_weight * role_state.assist_multiplier * profile['assist_bias'] * team_ctx['assist_boost']
            rebound_weight = minutes_weight * role_state.rebound_multiplier * profile['rebound_bias'] * team_ctx['rebound_boost']
            tov_weight = minutes_weight * role_state.turnover_multiplier * profile['turnover_bias'] * team_ctx['turnover_pressure']
            stl_weight = minutes_weight * profile['defense_bias'] * opp_ctx.get('turnover_pressure', 1.0)
            blk_weight = minutes_weight * profile['rim_bias'] * opp_ctx.get('paint_boost', 1.0)

            minute_weights.append(minutes_weight)
            usage_weights.append(usage_weight)
            shot_weights.append(shot_weight)
            assist_weights.append(assist_weight)
            rebound_weights.append(rebound_weight)
            tov_weights.append(tov_weight)
            stl_weights.append(stl_weight)
            blk_weights.append(blk_weight)
            profiles.append(profile)
            role_states_list.append(role_state)

        minute_weights_arr = np.asarray(minute_weights, dtype=float)
        if minute_weights_arr.sum() <= 0:
            minute_weights_arr = np.ones(len(roster), dtype=float)
        minute_share = minute_weights_arr / minute_weights_arr.sum()
        phase_minutes = minute_share * phase_minutes_total

        shot_weights_arr = np.asarray(shot_weights, dtype=float)
        assist_weights_arr = np.asarray(assist_weights, dtype=float)
        rebound_weights_arr = np.asarray(rebound_weights, dtype=float)
        tov_weights_arr = np.asarray(tov_weights, dtype=float)
        stl_weights_arr = np.asarray(stl_weights, dtype=float)
        blk_weights_arr = np.asarray(blk_weights, dtype=float)

        shot_pool = int(np.clip(round(
            phase_possessions * (0.88 + 0.04 * team_ctx['efficiency_boost'] + 0.04 * game_env.close_factor)
        ), 1, 60))
        turnover_pool = int(np.clip(round(
            phase_possessions * (0.10 + 0.02 * team_ctx['turnover_pressure'] + 0.02 * game_env.blowout_factor)
        ), 0, 18))
        fta_pool = int(np.clip(round(
            phase_possessions * (0.18 + 0.03 * team_ctx['ft_rate'] + 0.02 * game_env.close_factor)
        ), 0, 22))
        stl_pool = int(np.clip(round(
            phase_possessions * (0.03 + 0.01 * opp_ctx.get('turnover_pressure', 1.0))
        ), 0, 10))
        blk_pool = int(np.clip(round(
            phase_possessions * (0.02 + 0.01 * opp_ctx.get('opp_efficiency_penalty', 1.0))
        ), 0, 8))

        shot_alloc = self.allocate_pool(np_rng, shot_pool, shot_weights_arr)
        fta_alloc = self.allocate_pool(
            np_rng, fta_pool,
            minute_weights_arr * np.array(
                [p['ft_pct'] for p in profiles], dtype=float
            ),
        )
        tov_alloc = self.allocate_pool(np_rng, turnover_pool, tov_weights_arr)
        stl_alloc = self.allocate_pool(np_rng, stl_pool, stl_weights_arr)
        blk_alloc = self.allocate_pool(np_rng, blk_pool, blk_weights_arr)

        made_fg_total = 0
        player_make_data: List[Dict[str, Any]] = []

        for idx, player in enumerate(roster):
            profile = profiles[idx]
            role_state = role_states_list[idx]
            totals = player_totals[player['name']]
            minutes = float(phase_minutes[idx])
            totals['minutes'] += minutes
            totals['played'] = True

            base_three_rate = profile['three_rate']
            three_rate = float(np.clip(
                base_three_rate * (0.92 + 0.10 * team_ctx['usage_boost'])
                * (1.06 if close_game else 0.94 if blowout else 1.0),
                0.02, 0.72,
            ))
            fg2_pct = float(np.clip(
                profile['fg2_pct'] * team_ctx['efficiency_boost']
                * opp_ctx.get('paint_boost', 1.0)
                * role_state.efficiency_multiplier,
                0.25, 0.78,
            ))
            fg3_pct = float(np.clip(
                profile['fg3_pct'] * team_ctx['efficiency_boost']
                * opp_ctx.get('three_defense', 1.0)
                * role_state.efficiency_multiplier,
                0.18, 0.60,
            ))
            ft_pct = float(np.clip(
                profile['ft_pct'] * team_ctx['ft_rate'], 0.45, 0.94,
            ))

            fga = int(shot_alloc[idx])
            fg3a = int(np_rng.binomial(fga, three_rate)) if fga > 0 else 0
            fg2a = max(0, fga - fg3a)
            fg3m = int(np_rng.binomial(fg3a, fg3_pct)) if fg3a > 0 else 0
            fg2m = int(np_rng.binomial(fg2a, fg2_pct)) if fg2a > 0 else 0
            fta = int(fta_alloc[idx])
            ftm = int(np_rng.binomial(fta, ft_pct)) if fta > 0 else 0
            points = 3 * fg3m + 2 * fg2m + ftm

            zero_inflation = float(profile['zero_inflation'])
            if role_state.state in {'limited', 'bench'} and minutes < 14 and np_rng.random() < zero_inflation:
                points = min(points, int(np_rng.poisson(1.2)))
                fg2m = min(fg2m, points // 2)
                fg3m = min(fg3m, points // 3)

            totals['pts'] += points
            totals['tov'] += int(tov_alloc[idx])
            totals['stl'] += int(stl_alloc[idx])
            totals['blk'] += int(blk_alloc[idx])

            made_fg = fg2m + fg3m
            made_fg_total += made_fg
            player_make_data.append({
                'name': player['name'],
                'made_fg': made_fg,
                'missed_fg': max(0, fga - made_fg),
                'assist_weight': float(assist_weights_arr[idx]),
                'rebound_weight': float(rebound_weights_arr[idx]),
            })

            foul_mean = max(0.0, minutes / 11.5 * (0.78 + 0.18 * role_state.volatility + 0.08 * float(player.get('usage', 0.15))))
            if blowout and player.get('is_starter', False):
                foul_mean *= 0.88
            if close_game and (player.get('is_starter', False) or role_state.state == 'closer'):
                foul_mean *= 1.06
            foul_draw = int(np_rng.poisson(foul_mean))
            totals['fouls'] = min(6, totals['fouls'] + foul_draw)

        assist_pool = int(np.clip(round(
            max(0, made_fg_total) * (0.58 + 0.05 * team_ctx['assist_boost'] + 0.04 * game_env.close_factor)
        ), 0, 18))
        rebound_pool = int(np.clip(round(
            (shot_pool - made_fg_total) * (0.90 * team_ctx['rebound_boost']) + fta_pool * 0.20
        ), 0, 25))

        assist_alloc = self.allocate_pool(np_rng, assist_pool, assist_weights_arr)
        rebound_alloc = self.allocate_pool(np_rng, rebound_pool, rebound_weights_arr)

        for idx, player in enumerate(roster):
            totals = player_totals[player['name']]
            totals['ast'] += int(assist_alloc[idx])
            totals['reb'] += int(rebound_alloc[idx])

        team_points = int(sum(player_totals[p['name']]['pts'] for p in roster))
        return team_points