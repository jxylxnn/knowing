import logging
import sys
from typing import Optional, List
from datetime import datetime

from src.query.probability_calculator import ProbabilityCalculator, ProbabilityResult
from src.query.projection_loader import ProjectionLoader, PlayerProjection
from src.query.query_parser import QueryParser, ParsedQuery, QueryType

logger = logging.getLogger(__name__)


class InteractiveCLI:
    BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║           NBA Player Stat Over/Under Probability Query           ║
╠══════════════════════════════════════════════════════════════════╣
║  Type a question like:                                           ║
║    • LeBron over 25.5 points vs Boston?                          ║
║    • Jokic under 12.5 rebounds tonight?                          ║
║    • What's Curry's projection?                                  ║
║                                                                  ║
║  Commands: help, players, teams, reload, clear, exit             ║
╚══════════════════════════════════════════════════════════════════╝
"""

    HELP_TEXT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY FORMATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Over/Under Probability:
    <player> over <line> <stat> [vs <opponent>]?
    <player> under <line> <stat> [vs <opponent>]?
    
    Examples:
      • LeBron over 25.5 pts
      • Jokic under 12.5 rebounds vs DEN
      • Curry o 30.5 points tonight?

  Get Projection:
    <player> projection [stat]
    What's <player>'s <stat> projection?
    
    Examples:
      • Giannis projection
      • What's Durant's points projection?

  Compare Players:
    compare <player> and <player> [stat]
    
    Examples:
      • compare LeBron and Curry points
      • compare Jokic and Embiid rebounds

  Change Settings:
    sims <number> - Change number of simulations (default: 100)
    
    Examples:
      • sims 500
      • sims 1000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATS: pts/points, reb/rebounds, ast/assists, stl/steals, blk/blocks, tov/turnovers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def __init__(
        self,
        data_dir: str = 'data/sim_results',
        model_dir: str = 'models',
        num_sims: int = 100
    ):
        self.calculator = ProbabilityCalculator()
        self.loader = ProjectionLoader(data_dir=data_dir)
        self.parser = QueryParser()
        self.model_dir = model_dir
        self.num_sims = num_sims
        self._simulator = None
        self._running = True
    
    def run(self):
        self._print_banner()
        
        projections = self.loader.load_projections()
        if projections.empty:
            print("\n⚠ No cached projections found.")
            print("  Run 'python simulate_season.py --today' first to generate projections.")
            print("  Or I can run a live simulation when you query a player.\n")
        else:
            cache_info = self.loader.get_cache_info()
            print(f"\n✓ Loaded {cache_info['num_projections']} projections")
            if cache_info.get('cache_file'):
                print(f"  from: {cache_info['cache_file'].split('/')[-1]}")
            print()
        
        while self._running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                self._handle_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                self._running = False
                break
            except EOFError:
                print("\n\nGoodbye!")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                print(f"\n❌ Error: {e}")
    
    def _print_banner(self):
        print(self.BANNER)
    
    def _handle_input(self, user_input: str):
        parsed = self.parser.parse(user_input)
        
        if parsed.query_type == QueryType.EXIT:
            print("\nGoodbye!")
            self._running = False
            return
        
        if parsed.query_type == QueryType.HELP:
            print(self.HELP_TEXT)
            return
        
        if parsed.query_type == QueryType.LIST_PLAYERS:
            self._list_players()
            return
        
        if user_input.lower() in ('teams', 'list teams'):
            self._list_teams()
            return
        
        if user_input.lower() in ('reload', 'refresh'):
            self._reload()
            return
        
        if user_input.lower() in ('clear', 'cls'):
            self.parser.clear_context()
            print("Context cleared.")
            return
        
        if user_input.lower() == 'info':
            self._show_info()
            return
        
        if user_input.lower().startswith('sims '):
            try:
                new_sims = int(user_input.split()[1])
                if new_sims < 1:
                    print("Number of simulations must be at least 1.")
                    return
                self.num_sims = new_sims
                print(f"Simulations set to {new_sims}")
                return
            except (ValueError, IndexError):
                print("Usage: sims <number> (e.g., sims 500)")
                return
        
        if parsed.query_type == QueryType.OVER_UNDER:
            self._handle_over_under(parsed)
            return
        
        if parsed.query_type == QueryType.PROJECTION:
            self._handle_projection(parsed)
            return
        
        if parsed.query_type == QueryType.COMPARE:
            self._handle_compare(parsed)
            return
        
        if parsed.query_type == QueryType.UNKNOWN:
            print("\n❓ I didn't understand that query.")
            print("  Type 'help' for usage examples.\n")
    
    def _handle_over_under(self, parsed: ParsedQuery):
        player_name = parsed.player_name
        stat = parsed.stat or 'pts'
        line = parsed.line
        direction = parsed.direction
        opponent = parsed.opponent
        
        projection = self.loader.find_player(
            player_name=player_name,
            opponent=opponent
        )
        
        if projection is None:
            print(f"\n🔍 Player '{player_name}' not found in cached projections.")
            
            should_sim = self._prompt_live_simulation(player_name)
            if should_sim:
                projection = self._run_live_simulation(player_name, opponent)
            
            if projection is None:
                return
        
        if line is None:
            mean_val = projection.get_stat_mean(stat)
            line = round(mean_val - 0.5) + 0.5
            print(f"\n📊 No line specified. Using implied line: {line}")
        
        context = self.loader.get_player_context(
            player_name=projection.player_name,
            opponent=projection.opponent,
            stat=stat
        )
        
        base_mean = projection.get_stat_mean(stat)
        adjusted_mean = base_mean
        adjustments = []
        
        recent_avg = context.get('recent_avg', {})
        matchup_avg = context.get('matchup_avg', {})
        opponent_defense = context.get('opponent_defense', {})
        trend = context.get('trend', {})
        
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        recent_stat = safe_float(recent_avg.get(stat), base_mean)
        matchup_stat = safe_float(matchup_avg.get(stat), base_mean)
        
        archetype_info = self._infer_player_archetype(recent_avg, base_mean)
        position = archetype_info['position']
        archetype = archetype_info['archetype']
        archetype_params = self._get_archetype_params(archetype)
        
        adjustments.append(("Archetype", f"{archetype.replace('_', ' ').title()} ({position})"))
        
        recent_weight = safe_float(archetype_params.get('recent_weight'), 0.35)
        trend_mod = safe_float(archetype_params.get('trend_modifier'), 1.0)
        std_mult = safe_float(archetype_params.get('std_multiplier'), 1.0)
        
        pct_change_val = safe_float(trend.get('pct_change'), 0)
        if trend.get('direction') == 'up' and pct_change_val > 25:
            recent_weight = min(0.55, recent_weight + 0.10)
        elif trend.get('direction') == 'down' and pct_change_val < -25:
            recent_weight = max(0.20, recent_weight - 0.10)
        
        blended = (base_mean * (1 - recent_weight)) + (recent_stat * recent_weight)
        if abs(blended - base_mean) > 0.1:
            adjustments.append(("Recent Form Blend", round(blended - base_mean, 2)))
        
        exp_min = safe_float(recent_avg.get('min'), 30)
        baseline_min = 32.0
        min_factor = exp_min / baseline_min
        if abs(min_factor - 1.0) > 0.05:
            scaled = blended * min_factor
            adjustments.append(("Minutes Scale", round(scaled - blended, 2)))
        else:
            scaled = blended
        
        defense_factor = self._calculate_position_defense_factor(
            opponent_defense, position, stat, archetype_params
        )
        if abs(defense_factor - 1.0) > 0.01:
            scaled = scaled * defense_factor
            pct_change = round((defense_factor - 1.0) * 100, 1)
            if pct_change > 0:
                adjustments.append(("Position Defense", f"+{pct_change}%"))
            else:
                adjustments.append(("Position Defense", f"{pct_change}%"))
        
        trend_adj = self._calculate_archetype_trend_adjustment(
            trend, scaled, archetype_params
        )
        if abs(trend_adj) > 0.1:
            scaled += trend_adj
            adjustments.append(("Trend", round(trend_adj, 2)))
        
        if matchup_stat > 0 and base_mean > 0:
            matchup_adj = self._calculate_archetype_matchup_adjustment(
                matchup_stat, base_mean, scaled, archetype_params
            )
            if abs(matchup_adj) > 0.1:
                scaled += matchup_adj
                adjustments.append(("Matchup History", round(matchup_adj, 2)))
        
        home_adj = safe_float(archetype_params.get('home_adj'), 0.5) if stat == 'pts' else 0.3
        away_adj = safe_float(archetype_params.get('away_adj'), -0.3) if stat == 'pts' else -0.2
        
        if projection.is_home:
            scaled += home_adj
            adjustments.append(("Home Court", home_adj))
        else:
            scaled += away_adj
            adjustments.append(("Away Game", away_adj))
        
        adjusted_mean = scaled
        
        result = self.calculator.run_monte_carlo_simulation(
            player_name=projection.player_name,
            stat=stat,
            line=line,
            mean=adjusted_mean,
            std=projection.get_stat_std(stat),
            ci_low=projection.get_stat_ci(stat)[0],
            ci_high=projection.get_stat_ci(stat)[1],
            opponent=projection.opponent,
            date=projection.date,
            play_probability=projection.play_probability,
            num_sims=self.num_sims,
            team=projection.team,
            is_home=projection.is_home,
            recent_games=context.get('recent_games'),
            recent_avg=context.get('recent_avg'),
            matchup_history=context.get('matchup_history'),
            matchup_avg=context.get('matchup_avg'),
            opponent_defense=context.get('opponent_defense'),
            trend=context.get('trend'),
            base_mean=base_mean,
            adjustments=adjustments
        )
        
        print("\n" + self.calculator.format_detailed_result(result))
    
    def _infer_player_archetype(self, recent_avg: dict, base_mean: float) -> dict:
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        pts = safe_float(recent_avg.get('pts'), base_mean if base_mean > 0 else 10)
        reb = safe_float(recent_avg.get('reb'), 4)
        ast = safe_float(recent_avg.get('ast'), 2)
        stl = safe_float(recent_avg.get('stl'), 0.5)
        blk = safe_float(recent_avg.get('blk'), 0.3)
        min_val = safe_float(recent_avg.get('min'), 25)
        
        if reb > 7:
            if blk > 1.5:
                position = 'C'
            elif reb > 9:
                position = 'C'
            else:
                position = 'PF'
        elif ast > 5:
            if ast > 8:
                position = 'PG'
            else:
                position = 'SG'
        elif pts > 15 and reb < 5:
            if ast > 3:
                position = 'SG'
            else:
                position = 'SF'
        elif reb > 4 and pts < 14:
            position = 'PF'
        else:
            position = 'SF'
        
        if pts >= 25:
            if ast >= 6:
                archetype = 'elite_playmaker'
            elif reb >= 8:
                archetype = 'dominant_big'
            else:
                archetype = 'primary_scorer'
        elif pts >= 20:
            if ast >= 5:
                archetype = 'floor_general'
            elif reb >= 7:
                archetype = 'interior_scorer'
            elif stl >= 1.2:
                archetype = 'two_way_wing'
            else:
                archetype = 'secondary_scorer'
        elif pts >= 15:
            if ast >= 4:
                archetype = 'playmaker'
            elif reb >= 6:
                archetype = 'stretch_big'
            elif blk >= 1.0 or stl >= 1.0:
                archetype = 'defensive_anchor'
            else:
                archetype = 'three_level_scorer'
        elif pts >= 10:
            if reb >= 6:
                archetype = 'rim_protector'
            elif ast >= 3:
                archetype = 'connector'
            elif stl >= 0.8:
                archetype = 'three_and_d'
            else:
                archetype = 'role_player'
        else:
            if reb >= 5:
                archetype = 'rim_protector'
            elif ast >= 2:
                archetype = 'facilitator'
            else:
                archetype = 'role_player'
        
        return {
            'position': position,
            'archetype': archetype
        }
    
    def _get_archetype_params(self, archetype: str) -> dict:
        params = {
            'elite_playmaker': {
                'recent_weight': 0.45,
                'trend_modifier': 1.2,
                'std_multiplier': 1.10,
                'matchup_weight': 0.35,
                'defense_sensitivity': 0.08,
                'home_adj': 0.6,
                'away_adj': -0.4
            },
            'dominant_big': {
                'recent_weight': 0.40,
                'trend_modifier': 1.0,
                'std_multiplier': 0.90,
                'matchup_weight': 0.40,
                'defense_sensitivity': 0.06,
                'home_adj': 0.4,
                'away_adj': -0.2
            },
            'primary_scorer': {
                'recent_weight': 0.45,
                'trend_modifier': 1.3,
                'std_multiplier': 1.15,
                'matchup_weight': 0.30,
                'defense_sensitivity': 0.07,
                'home_adj': 0.6,
                'away_adj': -0.4
            },
            'floor_general': {
                'recent_weight': 0.40,
                'trend_modifier': 1.1,
                'std_multiplier': 1.0,
                'matchup_weight': 0.35,
                'defense_sensitivity': 0.05,
                'home_adj': 0.5,
                'away_adj': -0.3
            },
            'interior_scorer': {
                'recent_weight': 0.38,
                'trend_modifier': 0.9,
                'std_multiplier': 0.85,
                'matchup_weight': 0.45,
                'defense_sensitivity': 0.08,
                'home_adj': 0.4,
                'away_adj': -0.2
            },
            'two_way_wing': {
                'recent_weight': 0.35,
                'trend_modifier': 1.0,
                'std_multiplier': 0.95,
                'matchup_weight': 0.35,
                'defense_sensitivity': 0.06,
                'home_adj': 0.5,
                'away_adj': -0.3
            },
            'secondary_scorer': {
                'recent_weight': 0.40,
                'trend_modifier': 1.1,
                'std_multiplier': 1.05,
                'matchup_weight': 0.30,
                'defense_sensitivity': 0.07,
                'home_adj': 0.5,
                'away_adj': -0.3
            },
            'playmaker': {
                'recent_weight': 0.38,
                'trend_modifier': 1.0,
                'std_multiplier': 1.0,
                'matchup_weight': 0.35,
                'defense_sensitivity': 0.05,
                'home_adj': 0.4,
                'away_adj': -0.3
            },
            'stretch_big': {
                'recent_weight': 0.35,
                'trend_modifier': 1.1,
                'std_multiplier': 1.05,
                'matchup_weight': 0.40,
                'defense_sensitivity': 0.07,
                'home_adj': 0.4,
                'away_adj': -0.2
            },
            'defensive_anchor': {
                'recent_weight': 0.30,
                'trend_modifier': 0.8,
                'std_multiplier': 0.80,
                'matchup_weight': 0.45,
                'defense_sensitivity': 0.05,
                'home_adj': 0.3,
                'away_adj': -0.2
            },
            'three_level_scorer': {
                'recent_weight': 0.40,
                'trend_modifier': 1.15,
                'std_multiplier': 1.10,
                'matchup_weight': 0.30,
                'defense_sensitivity': 0.08,
                'home_adj': 0.5,
                'away_adj': -0.3
            },
            'rim_protector': {
                'recent_weight': 0.25,
                'trend_modifier': 0.7,
                'std_multiplier': 0.70,
                'matchup_weight': 0.50,
                'defense_sensitivity': 0.04,
                'home_adj': 0.2,
                'away_adj': -0.1
            },
            'connector': {
                'recent_weight': 0.35,
                'trend_modifier': 0.9,
                'std_multiplier': 0.90,
                'matchup_weight': 0.40,
                'defense_sensitivity': 0.05,
                'home_adj': 0.3,
                'away_adj': -0.2
            },
            'three_and_d': {
                'recent_weight': 0.32,
                'trend_modifier': 1.0,
                'std_multiplier': 1.0,
                'matchup_weight': 0.35,
                'defense_sensitivity': 0.06,
                'home_adj': 0.4,
                'away_adj': -0.2
            },
            'role_player': {
                'recent_weight': 0.30,
                'trend_modifier': 0.8,
                'std_multiplier': 0.90,
                'matchup_weight': 0.40,
                'defense_sensitivity': 0.05,
                'home_adj': 0.3,
                'away_adj': -0.2
            },
            'facilitator': {
                'recent_weight': 0.30,
                'trend_modifier': 0.9,
                'std_multiplier': 0.85,
                'matchup_weight': 0.35,
                'defense_sensitivity': 0.04,
                'home_adj': 0.2,
                'away_adj': -0.1
            }
        }
        
        return params.get(archetype, params['role_player'])
    
    def _calculate_position_defense_factor(
        self, 
        opponent_defense: dict, 
        position: str, 
        stat: str,
        archetype_params: dict
    ) -> float:
        def safe_int(val, default=15):
            try:
                return int(float(val)) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        league_rank = safe_int(opponent_defense.get('league_rank'), 15)
        pts_allowed = opponent_defense.get('pts_allowed_per_100', 114.0)
        position_ranks = opponent_defense.get('position_ranks', {})
        
        defense_sensitivity = archetype_params.get('defense_sensitivity', 0.06)
        
        position_rank = safe_int(position_ranks.get(position), league_rank)
        
        if stat == 'pts':
            if position_rank >= 26:
                return 1.0 + (defense_sensitivity * 1.8)
            elif position_rank >= 21:
                return 1.0 + (defense_sensitivity * 1.2)
            elif position_rank >= 16:
                return 1.0 + (defense_sensitivity * 0.5)
            elif position_rank <= 3:
                return 1.0 - (defense_sensitivity * 1.5)
            elif position_rank <= 5:
                return 1.0 - (defense_sensitivity * 1.0)
            elif position_rank <= 10:
                return 1.0 - (defense_sensitivity * 0.5)
            return 1.0
        
        elif stat == 'reb':
            if position in ['C', 'PF']:
                if position_rank >= 21:
                    return 1.0 + 0.05
                elif position_rank <= 10:
                    return 1.0 - 0.04
            else:
                if position_rank >= 21:
                    return 1.0 + 0.03
                elif position_rank <= 10:
                    return 1.0 - 0.02
            return 1.0
        
        elif stat == 'ast':
            if position in ['PG', 'SG']:
                if position_rank >= 21:
                    return 1.0 + 0.04
                elif position_rank <= 10:
                    return 1.0 - 0.03
            return 1.0
        
        return 1.0
    
    def _calculate_archetype_trend_adjustment(
        self, 
        trend: dict, 
        current_mean: float,
        archetype_params: dict
    ) -> float:
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        direction = trend.get('direction', 'neutral')
        pct_change = safe_float(trend.get('pct_change'), 0)
        recent_3_avg = safe_float(trend.get('recent_3_avg'), current_mean)
        
        trend_mod = safe_float(archetype_params.get('trend_modifier'), 1.0)
        
        if direction == 'up':
            if pct_change > 40:
                base_adj = min(recent_3_avg - current_mean, current_mean * 0.15)
            elif pct_change > 25:
                base_adj = min(recent_3_avg - current_mean, current_mean * 0.12) * 0.8
            elif pct_change > 15:
                base_adj = min(recent_3_avg - current_mean, current_mean * 0.10) * 0.6
            elif pct_change > 5:
                base_adj = min(recent_3_avg - current_mean, current_mean * 0.08) * 0.4
            else:
                base_adj = 0.0
            return base_adj * trend_mod
        
        elif direction == 'down':
            if pct_change < -40:
                base_adj = max(recent_3_avg - current_mean, -current_mean * 0.15)
            elif pct_change < -25:
                base_adj = max(recent_3_avg - current_mean, -current_mean * 0.12) * 0.8
            elif pct_change < -15:
                base_adj = max(recent_3_avg - current_mean, -current_mean * 0.10) * 0.6
            elif pct_change < -5:
                base_adj = max(recent_3_avg - current_mean, -current_mean * 0.08) * 0.4
            else:
                base_adj = 0.0
            return base_adj * trend_mod
        
        return 0.0
    
    def _calculate_archetype_matchup_adjustment(
        self,
        matchup_stat: float,
        base_mean: float,
        current_mean: float,
        archetype_params: dict
    ) -> float:
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        matchup_stat = safe_float(matchup_stat, 0)
        base_mean = safe_float(base_mean, 0)
        current_mean = safe_float(current_mean, 0)
        
        if matchup_stat <= 0 or base_mean <= 0:
            return 0.0
        
        matchup_weight = safe_float(archetype_params.get('matchup_weight'), 0.35)
        
        diff_pct = (matchup_stat - base_mean) / base_mean
        
        if diff_pct > 0.30:
            raw_adj = min(matchup_stat - current_mean, current_mean * 0.15)
        elif diff_pct > 0.20:
            raw_adj = min(matchup_stat - current_mean, current_mean * 0.12) * 0.85
        elif diff_pct > 0.10:
            raw_adj = min(matchup_stat - current_mean, current_mean * 0.08) * 0.7
        elif diff_pct > 0.05:
            raw_adj = (matchup_stat - current_mean) * 0.5
        elif diff_pct < -0.30:
            raw_adj = max(matchup_stat - current_mean, -current_mean * 0.15)
        elif diff_pct < -0.20:
            raw_adj = max(matchup_stat - current_mean, -current_mean * 0.12) * 0.85
        elif diff_pct < -0.10:
            raw_adj = max(matchup_stat - current_mean, -current_mean * 0.08) * 0.7
        elif diff_pct < -0.05:
            raw_adj = (matchup_stat - current_mean) * 0.5
        else:
            raw_adj = 0.0
        
        return raw_adj * matchup_weight

    def _handle_projection(self, parsed: ParsedQuery):
        player_name = parsed.player_name
        stat = parsed.stat
        opponent = parsed.opponent
        
        projection = self.loader.find_player(
            player_name=player_name,
            opponent=opponent
        )
        
        if projection is None:
            print(f"\n🔍 Player '{player_name}' not found in cached projections.")
            
            should_sim = self._prompt_live_simulation(player_name)
            if should_sim:
                projection = self._run_live_simulation(player_name, opponent)
            
            if projection is None:
                return
        
        self._print_projection(projection, stat)
    
    def _handle_compare(self, parsed: ParsedQuery):
        players = parsed.compare_players
        stat = parsed.stat or 'pts'
        
        if not players or len(players) < 2:
            print("\n❌ Need at least 2 players to compare.")
            return
        
        print(f"\n{'═' * 60}")
        print(f"Comparing {players[0]} vs {players[1]} - {stat.upper()}")
        print(f"{'═' * 60}")
        
        results = []
        for player_name in players:
            projection = self.loader.find_player(player_name)
            if projection:
                results.append({
                    'name': projection.player_name,
                    'mean': projection.get_stat_mean(stat),
                    'std': projection.get_stat_std(stat),
                    'ci': projection.get_stat_ci(stat)
                })
            else:
                print(f"\n⚠ Player '{player_name}' not found.")
        
        if len(results) < 2:
            return
        
        print(f"\n{'Player':<25} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
        print(f"{'─' * 60}")
        
        for r in results:
            ci_str = f"({r['ci'][0]:.1f} - {r['ci'][1]:.1f})"
            print(f"{r['name']:<25} {r['mean']:>8.1f} {r['std']:>8.2f} {ci_str:>20}")
        
        diff = results[0]['mean'] - results[1]['mean']
        favored = results[0]['name'] if diff > 0 else results[1]['name']
        
        print(f"\n▸ {favored} projected {abs(diff):.1f} {stat} higher")
    
    def _print_projection(self, projection: PlayerProjection, stat: Optional[str] = None):
        print(f"\n{'─' * 50}")
        print(f"{projection.player_name} ({projection.team} vs {projection.opponent})")
        print(f"Date: {projection.date}")
        if projection.play_probability < 1.0:
            print(f"Play Probability: {projection.play_probability * 100:.0f}%")
        print(f"{'─' * 50}")
        
        if stat:
            mean = projection.get_stat_mean(stat)
            std = projection.get_stat_std(stat)
            ci = projection.get_stat_ci(stat)
            print(f"\n{stat.upper()}: {mean:.1f} ± {std:.2f}")
            print(f"95% CI: {ci[0]:.1f} - {ci[1]:.1f}")
        else:
            print(f"\n{'Stat':<12} {'Mean':>8} {'Mode':>8} {'95% CI':>18}")
            print(f"{'─' * 50}")
            
            for s in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']:
                mean = projection.get_stat_mean(s)
                mode = getattr(projection, f'{s}_mode', mean)
                ci = projection.get_stat_ci(s)
                ci_str = f"({ci[0]:.1f} - {ci[1]:.1f})"
                print(f"{s.upper():<12} {mean:>8.1f} {mode:>8.1f} {ci_str:>18}")
        
        print(f"{'─' * 50}\n")
    
    def _list_players(self):
        players = self.loader.get_available_players()
        
        if not players:
            print("\n⚠ No players available. Run a simulation first.")
            return
        
        print(f"\n{'═' * 50}")
        print(f"Available Players ({len(players)})")
        print(f"{'═' * 50}")
        
        for i, player in enumerate(players):
            if i % 3 == 0:
                print()
            print(f"  {player:<20}", end='')
            if (i + 1) % 3 == 0:
                print()
        
        print("\n")
    
    def _list_teams(self):
        teams = self.loader.get_available_teams()
        
        if not teams:
            print("\n⚠ No teams available. Run a simulation first.")
            return
        
        print(f"\n{'═' * 50}")
        print(f"Available Teams ({len(teams)})")
        print(f"{'═' * 50}")
        
        for team in teams:
            print(f"  {team}")
        
        print()
    
    def _reload(self):
        print("\n🔄 Reloading projections...")
        self.loader.load_projections(force_reload=True)
        cache_info = self.loader.get_cache_info()
        print(f"✓ Loaded {cache_info['num_projections']} projections\n")
    
    def _show_info(self):
        cache_info = self.loader.get_cache_info()
        
        print(f"\n{'═' * 50}")
        print("System Info")
        print(f"{'═' * 50}")
        print(f"Cache file: {cache_info.get('cache_file', 'None')}")
        print(f"Last load: {cache_info.get('last_load_time', 'Never')}")
        print(f"Projections loaded: {cache_info.get('num_projections', 0)}")
        print(f"{'═' * 50}\n")
    
    def _prompt_live_simulation(self, player_name: str) -> bool:
        print(f"\n  Run a live simulation for {player_name}? [y/N]: ", end='')
        try:
            response = input().strip().lower()
            return response in ('y', 'yes')
        except (EOFError, OSError):
            return False
    
    def _run_live_simulation(self, player_name: str, opponent: Optional[str]) -> Optional[PlayerProjection]:
        print(f"\n⏳ Running live simulation for {player_name}...")
        
        try:
            if self._simulator is None:
                from src.models.model_manager import ModelManager
                from src.simulation.game_simulator import GameSimulator
                
                manager = ModelManager()
                manager._load_models()
                self._simulator = GameSimulator(manager)
            
            self._simulator.prepare_simulation_context()
            
            teams = self._find_player_team(player_name)
            
            if not teams:
                print(f"❌ Could not find team for {player_name}")
                return None
            
            team = teams[0]
            if not opponent:
                available_teams = [t for t in self._simulator.get_available_teams() if t != team]
                if available_teams:
                    import random
                    opponent = random.choice(available_teams)
                else:
                    opponent = "LAL"
            
            is_home = True
            
            print(f"  Simulating {team} vs {opponent}...")
            
            result = self._simulator.simulate_matchup(
                team_a=team if is_home else opponent,
                team_b=opponent if is_home else team,
                num_sims=self.num_sims
            )
            
            if 'error' in result:
                print(f"❌ Simulation error: {result['error']}")
                return None
            
            for player_avg in result.get('player_averages', []):
                if player_name.lower() in player_avg['name'].lower():
                    return PlayerProjection(
                        player_name=player_avg['name'],
                        team=team,
                        opponent=opponent,
                        is_home=is_home,
                        date=datetime.now().strftime('%Y-%m-%d'),
                        pts_mean=player_avg['pts'],
                        pts_mode=player_avg.get('pts_mode', player_avg['pts']),
                        pts_std=player_avg.get('pts_std', 3.0),
                        pts_ci_low=player_avg.get('pts_95_ci', [0, 0])[0],
                        pts_ci_high=player_avg.get('pts_95_ci', [0, 0])[1],
                        reb_mean=player_avg['reb'],
                        reb_mode=player_avg.get('reb_mode', player_avg['reb']),
                        reb_std=player_avg.get('reb_std', 2.0),
                        reb_ci_low=player_avg.get('reb_95_ci', [0, 0])[0],
                        reb_ci_high=player_avg.get('reb_95_ci', [0, 0])[1],
                        ast_mean=player_avg['ast'],
                        ast_mode=player_avg.get('ast_mode', player_avg['ast']),
                        ast_std=player_avg.get('ast_std', 2.0),
                        ast_ci_low=player_avg.get('ast_95_ci', [0, 0])[0],
                        ast_ci_high=player_avg.get('ast_95_ci', [0, 0])[1],
                        stl_mean=player_avg.get('stl', 0.0),
                        stl_mode=player_avg.get('stl_mode', player_avg.get('stl', 0.0)),
                        stl_std=player_avg.get('stl_std', 0.8),
                        stl_ci_low=player_avg.get('stl_95_ci', [0, 0])[0],
                        stl_ci_high=player_avg.get('stl_95_ci', [0, 0])[1],
                        blk_mean=player_avg.get('blk', 0.0),
                        blk_mode=player_avg.get('blk_mode', player_avg.get('blk', 0.0)),
                        blk_std=player_avg.get('blk_std', 0.6),
                        blk_ci_low=player_avg.get('blk_95_ci', [0, 0])[0],
                        blk_ci_high=player_avg.get('blk_95_ci', [0, 0])[1],
                        tov_mean=player_avg.get('tov', 0.0),
                        tov_mode=player_avg.get('tov_mode', player_avg.get('tov', 0.0)),
                        tov_std=player_avg.get('tov_std', 1.0),
                        tov_ci_low=player_avg.get('tov_95_ci', [0, 0])[0],
                        tov_ci_high=player_avg.get('tov_95_ci', [0, 0])[1],
                        play_probability=player_avg.get('play_probability', 1.0)
                    )
            
            print(f"❌ Player {player_name} not found in simulation results")
            return None
            
        except Exception as e:
            logger.error(f"Live simulation failed: {e}")
            print(f"❌ Simulation failed: {e}")
            return None
    
    def _find_player_team(self, player_name: str) -> List[str]:
        if self._simulator is None:
            return []
        
        self._simulator.load_context()
        
        latest = self._simulator.latest_player_stats
        if latest is None:
            return []
        
        matches = latest[latest['PLAYER_NAME'].str.lower().str.contains(player_name.lower(), na=False)]
        
        if matches.empty:
            return []
        
        return matches['TEAM_ABBREVIATION'].unique().tolist()
    
    def query_one_shot(
        self,
        player_name: str,
        stat: str,
        line: float,
        opponent: Optional[str] = None
    ) -> Optional[ProbabilityResult]:
        projection = self.loader.find_player(
            player_name=player_name,
            opponent=opponent
        )
        
        if projection is None:
            return None

        if opponent:
            context = self.loader.get_player_context(
                player_name=projection.player_name,
                opponent=projection.opponent,
                stat=stat
            )
        else:
            recent_games = self.loader.get_recent_games(projection.player_name, n=5)
            recent_avg = {'pts': 0, 'reb': 0, 'ast': 0, 'stl': 0, 'blk': 0, 'tov': 0, 'min': 0}
            if recent_games:
                for game in recent_games:
                    for key in recent_avg:
                        recent_avg[key] += float(game.get(key, 0) or 0)
                count = len(recent_games)
                recent_avg = {k: v / count for k, v in recent_avg.items()}
            context = {
                'recent_games': recent_games,
                'recent_avg': recent_avg,
                'recent_sample_size': len(recent_games),
                'matchup_history': [],
                'matchup_avg': {},
                'matchup_sample_size': 0,
                'opponent_defense': {},
                'trend': {},
            }
        
        return self.calculator.calculate_from_projection(
            player_name=projection.player_name,
            stat=stat,
            line=line,
            mean=projection.get_stat_mean(stat),
            std=projection.get_stat_std(stat),
            ci_low=projection.get_stat_ci(stat)[0],
            ci_high=projection.get_stat_ci(stat)[1],
            opponent=projection.opponent,
            date=projection.date,
            play_probability=projection.play_probability,
            num_sims=self.num_sims,
            team=projection.team,
            is_home=projection.is_home,
            recent_games=context.get('recent_games'),
            recent_avg=context.get('recent_avg'),
            matchup_history=context.get('matchup_history'),
            matchup_avg=context.get('matchup_avg'),
            opponent_defense=context.get('opponent_defense'),
            trend=context.get('trend'),
        )
