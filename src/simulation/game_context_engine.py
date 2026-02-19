"""
NBA Game Context Engine for realistic game situation adjustments.
Handles blowout logic, clutch time, fatigue, and rest day impacts.
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class GameSituation(Enum):
    NORMAL = "normal"
    BLOWOUT_HOME = "blowout_home"
    BLOWOUT_AWAY = "blowout_away"
    CLUTCH = "clutch"
    OVERTIME = "overtime"
    FOUL_TROUBLE = "foul_trouble"
    B2B = "back_to_back"


@dataclass
class PlayerContext:
    """Context information for a player affecting their performance."""
    name: str
    rest_days: int = 2
    is_b2b: bool = False
    games_last_7_days: int = 0
    minutes_last_3_games: float = 0.0
    fouls: int = 0
    is_home: bool = True
    recent_form: float = 1.0  # 0.8 = cold, 1.0 = normal, 1.2 = hot
    injury_status: float = 1.0  # 0.0 = out, 0.5 = questionable, 1.0 = healthy
    starter: bool = True


@dataclass
class GameContext:
    """Full game context for making adjustments."""
    quarter: int
    time_remaining: float  # seconds
    home_score: int
    away_score: int
    possession: str  # 'home' or 'away'
    home_timeouts: int = 7
    away_timeouts: int = 7
    home_fouls_qtr: int = 0
    away_fouls_qtr: int = 0
    is_overtime: bool = False
    rest_days_home: int = 2
    rest_days_away: int = 2
    
    @property
    def score_diff(self) -> int:
        return self.home_score - self.away_score
    
    @property
    def is_close(self) -> bool:
        return abs(self.score_diff) <= 8
    
    @property
    def is_blowout(self) -> bool:
        if self.quarter < 3:
            return abs(self.score_diff) >= 25
        elif self.quarter == 3:
            return abs(self.score_diff) >= 20
        else:
            return abs(self.score_diff) >= 15
    
    @property
    def is_clutch(self) -> bool:
        return (
            self.quarter == 4 and 
            self.time_remaining <= 300 and 
            abs(self.score_diff) <= 5
        )
    
    @property
    def garbage_time(self) -> bool:
        if self.quarter < 4:
            return False
        if self.is_overtime:
            return False
        return abs(self.score_diff) >= 20 and self.time_remaining <= 360
    
    @property
    def situation(self) -> GameSituation:
        if self.is_overtime:
            return GameSituation.OVERTIME
        if self.is_clutch:
            return GameSituation.CLUTCH
        if self.is_blowout:
            return GameSituation.BLOWOUT_HOME if self.score_diff > 0 else GameSituation.BLOWOUT_AWAY
        return GameSituation.NORMAL


class GameContextEngine:
    """
    Provides context-aware adjustments for realistic NBA simulation.
    
    Key features:
    1. Blowout detection and minutes reduction
    2. Clutch time performance adjustments
    3. Fatigue modeling (B2B, games in 7 days, minutes load)
    4. Rest day impact
    5. Home court advantage adjustments
    6. Foul trouble impact
    """
    
    FATIGUE_COEFFICIENTS = {
        'b2b_penalty': {'pts': 0.92, 'reb': 0.95, 'ast': 0.90, 'min': 0.88},
        '3_in_4_penalty': {'pts': 0.95, 'reb': 0.97, 'ast': 0.94, 'min': 0.92},
        '4_in_5_penalty': {'pts': 0.88, 'reb': 0.92, 'ast': 0.86, 'min': 0.85},
        'rest_advantage': {'pts': 1.03, 'reb': 1.02, 'ast': 1.04, 'min': 1.02},
        'home_advantage': {'pts': 1.015, 'reb': 1.01, 'ast': 1.02, 'min': 1.0},
        'clutch_star_boost': {'pts': 1.12, 'reb': 1.05, 'ast': 1.08},
        'blowout_bench_boost': {'pts': 1.15, 'reb': 1.20, 'ast': 1.10, 'min': 1.25},
        'blowout_star_reduction': {'pts': 0.70, 'reb': 0.75, 'ast': 0.60, 'min': 0.60},
        'garbage_time_usage': {'pts': 1.0, 'reb': 1.15, 'ast': 0.85},
    }
    
    CLUTCH_STAR_USAGE_BOOST = 1.25
    HOME_COURT_ADJUSTMENT = 2.5
    
    def __init__(self):
        self._context_cache: Dict[str, float] = {}
    
    def get_performance_multiplier(
        self,
        player: PlayerContext,
        game_context: GameContext,
        stat_type: str
    ) -> float:
        """
        Get overall performance multiplier for a player in given context.
        
        Combines:
        - Fatigue (B2B, rest, load)
        - Game situation (blowout, clutch)
        - Home/away
        - Foul trouble
        
        Returns:
            Multiplier (1.0 = no adjustment, <1.0 = worse, >1.0 = better)
        """
        multiplier = 1.0
        
        multiplier *= self._get_fatigue_multiplier(player, stat_type)
        
        multiplier *= self._get_situation_multiplier(player, game_context, stat_type)
        
        if not player.is_home:
            multiplier *= self.FATIGUE_COEFFICIENTS['home_advantage'].get(stat_type, 1.0) ** -1
        
        multiplier *= player.injury_status
        
        if player.recent_form != 1.0:
            regressed_form = 0.6 + 0.4 * player.recent_form
            multiplier *= regressed_form
        
        return np.clip(multiplier, 0.5, 1.5)
    
    def get_minutes_adjustment(
        self,
        player: PlayerContext,
        game_context: GameContext,
        projected_minutes: float
    ) -> float:
        """
        Adjust projected minutes based on context.
        """
        adjustment = 1.0
        
        if player.is_b2b:
            adjustment *= self.FATIGUE_COEFFICIENTS['b2b_penalty'].get('min', 0.88)
        elif player.rest_days >= 3:
            adjustment *= self.FATIGUE_COEFFICIENTS['rest_advantage'].get('min', 1.02)
        
        if player.games_last_7_days >= 4:
            adjustment *= self.FATIGUE_COEFFICIENTS['4_in_5_penalty'].get('min', 0.85)
        elif player.games_last_7_days >= 3:
            adjustment *= self.FATIGUE_COEFFICIENTS['3_in_4_penalty'].get('min', 0.92)
        
        if player.minutes_last_3_games > 108:
            adjustment *= 0.95
        elif player.minutes_last_3_games > 120:
            adjustment *= 0.90
        
        if game_context.is_blowout:
            if player.starter:
                adjustment *= self.FATIGUE_COEFFICIENTS['blowout_star_reduction'].get('min', 0.60)
            else:
                adjustment *= self.FATIGUE_COEFFICIENTS['blowout_bench_boost'].get('min', 1.25)
        
        if game_context.garbage_time:
            if player.starter:
                adjustment *= 0.40
            else:
                adjustment *= 1.50
        
        if player.fouls >= 5:
            adjustment *= 0.5
        elif player.fouls >= 4:
            adjustment *= 0.85
        
        adjusted = projected_minutes * adjustment
        return np.clip(adjusted, 5, 48)
    
    def get_usage_adjustment(
        self,
        player: PlayerContext,
        game_context: GameContext,
        base_usage: float
    ) -> float:
        """
        Adjust usage rate based on game context.
        """
        usage = base_usage
        
        if game_context.is_clutch:
            if player.starter:
                usage *= self.CLUTCH_STAR_USAGE_BOOST
            else:
                usage *= 0.6
        
        if game_context.garbage_time:
            usage *= 0.8
        
        if player.fouls >= 4:
            usage *= 0.7
        
        return np.clip(usage, 0.05, 0.45)
    
    def _get_fatigue_multiplier(self, player: PlayerContext, stat_type: str) -> float:
        """Calculate fatigue impact on performance."""
        multiplier = 1.0
        
        if player.is_b2b:
            multiplier *= self.FATIGUE_COEFFICIENTS['b2b_penalty'].get(stat_type, 0.92)
        elif player.rest_days >= 3:
            multiplier *= self.FATIGUE_COEFFICIENTS['rest_advantage'].get(stat_type, 1.03)
        
        if player.games_last_7_days >= 4:
            multiplier *= self.FATIGUE_COEFFICIENTS['4_in_5_penalty'].get(stat_type, 0.88)
        elif player.games_last_7_days >= 3:
            multiplier *= self.FATIGUE_COEFFICIENTS['3_in_4_penalty'].get(stat_type, 0.95)
        
        if player.minutes_last_3_games > 108:
            fatigue_factor = 1 - (player.minutes_last_3_games - 108) / 120
            multiplier *= max(0.90, fatigue_factor)
        
        return multiplier
    
    def _get_situation_multiplier(
        self,
        player: PlayerContext,
        game_context: GameContext,
        stat_type: str
    ) -> float:
        """Calculate game situation impact on performance."""
        multiplier = 1.0
        
        if game_context.is_clutch:
            if player.starter:
                multiplier *= self.FATIGUE_COEFFICIENTS['clutch_star_boost'].get(stat_type, 1.08)
            else:
                multiplier *= 0.7
        
        elif game_context.is_blowout:
            if game_context.score_diff > 0:
                if player.is_home:
                    if player.starter:
                        multiplier *= self.FATIGUE_COEFFICIENTS['blowout_star_reduction'].get(stat_type, 0.70)
                    else:
                        multiplier *= self.FATIGUE_COEFFICIENTS['blowout_bench_boost'].get(stat_type, 1.15)
                else:
                    if player.starter:
                        multiplier *= 0.8
            else:
                if not player.is_home:
                    if player.starter:
                        multiplier *= self.FATIGUE_COEFFICIENTS['blowout_star_reduction'].get(stat_type, 0.70)
                    else:
                        multiplier *= self.FATIGUE_COEFFICIENTS['blowout_bench_boost'].get(stat_type, 1.15)
                else:
                    if player.starter:
                        multiplier *= 0.8
        
        if game_context.garbage_time:
            multiplier *= self.FATIGUE_COEFFICIENTS['garbage_time_usage'].get(stat_type, 1.0)
        
        return multiplier
    
    def get_home_court_advantage(self, game_context: GameContext) -> float:
        """
        Get home court advantage in points.
        Adjusted for game situation.
        """
        base_advantage = self.HOME_COURT_ADJUSTMENT
        
        if game_context.is_blowout:
            return base_advantage * 0.3
        
        if game_context.is_clutch:
            return base_advantage * 1.4
        
        return base_advantage
    
    def predict_substitutions(
        self,
        home_players: List[PlayerContext],
        away_players: List[PlayerContext],
        game_context: GameContext,
        current_lineup: Tuple[List[str], List[str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Predict likely substitutions based on context.
        
        Returns:
            Tuple of (new_home_lineup_names, new_away_lineup_names)
        """
        home_lineup, away_lineup = current_lineup
        
        if game_context.garbage_time:
            home_lineup = [p.name for p in home_players if not p.starter][:5] or home_lineup
            away_lineup = [p.name for p in away_players if not p.starter][:5] or away_lineup
        
        elif game_context.is_clutch:
            home_lineup = sorted(home_players, key=lambda p: (-p.starter, -p.recent_form))[:5]
            home_lineup = [p.name for p in home_lineup]
            away_lineup = sorted(away_players, key=lambda p: (-p.starter, -p.recent_form))[:5]
            away_lineup = [p.name for p in away_lineup]
        
        return home_lineup, away_lineup
    
    def calculate_possession_tempo(
        self,
        home_pace: float,
        away_pace: float,
        game_context: GameContext,
        home_players: List[PlayerContext],
        away_players: List[PlayerContext]
    ) -> float:
        """
        Calculate possession duration based on game situation.
        
        Returns:
            Expected possession time in seconds
        """
        base_pace = (home_pace + away_pace) / 2
        base_possession_time = 2880 / (base_pace * 2)
        
        if game_context.is_clutch:
            base_possession_time *= 1.10
        
        if game_context.garbage_time:
            base_possession_time *= 0.90
        
        home_fatigue = np.mean([p.games_last_7_days for p in home_players[:5]])
        away_fatigue = np.mean([p.games_last_7_days for p in away_players[:5]])
        
        if home_fatigue > 3 or away_fatigue > 3:
            base_possession_time *= 1.03
        
        return base_possession_time
    
    def get_foul_probability_adjustment(
        self,
        player: PlayerContext,
        game_context: GameContext,
        base_foul_rate: float = 0.12
    ) -> float:
        """
        Adjust foul probability based on context.
        """
        foul_rate = base_foul_rate
        
        if player.is_b2b:
            foul_rate *= 1.05
        
        if game_context.is_close:
            foul_rate *= 0.92
        
        if game_context.quarter == 4 and game_context.time_remaining < 300:
            if game_context.home_fouls_qtr >= 4 or game_context.away_fouls_qtr >= 4:
                foul_rate *= 0.85
        
        if player.fouls >= 4:
            foul_rate *= 0.70
        
        return foul_rate


class ContextAwareAdjuster:
    """
    High-level interface for applying context adjustments to predictions.
    """
    
    def __init__(self):
        self.context_engine = GameContextEngine()
    
    def adjust_predictions(
        self,
        player_predictions: Dict[str, Dict[str, float]],
        player_contexts: Dict[str, PlayerContext],
        game_context: GameContext
    ) -> Dict[str, Dict[str, float]]:
        """
        Adjust all player predictions based on game context.
        
        Args:
            player_predictions: Dict of player_name -> {pts, reb, ast, min}
            player_contexts: Dict of player_name -> PlayerContext
            game_context: Current game situation
            
        Returns:
            Adjusted predictions dict
        """
        adjusted = {}
        
        for player_name, predictions in player_predictions.items():
            if player_name not in player_contexts:
                adjusted[player_name] = predictions.copy()
                continue
            
            context = player_contexts[player_name]
            
            adj_preds = {}
            
            for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']:
                if stat in predictions:
                    multiplier = self.context_engine.get_performance_multiplier(
                        context, game_context, stat
                    )
                    adj_preds[stat] = predictions[stat] * multiplier
            
            if 'min' in predictions:
                adj_preds['min'] = self.context_engine.get_minutes_adjustment(
                    context, game_context, predictions['min']
                )
            
            adjusted[player_name] = adj_preds
        
        return adjusted
    
    def create_game_context(
        self,
        quarter: int = 1,
        time_remaining: float = 720.0,
        home_score: int = 0,
        away_score: int = 0,
        home_timeouts: int = 7,
        away_timeouts: int = 7,
        home_fouls: int = 0,
        away_fouls: int = 0,
        is_overtime: bool = False,
        rest_home: int = 2,
        rest_away: int = 2
    ) -> GameContext:
        """Factory method to create a GameContext."""
        return GameContext(
            quarter=quarter,
            time_remaining=time_remaining,
            home_score=home_score,
            away_score=away_score,
            possession='home',
            home_timeouts=home_timeouts,
            away_timeouts=away_timeouts,
            home_fouls_qtr=home_fouls,
            away_fouls_qtr=away_fouls,
            is_overtime=is_overtime,
            rest_days_home=rest_home,
            rest_days_away=rest_away
        )
    
    def create_player_context(
        self,
        name: str,
        rest_days: int = 2,
        is_b2b: bool = False,
        games_last_7: int = 0,
        minutes_last_3: float = 0.0,
        fouls: int = 0,
        is_home: bool = True,
        recent_form: float = 1.0,
        injury_status: float = 1.0,
        starter: bool = True
    ) -> PlayerContext:
        """Factory method to create a PlayerContext."""
        return PlayerContext(
            name=name,
            rest_days=rest_days,
            is_b2b=is_b2b,
            games_last_7_days=games_last_7,
            minutes_last_3_games=minutes_last_3,
            fouls=fouls,
            is_home=is_home,
            recent_form=recent_form,
            injury_status=injury_status,
            starter=starter
        )
    
    def simulate_game_progression(
        self,
        initial_context: GameContext,
        home_players: List[PlayerContext],
        away_players: List[PlayerContext],
        num_steps: int = 24
    ) -> List[GameContext]:
        """
        Simulate how game context evolves over time.
        
        Returns list of context snapshots for each step.
        """
        contexts = []
        current = initial_context
        
        step_duration = 2880 / num_steps  # 48 minutes / steps
        
        for step in range(num_steps):
            contexts.append(current)
            
            quarter = (step // 6) + 1
            time_in_qtr = 720 - ((step % 6) + 1) * 120
            
            current = GameContext(
                quarter=quarter,
                time_remaining=time_in_qtr,
                home_score=current.home_score,
                away_score=current.away_score,
                possession='home' if step % 2 == 0 else 'away',
                home_timeouts=current.home_timeouts,
                away_timeouts=current.away_timeouts,
                is_overtime=current.is_overtime,
                rest_days_home=current.rest_days_home,
                rest_days_away=current.rest_days_away
            )
        
        return contexts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = GameContextEngine()
    adjuster = ContextAwareAdjuster()
    
    print("Testing Game Context Engine...")
    
    player = PlayerContext(
        name="Tatum",
        rest_days=1,
        is_b2b=True,
        games_last_7_days=4,
        minutes_last_3_games=110,
        fouls=3,
        is_home=True,
        recent_form=1.1,
        injury_status=1.0,
        starter=True
    )
    
    contexts = [
        ("Normal 1st Quarter", GameContext(1, 720, 20, 18)),
        ("Close Game Q4", GameContext(4, 180, 95, 93)),
        ("Blowout", GameContext(4, 360, 108, 85)),
        ("Clutch", GameContext(4, 60, 102, 101)),
        ("Garbage Time", GameContext(4, 180, 115, 90)),
    ]
    
    print(f"\nPlayer: {player.name} (B2B, 4 games in 7 days)")
    print(f"Recent form: {player.recent_form}")
    print("-" * 60)
    
    for name, ctx in contexts:
        pts_mult = engine.get_performance_multiplier(player, ctx, 'pts')
        min_adj = engine.get_minutes_adjustment(player, ctx, 36)
        usage = engine.get_usage_adjustment(player, ctx, 0.28)
        
        print(f"\n{name}:")
        print(f"  Situation: {ctx.situation.value}")
        print(f"  PTS Multiplier: {pts_mult:.3f}")
        print(f"  Minutes Adjusted: {min_adj:.1f} (from 36)")
        print(f"  Usage: {usage:.3f}")