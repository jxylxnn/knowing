"""
NBA Possession-by-Possession Game Simulator.
Simulates games at the possession level for maximum realism.
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class PossessionOutcome(Enum):
    """Possible outcomes of a single possession."""
    FG2_MAKE = "fg2_make"
    FG2_MISS = "fg2_miss"
    FG3_MAKE = "fg3_make"
    FG3_MISS = "fg3_miss"
    FT_MAKE = "ft_make"
    FT_MISS = "ft_miss"
    TURNOVER = "turnover"
    OFF_REBOUND = "off_rebound"
    FOUL = "foul"


@dataclass
class PlayerSimState:
    """Runtime state for a player during simulation."""
    name: str
    team: str
    position: str
    usage_rate: float
    fg_pct: float
    fg3_pct: float
    ft_pct: float
    reb_rate: float
    ast_rate: float
    tov_rate: float
    projected_minutes: float
    minutes_played: float = 0.0
    fouls: int = 0
    pts: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    fga: int = 0
    fgm: int = 0
    fg3a: int = 0
    fg3m: int = 0
    fta: int = 0
    ftm: int = 0
    is_on_court: bool = True
    is_starter: bool = False
    fatigue_level: float = 0.0


@dataclass
class TeamSimState:
    """Runtime state for a team during simulation."""
    team_abbr: str
    players: List[PlayerSimState]
    pts: int = 0
    fga: int = 0
    fgm: int = 0
    fg3a: int = 0
    fg3m: int = 0
    fta: int = 0
    ftm: int = 0
    orb: int = 0
    drb: int = 0
    ast: int = 0
    tov: int = 0
    stl: int = 0
    blk: int = 0
    pf: int = 0
    possessions: int = 0
    timeouts_remaining: int = 7
    is_home: bool = False
    
    def get_on_court_players(self) -> List[PlayerSimState]:
        return [p for p in self.players if p.is_on_court]
    
    def get_total_minutes(self) -> float:
        return sum(p.minutes_played for p in self.players)


@dataclass
class GameState:
    """Full game state for simulation."""
    home_team: TeamSimState
    away_team: TeamSimState
    quarter: int = 1
    time_remaining: float = 720.0  # seconds
    period: int = 1
    is_overtime: bool = False
    possession_arrow: str = "home"
    last_event: str = ""
    
    @property
    def score_diff(self) -> int:
        return self.home_team.pts - self.away_team.pts
    
    @property
    def is_close_game(self) -> bool:
        return abs(self.score_diff) <= 5
    
    @property
    def is_blowout(self) -> bool:
        if self.quarter < 3:
            return False
        return abs(self.score_diff) >= 20
    
    @property
    def is_clutch_time(self) -> bool:
        return self.quarter == 4 and self.time_remaining <= 300 and self.is_close_game
    
    def get_offensive_team(self) -> TeamSimState:
        return self.home_team if self.possession_arrow == "home" else self.away_team
    
    def get_defensive_team(self) -> TeamSimState:
        return self.away_team if self.possession_arrow == "home" else self.home_team


class PossessionSimulator:
    """
    Simulates NBA games possession by possession for maximum realism.
    
    This is the core engine that drives realistic stat generation by:
    1. Simulating each possession independently
    2. Tracking player minutes and fatigue
    3. Applying game context (blowouts, clutch time)
    4. Enforcing team total constraints via possession flow
    """
    
    LEAGUE_AVERAGES = {
        'fg_pct': 0.472,
        'fg3_pct': 0.362,
        'ft_pct': 0.780,
        'tov_pct': 0.135,
        'orb_pct': 0.250,
        'fg3_freq': 0.39,
        'ft_rate': 0.230,
        'ast_pct': 0.60
    }
    
    PLAYER_TYPE_PROFILES = {
        'star': {'usage': 0.28, 'fg3_freq': 0.35, 'ft_rate': 0.25, 'tov_rate': 0.12},
        'starter': {'usage': 0.20, 'fg3_freq': 0.38, 'ft_rate': 0.20, 'tov_rate': 0.14},
        'role_player': {'usage': 0.14, 'fg3_freq': 0.45, 'ft_rate': 0.15, 'tov_rate': 0.16},
        'bench': {'usage': 0.12, 'fg3_freq': 0.42, 'ft_rate': 0.12, 'tov_rate': 0.18}
    }
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self._game_log: List[dict] = []
    
    def simulate_game(
        self,
        home_roster: List[dict],
        away_roster: List[dict],
        home_pace: float = 100.0,
        away_pace: float = 100.0,
        home_off_rating: float = 114.0,
        away_off_rating: float = 114.0,
        home_def_rating: float = 114.0,
        away_def_rating: float = 114.0
    ) -> Dict:
        """
        Simulate a full game possession by possession.
        
        Args:
            home_roster: List of player dicts with projection info
            away_roster: List of player dicts with projection info
            home_pace: Home team pace (possessions per game)
            away_pace: Away team pace
            home_off_rating: Home offensive rating
            away_off_rating: Away offensive rating
            home_def_rating: Home defensive rating
            away_def_rating: Away defensive rating
            
        Returns:
            Game result with player stats and final score
        """
        home_state = self._initialize_team_state(home_roster, "home")
        away_state = self._initialize_team_state(away_roster, "away")
        
        game_state = GameState(
            home_team=home_state,
            away_team=away_state,
            quarter=1,
            time_remaining=720.0
        )
        
        avg_pace = (home_pace + away_pace) / 2
        expected_possessions = int(avg_pace)
        
        # Guard against division by zero (defensive ratings should never be 0)
        safe_away_def = max(away_def_rating, 1.0)
        safe_home_def = max(home_def_rating, 1.0)
        home_eff_factor = home_off_rating / 114.0
        away_eff_factor = away_def_rating / 114.0
        home_adj = home_eff_factor * (114.0 / safe_away_def)
        away_adj = (away_off_rating / 114.0) * (114.0 / safe_home_def)
        
        possession_count = 0
        max_possessions = int(expected_possessions * 2 * 2 + 20)
        
        for quarter in range(1, 5):
            game_state.quarter = quarter
            game_state.time_remaining = 720.0
            
            quarter_possessions = int(avg_pace * 2 / 4) + self.rng.integers(-6, 7)
            
            for _ in range(quarter_possessions):
                if possession_count >= max_possessions:
                    break
                
                self._simulate_possession(
                    game_state,
                    home_adj if game_state.possession_arrow == "home" else away_adj
                )
                
                game_state.possession_arrow = "away" if game_state.possession_arrow == "home" else "home"
                possession_count += 1
            
            self._handle_substitutions(game_state, quarter)
        
        while game_state.score_diff == 0:
            game_state.is_overtime = True
            game_state.time_remaining = 300.0
            
            for _ in range(int(avg_pace / 5)):
                self._simulate_possession(
                    game_state,
                    home_adj if game_state.possession_arrow == "home" else away_adj
                )
                game_state.possession_arrow = "away" if game_state.possession_arrow == "home" else "home"
                
                if game_state.time_remaining <= 0:
                    break
        
        return self._compile_game_results(game_state)
    
    def _initialize_team_state(self, roster: List[dict], side: str) -> TeamSimState:
        """Initialize team state from roster projections."""
        players = []
        
        for i, p in enumerate(roster):
            player = PlayerSimState(
                name=p.get('name', f"Player_{i}"),
                team=p.get('team', 'UNK'),
                position=p.get('position', 'SG'),
                usage_rate=p.get('usage', p.get('usage_rate', 0.15)),
                fg_pct=p.get('fg_pct', 0.47),
                fg3_pct=p.get('fg3_pct', 0.36),
                ft_pct=p.get('ft_pct', 0.78),
                reb_rate=p.get('reb_rate', p.get('reb', 5) / 10),
                ast_rate=p.get('ast_rate', p.get('ast', 2) / 5),
                tov_rate=p.get('tov_rate', 0.14),
                projected_minutes=p.get('exp_min', p.get('projected_minutes', 24)),
                is_starter=i < 5,
                is_on_court=i < 5
            )
            players.append(player)
        
        return TeamSimState(
            team_abbr=roster[0].get('team', 'UNK') if roster else 'UNK',
            players=players,
            is_home=(side == "home")
        )
    
    def _simulate_possession(self, game_state: GameState, efficiency_adj: float):
        """Simulate a single possession."""
        offense = game_state.get_offensive_team()
        defense = game_state.get_defensive_team()
        
        offense.possessions += 1
        
        ball_handler = self._select_ball_handler(offense, game_state)
        
        if ball_handler is None:
            return
        
        tov_occurred = self._check_turnover(ball_handler, defense, efficiency_adj)
        if tov_occurred:
            ball_handler.tov += 1
            offense.tov += 1
            return
        
        shot_outcome = self._simulate_shot(ball_handler, offense, defense, efficiency_adj, game_state)
        
        if not shot_outcome['made']:
            reb_outcome = self._simulate_rebound(offense, defense)
            if reb_outcome['offensive']:
                offense.orb += 1
                ball_handler.reb += 1 if self.rng.random() < ball_handler.reb_rate else 0
            else:
                defense.drb += 1
                for d_player in defense.get_on_court_players():
                    if self.rng.random() < d_player.reb_rate:
                        d_player.reb += 1
                        break
        
        ball_handler.minutes_played += 0.2
    
    def _select_ball_handler(self, offense: TeamSimState, game_state: GameState) -> Optional[PlayerSimState]:
        """Select which player handles the ball based on usage and game context."""
        on_court = offense.get_on_court_players()
        if not on_court:
            return None
        
        weights = []
        for p in on_court:
            weight = p.usage_rate
            if game_state.is_clutch_time and p.is_starter:
                weight *= 1.3
            if game_state.is_blowout and not p.is_starter:
                weight *= 1.5
            weight *= (1 - p.fatigue_level * 0.1)
            weights.append(weight)
        
        total = sum(weights)
        if total == 0:
            total = 1
        weights = [w / total for w in weights]
        
        return self.rng.choice(on_court, p=weights)
    
    def _check_turnover(self, ball_handler: PlayerSimState, defense: TeamSimState, eff_adj: float) -> bool:
        """Check if turnover occurs."""
        base_tov_rate = ball_handler.tov_rate * self.LEAGUE_AVERAGES['tov_pct']
        safe_eff_adj = max(eff_adj, 0.01)  # Guard against division by zero
        adj_tov_rate = base_tov_rate / safe_eff_adj
        return self.rng.random() < adj_tov_rate
    
    def _simulate_shot(
        self,
        ball_handler: PlayerSimState,
        offense: TeamSimState,
        defense: TeamSimState,
        eff_adj: float,
        game_state: GameState
    ) -> dict:
        """Simulate a shot attempt and result."""
        shot_type = self._determine_shot_type(ball_handler, game_state)
        
        if shot_type == 'fg3':
            fg_pct = ball_handler.fg3_pct * eff_adj
            ball_handler.fg3a += 1
            offense.fg3a += 1
            ball_handler.fga += 1
            offense.fga += 1
            
            made = self.rng.random() < fg_pct
            if made:
                ball_handler.fg3m += 1
                ball_handler.fgm += 1
                offense.fg3m += 1
                offense.fgm += 1
                ball_handler.pts += 3
                offense.pts += 3
                
                if self.rng.random() < ball_handler.ast_rate:
                    assister = self._select_assister(offense, exclude=ball_handler)
                    if assister:
                        assister.ast += 1
                        offense.ast += 1
        else:
            fg_pct = ball_handler.fg_pct * eff_adj
            if shot_type == 'fg2':
                fg_pct *= 1.05
            
            ball_handler.fga += 1
            offense.fga += 1
            
            made = self.rng.random() < fg_pct
            if made:
                ball_handler.fgm += 1
                offense.fgm += 1
                pts = 2
                ball_handler.pts += pts
                offense.pts += pts
                
                if self.rng.random() < ball_handler.ast_rate:
                    assister = self._select_assister(offense, exclude=ball_handler)
                    if assister:
                        assister.ast += 1
                        offense.ast += 1
        
        ft_occurred = self._check_free_throws(ball_handler, offense, shot_type, made)
        if ft_occurred:
            self._shoot_free_throws(ball_handler, offense)
        
        if not made:
            for d_player in defense.get_on_court_players():
                if self.rng.random() < 0.03:
                    d_player.blk += 1
                    defense.blk += 1
                    break
        
        return {'made': made, 'shot_type': shot_type}
    
    def _determine_shot_type(self, player: PlayerSimState, game_state: GameState) -> str:
        """Determine field goal type (2PT vs 3PT)."""
        fg3_freq = player.usage_rate * 1.5 if player.fg3_pct > 0.38 else player.usage_rate * 0.8
        
        if game_state.is_clutch_time:
            fg3_freq *= 1.2
        elif game_state.is_blowout:
            fg3_freq *= 0.8
        
        fg3_freq = min(0.55, max(0.15, fg3_freq))
        
        if self.rng.random() < fg3_freq:
            return 'fg3'
        return 'fg2'
    
    def _check_free_throws(self, player: PlayerSimState, team: TeamSimState, shot_type: str, made: bool) -> bool:
        """Check if fouled and going to line."""
        ft_rate = self.LEAGUE_AVERAGES['ft_rate']
        
        if shot_type == 'fg3':
            ft_rate *= 0.8
        elif shot_type == 'fg2':
            ft_rate *= 1.2
        
        return self.rng.random() < ft_rate
    
    def _shoot_free_throws(self, player: PlayerSimState, team: TeamSimState):
        """Simulate free throw attempts."""
        num_fts = self.rng.choice([1, 2, 3], p=[0.15, 0.70, 0.15])
        
        player.fta += num_fts
        team.fta += num_fts
        
        for _ in range(num_fts):
            if self.rng.random() < player.ft_pct:
                player.ftm += 1
                team.ftm += 1
                player.pts += 1
                team.pts += 1
    
    def _select_assister(self, team: TeamSimState, exclude: PlayerSimState) -> Optional[PlayerSimState]:
        """Select player to credit assist."""
        candidates = [p for p in team.get_on_court_players() if p != exclude]
        if not candidates:
            return None
        
        weights = [p.ast_rate for p in candidates]
        total = sum(weights)
        if total == 0:
            return None
        weights = [w / total for w in weights]
        
        return self.rng.choice(candidates, p=weights)
    
    def _simulate_rebound(self, offense: TeamSimState, defense: TeamSimState) -> dict:
        """Simulate rebound after missed shot."""
        orb_pct = self.LEAGUE_AVERAGES['orb_pct']
        
        offensive_rebound = self.rng.random() < orb_pct
        
        return {'offensive': offensive_rebound}
    
    def _handle_substitutions(self, game_state: GameState, quarter: int):
        """Handle player substitutions based on minutes and context."""
        for team in [game_state.home_team, game_state.away_team]:
            on_court = team.get_on_court_players()
            bench = [p for p in team.players if not p.is_on_court]
            
            for player in on_court:
                if player.minutes_played >= player.projected_minutes * 0.9:
                    if game_state.is_blowout or player.fatigue_level > 0.8:
                        player.is_on_court = False
                        if bench:
                            replacement = max(bench, key=lambda p: p.projected_minutes - p.minutes_played)
                            replacement.is_on_court = True
                        
                player.fatigue_level = player.minutes_played / max(player.projected_minutes, 1)
            
            if quarter == 2 and len([p for p in team.players if p.is_on_court and not p.is_starter]) < 2:
                for starter in [p for p in team.players if p.is_starter and p.is_on_court][:2]:
                    starter.is_on_court = False
                    if bench:
                        bench_player = max(bench, key=lambda p: p.projected_minutes - p.minutes_played)
                        bench_player.is_on_court = True
    
    def _compile_game_results(self, game_state: GameState) -> Dict:
        """Compile final game results."""
        results = {
            'home_team': game_state.home_team.team_abbr,
            'away_team': game_state.away_team.team_abbr,
            'home_pts': game_state.home_team.pts,
            'away_pts': game_state.away_team.pts,
            'home_fga': game_state.home_team.fga,
            'away_fga': game_state.away_team.fga,
            'home_fgm': game_state.home_team.fgm,
            'away_fgm': game_state.away_team.fgm,
            'home_fg3a': game_state.home_team.fg3a,
            'away_fg3a': game_state.away_team.fg3a,
            'home_fg3m': game_state.home_team.fg3m,
            'away_fg3m': game_state.away_team.fg3m,
            'home_fta': game_state.home_team.fta,
            'away_fta': game_state.away_team.fta,
            'home_ftm': game_state.home_team.ftm,
            'away_ftm': game_state.away_team.ftm,
            'home_orb': game_state.home_team.orb,
            'away_orb': game_state.away_team.orb,
            'home_drb': game_state.home_team.drb,
            'away_drb': game_state.away_team.drb,
            'home_ast': game_state.home_team.ast,
            'away_ast': game_state.away_team.ast,
            'home_tov': game_state.home_team.tov,
            'away_tov': game_state.away_team.tov,
            'possessions': game_state.home_team.possessions,
            'was_overtime': game_state.is_overtime,
            'player_stats': {
                'home': [],
                'away': []
            }
        }
        
        results['home_fg_pct'] = results['home_fgm'] / max(results['home_fga'], 1)
        results['away_fg_pct'] = results['away_fgm'] / max(results['away_fga'], 1)
        
        results['home_or_rating'] = (results['home_pts'] / max(results['possessions'], 1)) * 100
        results['away_or_rating'] = (results['away_pts'] / max(results['possessions'], 1)) * 100
        
        for player in game_state.home_team.players:
            results['player_stats']['home'].append({
                'name': player.name,
                'pts': player.pts,
                'reb': player.reb,
                'ast': player.ast,
                'stl': player.stl,
                'blk': player.blk,
                'tov': player.tov,
                'fgm': player.fgm,
                'fga': player.fga,
                'fg3m': player.fg3m,
                'fg3a': player.fg3a,
                'ftm': player.ftm,
                'fta': player.fta,
                'min': round(player.minutes_played, 1),
                'is_starter': player.is_starter
            })
        
        for player in game_state.away_team.players:
            results['player_stats']['away'].append({
                'name': player.name,
                'pts': player.pts,
                'reb': player.reb,
                'ast': player.ast,
                'stl': player.stl,
                'blk': player.blk,
                'tov': player.tov,
                'fgm': player.fgm,
                'fga': player.fga,
                'fg3m': player.fg3m,
                'fg3a': player.fg3a,
                'ftm': player.ftm,
                'fta': player.fta,
                'min': round(player.minutes_played, 1),
                'is_starter': player.is_starter
            })
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sim = PossessionSimulator(seed=42)
    
    home_roster = [
        {'name': 'Tatum', 'team': 'BOS', 'position': 'SF', 'usage': 0.28, 'fg_pct': 0.47, 'fg3_pct': 0.38, 'ft_pct': 0.85, 'reb_rate': 0.07, 'ast_rate': 0.25, 'projected_minutes': 36},
        {'name': 'Brown', 'team': 'BOS', 'position': 'SG', 'usage': 0.25, 'fg_pct': 0.48, 'fg3_pct': 0.36, 'ft_pct': 0.78, 'reb_rate': 0.06, 'ast_rate': 0.20, 'projected_minutes': 34},
        {'name': 'Horford', 'team': 'BOS', 'position': 'C', 'usage': 0.14, 'fg_pct': 0.52, 'fg3_pct': 0.38, 'ft_pct': 0.82, 'reb_rate': 0.12, 'ast_rate': 0.22, 'projected_minutes': 30},
        {'name': 'White', 'team': 'BOS', 'position': 'PG', 'usage': 0.15, 'fg_pct': 0.46, 'fg3_pct': 0.40, 'ft_pct': 0.90, 'reb_rate': 0.04, 'ast_rate': 0.28, 'projected_minutes': 32},
        {'name': 'Holiday', 'team': 'BOS', 'position': 'PG', 'usage': 0.16, 'fg_pct': 0.46, 'fg3_pct': 0.37, 'ft_pct': 0.85, 'reb_rate': 0.05, 'ast_rate': 0.30, 'projected_minutes': 33},
        {'name': 'Porzingis', 'team': 'BOS', 'position': 'C', 'usage': 0.22, 'fg_pct': 0.48, 'fg3_pct': 0.36, 'ft_pct': 0.85, 'reb_rate': 0.11, 'ast_rate': 0.12, 'projected_minutes': 28},
        {'name': 'Pritchard', 'team': 'BOS', 'position': 'PG', 'usage': 0.18, 'fg_pct': 0.44, 'fg3_pct': 0.40, 'ft_pct': 0.92, 'reb_rate': 0.03, 'ast_rate': 0.25, 'projected_minutes': 22},
        {'name': 'Hauser', 'team': 'BOS', 'position': 'SF', 'usage': 0.12, 'fg_pct': 0.45, 'fg3_pct': 0.42, 'ft_pct': 0.88, 'reb_rate': 0.04, 'ast_rate': 0.15, 'projected_minutes': 18},
    ]
    
    away_roster = [
        {'name': 'LeBron', 'team': 'LAL', 'position': 'SF', 'usage': 0.28, 'fg_pct': 0.52, 'fg3_pct': 0.36, 'ft_pct': 0.73, 'reb_rate': 0.10, 'ast_rate': 0.35, 'projected_minutes': 35},
        {'name': 'Davis', 'team': 'LAL', 'position': 'C', 'usage': 0.26, 'fg_pct': 0.55, 'fg3_pct': 0.28, 'ft_pct': 0.80, 'reb_rate': 0.15, 'ast_rate': 0.18, 'projected_minutes': 36},
        {'name': 'Reaves', 'team': 'LAL', 'position': 'SG', 'usage': 0.20, 'fg_pct': 0.49, 'fg3_pct': 0.38, 'ft_pct': 0.86, 'reb_rate': 0.05, 'ast_rate': 0.28, 'projected_minutes': 33},
        {'name': 'Russell', 'team': 'LAL', 'position': 'PG', 'usage': 0.22, 'fg_pct': 0.46, 'fg3_pct': 0.40, 'ft_pct': 0.82, 'reb_rate': 0.04, 'ast_rate': 0.32, 'projected_minutes': 30},
        {'name': 'Hachimura', 'team': 'LAL', 'position': 'PF', 'usage': 0.14, 'fg_pct': 0.50, 'fg3_pct': 0.34, 'ft_pct': 0.75, 'reb_rate': 0.08, 'ast_rate': 0.10, 'projected_minutes': 26},
        {'name': 'Vincent', 'team': 'LAL', 'position': 'PG', 'usage': 0.12, 'fg_pct': 0.42, 'fg3_pct': 0.35, 'ft_pct': 0.80, 'reb_rate': 0.04, 'ast_rate': 0.25, 'projected_minutes': 20},
        {'name': 'Wood', 'team': 'LAL', 'position': 'C', 'usage': 0.16, 'fg_pct': 0.48, 'fg3_pct': 0.36, 'ft_pct': 0.72, 'reb_rate': 0.13, 'ast_rate': 0.12, 'projected_minutes': 18},
        {'name': 'Reddish', 'team': 'LAL', 'position': 'SF', 'usage': 0.10, 'fg_pct': 0.40, 'fg3_pct': 0.32, 'ft_pct': 0.78, 'reb_rate': 0.04, 'ast_rate': 0.12, 'projected_minutes': 15},
    ]
    
    result = sim.simulate_game(
        home_roster=home_roster,
        away_roster=away_roster,
        home_pace=100.0,
        away_pace=99.0,
        home_off_rating=120.0,
        away_off_rating=115.0,
        home_def_rating=110.0,
        away_def_rating=112.0
    )
    
    print(f"\nGame Result: {result['away_team']} @ {result['home_team']}")
    print(f"Score: {result['away_pts']} - {result['home_pts']}")
    print(f"Possessions: {result['possessions']}")
    print(f"Home ORtg: {result['home_or_rating']:.1f}")
    print(f"Away ORtg: {result['away_or_rating']:.1f}")
    print(f"\nHome Player Stats:")
    for p in result['player_stats']['home'][:5]:
        print(f"  {p['name']}: {p['pts']}pts, {p['reb']}reb, {p['ast']}ast ({p['min']}min)")
    print(f"\nAway Player Stats:")
    for p in result['player_stats']['away'][:5]:
        print(f"  {p['name']}: {p['pts']}pts, {p['reb']}reb, {p['ast']}ast ({p['min']}min)")