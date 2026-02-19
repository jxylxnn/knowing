"""
Enhanced NBA Game Simulator with Realistic Simulation Features.
Integrates Four Factors, Possession Simulation, Game Context, and Player Correlations.
"""
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
import hashlib

from src.models.model_manager import ModelManager
from src.simulation.four_factors_engine import FourFactorsEngine, TeamTotalPredictor
from src.simulation.possession_simulator import PossessionSimulator
from src.simulation.game_context_engine import GameContextEngine, ContextAwareAdjuster, PlayerContext, GameContext
from src.simulation.player_correlation_engine import PlayerCorrelationEngine
from src.data.basketball_ref_scraper import BasketballRefScraper
from src.data.nba_defense_scraper import NBADefenseScraper, DefensiveMatchupAnalyzer
from src.data.rotowire_lineup_scraper import RotoWireLineupScraper, LineupManager
from src.data.injury_scraper import InjuryScraper
from src.utils.team_mappings import normalize_team

logger = logging.getLogger(__name__)


class EnhancedGameSimulator:
    """
    Enhanced NBA Game Simulator with 100x better realism.
    
    Features:
    1. Four Factors-based team total prediction
    2. Possession-by-possession simulation
    3. Game context awareness (blowout, clutch, fatigue)
    4. Player stat correlations (assists → points, rebound competition)
    5. Real-time lineup data integration
    6. Position-specific defensive matchups
    """
    
    def __init__(
        self,
        manager: ModelManager,
        use_possession_sim: bool = True,
        use_context_engine: bool = True,
        use_correlations: bool = True,
        use_real_data: bool = True,
        cache_dir: str = 'data/sim_cache',
        device: str = 'auto'
    ):
        self.manager = manager
        self.use_possession_sim = use_possession_sim
        self.use_context_engine = use_context_engine
        self.use_correlations = use_correlations
        self.use_real_data = use_real_data
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"EnhancedGameSimulator initialized on device: {self.device}")
        
        self.players_df = None
        self.games_df = None
        self.merged_data = None
        self.all_merged_with_features = None
        self.latest_player_stats = None
        
        try:
            self.bref_scraper = BasketballRefScraper(cache_dir) if use_real_data else None
            self.defense_scraper = NBADefenseScraper(cache_dir) if use_real_data else None
            self.defense_analyzer = DefensiveMatchupAnalyzer(self.defense_scraper) if use_real_data else None
            self.lineup_scraper = RotoWireLineupScraper(cache_dir) if use_real_data else None
            self.lineup_manager = LineupManager(self.lineup_scraper) if use_real_data else None
            self.injury_scraper = InjuryScraper()
        except Exception as e:
            logger.warning(f"Failed to initialize data scrapers: {e}")
            self.bref_scraper = None
            self.defense_scraper = None
            self.defense_analyzer = None
            self.lineup_scraper = None
            self.lineup_manager = None
        
        self.four_factors = FourFactorsEngine()
        self.team_predictor = TeamTotalPredictor(self.bref_scraper)
        self.possession_sim = PossessionSimulator()
        self.context_engine = GameContextEngine()
        self.context_adjuster = ContextAwareAdjuster()
        self.correlation_engine = PlayerCorrelationEngine(str(self.device))
        
        self.cache_dir = cache_dir
        self._team_stats_cache: Dict[str, dict] = {}
        self._lineup_cache: Dict[str, dict] = {}
        self._correlation_cache: Dict[str, Any] = {}
    
    def load_context(self):
        """Loads the raw data to extract rosters and recent stats."""
        if self.players_df is None:
            self.players_df = pd.read_csv(self.manager.data_dir + '/nba_players.csv')
            self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
            self.games_df = pd.read_csv(self.manager.data_dir + '/nba_games.csv')
            self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
    
    def get_available_teams(self) -> List[str]:
        """Returns list of available team abbreviations."""
        self.load_context()
        if self.games_df is not None:
            return sorted(self.games_df['TEAM_ABBREVIATION'].unique().tolist())
        return []
    
    def simulate_matchup(
        self,
        team_a: str,
        team_b: str,
        num_sims: int = 100,
        date: str = None
    ) -> Dict[str, Any]:
        """
        Simulate a matchup with enhanced realism.
        
        Args:
            team_a: Home team abbreviation
            team_b: Away team abbreviation
            num_sims: Number of Monte Carlo simulations
            date: Game date (YYYY-MM-DD) for context
            
        Returns:
            Simulation results dictionary
        """
        team_a = normalize_team(team_a)
        team_b = normalize_team(team_b)
        logger.info(f"Simulating {team_b} @ {team_a} with {num_sims} simulations")
        
        self.prepare_simulation_context()
        
        team_a_stats = self._get_team_stats(team_a)
        team_b_stats = self._get_team_stats(team_b)
        
        matchup_pace = self._calculate_matchup_pace(team_a_stats, team_b_stats)
        
        four_factors_pred = self.four_factors.predict_matchup(
            self._stats_to_efficiency(team_a_stats),
            self._stats_to_efficiency(team_b_stats),
            num_samples=max(num_sims, 500)
        ) if team_a_stats and team_b_stats else None
        
        injury_probs_a = self._get_injury_probs(team_a)
        injury_probs_b = self._get_injury_probs(team_b)
        
        lineup_a = self._get_lineup(team_a, injury_probs_a)
        lineup_b = self._get_lineup(team_b, injury_probs_b)
        
        if not lineup_a or not lineup_b:
            logger.warning("Insufficient lineup data, using fallback")
            lineup_a = self._get_fallback_lineup(team_a, injury_probs_a)
            lineup_b = self._get_fallback_lineup(team_b, injury_probs_b)
        
        player_contexts_a = self._build_player_contexts(lineup_a, team_a, vs_team=team_b, is_home=True)
        player_contexts_b = self._build_player_contexts(lineup_b, team_b, vs_team=team_a, is_home=False)
        
        predictions_a = self._predict_player_stats(lineup_a, team_a, vs_team=team_b)
        predictions_b = self._predict_player_stats(lineup_b, team_b, vs_team=team_a)
        
        if self.use_context_engine:
            game_context = self._create_game_context(team_a, team_b, date)
            predictions_a = self.context_adjuster.adjust_predictions(
                predictions_a, player_contexts_a, game_context
            )
            predictions_b = self.context_adjuster.adjust_predictions(
                predictions_b, player_contexts_b, game_context
            )
        
        if self.use_possession_sim:
            results = self._run_possession_simulation(
                lineup_a, lineup_b,
                predictions_a, predictions_b,
                team_a_stats, team_b_stats,
                matchup_pace,
                num_sims
            )
        else:
            results = self._run_fast_simulation(
                lineup_a, lineup_b,
                predictions_a, predictions_b,
                team_a_stats, team_b_stats,
                matchup_pace,
                num_sims
            )
        
        if four_factors_pred:
            results['four_factors_prediction'] = four_factors_pred
            results['predicted_pace'] = matchup_pace
        
        results['team_a'] = team_a
        results['team_b'] = team_b
        results['lineups'] = {
            'home': [{'name': p.get('name', ''), 'min': p.get('projected_minutes', 0)} for p in lineup_a[:10]],
            'away': [{'name': p.get('name', ''), 'min': p.get('projected_minutes', 0)} for p in lineup_b[:10]]
        }
        
        return results
    
    def _get_team_stats(self, team_abbr: str) -> Optional[dict]:
        """Get team stats from cache or scraper."""
        if team_abbr in self._team_stats_cache:
            return self._team_stats_cache[team_abbr]
        
        if self.bref_scraper:
            try:
                stats = self.bref_scraper.get_team_stats(team_abbr)
                self._team_stats_cache[team_abbr] = stats
                return stats
            except Exception as e:
                logger.debug(f"Failed to get team stats for {team_abbr}: {e}")
        
        return None
    
    def _get_injury_probs(self, team_abbr: str) -> Dict[str, float]:
        """Get injury probabilities for team players."""
        try:
            if self.injury_scraper:
                return self.injury_scraper.get_player_availability(team_abbr)
        except Exception as e:
            logger.debug(f"Failed to get injury probs for {team_abbr}: {e}")
        return {}
    
    def _get_lineup(self, team_abbr: str, injury_probs: Dict[str, float]) -> List[dict]:
        """Get projected lineup with minutes."""
        if self.lineup_manager:
            try:
                lineup_data = self.lineup_manager.get_enhanced_lineup(team_abbr)
                if lineup_data and lineup_data.get('starters'):
                    return lineup_data.get('starters', []) + lineup_data.get('bench', [])[:7]
            except Exception as e:
                logger.debug(f"Failed to get lineup for {team_abbr}: {e}")
        
        return self._get_fallback_lineup(team_abbr, injury_probs)
    
    def _get_fallback_lineup(self, team_abbr: str, injury_probs: Dict[str, float]) -> List[dict]:
        """Build fallback lineup from historical data."""
        if self.latest_player_stats is None:
            return []
        
        team_players = self.latest_player_stats[
            self.latest_player_stats['TEAM_ABBREVIATION'] == team_abbr
        ]
        
        if team_players.empty:
            return []
        
        recent = team_players[
            (team_players['GAME_DATE'] >= self.latest_player_stats['GAME_DATE'].max() - pd.Timedelta(days=30)) &
            (team_players['MIN'] >= 5)
        ]
        
        if recent.empty:
            recent = team_players
        
        top_players = recent.nlargest(12, 'MIN')
        
        lineup = []
        for _, row in top_players.iterrows():
            player_name = row.get('PLAYER_NAME', '')
            prob = injury_probs.get(player_name, 1.0)
            
            lineup.append({
                'name': player_name,
                'player_id': row.get('PLAYER_ID'),
                'position': self._infer_position(row),
                'usage': float(row.get('USAGE_PROXY_10', row.get('USG_PCT', 0.15))),
                'projected_minutes': float(row.get('ROLL_MIN_AVG_10', row.get('MIN', 20))) * prob,
                'play_probability': prob,
                'pts_avg': float(row.get('ROLL_PTS_AVG_10', row.get('PTS', 10))),
                'reb_avg': float(row.get('ROLL_REB_AVG_10', row.get('REB', 4))),
                'ast_avg': float(row.get('ROLL_AST_AVG_10', row.get('AST', 2))),
                'fg_pct': float(row.get('ROLL_FG_PCT_10', 0.47)),
                'fg3_pct': float(row.get('ROLL_3PT_PCT_10', 0.36)),
                'ft_pct': float(row.get('ROLL_FT_PCT_10', 0.78)),
                'is_starter': len(lineup) < 5
            })
        
        return lineup
    
    def _infer_position(self, player_row) -> str:
        """Infer player position from stats."""
        reb = player_row.get('REB', 0)
        ast = player_row.get('AST', 0)
        height = player_row.get('HEIGHT', 0)
        
        if reb >= 8:
            return 'C'
        elif reb >= 6:
            return 'PF'
        elif ast >= 6:
            return 'PG'
        elif ast >= 4:
            return 'SG'
        else:
            return 'SF'
    
    def _build_player_contexts(
        self,
        lineup: List[dict],
        team: str,
        vs_team: str,
        is_home: bool
    ) -> Dict[str, PlayerContext]:
        """Build player context objects for context engine."""
        contexts = {}
        
        for p in lineup:
            rest_days = 2
            is_b2b = False
            games_last_7 = 3
            minutes_last_3 = p.get('projected_minutes', 24) * 3
            
            context = PlayerContext(
                name=p.get('name', ''),
                rest_days=rest_days,
                is_b2b=is_b2b,
                games_last_7_days=games_last_7,
                minutes_last_3_games=minutes_last_3,
                fouls=0,
                is_home=is_home,
                recent_form=1.0,
                injury_status=p.get('play_probability', 1.0),
                starter=p.get('is_starter', False)
            )
            contexts[p.get('name', '')] = context
        
        return contexts
    
    def _create_game_context(self, team_a: str, team_b: str, date: str = None) -> GameContext:
        """Create game context for simulation."""
        return GameContext(
            quarter=1,
            time_remaining=720.0,
            home_score=0,
            away_score=0,
            possession='home',
            home_timeouts=7,
            away_timeouts=7,
            home_fouls_qtr=0,
            away_fouls_qtr=0,
            is_overtime=False,
            rest_days_home=2,
            rest_days_away=2
        )
    
    def _predict_player_stats(
        self,
        lineup: List[dict],
        team: str,
        vs_team: str
    ) -> Dict[str, Dict[str, float]]:
        """Predict stats for each player in lineup."""
        predictions = {}
        
        defense_factors = {}
        if self.defense_analyzer:
            for p in lineup:
                pos = p.get('position', 'SG')
                defense_factors[p.get('name', '')] = self.defense_analyzer.get_matchup_adjustment(
                    pos, vs_team, 'pts'
                )
        
        for p in lineup:
            name = p.get('name', '')
            prob = p.get('play_probability', 1.0)
            proj_min = p.get('projected_minutes', 20) * prob
            
            pts_base = p.get('pts_avg', 10)
            reb_base = p.get('reb_avg', 4)
            ast_base = p.get('ast_avg', 2)
            
            def_adj = defense_factors.get(name, 1.0)
            
            predictions[name] = {
                'pts': pts_base * def_adj * (proj_min / 36),
                'reb': reb_base * (proj_min / 36),
                'ast': ast_base * (proj_min / 36),
                'pts_std': pts_base * 0.4,
                'reb_std': reb_base * 0.4,
                'ast_std': ast_base * 0.5,
                'min': proj_min,
                'fg_pct': p.get('fg_pct', 0.47),
                'fg3_pct': p.get('fg3_pct', 0.36),
                'ft_pct': p.get('ft_pct', 0.78),
                'reb_rate': reb_base / 10,
                'ast_rate': ast_base / 5,
                'tov_rate': 0.12,
                'position': p.get('position', 'SG'),
                'team': team
            }
        
        return predictions
    
    def _calculate_matchup_pace(self, team_a_stats: dict, team_b_stats: dict) -> float:
        """Calculate expected pace for matchup."""
        if team_a_stats and team_b_stats:
            if self.bref_scraper:
                try:
                    return self.bref_scraper.get_matchup_pace(
                        team_a_stats.get('team', 'UNK'),
                        team_b_stats.get('team', 'UNK')
                    )
                except:
                    pass
            
            pace_a = team_a_stats.get('pace', 100)
            pace_b = team_b_stats.get('pace', 100)
            return (pace_a + pace_b) / 2
        
        return 100.0
    
    def _stats_to_efficiency(self, stats: dict):
        """Convert team stats to efficiency object."""
        from src.simulation.four_factors_engine import TeamEfficiency, FourFactors
        
        if not stats:
            return TeamEfficiency(
                offensive_rating=114.0,
                defensive_rating=114.0,
                pace=100.0,
                net_rating=0.0,
                four_factors_off=FourFactors(0.54, 0.135, 0.25, 0.23),
                four_factors_def=FourFactors(0.54, 0.135, 0.25, 0.23)
            )
        
        return TeamEfficiency(
            offensive_rating=stats.get('offensive_rating', 114.0),
            defensive_rating=stats.get('defensive_rating', 114.0),
            pace=stats.get('pace', 100.0),
            net_rating=stats.get('offensive_rating', 114.0) - stats.get('defensive_rating', 114.0),
            four_factors_off=FourFactors(
                stats.get('efg_pct', 0.54),
                stats.get('tov_pct', 0.135),
                stats.get('orb_pct', 0.25),
                stats.get('ft_rate', 0.23)
            ),
            four_factors_def=FourFactors(0.54, 0.135, 0.25, 0.23)
        )
    
    def _run_possession_simulation(
        self,
        lineup_a: List[dict],
        lineup_b: List[dict],
        predictions_a: Dict[str, Dict[str, float]],
        predictions_b: Dict[str, Dict[str, float]],
        team_a_stats: dict,
        team_b_stats: dict,
        pace: float,
        num_sims: int
    ) -> Dict[str, Any]:
        """Run possession-by-possession simulation."""
        results_list = []
        
        home_eff = self._stats_to_efficiency(team_a_stats)
        away_eff = self._stats_to_efficiency(team_b_stats)
        
        home_or = home_eff.offensive_rating
        home_dr = home_eff.defensive_rating
        away_or = away_eff.offensive_rating
        away_dr = away_eff.defensive_rating
        
        for sim_idx in range(min(num_sims, 100)):
            sim_result = self.possession_sim.simulate_game(
                home_roster=[{
                    'name': p.get('name', f'Player_{i}'),
                    'team': predictions_a.get(p.get('name', ''), {}).get('team', 'HOME'),
                    'position': p.get('position', 'SG'),
                    'usage': predictions_a.get(p.get('name', ''), {}).get('usage', 0.15),
                    'fg_pct': predictions_a.get(p.get('name', ''), {}).get('fg_pct', 0.47),
                    'fg3_pct': predictions_a.get(p.get('name', ''), {}).get('fg3_pct', 0.36),
                    'ft_pct': predictions_a.get(p.get('name', ''), {}).get('ft_pct', 0.78),
                    'reb_rate': predictions_a.get(p.get('name', ''), {}).get('reb_rate', 0.05),
                    'ast_rate': predictions_a.get(p.get('name', ''), {}).get('ast_rate', 0.15),
                    'tov_rate': predictions_a.get(p.get('name', ''), {}).get('tov_rate', 0.12),
                    'projected_minutes': p.get('projected_minutes', 20)
                } for i, p in enumerate(lineup_a)],
                away_roster=[{
                    'name': p.get('name', f'Player_{i}'),
                    'team': predictions_b.get(p.get('name', ''), {}).get('team', 'AWAY'),
                    'position': p.get('position', 'SG'),
                    'usage': predictions_b.get(p.get('name', ''), {}).get('usage', 0.15),
                    'fg_pct': predictions_b.get(p.get('name', ''), {}).get('fg_pct', 0.47),
                    'fg3_pct': predictions_b.get(p.get('name', ''), {}).get('fg3_pct', 0.36),
                    'ft_pct': predictions_b.get(p.get('name', ''), {}).get('ft_pct', 0.78),
                    'reb_rate': predictions_b.get(p.get('name', ''), {}).get('reb_rate', 0.05),
                    'ast_rate': predictions_b.get(p.get('name', ''), {}).get('ast_rate', 0.15),
                    'tov_rate': predictions_b.get(p.get('name', ''), {}).get('tov_rate', 0.12),
                    'projected_minutes': p.get('projected_minutes', 20)
                } for i, p in enumerate(lineup_b)],
                home_pace=pace,
                away_pace=pace,
                home_off_rating=home_or,
                away_off_rating=away_or,
                home_def_rating=home_dr,
                away_def_rating=away_dr
            )
            results_list.append(sim_result)
        
        return self._aggregate_simulation_results(results_list, lineup_a, lineup_b, predictions_a, predictions_b)
    
    def _run_fast_simulation(
        self,
        lineup_a: List[dict],
        lineup_b: List[dict],
        predictions_a: Dict[str, Dict[str, float]],
        predictions_b: Dict[str, Dict[str, float]],
        team_a_stats: dict,
        team_b_stats: dict,
        pace: float,
        num_sims: int
    ) -> Dict[str, Any]:
        """Run fast vectorized simulation."""
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        
        team_assignments = {}
        for p in lineup_a:
            team_assignments[p.get('name', '')] = 'HOME'
        for p in lineup_b:
            team_assignments[p.get('name', '')] = 'AWAY'
        
        all_predictions = {**predictions_a, **predictions_b}
        
        if self.use_correlations:
            correlated_samples = self.correlation_engine.apply_correlations(
                all_predictions, team_assignments, num_samples=num_sims, seed=42
            )
        else:
            correlated_samples = self._generate_uncorrelated_samples(
                all_predictions, num_sims, rng
            )
        
        home_pts_samples = []
        home_reb_samples = []
        home_ast_samples = []
        away_pts_samples = []
        away_reb_samples = []
        away_ast_samples = []
        
        player_results = {}
        
        for name in predictions_a:
            pts = np.array(correlated_samples[name]['pts'])
            reb = np.array(correlated_samples[name]['reb'])
            ast = np.array(correlated_samples[name]['ast'])
            
            player_results[name] = {
                'pts': pts,
                'reb': reb,
                'ast': ast,
                'team': 'HOME'
            }
            home_pts_samples.append(pts)
            home_reb_samples.append(reb)
            home_ast_samples.append(ast)
        
        for name in predictions_b:
            pts = np.array(correlated_samples[name]['pts'])
            reb = np.array(correlated_samples[name]['reb'])
            ast = np.array(correlated_samples[name]['ast'])
            
            player_results[name] = {
                'pts': pts,
                'reb': reb,
                'ast': ast,
                'team': 'AWAY'
            }
            away_pts_samples.append(pts)
            away_reb_samples.append(reb)
            away_ast_samples.append(ast)
        
        home_pts = np.sum(home_pts_samples, axis=0)
        home_reb = np.sum(home_reb_samples, axis=0)
        home_ast = np.sum(home_ast_samples, axis=0)
        away_pts = np.sum(away_pts_samples, axis=0)
        away_reb = np.sum(away_reb_samples, axis=0)
        away_ast = np.sum(away_ast_samples, axis=0)
        
        team_totals = self.correlation_engine.get_team_constraints(all_predictions, team_assignments)
        
        home_pts_adj = np.clip(home_pts, 85, 160)
        away_pts_adj = np.clip(away_pts, 85, 160)
        
        return {
            'win_prob_a': float(np.mean(home_pts > away_pts)),
            'team_summaries': {
                'HOME': {
                    'pts': {'mean': float(np.mean(home_pts_adj)), 'std': float(np.std(home_pts_adj))},
                    'reb': {'mean': float(np.mean(home_reb)), 'std': float(np.std(home_reb))},
                    'ast': {'mean': float(np.mean(home_ast)), 'std': float(np.std(home_ast))}
                },
                'AWAY': {
                    'pts': {'mean': float(np.mean(away_pts_adj)), 'std': float(np.std(away_pts_adj))},
                    'reb': {'mean': float(np.mean(away_reb)), 'std': float(np.std(away_reb))},
                    'ast': {'mean': float(np.mean(away_ast)), 'std': float(np.std(away_ast))}
                }
            },
            'player_averages': [
                {
                    'name': name,
                    'team': data['team'],
                    'pts': float(np.mean(data['pts'])),
                    'reb': float(np.mean(data['reb'])),
                    'ast': float(np.mean(data['ast'])),
                    'pts_std': float(np.std(data['pts'])),
                    'reb_std': float(np.std(data['reb'])),
                    'ast_std': float(np.std(data['ast']))
                }
                for name, data in player_results.items()
            ],
            'simulations': [
                {
                    'HOME': {'pts': float(home_pts_adj[i]), 'reb': float(home_reb[i]), 'ast': float(home_ast[i])},
                    'AWAY': {'pts': float(away_pts_adj[i]), 'reb': float(away_reb[i]), 'ast': float(away_ast[i])}
                }
                for i in range(min(100, len(home_pts_adj)))
            ]
        }
    
    def _generate_uncorrelated_samples(
        self,
        predictions: Dict[str, Dict[str, float]],
        num_sims: int,
        rng: np.random.Generator
    ) -> Dict[str, Dict[str, List[float]]]:
        """Generate uncorrelated samples for fast path."""
        samples = {}
        for name, preds in predictions.items():
            samples[name] = {
                'pts': (rng.normal(preds['pts'], preds.get('pts_std', preds['pts'] * 0.4), num_sims) * preds.get('play_probability', 1.0)).tolist(),
                'reb': (rng.normal(preds['reb'], preds.get('reb_std', preds['reb'] * 0.4), num_sims) * preds.get('play_probability', 1.0)).tolist(),
                'ast': (rng.normal(preds['ast'], preds.get('ast_std', preds['ast'] * 0.5), num_sims) * preds.get('play_probability', 1.0)).tolist()
            }
        return samples
    
    def _aggregate_simulation_results(
        self,
        results_list: List[dict],
        lineup_a: List[dict],
        lineup_b: List[dict],
        predictions_a: Dict[str, Dict[str, float]],
        predictions_b: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Aggregate possession simulation results."""
        if not results_list:
            return {'error': 'No simulation results'}
        
        home_pts = [r['home_pts'] for r in results_list]
        away_pts = [r['away_pts'] for r in results_list]
        home_reb = [r.get('home_orb', 0) + r.get('home_drb', 0) for r in results_list]
        away_reb = [r.get('away_orb', 0) + r.get('away_drb', 0) for r in results_list]
        home_ast = [r['home_ast'] for r in results_list]
        away_ast = [r['away_ast'] for r in results_list]
        
        player_aggregates = {}
        for r in results_list:
            for p in r.get('player_stats', {}).get('home', []):
                name = p['name']
                if name not in player_aggregates:
                    player_aggregates[name] = {'pts': [], 'reb': [], 'ast': [], 'team': 'HOME'}
                player_aggregates[name]['pts'].append(p['pts'])
                player_aggregates[name]['reb'].append(p['reb'])
                player_aggregates[name]['ast'].append(p['ast'])
            
            for p in r.get('player_stats', {}).get('away', []):
                name = p['name']
                if name not in player_aggregates:
                    player_aggregates[name] = {'pts': [], 'reb': [], 'ast': [], 'team': 'AWAY'}
                player_aggregates[name]['pts'].append(p['pts'])
                player_aggregates[name]['reb'].append(p['reb'])
                player_aggregates[name]['ast'].append(p['ast'])
        
        player_averages = [
            {
                'name': name,
                'team': data['team'],
                'pts': float(np.mean(data['pts'])),
                'reb': float(np.mean(data['reb'])),
                'ast': float(np.mean(data['ast'])),
                'pts_std': float(np.std(data['pts'])),
                'reb_std': float(np.std(data['reb'])),
                'ast_std': float(np.std(data['ast']))
            }
            for name, data in player_aggregates.items()
        ]
        
        return {
            'win_prob_a': float(np.mean(np.array(home_pts) > np.array(away_pts))),
            'team_summaries': {
                'HOME': {
                    'pts': {'mean': float(np.mean(home_pts)), 'std': float(np.std(home_pts))},
                    'reb': {'mean': float(np.mean(home_reb)), 'std': float(np.std(home_reb))},
                    'ast': {'mean': float(np.mean(home_ast)), 'std': float(np.std(home_ast))}
                },
                'AWAY': {
                    'pts': {'mean': float(np.mean(away_pts)), 'std': float(np.std(away_pts))},
                    'reb': {'mean': float(np.mean(away_reb)), 'std': float(np.std(away_reb))},
                    'ast': {'mean': float(np.mean(away_ast)), 'std': float(np.std(away_ast))}
                }
            },
            'player_averages': player_averages,
            'simulations': [
                {
                    'HOME': {'pts': home_pts[i], 'reb': home_reb[i], 'ast': home_ast[i]},
                    'AWAY': {'pts': away_pts[i], 'reb': away_reb[i], 'ast': away_ast[i]}
                }
                for i in range(min(100, len(home_pts)))
            ],
            'possession_results': True
        }
    
    def prepare_simulation_context(self):
        """Prepare shared simulation context."""
        self.load_context()
        if self.all_merged_with_features is None:
            logger.info("Preparing shared simulation context...")
            from src.preprocessing.data_loader import DataLoader
            loader = DataLoader(
                self.manager.data_dir + '/nba_players.csv',
                self.manager.data_dir + '/nba_games.csv'
            )
            self.merged_data = loader.merge_datasets()
            self.all_merged_with_features = self.manager.feature_engineer.create_features(self.merged_data)
            self.latest_player_stats = self.all_merged_with_features.sort_values('GAME_DATE').groupby('PLAYER_ID').tail(1)
            logger.info("Shared context prepared.")


import pandas as pd