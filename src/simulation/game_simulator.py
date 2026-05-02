import logging
import json
import os
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats as scipy_stats
from functools import lru_cache
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from src.preprocessing.data_loader import DataLoader
from src.models.model_manager import ModelManager
from src.data.injury_scraper import InjuryScraper
from src.utils.team_mappings import normalize_team
from src.models.gpu_utils import get_device
from src.data.lineup_scraper import LineupScraper
from src.data.nba_defense_scraper import NBADefenseScraper, DefensiveMatchupAnalyzer
from src.models.minutes_predictor import MinutesPredictor
from src.models.error_calibration import ErrorCalibrator
from src.data.betting_scraper import BettingScraper
from src.data.schedule_scraper import ScheduleScraper
from src.config.config import get_config
from src.simulation.four_factors_engine import FourFactorsEngine
from src.simulation.game_context_engine import (
    ContextAwareAdjuster,
    GameContext,
    PlayerContext,
)
from src.simulation.player_correlation_engine import PlayerCorrelationEngine
from src.simulation.input_health import build_input_health, summarize_input_health

logger = logging.getLogger(__name__)


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


class GameSimulator:
    """
    Advanced Monte Carlo Game Simulator with GPU acceleration.
    
    Features:
    - Defensive matchup adjustments (position-specific)
    - Vegas line calibration for team totals
    - Rest day / back-to-back fatigue modeling
    - Pace-adjusted team totals using actual team pace data
    - Overtime simulation for close games
    - 6-stat correlation matrix (PTS/REB/AST/STL/BLK/TOV)
    - Dynamic home court advantage scaled by team strength
    - Blowout/garbage time minutes redistribution
    """
    
    def __init__(self, manager: ModelManager, cache_dir='data/sim_cache', config: Optional[Any] = None):
        self._config = config if config else get_config()
        simulation_cfg = getattr(self._config, 'simulation', None)
        self.manager = manager
        self.players_df = None
        self.games_df = None
        self.injury_scraper = InjuryScraper(config=self._config)
        self.merged_data = None
        self.all_merged_with_features = None
        self.latest_player_stats = None
        
        self.lineup_scraper = LineupScraper(config=self._config)
        self.defense_scraper = NBADefenseScraper(config=self._config)
        self.defense_analyzer = DefensiveMatchupAnalyzer(self.defense_scraper)
        self.minutes_predictor = MinutesPredictor()
        self.error_calibrator = ErrorCalibrator()
        self.betting_scraper = BettingScraper(config=self._config)
        self.schedule_scraper = ScheduleScraper(config=self._config)
        self.context_adjuster = ContextAwareAdjuster()
        self.context_engine = self.context_adjuster.context_engine
        self.four_factors_engine = FourFactorsEngine()
        self.simulation_seed = int(getattr(simulation_cfg, 'seed', 42)) if simulation_cfg else 42
        self.use_gpu = bool(getattr(simulation_cfg, 'use_gpu', True)) if simulation_cfg else True
        
        if self.use_gpu:
            self.device = get_device() or torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.correlation_engine = PlayerCorrelationEngine(str(self.device))
        logger.info(f"GameSimulator initialized on device: {self.device}")
        
        self._init_simulation_params()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._roster_cache = {}
        self._synergy_cache = {}
        self._defense_cache = {}
        self._pace_cache = {}
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _init_simulation_params(self):
        """Initialize simulation parameters from config."""
        params = self._config.simulation_params if hasattr(self._config, 'simulation_params') else None
        
        corr_matrix = params.stat_correlation_matrix if params else None
        if corr_matrix is None:
            corr_matrix = [
                [1.0, 0.35, 0.45, 0.15, 0.08, 0.20],
                [0.35, 1.0, 0.25, 0.12, 0.18, 0.10],
                [0.45, 0.25, 1.0, 0.18, 0.06, 0.28],
                [0.15, 0.12, 0.18, 1.0, 0.22, 0.35],
                [0.08, 0.18, 0.06, 0.22, 1.0, 0.12],
                [0.20, 0.10, 0.28, 0.35, 0.12, 1.0],
            ]
        
        self.CORR_MATRIX = np.array(corr_matrix, dtype=np.float32)
        self.COV_CHOLESKY = np.linalg.cholesky(self.CORR_MATRIX)
        self.NUM_STATS = 6
        self.STAT_NAMES = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        
        self.league_avg_pts = self._get_config_value('league_averages.points_per_100', 114.0)
        self.vegas_weight = self._get_config_value('league_averages.vegas_weight', 0.30)
        self.home_edge = self._get_config_value('simulation_params.home_edge', 2.5)
        self.four_factors_weight = self._get_config_value('simulation.four_factors_weight', 0.25)
        self.use_four_factors = self._get_config_value('simulation.use_four_factors', True)
        self.use_betting_calibration = self._get_config_value('simulation.use_betting_calibration', True)
        self.use_context_engine = self._get_config_value('simulation.use_context_engine', True)
        self.use_player_correlations = self._get_config_value('simulation.use_player_correlations', True)
        self.use_minutes_model = self._get_config_value('simulation.use_minutes_model', True)
        self.use_error_calibration = self._get_config_value('simulation.use_error_calibration', True)
        self.detailed_path_threshold = int(self._get_config_value('simulation.detailed_path_threshold', 250))
        self.fast_path_threshold = int(self._get_config_value('simulation.fast_path_threshold', 1000))
        self.overtime_margin_threshold = self._get_config_value('simulation_params.overtime_margin_threshold', 3.0)
        self.overtime_home_win_prob = self._get_config_value('simulation_params.overtime_home_win_prob', 0.48)
        self.clutch_score_threshold = self._get_config_value('simulation_params.clutch_score_threshold', 118.0)
        self.correlation_noise_intensity = self._get_config_value('simulation_params.noise_intensity', 0.25)
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get config value using dot notation."""
        parts = key.split('.')
        obj = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj
    
    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_for_cache(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, np.ndarray) or (
            hasattr(data, 'detach') and hasattr(data, 'cpu') and hasattr(data, 'numpy')
        ):
            return {'__type__': 'array', 'data': data.tolist()}
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, dict):
            return {k: self._serialize_for_cache(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_cache(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            return {'__type__': 'dataframe', 'data': data.to_dict(orient='records')}
        return data
    
    def _deserialize_from_cache(self, data: Any) -> Any:
        """Convert JSON data back to original types."""
        if isinstance(data, dict):
            if data.get('__type__') == 'array':
                return np.array(data['data'])
            elif data.get('__type__') == 'dataframe':
                return pd.DataFrame(data['data'])
            return {k: self._deserialize_from_cache(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_from_cache(item) for item in data]
        return data

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from disk cache if available (JSON format for security)."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._deserialize_from_cache(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to disk cache (JSON format for security)."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            serializable_data = self._serialize_for_cache(data)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to save cache {cache_key}: {e}")
        
    def load_context(self):
        """Loads the raw data to extract rosters and recent stats."""
        if self.players_df is None:
            self.players_df = pd.read_csv(os.path.join(self.manager.data_dir, 'nba_players.csv'))
            self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
            self.games_df = pd.read_csv(os.path.join(self.manager.data_dir, 'nba_games.csv'))
            self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])

    def get_available_teams(self) -> List[str]:
        self.load_context()
        return sorted(self.games_df['TEAM_ABBREVIATION'].unique().tolist())

    def prepare_simulation_context(self):
        self.load_context()
        if self.all_merged_with_features is None:
            logger.info("Preparing shared simulation context...")
            loader = DataLoader(
                os.path.join(self.manager.data_dir, 'nba_players.csv'),
                os.path.join(self.manager.data_dir, 'nba_games.csv')
            )
            self.merged_data = loader.merge_datasets()
            self.all_merged_with_features = self.manager.feature_engineer.create_features(self.merged_data)
            self.latest_player_stats = self.all_merged_with_features.sort_values('GAME_DATE').groupby('PLAYER_ID').tail(1)
            logger.info("Shared context prepared.")

    def _compute_mode(self, values: np.ndarray) -> float:
        """Compute the mode using KDE for continuous, histogram for discrete."""
        values = np.array(values)
        if len(values) < 3: return float(np.mean(values))
        values = values[np.isfinite(values)]
        if len(values) < 3: return float(np.mean(values))
        
        rounded = np.round(values * 2) / 2
        if np.allclose(values, rounded, atol=0.1):
            bins = np.arange(values.min() - 0.25, values.max() + 0.75, 0.5)
            if len(bins) < 2: return float(np.mean(values))
            hist, bin_edges = np.histogram(values, bins=bins)
            mode_idx = np.argmax(hist)
            return float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)
        
        try:
            kde = scipy_stats.gaussian_kde(values)
            x_grid = np.linspace(values.min(), values.max(), 200)
            densities = kde(x_grid)
            mode_idx = np.argmax(densities)
            return float(x_grid[mode_idx])
        except Exception:
            return float(np.median(values))

    def _build_roster_context(
        self, 
        team: str, 
        opponent: str, 
        is_home: bool,
        injury_probs: Dict[str, float],
        lineup_data: Optional[dict] = None,
        game_date: Optional[str] = None,
        rest_info: Optional[dict] = None
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], List[Dict]]:
        """
        Builds full roster context for batch prediction with caching.
        """
        # Check cache first
        lineup_key = tuple(sorted((lineup_data or {}).get('starters', []))) if lineup_data else ()
        rest_key = tuple(sorted((rest_info or {}).items())) if rest_info else ()
        cache_key = self._get_cache_key(team, opponent, is_home, game_date, lineup_key, rest_key, tuple(sorted(injury_probs.items())))
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        max_date = self.all_merged_with_features['GAME_DATE'].max()
        recent_cutoff = max_date - pd.Timedelta(days=30)
        
        team_players = self.latest_player_stats[
            self.latest_player_stats['TEAM_ABBREVIATION'] == team
        ]
        
        recent_players = team_players[
            (team_players['GAME_DATE'] >= recent_cutoff) &
            (team_players['MIN'] >= 5)
        ].copy()
        
        if 'ROLL_MIN_AVG_10' in recent_players.columns:
            recent_players = recent_players[recent_players['ROLL_MIN_AVG_10'] >= 8]

        active_players = recent_players.nlargest(12, 'MIN')
        if len(active_players) < 5:
            active_players = team_players.nlargest(12, 'MIN')

        if active_players.empty:
            return pd.DataFrame(), {}, []

        contexts = []
        histories_map = {}
        roster_info = []
        lineup_starters = set()
        if lineup_data and isinstance(lineup_data, dict):
            lineup_starters = {str(name).strip() for name in lineup_data.get('starters', []) if name}
        team_rest = rest_info or self._get_team_rest_days(team, game_date)
        opp_pace = self._get_team_pace(opponent)
        opp_def_rating = self._get_team_efficiency_snapshot(opponent).get('defensive_rating', 114.0)
        coach_tightness = self.minutes_predictor.get_coach_tendency(team) if self.minutes_predictor else 0.5

        for _, player_row in active_players.iterrows():
            pid = player_row['PLAYER_ID']
            pname = player_row['PLAYER_NAME']
            play_prob = injury_probs.get(pname, 1.0)
            is_starter = pname in lineup_starters
            
            history = self.all_merged_with_features[
                self.all_merged_with_features['PLAYER_ID'] == pid
            ].sort_values('GAME_DATE').tail(60)
            
            if len(history) < 3: continue
            
            histories_map[pid] = history
            context = player_row.to_frame().T.copy()
            context['IS_HOME'] = 1 if is_home else 0
            context['OPPONENT_ABBR'] = opponent
            
            opp_team_row = self.games_df[self.games_df['TEAM_ABBREVIATION'] == opponent].tail(1)
            context['OPPONENT_ID'] = opp_team_row['TEAM_ID'].values[0] if not opp_team_row.empty else -1
            
            contexts.append(context)
            
            usage = float(player_row.get('USAGE_PROXY_10', 0.15))
            usage = float(np.clip(usage, 0.03, 0.38))
            
            exp_min = float(player_row.get('ROLL_MIN_AVG_10', player_row.get('MIN', 20)))
            exp_min = float(np.clip(exp_min, 5.0, 42.0))
            if is_starter:
                exp_min = float(np.clip(exp_min + 1.5, 5.0, 42.0))
                play_prob = float(np.clip(max(play_prob, 0.85), 0.0, 1.0))

            minutes_std = float(player_row.get('ROLL_MIN_STD_10', 5.0))
            if self.use_minutes_model and self.minutes_predictor is not None:
                game_context = {
                    'is_home': is_home,
                    'is_b2b': bool(team_rest.get('is_b2b', False)),
                    'rest_days': int(team_rest.get('rest_days', 2)),
                    'fatigue_score': float(np.clip(1.0 - (team_rest.get('rest_days', 2) / 8.0), 0.0, 1.0)),
                    'opp_pace': opp_pace,
                    'opp_def_rating': opp_def_rating,
                    'game_importance': 0.5 if game_date else 0.45,
                    'coach_rotation_tightness': coach_tightness,
                    'injury_risk': float(np.clip(1.0 - play_prob, 0.0, 1.0)),
                    'starter_probability': 0.9 if is_starter else 0.35,
                }
                try:
                    pred_min, pred_std = self.minutes_predictor.predict_minutes(player_row, game_context)
                    exp_min = float(np.clip(0.65 * exp_min + 0.35 * pred_min, 5.0, 42.0))
                    minutes_std = float(max(minutes_std, pred_std))
                except Exception as e:
                    logger.debug(f"Minutes prediction failed for {pname}: {e}")

            position = self._infer_position(player_row)
            
            roster_info.append({
                'id': pid, 'name': pname, 'usage': usage,
                'exp_min': exp_min, 'play_probability': play_prob,
                'min_std': minutes_std,
                'position': position,
                'is_starter': is_starter,
            })
            
        if not contexts: 
            empty_result = (pd.DataFrame(), {}, [])
            return empty_result
        
        result = (pd.concat(contexts, ignore_index=True), histories_map, roster_info)
        
        # Cache the result
        self._save_to_cache(cache_key, result)
        
        return result

    def _get_team_rest_days(self, team: str, game_date: str = None) -> dict:
        """Calculate rest days and back-to-back status from schedule data."""
        try:
            if game_date is None:
                game_date = datetime.now().strftime('%Y-%m-%d')
            
            target_date = pd.to_datetime(game_date)
            
            if self.games_df is not None:
                team_games = self.games_df[
                    (self.games_df['TEAM_ABBREVIATION'] == team) &
                    (self.games_df['GAME_DATE'] < target_date)
                ].sort_values('GAME_DATE')
                
                if len(team_games) >= 1:
                    last_game = team_games['GAME_DATE'].iloc[-1]
                    rest_days = (target_date - last_game).days
                    
                    games_last_7 = len(team_games[team_games['GAME_DATE'] >= target_date - pd.Timedelta(days=7)])
                    games_last_14 = len(team_games[team_games['GAME_DATE'] >= target_date - pd.Timedelta(days=14)])
                    
                    return {
                        'rest_days': rest_days,
                        'is_b2b': rest_days <= 1,
                        'is_3_in_4': games_last_7 >= 3 and rest_days <= 1,
                        'games_last_7': games_last_7,
                        'games_last_14': games_last_14,
                    }
        except Exception as e:
            logger.debug(f"Rest day calculation failed for {team}: {e}")
        
        return {'rest_days': 2, 'is_b2b': False, 'is_3_in_4': False, 'games_last_7': 3, 'games_last_14': 6}

    def _get_rest_fatigue_multiplier(self, rest_info: dict) -> float:
        """Convert rest days into a performance multiplier."""
        rest = rest_info['rest_days']
        
        if rest_info.get('is_3_in_4'):
            return self._get_config_value('simulation_params.fatigue_3in4', 0.96)
        elif rest_info.get('is_b2b'):
            return self._get_config_value('simulation_params.fatigue_back_to_back', 0.975)
        elif rest == 2:
            return self._get_config_value('simulation_params.fatigue_2_days', 1.0)
        elif rest == 3:
            return self._get_config_value('simulation_params.fatigue_3_days', 1.01)
        elif rest >= 4:
            return self._get_config_value('simulation_params.fatigue_4plus_days', 1.005)
        return 1.0

    def _get_defensive_adjustments(self, opponent: str, roster_info: list) -> dict:
        """Get position-specific defensive adjustments for each player."""
        adjustments = {}
        try:
            opp_defense = self.defense_scraper.get_team_defense_allowed(opponent)
            if not opp_defense:
                return adjustments
            
            league_avg_pts = self.league_avg_pts
            opp_pts_allowed = opp_defense.get('pts_allowed_per_100', league_avg_pts)
            
            team_def_factor = opp_pts_allowed / league_avg_pts
            
            pts_range = self._get_config_value('simulation_params.defense_pts_range', (0.85, 1.15))
            reb_range = self._get_config_value('simulation_params.defense_reb_range', (0.90, 1.10))
            ast_range = self._get_config_value('simulation_params.defense_ast_range', (0.92, 1.08))
            tov_range = self._get_config_value('simulation_params.defense_tov_range', (0.90, 1.10))
            
            for player in roster_info:
                position = player.get('position', 'SF')
                try:
                    pos_defense = self.defense_scraper.get_position_defense(opponent, position)
                    if pos_defense:
                        pos_pts = pos_defense.get('pts_allowed', 0)
                        pos_avg = pos_defense.get('league_avg', pos_pts)
                        if pos_avg > 0 and pos_pts > 0:
                            pos_factor = pos_pts / pos_avg
                            adj = 0.5 * team_def_factor + 0.5 * pos_factor
                        else:
                            adj = team_def_factor
                    else:
                        adj = team_def_factor
                except Exception:
                    adj = team_def_factor
                
                adjustments[player['name']] = {
                    'pts': float(np.clip(adj, pts_range[0], pts_range[1])),
                    'reb': float(np.clip(1.0 + (adj - 1.0) * 0.5, reb_range[0], reb_range[1])),
                    'ast': float(np.clip(1.0 + (adj - 1.0) * 0.3, ast_range[0], ast_range[1])),
                    'stl': 1.0,
                    'blk': 1.0,
                    'tov': float(np.clip(2.0 - adj, tov_range[0], tov_range[1])),
                }
        except Exception as e:
            logger.debug(f"Defensive adjustment failed for opponent {opponent}: {e}")
        
        return adjustments

    def _get_team_pace(self, team: str) -> float:
        """Get team's pace (possessions per 48 minutes) from recent data."""
        if team in self._pace_cache:
            return self._pace_cache[team]
        
        pace = 100.0
        try:
            if self.games_df is not None:
                team_games = self.games_df[self.games_df['TEAM_ABBREVIATION'] == team].tail(20)
                if 'PACE' in team_games.columns and len(team_games) > 0:
                    pace = float(team_games['PACE'].mean())
                elif len(team_games) > 5:
                    if 'PTS' in team_games.columns:
                        avg_pts = team_games['PTS'].mean()
                        pace = float(np.clip(avg_pts / 1.12, 90, 110))
        except Exception:
            pass
        
        self._pace_cache[team] = pace
        return pace

    def _calibrate_with_vegas(self, team_totals_mean: dict, team_a: str, team_b: str, betting_lines: dict) -> dict:
        """Blend model predictions with Vegas implied totals (30/70 split)."""
        if not betting_lines or betting_lines.get('total') is None:
            return team_totals_mean
        
        vegas_total = betting_lines.get('total', 0)
        vegas_spread = betting_lines.get('spread', 0)
        
        if vegas_total <= 0:
            return team_totals_mean
        
        vegas_home = (vegas_total - vegas_spread) / 2
        vegas_away = (vegas_total + vegas_spread) / 2
        
        model_home = team_totals_mean.get(team_a, 110)
        model_away = team_totals_mean.get(team_b, 108)
        
        vegas_weight = self.vegas_weight
        
        calibrated = {
            team_a: model_home * (1 - vegas_weight) + vegas_home * vegas_weight,
            team_b: model_away * (1 - vegas_weight) + vegas_away * vegas_weight,
        }
        
        logger.debug(f"Vegas calibration: model=[{model_home:.1f}, {model_away:.1f}] "
                     f"vegas=[{vegas_home:.1f}, {vegas_away:.1f}] "
                     f"final=[{calibrated[team_a]:.1f}, {calibrated[team_b]:.1f}]")
        
        return calibrated

    def _infer_position(self, player_row) -> str:
        """Infer player position from stats."""
        reb = float(player_row.get('ROLL_REB_AVG_10', player_row.get('REB', 4)))
        ast = float(player_row.get('ROLL_AST_AVG_10', player_row.get('AST', 2)))
        blk = float(player_row.get('ROLL_BLK_AVG_10', player_row.get('BLK', 0.5)))
        
        if reb >= 8 and blk >= 1.5:
            return 'C'
        elif reb >= 7:
            return 'PF'
        elif ast >= 6:
            return 'PG'
        elif ast >= 3:
            return 'SG'
        return 'SF'

    def _build_player_projection(self, pinfo: Dict, pred_row: pd.Series, context_row: pd.Series,
                                  def_adj: dict = None) -> Dict:
        """Build projection dict with defensive matchup adjustments."""
        def fallback(stat):
            for col in (f'ROLL_{stat}_AVG_10', f'ROLL_{stat}_AVG_20', f'{stat}_EWMA_5', stat):
                v = context_row.get(col, np.nan)
                if pd.notna(v) and float(v) > 0: return float(v)
            return 0.0
        
        means = {}
        stds = {}
        default_cv = {'PTS': 0.45, 'REB': 0.40, 'AST': 0.50, 'STL': 0.80, 'BLK': 0.90, 'TOV': 0.60}
        min_std = {'PTS': 1.0, 'REB': 0.5, 'AST': 0.5, 'STL': 0.3, 'BLK': 0.3, 'TOV': 0.3}
        
        for stat in self.STAT_NAMES:
            m = pred_row.get(stat, np.nan)
            if np.isnan(m) or m <= 0:
                m = fallback(stat)
            means[stat] = max(0.0, float(m))
            
            s = pred_row.get(f'{stat}_STD', context_row.get(f'ROLL_{stat}_STD_10', np.nan))
            if pd.isna(s) or float(s) <= 0:
                s = max(min_std[stat], default_cv[stat] * means[stat])
            stds[stat] = max(min_std[stat], float(s))
        
        if def_adj:
            player_adj = def_adj.get(pinfo['name'], {})
            for stat in self.STAT_NAMES:
                adj_key = stat.lower()
                if adj_key in player_adj:
                    means[stat] *= player_adj[adj_key]
        
        proj = {
            'id': pinfo['id'], 'name': pinfo['name'], 'usage': pinfo['usage'], 
            'exp_min': pinfo['exp_min'], 'play_probability': pinfo['play_probability'],
            'position': pinfo.get('position', 'SF'),
        }
        for stat in self.STAT_NAMES:
            proj[f'mean_{stat.lower()}'] = means[stat]
            proj[f'std_{stat.lower()}'] = stds[stat]
        
        return proj

    def _get_matchup_seed(self, team_a: str, team_b: str, num_sims: int, seed: Optional[int] = None) -> int:
        """Build a deterministic seed for a matchup."""
        base_seed = self.simulation_seed if seed is None else int(seed)
        payload = f"{base_seed}:{team_a}:{team_b}:{num_sims}"
        digest = hashlib.md5(payload.encode('utf-8')).hexdigest()
        return (base_seed + int(digest[:8], 16)) % (2**31 - 1)

    def _seed_random_generators(self, seed: int) -> Tuple[np.random.Generator, Any]:
        """Seed NumPy and Torch so the simulation is reproducible."""
        np.random.seed(seed)
        np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return np_rng, None

    def _safe_get_game_lines(self, team_a: str, team_b: str, game_date: Optional[str] = None) -> dict:
        """Fetch betting lines and expose health metadata."""
        try:
            lines = self.betting_scraper.get_game_lines(team_a, team_b, game_date)
            health = self.betting_scraper.get_last_fetch_status()
            if not health:
                health = build_input_health(
                    'betting',
                    'success' if lines.get('total') is not None else 'fallback',
                    required=False,
                    message=f"Retrieved betting context for {team_b} @ {team_a}",
                    details={'source': lines.get('source')},
                )
            return {'data': lines, 'health': health}
        except Exception as e:
            logger.warning(f"Betting line fetch failed for {team_b} @ {team_a}: {e}")
            return {
                'data': {
                    'home_team': team_a,
                    'away_team': team_b,
                    'total': None,
                    'spread': None,
                    'source': 'fallback',
                },
                'health': build_input_health(
                    'betting',
                    'failed',
                    required=False,
                    message=f"Betting context failed for {team_b} @ {team_a}",
                    details={'error': str(e)},
                ),
            }

    def _safe_get_lineup(self, team: str, game_date: Optional[str] = None) -> dict:
        """Fetch lineups and expose health metadata."""
        try:
            lineup = self.lineup_scraper.get_starting_lineup(team, game_date)
            payload = lineup if isinstance(lineup, dict) else {}
            health = self.lineup_scraper.get_last_fetch_status()
            if not health:
                health = build_input_health(
                    f'lineup_{team.lower()}',
                    'success' if payload.get('starters') else 'fallback',
                    required=False,
                    message=f"Retrieved lineup context for {team}",
                    details={'source': payload.get('source')},
                )
            else:
                health['source_key'] = f"lineup_{team.lower()}"
            return {'data': payload, 'health': health}
        except Exception as e:
            logger.warning(f"Lineup fetch failed for {team}: {e}")
            return {
                'data': {},
                'health': build_input_health(
                    f'lineup_{team.lower()}',
                    'failed',
                    required=False,
                    message=f"Lineup context failed for {team}",
                    details={'error': str(e)},
                ),
            }

    def _safe_get_injury_probs(self, team: str) -> dict:
        """Fetch injury probabilities and expose health metadata."""
        try:
            probs = self.injury_scraper.get_player_availability(team)
            payload = probs if isinstance(probs, dict) else {}
            health = self.injury_scraper.get_last_fetch_status()
            if not health:
                health = build_input_health(
                    f'injury_{team.lower()}',
                    'success',
                    required=False,
                    message=f"Retrieved injury context for {team}",
                    details={'players_listed': len(payload)},
                )
            else:
                health['source_key'] = f"injury_{team.lower()}"
                health.setdefault('details', {})
                health['details']['players_listed'] = len(payload)
            return {'data': payload, 'health': health}
        except Exception as e:
            logger.warning(f"Injury fetch failed for {team}: {e}")
            return {
                'data': {},
                'health': build_input_health(
                    f'injury_{team.lower()}',
                    'failed',
                    required=False,
                    message=f"Injury context failed for {team}",
                    details={'error': str(e)},
                ),
            }

    def _safe_get_defensive_adjustments(self, opponent: str, roster_info: list) -> dict:
        """Fetch defensive adjustments and expose health metadata."""
        try:
            adjustments = self._get_defensive_adjustments(opponent, roster_info)
            return {
                'data': adjustments,
                'health': build_input_health(
                    f'defense_{opponent.lower()}',
                    'success' if adjustments else 'fallback',
                    required=False,
                    message=(
                        f"Applied defensive adjustments for opponent {opponent}"
                        if adjustments else
                        f"Defensive adjustments unavailable for opponent {opponent}"
                    ),
                    details={
                        'opponent': opponent,
                        'adjustments_applied': bool(adjustments),
                        'players_adjusted': len(adjustments),
                    },
                ),
            }
        except Exception as e:
            logger.warning(f"Defensive adjustment fetch failed for opponent {opponent}: {e}")
            return {
                'data': {},
                'health': build_input_health(
                    f'defense_{opponent.lower()}',
                    'failed',
                    required=False,
                    message=f"Defensive adjustment lookup failed for opponent {opponent}",
                    details={'error': str(e), 'adjustments_applied': False},
                ),
            }

    def _create_game_context(
        self,
        team_a: str,
        team_b: str,
        rest_a: dict,
        rest_b: dict,
        betting_lines: dict,
        lineup_a: dict,
        lineup_b: dict
    ) -> GameContext:
        """Create a pregame context snapshot for both teams."""
        home_timeouts = 7
        away_timeouts = 7
        return GameContext(
            quarter=1,
            time_remaining=720.0,
            home_score=0,
            away_score=0,
            possession='home',
            home_timeouts=home_timeouts,
            away_timeouts=away_timeouts,
            home_fouls_qtr=0,
            away_fouls_qtr=0,
            is_overtime=False,
            rest_days_home=int(rest_a.get('rest_days', 2)),
            rest_days_away=int(rest_b.get('rest_days', 2)),
        )

    def _get_team_efficiency_snapshot(self, team: str) -> dict:
        """Build a team efficiency snapshot from historical game logs."""
        self.load_context()
        if self.games_df is None or self.games_df.empty:
            return {}

        team_games = self.games_df[self.games_df['TEAM_ABBREVIATION'] == team].sort_values('GAME_DATE').tail(20)
        if team_games.empty:
            return {}

        def _mean_if(col: str, default: float) -> float:
            return float(team_games[col].mean()) if col in team_games.columns else default

        pace = _mean_if('PACE', 100.0)
        pts = _mean_if('PTS', 114.0)
        opp_pts = _mean_if('OPP_PTS', 114.0)
        fga = _mean_if('FGA', 88.0)
        fgm = _mean_if('FGM', 41.0)
        fg3m = _mean_if('FG3M', 13.0)
        fta = _mean_if('FTA', 22.0)
        tov = _mean_if('TOV', 14.0)
        orb = _mean_if('OREB', 10.0)
        opp_drb = _mean_if('OPP_DREB', 35.0)

        efg_pct = ((fgm + 0.5 * fg3m) / max(fga, 1.0))
        tov_pct = tov / max(fga + 0.44 * fta + tov, 1.0)
        orb_pct = orb / max(orb + opp_drb, 1.0)
        ft_rate = fta / max(fga, 1.0)

        return {
            'team': team,
            'pace': pace,
            'offensive_rating': float(np.clip(pts / max(pace, 1.0) * 100.0, 95.0, 130.0)),
            'defensive_rating': float(np.clip(opp_pts / max(pace, 1.0) * 100.0, 95.0, 130.0)),
            'efg_pct': float(np.clip(efg_pct, 0.40, 0.65)),
            'tov_pct': float(np.clip(tov_pct, 0.08, 0.20)),
            'orb_pct': float(np.clip(orb_pct, 0.15, 0.35)),
            'ft_rate': float(np.clip(ft_rate, 0.12, 0.35)),
        }

    def _prepare_roster_projection(
        self,
        roster: List[Dict],
        pred_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        def_adj: dict,
    ) -> List[Dict]:
        """Convert model predictions into simulation-ready roster projections."""
        projections = []
        for idx, pinfo in enumerate(roster):
            pred_row = pred_df.iloc[idx] if idx < len(pred_df) else pd.Series(dtype=float)
            ctx_row = ctx_df.iloc[idx] if idx < len(ctx_df) else pd.Series(dtype=float)
            projections.append(self._build_player_projection(pinfo, pred_row, ctx_row, def_adj))
        return projections

    def _apply_context_adjustments(
        self,
        roster: List[Dict],
        is_home: bool,
        rest_info: dict,
        game_context: GameContext
    ) -> List[Dict]:
        """Apply game-context adjustments to roster projections."""
        player_contexts = {}
        predictions = {}
        for player in roster:
            player_contexts[player['name']] = PlayerContext(
                name=player['name'],
                rest_days=int(rest_info.get('rest_days', 2)),
                is_b2b=bool(rest_info.get('is_b2b', False)),
                games_last_7_days=int(rest_info.get('games_last_7', 3)),
                minutes_last_3_games=float(player.get('exp_min', 20.0)) * 3.0,
                fouls=0,
                is_home=is_home,
                recent_form=1.0,
                injury_status=float(player.get('play_probability', 1.0)),
                starter=bool(player.get('is_starter', False)),
            )
            predictions[player['name']] = {
                'pts': float(player.get('mean_pts', 0.0)),
                'reb': float(player.get('mean_reb', 0.0)),
                'ast': float(player.get('mean_ast', 0.0)),
                'stl': float(player.get('mean_stl', 0.0)),
                'blk': float(player.get('mean_blk', 0.0)),
                'tov': float(player.get('mean_tov', 0.0)),
                'min': float(player.get('exp_min', 20.0)),
            }

        adjusted = self.context_adjuster.adjust_predictions(predictions, player_contexts, game_context)

        for player in roster:
            adj = adjusted.get(player['name'], {})
            for stat in self.STAT_NAMES:
                key = f'mean_{stat.lower()}'
                if stat.lower() in adj:
                    player[key] = max(0.0, float(adj[stat.lower()]))
            if 'min' in adj:
                player['exp_min'] = float(np.clip(adj['min'], 5.0, 48.0))

        return roster

    def _apply_error_calibration(self, roster: List[Dict]) -> List[Dict]:
        """Apply historical error calibration when available."""
        if not self.use_error_calibration or self.error_calibrator is None:
            return roster

        for player in roster:
            player_id = player.get('id', 0)
            for stat in self.STAT_NAMES:
                mean_key = f'mean_{stat.lower()}'
                std_key = f'std_{stat.lower()}'
                try:
                    calibrated, std_adjust = self.error_calibrator.get_calibrated_prediction(
                        player_id,
                        stat,
                        float(player.get(mean_key, 0.0)),
                        player.get('name'),
                    )
                    player[mean_key] = calibrated
                    if std_adjust > 0:
                        player[std_key] = float(max(float(player.get(std_key, 0.0)), std_adjust))
                except Exception as e:
                    logger.debug(f"Calibration skipped for {player.get('name')}/{stat}: {e}")

        return roster

    def _build_team_target_means(
        self,
        team_a: str,
        team_b: str,
        roster_a: List[Dict],
        roster_b: List[Dict],
        betting_lines: dict,
        team_a_eff: dict,
        team_b_eff: dict,
    ) -> Dict[str, Dict[str, float]]:
        """Build blended target means for each team."""
        model_targets = {
            team_a: {
                'pts': float(sum(p.get('mean_pts', 0.0) for p in roster_a)),
                'reb': float(sum(p.get('mean_reb', 0.0) for p in roster_a)),
                'ast': float(sum(p.get('mean_ast', 0.0) for p in roster_a)),
            },
            team_b: {
                'pts': float(sum(p.get('mean_pts', 0.0) for p in roster_b)),
                'reb': float(sum(p.get('mean_reb', 0.0) for p in roster_b)),
                'ast': float(sum(p.get('mean_ast', 0.0) for p in roster_b)),
            },
        }

        if self.use_four_factors and team_a_eff and team_b_eff:
            try:
                ff_pred = self.four_factors_engine.predict_matchup(
                    self.four_factors_engine.calculate_efficiency(team_a_eff, team_b_eff),
                    self.four_factors_engine.calculate_efficiency(team_b_eff, team_a_eff),
                    num_samples=500,
                )
                ff_targets = {
                    team_a: {'pts': float(ff_pred['home_pts_mean'])},
                    team_b: {'pts': float(ff_pred['away_pts_mean'])},
                }
                model_targets[team_a]['pts'] = 0.75 * model_targets[team_a]['pts'] + 0.25 * ff_targets[team_a]['pts']
                model_targets[team_b]['pts'] = 0.75 * model_targets[team_b]['pts'] + 0.25 * ff_targets[team_b]['pts']
            except Exception as e:
                logger.debug(f"Four factors calibration skipped: {e}")

        if self.use_betting_calibration and betting_lines and betting_lines.get('total') is not None:
            try:
                model_home = model_targets[team_a]['pts']
                model_away = model_targets[team_b]['pts']
                blended_home, blended_away = self.betting_scraper.blend_with_model(
                    model_home,
                    model_away,
                    team_a,
                    team_b,
                )
                model_targets[team_a]['pts'] = float(blended_home)
                model_targets[team_b]['pts'] = float(blended_away)
            except Exception as e:
                logger.debug(f"Vegas calibration skipped: {e}")

        return model_targets

    def _normalize_player_samples(
        self,
        player_samples: Dict[str, Dict[str, np.ndarray]],
        team_assignments: Dict[str, str],
        target_means: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Normalize player samples to team target means."""
        if not self.use_player_correlations:
            return player_samples

        try:
            sample_payload = {
                name: {
                    'pts': values['pts'].tolist(),
                    'reb': values['reb'].tolist(),
                    'ast': values['ast'].tolist(),
                }
                for name, values in player_samples.items()
            }
            normalized = self.correlation_engine.normalize_to_team_totals(
                sample_payload,
                team_assignments,
                target_means=target_means,
            )
            for name, values in normalized.items():
                for stat in ['pts', 'reb', 'ast']:
                    player_samples[name][stat] = np.array(values[stat], dtype=float)
        except Exception as e:
            logger.debug(f"Correlation normalization skipped: {e}")

        return player_samples

    def simulate_matchup(
        self,
        team_a: str,
        team_b: str,
        num_sims: int = 100,
        game_date: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        High-fidelity vectorized Monte Carlo simulation with contextual priors.
        """
        if num_sims < 1:
            return {'error': 'num_sims must be at least 1'}

        team_a = normalize_team(team_a)
        team_b = normalize_team(team_b)
        game_date_str = None
        if game_date is not None:
            game_date_str = pd.to_datetime(game_date).strftime('%Y-%m-%d')

        logger.info(f"Simulating {team_b} @ {team_a} ({num_sims} sims)")
        self.prepare_simulation_context()

        matchup_seed = self._get_matchup_seed(team_a, team_b, num_sims, seed)
        np_rng, torch_rng = self._seed_random_generators(matchup_seed)
        apply_detailed_context = num_sims <= self.fast_path_threshold

        betting_result = self._safe_get_game_lines(team_a, team_b, game_date_str)
        lineup_result_a = self._safe_get_lineup(team_a, game_date_str)
        lineup_result_b = self._safe_get_lineup(team_b, game_date_str)
        injury_result_a = self._safe_get_injury_probs(team_a)
        injury_result_b = self._safe_get_injury_probs(team_b)
        betting_lines = betting_result['data']
        lineup_a = lineup_result_a['data']
        lineup_b = lineup_result_b['data']
        injury_probs_a = injury_result_a['data']
        injury_probs_b = injury_result_b['data']

        rest_a = self._get_team_rest_days(team_a, game_date_str)
        rest_b = self._get_team_rest_days(team_b, game_date_str)
        fatigue_a = self._get_rest_fatigue_multiplier(rest_a)
        fatigue_b = self._get_rest_fatigue_multiplier(rest_b)

        pace_a = self._get_team_pace(team_a)
        pace_b = self._get_team_pace(team_b)
        expected_pace = (pace_a + pace_b) / 2.0
        pace_multiplier = expected_pace / 100.0

        game_context = self._create_game_context(team_a, team_b, rest_a, rest_b, betting_lines, lineup_a, lineup_b)

        ctx_a, hist_a, info_a = self._build_roster_context(
            team_a, team_b, True, injury_probs_a, lineup_a, game_date_str, rest_a
        )
        ctx_b, hist_b, info_b = self._build_roster_context(
            team_b, team_a, False, injury_probs_b, lineup_b, game_date_str, rest_b
        )

        if ctx_a.empty or ctx_b.empty:
            return {'error': 'Insufficient roster data'}

        defense_result_a = self._safe_get_defensive_adjustments(team_b, info_a)
        defense_result_b = self._safe_get_defensive_adjustments(team_a, info_b)
        def_adj_a = defense_result_a['data']
        def_adj_b = defense_result_b['data']

        preds_a = self.manager.predict_player_stats_batch(ctx_a, hist_a)
        preds_b = self.manager.predict_player_stats_batch(ctx_b, hist_b)
        if preds_a.empty or preds_b.empty:
            return {'error': 'Model predictions unavailable'}

        roster_a = self._prepare_roster_projection(info_a, preds_a, ctx_a, def_adj_a)
        roster_b = self._prepare_roster_projection(info_b, preds_b, ctx_b, def_adj_b)
        roster_a = self._apply_error_calibration(roster_a)
        roster_b = self._apply_error_calibration(roster_b)

        if apply_detailed_context:
            roster_a = self._apply_context_adjustments(roster_a, True, rest_a, game_context)
            roster_b = self._apply_context_adjustments(roster_b, False, rest_b, game_context)

        team_a_eff = self._get_team_efficiency_snapshot(team_a)
        team_b_eff = self._get_team_efficiency_snapshot(team_b)
        team_targets = self._build_team_target_means(
            team_a,
            team_b,
            roster_a,
            roster_b,
            betting_lines,
            team_a_eff,
            team_b_eff,
        )

        input_health = summarize_input_health([
            betting_result['health'],
            lineup_result_a['health'],
            lineup_result_b['health'],
            injury_result_a['health'],
            injury_result_b['health'],
            defense_result_a['health'],
            defense_result_b['health'],
        ])
        input_health['betting_calibration_applied'] = bool(
            self.use_betting_calibration and betting_lines and betting_lines.get('total') is not None
        )
        input_health['defensive_adjustments_applied'] = bool(def_adj_a or def_adj_b)

        for source in input_health['sources']:
            if source['status'] != 'success':
                logger.warning(
                    "Simulation input degraded for %s @ %s: %s (%s)",
                    team_b,
                    team_a,
                    source['source_key'],
                    source['status'],
                )

        return self._simulate_matchup_reactive(
            team_a,
            team_b,
            num_sims,
            betting_lines,
            lineup_a,
            lineup_b,
            rest_a,
            rest_b,
            pace_a,
            pace_b,
            team_a_eff,
            team_b_eff,
            roster_a,
            roster_b,
            team_targets,
            matchup_seed,
            input_health,
        )


    def _calculate_team_synergy(self, player_ids: List[int]) -> float:
        """Calculate team synergy score with caching."""
        # Sort player IDs for consistent cache keys
        sorted_ids = tuple(sorted(player_ids))
        
        # Check cache
        if sorted_ids in self._synergy_cache:
            return self._synergy_cache[sorted_ids]
        
        synergy_score = 1.0

        # Cache the result
        self._synergy_cache[sorted_ids] = synergy_score
        
        return synergy_score

    def _infer_player_archetype(self, player: Dict) -> str:
        """Infer a coarse archetype from projection shape and position."""
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

    def _get_archetype_profile(self, archetype: str) -> Dict[str, float]:
        """Return volatility and style priors for a player archetype."""
        profiles = {
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
        return profiles.get(archetype, profiles['balanced'])

    def _sample_role_state(
        self,
        player: Dict,
        np_rng: np.random.Generator,
        coach_tightness: float,
        close_game_prob: float,
    ) -> RoleSample:
        """Sample a role state for a player before a simulation run."""
        is_starter = bool(player.get('is_starter', False))
        archetype = self._infer_player_archetype(player)

        if is_starter:
            state_names = ['limited', 'normal', 'expanded', 'starter', 'closer']
            base_probs = np.array([0.08, 0.32, 0.18, 0.26, 0.16], dtype=float)
        else:
            state_names = ['limited', 'normal', 'expanded', 'bench', 'closer']
            base_probs = np.array([0.18, 0.34, 0.18, 0.20, 0.10], dtype=float)

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

        base_probs = np.clip(base_probs, 0.01, None)
        base_probs /= base_probs.sum()
        state = str(np_rng.choice(state_names, p=base_probs))

        state_profiles = {
            'limited': RoleSample('limited', 0.68, 0.84, 0.95, 0.82, 0.90, 0.92, 0.78, 1.10, 0.24, 1.16),
            'normal': RoleSample('normal', 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.10, 1.00),
            'expanded': RoleSample('expanded', 1.12, 1.10, 1.04, 1.10, 0.95, 1.03, 1.08, 0.96, 0.08, 1.08),
            'starter': RoleSample('starter', 1.16, 1.12, 1.03, 1.08, 0.96, 1.02, 1.12, 0.92, 0.06, 1.04),
            'bench': RoleSample('bench', 0.90, 0.96, 1.02, 0.92, 1.06, 1.00, 0.92, 1.06, 0.18, 1.22),
            'closer': RoleSample('closer', 1.10, 1.16, 1.08, 1.14, 0.92, 1.04, 1.24, 0.88, 0.05, 1.10),
            'non-closer': RoleSample('non-closer', 0.86, 0.90, 0.96, 0.82, 1.04, 0.96, 0.82, 1.08, 0.14, 1.12),
        }

        sampled = state_profiles[state]
        if archetype == 'microwave_bench_scorer' and state in {'bench', 'expanded'}:
            sampled = RoleSample(
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
            sampled = RoleSample(
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

    def _build_team_lineup_context(
        self,
        roster: List[Dict],
        lineup_data: Optional[dict],
        coach_tightness: float,
    ) -> Dict[str, float]:
        """Build soft priors for lineup interaction and coach behavior."""
        starter_names = {
            str(name).strip()
            for name in (lineup_data or {}).get('starters', [])
            if name
        }
        if not starter_names:
            starter_names = {str(p['name']).strip() for p in roster if p.get('is_starter', False)}

        if not roster:
            return {
                'usage_boost': 1.0,
                'assist_boost': 1.0,
                'rebound_boost': 1.0,
                'efficiency_boost': 1.0,
                'opp_efficiency_penalty': 1.0,
                'paint_boost': 1.0,
                'shot_volume': 1.0,
                'ft_rate': 1.0,
                'turnover_pressure': 1.0,
                'closing_bonus': 1.0,
                'starter_overlap': 0.0,
                'coach_tightness': float(np.clip(coach_tightness, 0.0, 1.0)),
            }

        primary = max(roster, key=lambda p: float(p.get('usage', 0.0)))
        rebounder = max(roster, key=lambda p: float(p.get('mean_reb', 0.0)))
        rim_protector = max(roster, key=lambda p: float(p.get('mean_blk', 0.0)))
        bench_starters = sum(
            1
            for p in roster
            if str(p['name']).strip() in starter_names and not bool(p.get('is_starter', False))
        )
        starter_overlap = sum(
            1
            for p in roster
            if str(p['name']).strip() in starter_names and bool(p.get('is_starter', False))
        ) / max(len(starter_names), 1)

        primary_out = str(primary['name']).strip() not in starter_names
        rebounder_out = str(rebounder['name']).strip() not in starter_names
        rim_out = str(rim_protector['name']).strip() not in starter_names

        coach_tightness = float(np.clip(coach_tightness, 0.0, 1.0))
        usage_boost = 1.0 + (0.09 if primary_out else 0.0) + 0.03 * bench_starters
        assist_boost = 1.0 + (0.10 if primary_out else 0.0) + 0.03 * coach_tightness
        rebound_boost = 1.0 + (0.10 if rebounder_out else 0.0)
        efficiency_boost = 1.0 + (0.03 if primary_out else 0.0) + 0.02 * coach_tightness
        opp_efficiency_penalty = 1.0 + (0.06 if rim_out else 0.0)
        paint_boost = 1.0 + (0.05 if rim_out else 0.0)
        shot_volume = 1.0 + (0.03 if coach_tightness >= 0.6 else -0.01)
        ft_rate = 1.0 + (0.03 if coach_tightness >= 0.55 else 0.0)
        turnover_pressure = 1.0 + (0.03 if primary_out else 0.0)
        closing_bonus = 1.0 + 0.06 * coach_tightness

        return {
            'usage_boost': float(np.clip(usage_boost, 0.9, 1.25)),
            'assist_boost': float(np.clip(assist_boost, 0.9, 1.25)),
            'rebound_boost': float(np.clip(rebound_boost, 0.9, 1.25)),
            'efficiency_boost': float(np.clip(efficiency_boost, 0.9, 1.18)),
            'opp_efficiency_penalty': float(np.clip(opp_efficiency_penalty, 0.9, 1.18)),
            'paint_boost': float(np.clip(paint_boost, 0.9, 1.20)),
            'shot_volume': float(np.clip(shot_volume, 0.92, 1.10)),
            'ft_rate': float(np.clip(ft_rate, 0.92, 1.10)),
            'turnover_pressure': float(np.clip(turnover_pressure, 0.92, 1.12)),
            'closing_bonus': float(np.clip(closing_bonus, 1.0, 1.18)),
            'starter_overlap': float(np.clip(starter_overlap, 0.0, 1.0)),
            'coach_tightness': float(np.clip(coach_tightness, 0.0, 1.0)),
        }

    def _phase_definitions(self) -> List[PhaseDefinition]:
        """Return the standard phase schedule for a single game."""
        return [
            PhaseDefinition('first_half', 24.0),
            PhaseDefinition('second_half', 18.0),
            PhaseDefinition('clutch', 6.0, clutch_window=True),
            PhaseDefinition('overtime', 5.0, overtime=True),
        ]

    def _sample_game_environment(
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
    ) -> Dict[str, float]:
        """Sample shared game environment uncertainty for a run."""
        model_total = float(team_targets.get(team_a, {}).get('pts', 110.0) + team_targets.get(team_b, {}).get('pts', 108.0))
        vegas_total = betting_lines.get('total')
        vegas_spread = betting_lines.get('spread')

        if vegas_total is not None and float(vegas_total) > 0:
            total_anchor = 0.55 * float(vegas_total) + 0.45 * model_total
        else:
            total_anchor = model_total

        if vegas_spread is not None:
            margin_anchor = -float(vegas_spread)
        else:
            margin_anchor = float(team_targets.get(team_a, {}).get('pts', 110.0) - team_targets.get(team_b, {}).get('pts', 108.0))

        rest_penalty = 0.0
        if rest_a.get('is_b2b'):
            rest_penalty += 0.5
        if rest_b.get('is_b2b'):
            rest_penalty += 0.5

        pace_anchor = float(np.mean([
            float(team_a_eff.get('pace', 100.0)),
            float(team_b_eff.get('pace', 100.0)),
        ]))
        pace_shock = float(np.clip(np_rng.normal(1.0 + (total_anchor / max(model_total, 1.0) - 1.0) * 0.12 - rest_penalty * 0.01, 0.04), 0.88, 1.14))
        total_shock = float(np.clip(np_rng.normal(1.0, 0.055), 0.85, 1.16))
        margin_draw = float(np_rng.normal(margin_anchor, 8.5))
        close_factor = float(np.clip(np.exp(-abs(margin_draw) / 7.0), 0.0, 1.0))
        blowout_factor = float(np.clip(max(0.0, (abs(margin_draw) - 10.0) / 20.0), 0.0, 1.0))
        game_total = float(total_anchor * total_shock)

        return {
            'pace_anchor': pace_anchor,
            'pace_shock': pace_shock,
            'total_anchor': total_anchor,
            'total_shock': total_shock,
            'margin_draw': margin_draw,
            'close_factor': close_factor,
            'blowout_factor': blowout_factor,
            'game_total': game_total,
        }

    def _sample_phase_possessions(
        self,
        np_rng: np.random.Generator,
        team_pace: float,
        phase: PhaseDefinition,
        game_env: Dict[str, float],
        score_diff: float,
    ) -> int:
        """Sample offensive possessions for a team in a specific phase."""
        base_possessions = team_pace * phase.minutes / 48.0 * game_env['pace_shock']
        if phase.clutch_window:
            if abs(score_diff) <= 5:
                base_possessions *= 1.04 + 0.05 * game_env['close_factor']
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

    def _allocate_pool(self, np_rng: np.random.Generator, total: int, weights: np.ndarray) -> np.ndarray:
        """Allocate an integer pool to players using a multinomial draw."""
        total = int(max(0, total))
        if total == 0:
            return np.zeros(len(weights), dtype=int)
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 1e-8, None)
        probs = weights / weights.sum()
        return np_rng.multinomial(total, probs)

    def _simulate_team_phase(
        self,
        np_rng: np.random.Generator,
        roster: List[Dict],
        team_name: str,
        is_home: bool,
        phase: PhaseDefinition,
        current_score_diff: float,
        team_context: Dict[str, float],
        opponent_context: Dict[str, float],
        game_env: Dict[str, float],
        player_totals: Dict[str, Dict[str, Any]],
    ) -> int:
        """Simulate one team across one game phase."""
        if not roster:
            return 0

        phase_possessions = self._sample_phase_possessions(
            np_rng,
            float(team_context.get('pace', 100.0)),
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
        role_states = []

        for player in roster:
            totals = player_totals[player['name']]
            archetype = player['archetype']
            profile = self._get_archetype_profile(archetype)
            role_state: RoleSample = player['role_state']
            base_minutes = float(player['exp_min']) * role_state.minute_multiplier

            if close_game:
                if player.get('is_starter', False) or role_state.state == 'closer':
                    base_minutes *= role_state.close_game_multiplier * team_context.get('closing_bonus', 1.0)
                else:
                    base_minutes *= 0.88
            elif blowout:
                if player.get('is_starter', False):
                    base_minutes *= role_state.blowout_multiplier * 0.82
                else:
                    base_minutes *= 1.12

            if totals['fouls'] >= 5:
                base_minutes *= 0.50
            elif totals['fouls'] >= 4:
                base_minutes *= 0.82

            if float(player.get('play_probability', 1.0)) < 0.85:
                base_minutes *= 0.88

            minutes_weight = max(0.0, base_minutes) * role_state.volatility
            usage_weight = float(player['usage']) * role_state.usage_multiplier * profile['usage_bias'] * team_context['usage_boost']
            if close_game:
                usage_weight *= role_state.close_game_multiplier
            elif blowout:
                usage_weight *= role_state.blowout_multiplier

            shot_weight = minutes_weight * usage_weight * profile['shot_bias'] * team_context['shot_volume']
            assist_weight = minutes_weight * role_state.assist_multiplier * profile['assist_bias'] * team_context['assist_boost']
            rebound_weight = minutes_weight * role_state.rebound_multiplier * profile['rebound_bias'] * team_context['rebound_boost']
            tov_weight = minutes_weight * role_state.turnover_multiplier * profile['turnover_bias'] * team_context['turnover_pressure']
            stl_weight = minutes_weight * profile['defense_bias'] * opponent_context.get('turnover_pressure', 1.0)
            blk_weight = minutes_weight * profile['rim_bias'] * opponent_context.get('paint_boost', 1.0)

            minute_weights.append(minutes_weight)
            usage_weights.append(usage_weight)
            shot_weights.append(shot_weight)
            assist_weights.append(assist_weight)
            rebound_weights.append(rebound_weight)
            tov_weights.append(tov_weight)
            stl_weights.append(stl_weight)
            blk_weights.append(blk_weight)
            profiles.append(profile)
            role_states.append(role_state)

        minute_weights = np.asarray(minute_weights, dtype=float)
        if minute_weights.sum() <= 0:
            minute_weights = np.ones(len(roster), dtype=float)
        minute_share = minute_weights / minute_weights.sum()
        phase_minutes = minute_share * phase_minutes_total

        shot_weights = np.asarray(shot_weights, dtype=float)
        assist_weights = np.asarray(assist_weights, dtype=float)
        rebound_weights = np.asarray(rebound_weights, dtype=float)
        tov_weights = np.asarray(tov_weights, dtype=float)
        stl_weights = np.asarray(stl_weights, dtype=float)
        blk_weights = np.asarray(blk_weights, dtype=float)

        shot_pool = int(np.clip(round(phase_possessions * (0.88 + 0.04 * team_context['efficiency_boost'] + 0.04 * game_env['close_factor'])), 1, 60))
        turnover_pool = int(np.clip(round(phase_possessions * (0.10 + 0.02 * team_context['turnover_pressure'] + 0.02 * game_env['blowout_factor'])), 0, 18))
        fta_pool = int(np.clip(round(phase_possessions * (0.18 + 0.03 * team_context['ft_rate'] + 0.02 * game_env['close_factor'])), 0, 22))
        stl_pool = int(np.clip(round(phase_possessions * (0.03 + 0.01 * opponent_context.get('turnover_pressure', 1.0))), 0, 10))
        blk_pool = int(np.clip(round(phase_possessions * (0.02 + 0.01 * opponent_context.get('opp_efficiency_penalty', 1.0))), 0, 8))

        shot_alloc = self._allocate_pool(np_rng, shot_pool, shot_weights)
        fta_alloc = self._allocate_pool(np_rng, fta_pool, minute_weights * np.array([p['archetype_profile']['ft_pct'] for p in roster], dtype=float))
        tov_alloc = self._allocate_pool(np_rng, turnover_pool, tov_weights)
        stl_alloc = self._allocate_pool(np_rng, stl_pool, stl_weights)
        blk_alloc = self._allocate_pool(np_rng, blk_pool, blk_weights)

        made_fg_total = 0
        player_make_data: List[Dict[str, int]] = []

        for idx, player in enumerate(roster):
            profile = profiles[idx]
            role_state = role_states[idx]
            totals = player_totals[player['name']]
            minutes = float(phase_minutes[idx])
            totals['minutes'] += minutes
            totals['played'] = True

            base_three_rate = profile['three_rate']
            three_rate = float(np.clip(base_three_rate * (0.92 + 0.10 * team_context['usage_boost']) * (1.06 if close_game else 0.94 if blowout else 1.0), 0.02, 0.72))
            fg2_pct = float(np.clip(profile['fg2_pct'] * team_context['efficiency_boost'] * opponent_context.get('paint_boost', 1.0) * role_state.efficiency_multiplier, 0.25, 0.78))
            fg3_pct = float(np.clip(profile['fg3_pct'] * team_context['efficiency_boost'] * opponent_context.get('three_defense', 1.0) * role_state.efficiency_multiplier, 0.18, 0.60))
            ft_pct = float(np.clip(profile['ft_pct'] * team_context['ft_rate'], 0.45, 0.94))

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
                'assist_weight': float(assist_weights[idx]),
                'rebound_weight': float(rebound_weights[idx]),
            })

            foul_mean = max(0.0, minutes / 11.5 * (0.78 + 0.18 * role_state.volatility + 0.08 * float(player['usage'])))
            if blowout and player.get('is_starter', False):
                foul_mean *= 0.88
            if close_game and (player.get('is_starter', False) or role_state.state == 'closer'):
                foul_mean *= 1.06
            foul_draw = int(np_rng.poisson(foul_mean))
            totals['fouls'] = min(6, totals['fouls'] + foul_draw)

        assist_pool = int(np.clip(round(max(0, made_fg_total) * (0.58 + 0.05 * team_context['assist_boost'] + 0.04 * game_env['close_factor'])), 0, 18))
        rebound_pool = int(np.clip(round((shot_pool - made_fg_total) * (0.90 * team_context['rebound_boost']) + fta_pool * 0.20), 0, 25))

        assist_alloc = self._allocate_pool(np_rng, assist_pool, assist_weights)
        rebound_alloc = self._allocate_pool(np_rng, rebound_pool, rebound_weights)

        for idx, player in enumerate(roster):
            totals = player_totals[player['name']]
            totals['ast'] += int(assist_alloc[idx])
            totals['reb'] += int(rebound_alloc[idx])

        team_points = int(sum(player_totals[player['name']]['pts'] for player in roster))
        return team_points

    def _simulate_matchup_reactive(
        self,
        team_a: str,
        team_b: str,
        num_sims: int,
        betting_lines: dict,
        lineup_a: dict,
        lineup_b: dict,
        rest_a: dict,
        rest_b: dict,
        pace_a: float,
        pace_b: float,
        team_a_eff: dict,
        team_b_eff: dict,
        roster_a: List[Dict],
        roster_b: List[Dict],
        team_targets: Dict[str, Dict[str, float]],
        seed: int,
        input_health: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the reactive phase-based simulation."""
        np_rng, _ = self._seed_random_generators(seed)
        phases = self._phase_definitions()

        coach_tightness_a = float(self.minutes_predictor.get_coach_tendency(team_a)) if self.minutes_predictor else 0.5
        coach_tightness_b = float(self.minutes_predictor.get_coach_tendency(team_b)) if self.minutes_predictor else 0.5
        team_context_a = self._build_team_lineup_context(roster_a, lineup_a, coach_tightness_a)
        team_context_b = self._build_team_lineup_context(roster_b, lineup_b, coach_tightness_b)

        base_env = self._sample_game_environment(
            np_rng,
            betting_lines,
            team_targets,
            team_a,
            team_b,
            team_a_eff,
            team_b_eff,
            rest_a,
            rest_b,
        )

        # Blend environmental priors with team matchup context.
        team_context_a = {
            **team_context_a,
            'pace': float(team_a_eff.get('pace', pace_a)),
            'coach_tightness': coach_tightness_a,
            'off_env': float(np.clip(team_targets[team_a]['pts'] / max(self.league_avg_pts, 1.0), 0.82, 1.22)),
            'efficiency_boost': float(np.clip(team_context_a['efficiency_boost'] * (team_a_eff.get('offensive_rating', 114.0) / 114.0), 0.85, 1.20)),
            'paint_boost': float(np.clip(team_context_a['paint_boost'] * (1.0 + (team_b_eff.get('defensive_rating', 114.0) - 114.0) / 260.0), 0.85, 1.20)),
            'three_defense': float(np.clip(1.0 + (114.0 - team_b_eff.get('defensive_rating', 114.0)) / 320.0, 0.85, 1.15)),
            'turnover_pressure': float(np.clip(team_context_a['turnover_pressure'] * (team_b_eff.get('defensive_rating', 114.0) / 114.0), 0.85, 1.20)),
        }
        team_context_b = {
            **team_context_b,
            'pace': float(team_b_eff.get('pace', pace_b)),
            'coach_tightness': coach_tightness_b,
            'off_env': float(np.clip(team_targets[team_b]['pts'] / max(self.league_avg_pts, 1.0), 0.82, 1.22)),
            'efficiency_boost': float(np.clip(team_context_b['efficiency_boost'] * (team_b_eff.get('offensive_rating', 114.0) / 114.0), 0.85, 1.20)),
            'paint_boost': float(np.clip(team_context_b['paint_boost'] * (1.0 + (team_a_eff.get('defensive_rating', 114.0) - 114.0) / 260.0), 0.85, 1.20)),
            'three_defense': float(np.clip(1.0 + (114.0 - team_a_eff.get('defensive_rating', 114.0)) / 320.0, 0.85, 1.15)),
            'turnover_pressure': float(np.clip(team_context_b['turnover_pressure'] * (team_a_eff.get('defensive_rating', 114.0) / 114.0), 0.85, 1.20)),
        }

        roster_a = [
            {
                **player,
                'archetype': self._infer_player_archetype(player),
                'archetype_profile': self._get_archetype_profile(self._infer_player_archetype(player)),
            }
            for player in roster_a
        ]
        roster_b = [
            {
                **player,
                'archetype': self._infer_player_archetype(player),
                'archetype_profile': self._get_archetype_profile(self._infer_player_archetype(player)),
            }
            for player in roster_b
        ]

        player_names = [p['name'] for p in roster_a + roster_b]
        roster_a_names = {p['name'] for p in roster_a}
        roster_b_names = {p['name'] for p in roster_b}
        player_play_prob = {p['name']: float(p.get('play_probability', 1.0)) for p in roster_a + roster_b}
        results = {
            team_a: {stat.lower(): np.zeros(num_sims, dtype=float) for stat in self.STAT_NAMES},
            team_b: {stat.lower(): np.zeros(num_sims, dtype=float) for stat in self.STAT_NAMES},
            'player_stats': {
                name: {
                    'team': team_a if name in roster_a_names else team_b,
                    'played': np.zeros(num_sims, dtype=bool),
                    'play_probability': player_play_prob.get(name, 1.0),
                    'pts': np.zeros(num_sims, dtype=float),
                    'reb': np.zeros(num_sims, dtype=float),
                    'ast': np.zeros(num_sims, dtype=float),
                    'stl': np.zeros(num_sims, dtype=float),
                    'blk': np.zeros(num_sims, dtype=float),
                    'tov': np.zeros(num_sims, dtype=float),
                }
                for name in player_names
            },
        }

        roster_map = {
            team_a: roster_a,
            team_b: roster_b,
        }
        team_context_map = {
            team_a: team_context_a,
            team_b: team_context_b,
        }
        team_pace_map = {
            team_a: pace_a,
            team_b: pace_b,
        }
        opponent_map = {
            team_a: team_b,
            team_b: team_a,
        }

        team_summaries_arrays = {
            team_a: {stat.lower(): np.zeros(num_sims, dtype=float) for stat in self.STAT_NAMES},
            team_b: {stat.lower(): np.zeros(num_sims, dtype=float) for stat in self.STAT_NAMES},
        }

        for sim_idx in range(num_sims):
            game_state_diff = float(base_env['margin_draw'])
            game_points = {team_a: 0, team_b: 0}
            sim_player_totals = {
                name: {'pts': 0, 'reb': 0, 'ast': 0, 'stl': 0, 'blk': 0, 'tov': 0, 'minutes': 0.0, 'fouls': 0, 'played': False}
                for name in player_names
            }

            role_states = {}
            available_masks = {}
            close_prob = float(np.clip(base_env['close_factor'], 0.0, 1.0))
            for team in [team_a, team_b]:
                roster = roster_map[team]
                starter_names = {
                    str(name).strip()
                    for name in (lineup_a if team == team_a else lineup_b).get('starters', [])
                    if name
                }
                if not starter_names:
                    starter_names = {str(p['name']).strip() for p in roster if p.get('is_starter', False)}
                avail = []
                for player in roster:
                    play_prob = float(np.clip(player.get('play_probability', 1.0), 0.0, 1.0))
                    if str(player['name']).strip() in starter_names:
                        play_prob = max(play_prob, 0.82)
                    avail.append(np_rng.random() < play_prob)
                    sampled_role = self._sample_role_state(
                        player,
                        np_rng,
                        float(team_context_map[team].get('coach_tightness', 0.5)),
                        close_prob,
                    )
                    role_states[player['name']] = sampled_role
                    player['role_state'] = sampled_role
                if sum(avail) < 5:
                    top_idx = np.argsort([-float(p.get('play_probability', 1.0)) for p in roster])[:5]
                    for idx in top_idx:
                        avail[idx] = True
                available_masks[team] = np.array(avail, dtype=bool)

            for phase in phases:
                game_state_diff = float(game_points[team_a] - game_points[team_b])
                if phase.overtime and abs(game_state_diff) > self.overtime_margin_threshold:
                    break

                phase_close = phase.clutch_window and abs(game_state_diff) <= 5
                phase_blowout = abs(game_state_diff) >= (20 if phase.clutch_window else 15)

                phase_team_points = {}
                for team in [team_a, team_b]:
                    roster = roster_map[team]
                    active_roster = [player for idx, player in enumerate(roster) if available_masks[team][idx]]
                    if len(active_roster) < 5:
                        active_roster = roster[:]

                    adjusted_context = dict(team_context_map[team])
                    if phase_close:
                        adjusted_context['usage_boost'] *= adjusted_context['closing_bonus']
                        adjusted_context['assist_boost'] *= adjusted_context['closing_bonus']
                        adjusted_context['efficiency_boost'] *= 1.03
                    if phase_blowout:
                        adjusted_context['efficiency_boost'] *= 0.96
                        adjusted_context['shot_volume'] *= 0.94

                    opponent_context = dict(team_context_map[opponent_map[team]])
                    opponent_context['three_defense'] = float(np.clip(opponent_context.get('three_defense', 1.0), 0.85, 1.15))

                    phase_points = self._simulate_team_phase(
                        np_rng,
                        active_roster,
                        team,
                        team == team_a,
                        phase,
                        game_state_diff if team == team_a else -game_state_diff,
                        adjusted_context,
                        opponent_context,
                        base_env,
                        sim_player_totals,
                    )
                    phase_team_points[team] = phase_points

                game_points[team_a] += phase_team_points.get(team_a, 0)
                game_points[team_b] += phase_team_points.get(team_b, 0)
                game_state_diff = float(game_points[team_a] - game_points[team_b])

            if abs(game_state_diff) <= self.overtime_margin_threshold:
                ot_phase = self._phase_definitions()[-1]
                for team in [team_a, team_b]:
                    roster = roster_map[team]
                    active_roster = [player for idx, player in enumerate(roster) if available_masks[team][idx]]
                    adjusted_context = dict(team_context_map[team])
                    adjusted_context['usage_boost'] *= adjusted_context['closing_bonus']
                    adjusted_context['assist_boost'] *= adjusted_context['closing_bonus']
                    adjusted_context['efficiency_boost'] *= 1.03
                    opponent_context = dict(team_context_map[opponent_map[team]])
                    phase_points = self._simulate_team_phase(
                        np_rng,
                        active_roster,
                        team,
                        team == team_a,
                        ot_phase,
                        game_state_diff if team == team_a else -game_state_diff,
                        adjusted_context,
                        opponent_context,
                        base_env,
                        sim_player_totals,
                    )
                    game_points[team] += phase_points
                game_state_diff = float(game_points[team_a] - game_points[team_b])

            results[team_a]['pts'][sim_idx] = float(game_points[team_a])
            results[team_b]['pts'][sim_idx] = float(game_points[team_b])

            for team in [team_a, team_b]:
                roster = roster_map[team]
                for stat in ['reb', 'ast', 'stl', 'blk', 'tov']:
                    values = [sim_player_totals[player['name']][stat] for player in roster]
                    team_summaries_arrays[team][stat][sim_idx] = float(sum(values))
                results[team][ 'reb'][sim_idx] = team_summaries_arrays[team]['reb'][sim_idx]
                results[team][ 'ast'][sim_idx] = team_summaries_arrays[team]['ast'][sim_idx]
                results[team][ 'stl'][sim_idx] = team_summaries_arrays[team]['stl'][sim_idx]
                results[team][ 'blk'][sim_idx] = team_summaries_arrays[team]['blk'][sim_idx]
                results[team][ 'tov'][sim_idx] = team_summaries_arrays[team]['tov'][sim_idx]

            for name, totals in sim_player_totals.items():
                player_record = results['player_stats'][name]
                player_record['played'][sim_idx] = bool(totals['played'])
                player_record['play_probability'] = float(next((p['play_probability'] for p in roster_a + roster_b if p['name'] == name), 1.0))
                player_record['pts'][sim_idx] = float(totals['pts'])
                player_record['reb'][sim_idx] = float(totals['reb'])
                player_record['ast'][sim_idx] = float(totals['ast'])
                player_record['stl'][sim_idx] = float(totals['stl'])
                player_record['blk'][sim_idx] = float(totals['blk'])
                player_record['tov'][sim_idx] = float(totals['tov'])

        player_samples = {
            name: {
                'pts': data['pts'],
                'reb': data['reb'],
                'ast': data['ast'],
            }
            for name, data in results['player_stats'].items()
        }
        player_samples = self._normalize_player_samples(player_samples, {name: data['team'] for name, data in results['player_stats'].items()}, team_targets)
        for name, samples in player_samples.items():
            results['player_stats'][name]['pts'] = samples['pts']
            results['player_stats'][name]['reb'] = samples['reb']
            results['player_stats'][name]['ast'] = samples['ast']

        for team in [team_a, team_b]:
            team_players = [name for name, stats in results['player_stats'].items() if stats['team'] == team]
            for stat in self.STAT_NAMES:
                stat_lower = stat.lower()
                results[team][stat_lower] = np.sum([results['player_stats'][name][stat_lower] for name in team_players], axis=0)

        win_prob_a = float(np.mean(results[team_a]['pts'] > results[team_b]['pts']) * 100.0)
        team_summaries = {}
        for team in [team_a, team_b]:
            summary = {}
            for stat in self.STAT_NAMES:
                vals = results[team][stat.lower()]
                stat_summary = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'mode': self._compute_mode(vals),
                }
                if stat == 'PTS':
                    stat_summary.update({
                        'p0.5': float(np.percentile(vals, 0.5)),
                        'p99.5': float(np.percentile(vals, 99.5)),
                        'p5': float(np.percentile(vals, 5)),
                        'p95': float(np.percentile(vals, 95)),
                    })
                summary[stat.lower()] = stat_summary
            team_summaries[team] = summary

        simulations = []
        for s in range(min(num_sims, 1000)):
            game = {
                team_a: {stat.lower(): float(results[team_a][stat.lower()][s]) for stat in self.STAT_NAMES},
                team_b: {stat.lower(): float(results[team_b][stat.lower()][s]) for stat in self.STAT_NAMES},
                'players': {},
            }
            for name, stats in results['player_stats'].items():
                game['players'][name] = {stat.lower(): float(stats[stat.lower()][s]) for stat in self.STAT_NAMES}
                game['players'][name]['played'] = bool(stats['played'][s])
            simulations.append(game)

        player_averages = []
        for name, stats in results['player_stats'].items():
            played = stats['played']
            pa = {
                'name': name,
                'team': stats['team'],
                'play_probability': stats['play_probability'],
                'games_played_pct': played.mean() * 100,
            }
            for stat in self.STAT_NAMES:
                sl = stat.lower()
                pa[sl] = round(float(stats[sl].mean()), 1)
                pa[f'{sl}_mode'] = round(self._compute_mode(stats[sl][played]) if played.any() else 0, 1)
                pa[f'{sl}_95_ci'] = [round(float(np.percentile(stats[sl], 2.5)), 1), round(float(np.percentile(stats[sl], 97.5)), 1)]
                pa[f'{sl}_99_ci'] = [round(float(np.percentile(stats[sl], 0.5)), 1), round(float(np.percentile(stats[sl], 99.5)), 1)]
                pa[f'{sl}_std'] = round(float(stats[sl].std()), 2)
            player_averages.append(pa)

        return {
            'team_a': team_a,
            'team_b': team_b,
            'win_prob_a': win_prob_a,
            'team_summaries': team_summaries,
            'simulations': simulations,
            'player_averages': player_averages,
            'betting_lines': betting_lines,
            'lineup_a': lineup_a,
            'lineup_b': lineup_b,
            'metadata': {
                'seed': seed,
                'device': str(self.device),
                'simulation_mode': 'high_fidelity',
                'simulation_engine': 'reactive_v2',
                'use_context_engine': True,
                'use_player_correlations': self.use_player_correlations,
                'use_betting_calibration': self.use_betting_calibration,
                'input_health': input_health or summarize_input_health([]),
            },
            'context': {
                'rest_a': rest_a,
                'rest_b': rest_b,
                'pace_a': pace_a,
                'pace_b': pace_b,
                'expected_pace': float(np.mean([pace_a, pace_b])),
                'team_targets': team_targets,
                'game_environment': base_env,
                'lineup_context_a': team_context_a,
                'lineup_context_b': team_context_b,
                'input_overall_status': (input_health or {}).get('overall_status', 'healthy'),
            }
        }
