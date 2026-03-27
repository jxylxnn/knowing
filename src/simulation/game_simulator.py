import logging
import json
import os
import pandas as pd
import numpy as np
import torch
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

logger = logging.getLogger(__name__)

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
        self.injury_scraper = InjuryScraper()
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
        """Fetch betting lines with a safe fallback."""
        try:
            return self.betting_scraper.get_game_lines(team_a, team_b, game_date)
        except Exception as e:
            logger.debug(f"Betting line fetch failed for {team_b} @ {team_a}: {e}")
            return {'home_team': team_a, 'away_team': team_b, 'total': None, 'spread': None, 'source': 'fallback'}

    def _safe_get_lineup(self, team: str, game_date: Optional[str] = None) -> dict:
        """Fetch lineups with a safe fallback."""
        try:
            lineup = self.lineup_scraper.get_starting_lineup(team, game_date)
            return lineup if isinstance(lineup, dict) else {}
        except Exception as e:
            logger.debug(f"Lineup fetch failed for {team}: {e}")
            return {}

    def _safe_get_injury_probs(self, team: str) -> Dict[str, float]:
        """Fetch injury probabilities with a safe fallback."""
        try:
            probs = self.injury_scraper.get_player_availability(team)
            return probs if isinstance(probs, dict) else {}
        except Exception as e:
            logger.debug(f"Injury fetch failed for {team}: {e}")
            return {}

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

        betting_lines = self._safe_get_game_lines(team_a, team_b, game_date_str)
        lineup_a = self._safe_get_lineup(team_a, game_date_str)
        lineup_b = self._safe_get_lineup(team_b, game_date_str)
        injury_probs_a = self._safe_get_injury_probs(team_a)
        injury_probs_b = self._safe_get_injury_probs(team_b)

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

        def_adj_a = self._get_defensive_adjustments(team_b, info_a)
        def_adj_b = self._get_defensive_adjustments(team_a, info_b)

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

        results = {team_a: {}, team_b: {}, 'player_stats': {}}
        player_samples: Dict[str, Dict[str, np.ndarray]] = {}
        team_assignments: Dict[str, str] = {}

        pace_var = np.clip(
            np_rng.normal(float(pace_multiplier), 0.04, size=(num_sims, 1)),
            float(pace_multiplier) * 0.88,
            float(pace_multiplier) * 1.12,
        )
        env_factor = np.clip(
            np_rng.normal(1.0, 0.05, size=(num_sims, 1)),
            0.88,
            1.15,
        )

        def run_team_sim(team_name: str, roster: List[Dict], is_home: bool, fatigue_mult: float, target_means: Dict[str, float]):
            n = len(roster)
            S = self.NUM_STATS
            play_prob = np.asarray([p['play_probability'] for p in roster], dtype=float)
            usage = np.asarray([p['usage'] for p in roster], dtype=float)
            exp_min = np.asarray([p['exp_min'] for p in roster], dtype=float)

            stat_means = np.zeros((n, S), dtype=float)
            stat_stds = np.zeros((n, S), dtype=float)
            for si, stat in enumerate(self.STAT_NAMES):
                stat_means[:, si] = np.asarray([p[f'mean_{stat.lower()}'] for p in roster], dtype=float)
                stat_stds[:, si] = np.asarray([p[f'std_{stat.lower()}'] for p in roster], dtype=float)

            synergy_mod = self._calculate_team_synergy([p['id'] for p in roster])

            injury_roll = np_rng.random((num_sims, n))
            active_mask = injury_roll < play_prob[None, :]
            if (active_mask.sum(axis=1) < 5).any():
                top5_indices = np.argsort(-play_prob)[: min(5, n)]
                active_mask[:, top5_indices] = True

            team_total_mins = 240.0
            mins_base = active_mask.astype(float) * exp_min[None, :]
            missing_mins = np.clip(team_total_mins - mins_base.sum(axis=1, keepdims=True), 0.0, None)

            usage_active = active_mask * usage[None, :]
            usage_weights = usage_active / np.clip(usage_active.sum(axis=1, keepdims=True), 1e-6, None)
            exp_mins_final = mins_base + (missing_mins * usage_weights)

            mins_sd = np.where(exp_mins_final >= 32, 2.0, 3.5)
            mins = np.clip(np_rng.normal(exp_mins_final, mins_sd), 0.0, 48.0)
            mins = mins * (240.0 / np.clip(mins.sum(axis=1, keepdims=True), 1.0, None))

            scale = (mins / np.clip(exp_min[None, :], 1e-6, None))[:, :, None]
            synergy_boost = 1.0 + (synergy_mod - 1.0) * 0.5
            eff = env_factor[:, :, None] * synergy_boost * fatigue_mult

            p_means = stat_means[None, :, :] * scale * eff
            p_means[:, :, 1] *= (0.97 + 0.06 * pace_var)
            p_means[:, :, 0] *= pace_var

            p_stds = stat_stds[None, :, :] * np.sqrt(np.clip(scale, 0.2, None)) * env_factor[:, :, None]

            home_edge = self.home_edge if is_home else -self.home_edge
            if is_home:
                strength_diff = float(stat_means[:, 0].sum() - 100) / 50.0
                home_edge += float(np.clip(strength_diff, -0.5, 0.5))

            team_m = p_means.sum(axis=1)
            team_s = np.sqrt((p_stds**2).sum(axis=1)) + np.array([5.0, 3.0, 2.5, 1.0, 0.8, 1.0], dtype=float)

            team_totals = team_m + np_rng.normal(0.0, 1.0, size=team_m.shape) * team_s
            team_totals[:, 0] += home_edge

            target_pts = float(target_means.get('pts', team_totals[:, 0].mean().item()))
            target_reb = float(target_means.get('reb', team_totals[:, 1].mean().item()))
            target_ast = float(target_means.get('ast', team_totals[:, 2].mean().item()))
            team_totals[:, 0] = team_totals[:, 0] * 0.70 + target_pts * 0.30
            team_totals[:, 1] = team_totals[:, 1] * 0.85 + target_reb * 0.15
            team_totals[:, 2] = team_totals[:, 2] * 0.85 + target_ast * 0.15

            team_totals_min = self._get_config_value('simulation_params.team_totals_min', [70, 30, 15, 3, 2, 5])
            team_totals_max = self._get_config_value('simulation_params.team_totals_max', [160, 70, 45, 20, 15, 28])
            team_totals = np.clip(team_totals, np.array(team_totals_min, dtype=float), np.array(team_totals_max, dtype=float))

            dirichlet_conc = self._get_config_value('simulation_params.dirichlet_concentrations', [70.0, 90.0, 85.0, 95.0, 95.0, 90.0])
            conc = np.asarray(dirichlet_conc, dtype=float)
            alpha = (p_means / np.clip(p_means.sum(axis=1, keepdims=True), 1e-6, None)) * conc[None, None, :]
            gamma_draws = np_rng.gamma(shape=np.clip(alpha, 1e-6, None), scale=1.0)
            shares = gamma_draws / np.clip(gamma_draws.sum(axis=1, keepdims=True), 1e-12, None)

            p_stats_raw = team_totals[:, None, :] * shares
            Z = np_rng.normal(size=(num_sims, n, S))
            Z_corr = np.matmul(Z, self.COV_CHOLESKY.T)
            p_stats = np.clip(p_stats_raw + Z_corr * p_stds * self.correlation_noise_intensity, 0.0, None)

            pts_sum = np.clip(p_stats[:, :, 0].sum(axis=1, keepdims=True), 1.0, None)
            p_stats[:, :, 0] = p_stats[:, :, 0] * (team_totals[:, 0:1] / pts_sum)

            clutch_mask = team_totals[:, 0] > self.clutch_score_threshold
            if clutch_mask.any():
                top2_idx = np.argsort(-usage)[: min(2, n)]
                bot_idx = np.argsort(usage)[: min(2, n)]
                clutch_idxs = np.where(clutch_mask)[0]
                for idx in clutch_idxs:
                    p_stats[idx, top2_idx, 0] += 1.5
                    p_stats[idx, bot_idx, 0] = np.clip(p_stats[idx, bot_idx, 0] - 1.0, 0.0, None)
                    p_stats[idx, :, 0] *= (team_totals[idx, 0] / np.clip(p_stats[idx, :, 0].sum(), 1.0, None))

            stat_arrays = {}
            for si, stat in enumerate(self.STAT_NAMES):
                stat_lower = stat.lower()
                stat_arrays[stat_lower] = p_stats[:, :, si].sum(axis=1)
            results[team_name] = stat_arrays

            for i, p in enumerate(roster):
                player_stats = {
                    'team': team_name,
                    'played': active_mask[:, i],
                    'play_probability': p['play_probability'],
                }
                for si, stat in enumerate(self.STAT_NAMES):
                    player_stats[stat.lower()] = p_stats[:, i, si]
                results['player_stats'][p['name']] = player_stats
                player_samples[p['name']] = {
                    'pts': player_stats['pts'],
                    'reb': player_stats['reb'],
                    'ast': player_stats['ast'],
                }
                team_assignments[p['name']] = team_name

        run_team_sim(team_a, roster_a, True, fatigue_a, team_targets[team_a])
        run_team_sim(team_b, roster_b, False, fatigue_b, team_targets[team_b])

        player_samples = self._normalize_player_samples(player_samples, team_assignments, team_targets)
        for name, samples in player_samples.items():
            results['player_stats'][name]['pts'] = samples['pts']
            results['player_stats'][name]['reb'] = samples['reb']
            results['player_stats'][name]['ast'] = samples['ast']

        for team in [team_a, team_b]:
            team_players = [name for name, stats in results['player_stats'].items() if stats['team'] == team]
            for stat in self.STAT_NAMES:
                stat_lower = stat.lower()
                results[team][stat_lower] = np.sum([results['player_stats'][name][stat_lower] for name in team_players], axis=0)

        pts_a = results[team_a]['pts']
        pts_b = results[team_b]['pts']

        margin = np.abs(pts_a - pts_b)
        ot_mask = margin <= self.overtime_margin_threshold
        if ot_mask.any():
            n_ot = int(ot_mask.sum())
            ot_pts = np_rng.normal(self.four_factors_engine.LEAGUE_AVERAGES['pace'] / 20.0, 2.5, n_ot)
            ot_pts = np.clip(ot_pts, 0, 15)
            a_wins_ot = np_rng.random(n_ot) > self.overtime_home_win_prob
            pts_a[ot_mask] += np.where(a_wins_ot, ot_pts + 1, ot_pts - 1)
            pts_b[ot_mask] += np.where(a_wins_ot, ot_pts - 1, ot_pts + 1)
            results[team_a]['pts'] = pts_a
            results[team_b]['pts'] = pts_b
            logger.debug(f"Overtime simulated in {n_ot}/{num_sims} games")

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
                'seed': matchup_seed,
                'device': str(self.device),
                'simulation_mode': 'high_fidelity' if apply_detailed_context else 'vectorized_fast',
                'use_context_engine': apply_detailed_context,
                'use_player_correlations': self.use_player_correlations,
                'use_betting_calibration': self.use_betting_calibration,
            },
            'context': {
                'rest_a': rest_a,
                'rest_b': rest_b,
                'pace_a': pace_a,
                'pace_b': pace_b,
                'expected_pace': expected_pace,
                'fatigue_a': fatigue_a,
                'fatigue_b': fatigue_b,
                'has_defensive_adjustments': bool(def_adj_a or def_adj_b),
                'has_vegas_calibration': bool(betting_lines and betting_lines.get('total')),
                'team_targets': team_targets,
            }
        }

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
