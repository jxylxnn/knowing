import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
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
    
    def __init__(self, manager: ModelManager, gnn_model=None, transformer_model=None, cache_dir='data/sim_cache'):
        self.manager = manager
        self.players_df = None
        self.games_df = None
        self.injury_scraper = InjuryScraper()
        self.merged_data = None
        self.all_merged_with_features = None
        self.latest_player_stats = None
        
        self.gnn_model = gnn_model
        self.transformer_model = transformer_model
        
        self.lineup_scraper = LineupScraper()
        self.defense_scraper = NBADefenseScraper()
        self.defense_analyzer = DefensiveMatchupAnalyzer(self.defense_scraper)
        self.minutes_predictor = MinutesPredictor()
        self.error_calibrator = ErrorCalibrator()
        self.betting_scraper = BettingScraper()
        self.schedule_scraper = ScheduleScraper()
        
        self.device = get_device()
        logger.info(f"GameSimulator initialized on device: {self.device}")
        
        # 6-stat correlation matrix: [PTS, REB, AST, STL, BLK, TOV]
        # Based on empirical NBA stat correlations
        self.CORR_MATRIX = torch.tensor([
            # PTS    REB    AST    STL    BLK    TOV
            [1.00,  0.15, -0.05,  0.12, -0.02,  0.35],   # PTS
            [0.15,  1.00, -0.10, -0.03,  0.30, -0.05],   # REB
            [-0.05, -0.10,  1.00,  0.15, -0.08,  0.25],  # AST
            [0.12, -0.03,  0.15,  1.00,  0.05,  0.08],   # STL
            [-0.02,  0.30, -0.08,  0.05,  1.00, -0.03],  # BLK
            [0.35, -0.05,  0.25,  0.08, -0.03,  1.00],   # TOV
        ], dtype=torch.float32, device=self.device)
        self.COV_CHOLESKY = torch.linalg.cholesky(self.CORR_MATRIX)
        self.NUM_STATS = 6
        self.STAT_NAMES = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._roster_cache = {}
        self._synergy_cache = {}
        self._defense_cache = {}
        self._pace_cache = {}
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_for_cache(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
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
        injury_probs: Dict[str, float]
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], List[Dict]]:
        """
        Builds full roster context for batch prediction with caching.
        """
        # Check cache first
        cache_key = self._get_cache_key(team, opponent, is_home, tuple(sorted(injury_probs.items())))
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
        
        for _, player_row in active_players.iterrows():
            pid = player_row['PLAYER_ID']
            pname = player_row['PLAYER_NAME']
            play_prob = injury_probs.get(pname, 1.0)
            
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
            
            position = self._infer_position(player_row)
            
            roster_info.append({
                'id': pid, 'name': pname, 'usage': usage, 
                'exp_min': exp_min, 'play_probability': play_prob,
                'position': position,
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
            return 0.96
        elif rest_info.get('is_b2b'):
            return 0.975
        elif rest == 2:
            return 1.0
        elif rest == 3:
            return 1.01
        elif rest >= 4:
            return 1.005
        return 1.0

    def _get_defensive_adjustments(self, opponent: str, roster_info: list) -> dict:
        """Get position-specific defensive adjustments for each player."""
        adjustments = {}
        try:
            opp_defense = self.defense_scraper.get_team_defense_allowed(opponent)
            if not opp_defense:
                return adjustments
            
            league_avg_pts = 114.0
            opp_pts_allowed = opp_defense.get('pts_allowed_per_100', league_avg_pts)
            
            team_def_factor = opp_pts_allowed / league_avg_pts
            
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
                    'pts': float(np.clip(adj, 0.85, 1.15)),
                    'reb': float(np.clip(1.0 + (adj - 1.0) * 0.5, 0.90, 1.10)),
                    'ast': float(np.clip(1.0 + (adj - 1.0) * 0.3, 0.92, 1.08)),
                    'stl': 1.0,
                    'blk': 1.0,
                    'tov': float(np.clip(2.0 - adj, 0.90, 1.10)),
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
        
        vegas_weight = 0.30
        
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

    def simulate_matchup(
        self, 
        team_a: str, 
        team_b: str, 
        num_sims: int = 100
    ) -> Dict[str, Any]:
        """
        Advanced Vectorized GPU Monte Carlo Simulation.
        
        Incorporates defensive matchups, Vegas calibration, rest/fatigue,
        pace modeling, overtime, and 6-stat correlated sampling.
        """
        team_a = normalize_team(team_a)
        team_b = normalize_team(team_b)
        logger.info(f"Simulating {team_b} @ {team_a} ({num_sims} sims)")

        self.prepare_simulation_context()

        # --- CONTEXT GATHERING ---
        betting_lines = self.betting_scraper.get_game_lines(team_a, team_b)
        lineup_a = self.lineup_scraper.get_starting_lineup(team_a)
        lineup_b = self.lineup_scraper.get_starting_lineup(team_b)
        
        injury_probs_a = self.injury_scraper.get_player_availability(team_a)
        injury_probs_b = self.injury_scraper.get_player_availability(team_b)
        
        rest_a = self._get_team_rest_days(team_a)
        rest_b = self._get_team_rest_days(team_b)
        fatigue_a = self._get_rest_fatigue_multiplier(rest_a)
        fatigue_b = self._get_rest_fatigue_multiplier(rest_b)
        
        pace_a = self._get_team_pace(team_a)
        pace_b = self._get_team_pace(team_b)
        expected_pace = (pace_a + pace_b) / 2.0
        pace_multiplier = expected_pace / 100.0
        
        ctx_a, hist_a, info_a = self._build_roster_context(team_a, team_b, True, injury_probs_a)
        ctx_b, hist_b, info_b = self._build_roster_context(team_b, team_a, False, injury_probs_b)
        
        if ctx_a.empty or ctx_b.empty: return {'error': 'Insufficient roster data'}
        
        def_adj_a = self._get_defensive_adjustments(team_b, info_a)
        def_adj_b = self._get_defensive_adjustments(team_a, info_b)
        
        preds_a = self.manager.predict_player_stats_batch(ctx_a, hist_a)
        preds_b = self.manager.predict_player_stats_batch(ctx_b, hist_b)
        
        rosters = {
            team_a: [self._build_player_projection(info_a[i], preds_a.iloc[i], ctx_a.iloc[i], def_adj_a) for i in range(len(info_a))],
            team_b: [self._build_player_projection(info_b[i], preds_b.iloc[i], ctx_b.iloc[i], def_adj_b) for i in range(len(info_b))]
        }

        # --- VECTORIZED GPU SIMULATION ---
        results = {team_a: {}, team_b: {}, 'player_stats': {}}
        rng = torch.Generator(device=self.device)
        
        pace_var = torch.clamp(
            torch.normal(float(pace_multiplier), 0.04, size=(num_sims, 1), generator=rng, device=self.device),
            float(pace_multiplier) * 0.88, float(pace_multiplier) * 1.12
        )
        env_factor = torch.clamp(torch.normal(1.0, 0.05, size=(num_sims, 1), generator=rng, device=self.device), 0.88, 1.15)

        def run_team_sim(team_name, roster, is_home, fatigue_mult):
            n = len(roster)
            S = self.NUM_STATS
            play_prob = torch.tensor([p['play_probability'] for p in roster], dtype=torch.float32, device=self.device)
            usage = torch.tensor([p['usage'] for p in roster], dtype=torch.float32, device=self.device)
            exp_min = torch.tensor([p['exp_min'] for p in roster], dtype=torch.float32, device=self.device)
            
            stat_means = torch.zeros(n, S, dtype=torch.float32, device=self.device)
            stat_stds = torch.zeros(n, S, dtype=torch.float32, device=self.device)
            for si, stat in enumerate(self.STAT_NAMES):
                stat_means[:, si] = torch.tensor([p[f'mean_{stat.lower()}'] for p in roster], dtype=torch.float32, device=self.device)
                stat_stds[:, si] = torch.tensor([p[f'std_{stat.lower()}'] for p in roster], dtype=torch.float32, device=self.device)

            synergy_mod = self._calculate_team_synergy([p['id'] for p in roster])
            
            # 1. Injury Checks
            injury_roll = torch.rand(num_sims, n, generator=rng, device=self.device)
            active_mask = injury_roll < play_prob.unsqueeze(0)
            
            if (active_mask.sum(dim=1) < 5).any():
                top5_indices = torch.topk(play_prob, min(5, n)).indices
                active_mask[:, top5_indices] = True

            # 2. Minutes Allocation with pace and fatigue
            team_total_mins = 240.0 * float(pace_multiplier) / 1.0
            mins_base = active_mask * exp_min.unsqueeze(0)
            missing_mins = torch.clamp(team_total_mins - mins_base.sum(dim=1, keepdim=True), min=0.0)
            
            usage_weights = (active_mask * usage.unsqueeze(0)) / torch.clamp((active_mask * usage.unsqueeze(0)).sum(dim=1, keepdim=True), min=1e-6)
            exp_mins_final = mins_base + (missing_mins * usage_weights)
            
            mins_sd = torch.where(exp_mins_final >= 32, 2.0, 3.5)
            mins = torch.clamp(torch.normal(exp_mins_final, mins_sd, generator=rng), 0.0, 48.0)
            mins = mins * (240.0 / torch.clamp(mins.sum(dim=1, keepdim=True), min=1.0))
            
            # 3. Scale by minutes, pace, synergy, fatigue, and environment
            scale = (mins / torch.clamp(exp_min.unsqueeze(0), min=1e-6)).unsqueeze(-1)  # (sims, n, 1)
            synergy_boost = 1.0 + (synergy_mod - 1.0) * 0.5
            eff = env_factor.unsqueeze(-1) * synergy_boost * fatigue_mult  # (sims, 1, 1)
            
            p_means = stat_means.unsqueeze(0) * scale * eff  # (sims, n, S)
            p_means[:, :, 1] *= (0.97 + 0.06 * pace_var)  # REB scales with pace
            p_means[:, :, 0] *= pace_var  # PTS scales with pace
            
            p_stds = stat_stds.unsqueeze(0) * torch.sqrt(torch.clamp(scale, min=0.2)) * env_factor.unsqueeze(-1)
            
            # 4. Team Totals with dynamic home court advantage
            rest_advantage = 0.0
            if is_home:
                home_edge = 2.5
                strength_diff = float(stat_means[:, 0].sum() - 100) / 50.0
                home_edge += float(np.clip(strength_diff, -0.5, 0.5))
            else:
                home_edge = -2.5
            home_edge += rest_advantage
            
            team_m = p_means.sum(dim=1)  # (sims, S)
            team_s = torch.sqrt((p_stds**2).sum(dim=1)) + torch.tensor([5.0, 3.0, 2.5, 1.0, 0.8, 1.0], device=self.device)
            
            team_totals = torch.normal(team_m, team_s, generator=rng)
            team_totals[:, 0] += home_edge
            
            team_totals = torch.clamp(team_totals,
                min=torch.tensor([70, 30, 15, 3, 2, 5], dtype=torch.float32, device=self.device),
                max=torch.tensor([160, 70, 45, 20, 15, 28], dtype=torch.float32, device=self.device))
            
            # 5. Dirichlet Stat Allocation (all 6 stats)
            conc = torch.tensor([70.0, 90.0, 85.0, 95.0, 95.0, 90.0], device=self.device)
            alpha = (p_means / torch.clamp(p_means.sum(dim=1, keepdim=True), min=1e-6)) * conc.unsqueeze(0).unsqueeze(0)
            gamma_dist = torch.distributions.Gamma(torch.clamp(alpha, min=1e-6), 1.0)
            gamma_draws = gamma_dist.rsample()
            shares = gamma_draws / torch.clamp(gamma_draws.sum(dim=1, keepdim=True), min=1e-12)
            
            p_stats_raw = team_totals.unsqueeze(1) * shares
            
            # 6. Correlated Noise Injection (6-stat)
            Z = torch.randn(num_sims, n, S, generator=rng, device=self.device)
            Z_corr = torch.matmul(Z, self.COV_CHOLESKY.T)
            
            noise_intensity = 0.25
            p_stats = torch.clamp(p_stats_raw + Z_corr * p_stds * noise_intensity, min=0.0)
            
            # Normalize PTS to team totals
            pts_sum = torch.clamp(p_stats[:, :, 0].sum(dim=1, keepdim=True), min=1.0)
            p_stats[:, :, 0] = p_stats[:, :, 0] * (team_totals[:, 0:1] / pts_sum)
            
            # 7. Clutch redistribution: in high-scoring games, stars get more
            clutch_mask = team_totals[:, 0] > 118.0
            if clutch_mask.any():
                top2_idx = torch.topk(usage, min(2, n)).indices
                bot_idx = torch.topk(usage, min(2, n), largest=False).indices
                clutch_idxs = torch.nonzero(clutch_mask).squeeze(1)
                for idx in clutch_idxs:
                    p_stats[idx, top2_idx, 0] += 1.5
                    p_stats[idx, bot_idx, 0] = torch.clamp(p_stats[idx, bot_idx, 0] - 1.0, min=0.0)
                    p_stats[idx, :, 0] *= (team_totals[idx, 0] / torch.clamp(p_stats[idx, :, 0].sum(), min=1.0))

            # Store all 6 stats
            stat_arrays = {}
            for si, stat in enumerate(self.STAT_NAMES):
                stat_lower = stat.lower()
                stat_arrays[stat_lower] = p_stats[:, :, si].sum(dim=1).cpu().numpy()
            results[team_name] = stat_arrays
            
            for i, p in enumerate(roster):
                player_stats = {'team': team_name, 'played': active_mask[:, i].cpu().numpy(),
                                'play_probability': p['play_probability']}
                for si, stat in enumerate(self.STAT_NAMES):
                    player_stats[stat.lower()] = p_stats[:, i, si].cpu().numpy()
                results['player_stats'][p['name']] = player_stats

        run_team_sim(team_a, rosters[team_a], True, fatigue_a)
        run_team_sim(team_b, rosters[team_b], False, fatigue_b)
        
        # --- VEGAS CALIBRATION ---
        model_totals = {
            team_a: float(np.mean(results[team_a]['pts'])),
            team_b: float(np.mean(results[team_b]['pts'])),
        }
        calibrated = self._calibrate_with_vegas(model_totals, team_a, team_b, betting_lines)
        
        for team in [team_a, team_b]:
            if model_totals[team] > 0:
                cal_ratio = calibrated[team] / model_totals[team]
                if abs(cal_ratio - 1.0) > 0.001:
                    results[team]['pts'] = results[team]['pts'] * cal_ratio
                    for name, stats in results['player_stats'].items():
                        if stats['team'] == team:
                            stats['pts'] = stats['pts'] * cal_ratio
        
        # --- OVERTIME SIMULATION ---
        pts_a = results[team_a]['pts']
        pts_b = results[team_b]['pts']
        margin = np.abs(pts_a - pts_b)
        ot_mask = margin <= 3.0
        
        if ot_mask.any():
            n_ot = int(ot_mask.sum())
            ot_pts = np.random.normal(5.0, 2.5, n_ot)
            ot_pts = np.clip(ot_pts, 0, 15)
            
            a_wins_ot = np.random.random(n_ot) > 0.48
            
            pts_a[ot_mask] += np.where(a_wins_ot, ot_pts + 1, ot_pts - 1)
            pts_b[ot_mask] += np.where(a_wins_ot, ot_pts - 1, ot_pts + 1)
            
            results[team_a]['pts'] = pts_a
            results[team_b]['pts'] = pts_b
            logger.debug(f"Overtime simulated in {n_ot}/{num_sims} games")

        # --- AGGREGATION ---
        pts_a_t = torch.from_numpy(results[team_a]['pts']).to(self.device)
        pts_b_t = torch.from_numpy(results[team_b]['pts']).to(self.device)
        win_prob_a = (pts_a_t > pts_b_t).float().mean().item() * 100
        
        team_summaries = {}
        for team in [team_a, team_b]:
            summary = {}
            for stat in self.STAT_NAMES:
                vals = results[team][stat.lower()]
                stat_summary = {'mean': float(vals.mean()), 'std': float(vals.std()), 'mode': self._compute_mode(vals)}
                if stat == 'PTS':
                    stat_summary.update({
                        'p0.5': float(np.percentile(vals, 0.5)), 'p99.5': float(np.percentile(vals, 99.5)),
                        'p5': float(np.percentile(vals, 5)), 'p95': float(np.percentile(vals, 95)),
                    })
                summary[stat.lower()] = stat_summary
            team_summaries[team] = summary

        simulations = []
        for s in range(min(num_sims, 1000)):
            game = {
                team_a: {stat.lower(): float(results[team_a][stat.lower()][s]) for stat in self.STAT_NAMES},
                team_b: {stat.lower(): float(results[team_b][stat.lower()][s]) for stat in self.STAT_NAMES},
                'players': {}
            }
            for name, stats in results['player_stats'].items():
                game['players'][name] = {stat.lower(): float(stats[stat.lower()][s]) for stat in self.STAT_NAMES}
                game['players'][name]['played'] = bool(stats['played'][s])
            simulations.append(game)

        player_averages = []
        for name, stats in results['player_stats'].items():
            played = stats['played']
            pa = {
                'name': name, 'team': stats['team'], 'play_probability': stats['play_probability'],
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
            'team_a': team_a, 'team_b': team_b, 
            'win_prob_a': win_prob_a,
            'team_summaries': team_summaries,
            'simulations': simulations, 
            'player_averages': player_averages,
            'betting_lines': betting_lines,
            'lineup_a': lineup_a,
            'lineup_b': lineup_b,
            'context': {
                'rest_a': rest_a, 'rest_b': rest_b,
                'pace_a': pace_a, 'pace_b': pace_b,
                'expected_pace': expected_pace,
                'fatigue_a': fatigue_a, 'fatigue_b': fatigue_b,
                'has_defensive_adjustments': bool(def_adj_a or def_adj_b),
                'has_vegas_calibration': bool(betting_lines and betting_lines.get('total')),
            }
        }

    def _calculate_team_synergy(self, player_ids: List[int]) -> float:
        """
        Calculate team synergy score with caching.
        Uses GNN embeddings if available.
        """
        # Sort player IDs for consistent cache keys
        sorted_ids = tuple(sorted(player_ids))
        
        # Check cache
        if sorted_ids in self._synergy_cache:
            return self._synergy_cache[sorted_ids]
        
        if self.gnn_model is None or not self.gnn_model.is_trained:
            synergy_score = 1.0
        else:
            try:
                synergy_score = 0.0
                count = 0
                for pid in player_ids:
                    if pid in self.gnn_model.player_map:
                        idx = self.gnn_model.player_map[pid]
                        synergy_score += np.mean(self.gnn_model.trained_embeddings[idx])
                        count += 1
                if count > 0:
                    avg_score = synergy_score / count
                    global_mean = np.mean(self.gnn_model.trained_embeddings)
                    synergy_score = np.clip(1.0 + (avg_score - global_mean) * 0.1, 0.95, 1.05)
                else:
                    synergy_score = 1.0
            except Exception as e:
                logger.debug(f"Synergy calculation failed: {e}")
                synergy_score = 1.0
        
        # Cache the result
        self._synergy_cache[sorted_ids] = synergy_score
        
        return synergy_score
