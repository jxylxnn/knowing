import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
from scipy import stats as scipy_stats
from functools import lru_cache
import hashlib
from pathlib import Path

from src.preprocessing.data_loader import DataLoader
from src.models.model_manager import ModelManager
from src.data.injury_scraper import InjuryScraper
from src.utils.team_mappings import normalize_team
from src.models.gpu_utils import get_device
from src.data.lineup_scraper import LineupScraper
from src.data.nba_defense_scraper import NBADefenseScraper
from src.models.minutes_predictor import MinutesPredictor
from src.models.error_calibration import ErrorCalibrator
from src.data.betting_scraper import BettingScraper

logger = logging.getLogger(__name__)

class GameSimulator:
    """
    Top-Tier Monte Carlo Game Simulator with automatic GPU fallback and caching.
    
    Uses PyTorch vectorization to run thousands of simulations in parallel.
    Automatically falls back to CPU for unsupported GPU architectures.
    Includes intelligent caching for expensive computations.
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
        
        # --- NEW: Advanced Components ---
        self.lineup_scraper = LineupScraper()
        self.defense_scraper = NBADefenseScraper()
        self.minutes_predictor = MinutesPredictor()
        self.error_calibrator = ErrorCalibrator()
        self.betting_scraper = BettingScraper()
        
        # --- GPU ACCELERATION (with Blackwell fallback) ---
        self.device = get_device()
        logger.info(f"GameSimulator initialized on device: {self.device}")
        
        # Correlation Matrix for [PTS, REB, AST]
        self.CORR_MATRIX = torch.tensor([
            [1.00, 0.15, -0.05],
            [0.15, 1.00, -0.10],
            [-0.05, -0.10, 1.00]
        ], dtype=torch.float32, device=self.device)
        self.COV_CHOLESKY = torch.linalg.cholesky(self.CORR_MATRIX)
        
        # --- CACHING SETUP ---
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._roster_cache = {}
        self._synergy_cache = {}
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
            self.players_df = pd.read_csv(self.manager.data_dir + '/nba_players.csv')
            self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
            self.games_df = pd.read_csv(self.manager.data_dir + '/nba_games.csv')
            self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])

    def get_available_teams(self) -> List[str]:
        self.load_context()
        return sorted(self.games_df['TEAM_ABBREVIATION'].unique().tolist())

    def prepare_simulation_context(self):
        self.load_context()
        if self.all_merged_with_features is None:
            logger.info("Preparing shared simulation context...")
            loader = DataLoader(
                self.manager.data_dir + '/nba_players.csv',
                self.manager.data_dir + '/nba_games.csv'
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
        cache_key = self._get_cache_key(team, opponent, is_home)
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
            
            roster_info.append({
                'id': pid, 'name': pname, 'usage': usage, 
                'exp_min': exp_min, 'play_probability': play_prob
            })
            
        if not contexts: 
            empty_result = (pd.DataFrame(), {}, [])
            return empty_result
        
        result = (pd.concat(contexts, ignore_index=True), histories_map, roster_info)
        
        # Cache the result
        self._save_to_cache(cache_key, result)
        
        return result

    def _build_player_projection(self, pinfo: Dict, pred_row: pd.Series, context_row: pd.Series) -> Dict:
        """Helper to build consistent projection dict."""
        def fallback(stat):
            for col in (f'ROLL_{stat}_AVG_10', f'ROLL_{stat}_AVG_20', f'{stat}_EWMA_5', stat):
                v = context_row.get(col, np.nan)
                if pd.notna(v) and float(v) > 0: return float(v)
            return 0.0
        
        m_pts = pred_row.get('PTS', np.nan)
        if np.isnan(m_pts): m_pts = fallback('PTS')
        m_reb = pred_row.get('REB', np.nan)
        if np.isnan(m_reb): m_reb = fallback('REB')
        m_ast = pred_row.get('AST', np.nan)
        if np.isnan(m_ast): m_ast = fallback('AST')
        
        s_pts = pred_row.get('PTS_STD', context_row.get('ROLL_PTS_STD_10', max(2.0, 0.45 * m_pts)))
        s_reb = pred_row.get('REB_STD', context_row.get('ROLL_REB_STD_10', max(1.0, 0.40 * m_reb)))
        s_ast = pred_row.get('AST_STD', context_row.get('ROLL_AST_STD_10', max(1.0, 0.50 * m_ast)))
        
        return {
            'id': pinfo['id'], 'name': pinfo['name'], 'usage': pinfo['usage'], 
            'exp_min': pinfo['exp_min'], 'play_probability': pinfo['play_probability'],
            'mean_pts': max(0.0, float(m_pts)), 'std_pts': max(1.0, float(s_pts)),
            'mean_reb': max(0.0, float(m_reb)), 'std_reb': max(0.5, float(s_reb)),
            'mean_ast': max(0.0, float(m_ast)), 'std_ast': max(0.5, float(s_ast)),
        }

    def simulate_matchup(
        self, 
        team_a: str, 
        team_b: str, 
        num_sims: int = 100
    ) -> Dict[str, Any]:
        """
        Vectorized GPU Monte Carlo Simulation.
        """
        team_a = normalize_team(team_a)
        team_b = normalize_team(team_b)
        logger.info(f"Simulating {team_b} @ {team_a} ({num_sims} sims on GPU)")

        self.prepare_simulation_context()

        # --- NEW: Get betting lines for calibration ---
        betting_lines = self.betting_scraper.get_game_lines(team_a, team_b)
        
        # --- NEW: Get real starting lineups ---
        lineup_a = self.lineup_scraper.get_starting_lineup(team_a)
        lineup_b = self.lineup_scraper.get_starting_lineup(team_b)
        
        # --- NEW: Get opponent defensive matchup factors ---
        matchup_factors_a = {}
        matchup_factors_b = {}
        
        injury_probs_a = self.injury_scraper.get_player_availability(team_a)
        injury_probs_b = self.injury_scraper.get_player_availability(team_b)
        
        ctx_a, hist_a, info_a = self._build_roster_context(team_a, team_b, True, injury_probs_a)
        ctx_b, hist_b, info_b = self._build_roster_context(team_b, team_a, False, injury_probs_b)
        
        if ctx_a.empty or ctx_b.empty: return {'error': 'Insufficient roster data'}
        
        # Batch Projections
        preds_a = self.manager.predict_player_stats_batch(ctx_a, hist_a)
        preds_b = self.manager.predict_player_stats_batch(ctx_b, hist_b)
        
        rosters = {
            team_a: [self._build_player_projection(info_a[i], preds_a.iloc[i], ctx_a.iloc[i]) for i in range(len(info_a))],
            team_b: [self._build_player_projection(info_b[i], preds_b.iloc[i], ctx_b.iloc[i]) for i in range(len(info_b))]
        }

        # --- VECTORIZED GPU SIMULATION ---
        results = {team_a: {}, team_b: {}, 'player_stats': {}}
        rng = torch.Generator(device=self.device)
        rng.manual_seed(42)
        
        # Global Game Factors
        pace_factor = torch.clamp(torch.normal(1.0, 0.05, size=(num_sims, 1), generator=rng, device=self.device), 0.88, 1.15)
        env_factor = torch.clamp(torch.normal(1.0, 0.06, size=(num_sims, 1), generator=rng, device=self.device), 0.85, 1.20)

        def run_team_sim(team_name, roster, is_home):
            n = len(roster)
            play_prob = torch.tensor([p['play_probability'] for p in roster], dtype=torch.float32, device=self.device)
            usage = torch.tensor([p['usage'] for p in roster], dtype=torch.float32, device=self.device)
            exp_min = torch.tensor([p['exp_min'] for p in roster], dtype=torch.float32, device=self.device)
            
            mean_pts = torch.tensor([p['mean_pts'] for p in roster], dtype=torch.float32, device=self.device)
            std_pts = torch.tensor([p['std_pts'] for p in roster], dtype=torch.float32, device=self.device)
            mean_reb = torch.tensor([p['mean_reb'] for p in roster], dtype=torch.float32, device=self.device)
            std_reb = torch.tensor([p['std_reb'] for p in roster], dtype=torch.float32, device=self.device)
            mean_ast = torch.tensor([p['mean_ast'] for p in roster], dtype=torch.float32, device=self.device)
            std_ast = torch.tensor([p['std_ast'] for p in roster], dtype=torch.float32, device=self.device)

            synergy_mod = self._calculate_team_synergy([p['id'] for p in roster])
            
            # 1. Injury Checks
            injury_roll = torch.rand(num_sims, n, generator=rng, device=self.device)
            active_mask = injury_roll < play_prob.unsqueeze(0)
            
            if (active_mask.sum(dim=1) < 5).any():
                top5_indices = torch.topk(play_prob, 5).indices
                active_mask[:, top5_indices] = True

            # 2. Minutes Allocation
            mins_base = active_mask * exp_min.unsqueeze(0)
            missing_mins = torch.clamp(240.0 - mins_base.sum(dim=1, keepdim=True), min=0.0)
            
            usage_weights = (active_mask * usage.unsqueeze(0)) / torch.clamp((active_mask * usage.unsqueeze(0)).sum(dim=1, keepdim=True), min=1e-6)
            exp_mins_final = mins_base + (missing_mins * usage_weights)
            
            mins_sd = torch.where(exp_mins_final >= 32, 2.0, 4.0)
            mins = torch.clamp(torch.normal(exp_mins_final, mins_sd, generator=rng), 0.0, 48.0)
            mins = mins * (240.0 / torch.clamp(mins.sum(dim=1, keepdim=True), min=1.0))
            
            # 3. Scale Means/Stds
            scale = (mins / torch.clamp(exp_min.unsqueeze(0), min=1e-6)).unsqueeze(-1)
            synergy_boost = 1.0 + (synergy_mod - 1.0) * 0.5
            eff = env_factor.unsqueeze(-1) * synergy_boost
            
            p_means = torch.stack([mean_pts, mean_reb, mean_ast], dim=1).unsqueeze(0) * scale * eff
            # Pace impact on REB
            p_means[:, :, 1] *= (0.98 + 0.04 * pace_factor)
            
            p_stds = torch.stack([std_pts, std_reb, std_ast], dim=1).unsqueeze(0) * torch.sqrt(torch.clamp(scale, min=0.2)) * env_factor.unsqueeze(-1)
            
            # 4. Dirichlet Stat Allocation
            # Concentrations for Dirichlet
            conc = torch.tensor([70.0, 90.0, 85.0], device=self.device)
            
            # Alpha for Gamma draws
            alpha = (p_means / torch.clamp(p_means.sum(dim=1, keepdim=True), min=1e-6)) * conc.unsqueeze(0).unsqueeze(0)
            # Note: torch.distributions doesn't support 'generator' arg, use rsample() for reparameterized sampling
            gamma_dist = torch.distributions.Gamma(torch.clamp(alpha, min=1e-6), 1.0)
            gamma_draws = gamma_dist.rsample()
            shares = gamma_draws / torch.clamp(gamma_draws.sum(dim=1, keepdim=True), min=1e-12)
            
            # Team Totals
            home_edge = 1.8 if is_home else -1.8
            team_m = p_means.sum(dim=1)
            team_s = torch.sqrt((p_stds**2).sum(dim=1)) + 5.0
            team_totals = torch.clamp(torch.normal(team_m, team_s, generator=rng), 
                                      min=torch.tensor([70,30,15], device=self.device),
                                      max=torch.tensor([160,70,45], device=self.device))
            team_totals[:, 0] += home_edge
            
            # Raw Allocation
            p_stats_raw = team_totals.unsqueeze(1) * shares
            
            # 5. Correlated Noise Injection
            Z = torch.randn(num_sims, n, 3, generator=rng, device=self.device)
            Z_corr = torch.matmul(Z, self.COV_CHOLESKY.T)
            
            noise_intensity = 0.3
            p_stats = torch.clamp(p_stats_raw + Z_corr * p_stds * noise_intensity, min=0.0)
            
            # Normalize PTS to team totals
            p_stats[:, :, 0] = p_stats[:, :, 0] * (team_totals[:, 0:1] / torch.clamp(p_stats[:, :, 0].sum(dim=1, keepdim=True), min=1.0))
            
            # 6. Clutch Logic
            clutch_mask = team_totals[:, 0] > 115.0
            if clutch_mask.any():
                top2_idx = torch.topk(usage, 2).indices
                bot2_idx = torch.topk(usage, 2, largest=False).indices
                indices = torch.nonzero(clutch_mask).squeeze(1)
                for idx in indices:
                    p_stats[idx, top2_idx, 0] += 1.0
                    p_stats[idx, bot2_idx, 0] = torch.clamp(p_stats[idx, bot2_idx, 0] - 1.0, min=0.0)
                    p_stats[idx, :, 0] *= (team_totals[idx, 0] / torch.clamp(p_stats[idx, :, 0].sum(), min=1.0))

            # Store Stats
            results[team_name]['pts'] = p_stats[:, :, 0].sum(dim=1).cpu().numpy()
            results[team_name]['reb'] = p_stats[:, :, 1].sum(dim=1).cpu().numpy()
            results[team_name]['ast'] = p_stats[:, :, 2].sum(dim=1).cpu().numpy()
            
            for i, p in enumerate(roster):
                results['player_stats'][p['name']] = {
                    'team': team_name, 'pts': p_stats[:, i, 0].cpu().numpy(),
                    'reb': p_stats[:, i, 1].cpu().numpy(), 'ast': p_stats[:, i, 2].cpu().numpy(),
                    'played': active_mask[:, i].cpu().numpy(), 'play_probability': p['play_probability']
                }

        run_team_sim(team_a, rosters[team_a], True)
        run_team_sim(team_b, rosters[team_b], False)
        
        # --- Aggregation & Reporting Hooks ---
        # Pre-calculate win prob on GPU
        pts_a_t = torch.from_numpy(results[team_a]['pts']).to(self.device)
        pts_b_t = torch.from_numpy(results[team_b]['pts']).to(self.device)
        win_prob_a = (pts_a_t > pts_b_t).float().mean().item() * 100
        
        team_summaries = {}
        for team in [team_a, team_b]:
            t_pts, t_reb, t_ast = results[team]['pts'], results[team]['reb'], results[team]['ast']
            team_summaries[team] = {
                'pts': {'mean': float(t_pts.mean()), 'std': float(t_pts.std()), 'mode': self._compute_mode(t_pts),
                        'p0.5': float(np.percentile(t_pts, 0.5)), 'p99.5': float(np.percentile(t_pts, 99.5)),
                        'p5': float(np.percentile(t_pts, 5)), 'p95': float(np.percentile(t_pts, 95))},
                'reb': {'mean': float(t_reb.mean()), 'std': float(t_reb.std()), 'mode': self._compute_mode(t_reb)},
                'ast': {'mean': float(t_ast.mean()), 'std': float(t_ast.std()), 'mode': self._compute_mode(t_ast)}
            }

        simulations = []
        for s in range(min(num_sims, 1000)):
            game = {team_a: {'pts': results[team_a]['pts'][s], 'reb': results[team_a]['reb'][s], 'ast': results[team_a]['ast'][s]},
                    team_b: {'pts': results[team_b]['pts'][s], 'reb': results[team_b]['reb'][s], 'ast': results[team_b]['ast'][s]},
                    'players': {name: {'pts': stats['pts'][s], 'reb': stats['reb'][s], 'ast': stats['ast'][s], 'played': bool(stats['played'][s])} 
                               for name, stats in results['player_stats'].items()}}
            simulations.append(game)

        player_averages = []
        for name, stats in results['player_stats'].items():
            played = stats['played']
            player_averages.append({
                'name': name, 'team': stats['team'], 'play_probability': stats['play_probability'],
                'games_played_pct': played.mean() * 100,
                'pts': round(float(stats['pts'].mean()), 1),
                'reb': round(float(stats['reb'].mean()), 1),
                'ast': round(float(stats['ast'].mean()), 1),
                'pts_mode': round(self._compute_mode(stats['pts'][played]) if played.any() else 0, 1),
                'reb_mode': round(self._compute_mode(stats['reb'][played]) if played.any() else 0, 1),
                'ast_mode': round(self._compute_mode(stats['ast'][played]) if played.any() else 0, 1),
                'pts_95_ci': [round(float(np.percentile(stats['pts'], 2.5)), 1), round(float(np.percentile(stats['pts'], 97.5)), 1)],
                'reb_95_ci': [round(float(np.percentile(stats['reb'], 2.5)), 1), round(float(np.percentile(stats['reb'], 97.5)), 1)],
                'ast_95_ci': [round(float(np.percentile(stats['ast'], 2.5)), 1), round(float(np.percentile(stats['ast'], 97.5)), 1)],
                'pts_99_ci': [round(float(np.percentile(stats['pts'], 0.5)), 1), round(float(np.percentile(stats['pts'], 99.5)), 1)],
                'reb_99_ci': [round(float(np.percentile(stats['reb'], 0.5)), 1), round(float(np.percentile(stats['reb'], 99.5)), 1)],
                'ast_99_ci': [round(float(np.percentile(stats['ast'], 0.5)), 1), round(float(np.percentile(stats['ast'], 99.5)), 1)],
                'pts_std': round(float(stats['pts'].std()), 2), 'reb_std': round(float(stats['reb'].std()), 2), 'ast_std': round(float(stats['ast'].std()), 2),
            })

        return {
            'team_a': team_a, 'team_b': team_b, 
            'win_prob_a': win_prob_a,
            'team_summaries': team_summaries,
            'simulations': simulations, 
            'player_averages': player_averages,
            'betting_lines': betting_lines,
            'lineup_a': lineup_a,
            'lineup_b': lineup_b
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
