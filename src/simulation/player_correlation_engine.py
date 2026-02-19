"""
NBA Player Correlation Engine for realistic stat correlations.
Models how player stats correlate with each other (assists → points, teammates competing for rebounds).
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class PlayerCorrelation:
    """Correlation coefficients between two players."""
    player_a: str
    player_b: str
    pts_corr: float
    reb_corr: float
    ast_corr: float
    same_team: bool
    correlation_type: str  # 'synergy', 'competition', 'neutral'


class PlayerCorrelationEngine:
    """
    Models realistic correlations between player stats.
    
    Key correlations modeled:
    1. Assister → Scorer: Positive correlation when teammates
    2. Rebound competition: Negative correlation between teammates
    3. Usage rate trade-offs: When one player scores more, others score less
    4. Defensive matchups: Opposing player stat correlations
    """
    
    DEFAULT_CORRELATIONS = {
        'teammate_pts': {'base': -0.08, 'stars': -0.15},
        'teammate_reb': {'base': -0.15, 'centers': -0.25},
        'teammate_ast': {'base': 0.25, 'pg_scorer': 0.35},
        'assister_scorer': 0.35,
        'opposing_reb': 0.12,
        'opposing_pts': -0.02,
        'usage_tradeoff': -0.20,
    }
    
    POSITION_SYNERGY_MATRIX = {
        ('PG', 'SG'): {'ast_pts': 0.30, 'pts_tradeoff': -0.10},
        ('PG', 'SF'): {'ast_pts': 0.28, 'pts_tradeoff': -0.08},
        ('PG', 'PF'): {'ast_pts': 0.22, 'pts_tradeoff': -0.06},
        ('PG', 'C'): {'ast_pts': 0.25, 'pts_tradeoff': -0.05},
        ('SG', 'SF'): {'ast_pts': 0.20, 'pts_tradeoff': -0.10},
        ('SG', 'PF'): {'ast_pts': 0.15, 'pts_tradeoff': -0.08},
        ('SG', 'C'): {'ast_pts': 0.18, 'pts_tradeoff': -0.05},
        ('SF', 'PF'): {'ast_pts': 0.15, 'pts_tradeoff': -0.08},
        ('SF', 'C'): {'ast_pts': 0.20, 'pts_tradeoff': -0.06},
        ('PF', 'C'): {'ast_pts': 0.12, 'pts_tradeoff': -0.10},
    }
    
    REBOUND_COMPETITION = {
        ('C', 'C'): -0.30,
        ('C', 'PF'): -0.20,
        ('C', 'SF'): -0.08,
        ('PF', 'PF'): -0.25,
        ('PF', 'SF'): -0.12,
        ('SF', 'SF'): -0.15,
        ('SG', 'SG'): -0.10,
        ('PG', 'PG'): -0.05,
    }
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._correlation_cache: Dict[str, PlayerCorrelation] = {}
        self._team_correlations: Dict[str, np.ndarray] = {}
    
    def build_correlation_matrix(
        self,
        players: List[Dict],
        team_assignments: Dict[str, str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build correlation matrices for PTS, REB, AST across all players.
        
        Args:
            players: List of player dicts with name, position, usage
            team_assignments: Dict mapping player_name -> team_abbr
            
        Returns:
            Tuple of (pts_corr_matrix, reb_corr_matrix, ast_corr_matrix)
        """
        n = len(players)
        
        pts_corr = np.eye(n)
        reb_corr = np.eye(n)
        ast_corr = np.eye(n)
        
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players):
                if i >= j:
                    continue
                
                corr = self._calculate_player_correlation(p1, p2, team_assignments)
                
                pts_corr[i, j] = corr.pts_corr
                pts_corr[j, i] = corr.pts_corr
                reb_corr[i, j] = corr.reb_corr
                reb_corr[j, i] = corr.reb_corr
                ast_corr[i, j] = corr.ast_corr
                ast_corr[j, i] = corr.ast_corr
                
                cache_key = f"{p1['name']}_{p2['name']}"
                self._correlation_cache[cache_key] = corr
        
        pts_corr = self._ensure_positive_definite(pts_corr)
        reb_corr = self._ensure_positive_definite(reb_corr)
        ast_corr = self._ensure_positive_definite(ast_corr)
        
        return pts_corr, reb_corr, ast_corr
    
    def _calculate_player_correlation(
        self,
        p1: Dict,
        p2: Dict,
        team_assignments: Dict[str, str]
    ) -> PlayerCorrelation:
        """Calculate correlation between two players."""
        team1 = team_assignments.get(p1['name'], '')
        team2 = team_assignments.get(p2['name'], '')
        same_team = team1 == team2 and team1 != ''
        
        pos1 = p1.get('position', 'SG')
        pos2 = p2.get('position', 'SG')
        usage1 = p1.get('usage', 0.15)
        usage2 = p2.get('usage', 0.15)
        
        if same_team:
            pts_corr, reb_corr, ast_corr = self._teammate_correlations(
                pos1, pos2, usage1, usage2
            )
            corr_type = 'synergy' if ast_corr > 0.15 else 'competition'
        else:
            pts_corr, reb_corr, ast_corr = self._opponent_correlations(
                pos1, pos2, team1, team2
            )
            corr_type = 'neutral'
        
        return PlayerCorrelation(
            player_a=p1['name'],
            player_b=p2['name'],
            pts_corr=pts_corr,
            reb_corr=reb_corr,
            ast_corr=ast_corr,
            same_team=same_team,
            correlation_type=corr_type
        )
    
    def _teammate_correlations(
        self,
        pos1: str,
        pos2: str,
        usage1: float,
        usage2: float
    ) -> Tuple[float, float, float]:
        """Calculate correlations for teammates."""
        pts_corr = self.DEFAULT_CORRELATIONS['teammate_pts']['base']
        reb_corr = self.DEFAULT_CORRELATIONS['teammate_reb']['base']
        ast_corr = self.DEFAULT_CORRELATIONS['teammate_ast']['base']
        
        pos_key = (pos1, pos2) if (pos1, pos2) in self.POSITION_SYNERGY_MATRIX else (pos2, pos1)
        if pos_key in self.POSITION_SYNERGY_MATRIX:
            synergy = self.POSITION_SYNERGY_MATRIX[pos_key]
            ast_corr = synergy.get('ast_pts', ast_corr)
            pts_corr = synergy.get('pts_tradeoff', pts_corr)
        
        rebound_key = (pos1, pos2) if (pos1, pos2) in self.REBOUND_COMPETITION else (pos2, pos1)
        if rebound_key in self.REBOUND_COMPETITION:
            reb_corr = self.REBOUND_COMPETITION[rebound_key]
        
        usage_product = usage1 * usage2
        if usage_product > 0.05:
            pts_corr -= 0.05
        
        return pts_corr, reb_corr, ast_corr
    
    def _opponent_correlations(
        self,
        pos1: str,
        pos2: str,
        team1: str,
        team2: str
    ) -> Tuple[float, float, float]:
        """Calculate correlations for opposing players."""
        pts_corr = self.DEFAULT_CORRELATIONS['opposing_pts']
        reb_corr = self.DEFAULT_CORRELATIONS['opposing_reb']
        ast_corr = 0.05
        
        if pos1 == pos2:
            reb_corr = 0.15
            pts_corr = 0.08
        
        if pos1 in ['C', 'PF'] and pos2 in ['C', 'PF']:
            reb_corr = 0.20
        
        if pos1 in ['PG', 'SG'] and pos2 in ['PG', 'SG']:
            pts_corr = 0.05
            ast_corr = 0.10
        
        return pts_corr, reb_corr, ast_corr
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is valid (positive definite)."""
        matrix = (matrix + matrix.T) / 2
        
        min_eig = np.min(np.linalg.eigvalsh(matrix))
        if min_eig < 0:
            matrix = matrix + (-min_eig + 0.01) * np.eye(matrix.shape[0])
        
        np.fill_diagonal(matrix, 1.0)
        
        return matrix
    
    def apply_correlations(
        self,
        base_predictions: Dict[str, Dict[str, float]],
        team_assignments: Dict[str, str],
        num_samples: int = 100,
        seed: int = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Apply realistic correlations to base predictions.
        
        Args:
            base_predictions: Dict of player_name -> {pts, reb, ast}
            team_assignments: Dict of player_name -> team_abbr
            num_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            
        Returns:
            Dict of player_name -> {pts: [...], reb: [...], ast: [...]}
        """
        players = [{'name': name, 'position': 'SG', 'usage': 0.15} for name in base_predictions]
        
        pts_corr, reb_corr, ast_corr = self.build_correlation_matrix(
            players, team_assignments
        )
        
        n = len(players)
        player_names = list(base_predictions.keys())
        
        rng = np.random.default_rng(seed)
        
        pts_means = np.array([base_predictions[name]['pts'] for name in player_names])
        reb_means = np.array([base_predictions[name]['reb'] for name in player_names])
        ast_means = np.array([base_predictions[name]['ast'] for name in player_names])
        
        pts_stds = np.array([
            max(3.0, base_predictions[name].get('pts_std', 0.4 * base_predictions[name]['pts']))
            for name in player_names
        ])
        reb_stds = np.array([
            max(1.5, base_predictions[name].get('reb_std', 0.4 * base_predictions[name]['reb']))
            for name in player_names
        ])
        ast_stds = np.array([
            max(1.0, base_predictions[name].get('ast_std', 0.5 * base_predictions[name]['ast']))
            for name in player_names
        ])
        
        L_pts = np.linalg.cholesky(pts_corr)
        L_reb = np.linalg.cholesky(reb_corr)
        L_ast = np.linalg.cholesky(ast_corr)
        
        Z = rng.standard_normal((num_samples, n))
        
        pts_samples = pts_means + L_pts @ Z.T * pts_stds[:, np.newaxis]
        pts_samples = pts_samples.T
        
        Z = rng.standard_normal((num_samples, n))
        reb_samples = reb_means + L_reb @ Z.T * reb_stds[:, np.newaxis]
        reb_samples = reb_samples.T
        
        Z = rng.standard_normal((num_samples, n))
        ast_samples = ast_means + L_ast @ Z.T * ast_stds[:, np.newaxis]
        ast_samples = ast_samples.T
        
        pts_samples = np.maximum(pts_samples, 0)
        reb_samples = np.maximum(reb_samples, 0)
        ast_samples = np.maximum(ast_samples, 0)
        
        results = {}
        for i, name in enumerate(player_names):
            results[name] = {
                'pts': pts_samples[:, i].tolist(),
                'reb': reb_samples[:, i].tolist(),
                'ast': ast_samples[:, i].tolist()
            }
        
        return results
    
    def apply_correlations_torch(
        self,
        base_predictions: torch.Tensor,
        stds: torch.Tensor,
        correlation_matrix: torch.Tensor,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        GPU-accelerated correlation application using PyTorch.
        
        Args:
            base_predictions: Tensor of shape (n_players, 3) for [pts, reb, ast]
            stds: Tensor of shape (n_players, 3)
            correlation_matrix: Tensor of shape (n_players, n_players)
            num_samples: Number of samples
            
        Returns:
            Tensor of shape (num_samples, n_players, 3)
        """
        base_predictions = base_predictions.to(self.device)
        stds = stds.to(self.device)
        correlation_matrix = correlation_matrix.to(self.device)
        
        n_players = base_predictions.shape[0]
        
        L = torch.linalg.cholesky(correlation_matrix)
        
        Z = torch.randn(num_samples, n_players, device=self.device)
        
        correlated = torch.matmul(Z, L.T)
        
        samples = base_predictions.unsqueeze(0) + correlated.unsqueeze(-1) * stds.unsqueeze(0)
        
        samples = torch.maximum(samples, torch.zeros_like(samples))
        
        return samples
    
    def get_team_constraints(
        self,
        player_predictions: Dict[str, Dict[str, float]],
        team_assignments: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate team total constraints for normalization.
        
        Returns:
            Dict with team totals and per-player constrained allocations
        """
        team_totals = {}
        
        for name, preds in player_predictions.items():
            team = team_assignments.get(name, 'UNK')
            if team not in team_totals:
                team_totals[team] = {'pts': 0, 'reb': 0, 'ast': 0, 'players': []}
            
            team_totals[team]['pts'] += preds['pts']
            team_totals[team]['reb'] += preds['reb']
            team_totals[team]['ast'] += preds['ast']
            team_totals[team]['players'].append(name)
        
        realistic_totals = {
            'pts': {'min': 90, 'max': 150, 'mean': 115},
            'reb': {'min': 35, 'max': 60, 'mean': 44},
            'ast': {'min': 18, 'max': 35, 'mean': 26}
        }
        
        constraints = {}
        for team, totals in team_totals.items():
            constraints[team] = {
                'predicted_pts': totals['pts'],
                'predicted_reb': totals['reb'],
                'predicted_ast': totals['ast'],
                'pts_adjustment': np.clip(realistic_totals['pts']['mean'] / max(totals['pts'], 1), 0.85, 1.15),
                'reb_adjustment': np.clip(realistic_totals['reb']['mean'] / max(totals['reb'], 1), 0.80, 1.20),
                'ast_adjustment': np.clip(realistic_totals['ast']['mean'] / max(totals['ast'], 1), 0.75, 1.25),
            }
        
        return constraints
    
    def normalize_to_team_totals(
        self,
        samples: Dict[str, Dict[str, List[float]]],
        team_assignments: Dict[str, str],
        target_means: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Normalize player samples to enforce realistic team totals.
        
        Args:
            samples: Correlated samples per player
            team_assignments: Dict of player_name -> team
            target_means: Optional dict of team -> {pts, reb, ast} targets
            
        Returns:
            Normalized samples maintaining correlations
        """
        teams = set(team_assignments.values())
        
        for team in teams:
            team_players = [p for p in team_assignments if team_assignments[p] == team]
            
            if not team_players:
                continue
            
            num_samples = len(samples[team_players[0]]['pts'])
            
            for stat in ['pts', 'reb', 'ast']:
                team_totals = np.zeros(num_samples)
                
                for player in team_players:
                    team_totals += np.array(samples[player][stat])
                
                league_avg = {'pts': 115, 'reb': 44, 'ast': 26}[stat]
                target = target_means.get(team, {}).get(stat, league_avg) if target_means else league_avg
                
                adjustments = target / np.maximum(team_totals, 1)
                adjustments = np.clip(adjustments, 0.7, 1.3)
                
                for player in team_players:
                    samples[player][stat] = (
                        np.array(samples[player][stat]) * adjustments
                    ).tolist()
        
        return samples
    
    def calculate_assist_correlation(
        self,
        assister_predictions: Dict[str, float],
        scorer_predictions: Dict[str, float],
        team: str,
        team_assignments: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate which assists go to which scorers.
        
        Returns:
            Dict mapping (assister, scorer) -> number of assists
        """
        assist_allocations = {}
        
        team_assisters = {k: v for k, v in assister_predictions.items() 
                         if team_assignments.get(k) == team}
        team_scorers = {k: v for k, v in scorer_predictions.items() 
                       if team_assignments.get(k) == team}
        
        if not team_assisters or not team_scorers:
            return assist_allocations
        
        total_assists = sum(team_assisters.values())
        total_pts = sum(team_scorers.values())
        
        for assister, ast in team_assisters.items():
            for scorer, pts in team_scorers.items():
                if assister == scorer:
                    continue
                
                scorer_share = pts / max(total_pts, 1)
                expected_assists = ast * scorer_share * 0.5
                
                assist_allocations[f"{assister}_{scorer}"] = expected_assists
        
        return assist_allocations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = PlayerCorrelationEngine()
    
    players = [
        {'name': 'Tatum', 'position': 'SF', 'usage': 0.28},
        {'name': 'Brown', 'position': 'SG', 'usage': 0.25},
        {'name': 'Horford', 'position': 'C', 'usage': 0.14},
        {'name': 'White', 'position': 'PG', 'usage': 0.15},
        {'name': 'Holiday', 'position': 'PG', 'usage': 0.16},
    ]
    
    team_assignments = {p['name']: 'BOS' for p in players}
    
    base_predictions = {
        'Tatum': {'pts': 27, 'reb': 8, 'ast': 4, 'pts_std': 5, 'reb_std': 3, 'ast_std': 2},
        'Brown': {'pts': 23, 'reb': 6, 'ast': 3, 'pts_std': 4, 'reb_std': 2, 'ast_std': 2},
        'Horford': {'pts': 12, 'reb': 7, 'ast': 4, 'pts_std': 3, 'reb_std': 3, 'ast_std': 2},
        'White': {'pts': 14, 'reb': 4, 'ast': 5, 'pts_std': 3, 'reb_std': 2, 'ast_std': 2},
        'Holiday': {'pts': 13, 'reb': 5, 'ast': 6, 'pts_std': 3, 'reb_std': 2, 'ast_std': 2},
    }
    
    print("Building correlation matrices...")
    pts_corr, reb_corr, ast_corr = engine.build_correlation_matrix(players, team_assignments)
    
    print("\nPTS Correlation Matrix:")
    print(np.round(pts_corr, 3))
    
    print("\nREB Correlation Matrix:")
    print(np.round(reb_corr, 3))
    
    print("\nAST Correlation Matrix:")
    print(np.round(ast_corr, 3))
    
    print("\nApplying correlations to predictions...")
    correlated_samples = engine.apply_correlations(
        base_predictions, team_assignments, num_samples=1000, seed=42
    )
    
    print("\nCorrelated Sample Statistics (mean ± std):")
    for player in players:
        pts = np.array(correlated_samples[player['name']]['pts'])
        reb = np.array(correlated_samples[player['name']]['reb'])
        ast = np.array(correlated_samples[player['name']]['ast'])
        print(f"  {player['name']}: PTS {pts.mean():.1f}±{pts.std():.1f}, "
              f"REB {reb.mean():.1f}±{reb.std():.1f}, AST {ast.mean():.1f}±{ast.std():.1f}")
    
    constraints = engine.get_team_constraints(base_predictions, team_assignments)
    print(f"\nTeam Constraints for BOS:")
    for k, v in constraints['BOS'].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")