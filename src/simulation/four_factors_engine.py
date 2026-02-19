"""
NBA Four Factors Engine based on Dean Oliver's basketball analytics.
Implements proper possession-based team total prediction.
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FourFactors:
    """Dean Oliver's Four Factors for basketball efficiency."""
    efg_pct: float              # Effective FG% = (FGM + 0.5*FG3M) / FGA
    tov_pct: float              # Turnover % = TOV / (FGA + 0.44*FTA + TOV)
    orb_pct: float              # Off Reb % = ORB / (ORB + Opp DRB)
    ft_rate: float              # FT Rate = FTM / FGA (or FTA/FGA)
    
    def to_dict(self) -> dict:
        return {
            'efg_pct': self.efg_pct,
            'tov_pct': self.tov_pct,
            'orb_pct': self.orb_pct,
            'ft_rate': self.ft_rate
        }


@dataclass
class TeamEfficiency:
    """Team efficiency metrics for prediction."""
    offensive_rating: float     # Points per 100 possessions
    defensive_rating: float     # Points allowed per 100 possessions
    pace: float                 # Possessions per game
    net_rating: float           # ORtg - DRtg
    four_factors_off: FourFactors
    four_factors_def: FourFactors
    
    def to_dict(self) -> dict:
        return {
            'offensive_rating': self.offensive_rating,
            'defensive_rating': self.defensive_rating,
            'pace': self.pace,
            'net_rating': self.net_rating,
            'four_factors_off': self.four_factors_off.to_dict(),
            'four_factors_def': self.four_factors_def.to_dict()
        }


class FourFactorsEngine:
    """
    Implements Dean Oliver's Four Factors model for realistic NBA game prediction.
    
    The four factors explain basketball efficiency:
    1. Shooting (eFG%) - ~40% of success
    2. Turnovers (TOV%) - ~25% of success  
    3. Rebounding (ORB%) - ~20% of success
    4. Free Throws (FT Rate) - ~15% of success
    """
    
    LEAGUE_AVERAGES = {
        'efg_pct': 0.540,
        'tov_pct': 0.135,
        'orb_pct': 0.250,
        'ft_rate': 0.230,
        'pace': 100.0,
        'offensive_rating': 114.0,
        'defensive_rating': 114.0
    }
    
    FACTOR_WEIGHTS = {
        'efg_pct': 0.40,
        'tov_pct': 0.25,
        'orb_pct': 0.20,
        'ft_rate': 0.15
    }
    
    def __init__(self):
        self._team_cache: Dict[str, TeamEfficiency] = {}
    
    def calculate_efficiency(
        self,
        team_stats: dict,
        opponent_stats: dict = None
    ) -> TeamEfficiency:
        """
        Calculate team efficiency from stats dictionary.
        
        Args:
            team_stats: Dict with offensive stats
            opponent_stats: Dict with defensive stats (uses league avg if None)
        """
        opp = opponent_stats or self.LEAGUE_AVERAGES
        
        off_factors = FourFactors(
            efg_pct=team_stats.get('efg_pct', team_stats.get('fg_pct', 0.54)),
            tov_pct=team_stats.get('tov_pct', 0.135),
            orb_pct=team_stats.get('orb_pct', 0.25),
            ft_rate=team_stats.get('ft_rate', team_stats.get('fta_rate', 0.23))
        )
        
        def_factors = FourFactors(
            efg_pct=opp.get('opp_efg_pct', opp.get('efg_pct', 0.54)),
            tov_pct=opp.get('opp_tov_pct', opp.get('tov_pct', 0.135)),
            orb_pct=1 - opp.get('opp_dreb_pct', 0.75),
            ft_rate=opp.get('opp_ft_rate', opp.get('ft_rate', 0.23))
        )
        
        pace = team_stats.get('pace', self.LEAGUE_AVERAGES['pace'])
        ortg = team_stats.get('offensive_rating', self._estimate_ortg(off_factors))
        drtg = opponent_stats.get('defensive_rating', self._estimate_drtg(def_factors)) if opponent_stats else self.LEAGUE_AVERAGES['defensive_rating']
        
        return TeamEfficiency(
            offensive_rating=ortg,
            defensive_rating=drtg,
            pace=pace,
            net_rating=ortg - drtg,
            four_factors_off=off_factors,
            four_factors_def=def_factors
        )
    
    def _estimate_ortg(self, factors: FourFactors) -> float:
        """Estimate offensive rating from four factors."""
        base_ortg = 114.0
        
        efg_diff = factors.efg_pct - self.LEAGUE_AVERAGES['efg_pct']
        tov_diff = self.LEAGUE_AVERAGES['tov_pct'] - factors.tov_pct
        orb_diff = factors.orb_pct - self.LEAGUE_AVERAGES['orb_pct']
        ft_diff = factors.ft_rate - self.LEAGUE_AVERAGES['ft_rate']
        
        ortg_adj = (
            efg_diff * 200 * self.FACTOR_WEIGHTS['efg_pct'] +
            tov_diff * 200 * self.FACTOR_WEIGHTS['tov_pct'] +
            orb_diff * 80 * self.FACTOR_WEIGHTS['orb_pct'] +
            ft_diff * 100 * self.FACTOR_WEIGHTS['ft_rate']
        )
        
        return base_ortg + ortg_adj
    
    def _estimate_drtg(self, factors: FourFactors) -> float:
        """Estimate defensive rating from opponent four factors."""
        base_drtg = 114.0
        
        efg_diff = factors.efg_pct - self.LEAGUE_AVERAGES['efg_pct']
        tov_diff = self.LEAGUE_AVERAGES['tov_pct'] - factors.tov_pct
        orb_diff = factors.orb_pct - self.LEAGUE_AVERAGES['orb_pct']
        ft_diff = factors.ft_rate - self.LEAGUE_AVERAGES['ft_rate']
        
        drtg_adj = (
            efg_diff * 200 * self.FACTOR_WEIGHTS['efg_pct'] +
            tov_diff * 200 * self.FACTOR_WEIGHTS['tov_pct'] +
            orb_diff * 80 * self.FACTOR_WEIGHTS['orb_pct'] +
            ft_diff * 100 * self.FACTOR_WEIGHTS['ft_rate']
        )
        
        return base_drtg + drtg_adj
    
    def predict_matchup(
        self,
        home_efficiency: TeamEfficiency,
        away_efficiency: TeamEfficiency,
        num_samples: int = 1000
    ) -> Dict[str, any]:
        """
        Predict a matchup using four factors model.
        
        Returns dict with:
            - predicted_possessions
            - home_pts_mean, home_pts_std
            - away_pts_mean, away_pts_std
            - home_win_prob
            - predicted_total
        """
        predicted_pace = (home_efficiency.pace + away_efficiency.pace) / 2
        
        possession_variance = 4.0
        pace_samples = np.random.normal(predicted_pace, possession_variance, num_samples)
        pace_samples = np.clip(pace_samples, 90, 115)
        
        home_ortg = home_efficiency.offensive_rating
        away_ortg = away_efficiency.offensive_rating
        
        home_drtg = home_efficiency.defensive_rating
        away_drtg = away_efficiency.defensive_rating
        
        matchup_home_ortg = self._calculate_matchup_ortg(
            home_efficiency.four_factors_off,
            away_efficiency.four_factors_def
        )
        matchup_away_ortg = self._calculate_matchup_ortg(
            away_efficiency.four_factors_off,
            home_efficiency.four_factors_def
        )
        
        home_ortg = (home_ortg + matchup_home_ortg) / 2
        away_ortg = (away_ortg + matchup_away_ortg) / 2
        
        home_advantage = 2.5
        home_ortg_adj = home_ortg + (home_advantage * 100 / predicted_pace)
        away_ortg_adj = away_ortg - (home_advantage * 100 / predicted_pace)
        
        ortg_std = 6.0
        home_ortg_samples = np.random.normal(home_ortg_adj, ortg_std, num_samples)
        away_ortg_samples = np.random.normal(away_ortg_adj, ortg_std, num_samples)
        
        home_pts_samples = pace_samples * home_ortg_samples / 100
        away_pts_samples = pace_samples * away_ortg_samples / 100
        
        home_pts_samples = np.clip(home_pts_samples, 70, 160)
        away_pts_samples = np.clip(away_pts_samples, 70, 160)
        
        home_wins = (home_pts_samples > away_pts_samples).sum()
        home_win_prob = home_wins / num_samples
        
        return {
            'predicted_possessions': float(np.mean(pace_samples)),
            'pace_std': float(np.std(pace_samples)),
            'home_pts_mean': float(np.mean(home_pts_samples)),
            'home_pts_std': float(np.std(home_pts_samples)),
            'home_pts_median': float(np.median(home_pts_samples)),
            'away_pts_mean': float(np.mean(away_pts_samples)),
            'away_pts_std': float(np.std(away_pts_samples)),
            'away_pts_median': float(np.median(away_pts_samples)),
            'predicted_total': float(np.mean(home_pts_samples + away_pts_samples)),
            'home_win_prob': float(home_win_prob),
            'home_ortg': float(home_ortg_adj),
            'away_ortg': float(away_ortg_adj),
            'spread': float(np.mean(home_pts_samples - away_pts_samples))
        }
    
    def _calculate_matchup_ortg(
        self,
        offense_factors: FourFactors,
        defense_factors: FourFactors
    ) -> float:
        """
        Calculate offensive rating for a specific matchup.
        
        Uses interaction between offensive and defensive four factors.
        """
        base_ortg = 114.0
        
        efg_matchup = (offense_factors.efg_pct + (1 - defense_factors.efg_pct) + 0.54) / 2.5
        tov_matchup = (offense_factors.tov_pct + defense_factors.tov_pct) / 2
        orb_matchup = (offense_factors.orb_pct + (1 - defense_factors.orb_pct)) / 2
        
        efg_adj = (efg_matchup - 0.54) * 200
        tov_adj = (0.135 - tov_matchup) * 200
        orb_adj = (orb_matchup - 0.25) * 80
        ft_adj = (offense_factors.ft_rate - 0.23) * 100
        
        adjusted_ortg = base_ortg + (
            efg_adj * self.FACTOR_WEIGHTS['efg_pct'] +
            tov_adj * self.FACTOR_WEIGHTS['tov_pct'] +
            orb_adj * self.FACTOR_WEIGHTS['orb_pct'] +
            ft_adj * self.FACTOR_WEIGHTS['ft_rate']
        )
        
        return np.clip(adjusted_ortg, 95, 135)
    
    def _calculate_matchup_drtg(
        self,
        opponent_offense: FourFactors,
        defense_factors: FourFactors
    ) -> float:
        """Calculate defensive rating given opponent offense and defense."""
        ortg_allowed = self._calculate_matchup_ortg(opponent_offense, defense_factors)
        return ortg_allowed
    
    def get_possession_estimate(
        self,
        team_a_pace: float,
        team_b_pace: float,
        rest_days_a: int = 2,
        rest_days_b: int = 2,
        is_b2b_a: bool = False,
        is_b2b_b: bool = False
    ) -> Tuple[float, float]:
        """
        Estimate possessions for a game with rest adjustments.
        
        Returns:
            (mean_possessions, std_possessions)
        """
        base_pace = (team_a_pace + team_b_pace) / 2
        
        if rest_days_a == 0 or is_b2b_a:
            base_pace *= 0.99
        elif rest_days_a >= 3:
            base_pace *= 1.01
        
        if rest_days_b == 0 or is_b2b_b:
            base_pace *= 0.99
        elif rest_days_b >= 3:
            base_pace *= 1.01
        
        pace_std = 3.5
        
        return base_pace, pace_std
    
    def distribute_team_points_to_players(
        self,
        team_total: float,
        player_usages: Dict[str, float],
        player_efg: Dict[str, float],
        player_ft_rate: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Distribute team total points to players based on usage and efficiency.
        
        Uses four factors to allocate points proportionally.
        """
        total_usage = sum(player_usages.values())
        if total_usage == 0:
            return {name: team_total / len(player_usages) for name in player_usages}
        
        points_allocation = {}
        
        player_fts = {}
        player_fgs = {}
        
        for name, usage in player_usages.items():
            weight = usage / total_usage
            efg = player_efg.get(name, 0.54)
            ft_rate = player_ft_rate.get(name, 0.23)
            
            efg_contribution = efg * 2 * self.FACTOR_WEIGHTS['efg_pct']
            ft_contribution = ft_rate * 1 * self.FACTOR_WEIGHTS['ft_rate']
            efficiency_factor = efg_contribution + ft_contribution
            
            points_allocation[name] = team_total * weight * (efficiency_factor / 0.22)
        
        total_estimated = sum(points_allocation.values())
        if total_estimated > 0:
            points_allocation = {
                name: pts * (team_total / total_estimated)
                for name, pts in points_allocation.items()
            }
        
        return points_allocation


class TeamTotalPredictor:
    """
    High-level interface for predicting team totals using four factors model.
    Integrates with BasketballRefScraper for real data.
    """
    
    def __init__(self, bref_scraper=None):
        from src.data.basketball_ref_scraper import BasketballRefScraper
        self.bref_scraper = bref_scraper or BasketballRefScraper()
        self.four_factors_engine = FourFactorsEngine()
        self._cache: Dict[str, TeamEfficiency] = {}
    
    def get_team_efficiency(self, team_abbr: str, season: str = None) -> TeamEfficiency:
        """Get team efficiency from cached data or fetch."""
        cache_key = f"{team_abbr}_{season}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        stats = self.bref_scraper.get_team_stats(team_abbr, season)
        
        efficiency = self.four_factors_engine.calculate_efficiency(stats)
        self._cache[cache_key] = efficiency
        
        return efficiency
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        season: str = None,
        num_samples: int = 1000
    ) -> dict:
        """
        Predict a game outcome using four factors.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season string (e.g., '2024-25')
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Prediction dictionary with scores, win prob, spread
        """
        home_eff = self.get_team_efficiency(home_team, season)
        away_eff = self.get_team_efficiency(away_team, season)
        
        return self.four_factors_engine.predict_matchup(home_eff, away_eff, num_samples)
    
    def get_expected_total(self, team_a: str, team_b: str, season: str = None) -> float:
        """Get expected total points for a game."""
        prediction = self.predict_game(team_a, team_b, season, num_samples=500)
        return prediction['predicted_total']
    
    def get_spread(self, home_team: str, away_team: str, season: str = None) -> float:
        """
        Get predicted spread (home points - away points).
        Positive = home favored, Negative = away favored.
        """
        prediction = self.predict_game(home_team, away_team, season, num_samples=500)
        return prediction['spread']


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = FourFactorsEngine()
    
    lakers_factors = FourFactors(
        efg_pct=0.552,
        tov_pct=0.132,
        orb_pct=0.268,
        ft_rate=0.245
    )
    
    celtics_factors = FourFactors(
        efg_pct=0.568,
        tov_pct=0.128,
        orb_pct=0.252,
        ft_rate=0.218
    )
    
    lakers_eff = TeamEfficiency(
        offensive_rating=116.5,
        defensive_rating=112.0,
        pace=101.5,
        net_rating=4.5,
        four_factors_off=lakers_factors,
        four_factors_def=FourFactors(0.54, 0.135, 0.25, 0.23)
    )
    
    celtics_eff = TeamEfficiency(
        offensive_rating=120.2,
        defensive_rating=109.5,
        pace=99.8,
        net_rating=10.7,
        four_factors_off=celtics_factors,
        four_factors_def=FourFactors(0.52, 0.140, 0.245, 0.22)
    )
    
    print("Predicting LAL @ BOS...")
    prediction = engine.predict_matchup(celtics_eff, lakers_eff)
    
    print(f"\nGame Prediction:")
    print(f"  Pace: {prediction['predicted_possessions']:.1f} possessions")
    print(f"  Home (BOS): {prediction['home_pts_mean']:.1f} ± {prediction['home_pts_std']:.1f} pts")
    print(f"  Away (LAL): {prediction['away_pts_mean']:.1f} ± {prediction['away_pts_std']:.1f} pts")
    print(f"  Total: {prediction['predicted_total']:.1f} pts")
    print(f"  Spread: {prediction['spread']:+.1f}")
    print(f"  Home Win Prob: {prediction['home_win_prob']:.1%}")