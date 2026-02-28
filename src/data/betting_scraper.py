"""
Betting Lines Scraper - Integrates Vegas betting lines as priors.
Uses betting market efficiency to improve simulation accuracy.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BettingScraper:
    """
    Scrapes betting lines from multiple sources.
    
    Uses Vegas lines as efficient priors - betting markets aggregate
    enormous information and are highly predictive.
    
    Key metrics:
    - Game totals (over/under)
    - Point spreads
    - Moneyline odds
    - Implied team totals
    """
    
    CACHE_TTL_HOURS = 4
    MAX_RETRIES = 3
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_game_lines(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str = None
    ) -> dict:
        """
        Get betting lines for a specific game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Dictionary with betting lines
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        home_team = home_team.upper()
        away_team = away_team.upper()
        
        cache_key = f"{away_team}_{home_team}_{game_date}"
        cache_file = os.path.join(self.cache_dir, f"betting_{cache_key}.json")
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.CACHE_TTL_HOURS):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        
        lines = self._fetch_game_lines(home_team, away_team, game_date)
        
        if lines.get('total') is None:
            lines = self._get_fallback_lines(home_team, away_team)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(lines, f, indent=2, default=str)
        except Exception:
            pass
        
        return lines
    
    def _fetch_game_lines(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str
    ) -> dict:
        """Fetch lines from multiple sources."""
        lines = {
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date,
            'total': None,
            'spread': None,
            'home_ml': None,
            'away_ml': None,
            'home_implied_pts': None,
            'away_implied_pts': None,
            'source': None,
            'fetched_at': datetime.now().isoformat()
        }
        
        action_lines = self._fetch_from_action_network(home_team, away_team, game_date)
        if action_lines:
            lines.update(action_lines)
            lines['source'] = 'action_network'
            return lines
        
        oddsjam_lines = self._fetch_from_oddsjam(home_team, away_team, game_date)
        if oddsjam_lines:
            lines.update(oddsjam_lines)
            lines['source'] = 'oddsjam'
            return lines
        
        return lines
    
    def _fetch_from_action_network(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str
    ) -> Optional[dict]:
        """Try to scrape from Action Network (free)."""
        try:
            url = f"https://www.actionnetwork.com/nba/{away_team.lower()}-{home_team.lower()}-odds"
            
            response = self._session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            total_elem = soup.find('div', class_='total-cell')
            spread_elem = soup.find('div', class_='spread-cell')
            
            lines = {}
            
            if total_elem:
                total_text = total_elem.get_text(strip=True)
                over_under = float(''.join(filter(lambda x: x.isdigit() or x == '.', total_text.split('O')[0])))
                lines['total'] = over_under
            
            if spread_elem:
                spread_text = spread_elem.get_text(strip=True)
                spread = float(''.join(filter(lambda x: x.isdigit() or x == '.', spread_text)))
                lines['spread'] = spread
            
            return lines if lines else None
            
        except Exception as e:
            logger.debug(f"Action Network scrape failed: {e}")
            return None
    
    def _fetch_from_oddsjam(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str
    ) -> Optional[dict]:
        """Try to get lines from OddsJam."""
        return None
    
    def _get_fallback_lines(
        self, 
        home_team: str, 
        away_team: str
    ) -> dict:
        """Get fallback lines based on team ratings."""
        from src.data.basketball_ref_scraper import BasketballRefScraper
        
        try:
            scraper = BasketballRefScraper()
            
            home_stats = scraper.get_team_stats(home_team)
            away_stats = scraper.get_team_stats(away_team)
            
            home_ortg = home_stats.get('offensive_rating', 114)
            away_ortg = away_stats.get('offensive_rating', 114)
            home_drtg = home_stats.get('defensive_rating', 114)
            away_drtg = away_stats.get('defensive_rating', 114)
            
            home_pace = home_stats.get('pace', 100)
            away_pace = away_stats.get('pace', 100)
            avg_pace = (home_pace + away_pace) / 2
            
            home_expected = (home_ortg / 114) * (away_drtg / 114) * 111
            away_expected = (away_ortg / 114) * (home_drtg / 114) * 111
            
            home_implied = home_expected * (avg_pace / 100)
            away_implied = away_expected * (avg_pace / 100)
            
            total = home_implied + away_implied
            
            home_spread = (away_implied - home_implied) / 2
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'total': round(total, 1),
                'spread': round(-home_spread, 1) if home_spread > 0 else round(home_spread, 1),
                'home_implied_pts': round(home_implied, 1),
                'away_implied_pts': round(away_implied, 1),
                'source': 'estimated_from_ratings',
                'fetched_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Fallback lines failed: {e}")
            return {
                'home_team': home_team,
                'away_team': away_team,
                'total': 225,
                'spread': 0,
                'home_implied_pts': 112.5,
                'away_implied_pts': 112.5,
                'source': 'league_average',
                'fetched_at': datetime.now().isoformat()
            }
    
    def get_team_implied_totals(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str = None
    ) -> Tuple[float, float]:
        """
        Get implied points for each team from betting lines.
        
        Returns:
            Tuple of (home_implied_pts, away_implied_pts)
        """
        lines = self.get_game_lines(home_team, away_team, game_date)
        
        home_pts = lines.get('home_implied_pts')
        away_pts = lines.get('away_implied_pts')
        
        if home_pts is not None and away_pts is not None:
            return home_pts, away_pts
        
        total = lines.get('total', 225)
        spread = lines.get('spread', 0)
        
        home_pts = (total / 2) + (spread / 2)
        away_pts = (total / 2) - (spread / 2)
        
        return home_pts, away_pts
    
    def get_calibration_weight(self) -> float:
        """
        Get how much to weight Vegas lines vs model predictions.
        
        Vegas is highly efficient but not perfect.
        Weight of 0.3 means 30% Vegas, 70% model.
        """
        return 0.3
    
    def blend_with_model(
        self,
        model_home_pts: float,
        model_away_pts: float,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> Tuple[float, float]:
        """
        Blend model predictions with Vegas lines.
        
        Args:
            model_home_pts: Model's predicted home team points
            model_away_pts: Model's predicted away team points
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date
            
        Returns:
            Tuple of (blended_home_pts, blended_away_pts)
        """
        vegas_weight = self.get_calibration_weight()
        
        vegas_home, vegas_away = self.get_team_implied_totals(
            home_team, away_team, game_date
        )
        
        blended_home = (model_home_pts * (1 - vegas_weight)) + (vegas_home * vegas_weight)
        blended_away = (model_away_pts * (1 - vegas_weight)) + (vegas_away * vegas_weight)
        
        return blended_home, blended_away
    
    def get_sharp_money_indicator(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str = None
    ) -> dict:
        """
        Get indicator of where sharp money is going.
        
        Returns:
            Dictionary with betting percentages and sharp indicators
        """
        return {
            'home_pct': 50,
            'away_pct': 50,
            'sharp_indicator': 'neutral',
            'line_movement': 0,
            'public_pct_home': 50
        }
    
    def get_over_under_probability(
        self,
        home_team: str,
        away_team: str,
        predicted_total: float,
        game_date: str = None
    ) -> float:
        """
        Calculate probability that game goes over/under the total.
        
        Uses historical variance around the total to estimate probability.
        """
        lines = self.get_game_lines(home_team, away_team, game_date)
        vegas_total = lines.get('total', 225)
        
        if vegas_total is None:
            return 0.5
        
        diff = predicted_total - vegas_total
        
        historical_std = 11.0
        
        z_score = diff / historical_std
        
        from scipy import stats as sp_stats
        prob_over = sp_stats.norm.cdf(z_score)
        
        return float(np.clip(prob_over, 0.05, 0.95))
    
    def get_spread_cover_probability(
        self,
        home_team: str,
        away_team: str,
        predicted_margin: float,
        game_date: str = None
    ) -> float:
        """
        Calculate probability that home team covers the spread.
        """
        lines = self.get_game_lines(home_team, away_team, game_date)
        spread = lines.get('spread', 0)
        
        diff = predicted_margin - spread
        
        historical_std = 10.5
        
        z_score = diff / historical_std
        
        from scipy import stats as sp_stats
        prob_cover = sp_stats.norm.cdf(z_score)
        
        return float(np.clip(prob_cover, 0.05, 0.95))


class BettingFeatureGenerator:
    """
    Generates features from betting lines for the model.
    """
    
    def __init__(self, scraper: BettingScraper = None):
        self.scraper = scraper or BettingScraper()
    
    def get_betting_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> dict:
        """
        Generate betting-derived features for a game.
        
        These features can be used as additional inputs to the model.
        """
        lines = self.scraper.get_game_lines(home_team, away_team, game_date)
        
        home_implied, away_implied = self.scraper.get_team_implied_totals(
            home_team, away_team, game_date
        )
        
        total = lines.get('total', 225)
        spread = lines.get('spread', 0)
        
        features = {
            'vegas_total': total,
            'vegas_spread': spread,
            'home_implied_pts': home_implied,
            'away_implied_pts': away_implied,
            'implied_total': home_implied + away_implied,
            'total_vs_implied_diff': total - (home_implied + away_implied),
            'home_team_value': -spread / 2,
            'away_team_value': spread / 2,
            'line_source': lines.get('source', 'unknown')
        }
        
        return features
    
    def apply_vegas_priors(
        self,
        predictions: dict,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> dict:
        """
        Apply Vegas lines as priors to predictions.
        
        Returns adjusted predictions.
        """
        home_implied, away_implied = self.scraper.get_team_implied_totals(
            home_team, away_team, game_date
        )
        
        adjusted = predictions.copy()
        
        vegas_weight = self.scraper.get_calibration_weight()
        
        if 'team_totals' in predictions:
            home_pred = predictions['team_totals'].get(home_team, {}).get('pts', home_implied)
            away_pred = predictions['team_totals'].get(away_team, {}).get('pts', away_implied)
            
            adjusted_home = home_pred * (1 - vegas_weight) + home_implied * vegas_weight
            adjusted_away = away_pred * (1 - vegas_weight) + away_implied * vegas_weight
            
            if 'team_totals' not in adjusted:
                adjusted['team_totals'] = {}
            
            adjusted['team_totals'][home_team] = {'pts': adjusted_home}
            adjusted['team_totals'][away_team] = {'pts': adjusted_away}
        
        return adjusted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = BettingScraper()
    
    print("Testing Betting Scraper...")
    
    lines = scraper.get_game_lines('BOS', 'LAL')
    print(f"\nBOS vs LAI Lines:")
    print(f"  Total: {lines.get('total')}")
    print(f"  Spread: {lines.get('spread')}")
    print(f"  Home implied: {lines.get('home_implied_pts')}")
    print(f"  Away implied: {lines.get('away_implied_pts')}")
    print(f"  Source: {lines.get('source')}")
    
    home, away = scraper.get_team_implied_totals('BOS', 'LAL')
    print(f"\nImplied points: Home {home}, Away {away}")
    
    blended_home, blended_away = scraper.blend_with_model(115, 110, 'BOS', 'LAL')
    print(f"\nBlended (model 115/110): Home {blended_home:.1f}, Away {blended_away:.1f}")