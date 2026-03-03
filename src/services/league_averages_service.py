"""
League averages service for dynamic NBA statistics.

This service calculates current league averages from recent game data,
with configurable fallback defaults when dynamic data is unavailable.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LeagueAveragesService:
    def __init__(self, config: Optional[Any] = None):
        self._config = config
        self._cache: Dict[str, Any] = {}
        self._cache_loaded = False
    
    @property
    def config(self):
        """Get config, using defaults if not set."""
        if self._config is None:
            from ..config.config import get_config
            return get_config()
        return self._config
    
    def _get_fallback(self, key: str, default: Any = None) -> Any:
        """Get fallback value from config league_averages."""
        if self._config and hasattr(self._config, 'league_averages'):
            return getattr(self._config.league_averages, key, default)
        return default
    
    def get_current_averages(self, season: Optional[int] = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get current league averages, calculating from recent data if available.
        
        Args:
            season: NBA season year (e.g., 2024 for 2024-25 season). If None, uses current.
            force_refresh: Ignore cache and recalculate.
        
        Returns:
            Dict with league average statistics.
        """
        cache_key = f"league_averages_{season}"
        
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._calculate_from_data(season)
        
        if result is None:
            logger.info("Using fallback league averages from config")
            result = self._get_fallback_dict()
        
        self._cache[cache_key] = result
        return result
    
    def _calculate_from_data(self, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Calculate league averages from loaded game data."""
        try:
            data_dir = self.config.data.data_dir
            games_file = data_dir / "nba_games.csv"
            
            if not games_file.exists():
                return None
            
            df = pd.read_csv(games_file, parse_dates=['GAME_DATE'])
            
            if season is not None:
                season_start = pd.Timestamp(f"{season}-10-01")
                season_end = pd.Timestamp(f"{season + 1}-07-31")
                df = df[(df['GAME_DATE'] >= season_start) & (df['GAME_DATE'] <= season_end)]
            
            recent_games = df.tail(500)
            
            if len(recent_games) < 50:
                logger.warning(f"Not enough games for reliable averages: {len(recent_games)}")
                return None
            
            result = {
                'points_per_100': float(recent_games['PTS'].mean()),
                'offensive_rating': float(recent_games['PTS'].mean()),
                'defensive_rating': float(recent_games['PTS'].mean()),
                'pace': float(recent_games.get('PACE', pd.Series([100] * len(recent_games))).mean()),
                'effective_fg_pct': float(recent_games.get('EFG_PCT', pd.Series([0.54] * len(recent_games))).mean()),
                'turnover_pct': float(recent_games.get('TOV_PCT', pd.Series([0.135] * len(recent_games))).mean()),
                'offensive_rebound_pct': float(recent_games.get('ORB_PCT', pd.Series([0.25] * len(recent_games))).mean()),
                'free_throw_rate': float(recent_games.get('FT_RATE', pd.Series([0.23] * len(recent_games))).mean()),
                'fg_pct': float(recent_games.get('FG_PCT', pd.Series([0.470] * len(recent_games))).mean()),
                'fg3_pct': float(recent_games.get('FG3_PCT', pd.Series([0.360] * len(recent_games))).mean()),
                'ft_pct': float(recent_games.get('FT_PCT', pd.Series([0.75] * len(recent_games))).mean()),
                'sample_size': len(recent_games),
            }
            
            logger.info(f"Calculated league averages from {len(recent_games)} recent games")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to calculate league averages from data: {e}")
            return None
    
    def _get_fallback_dict(self) -> Dict[str, Any]:
        """Get fallback values from config as dictionary."""
        return {
            'points_per_100': self._get_fallback('points_per_100', 114.0),
            'offensive_rating': self._get_fallback('offensive_rating', 114.0),
            'defensive_rating': self._get_fallback('defensive_rating', 114.0),
            'pace': self._get_fallback('pace', 100.0),
            'effective_fg_pct': self._get_fallback('effective_fg_pct', 0.54),
            'turnover_pct': self._get_fallback('turnover_pct', 0.135),
            'offensive_rebound_pct': self._get_fallback('offensive_rebound_pct', 0.25),
            'free_throw_rate': self._get_fallback('free_throw_rate', 0.23),
            'fg_pct': self._get_fallback('fg_pct', 0.470),
            'fg3_pct': self._get_fallback('fg3_pct', 0.360),
            'ft_pct': self._get_fallback('ft_pct', 0.75),
            'sample_size': 0,
        }
    
    def get_game_totals_defaults(self) -> Dict[str, float]:
        """Get default game totals and spreads."""
        return {
            'default_total': self._get_fallback('default_total', 225.0),
            'implied_home_pts': self._get_fallback('implied_home_pts', 112.5),
            'default_home_rating': self._get_fallback('default_home_rating', 114.0),
            'default_away_rating': self._get_fallback('default_away_rating', 114.0),
        }
    
    def get_historical_std(self) -> Dict[str, float]:
        """Get historical standard deviations for totals and spreads."""
        return {
            'total': self._get_fallback('historical_std_total', 11.0),
            'spread': self._get_fallback('historical_std_spread', 10.5),
        }
    
    def get_vegas_weight(self) -> float:
        """Get Vegas model weight for blending."""
        return self._get_fallback('vegas_weight', 0.30)
    
    def get_player_defaults(self) -> Dict[str, float]:
        """Get default player stat percentages by position."""
        return {
            'fg_pct': self._get_fallback('fg_pct', 0.470),
            'fg3_pct': self._get_fallback('fg3_pct', 0.360),
            'ft_pct': self._get_fallback('ft_pct', 0.75),
        }
    
    def get_position_averages(self) -> Dict[str, Dict[str, float]]:
        """Get position-based average statistics."""
        return {
            'PG': {'pts': 18, 'reb': 4, 'ast': 7, 'fg3_pct': 0.360},
            'SG': {'pts': 16, 'reb': 4, 'ast': 3, 'fg3_pct': 0.370},
            'SF': {'pts': 15, 'reb': 5, 'ast': 2, 'fg3_pct': 0.355},
            'PF': {'pts': 14, 'reb': 7, 'ast': 2, 'fg3_pct': 0.340},
            'C': {'pts': 13, 'reb': 9, 'ast': 2, 'fg3_pct': 0.250},
        }
    
    def get_team_averages(self, team_abbr: str) -> Dict[str, Any]:
        """Get team-specific average statistics (offensive/defensive ratings)."""
        return {
            'offensive_rating': self._get_fallback('offensive_rating', 114.0),
            'defensive_rating': self._get_fallback('defensive_rating', 114.0),
            'pace': self._get_fallback('pace', 100.0),
        }
    
    def clear_cache(self) -> None:
        """Clear the cache to force recalculation."""
        self._cache.clear()
        self._cache_loaded = False


_global_service: Optional[LeagueAveragesService] = None


def get_league_averages(config: Optional[Any] = None) -> LeagueAveragesService:
    """Get the global LeagueAveragesService instance."""
    global _global_service
    if _global_service is None:
        _global_service = LeagueAveragesService(config)
    return _global_service


def set_league_averages(service: LeagueAveragesService) -> None:
    """Set the global LeagueAveragesService instance."""
    global _global_service
    _global_service = service
