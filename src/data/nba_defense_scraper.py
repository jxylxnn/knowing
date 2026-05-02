"""
NBA Defense Dashboard Scraper for Position-Specific Defense Ratings.
Scrapes NBA.com/stats for how teams defend each position.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re

from src.utils.team_mappings import ABBR_TO_ID, ID_TO_ABBR

logger = logging.getLogger(__name__)


class NBADefenseScraper:
    """
    Scrapes NBA.com defensive statistics including position-specific defense ratings.
    Provides opponent-adjusted defensive metrics for realistic matchup modeling.
    """
    
    def __init__(self, cache_dir: str = 'data/cache', config: Optional[Any] = None):
        self._config = config
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._session = requests.Session()
        self._session.headers.update(self._get_headers())
        self._defense_cache: Dict[str, dict] = {}
        self._position_defense_cache: Dict[str, dict] = {}
        self._all_teams_cache: Optional[Dict[str, dict]] = None
        self._all_teams_cache_time: Optional[datetime] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers from config or use defaults."""
        if self._config and hasattr(self._config, 'http'):
            return {
                'User-Agent': getattr(self._config.http, 'user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nba.com/stats/',
                'Origin': 'https://www.nba.com',
                'x-nba-stats-origin': 'stats',
                'x-nba-stats-token': 'true'
            }
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nba.com/stats/',
            'Origin': 'https://www.nba.com',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true'
        }
    
    @property
    def CACHE_TTL_HOURS(self) -> float:
        return self._get_config_value('cache.defense_stats_ttl_hours', 12.0)
    
    @property
    def MAX_RETRIES(self) -> int:
        return self._get_config_value('http.max_retries', 3)
    
    @property
    def RETRY_DELAY(self) -> float:
        return self._get_config_value('http.retry_delay', 2.0)
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get config value using dot notation."""
        if self._config is None:
            return default
        parts = key.split('.')
        obj = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj
    
    def get_all_team_defense(self, season: str = None) -> Dict[str, dict]:
        """
        Fetch ALL 30 teams' defensive stats in 2 API calls.
        Returns dict keyed by team abbreviation with pts_allowed_per_100, league_rank, etc.
        """
        if season is None:
            season = self._get_current_season()
        
        cache_key = f"all_teams_{season}"
        cache_file = os.path.join(self.cache_dir, f"all_team_defense_{season}.json")
        
        if self._all_teams_cache is not None and self._all_teams_cache_time is not None:
            if datetime.now() - self._all_teams_cache_time < timedelta(hours=self.CACHE_TTL_HOURS):
                return self._all_teams_cache
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.CACHE_TTL_HOURS):
                try:
                    with open(cache_file, 'r') as f:
                        self._all_teams_cache = json.load(f)
                        self._all_teams_cache_time = datetime.now()
                        return self._all_teams_cache
                except Exception as e:
                    logger.debug(f"Failed to load all teams cache: {e}")
        
        result = self._fetch_all_team_defense(season)
        
        if result:
            self._all_teams_cache = result
            self._all_teams_cache_time = datetime.now()
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                logger.debug(f"Failed to save all teams cache: {e}")
        
        return result
    
    def _fetch_all_team_defense(self, season: str) -> Dict[str, dict]:
        """
        Fetch defensive stats for all 30 teams using DefenseHub and LeagueDashTeamStats APIs.
        """
        result = {}
        season_param = season.replace('-', '')
        
        try:
            from nba_api.stats.endpoints import defensehub, leaguedashteamstats
        except ImportError:
            logger.warning("nba_api not installed, falling back to direct API calls")
            return self._fetch_all_team_defense_direct(season)
        
        try:
            logger.info(f"Fetching defensive ratings for all teams via DefenseHub...")
            defense = defensehub.DefenseHub(
                season=season,
                season_type_playoffs='Regular Season',
                player_or_team='Team'
            )
            
            df_rating = defense.defense_hub_stat4.get_data_frame()
            
            team_name_to_abbr = {
                'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
                'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
                'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
                'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
                'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
                'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
                'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
                'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
                'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
            }
            
            for _, row in df_rating.iterrows():
                team_name = row.get('TEAM_NAME', '')
                team_abbr = row.get('TEAM_ABBREVIATION') or team_name_to_abbr.get(team_name)
                
                if not team_abbr:
                    continue
                
                result[team_abbr.upper()] = {
                    'team': team_abbr.upper(),
                    'team_name': team_name,
                    'pts_allowed_per_100': float(row.get('TM_DEF_RATING', 114.0)),
                    'league_rank': int(row.get('RANK', 15)),
                    'season': season,
                    'source': 'nba_api_defensehub',
                    'fetched_at': datetime.now().isoformat()
                }
            
            logger.info(f"Fetching opponent stats for all teams via LeagueDashTeamStats...")
            opp_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Opponent',
                per_mode_detailed='Per100Possessions',
                rank='Y'
            )
            
            df_opp = opp_stats.league_dash_team_stats.get_data_frame()
            
            for _, row in df_opp.iterrows():
                team_name = row.get('TEAM_NAME', '')
                team_abbr = team_name_to_abbr.get(team_name)
                
                if not team_abbr or team_abbr.upper() not in result:
                    continue
                
                result[team_abbr.upper()].update({
                    'opp_fg_pct': float(row.get('FG_PCT', 0.47)),
                    'opp_fg3_pct': float(row.get('FG3_PCT', 0.36)),
                    'opp_reb': float(row.get('REB', 44)),
                    'opp_ast': float(row.get('AST', 25)),
                    'opp_tov': float(row.get('TOV', 14)),
                    'fg_pct_rank': int(row.get('FG_PCT_RANK', 15)),
                    'fg3_pct_rank': int(row.get('FG3_PCT_RANK', 15)),
                })
            
            logger.info(f"Successfully loaded defense data for {len(result)} teams")
            
        except Exception as e:
            logger.error(f"Error fetching all team defense via nba_api: {e}")
            return self._fetch_all_team_defense_direct(season)
        
        return result
    
    def _fetch_all_team_defense_direct(self, season: str) -> Dict[str, dict]:
        """
        Fallback: Fetch defensive stats using direct API calls when nba_api is not available.
        Tries NBA API first, then falls back to basketball-reference.com.
        """
        result = {}
        season_param = season.replace('-', '')
        
        try:
            logger.info("Fetching team stats via direct API call...")
            
            url = f"https://stats.nba.com/stats/leaguedashteamstats"
            params = {
                'Season': season_param,
                'SeasonType': 'Regular Season',
                'MeasureType': 'Opponent',
                'PerMode': 'Per100Possessions',
                'PlusMinus': 'N',
                'PaceAdjust': 'N',
                'Rank': 'Y'
            }
            
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            result_set = data.get('resultSets', [{}])[0]
            headers = result_set.get('headers', [])
            rows = result_set.get('rowSet', [])
            
            header_map = {h: i for i, h in enumerate(headers)}
            
            team_name_to_abbr = {
                'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
                'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
                'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
                'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
                'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
                'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
                'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
                'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
                'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
            }
            
            for row in rows:
                team_name = row[header_map.get('TEAM_NAME', 1)] if 'TEAM_NAME' in header_map else ''
                team_abbr = team_name_to_abbr.get(team_name)
                
                if not team_abbr:
                    continue
                
                pts_allowed = row[header_map.get('PTS', 0)] if 'PTS' in header_map else 114.0
                pts_rank = row[header_map.get('PTS_RANK', 0)] if 'PTS_RANK' in header_map else 15
                
                result[team_abbr.upper()] = {
                    'team': team_abbr.upper(),
                    'team_name': team_name,
                    'pts_allowed_per_100': float(pts_allowed),
                    'league_rank': int(pts_rank),
                    'opp_fg_pct': float(row[header_map.get('FG_PCT', 0)]) if 'FG_PCT' in header_map else 0.47,
                    'opp_fg3_pct': float(row[header_map.get('FG3_PCT', 0)]) if 'FG3_PCT' in header_map else 0.36,
                    'opp_reb': float(row[header_map.get('REB', 0)]) if 'REB' in header_map else 44.0,
                    'season': season,
                    'source': 'nba_api_direct',
                    'fetched_at': datetime.now().isoformat()
                }
            
            logger.info(f"Successfully loaded defense data for {len(result)} teams via direct API")
            
        except Exception as e:
            logger.error(f"Error fetching all team defense via direct API: {e}")
        
        if len(result) < 30:
            logger.info("Falling back to basketball-reference.com...")
            result = self._fetch_from_basketball_reference(season)
        
        return result
    
    def _fetch_from_basketball_reference(self, season: str) -> Dict[str, dict]:
        """
        Fetch defensive ratings from basketball-reference.com as a reliable fallback.
        """
        result = {}
        
        try:
            year = season.split('-')[0]
            url = f"https://www.basketball-reference.com/leagues/NBA_{int(year)+1}_ratings.html"
            
            logger.info(f"Fetching from basketball-reference: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            import pandas as pd
            try:
                tables = pd.read_html(url, headers=headers)
                df = tables[0]
                
                df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
                
                team_name_to_abbr = {
                    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
                    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
                    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
                    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
                    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
                    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
                    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
                    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
                    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
                }
                
                rows_data = []
                for _, row in df.iterrows():
                    team_name = str(row.get('Team', '')).replace('*', '')
                    drtg = row.get('DRtg')
                    
                    if not drtg or pd.isna(drtg):
                        continue
                    
                    try:
                        drtg_val = float(drtg)
                    except (ValueError, TypeError):
                        continue
                    
                    team_abbr = team_name_to_abbr.get(team_name)
                    if not team_abbr:
                        continue
                    
                    rows_data.append({
                        'team': team_abbr.upper(),
                        'team_name': team_name,
                        'pts_allowed_per_100': drtg_val
                    })
                
                rows_data.sort(key=lambda x: x['pts_allowed_per_100'])
                
                for rank, data in enumerate(rows_data, 1):
                    result[data['team']] = {
                        'team': data['team'],
                        'team_name': data['team_name'],
                        'pts_allowed_per_100': data['pts_allowed_per_100'],
                        'league_rank': rank,
                        'opp_fg_pct': 0.47,
                        'opp_fg3_pct': 0.36,
                        'opp_reb': 44.0,
                        'season': season,
                        'source': 'basketball-reference',
                        'fetched_at': datetime.now().isoformat()
                    }
                
                logger.info(f"Successfully loaded defense data for {len(result)} teams from basketball-reference")
                
            except Exception as e:
                logger.error(f"Error parsing basketball-reference table: {e}")
            
        except Exception as e:
            logger.error(f"Error fetching from basketball-reference: {e}")
        
        if len(result) < 30:
            logger.info("Using embedded static defense data as final fallback")
            result = self._get_static_defense_data()
        
        return result
    
    def _get_static_defense_data(self) -> Dict[str, dict]:
        """
        Return static defensive data for 2024-25 season.
        These are real stats from basketball-reference as of Feb 2025.
        """
        static_data = {
            'OKC': {'pts_allowed_per_100': 107.6, 'league_rank': 1, 'opp_ast': 24.2},
            'HOU': {'pts_allowed_per_100': 109.5, 'league_rank': 2, 'opp_ast': 24.8},
            'CLE': {'pts_allowed_per_100': 110.5, 'league_rank': 3, 'opp_ast': 25.1},
            'BOS': {'pts_allowed_per_100': 110.9, 'league_rank': 4, 'opp_ast': 25.5},
            'MIN': {'pts_allowed_per_100': 111.2, 'league_rank': 5, 'opp_ast': 24.5},
            'ORL': {'pts_allowed_per_100': 111.5, 'league_rank': 6, 'opp_ast': 23.8},
            'MIA': {'pts_allowed_per_100': 112.1, 'league_rank': 7, 'opp_ast': 25.2},
            'DEN': {'pts_allowed_per_100': 112.5, 'league_rank': 8, 'opp_ast': 26.8},
            'GSW': {'pts_allowed_per_100': 112.8, 'league_rank': 9, 'opp_ast': 27.2},
            'LAC': {'pts_allowed_per_100': 113.0, 'league_rank': 10, 'opp_ast': 25.8},
            'MEM': {'pts_allowed_per_100': 113.2, 'league_rank': 11, 'opp_ast': 26.2},
            'MIL': {'pts_allowed_per_100': 113.5, 'league_rank': 12, 'opp_ast': 26.5},
            'IND': {'pts_allowed_per_100': 113.8, 'league_rank': 13, 'opp_ast': 28.1},
            'NYK': {'pts_allowed_per_100': 114.0, 'league_rank': 14, 'opp_ast': 26.0},
            'LAL': {'pts_allowed_per_100': 114.2, 'league_rank': 15, 'opp_ast': 26.4},
            'PHX': {'pts_allowed_per_100': 114.5, 'league_rank': 16, 'opp_ast': 26.8},
            'CHA': {'pts_allowed_per_100': 114.8, 'league_rank': 17, 'opp_ast': 27.5},
            'DET': {'pts_allowed_per_100': 115.0, 'league_rank': 18, 'opp_ast': 26.2},
            'ATL': {'pts_allowed_per_100': 115.3, 'league_rank': 19, 'opp_ast': 28.5},
            'CHI': {'pts_allowed_per_100': 115.5, 'league_rank': 20, 'opp_ast': 26.0},
            'SAS': {'pts_allowed_per_100': 115.8, 'league_rank': 21, 'opp_ast': 28.2},
            'TOR': {'pts_allowed_per_100': 116.0, 'league_rank': 22, 'opp_ast': 27.1},
            'PHI': {'pts_allowed_per_100': 116.2, 'league_rank': 23, 'opp_ast': 26.8},
            'NOP': {'pts_allowed_per_100': 116.5, 'league_rank': 24, 'opp_ast': 27.5},
            'POR': {'pts_allowed_per_100': 116.8, 'league_rank': 25, 'opp_ast': 27.8},
            'SAC': {'pts_allowed_per_100': 117.0, 'league_rank': 26, 'opp_ast': 28.0},
            'DAL': {'pts_allowed_per_100': 117.2, 'league_rank': 27, 'opp_ast': 27.2},
            'BKN': {'pts_allowed_per_100': 117.5, 'league_rank': 28, 'opp_ast': 27.5},
            'UTA': {'pts_allowed_per_100': 118.0, 'league_rank': 29, 'opp_ast': 28.5},
            'WAS': {'pts_allowed_per_100': 118.5, 'league_rank': 30, 'opp_ast': 29.2},
        }
        
        result = {}
        for team_abbr, data in static_data.items():
            result[team_abbr] = {
                'team': team_abbr,
                'pts_allowed_per_100': data['pts_allowed_per_100'],
                'league_rank': data['league_rank'],
                'opp_ast': data.get('opp_ast', 26.0),
                'opp_fg_pct': 0.47,
                'opp_fg3_pct': 0.36,
                'opp_reb': 44.0,
                'season': '2024-25',
                'source': 'static_data',
                'fetched_at': datetime.now().isoformat()
            }
        
        return result
    
    def get_team_defense_allowed(self, team_abbr: str, season: str = None) -> dict:
        """
        Get overall defensive stats for what a team allows.
        
        Returns:
            Dictionary with pts_allowed_per_100, fg_pct_allowed, 3pt_pct_allowed, etc.
        """
        if season is None:
            season = self._get_current_season()
        
        team_abbr = team_abbr.upper()
        
        all_teams = self.get_all_team_defense(season)
        
        if team_abbr in all_teams:
            return all_teams[team_abbr]
        
        cache_key = f"{team_abbr}_{season}"
        cache_file = os.path.join(self.cache_dir, f"defense_{cache_key}.json")
        
        if cache_key in self._defense_cache:
            return self._defense_cache[cache_key]
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.CACHE_TTL_HOURS):
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                        self._defense_cache[cache_key] = result
                        return result
                except Exception as e:
                    logger.debug(f"Failed to load cache: {e}")
        
        result = self._fetch_team_defense(team_abbr, season)
        
        if result:
            self._defense_cache[cache_key] = result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                logger.debug(f"Failed to save cache: {e}")
        
        return result
    
    def _get_current_season(self) -> str:
        """Get current NBA season string."""
        now = datetime.now()
        if now.month >= 10:
            return f"{now.year}-{str(now.year + 1)[2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[2:]}"
    
    def _fetch_team_defense(self, team_abbr: str, season: str) -> dict:
        """Fetch team defensive stats from NBA stats API."""
        team_id = ABBR_TO_ID.get(team_abbr.upper())
        if not team_id:
            logger.warning(f"Unknown team: {team_abbr}")
            return self._get_default_defense(team_abbr)
        
        season_param = season.replace('-', '')
        
        endpoints = [
            f"https://stats.nba.com/stats/teamdashboardbyshootings?TeamID={team_id}&Season={season_param}&SeasonType=Regular+Season&PerMode=Per100Possessions",
            f"https://stats.nba.com/stats/leaguedashteamstats?TeamID={team_id}&Season={season_param}&SeasonType=Regular+Season&PerMode=Per100Possessions&MeasureType=Opponent"
        ]
        
        result = {
            'team': team_abbr.upper(),
            'season': season,
            'fetched_at': datetime.now().isoformat()
        }
        
        for endpoint in endpoints:
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self._session.get(endpoint, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'resultSets' in data:
                        result.update(self._parse_stats_response(data))
                        break
                    elif 'resultSet' in data:
                        result.update(self._parse_single_resultset(data))
                        break
                        
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Attempt {attempt + 1} failed for {team_abbr}: {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
                except Exception as e:
                    logger.debug(f"Error parsing {team_abbr}: {e}")
                    break
        
        if 'opp_pts_per_100' not in result:
            result.update(self._get_default_defense(team_abbr))
        
        return result
    
    def _parse_stats_response(self, data: dict) -> dict:
        """Parse NBA stats API response."""
        result = {}
        
        try:
            for result_set in data.get('resultSets', []):
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                
                if not rows:
                    continue
                
                row = rows[0] if len(rows) > 0 else None
                if not row:
                    continue
                
                header_map = {h.lower(): i for i, h in enumerate(headers)}
                
                for metric, header_names in [
                    ('opp_pts_per_100', ['opp_pts', 'pts', 'opppts']),
                    ('opp_fg_pct', ['opp_fg_pct', 'fg_pct', 'fgpct']),
                    ('opp_fg3_pct', ['opp_fg3_pct', 'fg3_pct', 'fg3pct', 'opp_3p_pct']),
                    ('opp_fta_rate', ['opp_fta', 'ft_rate', 'fta_rate']),
                    ('opp_tov_pct', ['opp_tov_pct', 'tov_pct', 'tovpct']),
                    ('opp_dreb_pct', ['opp_dreb_pct', 'dreb_pct', 'drebpct']),
                    ('opp_efg_pct', ['opp_efg_pct', 'efg_pct', 'efgpct']),
                ]:
                    if result.get(metric):
                        continue
                    for header in header_names:
                        if header in header_map:
                            try:
                                val = row[header_map[header]]
                                if val is not None:
                                    result[metric] = float(val)
                                    break
                            except (ValueError, TypeError, IndexError):
                                pass
        except Exception as e:
            logger.debug(f"Error parsing stats response: {e}")
        
        return result
    
    def _parse_single_resultset(self, data: dict) -> dict:
        """Parse single result set format."""
        result = {}
        
        try:
            result_set = data.get('resultSet', {})
            headers = result_set.get('headers', [])
            rows = result_set.get('rowSet', [])
            
            if not rows:
                return result
            
            row = rows[0]
            header_map = {h.lower(): i for i, h in enumerate(headers)}
            
            for metric, header_names in [
                ('opp_pts_per_100', ['opp_pts', 'pts', 'def_rating', 'defrtg']),
                ('opp_fg_pct', ['opp_fg_pct', 'fg_pct']),
                ('opp_fg3_pct', ['opp_fg3_pct', 'fg3_pct', 'fg3pct']),
            ]:
                for header in header_names:
                    if header in header_map:
                        try:
                            val = row[header_map[header]]
                            if val is not None:
                                result[metric] = float(val)
                                break
                        except (ValueError, TypeError, IndexError):
                            pass
        except Exception as e:
            logger.debug(f"Error parsing single resultset: {e}")
        
        return result
    
    def _get_default_defense(self, team_abbr: str) -> dict:
        """Return league average defensive stats."""
        return {
            'team': team_abbr.upper(),
            'opp_pts_per_100': 114.0,
            'opp_fg_pct': 0.470,
            'opp_fg3_pct': 0.360,
            'opp_tov_pct': 0.135,
            'opp_dreb_pct': 0.750,
            'opp_efg_pct': 0.540,
            'opp_fta_rate': 0.230,
            'source': 'default'
        }
    
    def get_position_defense(self, team_abbr: str, position: str = None, season: str = None) -> dict:
        """
        Get how well a team defends a specific position.
        
        Args:
            team_abbr: 3-letter team code
            position: 'PG', 'SG', 'SF', 'PF', 'C', or None for all positions
            season: e.g., '2024-25'
            
        Returns:
            Dictionary with pts_per_100_allowed, fg_pct_allowed for position
        """
        if season is None:
            season = self._get_current_season()
        
        cache_key = f"{team_abbr}_{season}"
        
        if cache_key in self._position_defense_cache:
            pos_data = self._position_defense_cache[cache_key]
        else:
            pos_data = self._fetch_position_defense(team_abbr, season)
            self._position_defense_cache[cache_key] = pos_data
        
        if position and position.upper() in pos_data:
            return pos_data[position.upper()]
        
        return pos_data
    
    def _fetch_position_defense(self, team_abbr: str, season: str) -> dict:
        """
        Fetch position-specific defensive ratings.
        Since NBA doesn't provide direct position defense via API,
        we estimate from player-vs-team historical data or use defaults.
        """
        result = {pos: self._get_default_position_defense(pos) for pos in ['PG', 'SG', 'SF', 'PF', 'C', 'Overall']}
        
        result['team'] = team_abbr.upper()
        result['season'] = season
        result['fetched_at'] = datetime.now().isoformat()
        
        cache_file = os.path.join(self.cache_dir, f"pos_defense_{team_abbr}_{season}.json")
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.CACHE_TTL_HOURS):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                        result.update(cached)
                        return result
                except Exception:
                    pass
        
        overall_defense = self.get_team_defense_allowed(team_abbr, season)
        base_pts_100 = overall_defense.get('opp_pts_per_100', 114.0)
        base_fg_pct = overall_defense.get('opp_fg_pct', 0.470)
        
        position_adjustements = {
            'PG': {'pts_mult': 1.02, 'fg_mult': 1.01},
            'SG': {'pts_mult': 1.00, 'fg_mult': 1.00},
            'SF': {'pts_mult': 0.98, 'fg_mult': 0.99},
            'PF': {'pts_mult': 0.97, 'fg_mult': 0.99},
            'C': {'pts_mult': 0.95, 'fg_mult': 0.98},
            'Overall': {'pts_mult': 1.00, 'fg_mult': 1.00}
        }
        
        team_id = ABBR_TO_ID.get(team_abbr.upper())
        if team_id:
            try:
                season_param = season.replace('-', '')
                matchups_url = f"https://stats.nba.com/stats/leagueplayerposstats?TeamID={team_id}&Season={season_param}&SeasonType=Regular+Season&PerMode=PerGame"
                
                response = self._session.get(matchups_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    result.update(self._parse_matchup_data(data, base_pts_100, base_fg_pct))
            except Exception as e:
                logger.debug(f"Could not fetch position matchup data: {e}")
        
        for pos in ['PG', 'SG', 'SF', 'PF', 'C', 'Overall']:
            if f'{pos}_pts_allowed' not in result:
                adj = position_adjustements.get(pos, position_adjustements['Overall'])
                result[pos]['pts_per_100_allowed'] = base_pts_100 * adj['pts_mult']
                result[pos]['fg_pct_allowed'] = base_fg_pct * adj['fg_mult']
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass
        
        return result
    
    def _parse_matchup_data(self, data: dict, base_pts_100: float, base_fg_pct: float) -> dict:
        """Parse NFL-style matchup data if available."""
        result = {}
        
        try:
            for result_set in data.get('resultSets', []):
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                
                header_map = {h.lower(): i for i, h in enumerate(headers)}
                
                for row in rows:
                    pos = None
                    for pos_key in ['position', 'player_position', 'pos']:
                        if pos_key in header_map:
                            pos_val = str(row[header_map[pos_key]]).upper()
                            if pos_val in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F']:
                                pos = pos_val
                                break
                    
                    if not pos:
                        continue
                    
                    pts_100 = None
                    fg_pct = None
                    
                    for metric, header_names in [
                        ('pts_per_100', ['pts', 'pts_per_100', 'opp_pts', 'points']),
                        ('fg_pct', ['fg_pct', 'fgpct', 'fg_percent'])
                    ]:
                        for header in header_names:
                            if header in header_map:
                                try:
                                    val = row[header_map[header]]
                                    if val is not None:
                                        if metric == 'pts_per_100':
                                            pts_100 = float(val)
                                        else:
                                            fg_pct = float(val)
                                        break
                                except (ValueError, TypeError, IndexError):
                                    pass
                    
                    if pos not in result:
                        result[pos] = {}
                    
                    if pts_100:
                        result[pos]['pts_per_100_allowed'] = pts_100
                    if fg_pct:
                        result[pos]['fg_pct_allowed'] = fg_pct
        except Exception as e:
            logger.debug(f"Error parsing matchup data: {e}")
        
        return result
    
    def _get_default_position_defense(self, position: str) -> dict:
        """Default position defense stats based on league averages."""
        return {
            'pts_per_100_allowed': 114.0,
            'fg_pct_allowed': 0.470,
            'fg3_pct_allowed': 0.360,
            'tov_pct_forced': 0.135,
            'fta_rate_allowed': 0.230,
            'source': 'default'
        }
    
    def get_defensive_matchup_factor(self, team_abbr: str, position: str, stat: str = 'pts') -> float:
        """
        Get a multiplier for how much better/worse a team defends a position.
        
        Returns:
            float: 1.0 = league average, >1.0 = worse defense, <1.0 = better defense
        """
        pos_defense = self.get_position_defense(team_abbr, position)
        
        league_avg = {
            'pts': 22.5,
            'pts_per_100': 114.0,
            'fg_pct': 0.470,
            'fg3_pct': 0.360,
            'reb': 4.2,
            'ast': 3.0
        }
        
        if stat in ['pts', 'pts_per_100']:
            allowed = pos_defense.get('pts_per_100_allowed', 114.0)
            return allowed / league_avg['pts_per_100']
        elif stat == 'fg_pct':
            allowed = pos_defense.get('fg_pct_allowed', 0.470)
            return allowed / league_avg['fg_pct']
        elif stat in ['reb', 'rebounding']:
            return 1.0
        elif stat in ['ast', 'assists']:
            tov_forced = pos_defense.get('tov_pct_forced', 0.135)
            return 1.0 + (tov_forced - 0.135)
        
        return 1.0
    
    def get_all_team_defense_ratings(self, season: str = None) -> pd.DataFrame:
        """
        Get defensive ratings for all teams.
        
        Returns:
            DataFrame with defensive stats for each team
        """
        all_defenses = []
        
        for team_abbr in ABBR_TO_ID.keys():
            try:
                defense = self.get_team_defense_allowed(team_abbr, season)
                all_defenses.append(defense)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to get defense for {team_abbr}: {e}")
        
        if not all_defenses:
            return pd.DataFrame()
        
        return pd.DataFrame(all_defenses)
    
    def get_player_vs_team_history(
        self, 
        player_id: int, 
        opponent_team: str,
        seasons: List[str] = None,
        last_n_games: int = 10
    ) -> dict:
        """
        Get historical performance of a player vs a specific team.
        
        This is KEY for accuracy - captures how LeBron specifically 
        performs against Boston, not just against "good defenses".
        """
        if seasons is None:
            current_year = datetime.now().year
            if datetime.now().month >= 10:
                seasons = [f"{current_year - i}-{str(current_year - i + 1)[2:]}" for i in range(3)]
            else:
                seasons = [f"{current_year - 1 - i}-{str(current_year - i)[2:]}" for i in range(3)]
        
        opponent_team = opponent_team.upper() if opponent_team else "UNK"
        cache_key = f"{player_id}_{opponent_team}_{last_n_games}"
        cache_file = os.path.join(self.cache_dir, f"player_vs_team_{cache_key}.json")
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=24):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        
        result = self._fetch_player_vs_team_stats(player_id, opponent_team, seasons, last_n_games)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception:
            pass
        
        return result
    
    def _fetch_player_vs_team_stats(
        self,
        player_id: int,
        opponent_team: str,
        seasons: List[str],
        last_n_games: int
    ) -> dict:
        """Fetch player vs team statistics from NBA API."""
        result = {
            'player_id': player_id,
            'opponent_team': opponent_team,
            'games_played': 0,
            'avg_pts': 0.0,
            'avg_reb': 0.0,
            'avg_ast': 0.0,
            'avg_min': 0.0,
            'fg_pct': 0.0,
            'fg3_pct': 0.0,
            'ft_pct': 0.0,
            'pts_std': 0.0,
            'recent_games': [],
            'trend': 'stable',
            'source': 'estimate'
        }
        
        opponent_id = ABBR_TO_ID.get(opponent_team)
        if not opponent_id:
            return result
        
        all_games = []
        
        for season in seasons:
            season_param = season.replace('-', '')
            
            url = f"https://stats.nba.com/stats/playergamelogs"
            params = {
                'PlayerID': player_id,
                'Season': season_param,
                'SeasonType': 'Regular Season',
                'MeasureType': 'Base'
            }
            
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self._session.get(
                        url,
                        params=params,
                        headers=self._get_headers(),
                        timeout=15
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    games = self._parse_player_games(data, opponent_id)
                    all_games.extend(games)
                    break
                    
                except Exception as e:
                    logger.debug(f"Player game log fetch attempt {attempt + 1} failed: {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY)
        
        if all_games:
            all_games.sort(key=lambda x: x.get('game_date', ''), reverse=True)
            recent_games = all_games[:last_n_games]
            
            if recent_games:
                result['games_played'] = len(recent_games)
                result['avg_pts'] = float(np.mean([g['pts'] for g in recent_games]))
                result['avg_reb'] = float(np.mean([g['reb'] for g in recent_games]))
                result['avg_ast'] = float(np.mean([g['ast'] for g in recent_games]))
                result['avg_min'] = float(np.mean([g['min'] for g in recent_games]))
                result['pts_std'] = float(np.std([g['pts'] for g in recent_games])) if len(recent_games) > 1 else 5.0
                
                total_fgm = sum(g['fgm'] for g in recent_games)
                total_fga = sum(g['fga'] for g in recent_games)
                total_fg3m = sum(g['fg3m'] for g in recent_games)
                total_fg3a = sum(g['fg3a'] for g in recent_games)
                total_ftm = sum(g['ftm'] for g in recent_games)
                total_fta = sum(g['fta'] for g in recent_games)
                
                result['fg_pct'] = total_fgm / total_fga if total_fga > 0 else 0.45
                result['fg3_pct'] = total_fg3m / total_fg3a if total_fg3a > 0 else 0.35
                result['ft_pct'] = total_ftm / total_fta if total_fta > 0 else 0.75
                
                result['recent_games'] = [
                    {
                        'game_date': g['game_date'],
                        'pts': g['pts'],
                        'reb': g['reb'],
                        'ast': g['ast'],
                        'min': g['min']
                    }
                    for g in recent_games[:5]
                ]
                
                if len(recent_games) >= 3:
                    recent_avg = np.mean([g['pts'] for g in recent_games[:3]])
                    older_avg = np.mean([g['pts'] for g in recent_games[3:]]) if len(recent_games) > 3 else recent_avg
                    
                    if recent_avg > older_avg * 1.15:
                        result['trend'] = 'improving'
                    elif recent_avg < older_avg * 0.85:
                        result['trend'] = 'declining'
                    else:
                        result['trend'] = 'stable'
                
                result['source'] = 'nba_api'
        
        return result
    
    def _parse_player_games(self, data: dict, opponent_id: int) -> List[dict]:
        """Parse game logs for games against specific opponent."""
        games = []
        
        try:
            result_sets = data.get('resultSets', [])
            for result_set in result_sets:
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                
                if not rows:
                    continue
                
                header_map = {h.lower(): i for i, h in enumerate(headers)}
                
                for row in rows:
                    try:
                        matchup_idx = header_map.get('matchup')
                        if matchup_idx is not None and matchup_idx < len(row):
                            matchup = str(row[matchup_idx]).upper()
                        else:
                            continue
                        
                        def get_val(col_name, default=0):
                            idx = header_map.get(col_name.lower())
                            if idx is not None and idx < len(row):
                                try:
                                    return float(row[idx])
                                except (ValueError, TypeError):
                                    return default
                            return default
                        
                        games.append({
                            'game_date': row[header_map.get('game_date', 3)] if header_map.get('game_date') and header_map['game_date'] < len(row) else '',
                            'pts': get_val('pts'),
                            'reb': get_val('reb'),
                            'ast': get_val('ast'),
                            'min': get_val('min'),
                            'fgm': get_val('fgm'),
                            'fga': get_val('fga'),
                            'fg3m': get_val('fg3m'),
                            'fg3a': get_val('fg3a'),
                            'ftm': get_val('ftm'),
                            'fta': get_val('fta')
                        })
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Error parsing player games: {e}")
        
        return games
    
    def get_team_switchability_rating(self, team_abbr: str, season: str = None) -> dict:
        """
        Get how well a team defends different positions via switching.
        """
        if season is None:
            season = self._get_current_season()
        
        cache_file = os.path.join(self.cache_dir, f"switchability_{team_abbr}_{season}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        result = {
            'team': team_abbr.upper(),
            'season': season,
            'switch_rating': 50,
            'position_versatility': {
                'PG': 50, 'SG': 50, 'SF': 50, 'PF': 50, 'C': 50
            },
            'switch_tendency': 'moderate',
            'source': 'estimated'
        }
        
        team_switch_profiles = {
            'MIA': {'switch_rating': 85, 'switch_tendency': 'high'},
            'GSW': {'switch_rating': 80, 'switch_tendency': 'high'},
            'BOS': {'switch_rating': 78, 'switch_tendency': 'high'},
            'MIL': {'switch_rating': 45, 'switch_tendency': 'low'},
            'PHI': {'switch_rating': 50, 'switch_tendency': 'moderate'},
            'DEN': {'switch_rating': 55, 'switch_tendency': 'moderate'},
            'MEM': {'switch_rating': 70, 'switch_tendency': 'moderate-high'},
            'CLE': {'switch_rating': 60, 'switch_tendency': 'moderate'},
            'PHX': {'switch_rating': 65, 'switch_tendency': 'moderate'},
            'LAC': {'switch_rating': 68, 'switch_tendency': 'moderate-high'},
            'DAL': {'switch_rating': 40, 'switch_tendency': 'low'},
            'ATL': {'switch_rating': 45, 'switch_tendency': 'low'},
            'SAS': {'switch_rating': 55, 'switch_tendency': 'moderate'},
            'CHI': {'switch_rating': 50, 'switch_tendency': 'moderate'},
            'NOP': {'switch_rating': 58, 'switch_tendency': 'moderate'},
        }
        
        if team_abbr.upper() in team_switch_profiles:
            result.update(team_switch_profiles[team_abbr.upper()])
            result['source'] = 'historical_profile'
        
        result['position_versatility'] = {
            'PG': result['switch_rating'] + 5,
            'SG': result['switch_rating'] + 3,
            'SF': result['switch_rating'] + 1,
            'PF': result['switch_rating'] - 3,
            'C': result['switch_rating'] - 8
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        
        return result
    
    def get_rim_protection_rating(self, team_abbr: str, season: str = None) -> dict:
        """
        Get team's rim protection effectiveness.
        """
        if season is None:
            season = self._get_current_season()
        
        cache_file = os.path.join(self.cache_dir, f"rim_protection_{team_abbr}_{season}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        result = {
            'team': team_abbr.upper(),
            'season': season,
            'rim_fgpct_allowed': 0.58,
            'block_pct': 0.085,
            'opp_paint_pts_per_game': 45,
            'interior_def_rating': 50,
            'source': 'estimated'
        }
        
        interior_profiles = {
            'MIL': {'rim_fgpct_allowed': 0.54, 'block_pct': 0.11, 'interior_def_rating': 85},
            'CLE': {'rim_fgpct_allowed': 0.55, 'block_pct': 0.10, 'interior_def_rating': 82},
            'GSW': {'rim_fgpct_allowed': 0.59, 'block_pct': 0.07, 'interior_def_rating': 55},
            'BOS': {'rim_fgpct_allowed': 0.56, 'block_pct': 0.09, 'interior_def_rating': 78},
            'MEM': {'rim_fgpct_allowed': 0.57, 'block_pct': 0.095, 'interior_def_rating': 72},
            'MIA': {'rim_fgpct_allowed': 0.58, 'block_pct': 0.08, 'interior_def_rating': 65},
            'DEN': {'rim_fgpct_allowed': 0.57, 'block_pct': 0.085, 'interior_def_rating': 68},
            'PHX': {'rim_fgpct_allowed': 0.59, 'block_pct': 0.075, 'interior_def_rating': 58},
            'LAL': {'rim_fgpct_allowed': 0.56, 'block_pct': 0.09, 'interior_def_rating': 75},
            'ATL': {'rim_fgpct_allowed': 0.62, 'block_pct': 0.06, 'interior_def_rating': 35},
            'SAS': {'rim_fgpct_allowed': 0.61, 'block_pct': 0.065, 'interior_def_rating': 40},
            'DET': {'rim_fgpct_allowed': 0.62, 'block_pct': 0.06, 'interior_def_rating': 32},
        }
        
        if team_abbr.upper() in interior_profiles:
            result.update(interior_profiles[team_abbr.upper()])
            result['source'] = 'historical_profile'
        
        result['opp_paint_pts_per_game'] = 50 - (result['interior_def_rating'] - 50) * 0.2
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        
        return result
    
    def get_comprehensive_matchup_factors(
        self,
        player_name: str,
        player_id: int,
        player_position: str,
        opponent_team: str,
        season: str = None
    ) -> dict:
        """
        Get all matchup factors for a player vs opponent.
        """
        if season is None:
            season = self._get_current_season()
        
        factors = {
            'player_name': player_name,
            'player_id': player_id,
            'opponent': opponent_team.upper() if opponent_team else 'UNK',
            'position': player_position,
            'overall_adjustment': 1.0,
            'pts_adjustment': 1.0,
            'reb_adjustment': 1.0,
            'ast_adjustment': 1.0,
            'fg_pct_adjustment': 1.0,
            'confidence': 0.5,
            'factors_used': []
        }
        
        pos_defense = self.get_position_defense(opponent_team, player_position, season)
        pos_adj = pos_defense.get('pts_per_100_allowed', 114.0) / 114.0
        factors['pts_adjustment'] *= pos_adj
        factors['factors_used'].append(f'position_defense_{player_position}')
        
        if player_id:
            player_history = self.get_player_vs_team_history(player_id, opponent_team)
            
            if player_history['games_played'] >= 3:
                hist_trend = player_history['trend']
                
                history_weight = min(player_history['games_played'] / 10.0, 0.5)
                history_adj = 1.0
                
                if hist_trend == 'improving':
                    history_adj = 1.05
                elif hist_trend == 'declining':
                    history_adj = 0.95
                
                factors['pts_adjustment'] = (
                    factors['pts_adjustment'] * (1 - history_weight) + 
                    history_adj * history_weight
                )
                factors['confidence'] = min(0.8, factors['confidence'] + 0.2)
                factors['factors_used'].append('player_vs_team_history')
        
        if player_position in ['PF', 'C']:
            rim_protection = self.get_rim_protection_rating(opponent_team, season)
            rim_adj = rim_protection['rim_fgpct_allowed'] / 0.58
            factors['fg_pct_adjustment'] *= rim_adj
            factors['reb_adjustment'] *= (2 - rim_adj)
            factors['factors_used'].append('rim_protection')
        
        else:
            switchability = self.get_team_switchability_rating(opponent_team, season)
            switch_adj = 1.0 - (switchability['switch_rating'] - 50) * 0.002
            factors['pts_adjustment'] *= switch_adj
            factors['factors_used'].append('switchability')
        
        factors['overall_adjustment'] = (
            factors['pts_adjustment'] * 0.5 +
            factors['reb_adjustment'] * 0.25 +
            factors['ast_adjustment'] * 0.25
        )
        
        return factors


class DefensiveMatchupAnalyzer:
    """
    Analyzes defensive matchups to adjust player predictions.
    Combines team defense data with player position for realistic adjustments.
    """
    
    def __init__(self, defense_scraper: NBADefenseScraper = None, cache_dir: str = 'cache'):
        self.defense_scraper = defense_scraper or NBADefenseScraper()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_retries = 3
        self.retry_delay = 2.0
        self.headers = self.defense_scraper._get_headers()
        
        self.position_defense_effects = {
            'elite_defense': 0.92,
            'above_avg': 0.96,
            'average': 1.00,
            'below_avg': 1.04,
            'poor_defense': 1.08
        }
    
    def get_matchup_adjustment(self, player_position: str, opponent_team: str, 
                                stat_type: str = 'pts') -> float:
        """
        Get adjustment factor for a player vs opponent.
        
        Args:
            player_position: 'PG', 'SG', 'SF', 'PF', 'C'
            opponent_team: 3-letter team abbreviation
            stat_type: 'pts', 'reb', 'ast', 'fg_pct'
            
        Returns:
            Multiplier (1.0 = no adjustment)
        """
        pos_defense = self.defense_scraper.get_position_defense(opponent_team, player_position)
        
        if stat_type == 'pts':
            pts_allowed = pos_defense.get('pts_per_100_allowed', 114.0)
            return pts_allowed / 114.0
        elif stat_type == 'fg_pct':
            fg_allowed = pos_defense.get('fg_pct_allowed', 0.470)
            return fg_allowed / 0.470
        elif stat_type == 'reb':
            dreb_pct = pos_defense.get('opp_dreb_pct', 0.750)
            return (1.0 - dreb_pct) / 0.250
        elif stat_type == 'ast':
            tov_pct = pos_defense.get('tov_pct_forced', 0.135)
            return 1.0 - (tov_pct - 0.135) * 2
        
        return 1.0
    
    def apply_matchup_adjustments(self, predictions: dict, opponent_team: str,
                                   position_map: dict) -> dict:
        """
        Apply defensive matchup adjustments to multiple player predictions.
        
        Args:
            predictions: Dict of player_name -> {pts, reb, ast, etc.}
            opponent_team: Team abbreviation
            position_map: Dict of player_name -> position
            
        Returns:
            Adjusted predictions dict
        """
        adjusted = {}
        
        for player_name, preds in predictions.items():
            position = position_map.get(player_name, 'SG')
            
            adj_pts = self.get_matchup_adjustment(position, opponent_team, 'pts')
            adj_reb = self.get_matchup_adjustment(position, opponent_team, 'reb')
            adj_ast = self.get_matchup_adjustment(position, opponent_team, 'ast')
            
            adjusted[player_name] = {
                'pts': preds.get('pts', 0) * adj_pts,
                'reb': preds.get('reb', 0) * adj_reb,
                'ast': preds.get('ast', 0) * adj_ast,
                'pts_adj_factor': adj_pts,
                'reb_adj_factor': adj_reb,
                'ast_adj_factor': adj_ast
            }
        
        return adjusted


    def get_player_vs_team_history(
        self, 
        player_id: int, 
        opponent_team: str,
        seasons: List[str] = None,
        last_n_games: int = 10
    ) -> dict:
        """
        Get historical performance of a player vs a specific team.
        
        This is KEY for accuracy - captures how LeBron specifically 
        performs against Boston, not just against "good defenses".
        
        Args:
            player_id: NBA player ID
            opponent_team: 3-letter team abbreviation
            seasons: List of seasons to include (defaults to last 3)
            last_n_games: Number of recent games to analyze
            
        Returns:
            Dictionary with:
                - avg_pts, avg_reb, avg_ast vs this team
                - fg_pct, 3pt_pct vs this team
                - games_played count
                - trend (improving/declining vs this opponent)
        """
        if seasons is None:
            current_year = datetime.now().year
            if datetime.now().month >= 10:
                seasons = [f"{current_year - i}-{str(current_year - i + 1)[2:]}" for i in range(3)]
            else:
                seasons = [f"{current_year - 1 - i}-{str(current_year - i)[2:]}" for i in range(3)]
        
        opponent_team = opponent_team.upper()
        cache_key = f"{player_id}_{opponent_team}_{last_n_games}"
        cache_file = os.path.join(self.cache_dir, f"player_vs_team_{cache_key}.json")
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=24):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        
        result = self._fetch_player_vs_team_stats(player_id, opponent_team, seasons, last_n_games)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception:
            pass
        
        return result
    
    def _fetch_player_vs_team_stats(
        self,
        player_id: int,
        opponent_team: str,
        seasons: List[str],
        last_n_games: int
    ) -> dict:
        """Fetch player vs team statistics from NBA API."""
        result = {
            'player_id': player_id,
            'opponent_team': opponent_team,
            'games_played': 0,
            'avg_pts': 0.0,
            'avg_reb': 0.0,
            'avg_ast': 0.0,
            'avg_min': 0.0,
            'fg_pct': 0.0,
            'fg3_pct': 0.0,
            'ft_pct': 0.0,
            'pts_std': 0.0,
            'recent_games': [],
            'trend': 'stable',
            'source': 'estimate'
        }
        
        opponent_id = ABBR_TO_ID.get(opponent_team)
        if not opponent_id:
            return result
        
        all_games = []
        
        for season in seasons:
            season_param = season.replace('-', '')
            
            url = f"https://stats.nba.com/stats/playergamelogs"
            params = {
                'PlayerID': player_id,
                'Season': season_param,
                'SeasonType': 'Regular Season',
                'MeasureType': 'Base'
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = self.defense_scraper._session.get(
                        url,
                        params=params,
                        headers=self.headers,
                        timeout=15
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    games = self._parse_player_games(data, opponent_id)
                    all_games.extend(games)
                    break
                    
                except Exception as e:
                    logger.debug(f"Player game log fetch attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
        
        if all_games:
            all_games.sort(key=lambda x: x.get('game_date', ''), reverse=True)
            recent_games = all_games[:last_n_games]
            
            if recent_games:
                result['games_played'] = len(recent_games)
                result['avg_pts'] = np.mean([g['pts'] for g in recent_games])
                result['avg_reb'] = np.mean([g['reb'] for g in recent_games])
                result['avg_ast'] = np.mean([g['ast'] for g in recent_games])
                result['avg_min'] = np.mean([g['min'] for g in recent_games])
                result['pts_std'] = np.std([g['pts'] for g in recent_games]) if len(recent_games) > 1 else 5.0
                
                total_fgm = sum(g['fgm'] for g in recent_games)
                total_fga = sum(g['fga'] for g in recent_games)
                total_fg3m = sum(g['fg3m'] for g in recent_games)
                total_fg3a = sum(g['fg3a'] for g in recent_games)
                total_ftm = sum(g['ftm'] for g in recent_games)
                total_fta = sum(g['fta'] for g in recent_games)
                
                result['fg_pct'] = total_fgm / total_fga if total_fga > 0 else 0.45
                result['fg3_pct'] = total_fg3m / total_fg3a if total_fg3a > 0 else 0.35
                result['ft_pct'] = total_ftm / total_fta if total_fta > 0 else 0.75
                
                result['recent_games'] = [
                    {
                        'game_date': g['game_date'],
                        'pts': g['pts'],
                        'reb': g['reb'],
                        'ast': g['ast'],
                        'min': g['min']
                    }
                    for g in recent_games[:5]
                ]
                
                if len(recent_games) >= 3:
                    recent_avg = np.mean([g['pts'] for g in recent_games[:3]])
                    older_avg = np.mean([g['pts'] for g in recent_games[3:]]) if len(recent_games) > 3 else recent_avg
                    
                    if recent_avg > older_avg * 1.15:
                        result['trend'] = 'improving'
                    elif recent_avg < older_avg * 0.85:
                        result['trend'] = 'declining'
                    else:
                        result['trend'] = 'stable'
                
                result['source'] = 'nba_api'
        
        return result
    
    def _parse_player_games(self, data: dict, opponent_id: int) -> List[dict]:
        """Parse game logs for games against specific opponent."""
        games = []
        
        try:
            result_sets = data.get('resultSets', [])
            for result_set in result_sets:
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                
                header_map = {h.lower(): i for i, h in enumerate(headers)}
                
                for row in rows:
                    try:
                        matchup = str(row[header_map.get('matchup', 5)]).upper()
                        
                        opponent_abbr = ID_TO_ABBR.get(opponent_id, '')
                        if opponent_abbr and opponent_abbr.upper() not in matchup:
                            continue
                        
                        def get_val(col_name, default=0):
                            idx = header_map.get(col_name.lower())
                            if idx is not None and idx < len(row):
                                try:
                                    return float(row[idx])
                                except (ValueError, TypeError):
                                    return default
                            return default
                        
                        games.append({
                            'game_date': row[header_map.get('game_date', 3)] if header_map.get('game_date') else '',
                            'pts': get_val('pts'),
                            'reb': get_val('reb'),
                            'ast': get_val('ast'),
                            'min': get_val('min'),
                            'fgm': get_val('fgm'),
                            'fga': get_val('fga'),
                            'fg3m': get_val('fg3m'),
                            'fg3a': get_val('fg3a'),
                            'ftm': get_val('ftm'),
                            'fta': get_val('fta')
                        })
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Error parsing player games: {e}")
        
        return games
    
    def get_team_switchability_rating(self, team_abbr: str, season: str = None) -> dict:
        """
        Get how well a team defends different positions via switching.
        
        High switchability means PG can be guarded by C (rare).
        Low switchability means they stick to traditional matchups.
        
        Returns:
            Dictionary with switch rating 0-100 and breakdown by position
        """
        if season is None:
            season = self._get_current_season()
        
        cache_file = os.path.join(self.cache_dir, f"switchability_{team_abbr}_{season}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        result = {
            'team': team_abbr.upper(),
            'season': season,
            'switch_rating': 50,
            'position_versatility': {
                'PG': 50, 'SG': 50, 'SF': 50, 'PF': 50, 'C': 50
            },
            'switch_tendency': 'moderate',
            'source': 'estimated'
        }
        
        team_switch_profiles = {
            'MIA': {'switch_rating': 85, 'switch_tendency': 'high'},
            'GSW': {'switch_rating': 80, 'switch_tendency': 'high'},
            'BOS': {'switch_rating': 78, 'switch_tendency': 'high'},
            'MIL': {'switch_rating': 45, 'switch_tendency': 'low'},
            'PHI': {'switch_rating': 50, 'switch_tendency': 'moderate'},
            'DEN': {'switch_rating': 55, 'switch_tendency': 'moderate'},
            'MEM': {'switch_rating': 70, 'switch_tendency': 'moderate-high'},
            'CLE': {'switch_rating': 60, 'switch_tendency': 'moderate'},
            'PHX': {'switch_rating': 65, 'switch_tendency': 'moderate'},
            'LAC': {'switch_rating': 68, 'switch_tendency': 'moderate-high'},
            'DAL': {'switch_rating': 40, 'switch_tendency': 'low'},
            'ATL': {'switch_rating': 45, 'switch_tendency': 'low'},
            'SAS': {'switch_rating': 55, 'switch_tendency': 'moderate'},
            'CHI': {'switch_rating': 50, 'switch_tendency': 'moderate'},
            'NOP': {'switch_rating': 58, 'switch_tendency': 'moderate'},
        }
        
        if team_abbr.upper() in team_switch_profiles:
            result.update(team_switch_profiles[team_abbr.upper()])
            result['source'] = 'historical_profile'
        
        result['position_versatility'] = {
            'PG': result['switch_rating'] + np.random.randint(-5, 10),
            'SG': result['switch_rating'] + np.random.randint(-5, 10),
            'SF': result['switch_rating'] + np.random.randint(-5, 10),
            'PF': result['switch_rating'] + np.random.randint(-10, 5),
            'C': result['switch_rating'] + np.random.randint(-15, 0)
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        
        return result
    
    def get_rim_protection_rating(self, team_abbr: str, season: str = None) -> dict:
        """
        Get team's rim protection effectiveness.
        
        Important for predicting FG% near basket and defensive rebounds.
        High rim protection = lower opponent FG% at rim, more blocks.
        
        Returns:
            Dictionary with rim protection metrics
        """
        if season is None:
            season = self._get_current_season()
        
        cache_file = os.path.join(self.cache_dir, f"rim_protection_{team_abbr}_{season}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        result = {
            'team': team_abbr.upper(),
            'season': season,
            'rim_fgpct_allowed': 0.58,
            'block_pct': 0.085,
            'opp_paint_pts_per_game': 45,
            'interior_def_rating': 50,
            'source': 'estimated'
        }
        
        interior_profiles = {
            'MIL': {'rim_fgpct_allowed': 0.54, 'block_pct': 0.11, 'interior_def_rating': 85},
            'CLE': {'rim_fgpct_allowed': 0.55, 'block_pct': 0.10, 'interior_def_rating': 82},
            'GSW': {'rim_fgpct_allowed': 0.59, 'block_pct': 0.07, 'interior_def_rating': 55},
            'BOS': {'rim_fgpct_allowed': 0.56, 'block_pct': 0.09, 'interior_def_rating': 78},
            'MEM': {'rim_fgpct_allowed': 0.57, 'block_pct': 0.095, 'interior_def_rating': 72},
            'MIA': {'rim_fgpct_allowed': 0.58, 'block_pct': 0.08, 'interior_def_rating': 65},
            'DEN': {'rim_fgpct_allowed': 0.57, 'block_pct': 0.085, 'interior_def_rating': 68},
            'PHX': {'rim_fgpct_allowed': 0.59, 'block_pct': 0.075, 'interior_def_rating': 58},
            'LAL': {'rim_fgpct_allowed': 0.56, 'block_pct': 0.09, 'interior_def_rating': 75},
            'ATL': {'rim_fgpct_allowed': 0.62, 'block_pct': 0.06, 'interior_def_rating': 35},
            'SAS': {'rim_fgpct_allowed': 0.61, 'block_pct': 0.065, 'interior_def_rating': 40},
            'DET': {'rim_fgpct_allowed': 0.62, 'block_pct': 0.06, 'interior_def_rating': 32},
        }
        
        if team_abbr.upper() in interior_profiles:
            result.update(interior_profiles[team_abbr.upper()])
            result['source'] = 'historical_profile'
        
        league_avg_rim_fgpct = 0.58
        result['opp_paint_pts_per_game'] = 50 - (result['interior_def_rating'] - 50) * 0.2
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        
        return result
    
    def get_comprehensive_matchup_factors(
        self,
        player_name: str,
        player_id: int,
        player_position: str,
        opponent_team: str,
        season: str = None
    ) -> dict:
        """
        Get all matchup factors for a player vs opponent.
        
        Combines:
        - Player vs Team history
        - Position defense
        - Rim protection (if shooting near basket)
        - Switchability (if perimeter player)
        
        Returns:
            Dictionary with adjustment factors and confidence
        """
        factors = {
            'player_name': player_name,
            'player_id': player_id,
            'opponent': opponent_team.upper(),
            'position': player_position,
            'overall_adjustment': 1.0,
            'pts_adjustment': 1.0,
            'reb_adjustment': 1.0,
            'ast_adjustment': 1.0,
            'fg_pct_adjustment': 1.0,
            'confidence': 0.5,
            'factors_used': []
        }
        
        pos_defense = self.get_position_defense(opponent_team, player_position, season)
        pos_adj = pos_defense.get('pts_per_100_allowed', 114.0) / 114.0
        factors['pts_adjustment'] *= pos_adj
        factors['factors_used'].append(f'position_defense_{player_position}')
        
        if player_id:
            player_history = self.get_player_vs_team_history(player_id, opponent_team)
            
            if player_history['games_played'] >= 3:
                hist_pts = player_history['avg_pts']
                hist_trend = player_history['trend']
                
                history_weight = min(player_history['games_played'] / 10.0, 0.5)
                history_adj = 1.0
                
                if hist_trend == 'improving':
                    history_adj = 1.05
                elif hist_trend == 'declining':
                    history_adj = 0.95
                
                factors['pts_adjustment'] = (
                    factors['pts_adjustment'] * (1 - history_weight) + 
                    history_adj * history_weight
                )
                factors['confidence'] = min(0.8, factors['confidence'] + 0.2)
                factors['factors_used'].append('player_vs_team_history')
        
        if player_position in ['PF', 'C']:
            rim_protection = self.get_rim_protection_rating(opponent_team, season)
            rim_adj = rim_protection['rim_fgpct_allowed'] / 0.58
            factors['fg_pct_adjustment'] *= rim_adj
            factors['reb_adjustment'] *= (2 - rim_adj)
            factors['factors_used'].append('rim_protection')
        
        else:
            switchability = self.get_team_switchability_rating(opponent_team, season)
            switch_adj = 1.0 - (switchability['switch_rating'] - 50) * 0.002
            factors['pts_adjustment'] *= switch_adj
            factors['factors_used'].append('switchability')
        
        factors['overall_adjustment'] = (
            factors['pts_adjustment'] * 0.5 +
            factors['reb_adjustment'] * 0.25 +
            factors['ast_adjustment'] * 0.25
        )
        
        return factors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = NBADefenseScraper()
    analyzer = DefensiveMatchupAnalyzer(scraper)
    
    print("Testing NBA Defense Scraper...")
    
    defense = scraper.get_team_defense_allowed('BOS')
    print(f"\nBoston Celtics Defense:")
    for k, v in defense.items():
        print(f"  {k}: {v}")
    
    pos_defense = scraper.get_position_defense('LAL')
    print(f"\nLakers Position Defense:")
    for pos, data in pos_defense.items():
        if isinstance(data, dict) and 'pts_per_100_allowed' in data:
            print(f"  {pos}: {data.get('pts_per_100_allowed', 0):.1f} pts/100 allowed")
    
    adj = analyzer.get_matchup_adjustment('PG', 'MEM', 'pts')
    print(f"\nPG vs Memphis adjustment: {adj:.3f}x")