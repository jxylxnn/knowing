"""
Basketball Reference Scraper for NBA Team Statistics.
Scrapes team pace, offensive/defensive rating, and four factors data.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re

from src.utils.team_mappings import TEAMS, get_bref_abbr

logger = logging.getLogger(__name__)

BREF_TO_TEAM_ABBR = {
    team_info['bref_abbr']: team_abbr
    for team_abbr, team_info in TEAMS.items()
}
TEAM_ABBR_TO_BREF = {
    team_abbr: team_info['bref_abbr']
    for team_abbr, team_info in TEAMS.items()
}


class BasketballRefScraper:
    """
    Scrapes basketball-reference.com for advanced team statistics.
    Focuses on pace, offensive/defensive rating, and four factors.
    """
    
    def __init__(self, cache_dir: str = 'data/cache', config: Optional[Any] = None):
        self._config = config
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._session = requests.Session()
        self._session.headers.update(self._get_headers())
        self._team_stats_cache: Dict[str, dict] = {}
        self._cache_timestamp: Optional[datetime] = None
        
        self.max_retries = self._get_config_value('http.max_retries', 3)
        self.retry_delay = self._get_config_value('http.retry_delay', 3.0)
        self.cache_ttl_hours = self._get_config_value('cache.basketball_ref_ttl_hours', 6.0)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers from config or use defaults."""
        if self._config and hasattr(self._config, 'http'):
            return {
                'User-Agent': getattr(self._config.http, 'user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    @property
    def BASE_URL(self) -> str:
        return self._get_config_value('api.basketball_reference_base_url', 'https://www.basketball-reference.com')
    
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
        
    def get_team_stats(self, team_abbr: str, season: str = None) -> dict:
        """
        Get comprehensive team statistics for a specific team.
        
        Args:
            team_abbr: 3-letter team abbreviation (e.g., 'BOS', 'LAL')
            season: Season in format '2024-25' (defaults to current season)
            
        Returns:
            Dictionary with pace, offensive_rating, defensive_rating, four_factors
        """
        if season is None:
            current_year = datetime.now().year
            if datetime.now().month >= 10:
                season = f"{current_year}-{str(current_year + 1)[2:]}"
            else:
                season = f"{current_year - 1}-{str(current_year)[2:]}"
        
        bref_abbr = TEAM_ABBR_TO_BREF.get(team_abbr.upper(), team_abbr.upper())
        cache_key = f"{bref_abbr}_{season}"
        
        if cache_key in self._team_stats_cache:
            return self._team_stats_cache[cache_key]
        
        csv_path = os.path.join(self.cache_dir, f"bref_{cache_key}.csv")
        if os.path.exists(csv_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(csv_path))
            if datetime.now() - file_time < timedelta(hours=self.cache_ttl_hours):
                try:
                    df = pd.read_csv(csv_path)
                    result = df.to_dict('records')[0] if not df.empty else {}
                    self._team_stats_cache[cache_key] = result
                    return result
                except Exception as e:
                    logger.debug(f"Failed to load cache: {e}")
        
        result = self._fetch_team_stats(bref_abbr, season)
        
        if result:
            self._team_stats_cache[cache_key] = result
            try:
                pd.DataFrame([result]).to_csv(csv_path, index=False)
            except Exception as e:
                logger.debug(f"Failed to save cache: {e}")
        
        return result
    
    def _fetch_team_stats(self, bref_abbr: str, season: str) -> dict:
        """Fetch team stats from basketball-reference team page."""
        url = f"{self.BASE_URL}/teams/{bref_abbr}/{season.replace('-', '')}.html"
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                result = {
                    'team': BREF_TO_TEAM_ABBR.get(bref_abbr, bref_abbr),
                    'season': season,
                    'fetched_at': datetime.now().isoformat()
                }
                
                misc_table = soup.find('table', {'id': 'team_misc'})
                if misc_table:
                    result.update(self._parse_misc_table(misc_table))
                
                advanced_div = soup.find('div', {'id': 'all_team_and_opponent'})
                if advanced_div:
                    result.update(self._parse_team_opponent_tables(advanced_div, soup))
                
                pace_table = soup.find('table', {'id': 'team_stats'})
                if pace_table:
                    result.update(self._parse_pace_from_team_stats(pace_table))
                
                result.update(self._extract_four_factors(soup))
                
                if result.get('pace') or result.get('offensive_rating'):
                    logger.info(f"Successfully fetched stats for {bref_abbr}")
                    return result
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {bref_abbr}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Error parsing {bref_abbr}: {e}")
                break
        
        logger.warning(f"Could not fetch stats for {bref_abbr}, using defaults")
        return self._get_defaults(bref_abbr)
    
    def _parse_misc_table(self, table) -> dict:
        """Parse the miscellaneous stats table."""
        result = {}
        try:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) < 2:
                    continue
                    
                header = cells[0].get_text(strip=True).lower()
                value_text = cells[1].get_text(strip=True)
                
                try:
                    value = float(value_text)
                except ValueError:
                    continue
                    
                if 'pace' in header:
                    result['pace'] = value
                elif 'offensive rating' in header or 'orating' in header:
                    result['offensive_rating'] = value
                elif 'defensive rating' in header or 'drating' in header:
                    result['defensive_rating'] = value
        except Exception as e:
            logger.debug(f"Error parsing misc table: {e}")
        
        return result
    
    def _parse_team_opponent_tables(self, div, soup) -> dict:
        """Parse team and opponent stats tables."""
        result = {}
        
        team_table = soup.find('table', {'id': 'team-stats-base'})
        opp_table = soup.find('table', {'id': 'opponent-stats-base'})
        
        if team_table:
            result.update(self._parse_base_stats(team_table, 'team'))
        if opp_table:
            result.update(self._parse_base_stats(opp_table, 'opponent'))
            
        return result
    
    def _parse_base_stats(self, table, prefix: str) -> dict:
        """Parse base stats from a table."""
        result = {}
        try:
            footer = table.find('tfoot')
            if footer:
                row = footer.find('tr')
                if row:
                    cells = row.find_all('td')
                    headers = [th.get_text(strip=True).lower() for th in table.find('thead').find_all('th')]
                    
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            header = headers[i]
                            try:
                                value = float(cell.get_text(strip=True))
                                
                                if 'fg' in header and 'a' in header and prefix == 'team':
                                    if 'fg%' not in header and '3p%' not in header:
                                        result['fga_per_game'] = value
                                elif 'tov' in header and prefix == 'team':
                                    result['tov_per_game'] = value
                                elif 'orb' in header and prefix == 'team':
                                    result['orb_per_game'] = value
                                elif 'ft' in header and 'a' in header and prefix == 'team':
                                    result['fta_per_game'] = value
                            except ValueError:
                                pass
        except Exception as e:
            logger.debug(f"Error parsing base stats: {e}")
        
        return result
    
    def _parse_pace_from_team_stats(self, table) -> dict:
        """Extract pace from team stats table."""
        result = {}
        try:
            footer = table.find('tfoot')
            if footer:
                cells = footer.find_all('td')
                pace_found = False
                for cell in cells:
                    text = cell.get_text(strip=True)
                    try:
                        val = float(text)
                        if not pace_found and 90 < val < 110:
                            result['possessions_per_game'] = val
                            pace_found = True
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"Error parsing pace: {e}")
        
        return result
    
    def _extract_four_factors(self, soup) -> dict:
        """Extract four factors data."""
        result = {}
        
        try:
            for table_id in ['team_misc', 'team_stats', 'opponent_stats']:
                table = soup.find('table', {'id': table_id})
                if not table:
                    continue
                    
                footers = table.find_all('tfoot')
                for footer in footers:
                    rows = footer.find_all('tr')
                    for row in rows:
                        th = row.find('th')
                        if th:
                            row_header = th.get_text(strip=True).lower()
                        else:
                            cells = row.find_all('td')
                            
                            for cell in cells:
                                text = cell.get_text(strip=True)
                                try:
                                    val = float(text)
                                except ValueError:
                                    continue
                                
                                data_stat = cell.get('data-stat', '')
                                
                                if 'efg' in data_stat or 'effective' in text.lower():
                                    if 0.4 < val < 0.7:
                                        result['efg_pct'] = val
                                elif 'tov' in data_stat and 'pct' in data_stat:
                                    if 0.08 < val < 0.20:
                                        result['tov_pct'] = val
                                elif 'orb' in data_stat and 'pct' in data_stat:
                                    if 0.15 < val < 0.35:
                                        result['orb_pct'] = val
                                elif 'ft' in data_stat and 'rate' in data_stat:
                                    if 0.15 < val < 0.35:
                                        result['ft_rate'] = val
        except Exception as e:
            logger.debug(f"Error extracting four factors: {e}")
        
        return result
    
    def _get_defaults(self, bref_abbr: str) -> dict:
        """Return league average defaults when scraping fails."""
        return {
            'team': BREF_TO_TEAM_ABBR.get(bref_abbr, bref_abbr),
            'pace': 100.0,
            'offensive_rating': 114.0,
            'defensive_rating': 114.0,
            'efg_pct': 0.54,
            'tov_pct': 0.135,
            'orb_pct': 0.25,
            'ft_rate': 0.23,
            'source': 'default'
        }
    
    def get_all_team_stats(self, season: str = None) -> pd.DataFrame:
        """
        Get stats for all teams in a season.
        
        Returns:
            DataFrame with one row per team
        """
        all_stats = []
        
        for team_abbr in TEAM_ABBR_TO_BREF.keys():
            try:
                stats = self.get_team_stats(team_abbr, season)
                if stats:
                    all_stats.append(stats)
                time.sleep(1.5)
            except Exception as e:
                logger.warning(f"Failed to get stats for {team_abbr}: {e}")
        
        if not all_stats:
            return pd.DataFrame()
        
        return pd.DataFrame(all_stats)
    
    def get_league_pace(self, season: str = None) -> float:
        """Get league average pace for a season."""
        url = f"{self.BASE_URL}/leagues/NBA_{season.replace('-', '')}_ratings.html"
        
        try:
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table', {'id': 'ratings'})
            
            if table:
                footer = table.find('tfoot')
                if footer:
                    for td in footer.find_all('td'):
                        text = td.get_text(strip=True)
                        try:
                            val = float(text)
                            if 95 < val < 105:
                                return val
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug(f"Could not fetch league pace: {e}")
        
        return 100.0
    
    def get_matchup_pace(self, team_a_abbr: str, team_b_abbr: str, season: str = None) -> float:
        """
        Estimate combined pace for a matchup between two teams.
        Uses average of both teams' pace with adjustment.
        """
        stats_a = self.get_team_stats(team_a_abbr, season)
        stats_b = self.get_team_stats(team_b_abbr, season)
        
        pace_a = stats_a.get('pace', 100.0)
        pace_b = stats_b.get('pace', 100.0)
        
        combined_pace = (pace_a + pace_b) / 2
        
        if combined_pace > 103:
            combined_pace = 103 + (combined_pace - 103) * 0.5
        elif combined_pace < 97:
            combined_pace = 97 - (97 - combined_pace) * 0.5
        
        return combined_pace
    
    def get_team_efficiency_diff(self, team_abbr: str, season: str = None) -> float:
        """
        Get the net rating (offensive - defensive) for a team.
        Positive = good team, negative = bad team.
        """
        stats = self.get_team_stats(team_abbr, season)
        
        ortg = stats.get('offensive_rating', 114.0)
        drtg = stats.get('defensive_rating', 114.0)
        
        return ortg - drtg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = BasketballRefScraper()
    
    print("Testing Basketball Reference Scraper...")
    
    stats = scraper.get_team_stats('BOS')
    print(f"\nBoston Celtics Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    pace = scraper.get_matchup_pace('BOS', 'LAL')
    print(f"\nBOS vs LAL Combined Pace: {pace:.1f}")
    
    diff = scraper.get_team_efficiency_diff('BOS')
    print(f"BOS Net Rating: {diff:.1f}")
