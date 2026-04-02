import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.utils.team_mappings import normalize_team

logger = logging.getLogger(__name__)

# Injury status to probability mapping
INJURY_PROBABILITY_MAP = {
    'OUT': 0.00,
    'OUT FOR SEASON': 0.00,
    'OUT INDEFINITELY': 0.00,
    'SUSPENDED': 0.00,
    'DOUBTFUL': 0.25,
    'QUESTIONABLE': 0.50,
    'GTD': 0.50,
    'GAME TIME DECISION': 0.50,
    'DAY-TO-DAY': 0.50,
    'DAY TO DAY': 0.50,
    'PROBABLE': 0.85,
    'AVAILABLE': 1.00,
    'ACTIVE': 1.00,
}


class InjuryScraper:
    """
    Robust NBA injury information scraper with multiple data sources and fallback strategies.
    Primary source: ESPN
    Fallback sources: ESPN API, Cached data
    """
    
    def __init__(self, cache_dir='data/cache', config: Optional[Any] = None):
        self._config = config
        self.cache_dir = cache_dir
        self.last_fetch_status: Dict[str, Any] = {}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._cache_timestamp = None
        self._cached_df = None
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': self._get_config_value('http.user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        })

    def _set_last_fetch_status(
        self,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_fetch_status = {
            'source_key': 'injury',
            'status': status,
            'required': False,
            'message': message,
            'details': details or {},
        }

    def get_last_fetch_status(self) -> Dict[str, Any]:
        return dict(self.last_fetch_status)
    
    @property
    def URL(self) -> str:
        return self._get_config_value('api.espn_injuries_url', 'https://www.espn.com/nba/injuries')
    
    @property
    def API_URL(self) -> str:
        return self._get_config_value('api.espn_health_api_url', 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/health')
    
    @property
    def CACHE_TTL_MINUTES(self) -> float:
        return self._get_config_value('cache.injury_ttl_minutes', 30.0)
    
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
            
    def fetch_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetches the latest injury report with retry logic and multiple fallback strategies.
        Tries in order: ESPN HTML -> ESPN API -> Cached data
        """
        if not force_refresh and self._is_cache_valid():
            logger.debug("Using in-memory cached injury data")
            if self._cached_df is not None:
                self._set_last_fetch_status(
                    'success',
                    "Using in-memory injury cache",
                    {
                        'source': 'memory_cache',
                        'rows': int(len(self._cached_df)),
                    },
                )
                return self._cached_df.copy()
        
        df = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Fetching injury data (attempt {attempt + 1}/{self.MAX_RETRIES})...")
                
                # Try HTML parsing first
                df = self._fetch_from_html()
                if df is not None and not df.empty:
                    self._set_last_fetch_status(
                        'success',
                        "Fetched injury data from ESPN HTML",
                        {
                            'source': 'espn_html',
                            'rows': int(len(df)),
                        },
                    )
                    break
                
                # Fallback to API
                logger.warning("HTML parsing failed, trying API fallback...")
                df = self._fetch_from_api()
                if df is not None and not df.empty:
                    self._set_last_fetch_status(
                        'fallback',
                        "Fetched injury data from ESPN API fallback",
                        {
                            'source': 'espn_api',
                            'rows': int(len(df)),
                        },
                    )
                    break
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
        
        if df is None or df.empty:
            logger.error("All fetch attempts failed, loading from cache...")
            df = self._load_from_cache()
            if df is not None and not df.empty:
                self._set_last_fetch_status(
                    'fallback',
                    "Loaded injury data from disk cache after fetch failure",
                    {
                        'source': 'disk_cache',
                        'rows': int(len(df)),
                    },
                )
            else:
                self._set_last_fetch_status(
                    'failed',
                    "Unable to load injury data from live sources or cache",
                    {'source': 'espn'},
                )
        else:
            if not df.empty:
                df['TEAM_ABBR'] = df['TEAM'].apply(normalize_team)
            self._cached_df = df
            self._cache_timestamp = datetime.now()
            self._save_to_cache(df)
            logger.info(f"Fetched {len(df)} injury records")
        
        return df if df is not None else pd.DataFrame()
    
    def _fetch_from_html(self) -> Optional[pd.DataFrame]:
        """Try to fetch injury data from ESPN HTML page."""
        try:
            response = self._session.get(self.URL, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            injury_data = []
            
            # Try multiple parsing strategies
            # Strategy 1: Look for team sections and tables
            team_sections = soup.find_all('div', class_='Table__Title')
            tables = soup.find_all('table', class_='Table')
            
            if len(team_sections) == len(tables):
                for team_div, table in zip(team_sections, tables):
                    team_name = team_div.text.strip()
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            injury_data.append(self._parse_injury_row(cols, team_name))
            else:
                # Strategy 2: Look for any table with injury data
                all_tables = soup.find_all('table')
                for table in all_tables:
                    rows = table.find_all('tr')
                    if len(rows) < 2:
                        continue
                    
                    # Try to extract team name from surrounding context
                    team_name = table.find_previous(['div', 'h2', 'h3'])
                    if team_name:
                        team_name = team_name.text.strip()
                    else:
                        team_name = "Unknown"
                    
                    for row in rows[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            injury_data.append(self._parse_injury_row(cols, team_name))
            
            if injury_data:
                return pd.DataFrame(injury_data)
            return None
            
        except Exception as e:
            logger.debug(f"HTML parsing failed: {e}")
            return None
    
    def _parse_injury_row(self, cols, team_name: str) -> Dict:
        """Parse a single injury row from HTML table."""
        player_name = cols[0].text.strip()
        status = cols[1].text.strip().upper()
        date = cols[2].text.strip()
        comment = cols[3].text.strip() if len(cols) > 3 else ""
        
        return {
            'TEAM': team_name,
            'PLAYER': player_name,
            'STATUS': status,
            'PLAY_PROBABILITY': self._status_to_probability(status),
            'DATE': date,
            'COMMENT': comment,
            'FETCHED_AT': datetime.now().isoformat()
        }
    
    def _fetch_from_api(self) -> Optional[pd.DataFrame]:
        """Try to fetch injury data from ESPN API."""
        try:
            response = self._session.get(self.API_URL, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            injury_data = []
            
            for team_data in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                team_name = team_data.get('displayName', 'Unknown')
                for athlete in team_data.get('athletes', []):
                    athlete_info = athlete.get('injury', {})
                    injury_data.append({
                        'TEAM': team_name,
                        'PLAYER': athlete.get('displayName', ''),
                        'STATUS': athlete_info.get('status', 'UNKNOWN').upper(),
                        'PLAY_PROBABILITY': self._status_to_probability(athlete_info.get('status', 'UNKNOWN')),
                        'DATE': athlete_info.get('date', ''),
                        'COMMENT': athlete_info.get('comments', ''),
                        'FETCHED_AT': datetime.now().isoformat()
                    })
            
            if injury_data:
                return pd.DataFrame(injury_data)
            return None
            
        except Exception as e:
            logger.debug(f"API fetching failed: {e}")
            return None
    
    def _status_to_probability(self, status: str) -> float:
        """Convert injury status string to play probability."""
        status_upper = status.upper().strip()
        
        # Direct match
        if status_upper in INJURY_PROBABILITY_MAP:
            return INJURY_PROBABILITY_MAP[status_upper]
        
        # Partial match for compound statuses
        for key, prob in INJURY_PROBABILITY_MAP.items():
            if key in status_upper:
                return prob
        
        # Unknown status - assume they might play
        logger.debug(f"Unknown injury status: '{status}', defaulting to 0.70 probability")
        return 0.70
    
    def _is_cache_valid(self) -> bool:
        """Check if in-memory cache is still valid."""
        if self._cached_df is None or self._cache_timestamp is None:
            return False
        age = datetime.now() - self._cache_timestamp
        return age < timedelta(minutes=self.CACHE_TTL_MINUTES)

    def _save_to_cache(self, df: pd.DataFrame):
        cache_path = os.path.join(self.cache_dir, 'latest_injuries.csv')
        df.to_csv(cache_path, index=False)
        logger.debug(f"Injury data cached to {cache_path}")

    def _load_from_cache(self) -> pd.DataFrame:
        cache_path = os.path.join(self.cache_dir, 'latest_injuries.csv')
        if os.path.exists(cache_path):
            logger.info(f"Loading cached injury data from {cache_path}")
            df = pd.read_csv(cache_path)
            self._cached_df = df
            return df
        return pd.DataFrame()

    def get_injured_players(self, team_abbr: str = None) -> List[str]:
        """Returns a list of names for players currently listed as OUT or similar."""
        df = self.fetch_injuries()
        if df.empty:
            return []
            
        # Filter for significant injuries (OUT, Doubtful)
        critically_injured = df[df['STATUS'].str.upper().isin(['OUT', 'DOUBTFUL', 'OUT FOR SEASON', 'SUSPENDED'])]
        
        if team_abbr:
            norm_abbr = normalize_team(team_abbr)
            return critically_injured[critically_injured['TEAM_ABBR'] == norm_abbr]['PLAYER'].tolist()
            
        return critically_injured['PLAYER'].tolist()
    
    def get_player_availability(self, team_abbr: str = None) -> Dict[str, float]:
        """
        Returns a dict mapping player names to their play probability.
        
        Args:
            team_abbr: Optional team filter
            
        Returns:
            Dict[player_name, probability] - only includes injured players
            (healthy players not on injury report are assumed 1.0)
        """
        df = self.fetch_injuries()
        if df.empty:
            return {}
        
        if team_abbr:
            norm_abbr = normalize_team(team_abbr)
            df = df[df['TEAM_ABBR'] == norm_abbr]
        
        return dict(zip(df['PLAYER'], df['PLAY_PROBABILITY']))
    
    def get_injury_details(self, player_name: str) -> Optional[Dict]:
        """Get full injury details for a specific player."""
        df = self.fetch_injuries()
        if df.empty:
            return None
        
        # Fuzzy match on player name
        matches = df[df['PLAYER'].str.lower().str.contains(player_name.lower())]
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        return {
            'player': row['PLAYER'],
            'team': row['TEAM'],
            'team_abbr': row['TEAM_ABBR'],
            'status': row['STATUS'],
            'play_probability': row['PLAY_PROBABILITY'],
            'date': row['DATE'],
            'comment': row['COMMENT']
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = InjuryScraper()
    df = scraper.fetch_injuries()
    print(df.head(10))
    print(f"\nTotal Injuries Found: {len(df)}")
    print(f"\nPlay Probability Distribution:")
    print(df['PLAY_PROBABILITY'].value_counts().sort_index())
