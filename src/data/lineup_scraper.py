"""
Lineup Scraper for NBA Starting Lineups.
Scrapes confirmed and projected starting lineups from multiple sources.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
import re

from src.utils.team_mappings import ID_TO_ABBR, ABBR_TO_ID, normalize_team

logger = logging.getLogger(__name__)


class LineupScraper:
    """
    Scrapes NBA starting lineups from multiple sources with reliability scoring.
    """
    
    def __init__(self, cache_dir: str = 'data/cache', config: Optional[Any] = None):
        self._config = config
        self.cache_dir = cache_dir
        self.last_fetch_status: Dict[str, Any] = {}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._session = requests.Session()
        self.nba_headers = self._get_headers()
        self._session.headers.update(self.nba_headers)
        
        self.max_retries = self._get_config_value('http.max_retries', 3)
        self.retry_delay = self._get_config_value('http.retry_delay', 2.0)
        self.cache_ttl_hours = self._get_config_value('cache.lineup_ttl_hours', 6.0)
        self._lineup_cache: Dict[str, dict] = {}
        self._coach_tendencies: Dict[str, dict] = {}
        self._load_coach_tendencies()

    def _set_last_fetch_status(
        self,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_fetch_status = {
            'source_key': 'lineup',
            'status': status,
            'required': False,
            'message': message,
            'details': details or {},
        }

    def get_last_fetch_status(self) -> Dict[str, Any]:
        return dict(self.last_fetch_status)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers from config or use defaults."""
        if self._config and hasattr(self._config, 'http'):
            return {
                'User-Agent': getattr(self._config.http, 'user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
                'Accept': 'application/json, text/html',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nba.com/',
                'x-nba-stats-origin': 'stats',
                'x-nba-stats-token': 'true',
                'Origin': 'https://www.nba.com',
            }
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nba.com/',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Origin': 'https://www.nba.com',
        }
    
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
        
    def _load_coach_tendencies(self):
        """Load cached coach tendencies from disk."""
        tendency_file = os.path.join(self.cache_dir, 'coach_tendencies.json')
        if os.path.exists(tendency_file):
            try:
                with open(tendency_file, 'r') as f:
                    self._coach_tendencies = json.load(f)
                logger.info(f"Loaded coach tendencies for {len(self._coach_tendencies)} teams")
            except Exception as e:
                logger.debug(f"Failed to load coach tendencies: {e}")
        
    def _save_coach_tendencies(self):
        """Save coach tendencies to disk."""
        tendency_file = os.path.join(self.cache_dir, 'coach_tendencies.json')
        try:
            with open(tendency_file, 'w') as f:
                json.dump(self._coach_tendencies, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Failed to save coach tendencies: {e}")

    def get_starting_lineup(
        self, 
        team_abbr: str, 
        game_date: str = None,
        include_projected: bool = True
    ) -> dict:
        """
        Get starting lineup for a team on a specific date.
        
        Args:
            team_abbr: 3-letter team abbreviation
            game_date: Date in 'YYYY-MM-DD' format (defaults to today)
            include_projected: If True, return projected lineup if confirmed not available
            
        Returns:
            Dictionary with:
                - starters: List of player names
                - starter_ids: List of player IDs
                - status: 'confirmed', 'projected', or 'inferred'
                - confidence: 0.0 to 1.0
                - source: Where the data came from
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
            
        team_abbr = team_abbr.upper()
        cache_key = f"{team_abbr}_{game_date}"
        
        if cache_key in self._lineup_cache:
            cached = self._lineup_cache[cache_key]
            self._set_last_fetch_status(
                'success' if cached.get('health_status') == 'success' else cached.get('health_status', 'fallback'),
                f"Using in-memory lineup cache for {team_abbr}",
                {
                    'team': team_abbr,
                    'game_date': game_date,
                    'source': 'memory_cache',
                    'lineup_source': cached.get('source'),
                },
            )
            return cached
        
        cache_file = os.path.join(self.cache_dir, f"lineup_{cache_key}.json")
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.cache_ttl_hours):
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                        self._lineup_cache[cache_key] = result
                        self._set_last_fetch_status(
                            result.get('health_status', 'success'),
                            f"Loaded cached lineup for {team_abbr}",
                            {
                                'team': team_abbr,
                                'game_date': game_date,
                                'source': 'disk_cache',
                                'lineup_source': result.get('source'),
                            },
                        )
                        return result
                except Exception as e:
                    logger.debug(f"Failed to load lineup cache: {e}")
        
        result = self._fetch_starting_lineup(team_abbr, game_date, include_projected)
        
        if result and result.get('starters'):
            self._lineup_cache[cache_key] = result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as e:
                logger.debug(f"Failed to save lineup cache: {e}")
        
        return result
    
    def _fetch_starting_lineup(
        self, 
        team_abbr: str, 
        game_date: str,
        include_projected: bool
    ) -> dict:
        """
        Fetch starting lineup from multiple sources with fallback logic.
        """
        result = {
            'team': team_abbr,
            'game_date': game_date,
            'starters': [],
            'starter_ids': [],
            'status': 'unknown',
            'confidence': 0.0,
            'source': None,
            'fetched_at': datetime.now().isoformat()
        }
        
        lineup = self._fetch_from_nba_stats(team_abbr, game_date)
        if lineup and len(lineup.get('starters', [])) >= 5:
            lineup['status'] = 'confirmed'
            lineup['confidence'] = 0.95
            lineup['health_status'] = 'success'
            lineup['health_message'] = f"Confirmed lineup found for {team_abbr}"
            self._set_last_fetch_status(
                'success',
                lineup['health_message'],
                {
                    'team': team_abbr,
                    'game_date': game_date,
                    'source': lineup.get('source'),
                },
            )
            return lineup
        
        if include_projected:
            projected = self._fetch_projected_lineup(team_abbr, game_date)
            if projected and len(projected.get('starters', [])) >= 5:
                projected['status'] = 'projected'
                projected['confidence'] = 0.75
                projected['health_status'] = 'fallback'
                projected['health_message'] = f"Using projected lineup fallback for {team_abbr}"
                self._set_last_fetch_status(
                    'fallback',
                    projected['health_message'],
                    {
                        'team': team_abbr,
                        'game_date': game_date,
                        'source': projected.get('source'),
                    },
                )
                return projected
        
        inferred = self._infer_lineup_from_history(team_abbr, game_date)
        if inferred and len(inferred.get('starters', [])) >= 5:
            inferred['status'] = 'inferred'
            inferred['confidence'] = 0.55
            inferred['health_status'] = 'fallback'
            inferred['health_message'] = f"Using inferred lineup fallback for {team_abbr}"
            self._set_last_fetch_status(
                'fallback',
                inferred['health_message'],
                {
                    'team': team_abbr,
                    'game_date': game_date,
                    'source': inferred.get('source'),
                },
            )
            return inferred
        
        result['source'] = 'failed'
        result['health_status'] = 'failed'
        result['health_message'] = f"Unable to determine lineup for {team_abbr}"
        self._set_last_fetch_status(
            'failed',
            result['health_message'],
            {
                'team': team_abbr,
                'game_date': game_date,
                'source': 'lineup_sources',
            },
        )
        return result
    
    def _fetch_from_nba_stats(self, team_abbr: str, game_date: str) -> Optional[dict]:
        """
        Fetch confirmed starters from NBA.com stats API.
        Uses the game book / boxscore endpoint.
        """
        team_id = ABBR_TO_ID.get(team_abbr.upper())
        if not team_id:
            return None
            
        season = self._get_season_for_date(game_date)
        season_param = season.replace('-', '')
        
        game_id = self._find_game_id(team_abbr, game_date)
        if not game_id:
            logger.debug(f"No game found for {team_abbr} on {game_date}")
            return None
        
        url = f"https://stats.nba.com/stats/boxscoretraditionalv3"
        params = {
            'GameID': game_id,
            'LeagueID': '00',
            'endPeriod': 0,
            'endRange': 0,
            'rangeType': 0,
            'startPeriod': 0,
            'startRange': 0
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.get(
                    url, 
                    params=params,
                    headers=self.nba_headers,
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                
                starters = self._parse_boxscore_starters(data, team_abbr)
                if starters:
                    return {
                        'team': team_abbr,
                        'game_date': game_date,
                        'starters': [s['name'] for s in starters],
                        'starter_ids': [s['id'] for s in starters],
                        'source': 'nba_stats_confirmed',
                        'positions': [s.get('position', 'Unknown') for s in starters]
                    }
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"NBA stats attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.debug(f"Error parsing NBA stats: {e}")
                break
        
        return None
    
    def _find_game_id(self, team_abbr: str, game_date: str) -> Optional[str]:
        """Find the game ID for a team on a specific date."""
        season = self._get_season_for_date(game_date)
        season_param = season.replace('-', '')
        team_id = ABBR_TO_ID.get(team_abbr.upper())
        
        url = "https://stats.nba.com/stats/scoreboardV3"
        params = {
            'LeagueID': '00',
            'OriginalDate': game_date
        }
        
        try:
            response = self._session.get(
                url, 
                params=params,
                headers=self.nba_headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            games = data.get('scoreboard', {}).get('games', [])
            for game in games:
                home_id = game.get('homeTeam', {}).get('teamId')
                away_id = game.get('awayTeam', {}).get('teamId')
                
                if str(home_id) == str(team_id) or str(away_id) == str(team_id):
                    return game.get('gameId')
                    
        except Exception as e:
            logger.debug(f"Error finding game ID: {e}")
        
        return None
    
    def _parse_boxscore_starters(self, data: dict, team_abbr: str) -> List[dict]:
        """Parse starters from boxscore response."""
        starters = []
        
        try:
            players = data.get('boxScoreTraditional', {}).get('playerStats', [])
            
            for player in players:
                if player.get('teamTricode', '').upper() != team_abbr.upper():
                    continue
                    
                if player.get('status') == 'STARTER' or str(player.get('position', '')).startswith('Start'):
                    starters.append({
                        'id': player.get('personId'),
                        'name': player.get('firstName', '') + ' ' + player.get('familyName', ''),
                        'position': player.get('position', 'Unknown')
                    })
            
            if len(starters) < 5:
                team_players = [p for p in players if p.get('teamTricode', '').upper() == team_abbr.upper()]
                team_players.sort(key=lambda x: float(x.get('minutes', '0:00').split(':')[0] or 0), reverse=True)
                starters = []
                for p in team_players[:5]:
                    mins_str = p.get('minutes', '0:00')
                    mins = float(mins_str.split(':')[0]) if ':' in mins_str else 0
                    if mins >= 15:
                        starters.append({
                            'id': p.get('personId'),
                            'name': p.get('firstName', '') + ' ' + p.get('familyName', ''),
                            'position': p.get('position', 'Unknown')
                        })
                        
        except Exception as e:
            logger.debug(f"Error parsing boxscore: {e}")
        
        return starters

    def _fetch_projected_lineup(self, team_abbr: str, game_date: str) -> Optional[dict]:
        """
        Fetch projected starting lineup from Rotowire/ESPN.
        """
        result = self._fetch_from_rotowire(team_abbr, game_date)
        if result and len(result.get('starters', [])) >= 5:
            return result
        
        result = self._fetch_from_espn(team_abbr, game_date)
        if result and len(result.get('starters', [])) >= 5:
            return result
        
        return None
    
    def _fetch_from_rotowire(self, team_abbr: str, game_date: str) -> Optional[dict]:
        """Scrape projected lineups from Rotowire."""
        slug_map = {
            'ATL': 'atlanta-hawks', 'BOS': 'boston-celtics', 'BKN': 'brooklyn-nets',
            'CHA': 'charlotte-hornets', 'CHI': 'chicago-bulls', 'CLE': 'cleveland-cavaliers',
            'DAL': 'dallas-mavericks', 'DEN': 'denver-nuggets', 'DET': 'detroit-pistons',
            'GSW': 'golden-state-warriors', 'HOU': 'houston-rockets', 'IND': 'indiana-pacers',
            'LAC': 'los-angeles-clippers', 'LAL': 'los-angeles-lakers', 'MEM': 'memphis-grizzlies',
            'MIA': 'miami-heat', 'MIL': 'milwaukee-bucks', 'MIN': 'minnesota-timberwolves',
            'NOP': 'new-orleans-pelicans', 'NYK': 'new-york-knicks', 'OKC': 'oklahoma-city-thunder',
            'ORL': 'orlando-magic', 'PHI': 'philadelphia-76ers', 'PHX': 'phoenix-suns',
            'POR': 'portland-trail-blazers', 'SAC': 'sacramento-kings', 'SAS': 'san-antonio-spurs',
            'TOR': 'toronto-raptors', 'UTA': 'utah-jazz', 'WAS': 'washington-wizards'
        }
        
        slug = slug_map.get(team_abbr.upper())
        if not slug:
            return None
            
        url = f"https://www.rotowire.com/basketball/nba-lineups.php"
        
        try:
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            lineup_divs = soup.find_all('div', class_='lineup')
            
            for lineup_div in lineup_divs:
                team_link = lineup_div.find('a', href=re.compile(team_abbr.lower()))
                if not team_link:
                    continue
                    
                starters_names = []
                starters_pdiv = lineup_div.find('div', class_='lineup__players')
                if starters_pdiv:
                    player_divs = starters_pdiv.find_all('div', class_='lineup__player')
                    for pd in player_divs[:5]:
                        name_div = pd.find('div', class_='lineup__player__name')
                        if name_div:
                            name = name_div.get_text(strip=True)
                            starters_names.append(name)
                
                if len(starters_names) >= 5:
                    return {
                        'team': team_abbr,
                        'game_date': game_date,
                        'starters': starters_names[:5],
                        'starter_ids': [],
                        'source': 'rotowire_projected'
                    }
                    
        except Exception as e:
            logger.debug(f"Rotowire scrape failed: {e}")
        
        return None
    
    def _fetch_from_espn(self, team_abbr: str, game_date: str) -> Optional[dict]:
        """Fetch projected lineups from ESPN API."""
        slug_map = {
            'ATL': 'atl', 'BOS': 'bos', 'BKN': 'bkn', 'CHA': 'cha', 'CHI': 'chi',
            'CLE': 'cle', 'DAL': 'dal', 'DEN': 'den', 'DET': 'det', 'GSW': 'gsw',
            'HOU': 'hou', 'IND': 'ind', 'LAC': 'lac', 'LAL': 'lal', 'MEM': 'mem',
            'MIA': 'mia', 'MIL': 'mil', 'MIN': 'min', 'NOP': 'no', 'NYK': 'nyk',
            'OKC': 'okc', 'ORL': 'orl', 'PHI': 'phi', 'PHX': 'phx', 'POR': 'por',
            'SAC': 'sac', 'SAS': 'sa', 'TOR': 'tor', 'UTA': 'utah', 'WAS': 'wsh'
        }
        
        espn_abbr = slug_map.get(team_abbr.upper())
        if not espn_abbr:
            return None
        
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{espn_abbr}"
        
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            athletes = data.get('team', {}).get('athletes', [])
            starters = []
            
            position_order = ['PG', 'SG', 'SF', 'PF', 'C']
            
            for pos in position_order:
                for athlete in athletes:
                    pos_name = athlete.get('position', {}).get('abbreviation', '')
                    if pos_name == pos and athlete.get('starter', False):
                        starters.append({
                            'name': athlete.get('displayName', ''),
                            'id': athlete.get('id'),
                            'position': pos_name
                        })
                        break
            
            if len(starters) >= 5:
                return {
                    'team': team_abbr,
                    'game_date': game_date,
                    'starters': [s['name'] for s in starters],
                    'starter_ids': [s['id'] for s in starters],
                    'positions': [s['position'] for s in starters],
                    'source': 'espn_projected'
                }
                
        except Exception as e:
            logger.debug(f"ESPN API failed: {e}")
        
        return None

    def _infer_lineup_from_history(self, team_abbr: str, game_date: str) -> Optional[dict]:
        """
        Infer expected lineup from recent game history.
        """
        tendency = self._coach_tendencies.get(team_abbr.upper(), {})
        
        return {
            'team': team_abbr,
            'game_date': game_date,
            'starters': tendency.get('recent_starters', []),
            'starter_ids': tendency.get('recent_starter_ids', []),
            'source': 'historical_inference',
            'coach_tendency_score': tendency.get('rotation_stability', 0.5)
        }
    
    def _get_season_for_date(self, date_str: str) -> str:
        """Get NBA season string for a given date."""
        date = datetime.strptime(date_str, '%Y-%m-%d')
        if date.month >= 10:
            return f"{date.year}-{str(date.year + 1)[2:]}"
        else:
            return f"{date.year - 1}-{str(date.year)[2:]}"

    def get_full_rotation(
        self, 
        team_abbr: str, 
        game_date: str = None
    ) -> dict:
        """
        Get expected full rotation (starters + bench) for a team.
        
        Returns:
            Dictionary with:
                - starters: List of starter dicts (name, id, position, expected_mins)
                - bench: List of bench players
                - inactive: List of injured/unavailable players
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
            
        team_abbr = team_abbr.upper()
        
        lineup = self.get_starting_lineup(team_abbr, game_date)
        
        rotation = self._fetch_rotation_from_history(team_abbr)
        
        inactive = self._get_inactive_players(team_abbr, game_date)
        
        return {
            'team': team_abbr,
            'game_date': game_date,
            'starters': lineup.get('starters', []),
            'starter_ids': lineup.get('starter_ids', []),
            'lineup_source': lineup.get('source'),
            'lineup_status': lineup.get('status'),
            'lineup_confidence': lineup.get('confidence', 0),
            'bench': rotation.get('bench', []),
            'rotation': rotation.get('full_rotation', []),
            'inactive': inactive,
            'expected_minutes': rotation.get('expected_minutes', {})
        }
    
    def _fetch_rotation_from_history(self, team_abbr: str) -> dict:
        """Get rotation based on recent minutes distribution."""
        cache_file = os.path.join(self.cache_dir, f"rotation_{team_abbr}.json")
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=24):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        
        return {
            'full_rotation': [],
            'bench': [],
            'expected_minutes': {}
        }
    
    def _get_inactive_players(self, team_abbr: str, game_date: str) -> List[dict]:
        """Get list of inactive/injured players for a game."""
        try:
            from src.data.injury_scraper import InjuryScraper
            scraper = InjuryScraper()
            
            injuries_df = scraper.fetch_injuries()
            if injuries_df.empty:
                return []
            if 'TEAM_ABBR' not in injuries_df.columns:
                if 'TEAM' in injuries_df.columns:
                    injuries_df['TEAM_ABBR'] = injuries_df['TEAM'].apply(normalize_team)
                else:
                    return []
            injuries_df = injuries_df[injuries_df['TEAM_ABBR'] == normalize_team(team_abbr)]
            
            inactive = []
            for _, injury in injuries_df.iterrows():
                status = str(injury.get('STATUS', '')).lower()
                if 'out' in status or 'injured' in status:
                    inactive.append({
                        'name': injury.get('PLAYER', ''),
                        'reason': injury.get('COMMENT', ''),
                        'status': injury.get('STATUS', 'OUT')
                    })
                    
            return inactive
            
        except Exception as e:
            logger.debug(f"Failed to get inactive players: {e}")
            return []

    def get_matchup_lineups(
        self, 
        home_team: str, 
        away_team: str,
        game_date: str = None
    ) -> dict:
        """
        Get lineups for both teams in a matchup.
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
            
        return {
            'game_date': game_date,
            'home_team': home_team.upper(),
            'away_team': away_team.upper(),
            'home_lineup': self.get_starting_lineup(home_team, game_date),
            'away_lineup': self.get_starting_lineup(away_team, game_date),
            'home_rotation': self.get_full_rotation(home_team, game_date),
            'away_rotation': self.get_full_rotation(away_team, game_date)
        }

    def update_coach_tendencies(self, team_abbr: str, starters: List[str], game_date: str):
        """
        Update coach tendency tracking after lineup is confirmed.
        Call this after games to learn coach patterns.
        """
        team_abbr = team_abbr.upper()
        
        if team_abbr not in self._coach_tendencies:
            self._coach_tendencies[team_abbr] = {
                'recent_starters': [],
                'recent_starter_ids': [],
                'rotation_stability': 0.5,
                'last_10_starters': [],
                'rest_day_changes': 0
            }
        
        tendency = self._coach_tendencies[team_abbr]
        
        if tendency.get('recent_starters'):
            matches = sum(1 for a, b in zip(tendency['recent_starters'], starters) if a == b)
            stability = matches / 5.0
            
            old_stability = tendency.get('rotation_stability', 0.5)
            tendency['rotation_stability'] = old_stability * 0.7 + stability * 0.3
        
        tendency['recent_starters'] = starters[:5]
        tendency['last_updated'] = game_date
        
        self._save_coach_tendencies()

    def get_all_games_lineups(self, game_date: str = None) -> List[dict]:
        """
        Get lineups for all games on a specific date.
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
            
        games = self._get_daily_schedule(game_date)
        
        results = []
        for game in games:
            home = game.get('home_team')
            away = game.get('away_team')
            
            if home and away:
                lineups = self.get_matchup_lineups(home, away, game_date)
                lineups['game_id'] = game.get('game_id')
                results.append(lineups)
                
        return results
    
    def _get_daily_schedule(self, game_date: str) -> List[dict]:
        """Get game schedule for a date."""
        url = "https://stats.nba.com/stats/scoreboardV3"
        params = {
            'LeagueID': '00',
            'OriginalDate': game_date
        }
        
        try:
            response = self._session.get(
                url, 
                params=params,
                headers=self.nba_headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            games = []
            for game in data.get('scoreboard', {}).get('games', []):
                games.append({
                    'game_id': game.get('gameId'),
                    'home_team': ID_TO_ABBR.get(game.get('homeTeam', {}).get('teamId'), 'UNK'),
                    'away_team': ID_TO_ABBR.get(game.get('awayTeam', {}).get('teamId'), 'UNK'),
                    'game_time': game.get('gameTimeUTC')
                })
            
            return games
            
        except Exception as e:
            logger.error(f"Failed to get schedule for {game_date}: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = LineupScraper()
    
    print("Testing Lineup Scraper...")
    
    lineup = scraper.get_starting_lineup('BOS')
    print(f"\nBoston Celtics Lineup ({lineup.get('status', 'unknown')}):")
    print(f"  Source: {lineup.get('source')}")
    print(f"  Confidence: {lineup.get('confidence', 0):.0%}")
    print(f"  Starters: {lineup.get('starters', [])}")
    
    rotation = scraper.get_full_rotation('LAL')
    print(f"\nLakers Full Rotation:")
    print(f"  Starters: {rotation.get('starters', [])}")
    print(f"  Inactive: {[p['name'] for p in rotation.get('inactive', [])]}")
