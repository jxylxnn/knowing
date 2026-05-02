"""
RotoWire Lineup Scraper for NBA Starting Lineups and Projected Minutes.
Scrapes projected starting lineups and player minutes from RotoWire.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from src.utils.team_mappings import normalize_team, get_all_abbrs, TEAMS


class RotoWireLineupScraper:
    """
    Scrapes RotoWire for NBA starting lineups and projected minutes.
    Provides real-time lineup information including injury status.
    """
    
    def __init__(self, cache_dir: str = 'data/cache', config: Optional[Any] = None):
        self._config = config
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._session = requests.Session()
        self._session.headers.update(self._get_headers())
        self._lineups_cache: Dict[str, dict] = {}
        self._cache_timestamp: Optional[datetime] = None
        
        self.max_retries = self._get_config_value('http.max_retries', 3)
        self.retry_delay = self._get_config_value('http.retry_delay', 2.0)
        self.cache_ttl_minutes = self._get_config_value('cache.rotowire_ttl_minutes', 30.0)
    
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
    def LINEUPS_URL(self) -> str:
        return self._get_config_value('api.rotowire_lineups_url', 'https://www.rotowire.com/basketball/nba-lineups.php')
    
    @property
    def DAILY_URL(self) -> str:
        return self._get_config_value('api.rotowire_lineups_daily_url', 'https://www.rotowire.com/basketball/nba-lineups-daily.php')
    
    @property
    def PROJECTED_MINUTES_URL(self) -> str:
        return self._get_config_value('api.rotowire_projections_url', 'https://www.rotowire.com/basketball/projections-daily.php')
    
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
        
    def get_todays_lineups(self) -> Dict[str, dict]:
        """
        Get today's projected starting lineups for all games.
        
        Returns:
            Dictionary mapping game_id -> {home_team, away_team, home_lineup, away_lineup}
        """
        return self.get_lineups_by_date(datetime.now().strftime('%Y-%m-%d'))
    
    def get_lineups_by_date(self, game_date: str) -> Dict[str, dict]:
        """
        Get projected starting lineups for a specific date.
        
        Args:
            game_date: Date string in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with lineup information
        """
        cache_key = f"lineups_{game_date}"
        
        if cache_key in self._lineups_cache:
            cached = self._lineups_cache[cache_key]
            if cached.get('timestamp'):
                cache_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cache_time < timedelta(minutes=self.cache_ttl_minutes):
                    return cached
        
        cache_file = os.path.join(self.cache_dir, f"lineups_{game_date}.json")
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(minutes=self.cache_ttl_minutes):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                        self._lineups_cache[cache_key] = cached
                        return cached
                except Exception as e:
                    logger.debug(f"Failed to load lineup cache: {e}")
        
        result = self._fetch_lineups(game_date)
        
        if result:
            result['timestamp'] = datetime.now().isoformat()
            self._lineups_cache[cache_key] = result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as e:
                logger.debug(f"Failed to save lineup cache: {e}")
        
        return result
    
    def _fetch_lineups(self, game_date: str) -> dict:
        """Fetch lineups from RotoWire."""
        result = {
            'date': game_date,
            'games': [],
            'fetched_at': datetime.now().isoformat()
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.get(self.LINEUPS_URL, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                games = self._parse_lineups_page(soup)
                if games:
                    result['games'] = games
                    result['source'] = 'rotowire'
                    logger.info(f"Fetched {len(games)} lineups for {game_date}")
                    return result
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Error parsing lineups: {e}")
                break
        
        result['error'] = 'Failed to fetch lineups'
        result['source'] = 'failed'
        return result
    
    def _parse_lineups_page(self, soup) -> List[dict]:
        """Parse the RotoWire lineups page."""
        games = []
        
        lineup_containers = soup.find_all('div', class_='lineup')
        
        if not lineup_containers:
            lineup_containers = soup.find_all('div', class_='game')
        
        if not lineup_containers:
            lineup_containers = soup.find_all('div', {'data-game': True})
        
        for container in lineup_containers:
            try:
                game = self._parse_lineup_container(container)
                if game:
                    games.append(game)
            except Exception as e:
                logger.debug(f"Error parsing lineup container: {e}")
                continue
        
        if not games:
            games = self._parse_lineups_fallback(soup)
        
        return games
    
    def _parse_lineup_container(self, container) -> Optional[dict]:
        """Parse a single lineup container."""
        game = {}
        
        team_divs = container.find_all('div', class_='lineup-team')
        if not team_divs:
            team_divs = container.find_all('div', class_='team')
        
        if len(team_divs) >= 2:
            away_team = self._extract_team_info(team_divs[0])
            home_team = self._extract_team_info(team_divs[1])
            
            if away_team and home_team:
                game = {
                    'away_team': away_team['abbr'],
                    'away_team_name': away_team['name'],
                    'away_lineup': away_team['lineup'],
                    'away_bench': away_team.get('bench', []),
                    'home_team': home_team['abbr'],
                    'home_team_name': home_team['name'],
                    'home_lineup': home_team['lineup'],
                    'home_bench': home_team.get('bench', []),
                    'game_time': self._extract_game_time(container)
                }
        
        return game if game else None
    
    def _extract_team_info(self, team_div) -> dict:
        """Extract team info from a team div."""
        result = {
            'abbr': None,
            'name': None,
            'lineup': [],
            'bench': []
        }
        
        name_elem = team_div.find(['h3', 'h4', 'span', 'a'], class_=lambda x: x and ('team' in x.lower() or 'name' in x.lower()))
        if not name_elem:
            name_elem = team_div.find(['h3', 'h4', 'span', 'a'])
        
        if name_elem:
            team_name = name_elem.get_text(strip=True)
            result['name'] = team_name
            result['abbr'] = self._get_team_abbr(team_name)
        
        players = team_div.find_all(['li', 'div', 'span'], class_=lambda x: x and ('player' in x.lower() or 'name' in x.lower()))
        if not players:
            players = team_div.find_all('li')
        
        for i, player_elem in enumerate(players[:10]):
            player_info = self._extract_player_info(player_elem)
            if player_info:
                if i < 5 and player_info.get('status') != 'OUT':
                    result['lineup'].append(player_info)
                else:
                    result['bench'].append(player_info)
        
        return result
    
    def _extract_player_info(self, player_elem) -> Optional[dict]:
        """Extract player info from a player element."""
        name = player_elem.get_text(strip=True)
        
        if not name or len(name) < 3:
            return None
        
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s*\d+\s*', ' ', name).strip()
        
        status = 'ACTIVE'
        status_match = re.search(r'(OUT|GTD|QUESTIONABLE|DOUBTFUL|PROBABLE)', name, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).upper()
            name = name.replace(status_match.group(0), '').strip()
        
        projected_minutes = None
        min_match = re.search(r'(\d+)\s*min', name, re.IGNORECASE)
        if min_match:
            projected_minutes = int(min_match.group(1))
            name = name.replace(min_match.group(0), '').strip()
        
        position = None
        pos_match = re.search(r'(PG|SG|SF|PF|C|G|F)\b', name, re.IGNORECASE)
        if pos_match:
            position = pos_match.group(1).upper()
        
        return {
            'name': name,
            'position': position,
            'status': status,
            'projected_minutes': projected_minutes
        }
    
    def _extract_game_time(self, container) -> Optional[str]:
        """Extract game time from container."""
        time_elem = container.find(['span', 'div', 'time'], class_=lambda x: x and ('time' in x.lower() or 'date' in x.lower()))
        if time_elem:
            return time_elem.get_text(strip=True)
        return None
    
    def _parse_lineups_fallback(self, soup) -> List[dict]:
        """Fallback parser for different page layouts."""
        games = []
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            header = rows[0]
            headers = [th.get_text(strip=True).lower() for th in header.find_all(['th', 'td'])]
            
            if 'team' in str(headers) or 'player' in str(headers):
                game = self._parse_table_to_game(rows[1:])
                if game:
                    games.append(game)
        
        return games
    
    def _parse_table_to_game(self, rows) -> Optional[dict]:
        """Parse table rows into a game."""
        game = {'away_lineup': [], 'home_lineup': []}
        current_team = None
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            for cell in cells:
                text = cell.get_text(strip=True)
                
                team_abbr = self._get_team_abbr(text)
                if team_abbr:
                    if current_team is None:
                        game['away_team'] = team_abbr
                        current_team = 'away'
                    else:
                        game['home_team'] = team_abbr
                        current_team = 'home'
                    continue
                
                player_info = self._extract_player_info(cell)
                if player_info and player_info.get('name'):
                    if current_team == 'away':
                        if len(game['away_lineup']) < 5:
                            game['away_lineup'].append(player_info)
                    elif current_team == 'home':
                        if len(game['home_lineup']) < 5:
                            game['home_lineup'].append(player_info)
        
        return game if game.get('away_team') and game.get('home_team') else None
    
    def _get_team_abbr(self, team_name: str) -> Optional[str]:
        """Convert team name to abbreviation."""
        team_lower = team_name.lower().strip()
        
        return normalize_team(team_name)
    
    def get_projected_minutes(self, team_abbr: str = None) -> Dict[str, float]:
        """
        Get projected minutes for players.
        
        Args:
            team_abbr: Optional team filter
            
        Returns:
            Dictionary mapping player_name -> projected_minutes
        """
        cache_key = f"minutes_{datetime.now().strftime('%Y-%m-%d')}"
        cache_file = os.path.join(self.cache_dir, f"minutes_{datetime.now().strftime('%Y%m%d')}.json")
        
        if cache_key in self._lineups_cache:
            return self._lineups_cache[cache_key]
        
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(minutes=self.cache_ttl_minutes):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        
        result = self._fetch_projected_minutes(team_abbr)
        
        if result:
            self._lineups_cache[cache_key] = result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
            except Exception:
                pass
        
        return result
    
    def _fetch_projected_minutes(self, team_abbr: str = None) -> Dict[str, float]:
        """Fetch projected minutes from RotoWire."""
        minutes = {}
        
        try:
            response = self._session.get(self.PROJECTED_MINUTES_URL, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        player_name = cells[0].get_text(strip=True)
                        try:
                            proj_min = float(cells[-1].get_text(strip=True))
                            if proj_min > 0:
                                minutes[player_name] = proj_min
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug(f"Error fetching projected minutes: {e}")
        
        return minutes
    
    def get_team_lineup(self, team_abbr: str) -> Optional[dict]:
        """
        Get lineup for a specific team.
        
        Returns:
            Dictionary with starters, bench, and injury status
        """
        lineups = self.get_todays_lineups()
        
        team_abbr = normalize_team(team_abbr)
        
        for game in lineups.get('games', []):
            if game.get('home_team') == team_abbr:
                return {
                    'team': team_abbr,
                    'starters': game.get('home_lineup', []),
                    'bench': game.get('home_bench', []),
                    'is_home': True
                }
            elif game.get('away_team') == team_abbr:
                return {
                    'team': team_abbr,
                    'starters': game.get('away_lineup', []),
                    'bench': game.get('away_bench', []),
                    'is_home': False
                }
        
        return None
    
    def get_starting_lineup(self, team_abbr: str) -> List[str]:
        """Get list of starter names for a team."""
        lineup = self.get_team_lineup(team_abbr)
        if lineup:
            return [p['name'] for p in lineup.get('starters', []) if p.get('status') != 'OUT']
        return []


class LineupManager:
    """
    Manages lineup data and provides enhanced lineup intelligence.
    Combines RotoWire data with injury status for accurate projections.
    """
    
    def __init__(self, lineup_scraper: RotoWireLineupScraper = None):
        from src.data.injury_scraper import InjuryScraper
        self.lineup_scraper = lineup_scraper or RotoWireLineupScraper()
        self.injury_scraper = InjuryScraper()
        self._lineup_cache: Dict[str, dict] = {}
    
    def get_enhanced_lineup(self, team_abbr: str, opponent_abbr: str = None) -> dict:
        """
        Get enhanced lineup data with injury and minutes projection.
        
        Returns:
            Dictionary with starters, their projected minutes, and confidence
        """
        team_abbr = normalize_team(team_abbr)
        
        cache_key = f"{team_abbr}_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self._lineup_cache:
            return self._lineup_cache[cache_key]
        
        lineup_data = self.lineup_scraper.get_team_lineup(team_abbr)
        injury_probs = self.injury_scraper.get_player_availability(team_abbr)
        projected_minutes = self.lineup_scraper.get_projected_minutes(team_abbr)
        
        result = {
            'team': team_abbr,
            'starters': [],
            'bench': [],
            'confirmed_lineup': True,
            'total_minutes_known': 0,
            'missing_players': []
        }
        
        if lineup_data:
            for player in lineup_data.get('starters', []):
                name = player.get('name', '')
                inj_prob = injury_probs.get(name, 1.0)
                proj_min = projected_minutes.get(name, player.get('projected_minutes', 28))
                
                enhanced_player = {
                    'name': name,
                    'position': player.get('position'),
                    'status': player.get('status', 'ACTIVE'),
                    'play_probability': inj_prob,
                    'projected_minutes': proj_min if inj_prob > 0.5 else proj_min * inj_prob,
                    'is_starter': True,
                    'confidence': 'confirmed' if player.get('status') == 'ACTIVE' else 'uncertain'
                }
                
                result['starters'].append(enhanced_player)
                result['total_minutes_known'] += enhanced_player['projected_minutes']
                
                if inj_prob < 0.5:
                    result['missing_players'].append(name)
            
            for player in lineup_data.get('bench', []):
                name = player.get('name', '')
                inj_prob = injury_probs.get(name, 1.0)
                proj_min = projected_minutes.get(name, player.get('projected_minutes', 15))
                
                enhanced_player = {
                    'name': name,
                    'position': player.get('position'),
                    'status': player.get('status', 'ACTIVE'),
                    'play_probability': inj_prob,
                    'projected_minutes': proj_min if inj_prob > 0.5 else proj_min * inj_prob,
                    'is_starter': False,
                    'confidence': 'bench'
                }
                
                result['bench'].append(enhanced_player)
        else:
            result['confirmed_lineup'] = False
        
        self._lineup_cache[cache_key] = result
        return result
    
    def estimate_minutes_distribution(self, team_abbr: str) -> Dict[str, float]:
        """
        Estimate minutes distribution for entire team.
        Uses lineup data + injury status to project realistic minutes.
        """
        enhanced = self.get_enhanced_lineup(team_abbr)
        
        minutes_dist = {}
        total_allocated = 0
        
        for player in enhanced.get('starters', []):
            name = player['name']
            proj_min = player['projected_minutes'] * player['play_probability']
            minutes_dist[name] = proj_min
            total_allocated += proj_min
        
        for player in enhanced.get('bench', []):
            name = player['name']
            proj_min = player['projected_minutes'] * player['play_probability']
            minutes_dist[name] = proj_min
            total_allocated += proj_min
        
        if total_allocated < 240:
            deficit = 240 - total_allocated
            num_players = len(minutes_dist)
            if num_players > 0:
                per_player_bonus = deficit / num_players
                for name in minutes_dist:
                    minutes_dist[name] = min(minutes_dist[name] + per_player_bonus, 42)
        elif total_allocated > 240:
            surplus = total_allocated - 240
            ratio = 240 / total_allocated
            for name in minutes_dist:
                minutes_dist[name] = minutes_dist[name] * ratio
        
        return minutes_dist
    
    def get_usage_adjustment(self, team_abbr: str, missing_players: List[str] = None) -> float:
        """
        Calculate usage rate adjustment when players are missing.
        
        Returns:
            Multiplier for remaining players' usage (1.0 = no adjustment)
        """
        if not missing_players:
            return 1.0
        
        enhanced = self.get_enhanced_lineup(team_abbr)
        
        total_missing_usage = 0.15
        
        num_remaining = len([p for p in enhanced.get('starters', []) if p['play_probability'] > 0.5])
        num_remaining = max(num_remaining, 5)
        
        usage_boost = 1.0 + (total_missing_usage / num_remaining)
        
        return min(usage_boost, 1.35)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = RotoWireLineupScraper()
    manager = LineupManager(scraper)
    
    print("Testing RotoWire Lineup Scraper...")
    
    lineups = scraper.get_todays_lineups()
    print(f"\nFound {len(lineups.get('games', []))} games today")
    
    for game in lineups.get('games', [])[:2]:
        print(f"\n{game.get('away_team', 'TBD')} @ {game.get('home_team', 'TBD')}")
        print(f"  Away starters: {[p['name'] for p in game.get('away_lineup', [])]}")
        print(f"  Home starters: {[p['name'] for p in game.get('home_lineup', [])]}")
    
    boston_lineup = manager.get_enhanced_lineup('BOS')
    print(f"\nBoston Enhanced Lineup:")
    for player in boston_lineup.get('starters', [])[:5]:
        print(f"  {player['name']}: {player['projected_minutes']:.0f} min ({player['status']})")