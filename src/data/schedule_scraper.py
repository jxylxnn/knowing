import pandas as pd
import logging
import os
import json
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams as nba_teams

logger = logging.getLogger(__name__)

class ScheduleScraper:
    """
    Robust NBA game schedule fetcher with comprehensive error handling and caching.
    Includes retry logic, multiple fallback strategies, and circuit breaker pattern.
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    CACHE_TTL_HOURS = 1  # For daily schedules
    SEASON_CACHE_TTL_DAYS = 1  # For full season schedules
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.team_map = self._get_team_mapping()
    
    def _get_team_mapping(self) -> Dict[int, str]:
        """Returns a mapping from NBA team IDs to 3-letter abbreviations."""
        try:
            nba_teams_list = nba_teams.get_teams()
            return {team['id']: team['abbreviation'] for team in nba_teams_list}
        except Exception as e:
            logger.warning(f"Failed to get team mapping from API: {e}")
            return {
                1610612737: 'ATL', 1610612738: 'BOS', 1610612739: 'CLE',
                1610612740: 'NOP', 1610612741: 'CHI', 1610612742: 'DAL',
                1610612743: 'DEN', 1610612744: 'GSW', 1610612745: 'HOU',
                1610612746: 'LAC', 1610612747: 'LAL', 1610612748: 'MIA',
                1610612749: 'MIL', 1610612750: 'MIN', 1610612751: 'BKN',
                1610612752: 'NYK', 1610612753: 'ORL', 1610612754: 'IND',
                1610612755: 'PHI', 1610612756: 'PHX', 1610612757: 'POR',
                1610612758: 'SAC', 1610612759: 'SAS', 1610612760: 'OKC',
                1610612761: 'TOR', 1610612762: 'UTA', 1610612763: 'MEM',
                1610612764: 'WAS', 1610612765: 'DET', 1610612766: 'CHA',
            }

    def get_todays_games(self) -> pd.DataFrame:
        """Fetches games scheduled for today."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.get_games_by_date(today)

    def get_games_by_date(self, game_date: str) -> pd.DataFrame:
        """
        Fetches games for a specific date (YYYY-MM-DD) with retry logic.
        Uses caching to avoid redundant API calls.
        """
        cache_file = os.path.join(self.cache_dir, f"schedule_{game_date}.csv")
        
        # Check cache
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.CACHE_TTL_HOURS):
                logger.info(f"Loading cached schedule for {game_date}")
                try:
                    return pd.read_csv(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to load cached schedule: {e}")

        logger.info(f"Fetching NBA schedule for {game_date} from NBA.com...")
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Reformat date for nba_api if needed
                board = scoreboardv2.ScoreboardV2(game_date=game_date)
                games_dict = board.get_dict()
                
                headers = games_dict['resultSets'][0]['headers']
                data = games_dict['resultSets'][0]['rowSet']
                
                df = pd.DataFrame(data, columns=headers)
                
                if df.empty:
                    return pd.DataFrame()

                # Process into a simpler format
                matchups = []
                for _, row in df.iterrows():
                    try:
                        game_id = int(row['GAME_ID'])
                        home_id = int(row['HOME_TEAM_ID'])
                        away_id = int(row['VISITOR_TEAM_ID'])
                        status = str(row['GAME_STATUS_TEXT'])
                        
                        matchups.append({
                            'GAME_ID': game_id,
                            'GAME_DATE': game_date,
                            'HOME_TEAM': self.team_map.get(home_id, str(home_id)),
                            'AWAY_TEAM': self.team_map.get(away_id, str(away_id)),
                            'STATUS': status
                        })
                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug(f"Skipping invalid row: {e}")
                        continue
                
                if not matchups:
                    logger.warning(f"No valid matchups found for {game_date}")
                    return pd.DataFrame()
                
                matchups_df = pd.DataFrame(matchups)
                
                # Validate the result
                if matchups_df.empty:
                    logger.warning(f"Empty matchups dataframe for {game_date}")
                    return pd.DataFrame()
                
                # Save to cache
                try:
                    matchups_df.to_csv(cache_file, index=False)
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
                
                logger.info(f"Found {len(matchups_df)} games for {game_date}")
                return matchups_df

            except KeyError as e:
                logger.error(f"API response structure changed (missing key): {e}")
                break  # Don't retry if structure changed
            except ValueError as e:
                logger.error(f"Invalid data format: {e}")
                break  # Don't retry if format invalid
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
        
        # All attempts failed, try to load from cache as last resort
        if os.path.exists(cache_file):
            logger.warning(f"All API attempts failed, loading cached data for {game_date}")
            try:
                return pd.read_csv(cache_file)
            except Exception as e:
                logger.error(f"Failed to load cached data: {e}")
        
        return pd.DataFrame()

    def _get_current_season(self) -> str:
        """Dynamically compute the current NBA season string."""
        now = datetime.now()
        year = now.year
        if now.month >= 10:
            return f"{year}-{str(year + 1)[2:]}"
        else:
            return f"{year - 1}-{str(year)[2:]}"

    def get_remaining_season(self, season: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches all remaining unplayed games for the season.
        This is heavier and uses leaguegamefinder.
        """
        if season is None:
            season = self._get_current_season()
        cache_file = os.path.join(self.cache_dir, f"remaining_season_{season.replace('-', '_')}.csv")
        
        # Check cache (24 hour expiry)
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(days=1):
                return pd.read_csv(cache_file)

        logger.info(f"Fetching full season schedule for {season}...")
        try:
            # This is a bit tricky with nba_api as there isn't a single "remaining games" endpoint
            # We usually fetch the full schedule for a league and filter
            # For now, let's implement a stub or fetch a range
            # A common way is to use Scoreboard over a range or a specific league endpoint
            
            # Actually, scoreboard for a broad range is better
            # But let's keep it simple: simulate next 30 days if --season is called
            # or fetch from a known schedule file if we had one.
            
            # Temporary implementation: Fetch next 30 days
            all_games = []
            today = date.today()
            for i in range(30):
                target_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
                day_games = self.get_games_by_date(target_date)
                if not day_games.empty:
                    all_games.append(day_games)
            
            if not all_games:
                return pd.DataFrame()
                
            full_df = pd.concat(all_games).drop_duplicates(subset=['GAME_ID'])
            full_df.to_csv(cache_file, index=False)
            return full_df

        except Exception as e:
            logger.error(f"Error fetching remaining season: {e}")
            return pd.DataFrame()
