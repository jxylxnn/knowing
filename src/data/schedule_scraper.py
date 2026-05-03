import pandas as pd
import logging
import os
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
    
    def __init__(self, cache_dir: str = 'data/cache', config: Optional[Any] = None):
        self._config = config
        self.cache_dir = cache_dir
        self.last_fetch_status: Dict[str, Any] = {}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.team_map = self._get_team_mapping()
        
        self.max_retries = self._get_config_value('http.max_retries', 3)
        self.retry_delay = self._get_config_value('http.retry_delay', 2.0)
        self.cache_ttl_hours = self._get_config_value('cache.schedule_ttl_hours', 1.0)
        self.season_cache_ttl_days = self._get_config_value('cache.season_schedule_ttl_days', 1.0)

    def _set_last_fetch_status(
        self,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_fetch_status = {
            'source_key': 'schedule',
            'status': status,
            'required': True,
            'message': message,
            'details': details or {},
        }

    def get_last_fetch_status(self) -> Dict[str, Any]:
        return dict(self.last_fetch_status)
    
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
            if datetime.now() - file_time < timedelta(hours=self.cache_ttl_hours):
                logger.info(f"Loading cached schedule for {game_date}")
                try:
                    cached_df = pd.read_csv(cache_file)
                    self._set_last_fetch_status(
                        'success',
                        f"Loaded cached schedule for {game_date}",
                        {
                            'game_date': game_date,
                            'source': 'cache',
                            'rows': int(len(cached_df)),
                        },
                    )
                    return cached_df
                except Exception as e:
                    logger.warning(f"Failed to load cached schedule: {e}")

        logger.info(f"Fetching NBA schedule for {game_date} from NBA.com...")
        
        for attempt in range(self.max_retries):
            try:
                # Reformat date for nba_api if needed
                board = scoreboardv2.ScoreboardV2(game_date=game_date)
                games_dict = board.get_dict()
                
                headers = games_dict['resultSets'][0]['headers']
                data = games_dict['resultSets'][0]['rowSet']
                
                df = pd.DataFrame(data, columns=headers)
                
                if df.empty:
                    self._set_last_fetch_status(
                        'success',
                        f"No games scheduled for {game_date}",
                        {
                            'game_date': game_date,
                            'source': 'nba_api',
                            'rows': 0,
                        },
                    )
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
                    self._set_last_fetch_status(
                        'failed',
                        f"Schedule response for {game_date} did not contain valid matchups",
                        {
                            'game_date': game_date,
                            'source': 'nba_api',
                        },
                    )
                    return pd.DataFrame()
                
                matchups_df = pd.DataFrame(matchups)
                
                # Validate the result
                if matchups_df.empty:
                    logger.warning(f"Empty matchups dataframe for {game_date}")
                    self._set_last_fetch_status(
                        'failed',
                        f"Schedule fetch for {game_date} returned an empty matchup frame",
                        {
                            'game_date': game_date,
                            'source': 'nba_api',
                        },
                    )
                    return pd.DataFrame()
                
                # Save to cache
                try:
                    matchups_df.to_csv(cache_file, index=False)
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
                
                logger.info(f"Found {len(matchups_df)} games for {game_date}")
                self._set_last_fetch_status(
                    'success',
                    f"Fetched {len(matchups_df)} scheduled games for {game_date}",
                    {
                        'game_date': game_date,
                        'source': 'nba_api',
                        'rows': int(len(matchups_df)),
                    },
                )
                return matchups_df

            except KeyError as e:
                logger.error(f"API response structure changed (missing key): {e}")
                self._set_last_fetch_status(
                    'failed',
                    f"Schedule API response changed for {game_date}",
                    {
                        'game_date': game_date,
                        'source': 'nba_api',
                        'error': str(e),
                    },
                )
                break  # Don't retry if structure changed
            except ValueError as e:
                logger.error(f"Invalid data format: {e}")
                self._set_last_fetch_status(
                    'failed',
                    f"Schedule API returned invalid data for {game_date}",
                    {
                        'game_date': game_date,
                        'source': 'nba_api',
                        'error': str(e),
                    },
                )
                break  # Don't retry if format invalid
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # All attempts failed, try to load from cache as last resort
        if os.path.exists(cache_file):
            logger.warning(f"All API attempts failed, loading cached data for {game_date}")
            try:
                cached_df = pd.read_csv(cache_file)
                self._set_last_fetch_status(
                    'fallback',
                    f"Schedule API failed for {game_date}; using cached schedule",
                    {
                        'game_date': game_date,
                        'source': 'cache',
                        'rows': int(len(cached_df)),
                    },
                )
                return cached_df
            except Exception as e:
                logger.error(f"Failed to load cached data: {e}")
                self._set_last_fetch_status(
                    'failed',
                    f"Schedule API failed and cache load also failed for {game_date}",
                    {
                        'game_date': game_date,
                        'source': 'cache',
                        'error': str(e),
                    },
                )
        
        self._set_last_fetch_status(
            'failed',
            f"Unable to load schedule for {game_date}",
            {
                'game_date': game_date,
                'source': 'nba_api',
            },
        )
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

        NOTE: nba_api does not expose a single "remaining games" endpoint, so this
        composes daily scoreboard calls from today through the expected season end.
        The iteration window is capped to avoid excessive API calls.
        """
        if season is None:
            season = self._get_current_season()
        cache_file = os.path.join(self.cache_dir, f"remaining_season_{season.replace('-', '_')}.csv")

        # Check cache (24 hour expiry)
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(days=self.season_cache_ttl_days):
                cached_df = pd.read_csv(cache_file)
                self._set_last_fetch_status(
                    'success',
                    f"Loaded cached remaining-season schedule for {season}",
                    {
                        'season': season,
                        'source': 'cache',
                        'rows': int(len(cached_df)),
                    },
                )
                return cached_df

        logger.info(f"Fetching remaining season schedule for {season}...")
        try:
            all_games = []
            today = date.today()

            # Compute a reasonable season-end horizon (June 30 of the season's second year).
            # The NBA regular season ends in April and playoffs run into June.
            season_end_year = int(season.split('-')[0]) + 1
            season_end = date(season_end_year, 6, 30)

            # Cap the forward window to avoid excessive API calls during the offseason
            # or if the season string is malformed.
            max_days = (season_end - today).days
            if max_days < 0:
                max_days = 0
            if max_days > 180:
                max_days = 180
                logger.warning(
                    f"Remaining-season window capped at {max_days} days "
                    f"(original horizon {(season_end - today).days} days). "
                    "This is a safeguard against excessive API calls."
                )

            for i in range(max_days + 1):
                target_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
                day_games = self.get_games_by_date(target_date)
                if not day_games.empty:
                    all_games.append(day_games)

            if not all_games:
                self._set_last_fetch_status(
                    'failed',
                    f"Unable to build remaining-season schedule for {season}",
                    {
                        'season': season,
                        'source': 'composed_daily_schedule',
                        'days_attempted': max_days + 1,
                    },
                )
                return pd.DataFrame()

            full_df = pd.concat(all_games).drop_duplicates(subset=['GAME_ID'])
            full_df.to_csv(cache_file, index=False)
            self._set_last_fetch_status(
                'success',
                f"Built remaining-season schedule for {season}",
                {
                    'season': season,
                    'source': 'composed_daily_schedule',
                    'rows': int(len(full_df)),
                    'days_attempted': max_days + 1,
                },
            )
            return full_df

        except Exception as e:
            logger.error(f"Error fetching remaining season: {e}")
            self._set_last_fetch_status(
                'failed',
                f"Error fetching remaining-season schedule for {season}",
                {
                    'season': season,
                    'source': 'composed_daily_schedule',
                    'error': str(e),
                },
            )
            return pd.DataFrame()
