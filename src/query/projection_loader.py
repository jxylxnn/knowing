import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlayerProjection:
    player_name: str
    team: str
    opponent: str
    is_home: bool
    date: str
    
    pts_mean: float
    pts_mode: float
    pts_std: float
    pts_ci_low: float
    pts_ci_high: float
    
    reb_mean: float
    reb_mode: float
    reb_std: float
    reb_ci_low: float
    reb_ci_high: float
    
    ast_mean: float
    ast_mode: float
    ast_std: float
    ast_ci_low: float
    ast_ci_high: float

    stl_mean: float
    stl_mode: float
    stl_std: float
    stl_ci_low: float
    stl_ci_high: float

    blk_mean: float
    blk_mode: float
    blk_std: float
    blk_ci_low: float
    blk_ci_high: float

    tov_mean: float
    tov_mode: float
    tov_std: float
    tov_ci_low: float
    tov_ci_high: float
    
    play_probability: float = 1.0
    game_id: Optional[str] = None
    
    def get_stat_mean(self, stat: str) -> float:
        stat = stat.lower()
        return getattr(self, f'{stat}_mean', 0.0)
    
    def get_stat_std(self, stat: str) -> float:
        stat = stat.lower()
        return getattr(self, f'{stat}_std', 0.0)
    
    def get_stat_ci(self, stat: str) -> Tuple[float, float]:
        stat = stat.lower()
        return (
            getattr(self, f'{stat}_ci_low', 0.0),
            getattr(self, f'{stat}_ci_high', 0.0)
        )


class ProjectionLoader:
    STAT_COLUMNS = {
        'pts': {
            'mean': 'PROJ_PTS_MEAN',
            'mode': 'PROJ_PTS_MODE',
            'std': None,
            'ci_low': 'PTS_CI_LOW',
            'ci_high': 'PTS_CI_HIGH'
        },
        'reb': {
            'mean': 'PROJ_REB_MEAN',
            'mode': 'PROJ_REB_MODE',
            'std': None,
            'ci_low': 'REB_CI_LOW',
            'ci_high': 'REB_CI_HIGH'
        },
        'ast': {
            'mean': 'PROJ_AST_MEAN',
            'mode': 'PROJ_AST_MODE',
            'std': None,
            'ci_low': 'AST_CI_LOW',
            'ci_high': 'AST_CI_HIGH'
        },
        'stl': {
            'mean': 'PROJ_STL_MEAN',
            'mode': 'PROJ_STL_MODE',
            'std': None,
            'ci_low': 'STL_CI_LOW',
            'ci_high': 'STL_CI_HIGH'
        },
        'blk': {
            'mean': 'PROJ_BLK_MEAN',
            'mode': 'PROJ_BLK_MODE',
            'std': None,
            'ci_low': 'BLK_CI_LOW',
            'ci_high': 'BLK_CI_HIGH'
        },
        'tov': {
            'mean': 'PROJ_TOV_MEAN',
            'mode': 'PROJ_TOV_MODE',
            'std': None,
            'ci_low': 'TOV_CI_LOW',
            'ci_high': 'TOV_CI_HIGH'
        }
    }
    
    STAT_DISPLAY_NAMES = {
        'pts': 'Points',
        'reb': 'Rebounds', 
        'ast': 'Assists',
        'stl': 'Steals',
        'blk': 'Blocks',
        'tov': 'Turnovers'
    }
    
    GAME_LOGS_COLUMNS = [
        'PLAYER_NAME', 'PLAYER_ID', 'GAME_DATE', 'GAME_ID', 'MATCHUP', 
        'WL', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
        'FGM', 'FGA', 'PLUS_MINUS'
    ]
    
    def __init__(self, data_dir: str = 'data/sim_results', players_data_path: str = 'data/nba_players.csv'):
        self.data_dir = Path(data_dir)
        self.players_data_path = Path(players_data_path)
        self._projections_cache: Optional[pd.DataFrame] = None
        self._player_index: Dict[str, List[int]] = {}
        self._cache_file: Optional[str] = None
        self._last_load_time: Optional[datetime] = None
        self._player_game_logs: Optional[pd.DataFrame] = None
        self._game_log_player_index: Optional[Dict[str, List[int]]] = None
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._player_games_cache: Dict[str, pd.DataFrame] = {}
        self._defense_scraper = None
    
    def _find_latest_projection_file(self) -> Optional[Path]:
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return None
        
        pattern = "player_projections_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No projection files found matching {pattern}")
            return None
        
        def extract_timestamp(f: Path) -> datetime:
            match = re.search(r'(\d{8}_\d{6})', f.name)
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
            return datetime.fromtimestamp(f.stat().st_mtime)
        
        files.sort(key=extract_timestamp, reverse=True)
        return files[0]
    
    def load_projections(self, force_reload: bool = False) -> pd.DataFrame:
        if self._projections_cache is not None and not force_reload:
            return self._projections_cache
        
        projection_file = self._find_latest_projection_file()
        
        if projection_file is None:
            logger.warning("No projection file found, returning empty DataFrame")
            return pd.DataFrame()
        
        if self._cache_file == str(projection_file) and self._projections_cache is not None:
            return self._projections_cache
        
        try:
            self._projections_cache = pd.read_csv(projection_file)
            self._cache_file = str(projection_file)
            self._last_load_time = datetime.now()
            self._build_player_index()
            
            logger.info(f"Loaded {len(self._projections_cache)} projections from {projection_file.name}")
            return self._projections_cache
            
        except Exception as e:
            logger.error(f"Failed to load projections: {e}")
            return pd.DataFrame()
    
    def _build_player_index(self):
        self._player_index = {}
        
        if self._projections_cache is None or self._projections_cache.empty:
            return
        
        for idx, row in self._projections_cache.iterrows():
            player_name = str(row.get('PLAYER_NAME', '')).lower()
            
            for name_part in player_name.split():
                if name_part not in self._player_index:
                    self._player_index[name_part] = []
                self._player_index[name_part].append(idx)
            
            if player_name not in self._player_index:
                self._player_index[player_name] = []
            self._player_index[player_name].append(idx)
    
    def find_player(
        self,
        player_name: str,
        team: Optional[str] = None,
        opponent: Optional[str] = None,
        date: Optional[str] = None
    ) -> Optional[PlayerProjection]:
        df = self.load_projections()
        
        if df.empty:
            return None
        
        player_name_lower = player_name.lower().strip()
        
        matches = df[df['PLAYER_NAME'].str.lower().str.contains(player_name_lower, na=False)]
        
        if matches.empty:
            name_parts = player_name_lower.split()
            for part in name_parts:
                if len(part) >= 3:
                    matches = df[df['PLAYER_NAME'].str.lower().str.contains(part, na=False)]
                    if not matches.empty:
                        break
        
        if matches.empty:
            return None
        
        if team:
            matches = matches[matches['TEAM'].str.upper() == team.upper()]
        
        if opponent:
            matches = matches[matches['OPPONENT'].str.upper() == opponent.upper()]
        
        if date:
            matches = matches[matches['DATE'].astype(str).str.contains(date)]
        
        if matches.empty:
            return None
        
        if len(matches) > 1:
            matches = matches.head(1)
        
        row = matches.iloc[0]
        
        # Surface degradation warning for projections using fallback data
        quality = row.get('DATA_QUALITY', 'FULL')
        if quality != 'FULL':
            print(f"\u26a0\ufe0f  [WARNING: Projection relies on fallback data - Quality: {quality}]")
        
        return self._row_to_projection(row)
    
    def find_all_players(
        self,
        team: Optional[str] = None,
        opponent: Optional[str] = None,
        date: Optional[str] = None
    ) -> List[PlayerProjection]:
        df = self.load_projections()
        
        if df.empty:
            return []
        
        matches = df.copy()
        
        if team:
            matches = matches[matches['TEAM'].str.upper() == team.upper()]
        
        if opponent:
            matches = matches[matches['OPPONENT'].str.upper() == opponent.upper()]
        
        if date:
            matches = matches[matches['DATE'].astype(str).str.contains(date)]
        
        if matches.empty:
            return []
        
        return [self._row_to_projection(row) for _, row in matches.iterrows()]
    
    def _row_to_projection(self, row: pd.Series) -> PlayerProjection:
        def safe_float(val, default=0.0):
            try:
                return float(val) if pd.notna(val) else default
            except (ValueError, TypeError):
                return default

        def get_required_float(column: str, default: float = 0.0) -> float:
            """Get a float from row, returning default if missing/None."""
            if column not in row.index:
                return default
            value = safe_float(row.get(column), default=None)
            return value if value is not None else default

        def get_required_stat_columns(stat: str) -> Dict[str, float]:
            """Get all columns for a stat using STAT_COLUMNS mapping."""
            cols = self.STAT_COLUMNS.get(stat, {})
            return {
                'mean': get_required_float(cols.get('mean', f'PROJ_{stat.upper()}_MEAN')),
                'mode': get_required_float(cols.get('mode', f'PROJ_{stat.upper()}_MODE')),
                'ci_low': get_required_float(cols.get('ci_low', f'{stat.upper()}_CI_LOW')),
                'ci_high': get_required_float(cols.get('ci_high', f'{stat.upper()}_CI_HIGH')),
            }
        
        def estimate_std(mean: float, ci_low: float, ci_high: float) -> float:
            if ci_low and ci_high and ci_high > ci_low:
                return (ci_high - ci_low) / 3.92
            return max(mean * 0.35, 1.5)
        
        # Get all stat values using STAT_COLUMNS mapping
        pts_data = get_required_stat_columns('pts')
        reb_data = get_required_stat_columns('reb')
        ast_data = get_required_stat_columns('ast')
        stl_data = get_required_stat_columns('stl')
        blk_data = get_required_stat_columns('blk')
        tov_data = get_required_stat_columns('tov')
        
        return PlayerProjection(
            player_name=str(row.get('PLAYER_NAME', 'Unknown')),
            team=str(row.get('TEAM', '')),
            opponent=str(row.get('OPPONENT', '')),
            is_home=bool(row.get('IS_HOME', False)),
            date=str(row.get('DATE', '')),
            
            pts_mean=pts_data['mean'],
            pts_mode=pts_data['mode'],
            pts_std=estimate_std(pts_data['mean'], pts_data['ci_low'], pts_data['ci_high']),
            pts_ci_low=pts_data['ci_low'],
            pts_ci_high=pts_data['ci_high'],
            
            reb_mean=reb_data['mean'],
            reb_mode=reb_data['mode'],
            reb_std=estimate_std(reb_data['mean'], reb_data['ci_low'], reb_data['ci_high']),
            reb_ci_low=reb_data['ci_low'],
            reb_ci_high=reb_data['ci_high'],
            
            ast_mean=ast_data['mean'],
            ast_mode=ast_data['mode'],
            ast_std=estimate_std(ast_data['mean'], ast_data['ci_low'], ast_data['ci_high']),
            ast_ci_low=ast_data['ci_low'],
            ast_ci_high=ast_data['ci_high'],

            stl_mean=stl_data['mean'],
            stl_mode=stl_data['mode'],
            stl_std=estimate_std(stl_data['mean'], stl_data['ci_low'], stl_data['ci_high']),
            stl_ci_low=stl_data['ci_low'],
            stl_ci_high=stl_data['ci_high'],

            blk_mean=blk_data['mean'],
            blk_mode=blk_data['mode'],
            blk_std=estimate_std(blk_data['mean'], blk_data['ci_low'], blk_data['ci_high']),
            blk_ci_low=blk_data['ci_low'],
            blk_ci_high=blk_data['ci_high'],

            tov_mean=tov_data['mean'],
            tov_mode=tov_data['mode'],
            tov_std=estimate_std(tov_data['mean'], tov_data['ci_low'], tov_data['ci_high']),
            tov_ci_low=tov_data['ci_low'],
            tov_ci_high=tov_data['ci_high'],
            
            play_probability=safe_float(row.get('PLAY_PROBABILITY', 1.0)),
            game_id=str(row.get('GAME_ID', ''))
        )
    
    def get_available_players(self) -> List[str]:
        df = self.load_projections()
        
        if df.empty:
            return []
        
        return sorted(df['PLAYER_NAME'].unique().tolist())
    
    def get_available_teams(self) -> List[str]:
        df = self.load_projections()
        
        if df.empty:
            return []
        
        teams = set(df['TEAM'].unique()) | set(df['OPPONENT'].unique())
        return sorted(teams)
    
    def get_available_dates(self) -> List[str]:
        df = self.load_projections()
        
        if df.empty:
            return []
        
        return sorted(df['DATE'].unique().tolist())
    
    def get_cache_info(self) -> Dict[str, Any]:
        return {
            'cache_file': self._cache_file,
            'last_load_time': self._last_load_time.isoformat() if self._last_load_time else None,
            'num_projections': len(self._projections_cache) if self._projections_cache is not None else 0,
            'num_players': len(self._player_index)
        }
    
    def _load_player_game_logs(self) -> pd.DataFrame:
        if self._player_game_logs is not None:
            return self._player_game_logs
        
        if not self.players_data_path.exists():
            logger.warning(f"Player data file not found: {self.players_data_path}")
            return pd.DataFrame()
        
        try:
            self._player_game_logs = pd.read_csv(
                self.players_data_path,
                usecols=self.GAME_LOGS_COLUMNS,
                low_memory=False
            )
            self._player_game_logs['GAME_DATE'] = pd.to_datetime(
                self._player_game_logs['GAME_DATE'], errors='coerce'
            )
            self._player_game_logs = self._player_game_logs.sort_values('GAME_DATE')
            self._player_game_logs = self._player_game_logs.reset_index(drop=True)
            logger.info(f"Loaded {len(self._player_game_logs)} player game logs")
            return self._player_game_logs
        except Exception as e:
            logger.error(f"Failed to load player game logs: {e}")
            return pd.DataFrame()
    
    def _get_player_games_df(self, player_name: str) -> Optional[pd.DataFrame]:
        cache_key = player_name.lower().strip()
        if cache_key in self._player_games_cache:
            return self._player_games_cache[cache_key]
        
        df = self._load_player_game_logs()
        if df.empty:
            return None
        
        name_lower = cache_key
        matches = df[df['PLAYER_NAME'].str.lower() == name_lower]
        
        if matches.empty:
            name_parts = name_lower.split()
            for part in name_parts:
                if len(part) >= 3:
                    matches = df[df['PLAYER_NAME'].str.lower().str.contains(part, case=False, na=False, regex=False)]
                    if not matches.empty:
                        break
        
        if matches.empty:
            return None
        
        self._player_games_cache[cache_key] = matches
        return matches
    
    def get_recent_games(self, player_name: str, n: int = 5) -> List[Dict[str, Any]]:
        matches = self._get_player_games_df(player_name)
        
        if matches is None or matches.empty:
            return []
        
        matches = matches.sort_values('GAME_DATE', ascending=False)
        matches = matches.drop_duplicates(subset=['GAME_DATE'], keep='first')
        matches = matches.head(n)
        
        games = []
        for _, row in matches.iterrows():
            matchup = str(row.get('MATCHUP', ''))
            is_home = 'vs.' in matchup.lower() or '@' not in matchup.lower()
            
            fga = row.get('FGA', 0)
            fgm = row.get('FGM', 0)
            fg_pct = (float(fgm) / float(fga) * 100) if fga and float(fga) > 0 else 0.0
            
            wl = str(row.get('WL', ''))
            result = 'W' if wl == 'W' else 'L'
            
            game_date = row.get('GAME_DATE')
            date_str = game_date.strftime('%Y-%m-%d') if pd.notna(game_date) else ''
            date_short = game_date.strftime('%b %d') if pd.notna(game_date) else ''
            
            games.append({
                'date': date_str,
                'date_short': date_short,
                'min': float(row.get('MIN', 0) or 0),
                'pts': float(row.get('PTS', 0) or 0),
                'reb': float(row.get('REB', 0) or 0),
                'ast': float(row.get('AST', 0) or 0),
                'stl': float(row.get('STL', 0) or 0),
                'blk': float(row.get('BLK', 0) or 0),
                'tov': float(row.get('TOV', 0) or 0),
                'fg_pct': fg_pct,
                'matchup': matchup,
                'result': result,
                'is_home': is_home,
                'plus_minus': float(row.get('PLUS_MINUS', 0) or 0)
            })
        
        return games
    
    def get_matchup_history(self, player_name: str, opponent: str, n: int = 5) -> List[Dict[str, Any]]:
        matches = self._get_player_games_df(player_name)
        
        if matches is None or matches.empty:
            return []
        
        opponent_upper = opponent.upper()
        matchup_matches = matches[
            matches['MATCHUP'].str.upper().str.contains(opponent_upper, na=False, regex=False)
        ]
        
        if matchup_matches.empty:
            return []
        
        matchup_matches = matchup_matches.sort_values('GAME_DATE', ascending=False)
        matchup_matches = matchup_matches.drop_duplicates(subset=['GAME_DATE'], keep='first')
        matchup_matches = matchup_matches.head(n)
        
        games = []
        for _, row in matchup_matches.iterrows():
            matchup = str(row.get('MATCHUP', ''))
            game_date = row.get('GAME_DATE')
            
            games.append({
                'date': game_date.strftime('%Y-%m-%d') if pd.notna(game_date) else '',
                'date_short': game_date.strftime('%b %d, %Y') if pd.notna(game_date) else '',
                'min': float(row.get('MIN', 0) or 0),
                'pts': float(row.get('PTS', 0) or 0),
                'reb': float(row.get('REB', 0) or 0),
                'ast': float(row.get('AST', 0) or 0),
                'matchup': matchup,
                'result': 'W' if row.get('WL') == 'W' else 'L'
            })
        
        return games
    
    def get_opponent_defense_profile(self, team_abbr: str) -> Dict[str, Any]:
        team_abbr = team_abbr.upper()
        
        cached_defense = self._load_cached_defense_data(team_abbr)
        if cached_defense:
            return cached_defense
        
        if self._defense_scraper is None:
            try:
                from src.data.nba_defense_scraper import NBADefenseScraper
                self._defense_scraper = NBADefenseScraper()
            except Exception as e:
                logger.debug(f"Could not initialize defense scraper: {e}")
                return self._get_default_defense_profile(team_abbr)
        
        try:
            defense_data = self._defense_scraper.get_team_defense_allowed(team_abbr)
            
            pts_allowed = (
                defense_data.get('pts_allowed_per_100') or 
                defense_data.get('opp_pts_per_100') or 
                115.0
            )
            league_rank = defense_data.get('league_rank') or defense_data.get('RANK') or 15
            
            return {
                'team': team_abbr,
                'pts_allowed_per_100': float(pts_allowed),
                'reb_allowed_per_game': float(defense_data.get('opp_reb', 44.0)),
                'ast_allowed_per_game': float(defense_data.get('opp_ast', 26.0)),
                'fg_pct_allowed': float(defense_data.get('opp_fg_pct', 0.47)),
                'three_pct_allowed': float(defense_data.get('opp_fg3_pct', 0.36)),
                'defensive_rating': float(pts_allowed),
                'league_rank': int(league_rank),
                'position_ranks': defense_data.get('position_ranks', {}),
                'source': defense_data.get('source', 'unknown')
            }
        except Exception as e:
            logger.debug(f"Could not fetch defense data: {e}")
            return self._get_default_defense_profile(team_abbr)
    
    def _load_cached_defense_data(self, team_abbr: str) -> Optional[Dict[str, Any]]:
        import json
        from pathlib import Path
        
        cache_dir = Path(self.data_dir).parent / 'cache'
        candidates = sorted(cache_dir.glob('all_team_defense_*.json'))
        if candidates:
            cache_file = candidates[-1]
        else:
            cache_file = cache_dir / 'all_team_defense_2025-26.json'
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                all_defense = json.load(f)
            
            team_data = all_defense.get(team_abbr)
            if not team_data:
                return None
            
            return {
                'team': team_abbr,
                'pts_allowed_per_100': float(team_data.get('pts_allowed_per_100', 115.0)),
                'reb_allowed_per_game': float(team_data.get('opp_reb', 44.0)),
                'ast_allowed_per_game': float(team_data.get('opp_ast', 26.0)),
                'fg_pct_allowed': float(team_data.get('opp_fg_pct', 0.47)),
                'three_pct_allowed': float(team_data.get('opp_fg3_pct', 0.36)),
                'defensive_rating': float(team_data.get('pts_allowed_per_100', 115.0)),
                'league_rank': int(team_data.get('league_rank', 15)),
                'position_ranks': {},
                'source': team_data.get('source', 'cached')
            }
        except Exception as e:
            logger.debug(f"Could not load cached defense data: {e}")
            return None
    
    def _get_default_defense_profile(self, team_abbr: str) -> Dict[str, Any]:
        default_profiles = {
            'BOS': {'pts_allowed_per_100': 110.2, 'reb_allowed_per_game': 43.1, 'league_rank': 2},
            'OKC': {'pts_allowed_per_100': 108.5, 'reb_allowed_per_game': 42.8, 'league_rank': 1},
            'MIN': {'pts_allowed_per_100': 111.8, 'reb_allowed_per_game': 44.2, 'league_rank': 4},
            'MIA': {'pts_allowed_per_100': 112.3, 'reb_allowed_per_game': 43.5, 'league_rank': 5},
            'CLE': {'pts_allowed_per_100': 111.5, 'reb_allowed_per_game': 43.8, 'league_rank': 3},
        }
        
        defaults = default_profiles.get(team_abbr, {
            'pts_allowed_per_100': 115.0,
            'reb_allowed_per_game': 44.5,
            'league_rank': 15
        })
        
        return {
            'team': team_abbr,
            'pts_allowed_per_100': defaults.get('pts_allowed_per_100', 115.0),
            'reb_allowed_per_game': defaults.get('reb_allowed_per_game', 44.5),
            'ast_allowed_per_game': 26.0,
            'fg_pct_allowed': 0.47,
            'three_pct_allowed': 0.36,
            'defensive_rating': defaults.get('pts_allowed_per_100', 115.0),
            'league_rank': defaults.get('league_rank', 15),
            'position_ranks': {}
        }
    
    def get_player_context(self, player_name: str, opponent: str, stat: str) -> Dict[str, Any]:
        cache_key = f"{player_name.lower()}_{opponent}_{stat}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        recent_games = self.get_recent_games(player_name, n=5)
        matchup_history = self.get_matchup_history(player_name, opponent, n=5)
        opponent_defense = self.get_opponent_defense_profile(opponent)
        
        recent_avg = {'pts': 0, 'reb': 0, 'ast': 0, 'stl': 0, 'blk': 0, 'tov': 0, 'min': 0}
        if recent_games:
            for game in recent_games:
                recent_avg['pts'] += game['pts']
                recent_avg['reb'] += game['reb']
                recent_avg['ast'] += game['ast']
                recent_avg['stl'] += game['stl']
                recent_avg['blk'] += game['blk']
                recent_avg['tov'] += game['tov']
                recent_avg['min'] += game['min']
            n = len(recent_games)
            recent_avg = {k: v / n for k, v in recent_avg.items()}
        recent_sample_size = len(recent_games) if recent_games else 0
        
        matchup_avg = {'pts': 0, 'reb': 0, 'ast': 0, 'stl': 0, 'blk': 0, 'tov': 0, 'min': 0}
        if matchup_history:
            for game in matchup_history:
                matchup_avg['pts'] += game['pts']
                matchup_avg['reb'] += game['reb']
                matchup_avg['ast'] += game['ast']
                matchup_avg['stl'] += game.get('stl', 0)
                matchup_avg['blk'] += game.get('blk', 0)
                matchup_avg['tov'] += game.get('tov', 0)
                matchup_avg['min'] += game['min']
            n = len(matchup_history)
            matchup_avg = {k: v / n for k, v in matchup_avg.items()}
        matchup_sample_size = len(matchup_history) if matchup_history else 0
        
        trend = self._calculate_trend(recent_games, stat)
        
        result = {
            'recent_games': recent_games,
            'recent_avg': recent_avg,
            'recent_sample_size': recent_sample_size,
            'matchup_history': matchup_history,
            'matchup_avg': matchup_avg,
            'matchup_sample_size': matchup_sample_size,
            'opponent_defense': opponent_defense,
            'trend': trend
        }
        
        self._context_cache[cache_key] = result
        return result
    
    def _calculate_trend(self, games: List[Dict], stat: str) -> Dict[str, Any]:
        if len(games) < 3:
            return {'direction': 'neutral', 'pct_change': 0, 'description': 'Insufficient data'}
        
        stat_key = stat.lower()
        recent_3 = sum(g.get(stat_key, 0) for g in games[:3]) / 3
        earlier = sum(g.get(stat_key, 0) for g in games[3:]) / max(len(games) - 3, 1)
        
        if earlier == 0:
            pct_change = 0
        else:
            pct_change = ((recent_3 - earlier) / earlier) * 100
        
        if pct_change > 10:
            direction = 'up'
            description = f'↗ Up {pct_change:.0f}% last 3 games'
        elif pct_change < -10:
            direction = 'down'
            description = f'↘ Down {abs(pct_change):.0f}% last 3 games'
        else:
            direction = 'neutral'
            description = f'→ Flat ({pct_change:+.0f}%)'
        
        return {
            'direction': direction,
            'pct_change': pct_change,
            'description': description,
            'recent_3_avg': recent_3,
            'earlier_avg': earlier
        }
