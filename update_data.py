"""
NBA Data Updater - Fetches latest player and team game data from NBA.com.

Usage:
    python update_data.py                           # Update current season
    python update_data.py --update                  # Fetch only new games since last update (fast)
    python update_data.py --season 2024-25          # Specific season
    python update_data.py --all-seasons             # Last 10 seasons
    python update_data.py --interactive             # Interactive selection (RECOMMENDED)
    python update_data.py --full-scrape             # All NBA seasons since 1946-47
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional, Set

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, teamgamelogs
from nba_api.stats.static import players as nba_players, teams as nba_teams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
PLAYERS_FILE = os.path.join(DATA_DIR, 'nba_players.csv')
GAMES_FILE = os.path.join(DATA_DIR, 'nba_games.csv')

SEASONS = [
    '1946-47', '1947-48', '1948-49', '1949-50',
    '1950-51', '1951-52', '1952-53', '1953-54',
    '1954-55', '1955-56', '1956-57', '1957-58',
    '1958-59', '1959-60', '1960-61', '1961-62',
    '1962-63', '1963-64', '1964-65', '1965-66',
    '1966-67', '1967-68', '1968-69', '1969-70',
    '1970-71', '1971-72', '1972-73', '1973-74',
    '1974-75', '1975-76', '1976-77', '1977-78',
    '1978-79', '1979-80', '1980-81', '1981-82',
    '1982-83', '1983-84', '1984-85', '1985-86',
    '1986-87', '1987-88', '1988-89', '1989-90',
    '1990-91', '1991-92', '1992-93', '1993-94',
    '1994-95', '1995-96', '1996-97', '1997-98',
    '1998-99', '1999-00', '2000-01', '2001-02',
    '2002-03', '2003-04', '2004-05', '2005-06',
    '2006-07', '2007-08', '2008-09', '2009-10',
    '2010-11', '2011-12', '2012-13', '2013-14',
    '2014-15', '2015-16', '2016-17', '2017-18',
    '2018-19', '2019-20', '2020-21', '2021-22',
    '2022-23', '2023-24', '2024-25', '2025-26'
]

RATE_LIMIT_DELAY = 0.6


def get_current_season() -> str:
    now = datetime.now()
    year = now.year
    if now.month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"


def get_latest_game_date(file_path: str) -> Optional[str]:
    """Get the most recent game date from existing data."""
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        if 'GAME_DATE' not in df.columns or df.empty:
            return None
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        latest = df['GAME_DATE'].max()
        return latest.strftime('%Y-%m-%d')
    except Exception as e:
        logger.warning(f"Could not read latest game date: {e}")
        return None


def get_season_for_date(date_str: str) -> str:
    """Determine which NBA season a date belongs to."""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    month = date.month
    
    if month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"


def get_seasons_between_dates(start_date: str, end_date: Optional[str] = None) -> List[str]:
    """Get all NBA seasons that overlap with a date range."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end = datetime.now()
    
    seasons = set()
    current = start
    
    while current <= end:
        season = get_season_for_date(current.strftime('%Y-%m-%d'))
        if season in SEASONS:
            seasons.add(season)
        current = current.replace(year=current.year + 1) if current.month < 10 else current.replace(year=current.year + 1, month=1)
    
    if not seasons:
        return [get_current_season()]
    
    return sorted(list(seasons), key=lambda s: SEASONS.index(s) if s in SEASONS else 999)


def fetch_player_logs(season: str) -> Optional[pd.DataFrame]:
    logger.info(f"Fetching player game logs for {season}...")
    
    try:
        result = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        df = result.get_data_frames()[0]
        
        if df.empty:
            logger.warning(f"No player data for {season}")
            return None
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
        
        if 'FANTASY_PTS' not in df.columns:
            df['FANTASY_PTS'] = (
                df['PTS'] + 
                1.2 * df['REB'] + 
                1.5 * df['AST'] + 
                2.0 * df['STL'] + 
                2.0 * df['BLK'] - 
                df['TOV']
            )
        
        logger.info(f"Fetched {len(df)} player game records for {season}")
        time.sleep(RATE_LIMIT_DELAY)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching player logs for {season}: {e}")
        return None


def fetch_team_logs(season: str) -> Optional[pd.DataFrame]:
    logger.info(f"Fetching team game logs for {season}...")
    
    try:
        result = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        df = result.get_data_frames()[0]
        
        if df.empty:
            logger.warning(f"No team data for {season}")
            return None
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
        
        logger.info(f"Fetched {len(df)} team game records for {season}")
        time.sleep(RATE_LIMIT_DELAY)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching team logs for {season}: {e}")
        return None


def fetch_player_logs_since(season: str, date_from: str) -> Optional[pd.DataFrame]:
    """Fetch player game logs from a specific date onwards."""
    logger.info(f"Fetching player game logs for {season} from {date_from}...")
    
    try:
        result = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable='Regular Season',
            date_from_nullable=date_from
        )
        df = result.get_data_frames()[0]
        
        if df.empty:
            logger.info(f"No new player games for {season} since {date_from}")
            return None
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
        
        if 'FANTASY_PTS' not in df.columns:
            df['FANTASY_PTS'] = (
                df['PTS'] + 
                1.2 * df['REB'] + 
                1.5 * df['AST'] + 
                2.0 * df['STL'] + 
                2.0 * df['BLK'] - 
                df['TOV']
            )
        
        logger.info(f"Fetched {len(df)} new player game records for {season} since {date_from}")
        time.sleep(RATE_LIMIT_DELAY)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching player logs for {season} since {date_from}: {e}")
        return None


def fetch_team_logs_since(season: str, date_from: str) -> Optional[pd.DataFrame]:
    """Fetch team game logs from a specific date onwards."""
    logger.info(f"Fetching team game logs for {season} from {date_from}...")
    
    try:
        result = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable='Regular Season',
            date_from_nullable=date_from
        )
        df = result.get_data_frames()[0]
        
        if df.empty:
            logger.info(f"No new team games for {season} since {date_from}")
            return None
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
        
        logger.info(f"Fetched {len(df)} new team game records for {season} since {date_from}")
        time.sleep(RATE_LIMIT_DELAY)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching team logs for {season} since {date_from}: {e}")
        return None


def load_existing_data(file_path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded existing data: {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
    return None


def merge_data(existing: Optional[pd.DataFrame], new: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if existing is None:
        return new
    
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep='last')
    return combined


def parse_selection_input(user_input: str) -> Set[int]:
    """Parse user input string into a set of season indices (1-indexed)."""
    selected = set()
    user_input = user_input.strip().lower()
    
    if user_input == 'all':
        return set(range(1, len(SEASONS) + 1))
    
    if user_input == 'recent':
        return set(range(len(SEASONS) - 9, len(SEASONS) + 1))
    
    parts = [p.strip() for p in user_input.split(',')]
    
    for part in parts:
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                if 1 <= start_idx <= len(SEASONS) and 1 <= end_idx <= len(SEASONS):
                    for i in range(start_idx, end_idx + 1):
                        selected.add(i)
            except ValueError:
                continue
        else:
            try:
                idx = int(part)
                if 1 <= idx <= len(SEASONS):
                    selected.add(idx)
            except ValueError:
                continue
    
    return selected


def select_seasons_interactive() -> List[str]:
    """Display interactive menu for season selection."""
    print("\n" + "=" * 70)
    print("                         NBA SEASONS")
    print("=" * 70)
    
    for i in range(0, len(SEASONS), 4):
        row = ""
        for j in range(4):
            if i + j < len(SEASONS):
                idx = i + j + 1
                season = SEASONS[i + j]
                row += f"  {idx:3}. {season}"
        print(row)
    
    print("=" * 70)
    print("\nInput options:")
    print("  - Single:     5")
    print("  - Multiple:   1,5,10,15")
    print("  - Range:      50-80")
    print("  - Combined:   1,5,10-15,20")
    print("  - 'all'       All NBA seasons (1946-47 to present)")
    print("  - 'recent'    Last 10 seasons")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\nSelect seasons: ").strip()
            
            if not user_input:
                print("Please enter a selection.")
                continue
            
            selected_indices = parse_selection_input(user_input)
            
            if not selected_indices:
                print("Invalid selection. Please try again.")
                continue
            
            selected_seasons = [SEASONS[i - 1] for i in sorted(selected_indices)]
            
            print(f"\nSelected {len(selected_seasons)} season(s):")
            if len(selected_seasons) <= 10:
                print(f"  {', '.join(selected_seasons)}")
            else:
                print(f"  {', '.join(selected_seasons[:5])} ... {', '.join(selected_seasons[-3:])}")
            
            confirm = input("\nProceed? [Y/n]: ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                return selected_seasons
            elif confirm in ['n', 'no']:
                print("Selection cancelled. Please select again.")
                continue
            else:
                return selected_seasons
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)


def check_empty_data() -> bool:
    """Check if data files exist and prompt for full scrape if not."""
    players_exist = os.path.exists(PLAYERS_FILE)
    games_exist = os.path.exists(GAMES_FILE)
    
    if not players_exist and not games_exist:
        print("\n" + "!" * 70)
        print("  WARNING: No existing data found!")
        print("  Data files will be created from scratch.")
        print("!" * 70)
        print("\nOptions:")
        print("  [1] Full scrape - All NBA seasons (1946-47 to present)")
        print("  [2] Recent seasons - Last 10 seasons (recommended)")
        print("  [3] Current season only")
        print("  [4] Interactive selection")
        print()
        
        while True:
            try:
                choice = input("Select option [1-4]: ").strip()
                if choice == '1':
                    return True, SEASONS
                elif choice == '2':
                    return True, SEASONS[-10:]
                elif choice == '3':
                    return True, [get_current_season()]
                elif choice == '4':
                    return True, None  # Signal to use interactive mode
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)
    
    return False, None


def update_season(season: str, existing_players: Optional[pd.DataFrame], 
                  existing_teams: Optional[pd.DataFrame]) -> tuple:
    player_df = fetch_player_logs(season)
    team_df = fetch_team_logs(season)
    
    if player_df is not None:
        existing_players = merge_data(
            existing_players, player_df, 
            ['PLAYER_ID', 'GAME_ID']
        )
    
    if team_df is not None:
        existing_teams = merge_data(
            existing_teams, team_df,
            ['TEAM_ID', 'GAME_ID']
        )
    
    return existing_players, existing_teams


def update_since_date(date_from: str, existing_players: Optional[pd.DataFrame], 
                      existing_teams: Optional[pd.DataFrame]) -> tuple:
    """Fetch only new games since the specified date."""
    seasons = get_seasons_between_dates(date_from)
    
    total_new_players = 0
    total_new_teams = 0
    
    for season in seasons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Fetching new games for {season} since {date_from}")
        logger.info(f"{'='*50}")
        
        player_df = fetch_player_logs_since(season, date_from)
        team_df = fetch_team_logs_since(season, date_from)
        
        if player_df is not None:
            if existing_players is not None:
                before_count = len(existing_players)
                existing_players = merge_data(
                    existing_players, player_df, 
                    ['PLAYER_ID', 'GAME_ID']
                )
                after_count = len(existing_players)
                new_count = after_count - before_count
            else:
                existing_players = player_df
                new_count = len(player_df)
            total_new_players += new_count
            logger.info(f"Added {new_count} new player game records")
        
        if team_df is not None:
            if existing_teams is not None:
                before_count = len(existing_teams)
                existing_teams = merge_data(
                    existing_teams, team_df,
                    ['TEAM_ID', 'GAME_ID']
                )
                after_count = len(existing_teams)
                new_count = after_count - before_count
            else:
                existing_teams = team_df
                new_count = len(team_df)
            total_new_teams += new_count
            logger.info(f"Added {new_count} new team game records")
    
    return existing_players, existing_teams, total_new_players, total_new_teams


def save_data(players_df: pd.DataFrame, teams_df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    players_df = players_df.sort_values(['GAME_DATE', 'PLAYER_NAME'])
    teams_df = teams_df.sort_values(['GAME_DATE', 'TEAM_ABBREVIATION'])
    
    players_df.to_csv(PLAYERS_FILE, index=False)
    teams_df.to_csv(GAMES_FILE, index=False)
    
    logger.info(f"Saved {len(players_df)} player records to {PLAYERS_FILE}")
    logger.info(f"Saved {len(teams_df)} team records to {GAMES_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description='Update NBA data from NBA.com',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_data.py                    # Update current season
  python update_data.py --update           # Fetch only new games since last update (fast)
  python update_data.py --season 2024-25   # Fetch specific season
  python update_data.py --all-seasons      # Last 10 seasons
  python update_data.py --interactive      # Interactive selection (RECOMMENDED)
  python update_data.py --full-scrape      # All NBA seasons since 1946-47
  python update_data.py --force            # Force re-fetch (ignore existing data)

Note: nba_api has reliable data from 1996-97 onward. Earlier seasons may have limited data.
        """
    )
    parser.add_argument('--season', type=str, action='append', dest='seasons',
                        help='Specific season(s) (e.g., --season 2023-24 --season 2024-25)')
    parser.add_argument('--all-seasons', action='store_true', 
                        help='Fetch last 10 seasons (2015-16 to present)')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive season selection menu')
    parser.add_argument('--full-scrape', action='store_true',
                        help='Fetch ALL NBA seasons since 1946-47')
    parser.add_argument('--update', action='store_true',
                        help='Fetch only new games since the last game in dataset (incremental update)')
    parser.add_argument('--force', action='store_true', 
                        help='Force re-fetch all data (ignore existing)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to save data files (default: data)')
    parser.add_argument('--current-season', action='store_true',
                        help='Fetch current season only')
    
    args = parser.parse_args()
    
    global DATA_DIR, PLAYERS_FILE, GAMES_FILE
    DATA_DIR = args.data_dir
    PLAYERS_FILE = os.path.join(DATA_DIR, 'nba_players.csv')
    GAMES_FILE = os.path.join(DATA_DIR, 'nba_games.csv')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    seasons = None
    incremental_update = False
    date_from = None
    
    if args.interactive:
        seasons = select_seasons_interactive()
    elif args.update:
        incremental_update = True
        latest_date = get_latest_game_date(PLAYERS_FILE)
        
        if latest_date:
            date_obj = datetime.strptime(latest_date, '%Y-%m-%d')
            date_from = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')
            
            if date_from > today:
                logger.info("\n" + "=" * 50)
                logger.info("Dataset is already up to date!")
                logger.info(f"Latest game: {latest_date}")
                logger.info("=" * 50)
                return
            
            print(f"\n{'='*50}")
            print("INCREMENTAL UPDATE MODE")
            print(f"{'='*50}")
            print(f"Latest game in dataset: {latest_date}")
            print(f"Fetching games from: {date_from}")
            print(f"{'='*50}\n")
        else:
            logger.info("No existing data found. Switching to current season fetch.")
            incremental_update = False
            seasons = [get_current_season()]
    elif args.full_scrape:
        seasons = SEASONS
        print(f"\nFull scrape: All {len(seasons)} NBA seasons will be fetched.")
    elif args.seasons:
        seasons = args.seasons
    elif args.all_seasons:
        seasons = SEASONS[-10:]
        print(f"\nFetching last 10 seasons: {', '.join(seasons)}")
    elif args.current_season:
        seasons = [get_current_season()]
        print(f"\nFetching current season: {seasons[0]}")
    else:
        is_empty, empty_seasons = check_empty_data()
        if is_empty:
            if empty_seasons is None:
                seasons = select_seasons_interactive()
            else:
                seasons = empty_seasons
        else:
            seasons = [get_current_season()]
    
    if seasons:
        logger.info(f"Seasons to update: {seasons}")
        logger.info(f"Total: {len(seasons)} season(s)")
    
    if args.force:
        existing_players = None
        existing_teams = None
        logger.info("Force mode: Ignoring existing data")
    else:
        existing_players = load_existing_data(PLAYERS_FILE)
        existing_teams = load_existing_data(GAMES_FILE)
    
    if incremental_update and date_from:
        existing_players, existing_teams, new_players, new_teams = update_since_date(
            date_from, existing_players, existing_teams
        )
        
        if existing_players is not None and existing_teams is not None:
            save_data(existing_players, existing_teams)
            logger.info("\n" + "=" * 50)
            logger.info("Incremental update complete!")
            logger.info(f"New player records added: {new_players}")
            logger.info(f"New team records added: {new_teams}")
            logger.info(f"Total player records: {len(existing_players)}")
            logger.info(f"Total team records: {len(existing_teams)}")
            logger.info("=" * 50)
        else:
            logger.info("No new games found. Dataset is up to date.")
    else:
        for i, season in enumerate(seasons, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing season: {season} ({i}/{len(seasons)})")
            logger.info(f"{'='*50}")
            
            existing_players, existing_teams = update_season(
                season, existing_players, existing_teams
            )
        
        if existing_players is not None and existing_teams is not None:
            save_data(existing_players, existing_teams)
            logger.info("\n" + "=" * 50)
            logger.info("Data update complete!")
            logger.info(f"Total player records: {len(existing_players)}")
            logger.info(f"Total team records: {len(existing_teams)}")
            logger.info("=" * 50)
        else:
            logger.error("No data was fetched. Please check your connection.")
            sys.exit(1)


if __name__ == "__main__":
    main()
