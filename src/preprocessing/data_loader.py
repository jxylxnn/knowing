import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic cleaning of NBA datasets."""
    
    def __init__(self, players_path: str, games_path: str):
        self.players_path = players_path
        self.games_path = games_path
        self.players_df = None
        self.games_df = None

    def _ensure_columns(self, df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
        """Ensure a DataFrame has all required columns with sensible defaults."""
        df = df.copy()
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        return df

    def _parse_minutes(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)): return float(val)
        s = str(val).strip()
        if s == '' or s.lower() in {'nan','none'}: return np.nan
        if ':' in s:
            parts = s.split(':')
            if len(parts) == 2: return float(parts[0]) + float(parts[1])/60.0
        try: return float(s)
        except ValueError: return np.nan

    def _add_team_rolling_features(self) -> None:
        """Compute safe team-level rolling context from prior games only."""
        if self.games_df is None or self.games_df.empty:
            return

        self.games_df = self.games_df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        for stat in ['PTS', 'REB', 'AST', 'FGA', 'FTA', 'FGM', 'OREB', 'DREB', 'TOV']:
            if stat not in self.games_df.columns:
                continue
            self.games_df[f'TEAM_{stat}_ROLL_10'] = self.games_df.groupby('TEAM_ID')[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            self.games_df[f'TEAM_{stat}_ROLL_5'] = self.games_df.groupby('TEAM_ID')[stat].transform(
                lambda x: x.shift(1).rolling(5, min_periods=2).mean()
            )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data from {self.players_path} and {self.games_path}")
        self.players_df = pd.read_csv(self.players_path)
        self.games_df = pd.read_csv(self.games_path)
        
        num_cols = ['PTS','REB','AST','STL','BLK','FGA','FGM','FTA','FTM','FG3A','FG3M','TOV']
        for c in num_cols:
            if c in self.players_df.columns: self.players_df[c] = pd.to_numeric(self.players_df[c], errors='coerce')

        if 'MIN' in self.players_df.columns:
            self.players_df['MIN'] = self.players_df['MIN'].apply(self._parse_minutes).clip(0, 60)

        self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
        self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
        self.players_df = self.players_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        self._add_team_rolling_features()

        missing_stats_cols = [c for c in ['PTS', 'REB', 'AST', 'MIN'] if c in self.players_df.columns]
        missing_players = self.players_df[missing_stats_cols].isna().sum().to_dict() if missing_stats_cols else {}
        if any(v > 0 for v in missing_players.values()):
            logger.warning("Player rows retain missing base stats: %s", missing_players)
        
        logger.info(f"Loaded {len(self.players_df)} player records and {len(self.games_df)} game records")
        return self.players_df, self.games_df

    def merge_datasets(self) -> pd.DataFrame:
        """Vectorized merge logic (Much Faster)."""
        if self.players_df is None or self.games_df is None: self.load_data()
            
        # 1. Prepare Game Stats for Self-Join
        # We need to join the game table on itself to get opponent stats
        games = self.games_df[['GAME_ID', 'TEAM_ID', 'PTS', 'REB', 'AST', 'FGA', 'FTA', 'FGM', 'OREB', 'DREB', 'TOV']].copy()
        
        # Self-merge to get opponent stats for each team
        # If Team A played Team B in Game 1, we get two rows. 
        # We merge Game 1 (Team A) with Game 1 (Team B) to get what B scored against A.
        merged_games = pd.merge(
            games, 
            games, 
            on='GAME_ID', 
            suffixes=('', '_OPP')
        )
        
        # Filter to ensure we aren't matching a team with itself (defensive check)
        merged_games = merged_games[merged_games['TEAM_ID'] != merged_games['TEAM_ID_OPP']]
        
        # Rename Opponent stats to Defensive Stats allowed by Team
        # Example: PTS_OPP is the points the opponent scored, which is the points Team_ID allowed
        merged_games.rename(columns={
            'PTS_OPP': 'OPP_PTS_ALLOWED', 'REB_OPP': 'OPP_REB_ALLOWED', 'AST_OPP': 'OPP_AST_ALLOWED',
            'FGA_OPP': 'OPP_FGA_ALLOWED', 'FGM_OPP': 'OPP_FGM_ALLOWED'
        }, inplace=True)
        
        # Calculate Rolling Defensive Ratings for the TEAM
        for stat in ['OPP_PTS_ALLOWED', 'OPP_REB_ALLOWED', 'OPP_AST_ALLOWED']:
            merged_games[f'TEAM_DEF_{stat}_ROLL_10'] = merged_games.groupby('TEAM_ID')[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            
        # Merge Defensive Stats back to main Games DF
        # We only need the team's defensive stats
        team_def = merged_games[['GAME_ID', 'TEAM_ID', 'TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10', 
                                  'TEAM_DEF_OPP_REB_ALLOWED_ROLL_10', 'TEAM_DEF_OPP_AST_ALLOWED_ROLL_10']]
        
        # Remove duplicates from self-merge
        team_def = team_def.drop_duplicates()

        self.games_df = pd.merge(self.games_df, team_def, on=['GAME_ID', 'TEAM_ID'], how='left')

        # Add safe opponent-team rolling context columns by merging on opponent identity.
        rolling_cols = [c for c in self.games_df.columns if c.startswith('TEAM_') and c.endswith(('_ROLL_10', '_ROLL_5'))]
        if rolling_cols:
            opp_roll = self.games_df[['GAME_ID', 'TEAM_ID'] + rolling_cols].copy()
            opp_roll = opp_roll.rename(columns={'TEAM_ID': 'OPPONENT_ID'})
            rename_map = {
                c: f'OPP_TEAM_{c.replace("TEAM_", "", 1).replace("DEF_OPP_", "DEF_")}'
                for c in rolling_cols
            }
            opp_roll = opp_roll.rename(columns=rename_map)
        else:
            opp_roll = pd.DataFrame(columns=['GAME_ID', 'OPPONENT_ID'])

        # 2. Merge with Player Stats
        game_merge_defaults = {
            'WL': '',
            'PTS': 0, 'REB': 0, 'AST': 0, 'FGA': 0, 'FTA': 0,
            'OREB': 0, 'DREB': 0, 'TOV': 0,
            'TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10': 0.0,
            'TEAM_DEF_OPP_REB_ALLOWED_ROLL_10': 0.0,
            'TEAM_DEF_OPP_AST_ALLOWED_ROLL_10': 0.0,
        }
        self.games_df = self._ensure_columns(self.games_df, game_merge_defaults)

        own_team_roll_cols = [c for c in self.games_df.columns if c.startswith('TEAM_') and c.endswith(('_ROLL_10', '_ROLL_5'))]
        cols_to_select = list(dict.fromkeys([
            'GAME_ID', 'TEAM_ID', 'WL', 'PTS', 'REB', 'AST', 'FGA', 'FTA', 'OREB', 'DREB', 'TOV',
            'TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10', 'TEAM_DEF_OPP_REB_ALLOWED_ROLL_10', 'TEAM_DEF_OPP_AST_ALLOWED_ROLL_10'
        ] + own_team_roll_cols))
        merged_df = pd.merge(
            self.players_df,
            self.games_df[cols_to_select],
            on=['GAME_ID', 'TEAM_ID'],
            how='left',
            suffixes=('', '_TEAM')
        )
        
        # 3. Add Opponent Identity (Using the pre-merged_games table is efficient)
        # We need the OPPONENT_ID for the player row
        opp_map = merged_games[['GAME_ID', 'TEAM_ID', 'TEAM_ID_OPP']].drop_duplicates()
        opp_map.rename(columns={'TEAM_ID_OPP': 'OPPONENT_ID_FROM_GAME'}, inplace=True)
        merged_df = pd.merge(merged_df, opp_map, on=['GAME_ID', 'TEAM_ID'], how='left')
        if 'OPPONENT_ID' in merged_df.columns:
            merged_df['OPPONENT_ID'] = merged_df['OPPONENT_ID'].fillna(merged_df['OPPONENT_ID_FROM_GAME'])
            merged_df = merged_df.drop(columns=['OPPONENT_ID_FROM_GAME'])
        else:
            merged_df = merged_df.rename(columns={'OPPONENT_ID_FROM_GAME': 'OPPONENT_ID'})

        if not opp_roll.empty:
            merged_df = pd.merge(
                merged_df,
                opp_roll,
                on=['GAME_ID', 'OPPONENT_ID'],
                how='left',
            )
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
