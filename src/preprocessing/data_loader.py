import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic cleaning of NBA datasets."""
    
    def __init__(self, players_path: str, games_path: str):
        self.players_path = players_path
        self.games_path = games_path
        self.players_df = None
        self.games_df = None

    @staticmethod
    def _parse_minutes_vectorized(series: pd.Series) -> pd.Series:
        """Vectorized minutes parsing — avoids row-by-row .apply()."""
        result = pd.to_numeric(series, errors='coerce')

        str_mask = result.isna() & series.notna()
        if not str_mask.any():
            return result

        str_vals = series[str_mask].astype(str).str.strip()
        colon_mask = str_vals.str.contains(':', na=False)
        if colon_mask.any():
            parts = str_vals[colon_mask].str.split(':', expand=True)
            mins = pd.to_numeric(parts[0], errors='coerce').fillna(0)
            secs = pd.to_numeric(parts[1], errors='coerce').fillna(0)
            result.loc[colon_mask[colon_mask].index] = mins + secs / 60.0

        remaining = str_mask & result.isna()
        if remaining.any():
            result.loc[remaining] = pd.to_numeric(series[remaining], errors='coerce')

        return result

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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data from {self.players_path} and {self.games_path}")
        self.players_df = pd.read_csv(self.players_path)
        self.games_df = pd.read_csv(self.games_path)
        
        num_cols = ['PTS','REB','AST','STL','BLK','FGA','FGM','FTA','FTM','FG3A','FG3M','TOV']
        for c in num_cols:
            if c in self.players_df.columns: self.players_df[c] = pd.to_numeric(self.players_df[c], errors='coerce')

        if 'MIN' in self.players_df.columns:
            self.players_df['MIN'] = self._parse_minutes_vectorized(self.players_df['MIN']).clip(0, 60)

        self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
        self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
        self.players_df = self.players_df.dropna(subset=['PTS', 'REB', 'AST', 'MIN'])
        self.players_df = self.players_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
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
        
        def_stats = ['OPP_PTS_ALLOWED', 'OPP_REB_ALLOWED', 'OPP_AST_ALLOWED']
        shifted_def = merged_games.groupby('TEAM_ID')[def_stats].shift(1)
        rolled_def = shifted_def.groupby(
            merged_games['TEAM_ID']
        ).rolling(10, min_periods=3).mean().reset_index(level=0, drop=True)
        for stat in def_stats:
            merged_games[f'TEAM_DEF_{stat}_ROLL_10'] = rolled_def[stat]
            
        # Merge Defensive Stats back to main Games DF
        # We only need the team's defensive stats
        team_def = merged_games[['GAME_ID', 'TEAM_ID', 'TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10', 
                                  'TEAM_DEF_OPP_REB_ALLOWED_ROLL_10', 'TEAM_DEF_OPP_AST_ALLOWED_ROLL_10']]
        
        # Remove duplicates from self-merge
        team_def = team_def.drop_duplicates()
        
        self.games_df = pd.merge(self.games_df, team_def, on=['GAME_ID', 'TEAM_ID'], how='left')
        
        # 2. Merge with Player Stats
        merged_df = pd.merge(
            self.players_df,
            self.games_df[['GAME_ID', 'TEAM_ID', 'WL', 'PTS', 'REB', 'AST', 'FGA', 'FTA', 'OREB', 'DREB', 'TOV',
                           'TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10', 'TEAM_DEF_OPP_REB_ALLOWED_ROLL_10', 'TEAM_DEF_OPP_AST_ALLOWED_ROLL_10']],
            on=['GAME_ID', 'TEAM_ID'],
            how='left',
            suffixes=('', '_TEAM')
        )
        
        # 3. Add Opponent Identity (Using the pre-merged_games table is efficient)
        # We need the OPPONENT_ID for the player row
        opp_map = merged_games[['GAME_ID', 'TEAM_ID', 'TEAM_ID_OPP']].drop_duplicates()
        opp_map.rename(columns={'TEAM_ID_OPP': 'OPPONENT_ID'}, inplace=True)
        merged_df = pd.merge(merged_df, opp_map, on=['GAME_ID', 'TEAM_ID'], how='left')
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
