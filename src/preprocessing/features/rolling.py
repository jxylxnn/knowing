"""Rolling statistics feature group."""

from typing import List, Optional
import pandas as pd
import numpy as np

from src.preprocessing.features.base import FeatureGroup


class RollingFeatureGroup(FeatureGroup):
    """Generates rolling window statistics for player stats.
    
    Creates rolling mean, std, min, max for various window sizes
    to capture player performance trends.
    
    Attributes:
        windows: List of rolling window sizes.
        target_cols: Statistics to compute windows for.
        efficiency_cols: Efficiency-related columns.
    """
    
    def __init__(
        self,
        windows: Optional[List[int]] = None,
        target_cols: Optional[List[str]] = None,
        efficiency_cols: Optional[List[str]] = None
    ):
        """Initialize rolling feature group.
        
        Args:
            windows: Rolling window sizes (default: [3, 5, 10, 20, 50]).
            target_cols: Target stat columns.
            efficiency_cols: Efficiency-related columns.
        """
        self.windows = windows or [3, 5, 10, 20, 50]
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
        self.efficiency_cols = efficiency_cols or [
            'FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'FG3A', 'TOV', 'MIN'
        ]
    
    @property
    def name(self) -> str:
        return "rolling_features"
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            df: Input DataFrame with player stats.
            
        Returns:
            DataFrame with added rolling feature columns.
        """
        df = df.copy()
        stat_cols = self.target_cols + self.efficiency_cols
        valid_cols = [c for c in stat_cols if c in df.columns]
        
        shifted = df.groupby('PLAYER_ID')[valid_cols].shift(1)
        
        for window in self.windows:
            min_periods = max(1, window // 3)
            
            core_agg_funcs = ['mean']
            if window >= 5:
                core_agg_funcs.append('std')
            if window >= 10:
                core_agg_funcs.extend(['min', 'max'])
            
            try:
                rolled = shifted.groupby(df['PLAYER_ID']).rolling(
                    window=window, min_periods=min_periods
                )
                
                window_stats = rolled.agg(core_agg_funcs)
                
                new_columns = []
                for col, func in window_stats.columns:
                    new_columns.append(f"ROLL_{col}_{func.upper()}_{window}")
                window_stats.columns = new_columns
                
                window_stats = window_stats.reset_index(level=0, drop=True)
                df = pd.concat([df, window_stats], axis=1)
                
            except Exception:
                continue
        
        for window in [10, 20]:
            base_col = f'ROLL_PTS_MIN_{window}'
            if base_col in df.columns:
                for col in valid_cols:
                    min_c = f'ROLL_{col}_MIN_{window}'
                    max_c = f'ROLL_{col}_MAX_{window}'
                    if min_c in df.columns and max_c in df.columns:
                        df[f'ROLL_{col}_RANGE_{window}'] = df[max_c] - df[min_c]
        
        return df


class EfficiencyFeatureGroup(FeatureGroup):
    """Generates efficiency metrics like TS%, eFG%, etc."""
    
    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [5, 10, 20]
    
    @property
    def name(self) -> str:
        return "efficiency_features"
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        new_cols = {}
        
        for window in self.windows:
            fg_sum = df.groupby('PLAYER_ID')['FGA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            ft_sum = df.groupby('PLAYER_ID')['FTA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            tov_sum = df.groupby('PLAYER_ID')['TOV'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            pts_sum = df.groupby('PLAYER_ID')['PTS'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            fgm_sum = df.groupby('PLAYER_ID')['FGM'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            fg3m_sum = df.groupby('PLAYER_ID')['FG3M'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            fg3a_sum = df.groupby('PLAYER_ID')['FG3A'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            ast_sum = df.groupby('PLAYER_ID')['AST'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)
            mins_sum = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(1)
            
            new_cols[f'ROLL_TS_PCT_{window}'] = pts_sum / (2 * (fg_sum + 0.44 * ft_sum + 1e-6))
            new_cols[f'ROLL_EFG_PCT_{window}'] = (fgm_sum + 0.5 * fg3m_sum) / (fg_sum + 1e-6)
            new_cols[f'ROLL_3PT_PCT_{window}'] = fg3m_sum / (fg3a_sum + 1e-6)
            new_cols[f'ROLL_AST_TOV_{window}'] = ast_sum / (tov_sum + 1e-6)
            
            for stat in ['PTS', 'REB', 'AST']:
                stat_sum = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).sum()
                ).fillna(0)
                new_cols[f'ROLL_{stat}_PER_MIN_{window}'] = stat_sum / (mins_sum + 1e-6)
        
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


class MomentumFeatureGroup(FeatureGroup):
    """Generates momentum and trend features using EWMA."""
    
    def __init__(self, target_cols: Optional[List[str]] = None):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
    
    @property
    def name(self) -> str:
        return "momentum_features"
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        new_cols = {}
        
        for stat in self.target_cols:
            for span in [3, 5, 10, 20]:
                new_cols[f'{stat}_EWMA_{span}'] = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
                )
            
            season_avg = df.groupby('PLAYER_ID')[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            new_cols[f'{stat}_SEASON_AVG'] = season_avg
            
            for short, long in [(3, 10), (5, 20)]:
                short_avg = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).rolling(short, min_periods=1).mean()
                )
                long_avg = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).rolling(long, min_periods=long//3).mean()
                )
                new_cols[f'{stat}_TREND_{short}_{long}'] = short_avg - long_avg
        
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


class ContextualFeatureGroup(FeatureGroup):
    """Generates contextual features (home/away, rest, fatigue)."""
    
    @property
    def name(self) -> str:
        return "contextual_features"
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        new_cols = {}
        
        if 'MATCHUP' in df.columns:
            new_cols['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        
        new_cols['DAYS_SINCE_LAST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(4)
        new_cols['REST_DAYS'] = new_cols['DAYS_SINCE_LAST'].clip(0, 7)
        new_cols['IS_B2B'] = (new_cols['DAYS_SINCE_LAST'] == 1).astype(int)
        
        mins_lag = df.groupby('PLAYER_ID')['MIN'].shift(1).fillna(0)
        new_cols['MINS_LAST_3'] = mins_lag.groupby(df['PLAYER_ID']).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
        new_cols['MINS_LAST_7'] = mins_lag.groupby(df['PLAYER_ID']).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
        
        new_cols['FATIGUE_SCORE'] = (
            (new_cols['MINS_LAST_3'] / 100) * 0.4 +
            (new_cols['IS_B2B'] * 0.3) +
            ((4 - new_cols['REST_DAYS'].clip(0, 4)) * 0.3)
        )
        
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)