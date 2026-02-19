"""Shared prediction and training utilities.

This module contains utility classes extracted from ModelManager and DataPipeline
to reduce code duplication and improve maintainability.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalWeightCalculator:
    """Calculate temporal decay weights for training samples.
    
    Games in the last 30 days get weight 1.0.
    Games 6 months ago get weight ~0.2.
    This teaches models to focus on recent performance.
    
    Attributes:
        lambda_decay: Decay rate for exponential weighting.
        min_weight: Minimum weight to avoid ignoring old data entirely.
    """
    
    def __init__(self, lambda_decay: float = 0.023, min_weight: float = 0.1):
        """Initialize the weight calculator.
        
        Args:
            lambda_decay: Exponential decay rate (default ~0.5 weight after 30 days).
            min_weight: Minimum weight floor for old games.
        """
        self.lambda_decay = lambda_decay
        self.min_weight = min_weight
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate sample weights based on recency.
        
        Args:
            df: DataFrame with 'GAME_DATE' column.
            
        Returns:
            Array of weights for each sample.
        """
        if 'GAME_DATE' not in df.columns:
            return np.ones(len(df))
        
        max_date = df['GAME_DATE'].max()
        days_ago = (max_date - df['GAME_DATE']).dt.days
        
        weights = np.exp(-self.lambda_decay * days_ago)
        weights = np.clip(weights, self.min_weight, 1.0)
        
        return weights


class FallbackPredictor:
    """Provides fallback predictions when models are unavailable.
    
    Uses historical averages from rolling statistics when available,
    falling back to league averages when no history exists.
    
    Attributes:
        league_averages: Default values for each stat when no history is available.
    """
    
    LEAGUE_AVERAGES: Dict[str, float] = {
        'PTS': 10.0,
        'REB': 4.5,
        'AST': 2.5,
        'STL': 0.8,
        'BLK': 0.6,
        'TOV': 1.5
    }
    
    FALLBACK_COLUMNS: Dict[str, List[str]] = {
        'PTS': ['ROLL_PTS_AVG_10', 'ROLL_PTS_AVG_20', 'PTS_EWMA_5', 'PTS'],
        'REB': ['ROLL_REB_AVG_10', 'ROLL_REB_AVG_20', 'REB_EWMA_5', 'REB'],
        'AST': ['ROLL_AST_AVG_10', 'ROLL_AST_AVG_20', 'AST_EWMA_5', 'AST'],
        'STL': ['ROLL_STL_AVG_10', 'ROLL_STL_AVG_20', 'STL_EWMA_5', 'STL'],
        'BLK': ['ROLL_BLK_AVG_10', 'ROLL_BLK_AVG_20', 'BLK_EWMA_5', 'BLK'],
        'TOV': ['ROLL_TOV_AVG_10', 'ROLL_TOV_AVG_20', 'TOV_EWMA_5', 'TOV']
    }
    
    def __init__(self, league_averages: Optional[Dict[str, float]] = None):
        """Initialize with optional custom league averages.
        
        Args:
            league_averages: Custom default values. Uses defaults if None.
        """
        self.league_averages = league_averages or self.LEAGUE_AVERAGES.copy()
    
    def predict(self, df: pd.DataFrame, targets: Optional[List[str]] = None) -> Dict[str, float]:
        """Generate fallback predictions for all targets.
        
        Args:
            df: DataFrame with player context (single row).
            targets: List of targets to predict. Uses all if None.
            
        Returns:
            Dictionary mapping target names to predicted values.
        """
        targets = targets or list(self.league_averages.keys())
        predictions = {}
        
        for target in targets:
            predictions[target] = self._get_fallback_value(df, target)
        
        return predictions
    
    def predict_batch(
        self, 
        df: pd.DataFrame, 
        targets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate fallback predictions for multiple players.
        
        Args:
            df: DataFrame with player contexts (multiple rows).
            targets: List of targets to predict. Uses all if None.
            
        Returns:
            DataFrame with predictions for each target.
        """
        targets = targets or list(self.league_averages.keys())
        predictions = {}
        
        for target in targets:
            predictions[target] = self._get_fallback_values_batch(df, target)
        
        return pd.DataFrame(predictions, index=df.index)
    
    def _get_fallback_value(self, df: pd.DataFrame, target: str) -> float:
        """Get fallback value for a single target from DataFrame.
        
        Args:
            df: Single-row DataFrame with player context.
            target: Target stat name.
            
        Returns:
            Fallback prediction value.
        """
        fallback_cols = self.FALLBACK_COLUMNS.get(target, [f'ROLL_{target}_AVG_10', target])
        
        for col in fallback_cols:
            if col in df.columns:
                val = df[col].iloc[0] if len(df) > 0 else None
                if pd.notna(val) and val > 0:
                    return float(val)
        
        return self.league_averages.get(target, 0.0)
    
    def _get_fallback_values_batch(self, df: pd.DataFrame, target: str) -> np.ndarray:
        """Get fallback values for a target across all rows.
        
        Args:
            df: Multi-row DataFrame with player contexts.
            target: Target stat name.
            
        Returns:
            Array of fallback prediction values.
        """
        fallback_cols = self.FALLBACK_COLUMNS.get(target, [f'ROLL_{target}_AVG_10', target])
        league_avg = self.league_averages.get(target, 0.0)
        
        values = np.full(len(df), league_avg)
        
        for col in fallback_cols:
            if col in df.columns:
                mask = (values == league_avg) & df[col].notna() & (df[col] > 0)
                values[mask] = df.loc[mask, col].values
        
        return values


class FeatureSelector:
    """Selects safe features for model training, avoiding data leakage.
    
    Identifies and filters features that could leak target information,
    keeping only lagged, rolling, and contextual features.
    """
    
    SAFE_PREFIXES = ['ROLL', 'EWMA_', 'VS_OPP_']
    SAFE_KEYWORDS = ['TREND', 'BAYESIAN', 'PROJ_', 'PACE', '_TE', '_SHARE_', 'ROLE_INDEX']
    SAFE_COLUMNS = [
        'IS_HOME', 'REST_DAYS', 'IS_B2B', 'FATIGUE_SCORE', 'MONTH', 'DAY_OF_WEEK',
        'EXP_PACE', 'EXP_TEAM_PTS', 'EXP_GAME_TOTAL', 'BLOWOUT_RISK', 'CLOSE_GAME', 'EXP_MARGIN'
    ]
    
    def __init__(self, targets: Optional[List[str]] = None):
        """Initialize with target columns to exclude.
        
        Args:
            targets: List of target column names.
        """
        self.targets = targets or ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select safe features from DataFrame.
        
        Args:
            df: DataFrame with all columns.
            
        Returns:
            List of safe feature column names.
        """
        exclude_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'OPPONENT_ID', 'OPPONENT_ABBR',
            'WL', 'SEASON_ID', 'VIDEO_AVAILABLE'
        ] + self.targets + self.targets
        
        candidate_cols = [
            c for c in df.columns
            if c not in exclude_cols and
            (df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'float', 'int'])
        ]
        
        leaky_suspects = [
            c for c in candidate_cols 
            if any(t.lower() in c.lower() for t in self.targets)
        ]
        if leaky_suspects:
            logger.warning(f"Potential leaky features detected: {leaky_suspects[:10]}...")
        
        safe_features = [
            c for c in candidate_cols
            if self._is_safe_feature(c)
        ]
        
        safe_features = [
            c for c in safe_features
            if not (c.endswith('_TEAM') and c.replace('_TEAM', '') in self.targets)
        ]
        
        logger.info(f"Selected {len(safe_features)} features for training.")
        return safe_features
    
    def _is_safe_feature(self, col: str) -> bool:
        """Check if a column is a safe feature.
        
        Args:
            col: Column name to check.
            
        Returns:
            True if the feature is safe to use.
        """
        if col in self.SAFE_COLUMNS:
            return True
        
        for prefix in self.SAFE_PREFIXES:
            if col.startswith(prefix):
                return True
        
        for keyword in self.SAFE_KEYWORDS:
            if keyword in col:
                return True
        
        return False


class TargetPreprocessor:
    """Preprocesses target variables to handle outliers.
    
    Caps target values at player-specific percentiles to prevent
    extreme values from skewing model training.
    """
    
    def __init__(self, percentile: float = 0.99, targets: Optional[List[str]] = None):
        """Initialize with capping parameters.
        
        Args:
            percentile: Percentile to use for capping (e.g., 0.99 for 99th).
            targets: List of target column names.
        """
        self.percentile = percentile
        self.targets = targets or ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap target values at player-specific percentiles.
        
        Args:
            df: DataFrame with target columns.
            
        Returns:
            DataFrame with capped targets (adds {TARGET}_CLEAN columns).
        """
        df = df.copy()
        
        for stat in self.targets:
            if stat not in df.columns:
                continue
            
            player_caps = df.groupby('PLAYER_ID')[stat].quantile(self.percentile)
            player_caps_mapped = df['PLAYER_ID'].map(player_caps)
            
            global_cap = df[stat].quantile(self.percentile)
            player_caps_mapped = player_caps_mapped.fillna(global_cap)
            
            df[f'{stat}_CLEAN'] = df[stat].clip(upper=player_caps_mapped)
        
        return df