"""Data pipeline for loading and preprocessing NBA data."""

import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from src.config import DataConfig, TrainingConfig
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils.prediction_utils import FeatureSelector, FeatureSchema

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles all data loading and preprocessing operations.
    
    This class encapsulates the data loading, feature engineering,
    and preprocessing logic that was previously in ModelManager.
    """
    
    def __init__(
        self, 
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        """Initialize data pipeline.
        
        Args:
            data_config: Configuration for data paths and storage
            training_config: Configuration for training parameters
        """
        self.data_config = data_config
        self.training_config = training_config
        self.feature_engineer = FeatureEngineer()
        self._feature_cols: Optional[List[str]] = None
        self.feature_schema: Optional[FeatureSchema] = None
        self.feature_selector = FeatureSelector(self.training_config.targets)
        
    @property
    def feature_cols(self) -> Optional[List[str]]:
        """Get the list of feature columns."""
        return self._feature_cols
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from files.
        
        Returns:
            Merged DataFrame with player and game data
            
        Raises:
            ValueError: If data files don't exist or are invalid
        """
        # Validate data directory
        if not self.data_config.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_config.data_dir}")
        
        # Validate required files
        players_file = self.data_config.data_dir / 'nba_players.csv'
        games_file = self.data_config.data_dir / 'nba_games.csv'
        
        if not players_file.exists():
            raise ValueError(f"Players file not found: {players_file}")
        if not games_file.exists():
            raise ValueError(f"Games file not found: {games_file}")
        
        loader = DataLoader(str(players_file), str(games_file))
        merged_df = loader.merge_datasets()
        
        # Validate merged data
        if merged_df.empty:
            raise ValueError("Merged dataset is empty after loading")
        
        if len(merged_df) < 1000:
            raise ValueError(f"Dataset too small: {len(merged_df)} rows (minimum 1000 required)")
        
        logger.info(f"Loaded raw data: {len(merged_df)} rows")
        return merged_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to the data.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        full_df = self.feature_engineer.create_features(df)
        
        if full_df.empty:
            raise ValueError("Feature engineering resulted in empty dataset")
        
        # Validate required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE'] + self.training_config.targets
        missing_cols = [c for c in required_cols if c not in full_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after feature engineering: {missing_cols}")
        
        logger.info(f"Engineered features: {len(full_df.columns)} columns")
        return full_df
    
    def split_train_test(
        self, 
        df: pd.DataFrame,
        split_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets by date.
        
        Args:
            df: DataFrame with engineered features
            split_date: Date to split on (uses config if not provided)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_date = split_date or self.training_config.test_split_date
        split_dt = pd.to_datetime(split_date)
        
        train_df = df[df['GAME_DATE'] < split_dt].copy()
        test_df = df[df['GAME_DATE'] >= split_dt].copy()
        
        # Validate splits
        if train_df.empty:
            raise ValueError("Training set is empty after split")
        if test_df.empty:
            raise ValueError("Test set is empty after split")
        
        logger.info(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
        return train_df, test_df
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Full data preparation pipeline.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        raw_df = self.load_raw_data()
        featured_df = self.engineer_features(raw_df)
        return self.split_train_test(featured_df)
    
    def calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate temporal decay sample weights.
        
        Games in the last 30 days get weight 1.0.
        Games 6 months ago get weight 0.2.
        
        Args:
            df: DataFrame with GAME_DATE column
            
        Returns:
            Array of sample weights
        """
        if 'GAME_DATE' not in df.columns:
            return np.ones(len(df))
        
        max_date = df['GAME_DATE'].max()
        days_ago = (max_date - df['GAME_DATE']).dt.days
        
        # Exponential decay: weight = exp(-lambda * days)
        lambda_decay = self.training_config.temporal_decay_lambda
        weights = np.exp(-lambda_decay * days_ago)
        
        # Clip minimum weight to avoid ignoring old data entirely
        weights = np.clip(weights, 0.1, 1.0)
        
        return weights
    
    def preprocess_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap targets to handle outliers.
        
        Prevents extreme values from skewing the loss landscape.
        Uses percentile caps per player.
        
        Args:
            df: DataFrame with target columns
            
        Returns:
            DataFrame with capped targets
        """
        df = df.copy()
        percentile = self.training_config.outlier_percentile
        
        for stat in self.training_config.targets:
            if stat in df.columns:
                # Calculate the percentile for each player
                caps = df.groupby('PLAYER_ID')[stat].quantile(percentile)
                
                # Map caps back to dataframe
                player_caps = df['PLAYER_ID'].map(caps)
                
                # Fill NaN caps with global percentile
                global_cap = df[stat].quantile(percentile)
                player_caps = player_caps.fillna(global_cap)
                
                # Cap the stat
                df[f'{stat}_CLEAN'] = df[stat].clip(upper=player_caps)
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features for training, avoiding leakage.
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            List of safe feature column names
        """
        self.feature_schema = self.feature_selector.fit(df, group_columns=self.feature_engineer.get_group_columns())
        self._feature_cols = self.feature_schema.feature_cols
        logger.info(f"Selected {len(self._feature_cols)} features for training")
        return self._feature_cols
    
    def get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of categorical columns for models that support them.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of categorical column names
        """
        cat_cols = ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID']
        return [c for c in cat_cols if c in df.columns]
    
    def clean_data_for_training(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        targets: List[str]
    ) -> pd.DataFrame:
        """Clean data for model training.
        
        Args:
            df: DataFrame to clean
            feature_cols: List of feature columns
            targets: List of target columns
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Fill NaN in features
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Clean targets
        for target in targets:
            t_col = f'{target}_CLEAN' if f'{target}_CLEAN' in df.columns else target
            df[target] = pd.to_numeric(df[t_col], errors='coerce').fillna(0)
        
        return df
