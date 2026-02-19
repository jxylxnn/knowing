"""Feature groups for modular feature engineering.

This module provides a base class for feature groups and individual
implementations for different feature categories.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class FeatureGroup(ABC):
    """Base class for feature groups.
    
    Each feature group creates a specific category of features
    (e.g., rolling averages, efficiency metrics, contextual features).
    
    Attributes:
        name: Human-readable name of the feature group.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the feature group name."""
        pass
    
    @property
    def depends_on(self) -> List[str]:
        """Return list of other feature groups this depends on.
        
        Returns:
            Empty list by default. Override if dependencies exist.
        """
        return []
    
    @abstractmethod
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for this group.
        
        Args:
            df: Input DataFrame with raw data.
            
        Returns:
            DataFrame with added feature columns.
        """
        pass
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get names of features created by this group.
        
        Override this method if feature names are not obvious.
        
        Args:
            df: DataFrame with features already created.
            
        Returns:
            List of feature column names.
        """
        return []