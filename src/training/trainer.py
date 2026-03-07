"""Base trainer class with unified interface for all model types."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a training run."""
    model: Any
    metrics: Dict[str, float]
    training_time: float
    best_iteration: Optional[int] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"TrainResult(metrics={{{metrics_str}}}, time={self.training_time:.1f}s)"


class BaseTrainer(ABC):
    """Abstract base class for all model trainers.
    
    Provides a unified interface for training, prediction, and model persistence.
    All concrete trainers must implement the abstract methods.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        use_gpu: bool = False,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """Initialize the trainer.
        
        Args:
            model_name: Name identifier for this model
            config: Model-specific configuration dictionary
            use_gpu: Whether to use GPU acceleration
            device: Specific device to use (e.g., 'cuda:0')
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.config = config
        self.use_gpu = use_gpu
        self.device = device or ('cuda' if use_gpu else 'cpu')
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.is_trained: bool = False
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {self.__class__.__name__} for '{model_name}' "
                   f"(GPU={use_gpu}, device={self.device})")
    
    @abstractmethod
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> TrainResult:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights for training (optional)
            **kwargs: Additional model-specific arguments
            
        Returns:
            TrainResult with model, metrics, and training info
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict on
            **kwargs: Additional model-specific arguments
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'BaseTrainer':
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional loading arguments
            
        Returns:
            Loaded trainer instance
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not supported by this model type.
        """
        return None
    
    def validate_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        check_finite: bool = True,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[np.ndarray]]:
        """Validate and clean input data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            check_finite: Whether to check for infinite values
            
        Returns:
            Tuple of (cleaned X, cleaned y)
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_clean = X.values
        else:
            X_clean = X.copy()
        
        # Check for NaN/Inf
        if check_finite:
            if np.any(np.isnan(X_clean)):
                logger.warning("NaN values found in features, filling with 0")
                X_clean = np.nan_to_num(X_clean, nan=0.0)
            if np.any(np.isinf(X_clean)):
                logger.warning("Infinite values found in features, replacing with large values")
                X_clean = np.nan_to_num(X_clean, posinf=1e10, neginf=-1e10)
        
        # Clean y if provided
        y_clean = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_clean = y.values
            else:
                y_clean = y.copy()
            
            if check_finite:
                y_clean = np.nan_to_num(y_clean, nan=0.0)
                y_clean = np.nan_to_num(y_clean, posinf=1e10, neginf=-1e10)
        
        return X_clean, y_clean
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names to values
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (handle zeros)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
        }