"""Abstract base classes and protocols for NBA prediction models."""

from abc import abstractmethod
from typing import Protocol, Optional, Dict, Any, List, Tuple, runtime_checkable
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseModel(Protocol):
    """Protocol defining interface for all prediction models.
    
    All models must implement these methods to be compatible with
    the training pipeline and prediction service.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "BaseModel":
        """Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
            
        Returns:
            Trained model instance (self)
        """
        ...
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on the provided data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        ...
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk.
        
        Args:
            path: Path to save the model
        """
        ...
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Deserialize model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        ...
    
    @property
    @abstractmethod
    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance if available.
        
        Returns:
            Series mapping feature names to importance scores,
            or None if not supported by the model
        """
        ...
class ModelMetadata:
    """Metadata for a trained model.
    
    Tracks model version, training info, and performance metrics.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str,
        version: str = "1.0.0",
        created_at: Optional[str] = None,
        training_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        feature_columns: Optional[List[str]] = None,
        checksum: Optional[str] = None
    ):
        """Initialize model metadata.
        
        Args:
            name: Model name
            model_type: Type of model (e.g., 'catboost', 'lstm')
            version: Semantic version string
            created_at: ISO format timestamp
            training_params: Hyperparameters used for training
            metrics: Performance metrics
            feature_columns: List of feature column names
            checksum: MD5 checksum of model file
        """
        from datetime import datetime
        
        self.name = name
        self.model_type = model_type
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.training_params = training_params or {}
        self.metrics = metrics or {}
        self.feature_columns = feature_columns or []
        self.checksum = checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'version': self.version,
            'created_at': self.created_at,
            'training_params': self.training_params,
            'metrics': self.metrics,
            'feature_columns': self.feature_columns,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


class PredictionResult:
    """Result of a prediction with uncertainty quantification.
    
    Provides both point estimates and uncertainty bounds.
    """
    
    def __init__(
        self,
        player_id: int,
        player_name: str,
        team: str,
        opponent: str,
        predictions: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
        model_contributions: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """Initialize prediction result.
        
        Args:
            player_id: Player ID
            player_name: Player name
            team: Team abbreviation
            opponent: Opponent team abbreviation
            predictions: Dict mapping stat names to predicted values
            uncertainties: Dict mapping stat names to std dev
            confidence_intervals: Dict mapping stat names to (lower, upper) bounds
            model_contributions: Dict of model_name -> stat -> contribution
        """
        self.player_id = player_id
        self.player_name = player_name
        self.team = team
        self.opponent = opponent
        self.predictions = predictions
        self.uncertainties = uncertainties or {}
        self.confidence_intervals = confidence_intervals or {}
        self.model_contributions = model_contributions or {}
    
    def get_prediction(self, stat: str) -> float:
        """Get prediction for a specific stat."""
        return self.predictions.get(stat, 0.0)
    
    def get_uncertainty(self, stat: str) -> float:
        """Get uncertainty for a specific stat."""
        return self.uncertainties.get(stat, 0.0)
    
    def get_confidence_interval(self, stat: str, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for a specific stat.
        
        Args:
            stat: Stat name
            confidence: Confidence level (default 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if stat in self.confidence_intervals:
            return self.confidence_intervals[stat]
        
        # Calculate from uncertainty if available
        if stat in self.uncertainties:
            pred = self.predictions[stat]
            std = self.uncertainties[stat]
            # Approximate 95% CI using 1.96 * std
            z_score = 1.96 if confidence == 0.95 else 2.58 if confidence == 0.99 else 1.645
            margin = z_score * std
            return (pred - margin, pred + margin)
        
        # Return point estimate as both bounds
        pred = self.predictions.get(stat, 0.0)
        return (pred, pred)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'team': self.team,
            'opponent': self.opponent,
            'predictions': self.predictions,
            'uncertainties': self.uncertainties,
            'confidence_intervals': self.confidence_intervals,
            'model_contributions': self.model_contributions
        }


class ModelRegistry:
    """Centralized registry for model storage and versioning.
    
    Provides save/load functionality with metadata tracking.
    """
    
    def __init__(self, models_dir: Path):
        """Initialize registry.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = str(Path(models_dir))
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, BaseModel] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
    
    def register(
        self, 
        name: str, 
        model: BaseModel, 
        metadata: ModelMetadata
    ) -> None:
        """Register a trained model.
        
        Args:
            name: Model identifier
            model: Trained model instance
            metadata: Model metadata
        """
        self._models[name] = model
        self._metadata[name] = metadata
    
    def get(self, name: str) -> BaseModel:
        """Get a registered model.
        
        Args:
            name: Model identifier
            
        Returns:
            Model instance
            
        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]
    
    def get_metadata(self, name: str) -> ModelMetadata:
        """Get model metadata.
        
        Args:
            name: Model identifier
            
        Returns:
            Model metadata
        """
        if name not in self._metadata:
            raise KeyError(f"Metadata for model '{name}' not found")
        return self._metadata[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def save(self, name: str, model: BaseModel, metadata: ModelMetadata) -> Path:
        """Save model to disk.
        
        Args:
            name: Model identifier
            model: Model to save
            metadata: Model metadata
            
        Returns:
            Path to saved model
        """
        model_path = self._models_dir / f"{name}.pkl"
        metadata_path = self._models_dir / f"{name}_metadata.json"
        
        # Save model
        model.save(model_path)
        
        # Save metadata
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Register in memory
        self.register(name, model, metadata)
        
        return model_path
    
    def load(self, name: str, model_class: type) -> BaseModel:
        """Load model from disk.
        
        Args:
            name: Model identifier
            model_class: Class to use for loading
            
        Returns:
            Loaded model instance
        """
        model_path = self._models_dir / f"{name}.pkl"
        metadata_path = self._models_dir / f"{name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = model_class.load(model_path)
        
        # Load metadata if available
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
        else:
            metadata = ModelMetadata(name=name, model_type=model_class.__name__)
        
        # Register
        self.register(name, model, metadata)
        
        return model


# ---------------------------------------------------------------------------
# Shared model-loading utilities used by ModelManager and TrainingPipeline
# ---------------------------------------------------------------------------


def validate_blend_contract(
    blend_weights: Dict[str, Any],
    transformer_model: Any,
    models_dir,
) -> None:
    """Raise when blend weights require a model that is not loaded.

    Blend weights are computed during training under the assumption that all
    models referenced by non-zero weights will contribute at runtime.  If a
    model is missing, the remaining predictions are scaled incorrectly,
    producing silently uncalibrated output.
    """
    if not blend_weights:
        return

    has_transformer_weight = any(
        float(weights.get("transformer", 0.0)) > 0.0
        for key, weights in blend_weights.items()
        if key != "_method"
    )
    if has_transformer_weight and transformer_model is None:
        transformer_path = Path(models_dir) / "attention_transformer.pkl"
        if transformer_path.exists():
            raise RuntimeError(
                "Blend weights require a Transformer model but "
                f"attention_transformer.pkl in {models_dir} failed to load. "
                "Fix the artifact or retrain."
            )
        raise FileNotFoundError(
            "Blend weights require a Transformer model but "
            f"attention_transformer.pkl is missing from {models_dir}. "
            "Provide the artifact or retrain with the Transformer disabled."
        )


def collect_quantile_dict(trainer) -> Dict[str, Any]:
    """Build a {qualifier: model} dict from a trainer's quantile attributes."""
    result: Dict[str, Any] = {}
    low = getattr(trainer, "quantile_low_model", None)
    high = getattr(trainer, "quantile_high_model", None)
    if low is not None:
        result["low"] = low
    if high is not None:
        result["high"] = high
    return result


def load_transformer_from_disk(models_dir) -> Optional[Any]:
    """Load the Transformer model from attention_transformer.pkl if present.

    Returns None if the file is missing or loading fails.
    """
    from src.models.transformer_model import TransformerWrapper

    transformer_path = Path(models_dir) / "attention_transformer.pkl"
    if not transformer_path.exists():
        return None
    try:
        model = TransformerWrapper.load(str(transformer_path))
        logger.info("Loaded Transformer model")
        return model
    except Exception as exc:
        logger.warning("Failed to load Transformer model: %s", exc)
        return None


def load_blend_weights_from_disk(models_dir) -> Dict[str, Any]:
    """Load blend weights from blend_weights.pkl if present.

    Returns an empty dict when the file is missing or loading fails.
    """
    import joblib

    blend_path = Path(models_dir) / "blend_weights.pkl"
    if not blend_path.exists():
        return {}
    try:
        weights = joblib.load(blend_path)
        logger.info("Loaded blend weights for %s targets", len(weights))
        return weights
    except Exception as exc:
        logger.warning("Failed to load blend weights: %s", exc)
        return {}
