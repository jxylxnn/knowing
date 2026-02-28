"""Model loading and saving service.

Extracts model persistence logic from ModelManager into a dedicated service.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, TypeVar

import joblib

from src.config import Config

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ModelLoader:
    """Handles loading and saving of trained models.
    
    Centralizes model persistence logic including:
    - CatBoost models
    - Neural network wrappers
    - Blenders (meta-learners)
    - Feature columns
    
    Attributes:
        models_dir: Directory where models are stored.
        targets: List of target stat names.
    """
    
    MODEL_FILES = {
        'joint_model': ('joint_stats_nn.pkl', 'MultiOutputWrapper'),
        'temporal_model': ('temporal_lstm.pkl', 'LSTMWrapper'),
        'attention_model': ('attention_transformer.pkl', 'TransformerWrapper'),
        'adv_temporal_model': ('adv_temporal_attention.pkl', 'TemporalAttentionWrapper'),
        'gnn_model': ('team_chemistry_gnn.pkl', 'GNNWrapper'),
    }
    
    def __init__(
        self,
        models_dir: str = 'models',
        targets: Optional[List[str]] = None
    ):
        """Initialize the model loader.
        
        Args:
            models_dir: Directory containing saved models.
            targets: List of target stat names (e.g., ['PTS', 'REB', 'AST']).
        """
        self.models_dir = Path(models_dir)
        self.targets = targets or ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        
        self.models: Dict[str, Any] = {}
        self.catboost_mae_models: Dict[str, Any] = {}
        self.catboost_quantile_models: Dict[str, Dict[str, Any]] = {}
        self.blenders: Dict[str, Any] = {}
        self.joint_model: Optional[Any] = None
        self.temporal_model: Optional[Any] = None
        self.attention_model: Optional[Any] = None
        self.adv_temporal_model: Optional[Any] = None
        self.gnn_model: Optional[Any] = None
        self.feature_cols: Optional[List[str]] = None
    
    def load_all(self, use_gpu: bool = False) -> Dict[str, int]:
        """Load all available models from disk.
        
        Args:
            use_gpu: Whether to configure models for GPU usage.
            
        Returns:
            Dictionary with counts of loaded models by type.
        """
        counts = {
            'catboost': 0,
            'advanced': 0,
            'blenders': 0,
            'failed': 0
        }
        
        counts['catboost'] = self._load_catboost_models()
        counts['blenders'] = self._load_blenders()
        counts['advanced'] = self._load_advanced_models(use_gpu)
        counts['failed'] = len(self.targets) - counts['catboost']
        
        self._load_feature_cols()
        
        logger.info(
            f"Loaded {counts['catboost']} CatBoost, "
            f"{counts['advanced']} advanced models, "
            f"{counts['blenders']} blenders"
        )
        
        return counts
    
    def _load_catboost_models(self) -> int:
        """Load CatBoost regression models for all targets.

        Also loads MAE companion and quantile (P10/P90) models when
        present on disk so the full multi-model architecture is available
        for blended prediction and calibrated uncertainty.
        """
        from catboost import CatBoostRegressor
        
        loaded = 0
        for target in self.targets:
            # Primary RMSE model
            model_path = self.models_dir / f'{target.lower()}_catboost.cbm'
            if model_path.exists():
                try:
                    model = CatBoostRegressor()
                    model.load_model(str(model_path))
                    self.models[target] = model
                    loaded += 1
                    logger.info(f"Loaded CatBoost model for {target}")
                except Exception as e:
                    logger.warning(f"Failed to load CatBoost for {target}: {e}")
                    continue

            # MAE companion model
            mae_path = self.models_dir / f'{target.lower()}_catboost_mae.cbm'
            if mae_path.exists():
                try:
                    mae_model = CatBoostRegressor()
                    mae_model.load_model(str(mae_path))
                    self.catboost_mae_models[target] = mae_model
                    logger.info(f"Loaded CatBoost MAE model for {target}")
                except Exception as e:
                    logger.debug(f"Failed to load MAE model for {target}: {e}")

            # Quantile models (P10/P90)
            for label in ('low', 'high'):
                q_path = self.models_dir / f'{target.lower()}_catboost_q{label}.cbm'
                if q_path.exists():
                    try:
                        q_model = CatBoostRegressor()
                        q_model.load_model(str(q_path))
                        self.catboost_quantile_models.setdefault(target, {})[label] = q_model
                        logger.info(f"Loaded CatBoost quantile-{label} for {target}")
                    except Exception as e:
                        logger.debug(f"Failed to load quantile-{label} for {target}: {e}")

        n_mae = len(self.catboost_mae_models)
        n_q = sum(len(v) for v in self.catboost_quantile_models.values())
        if n_mae or n_q:
            logger.info(f"Auxiliary CatBoost models: {n_mae} MAE, {n_q} quantile")

        return loaded
    
    def _load_blenders(self) -> int:
        """Load meta-learner blenders."""
        blender_path = self.models_dir / 'blenders.pkl'
        
        if blender_path.exists():
            try:
                self.blenders = joblib.load(blender_path)
                count = len(self.blenders)
                logger.info(f"Loaded {count} blenders")
                return count
            except Exception as e:
                logger.warning(f"Failed to load blenders: {e}")
        
        return 0
    
    def _load_advanced_models(self, use_gpu: bool = False) -> int:
        """Load neural network and GNN models."""
        loaded = 0
        
        for attr_name, (filename, wrapper_name) in self.MODEL_FILES.items():
            model_path = self.models_dir / filename
            
            if not model_path.exists():
                continue
            
            try:
                model = self._load_wrapper_model(model_path, wrapper_name, use_gpu)
                setattr(self, attr_name, model)
                loaded += 1
                logger.info(f"Loaded {attr_name} from {filename}")
            except Exception as e:
                logger.debug(f"Failed to load {attr_name}: {e}")
                setattr(self, attr_name, None)
        
        return loaded
    
    def _load_wrapper_model(
        self, 
        path: Path, 
        wrapper_name: str, 
        use_gpu: bool
    ) -> Any:
        """Load a wrapper model dynamically.
        
        Args:
            path: Path to the model file.
            wrapper_name: Class name of the wrapper.
            use_gpu: Whether to enable GPU mode.
            
        Returns:
            Loaded model instance.
        """
        from src.models.multi_output_nn import MultiOutputWrapper
        from src.models.lstm_model import LSTMWrapper
        from src.models.transformer_model import TransformerWrapper
        from src.models.temporal_attention import TemporalAttentionWrapper
        from src.models.gnn_model import GNNWrapper
        
        wrapper_classes = {
            'MultiOutputWrapper': MultiOutputWrapper,
            'LSTMWrapper': LSTMWrapper,
            'TransformerWrapper': TransformerWrapper,
            'TemporalAttentionWrapper': TemporalAttentionWrapper,
            'GNNWrapper': GNNWrapper,
        }
        
        wrapper_cls = wrapper_classes.get(wrapper_name)
        if wrapper_cls is None:
            raise ValueError(f"Unknown wrapper type: {wrapper_name}")
        
        model = wrapper_cls.load(str(path))
        
        if use_gpu and hasattr(model, 'use_gpu'):
            model.use_gpu = True
        
        return model
    
    def _load_feature_cols(self) -> Optional[List[str]]:
        """Load saved feature column names."""
        path = self.models_dir / 'feature_cols.pkl'
        
        if path.exists():
            try:
                self.feature_cols = joblib.load(path)
                logger.info(f"Loaded {len(self.feature_cols)} feature columns")
                return self.feature_cols
            except Exception as e:
                logger.warning(f"Failed to load feature columns: {e}")
        
        return None
    
    def save_feature_cols(self, feature_cols: List[str]) -> None:
        """Save feature column names.
        
        Args:
            feature_cols: List of feature column names.
        """
        self.feature_cols = feature_cols
        path = self.models_dir / 'feature_cols.pkl'
        joblib.dump(feature_cols, path)
        logger.info(f"Saved {len(feature_cols)} feature columns to {path}")
    
    def save_blenders(self, blenders: Dict[str, Any]) -> None:
        """Save blender models.
        
        Args:
            blenders: Dictionary of blender models by target.
        """
        self.blenders = blenders
        path = self.models_dir / 'blenders.pkl'
        joblib.dump(blenders, path)
        logger.info(f"Saved {len(blenders)} blenders to {path}")
    
    def save_catboost_model(
        self, 
        target: str, 
        model: Any
    ) -> None:
        """Save a CatBoost model.
        
        Args:
            target: Target stat name.
            model: Trained CatBoost model.
        """
        self.models[target] = model
        path = self.models_dir / f'{target.lower()}_catboost.cbm'
        model.save_model(str(path))
        logger.info(f"Saved {target} model to {path}")
    
    def get_model(self, target: str) -> Optional[Any]:
        """Get a loaded model by target name.
        
        Args:
            target: Target stat name.
            
        Returns:
            Model instance or None if not loaded.
        """
        return self.models.get(target)
    
    def has_models(self) -> bool:
        """Check if any models are loaded.
        
        Returns:
            True if at least one model is loaded.
        """
        return len(self.models) > 0
    
    def get_loaded_targets(self) -> List[str]:
        """Get list of targets with loaded models.
        
        Returns:
            List of target names with loaded models.
        """
        return list(self.models.keys())

    # ----- Multi-model prediction helpers -----

    def predict_blended(self, target: str, X, rmse_weight: float = 0.6) -> Any:
        """Blend RMSE + MAE CatBoost predictions for *target*.

        Falls back to RMSE-only when no MAE companion exists.
        """
        import numpy as np

        rmse_pred = self.models[target].predict(X)

        if target in self.catboost_mae_models:
            mae_pred = self.catboost_mae_models[target].predict(X)
            return rmse_pred * rmse_weight + mae_pred * (1 - rmse_weight)

        return rmse_pred

    def predict_quantiles(self, target: str, X) -> Optional[Dict[str, Any]]:
        """Return quantile predictions for *target* if available."""
        if target not in self.catboost_quantile_models:
            return None
        q = self.catboost_quantile_models[target]
        out: Dict[str, Any] = {}
        if 'low' in q:
            out['low'] = q['low'].predict(X)
        if 'high' in q:
            out['high'] = q['high'].predict(X)
        return out if out else None