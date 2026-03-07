"""CatBoost trainer with parallel target training and smart optimization."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.training.trainer import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class CatBoostTrainer(BaseTrainer):
    """CatBoost trainer with per-target hyperparameter tuning.
    
    Supports parallel training across multiple targets and includes
    multi-loss training (RMSE + MAE) and quantile regression.
    """
    
    # Per-target hyperparameter profiles
    TARGET_PROFILES: Dict[str, Dict[str, Any]] = {
        'PTS': {
            'depth': 9, 'iterations': 3500, 'learning_rate': 0.018,
            'l2_leaf_reg': 5.0, 'min_data_in_leaf': 8,
            'grow_policy': 'Depthwise',
        },
        'REB': {
            'depth': 8, 'iterations': 3000, 'learning_rate': 0.02,
            'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10,
            'grow_policy': 'Depthwise',
        },
        'AST': {
            'depth': 8, 'iterations': 3000, 'learning_rate': 0.02,
            'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10,
            'grow_policy': 'Depthwise',
        },
        'STL': {
            'depth': 6, 'iterations': 2500, 'learning_rate': 0.025,
            'l2_leaf_reg': 8.0, 'min_data_in_leaf': 20,
            'grow_policy': 'SymmetricTree',
        },
        'BLK': {
            'depth': 6, 'iterations': 2500, 'learning_rate': 0.025,
            'l2_leaf_reg': 8.0, 'min_data_in_leaf': 20,
            'grow_policy': 'SymmetricTree',
        },
        'TOV': {
            'depth': 7, 'iterations': 2500, 'learning_rate': 0.022,
            'l2_leaf_reg': 6.0, 'min_data_in_leaf': 15,
            'grow_policy': 'SymmetricTree',
        },
    }
    
    def __init__(
        self,
        model_name: str,
        target: str,
        config: Dict[str, Any],
        use_gpu: bool = False,
        device: Optional[str] = None,
        random_state: int = 42,
        use_multi_loss: bool = True,
        use_quantile: bool = True,
    ):
        """Initialize CatBoost trainer."""
        super().__init__(model_name, config, use_gpu, device, random_state)
        
        self.target = target
        self.use_multi_loss = use_multi_loss
        self.use_quantile = use_quantile
        
        self.primary_model: Optional[CatBoostRegressor] = None
        self.mae_model: Optional[CatBoostRegressor] = None
        self.quantile_low_model: Optional[CatBoostRegressor] = None
        self.quantile_high_model: Optional[CatBoostRegressor] = None
        
        self._feature_importance: Optional[Dict[str, float]] = None
        self.feature_cols: Optional[List[str]] = None
        self.cat_features: Optional[List[str]] = None
    
    def _build_params(self, loss_function: str = 'RMSE', model_type: str = 'primary') -> Dict[str, Any]:
        """Build CatBoost parameters."""
        params = {
            'iterations': self.config.get('iterations', 3000),
            'learning_rate': self.config.get('learning_rate', 0.02),
            'depth': self.config.get('depth', 8),
            'l2_leaf_reg': self.config.get('l2_leaf_reg', 5.0),
            'border_count': self.config.get('border_count', 254),
            'random_strength': self.config.get('random_strength', 1.0),
            'bagging_temperature': self.config.get('bagging_temperature', 0.5),
            'early_stopping_rounds': self.config.get('early_stopping_rounds', 150),
            'random_seed': self.random_state,
            'grow_policy': self.config.get('grow_policy', 'Depthwise'),
            'min_data_in_leaf': self.config.get('min_data_in_leaf', 10),
            'rsm': self.config.get('rsm', 0.8),
            'verbose': 200,
        }
        
        # Per-target overrides
        if self.config.get('use_per_target_tuning', True) and self.target in self.TARGET_PROFILES:
            params.update(self.TARGET_PROFILES[self.target])
        
        # Adjust for model type
        if model_type == 'mae':
            params['iterations'] = int(params['iterations'] * 0.65)
            params['early_stopping_rounds'] = max(20, int(params['iterations'] * 0.05))
        elif model_type.startswith('quantile'):
            params['iterations'] = int(params['iterations'] * 0.5)
            params['early_stopping_rounds'] = max(15, int(params['iterations'] * 0.08))
        
        # Loss function
        params['loss_function'] = loss_function
        if loss_function == 'RMSE':
            params['eval_metric'] = 'RMSE'
        elif loss_function == 'MAE':
            params['eval_metric'] = 'MAE'
        elif loss_function.startswith('Quantile'):
            params['eval_metric'] = loss_function
        
        # Boosting type
        if params['grow_policy'] == 'SymmetricTree':
            params['boosting_type'] = 'Ordered'
        else:
            params['boosting_type'] = 'Plain'
        
        if params['grow_policy'] in ('SymmetricTree', 'Depthwise'):
            params['score_function'] = self.config.get('score_function', 'Cosine')
        
        if self.config.get('langevin', False):
            params['langevin'] = True
            params['diffusion_temperature'] = self.config.get('diffusion_temperature', 10000.0)
        
        return params
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_cols: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        **kwargs
    ) -> TrainResult:
        """Train CatBoost models."""
        start_time = time.time()
        
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        
        X_train_clean, y_train_clean = self.validate_data(X_train, y_train)
        X_val_clean = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_val_clean = y_val.values if isinstance(y_val, pd.Series) else y_val
        
        # Train primary model
        logger.info(f"Training CatBoost for {self.target}: RMSE model")
        self.primary_model = self._train_single_model(
            X_train_clean, y_train_clean, X_val_clean, y_val_clean,
            'RMSE', 'primary', sample_weight
        )
        
        # Train MAE companion
        if self.use_multi_loss:
            logger.info(f"Training CatBoost for {self.target}: MAE model")
            self.mae_model = self._train_single_model(
                X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                'MAE', 'mae', sample_weight
            )
        
        # Train quantile models
        if self.use_quantile:
            q_low = self.config.get('quantile_alpha_low', 0.1)
            q_high = self.config.get('quantile_alpha_high', 0.9)
            
            logger.info(f"Training CatBoost for {self.target}: Quantile models")
            self.quantile_low_model = self._train_single_model(
                X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                f'Quantile:alpha={q_low}', 'quantile_low', sample_weight
            )
            self.quantile_high_model = self._train_single_model(
                X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                f'Quantile:alpha={q_high}', 'quantile_high', sample_weight
            )
        
        if self.primary_model is not None:
            self._compute_feature_importance()
        
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            metrics = self.compute_metrics(y_val_clean, y_pred)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        return TrainResult(
            model=self,
            metrics=metrics,
            training_time=training_time,
            best_iteration=self.primary_model.get_best_iteration() if self.primary_model else None,
            feature_importance=self._feature_importance,
        )
    
    def _train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        loss_function: str,
        model_type: str,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CatBoostRegressor:
        """Train a single CatBoost model."""
        params = self._build_params(loss_function, model_type)
        
        task_type = 'GPU' if self.use_gpu else 'CPU'
        
        model_params = {**params, 'cat_features': self.cat_features or []}
        
        if task_type == 'GPU':
            model_params['task_type'] = 'GPU'
            model_params['devices'] = '0'
        else:
            model_params['task_type'] = 'CPU'
        
        model = CatBoostRegressor(**model_params)
        
        fit_kwargs = {'eval_set': (X_val, y_val), 'use_best_model': True}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            if task_type == 'GPU':
                logger.warning(f"GPU training failed ({e}), falling back to CPU")
                model_params['task_type'] = 'CPU'
                del model_params['devices']
                model = CatBoostRegressor(**model_params)
                model.fit(X_train, y_train, **fit_kwargs)
            else:
                raise
        
        return model
    
    def _compute_feature_importance(self) -> None:
        """Compute feature importance."""
        if self.primary_model is None or self.feature_cols is None:
            return
        
        importance = self.primary_model.get_feature_importance()
        self._feature_importance = {
            name: float(imp)
            for name, imp in zip(self.feature_cols, importance)
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Make predictions (blended if multi-loss)."""
        if self.primary_model is None:
            raise RuntimeError("Model not trained")
        
        X_clean, _ = self.validate_data(X, None)
        rmse_pred = self.primary_model.predict(X_clean)
        
        if self.mae_model is not None and self.use_multi_loss:
            w_rmse = self.config.get('multi_loss_rmse_weight', 0.6)
            w_mae = self.config.get('multi_loss_mae_weight', 0.4)
            mae_pred = self.mae_model.predict(X_clean)
            return rmse_pred * w_rmse + mae_pred * w_mae
        
        return rmse_pred
    
    def predict_quantiles(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Get quantile predictions."""
        if not self.use_quantile or self.quantile_low_model is None:
            return None
        
        X_clean, _ = self.validate_data(X, None)
        result = {}
        
        if self.quantile_low_model is not None:
            result['low'] = self.quantile_low_model.predict(X_clean)
        if self.quantile_high_model is not None:
            result['high'] = self.quantile_high_model.predict(X_clean)
        
        return result if result else None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        return self._feature_importance
    
    def save(self, path: Union[str, Path]) -> None:
        """Save all models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.primary_model is not None:
            self.primary_model.save_model(str(path / f"{self.target.lower()}_catboost.cbm"))
        
        if self.mae_model is not None:
            self.mae_model.save_model(str(path / f"{self.target.lower()}_catboost_mae.cbm"))
        
        if self.quantile_low_model is not None:
            self.quantile_low_model.save_model(str(path / f"{self.target.lower()}_catboost_qlow.cbm"))
        
        if self.quantile_high_model is not None:
            self.quantile_high_model.save_model(str(path / f"{self.target.lower()}_catboost_qhigh.cbm"))
        
        metadata = {
            'target': self.target,
            'config': self.config,
            'use_multi_loss': self.use_multi_loss,
            'use_quantile': self.use_quantile,
            'feature_cols': self.feature_cols,
            'cat_features': self.cat_features,
            'feature_importance': self._feature_importance,
        }
        
        joblib.dump(metadata, path / f"{self.target.lower()}_metadata.joblib")
        logger.info(f"Saved CatBoost models for {self.target} to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], target: str, **kwargs) -> 'CatBoostTrainer':
        """Load a trained CatBoost trainer."""
        path = Path(path)
        
        metadata = joblib.load(path / f"{target.lower()}_metadata.joblib")
        
        trainer = cls(
            model_name=f"catboost_{target}",
            target=target,
            config=metadata['config'],
            use_multi_loss=metadata['use_multi_loss'],
            use_quantile=metadata['use_quantile'],
        )
        
        trainer.feature_cols = metadata.get('feature_cols')
        trainer.cat_features = metadata.get('cat_features')
        trainer._feature_importance = metadata.get('feature_importance')
        
        # Load models
        primary_path = path / f"{target.lower()}_catboost.cbm"
        if primary_path.exists():
            trainer.primary_model = CatBoostRegressor()
            trainer.primary_model.load_model(str(primary_path))
        
        mae_path = path / f"{target.lower()}_catboost_mae.cbm"
        if mae_path.exists():
            trainer.mae_model = CatBoostRegressor()
            trainer.mae_model.load_model(str(mae_path))
        
        qlow_path = path / f"{target.lower()}_catboost_qlow.cbm"
        if qlow_path.exists():
            trainer.quantile_low_model = CatBoostRegressor()
            trainer.quantile_low_model.load_model(str(qlow_path))
        
        qhigh_path = path / f"{target.lower()}_catboost_qhigh.cbm"
        if qhigh_path.exists():
            trainer.quantile_high_model = CatBoostRegressor()
            trainer.quantile_high_model.load_model(str(qhigh_path))
        
        trainer.is_trained = True
        logger.info(f"Loaded CatBoost trainer for {target}")
        
        return trainer


def train_catboost_target(
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any],
    cat_features: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    use_gpu: bool = False,
) -> Tuple[str, TrainResult]:
    """Train CatBoost for a single target (for parallel execution)."""
    trainer = CatBoostTrainer(
        model_name=f"catboost_{target}",
        target=target,
        config=config,
        use_gpu=use_gpu,
    )
    
    result = trainer.fit(
        X_train, y_train,
        X_val, y_val,
        sample_weight=sample_weight,
        feature_cols=list(X_train.columns),
        cat_features=cat_features,
    )
    
    return target, result