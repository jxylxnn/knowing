"""Training pipeline for NBA prediction models."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib

from src.config import Config, TrainingConfig, CatBoostConfig
from src.models.base import ModelRegistry, ModelMetadata
from src.models.stacked_ensemble import StackedEnsembleModel
from src.models.multi_output_nn import MultiOutputWrapper
from src.models.lstm_model import LSTMWrapper
from src.models.transformer_model import TransformerWrapper
from src.models.gnn_model import GNNWrapper
from src.models.temporal_attention import TemporalAttentionWrapper
from src.models.advanced_trainer import AdvancedTrainer
from src.models.gpu_utils import check_gpu_compatibility, get_device

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates model training with multiple model types.
    
    This class handles training of:
    - CatBoost models (one per target)
    - Joint Neural Network
    - LSTM temporal model
    - Transformer attention model
    - GNN model
    - Meta-learner blender
    """
    
    def __init__(
        self,
        config: Config,
        registry: Optional[ModelRegistry] = None
    ):
        """Initialize training pipeline.
        
        Args:
            config: Full configuration
            registry: Optional model registry for saving/loading
        """
        self.config = config
        self.training_config = config.training
        self.data_config = config.data
        self.registry = registry or ModelRegistry(self.data_config.models_dir)
        
        # Model containers
        self.models: Dict[str, Any] = {}
        self.catboost_mae_models: Dict[str, Any] = {}
        self.catboost_quantile_models: Dict[str, Dict[str, Any]] = {}
        self.joint_model: Optional[MultiOutputWrapper] = None
        self.temporal_model: Optional[LSTMWrapper] = None
        self.attention_model: Optional[TransformerWrapper] = None
        self.adv_temporal_model: Optional[TemporalAttentionWrapper] = None
        self.gnn_model: Optional[GNNWrapper] = None
        self.blenders: Dict[str, Ridge] = {}
        
        # State
        self.feature_cols: Optional[List[str]] = None
        self.advanced_trainer: Optional[AdvancedTrainer] = None
        
        # GPU setup
        self.use_gpu = check_gpu_compatibility()
        self.device = get_device()
        
        if self.use_gpu:
            logger.info("GPU Training ENABLED for TrainingPipeline")
        else:
            logger.info("CPU Training mode active")
    
    def train_all_models(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train all enabled models.
        
        Args:
            train_df: Training DataFrame with features and targets
            feature_cols: List of feature column names
            val_df: Optional validation DataFrame
            
        Returns:
            Dictionary of trained models
        """
        # Validate input
        if train_df is None or train_df.empty:
            raise ValueError("Training DataFrame is None or empty")
        
        if len(train_df) < 5000:
            raise ValueError(f"Training data too small: {len(train_df)} rows")
        
        self.feature_cols = feature_cols
        targets = self.training_config.targets
        
        # Validate required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE'] + targets
        missing_cols = [c for c in required_cols if c not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Split into fit/val if val not provided
        if val_df is None:
            train_df = train_df.sort_values('GAME_DATE')
            split_idx = int(len(train_df) * 0.85)
            fit_df = train_df.iloc[:split_idx].copy()
            val_df = train_df.iloc[split_idx:].copy()
        else:
            fit_df = train_df.copy()
        
        # Validate splits
        if len(fit_df) < 1000:
            raise ValueError(f"Fit dataset too small: {len(fit_df)} rows")
        if len(val_df) < 500:
            raise ValueError(f"Validation dataset too small: {len(val_df)} rows")
        
        # Get categorical columns
        cat_cols = self._get_categorical_columns(fit_df)
        
        # Initialize advanced trainer
        self.advanced_trainer = AdvancedTrainer(feature_cols, cat_features=cat_cols)
        if self.use_gpu:
            self.advanced_trainer.use_gpu = True
        
        # Perform adversarial validation if enabled
        adv_weights = None
        if self.training_config.use_adversarial_validation:
            adv_weights = self.advanced_trainer.perform_adversarial_validation(fit_df, val_df)
        
        # Feature selection if enabled
        if self.training_config.use_feature_selection:
            logger.info("Optimizing feature space...")
            optimized_features = self.advanced_trainer.select_best_features(
                fit_df[feature_cols], 
                fit_df['PTS']
            )
            self.feature_cols = optimized_features
            logger.info(f"Training with {len(self.feature_cols)} optimized features")
        
        # Clean data
        fit_df = self._clean_data(fit_df, targets)
        val_df = self._clean_data(val_df, targets)
        
        # Train CatBoost models
        if self.config.catboost.enabled:
            self._train_catboost_models(fit_df, val_df, cat_cols, adv_weights)
        
        # Train neural network models
        nn_features = [c for c in self.feature_cols if c not in cat_cols]
        
        if self.config.catboost.enabled:  # Joint model uses same flag for now
            self._train_joint_model(fit_df, nn_features)
        
        if self.config.lstm.enabled:
            self._train_lstm_model(fit_df, nn_features)
        
        if self.config.transformer.enabled:
            self._train_transformer_model(fit_df, nn_features)
        
        if self.config.gnn.enabled:
            self._train_gnn_model(fit_df, nn_features)
        
        # Train advanced temporal attention
        if self.config.ensemble.use_advanced_temporal:
            self._train_temporal_attention_model(fit_df, nn_features)
        
        # Train blender
        if self.config.ensemble.enabled:
            self._train_blender(val_df)
        
        # Save feature columns
        self._save_feature_cols()
        
        return self.models
    
    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get categorical columns for CatBoost."""
        cat_cols = ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID']
        return [c for c in cat_cols if c in df.columns]
    
    def _clean_data(self, df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
        """Clean data for training."""
        df = df.copy()
        
        # Fill NaN in features
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        # Clean targets
        for target in targets:
            t_col = f'{target}_CLEAN' if f'{target}_CLEAN' in df.columns else target
            df[target] = pd.to_numeric(df[t_col], errors='coerce').fillna(0)
        
        return df
    
    # Per-target CatBoost hyperparameter profiles matching ModelManager.
    _TARGET_PROFILES: Dict[str, Dict[str, Any]] = {
        'PTS': {'depth': 9, 'iterations': 3500, 'learning_rate': 0.018,
                'l2_leaf_reg': 5.0, 'min_data_in_leaf': 8, 'grow_policy': 'Depthwise'},
        'REB': {'depth': 8, 'iterations': 3000, 'learning_rate': 0.02,
                'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10, 'grow_policy': 'Depthwise'},
        'AST': {'depth': 8, 'iterations': 3000, 'learning_rate': 0.02,
                'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10, 'grow_policy': 'Depthwise'},
        'STL': {'depth': 6, 'iterations': 2500, 'learning_rate': 0.025,
                'l2_leaf_reg': 8.0, 'min_data_in_leaf': 20, 'grow_policy': 'SymmetricTree'},
        'BLK': {'depth': 6, 'iterations': 2500, 'learning_rate': 0.025,
                'l2_leaf_reg': 8.0, 'min_data_in_leaf': 20, 'grow_policy': 'SymmetricTree'},
        'TOV': {'depth': 7, 'iterations': 2500, 'learning_rate': 0.022,
                'l2_leaf_reg': 6.0, 'min_data_in_leaf': 15, 'grow_policy': 'SymmetricTree'},
    }

    def _build_catboost_params(
        self, target: str, loss_function: str = 'RMSE'
    ) -> Dict[str, Any]:
        """Build CatBoost param dict from config + per-target profile."""
        cfg = self.config.catboost
        params: Dict[str, Any] = {
            'iterations': cfg.iterations,
            'learning_rate': cfg.learning_rate,
            'depth': cfg.depth,
            'l2_leaf_reg': cfg.l2_leaf_reg,
            'border_count': cfg.border_count,
            'random_strength': cfg.random_strength,
            'bagging_temperature': cfg.bagging_temperature,
            'early_stopping_rounds': cfg.early_stopping_rounds,
            'random_seed': cfg.random_seed,
            'grow_policy': cfg.grow_policy,
            'min_data_in_leaf': cfg.min_data_in_leaf,
            'rsm': cfg.rsm,
        }

        # Per-target overrides first (before derived params)
        if cfg.use_per_target_tuning and target in self._TARGET_PROFILES:
            params.update(self._TARGET_PROFILES[target])

        # Ordered boosting is only supported for SymmetricTree
        gp = params['grow_policy']
        if gp == 'SymmetricTree':
            params['boosting_type'] = 'Ordered'
        else:
            params['boosting_type'] = 'Plain'

        if cfg.langevin:
            params['langevin'] = True
            params['diffusion_temperature'] = cfg.diffusion_temperature

        if gp in ('SymmetricTree', 'Depthwise'):
            params['score_function'] = cfg.score_function

        params['loss_function'] = loss_function
        if loss_function == 'RMSE':
            params['eval_metric'] = 'RMSE'
        elif loss_function == 'MAE':
            params['eval_metric'] = 'MAE'
        elif loss_function.startswith('Quantile'):
            params['eval_metric'] = loss_function

        return params

    def _fit_catboost(
        self,
        params: Dict[str, Any],
        cat_features: List[str],
        X_train, y_train, X_val, y_val,
        sample_weights=None,
    ):
        """Train a CatBoostRegressor with automatic GPU→CPU fallback."""
        from catboost import CatBoostRegressor

        task_type = 'GPU' if self.use_gpu else 'CPU'
        fit_kw: Dict[str, Any] = {
            'eval_set': (X_val, y_val), 'use_best_model': True,
        }
        if sample_weights is not None:
            fit_kw['sample_weight'] = sample_weights

        def _make(tt: str):
            p = {**params, 'cat_features': cat_features, 'verbose': 200,
                 'task_type': tt}
            if tt == 'GPU':
                p['devices'] = '0'
            return CatBoostRegressor(**p)

        try:
            model = _make(task_type)
            model.fit(X_train, y_train, **fit_kw)
            return model
        except Exception as exc:
            if task_type == 'GPU':
                logger.warning(f"GPU failed ({exc}), falling back to CPU")
                model = _make('CPU')
                model.fit(X_train, y_train, **fit_kw)
                return model
            raise

    def _train_catboost_models(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cat_cols: List[str],
        sample_weights: Optional[np.ndarray] = None
    ) -> None:
        """Train advanced multi-model CatBoost per target.

        For each target trains:
        1. Primary RMSE model with per-target tuned hyperparameters
        2. MAE companion model (if ``use_multi_loss``)
        3. Quantile P10/P90 models (if ``use_quantile_models``)
        """
        cfg = self.config.catboost
        targets = self.training_config.targets
        cat_features = [c for c in cat_cols if c in self.feature_cols]

        for target in targets:
            logger.info(f"=== Training Advanced CatBoost for: {target} ===")
            y_fit = fit_df[target]
            y_val = val_df[target]
            X_fit = fit_df[self.feature_cols]
            X_v = val_df[self.feature_cols]

            # 1. Primary RMSE model
            rmse_params = self._build_catboost_params(target, 'RMSE')
            logger.info(
                f"  RMSE: depth={rmse_params['depth']}, "
                f"iter={rmse_params['iterations']}, lr={rmse_params['learning_rate']:.4f}"
            )
            model = self._fit_catboost(
                rmse_params, cat_features, X_fit, y_fit, X_v, y_val, sample_weights
            )
            self.models[target] = model

            model_path = self.data_config.models_dir / f'{target.lower()}_catboost.cbm'
            model.save_model(str(model_path))

            metadata = ModelMetadata(
                name=f'{target}_catboost',
                model_type='catboost',
                training_params=rmse_params,
            )
            self.registry.register(f'{target}_catboost', model, metadata)

            # 2. MAE companion
            if cfg.use_multi_loss:
                mae_params = self._build_catboost_params(target, 'MAE')
                mae_params['iterations'] = int(mae_params['iterations'] * 0.8)
                mae_model = self._fit_catboost(
                    mae_params, cat_features, X_fit, y_fit, X_v, y_val, sample_weights
                )
                self.catboost_mae_models[target] = mae_model
                mae_path = self.data_config.models_dir / f'{target.lower()}_catboost_mae.cbm'
                mae_model.save_model(str(mae_path))
                logger.info(f"  Saved MAE model to {mae_path}")

            # 3. Quantile models
            if cfg.use_quantile_models:
                q_iter = int(rmse_params['iterations'] * 0.6)
                for alpha, label in [(cfg.quantile_alpha_low, 'low'),
                                     (cfg.quantile_alpha_high, 'high')]:
                    q_params = self._build_catboost_params(
                        target, f'Quantile:alpha={alpha}'
                    )
                    q_params['iterations'] = q_iter
                    q_model = self._fit_catboost(
                        q_params, cat_features, X_fit, y_fit, X_v, y_val, sample_weights
                    )
                    self.catboost_quantile_models.setdefault(target, {})[label] = q_model
                    q_path = self.data_config.models_dir / f'{target.lower()}_catboost_q{label}.cbm'
                    q_model.save_model(str(q_path))
                    logger.info(f"  Saved quantile-{label} model to {q_path}")

            logger.info(f"  {target} complete.")
    
    def _train_joint_model(self, fit_df: pd.DataFrame, nn_features: List[str]) -> None:
        """Train joint neural network model."""
        logger.info("Training Joint Stats NN...")
        
        core_targets = self.training_config.targets[:3]  # PTS, REB, AST
        
        self.joint_model = MultiOutputWrapper(
            input_dim=len(nn_features),
            target_names=core_targets,
            hidden_dim=512
        )
        
        self.joint_model.fit(fit_df[nn_features], fit_df[core_targets])
        
        # Save model
        model_path = self.data_config.models_dir / 'joint_stats_nn.pkl'
        self.joint_model.save(str(model_path))
        
        logger.info(f"Saved joint model to {model_path}")
    
    def _train_lstm_model(self, fit_df: pd.DataFrame, nn_features: List[str]) -> None:
        """Train LSTM temporal model."""
        logger.info("Training LSTM model...")
        
        core_targets = self.training_config.targets[:3]
        lstm_config = self.config.lstm
        
        self.temporal_model = LSTMWrapper(
            input_dim=len(nn_features),
            seq_len=lstm_config.sequence_length
        )
        
        self.temporal_model.fit(fit_df, nn_features, core_targets)
        
        # Save model
        model_path = self.data_config.models_dir / 'temporal_lstm.pkl'
        self.temporal_model.save(str(model_path))
        
        logger.info(f"Saved LSTM model to {model_path}")
    
    def _train_transformer_model(self, fit_df: pd.DataFrame, nn_features: List[str]) -> None:
        """Train Transformer attention model."""
        logger.info("Training Transformer model...")
        
        core_targets = self.training_config.targets[:3]
        transformer_config = self.config.transformer
        
        self.attention_model = TransformerWrapper(
            input_dim=len(nn_features),
            seq_len=transformer_config.max_seq_length
        )
        
        self.attention_model.fit(fit_df, nn_features, core_targets)
        
        # Save model
        model_path = self.data_config.models_dir / 'attention_transformer.pkl'
        self.attention_model.save(str(model_path))
        
        logger.info(f"Saved Transformer model to {model_path}")
    
    def _train_gnn_model(self, fit_df: pd.DataFrame, nn_features: List[str]) -> None:
        """Train GNN model."""
        logger.info("Training GNN model...")
        
        core_targets = self.training_config.targets[:3]
        
        self.gnn_model = GNNWrapper(
            input_dim=len(nn_features),
            target_names=core_targets
        )
        
        self.gnn_model.fit(fit_df, nn_features, core_targets)
        
        # Save model
        model_path = self.data_config.models_dir / 'team_chemistry_gnn.pkl'
        self.gnn_model.save(str(model_path))
        
        logger.info(f"Saved GNN model to {model_path}")
    
    def _train_temporal_attention_model(self, fit_df: pd.DataFrame, nn_features: List[str]) -> None:
        """Train advanced Temporal Attention model (context-aware)."""
        logger.info("Training Temporal Attention model...")
        
        core_targets = self.training_config.targets[:3]
        
        self.adv_temporal_model = TemporalAttentionWrapper(
            input_dim=len(nn_features),
            seq_len=20
        )
        
        self.adv_temporal_model.fit(fit_df, nn_features, core_targets)
        
        model_path = self.data_config.models_dir / 'adv_temporal_attention.pkl'
        self.adv_temporal_model.save(str(model_path))
        
        logger.info(f"Saved Temporal Attention model to {model_path}")
    
    def _predict_catboost_blended(self, target: str, X) -> np.ndarray:
        """Blend RMSE + MAE CatBoost predictions for *target*."""
        rmse_pred = self.models[target].predict(X)
        if target in self.catboost_mae_models:
            w = self.config.catboost.multi_loss_rmse_weight
            mae_pred = self.catboost_mae_models[target].predict(X)
            return rmse_pred * w + mae_pred * (1 - w)
        return rmse_pred

    def _train_blender(self, val_df: pd.DataFrame) -> None:
        """Train meta-learner blender using Ridge regression."""
        logger.info("Training Super Learner (Blender)...")
        
        core_targets = self.training_config.targets[:3]
        blender_y = val_df[core_targets].values
        
        # Get blended predictions from base models
        ens_preds_list = []
        for target in core_targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            X_val = val_df[self.feature_cols]
            
            if 'CatBoost' in str(type(model)):
                p = self._predict_catboost_blended(target, X_val)
            elif hasattr(model, 'predict'):
                try:
                    p = model.predict(X_val)
                except (TypeError, AttributeError):
                    p = model.predict(X_val, val_df)
            else:
                continue
            
            ens_preds_list.append(p.reshape(-1, 1))
        
        if not ens_preds_list:
            logger.warning("No ensemble models found for blending")
            return
        
        ens_meta = np.hstack(ens_preds_list)
        
        # Add NN predictions if available
        if self.joint_model and hasattr(self.joint_model, 'is_trained') and self.joint_model.is_trained:
            try:
                nn_m, _ = self.joint_model.predict(val_df[self.feature_cols])
                meta_features = np.hstack([ens_meta, nn_m])
            except (TypeError, AttributeError, ValueError) as e:
                logger.warning(f"Joint model prediction failed: {e}")
                meta_features = ens_meta
        else:
            meta_features = ens_meta
        
        # Train blenders
        self.blenders = {}
        for i, target in enumerate(core_targets):
            y_target = blender_y[:, i]
            
            blender = Ridge(alpha=5.0)
            blender.fit(meta_features, y_target)
            self.blenders[target] = blender
            
            logger.info(f"{target} Blender Weights: {blender.coef_}")
            logger.info(f"{target} Blend Intercept: {blender.intercept_:.4f}")
        
        # Save blenders
        self._save_blenders()
    
    def _save_blenders(self) -> None:
        """Save blender models."""
        if not self.blenders:
            return
        
        path = self.data_config.models_dir / 'blenders.pkl'
        joblib.dump(self.blenders, path)
        logger.info(f"Saved {len(self.blenders)} blenders to {path}")
    
    def _save_feature_cols(self) -> None:
        """Save feature column names."""
        if self.feature_cols is None:
            return
        
        path = self.data_config.models_dir / 'feature_cols.pkl'
        joblib.dump(self.feature_cols, path)
        logger.info(f"Saved {len(self.feature_cols)} feature columns to {path}")
    
    def load_models(self) -> None:
        """Load all saved models from disk."""
        logger.info("Loading models from disk...")
        
        # Load feature columns
        feature_cols_path = self.data_config.models_dir / 'feature_cols.pkl'
        if feature_cols_path.exists():
            self.feature_cols = joblib.load(feature_cols_path)
            logger.info(f"Loaded {len(self.feature_cols)} feature columns")
        
        # Load CatBoost models (primary + MAE + quantile)
        from catboost import CatBoostRegressor
        for target in self.training_config.targets:
            model_path = self.data_config.models_dir / f'{target.lower()}_catboost.cbm'
            if model_path.exists():
                model = CatBoostRegressor()
                model.load_model(str(model_path))
                self.models[target] = model
                logger.info(f"Loaded {target} CatBoost model")

            # MAE companion
            mae_path = self.data_config.models_dir / f'{target.lower()}_catboost_mae.cbm'
            if mae_path.exists():
                try:
                    mae_m = CatBoostRegressor()
                    mae_m.load_model(str(mae_path))
                    self.catboost_mae_models[target] = mae_m
                    logger.info(f"Loaded {target} MAE model")
                except Exception as e:
                    logger.debug(f"Failed to load MAE for {target}: {e}")

            # Quantile models
            for label in ('low', 'high'):
                q_path = self.data_config.models_dir / f'{target.lower()}_catboost_q{label}.cbm'
                if q_path.exists():
                    try:
                        q_m = CatBoostRegressor()
                        q_m.load_model(str(q_path))
                        self.catboost_quantile_models.setdefault(target, {})[label] = q_m
                        logger.info(f"Loaded {target} quantile-{label} model")
                    except Exception as e:
                        logger.debug(f"Failed to load quantile-{label} for {target}: {e}")
        
        # Load other models
        model_files = [
            ('joint_stats_nn.pkl', 'joint_model', MultiOutputWrapper),
            ('temporal_lstm.pkl', 'temporal_model', LSTMWrapper),
            ('attention_transformer.pkl', 'attention_model', TransformerWrapper),
            ('team_chemistry_gnn.pkl', 'gnn_model', GNNWrapper),
        ]
        
        for filename, attr_name, wrapper_class in model_files:
            model_path = self.data_config.models_dir / filename
            if model_path.exists():
                try:
                    model = wrapper_class.load(str(model_path))
                    setattr(self, attr_name, model)
                    logger.info(f"Loaded {attr_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {attr_name}: {e}")
        
        # Load blenders
        blenders_path = self.data_config.models_dir / 'blenders.pkl'
        if blenders_path.exists():
            self.blenders = joblib.load(blenders_path)
            logger.info(f"Loaded {len(self.blenders)} blenders")
    
    def evaluate_models(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on test set.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary of evaluation metrics per target
        """
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        
        results = {}
        
        for target in self.training_config.targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            X_test = test_df[self.feature_cols]
            y_test = test_df[target]
            
            try:
                if 'CatBoost' in str(type(model)):
                    y_pred = self._predict_catboost_blended(target, X_test)
                else:
                    y_pred = model.predict(X_test)
            except (TypeError, AttributeError):
                y_pred = model.predict(X_test, test_df)
            
            results[target] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': root_mean_squared_error(y_test, y_pred)
            }
        
        return results
