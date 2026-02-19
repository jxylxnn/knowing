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
    
    def _train_catboost_models(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cat_cols: List[str],
        sample_weights: Optional[np.ndarray] = None
    ) -> None:
        """Train CatBoost models for each target."""
        from catboost import CatBoostRegressor
        
        catboost_config = self.config.catboost
        targets = self.training_config.targets
        
        for target in targets:
            logger.info(f"Training CatBoost for: {target}")
            y = fit_df[target]
            
            # Get cat features that exist in feature_cols
            cat_features = [c for c in cat_cols if c in self.feature_cols]
            
            # Try GPU first, fallback to CPU
            task_type = "GPU" if self.use_gpu else "CPU"
            
            try:
                model = CatBoostRegressor(
                    iterations=catboost_config.iterations,
                    learning_rate=catboost_config.learning_rate,
                    depth=catboost_config.depth,
                    l2_leaf_reg=catboost_config.l2_leaf_reg,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    cat_features=cat_features,
                    verbose=200,
                    early_stopping_rounds=catboost_config.early_stopping_rounds,
                    task_type=task_type,
                    devices='0',
                    random_seed=catboost_config.random_seed
                )
                
                fit_kwargs = {
                    'eval_set': (val_df[self.feature_cols], val_df[target]),
                    'use_best_model': True
                }
                if sample_weights is not None:
                    fit_kwargs['sample_weight'] = sample_weights
                
                model.fit(fit_df[self.feature_cols], y, **fit_kwargs)
                
            except Exception as e:
                logger.error(f"GPU training failed for {target}: {e}. Falling back to CPU.")
                
                model = CatBoostRegressor(
                    iterations=catboost_config.iterations,
                    learning_rate=catboost_config.learning_rate,
                    depth=catboost_config.depth,
                    l2_leaf_reg=catboost_config.l2_leaf_reg,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    cat_features=cat_features,
                    verbose=200,
                    early_stopping_rounds=catboost_config.early_stopping_rounds,
                    task_type="CPU",
                    random_seed=catboost_config.random_seed
                )
                
                model.fit(fit_df[self.feature_cols], y, **fit_kwargs)
            
            self.models[target] = model
            
            # Save model
            model_path = self.data_config.models_dir / f'{target.lower()}_catboost.cbm'
            model.save_model(str(model_path))
            
            # Register in registry
            metadata = ModelMetadata(
                name=f'{target}_catboost',
                model_type='catboost',
                training_params=catboost_config.__dict__
            )
            self.registry.register(f'{target}_catboost', model, metadata)
            
            logger.info(f"Saved {target} model to {model_path}")
    
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
    
    def _train_blender(self, val_df: pd.DataFrame) -> None:
        """Train meta-learner blender using Ridge regression."""
        logger.info("Training Super Learner (Blender)...")
        
        core_targets = self.training_config.targets[:3]
        blender_y = val_df[core_targets].values
        
        # Get predictions from base models
        ens_preds_list = []
        for target in core_targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            X_val = val_df[self.feature_cols]
            
            # Handle different model types
            if hasattr(model, 'predict'):
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
        
        # Load CatBoost models
        for target in self.training_config.targets:
            model_path = self.data_config.models_dir / f'{target.lower()}_catboost.cbm'
            if model_path.exists():
                from catboost import CatBoostRegressor
                model = CatBoostRegressor()
                model.load_model(str(model_path))
                self.models[target] = model
                logger.info(f"Loaded {target} CatBoost model")
        
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
                y_pred = model.predict(X_test)
            except (TypeError, AttributeError):
                y_pred = model.predict(X_test, test_df)
            
            results[target] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': root_mean_squared_error(y_test, y_pred)
            }
        
        return results
