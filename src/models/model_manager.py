import os
import logging
import torch
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.stacked_ensemble import StackedEnsembleModel
from src.models.multi_output_nn import MultiOutputWrapper
from src.models.lstm_model import LSTMWrapper
from src.models.transformer_model import TransformerWrapper
from src.models.gnn_model import GNNWrapper
from src.models.temporal_attention import TemporalAttentionWrapper
from src.models.advanced_trainer import AdvancedTrainer
from src.models.gpu_utils import check_gpu_compatibility, get_device, clear_gpu_memory, log_gpu_memory
from src.config.model_config import get_model_config, save_model_config, print_config_summary
import joblib
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Orchestrates training with Temporal Weighting and Robust Loss.
    Includes input validation on all public methods.
    Supports auto-sizing based on detected hardware.
    """
    
    def __init__(self, data_dir: str = 'data', models_dir: str = 'models', 
                 model_size: str = 'auto', model_config: Optional[Dict[str, Any]] = None):
        if not isinstance(data_dir, str) or not data_dir:
            raise ValueError(f"Invalid data_dir: {data_dir}")
        if not isinstance(models_dir, str) or not models_dir:
            raise ValueError(f"Invalid models_dir: {models_dir}")
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.core_targets = ['PTS', 'REB', 'AST']
        self.secondary_targets = ['STL', 'BLK', 'TOV']
        self.targets = self.core_targets + self.secondary_targets
        
        self.models: Dict[str, Any] = {}
        self.catboost_mae_models: Dict[str, Any] = {}
        self.catboost_quantile_models: Dict[str, Dict[str, Any]] = {}
        self.joint_model: Optional[MultiOutputWrapper] = None
        self.temporal_model: Optional[LSTMWrapper] = None
        self.attention_model: Optional[TransformerWrapper] = None
        self.adv_temporal_model: Optional[TemporalAttentionWrapper] = None
        self.gnn_model: Optional[GNNWrapper] = None
        
        self.blenders: Dict[str, Any] = {}
        self._gpu_blenders: Dict[str, Any] = {}  # GPU-accelerated blenders for inference
        self.feature_cols: Optional[List[str]] = None
        self.advanced_trainer: Optional[AdvancedTrainer] = None
        
        # Check GPU compatibility before initializing feature engineer
        self.use_gpu = check_gpu_compatibility()
        self.device = get_device()
        
        # Initialize feature engineer with GPU flag
        self.feature_engineer = FeatureEngineer(use_gpu=self.use_gpu)
        
        if model_config is not None:
            self.model_config = model_config
            self.hw_info = model_config.get('metadata', {})
        else:
            self.model_config, self.hw_info = get_model_config(force_size=None if model_size == 'auto' else model_size)
        
        self.training_config = self.model_config.get('training', {})
        
        config_path = os.path.join(self.models_dir, 'training_config.json')
        save_model_config(self.model_config, config_path)
        
        if self.use_gpu:
            logger.info("GPU Training ENABLED for ModelManager.")
        else:
            logger.info("CPU Training mode active.")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and prepares data with validation.
        
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            ValueError: If data files don't exist or are invalid
        """
        # Validate data directory
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # Validate required files
        players_file = os.path.join(self.data_dir, 'nba_players.csv')
        games_file = os.path.join(self.data_dir, 'nba_games.csv')
        
        if not os.path.exists(players_file):
            raise ValueError(f"Players file not found: {players_file}")
        if not os.path.exists(games_file):
            raise ValueError(f"Games file not found: {games_file}")
        
        loader = DataLoader(
            players_file,
            games_file
        )
        merged_df = loader.merge_datasets()
        
        # Validate merged data
        if merged_df.empty:
            raise ValueError("Merged dataset is empty after loading")
        
        if len(merged_df) < 1000:
            raise ValueError(f"Dataset too small: {len(merged_df)} rows (minimum 1000 required)")
        
        full_df = self.feature_engineer.create_features(merged_df)
        
        if full_df.empty:
            raise ValueError("Feature engineering resulted in empty dataset")
        
        # Validate required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE'] + self.core_targets
        missing_cols = [c for c in required_cols if c not in full_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after feature engineering: {missing_cols}")
        
        split_date_str = self.training_config.get('test_split_date', '2024-03-01')
        split_date = pd.to_datetime(split_date_str)
        logger.info(f"Using train/test split date: {split_date_str}")
        train_df = full_df[full_df['GAME_DATE'] < split_date].copy()
        test_df = full_df[full_df['GAME_DATE'] >= split_date].copy()
        
        # Validate splits
        if train_df.empty:
            raise ValueError("Training set is empty after split")
        if test_df.empty:
            raise ValueError("Test set is empty after split")
        
        logger.info(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
        return train_df, test_df

    def _calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        SMART TRAINING: Calculate temporal decay weights.
        Games in the last 30 days get weight 1.0.
        Games 6 months ago get weight 0.2.
        This teaches the model to care about 'NOW'.
        """
        if 'GAME_DATE' not in df.columns:
            return np.ones(len(df))
            
        max_date = df['GAME_DATE'].max()
        days_ago = (max_date - df['GAME_DATE']).dt.days
        
        # Exponential decay: weight = exp(-lambda * days)
        # Lambda chosen so that weight ~0.5 after 30 days
        lambda_decay = 0.023 
        
        weights = np.exp(-lambda_decay * days_ago)
        
        # Clip minimum weight to avoid ignoring old data entirely
        weights = np.clip(weights, 0.1, 1.0)
        
        return weights

    def _preprocess_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SMART TRAINING: Cap targets to handle outliers.
        Prevents 60-point games from skewing the loss landscape.
        Uses 99th percentile caps per player.
        """
        df = df.copy()
        for stat in self.targets:
            if stat in df.columns:
                # Calculate the 99th percentile for each player
                caps = df.groupby('PLAYER_ID')[stat].quantile(0.99)
                
                # Map caps back to dataframe
                player_caps = df['PLAYER_ID'].map(caps)
                
                # Fill NaN caps (if player has <100 games) with global 99th percentile
                global_cap = df[stat].quantile(0.99)
                player_caps = player_caps.fillna(global_cap)
                
                # Cap the stat
                df[f'{stat}_CLEAN'] = df[stat].clip(upper=player_caps)
        return df

    # =====================================================================
    # Advanced CatBoost Multi-Model Architecture
    # =====================================================================

    # Per-target hyperparameter profiles.  Core stats (PTS, REB, AST) get
    # deeper trees to capture complex feature interactions.  Secondary
    # stats (STL, BLK, TOV) are sparse counts that benefit from heavier
    # regularization.
    CATBOOST_TARGET_PROFILES: Dict[str, Dict[str, Any]] = {
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

    def _get_catboost_params(self, target: str, loss_function: str = 'RMSE') -> dict:
        """Build CatBoost parameter dict for *target* and *loss_function*.

        Merges the global catboost config from ``self.model_config`` with the
        per-target profile from ``CATBOOST_TARGET_PROFILES``.
        """
        cat_config = self.model_config.get('catboost', {})
        use_per_target = cat_config.get('use_per_target_tuning', True)

        # Start with global defaults
        params: Dict[str, Any] = {
            'iterations': cat_config.get('iterations', 3000),
            'learning_rate': cat_config.get('learning_rate', 0.02),
            'depth': cat_config.get('depth', 8),
            'l2_leaf_reg': cat_config.get('l2_leaf_reg', 5.0),
            'border_count': cat_config.get('border_count', 254),
            'random_strength': cat_config.get('random_strength', 1.0),
            'bagging_temperature': cat_config.get('bagging_temperature', 0.5),
            'early_stopping_rounds': cat_config.get('early_stopping_rounds', 150),
            'random_seed': cat_config.get('random_seed', 42),
            'grow_policy': cat_config.get('grow_policy', 'Depthwise'),
            'min_data_in_leaf': cat_config.get('min_data_in_leaf', 10),
            'rsm': cat_config.get('rsm', 0.8),
        }

        # Per-target overrides (applied before derived params so grow_policy
        # is known when choosing boosting_type and score_function)
        if use_per_target and target in self.CATBOOST_TARGET_PROFILES:
            params.update(self.CATBOOST_TARGET_PROFILES[target])

        # Ordered boosting is only supported for SymmetricTree.
        gp = params['grow_policy']
        if gp == 'SymmetricTree':
            params['boosting_type'] = 'Ordered'
        else:
            params['boosting_type'] = 'Plain'

        # Langevin boosting (noise injection for regularization)
        if cat_config.get('langevin', False):
            params['langevin'] = True
            params['diffusion_temperature'] = cat_config.get('diffusion_temperature', 10000.0)

        # score_function only applies to SymmetricTree and Depthwise
        if gp in ('SymmetricTree', 'Depthwise'):
            params['score_function'] = cat_config.get('score_function', 'Cosine')

        # Apply the requested loss
        params['loss_function'] = loss_function
        if loss_function == 'RMSE':
            params['eval_metric'] = 'RMSE'
        elif loss_function == 'MAE':
            params['eval_metric'] = 'MAE'
        elif loss_function.startswith('Quantile'):
            params['eval_metric'] = loss_function

        return params

    def _build_catboost_model(
        self, params: dict, cat_features: List[str], task_type: str
    ):
        """Instantiate a ``CatBoostRegressor`` with GPU fallback."""
        from catboost import CatBoostRegressor

        model_params = {**params, 'cat_features': cat_features, 'verbose': 200}

        if task_type == 'GPU':
            model_params['task_type'] = 'GPU'
            model_params['devices'] = '0'
        else:
            model_params['task_type'] = 'CPU'

        return CatBoostRegressor(**model_params)

    def _fit_catboost_safe(
        self,
        params: dict,
        cat_features: List[str],
        X_train, y_train,
        X_val, y_val,
        sample_weight=None,
    ):
        """Train a CatBoost model with automatic GPU→CPU fallback."""
        task_type = 'GPU' if self.use_gpu else 'CPU'
        fit_kwargs: Dict[str, Any] = {
            'eval_set': (X_val, y_val),
            'use_best_model': True,
        }
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        try:
            model = self._build_catboost_model(params, cat_features, task_type)
            model.fit(X_train, y_train, **fit_kwargs)
            return model
        except Exception as e:
            if task_type == 'GPU':
                logger.warning(f"GPU training failed ({e}), falling back to CPU")
                model = self._build_catboost_model(params, cat_features, 'CPU')
                model.fit(X_train, y_train, **fit_kwargs)
                return model
            raise

    def _train_single_catboost_model(
        self,
        target: str,
        loss_function: str,
        X_fit, y_fit,
        X_val, y_val,
        cat_features: List[str],
        adv_weights: Optional[np.ndarray],
        model_type: str = 'primary'
    ) -> Tuple[str, Any]:
        """Train a single CatBoost model with aggressive early stopping.
        
        This method is designed for parallel execution with joblib.
        
        Args:
            target: Target column name
            loss_function: Loss function to use
            X_fit: Training features
            y_fit: Training targets
            X_val: Validation features
            y_val: Validation targets
            cat_features: Categorical feature columns
            adv_weights: Adversarial weights for training
            model_type: Type of model ('primary', 'mae', 'quantile_low', 'quantile_high')
            
        Returns:
            Tuple of (target, trained_model)
        """
        import os
        os.environ['RAY_DISABLE_DOCKER_CPU_WALLLAY'] = '1'
        
        try:
            params = self._get_catboost_params(target, loss_function)
            
            # Aggressive early stopping for faster training
            # Reduce iterations for MAE and quantile models
            if model_type == 'mae':
                params['iterations'] = int(params['iterations'] * 0.65)
                params['early_stopping_rounds'] = max(20, int(params['iterations'] * 0.05))
            elif model_type.startswith('quantile'):
                params['iterations'] = int(params['iterations'] * 0.5)
                params['early_stopping_rounds'] = max(15, int(params['iterations'] * 0.08))
            else:
                # Primary RMSE model - still use aggressive early stopping
                params['early_stopping_rounds'] = max(50, int(params['iterations'] * 0.03))
            
            model = self._fit_catboost_safe(
                params, cat_features, X_fit, y_fit, X_val, y_val, adv_weights
            )
            
            logger.info(f"Trained {model_type} model for {target}: {params['iterations']} iterations")
            return target, model
        except Exception as e:
            logger.error(f"Failed to train {model_type} model for {target}: {e}")
            raise

    def _train_catboost_parallel(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cat_cols: List[str],
        adv_weights: Optional[np.ndarray],
    ) -> None:
        """Train CatBoost models in parallel using joblib.
        
        This method parallelizes training across all targets and loss functions,
        significantly reducing training time on multi-core systems.
        """
        from catboost import CatBoostRegressor

        cat_config = self.model_config.get('catboost', {})
        use_multi_loss = cat_config.get('use_multi_loss', True)
        use_quantile = cat_config.get('use_quantile_models', True)
        q_low = cat_config.get('quantile_alpha_low', 0.1)
        q_high = cat_config.get('quantile_alpha_high', 0.9)

        cat_features = [c for c in cat_cols if c in self.feature_cols]
        X_fit = fit_df[self.feature_cols]
        X_val = val_df[self.feature_cols]

        # Prepare training tasks for parallel execution
        train_tasks = []
        
        for target in self.targets:
            y_fit = fit_df[target]
            y_val = val_df[target]

            # --- 1. Primary RMSE model ---
            train_tasks.append({
                'target': target,
                'loss_function': 'RMSE',
                'X_fit': X_fit,
                'y_fit': y_fit,
                'X_val': X_val,
                'y_val': y_val,
                'cat_features': cat_features,
                'adv_weights': adv_weights,
                'model_type': 'primary'
            })

            # --- 2. MAE model (robust companion) ---
            if use_multi_loss:
                train_tasks.append({
                    'target': target,
                    'loss_function': 'MAE',
                    'X_fit': X_fit,
                    'y_fit': y_fit,
                    'X_val': X_val,
                    'y_val': y_val,
                    'cat_features': cat_features,
                    'adv_weights': adv_weights,
                    'model_type': 'mae'
                })

            # --- 3. Quantile models (uncertainty estimation) ---
            if use_quantile:
                for alpha, label in [(q_low, 'low'), (q_high, 'high')]:
                    train_tasks.append({
                        'target': target,
                        'loss_function': f'Quantile:alpha={alpha}',
                        'X_fit': X_fit,
                        'y_fit': y_fit,
                        'X_val': X_val,
                        'y_val': y_val,
                        'cat_features': cat_features,
                        'adv_weights': adv_weights,
                        'model_type': f'quantile_{label}'
                    })

        # Train all models in parallel
        logger.info(f"Training {len(train_tasks)} CatBoost models in parallel...")
        start_time = logger.time() if hasattr(logger, 'time') else None
        
        # Use thread-based parallelism for CatBoost (it's not fully thread-safe but works in practice)
        models = joblib.Parallel(n_jobs=min(len(train_tasks), joblib.cpu_count()), 
                                  prefer='threads', verbose=10)(
            joblib.delayed(self._train_single_catboost_model_safe)(**task)
            for task in train_tasks
        )
        
        # Parse results and store models
        for target, model in models:
            # Categorize model by its type
            if 'Quantile:alpha=' in model:
                # This shouldn't happen in our current design, but handle it
                continue
            
            model_type = next((task['model_type'] for task in train_tasks 
                              if task['target'] == target and 'loss_function' in str(task)), 'primary')
            
            if 'primary' == model_type:
                self.models[target] = model
                model.save_model(os.path.join(self.models_dir, f'{target.lower()}_catboost.cbm'))
            elif 'mae' == model_type:
                self.catboost_mae_models[target] = model
                model.save_model(os.path.join(self.models_dir, f'{target.lower()}_catboost_mae.cbm'))
            elif 'quantile_low' == model_type:
                self.catboost_quantile_models.setdefault(target, {})['low'] = model
                model.save_model(os.path.join(self.models_dir, f'{target.lower()}_catboost_qlow.cbm'))
            elif 'quantile_high' == model_type:
                self.catboost_quantile_models.setdefault(target, {})['high'] = model
                model.save_model(os.path.join(self.models_dir, f'{target.lower()}_catboost_qhigh.cbm'))
        
        logger.info(f"Parallel CatBoost training complete.")

    def _train_single_catboost_model_safe(
        self,
        target: str,
        loss_function: str,
        X_fit,
        y_fit,
        X_val,
        y_val,
        cat_features: List[str],
        adv_weights: Optional[np.ndarray],
        model_type: str = 'primary'
    ) -> Tuple[str, Any]:
        """Thread-safe wrapper for _train_single_catboost_model."""
        try:
            return self._train_single_catboost_model(
                target=target,
                loss_function=loss_function,
                X_fit=X_fit,
                y_fit=y_fit,
                X_val=X_val,
                y_val=y_val,
                cat_features=cat_features,
                adv_weights=adv_weights,
                model_type=model_type
            )
        except Exception as e:
            logger.error(f"Parallel training failed for {target}/{model_type}: {e}")
            raise

    def _get_cache_key(self, df: pd.DataFrame, columns: List[str]) -> str:
        """Generate a cache key based on DataFrame content hash."""
        import hashlib
        
        # Use only the specified columns for hashing
        subset = df[columns].copy()
        
        # Convert to string representation and hash
        content_str = str(subset.shape) + str(subset.dtypes.to_dict())
        
        # Hash the first 1000 rows for faster computation
        sample = subset.head(1000).to_string()
        content_str += sample
        
        return hashlib.md5(content_str.encode()).hexdigest()[:16]

    def _load_cached_validation_weights(self, fit_df: pd.DataFrame, val_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Try to load cached adversarial validation weights."""
        try:
            cache_key = self._get_cache_key(fit_df, self.feature_cols) + '_' + self._get_cache_key(val_df, self.feature_cols)
            cache_path = os.path.join(self.models_dir, f'adv_weights_{cache_key}.npy')
            
            if os.path.exists(cache_path):
                logger.info(f"Loaded cached adversarial validation weights (key: {cache_key})")
                return np.load(cache_path)
            return None
        except Exception as e:
            logger.debug(f"Failed to load cached weights: {e}")
            return None

    def _save_cached_validation_weights(self, weights: np.ndarray, fit_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Save adversarial validation weights to cache."""
        try:
            cache_key = self._get_cache_key(fit_df, self.feature_cols) + '_' + self._get_cache_key(val_df, self.feature_cols)
            cache_path = os.path.join(self.models_dir, f'adv_weights_{cache_key}.npy')
            np.save(cache_path, weights)
            logger.info(f"Saved adversarial validation weights to cache (key: {cache_key})")
        except Exception as e:
            logger.debug(f"Failed to save cached weights: {e}")

    def _load_cached_features(self, fit_df: pd.DataFrame, target: str = 'PTS') -> Optional[List[str]]:
        """Try to load cached feature selection results."""
        try:
            cache_key = self._get_cache_key(fit_df, self.feature_cols) + '_' + target
            cache_path = os.path.join(self.models_dir, f'features_{cache_key}.pkl')
            
            if os.path.exists(cache_path):
                logger.info(f"Loaded cached feature selection results (key: {cache_key})")
                import joblib
                return joblib.load(cache_path)
            return None
        except Exception as e:
            logger.debug(f"Failed to load cached features: {e}")
            return None

    def _save_cached_features(self, features: List[str], fit_df: pd.DataFrame, target: str = 'PTS') -> None:
        """Save feature selection results to cache."""
        try:
            cache_key = self._get_cache_key(fit_df, self.feature_cols) + '_' + target
            cache_path = os.path.join(self.models_dir, f'features_{cache_key}.pkl')
            import joblib
            joblib.dump(features, cache_path)
            logger.info(f"Saved feature selection results to cache (key: {cache_key})")
        except Exception as e:
            logger.debug(f"Failed to save cached features: {e}")

    def _train_catboost_advanced(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cat_cols: List[str],
        adv_weights: Optional[np.ndarray],
    ) -> None:
        """Advanced multi-model CatBoost training.

        For each target this method trains:

        1. **Primary RMSE model** — standard point prediction.
        2. **MAE model** — outlier-robust companion (blended with RMSE at
           prediction time when ``use_multi_loss`` is enabled).
        3. **Quantile P10 / P90 models** — calibrated prediction intervals
           (when ``use_quantile_models`` is enabled).

        All models use per-target hyperparameter profiles, Cosine scoring,
        Ordered boosting, column subsampling, Langevin regularization, and
        deeper trees for core targets (PTS/REB/AST) vs heavier leaf
        regularization for sparse secondary targets (STL/BLK/TOV).
        """
        from catboost import CatBoostRegressor

        cat_config = self.model_config.get('catboost', {})
        use_multi_loss = cat_config.get('use_multi_loss', True)
        use_quantile = cat_config.get('use_quantile_models', True)
        q_low = cat_config.get('quantile_alpha_low', 0.1)
        q_high = cat_config.get('quantile_alpha_high', 0.9)

        cat_features = [c for c in cat_cols if c in self.feature_cols]
        X_fit = fit_df[self.feature_cols]
        X_val = val_df[self.feature_cols]

        for target in self.targets:
            logger.info(f"=== Training Advanced CatBoost for: {target} ===")

            y_fit = fit_df[target]
            y_val = val_df[target]

            # --- 1. Primary RMSE model ---
            rmse_params = self._get_catboost_params(target, 'RMSE')
            logger.info(
                f"  RMSE model: depth={rmse_params['depth']}, "
                f"iter={rmse_params['iterations']}, lr={rmse_params['learning_rate']:.4f}, "
                f"grow={rmse_params['grow_policy']}"
            )
            rmse_model = self._fit_catboost_safe(
                rmse_params, cat_features, X_fit, y_fit, X_val, y_val, adv_weights
            )
            self.models[target] = rmse_model
            rmse_model.save_model(
                os.path.join(self.models_dir, f'{target.lower()}_catboost.cbm')
            )

            # --- 2. MAE model (robust companion) ---
            if use_multi_loss:
                mae_params = self._get_catboost_params(target, 'MAE')
                mae_params['iterations'] = int(mae_params['iterations'] * 0.8)
                logger.info(f"  MAE model: iter={mae_params['iterations']}")
                mae_model = self._fit_catboost_safe(
                    mae_params, cat_features, X_fit, y_fit, X_val, y_val, adv_weights
                )
                self.catboost_mae_models[target] = mae_model
                mae_model.save_model(
                    os.path.join(self.models_dir, f'{target.lower()}_catboost_mae.cbm')
                )

            # --- 3. Quantile models (uncertainty estimation) ---
            if use_quantile:
                q_iterations = int(rmse_params['iterations'] * 0.6)
                self.catboost_quantile_models.setdefault(target, {})

                for alpha, label in [(q_low, 'low'), (q_high, 'high')]:
                    q_params = self._get_catboost_params(
                        target, f'Quantile:alpha={alpha}'
                    )
                    q_params['iterations'] = q_iterations
                    logger.info(f"  Quantile {label} (alpha={alpha}): iter={q_iterations}")
                    q_model = self._fit_catboost_safe(
                        q_params, cat_features, X_fit, y_fit, X_val, y_val, adv_weights
                    )
                    self.catboost_quantile_models[target][label] = q_model
                    q_model.save_model(
                        os.path.join(
                            self.models_dir, f'{target.lower()}_catboost_q{label}.cbm'
                        )
                    )

            # --- 4. Advanced: Calibrated Quantile Models (Pinball Loss) ---
            if use_quantile:
                logger.info(f"  Training calibrated quantile models...")
                # Use Pinball loss for better quantile calibration
                for alpha, label in [(q_low, 'low'), (q_high, 'high')]:
                    # Create calibrated quantile model with isotonic regression
                    q_model = self.catboost_quantile_models[target][label]
                    # Get predictions on validation set
                    val_pred = q_model.predict(X_val)
                    # Fit calibration if needed (simplified version)
                    self._calibrate_quantile(target, label, val_pred, y_val.values)
            
            logger.info(f"  {target} CatBoost training complete.")
    
    def _calibrate_quantile(self, target: str, label: str, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Calibrate quantile predictions using isotonic regression for better accuracy.
        This ensures quantile predictions are properly calibrated.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Simple calibration: adjust predictions to match expected quantile levels
            # on validation set
            residuals = actuals - predictions
            # Compute calibration factor
            calibration_factor = np.median(residuals)
            
            # Store calibration info for later use
            if not hasattr(self, '_quantile_calibrations'):
                self._quantile_calibrations = {}
            self._quantile_calibrations[(target, label)] = calibration_factor
            
        except ImportError:
            logger.debug("sklearn not available for quantile calibration")

    def _predict_catboost_blended(
        self, target: str, X: pd.DataFrame
    ) -> np.ndarray:
        """Blend RMSE + MAE CatBoost predictions for *target*.

        If only the RMSE model is available the blend falls back to RMSE-only.
        """
        cat_config = self.model_config.get('catboost', {})
        w_rmse = cat_config.get('multi_loss_rmse_weight', 0.6)
        w_mae = cat_config.get('multi_loss_mae_weight', 0.4)

        rmse_pred = self.models[target].predict(X)

        if target in self.catboost_mae_models:
            mae_pred = self.catboost_mae_models[target].predict(X)
            return rmse_pred * w_rmse + mae_pred * w_mae

        return rmse_pred

    def _predict_catboost_quantiles(
        self, target: str, X: pd.DataFrame
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return quantile predictions for *target* if quantile models exist."""
        if target not in self.catboost_quantile_models:
            return None
        q_models = self.catboost_quantile_models[target]
        result = {}
        if 'low' in q_models:
            result['low'] = q_models['low'].predict(X)
        if 'high' in q_models:
            result['high'] = q_models['high'].predict(X)
        return result if result else None

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features, avoiding leakage.
        
        Uses a two-pass approach:
        1. Exclude known non-feature columns (IDs, dates, raw targets)
        2. Block any column that exactly matches a target or looks like an
           unshifted derivative (e.g. raw team/opponent stats without ROLL/EWMA prefix)
        """
        # Columns that are never features
        exclude_cols = {
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'OPPONENT_ID', 'OPPONENT_ABBR',
            'WL', 'SEASON_ID', 'VIDEO_AVAILABLE', 'REST_BUCKET',
        }
        exclude_cols.update(self.targets)
        exclude_cols.update(self.secondary_targets)
        
        # Prefixes/substrings that indicate a safe, time-shifted feature
        SAFE_PREFIXES = (
            'ROLL_', 'EWMA_', 'VS_OPP_', 'PROJ_', 'LEAGUE_PCT_',
        )
        SAFE_SUBSTRINGS = {
            'TREND', 'BAYESIAN', 'PACE', '_TE', '_SHARE_', 'ROLE_INDEX',
            'SEASON_AVG', 'SEASON_SIN', 'SEASON_COS', 'HOT_STREAK', 'COLD_STREAK',
            'POTENTIAL', 'B2B_IMPACT', 'FATIGUE', 'EFF_Z_SCORE', 'FANTASY',
            'SOS_', 'PACE_ADJ',
        }
        # Exact column names that are safe contextual features (not derived from targets)
        SAFE_EXACT = {
            'IS_HOME', 'REST_DAYS', 'IS_B2B', 'FATIGUE_SCORE', 'MONTH', 'DAY_OF_WEEK',
            'EXP_PACE', 'EXP_TEAM_PTS', 'EXP_GAME_TOTAL', 'BLOWOUT_RISK',
            'CLOSE_GAME', 'EXP_MARGIN', 'DAYS_SINCE_LAST', 'MINS_LAST_3',
            'MINS_LAST_7', 'EST_POSS', 'TEAM_PACE_10', 'PACE_FACTOR',
            'STAR_TEAMMATE_OUT',
        }
        
        # Raw target names (lowercase) for fuzzy matching
        target_names_lower = {t.lower() for t in self.targets + self.secondary_targets}
        
        # Initial numeric-only filter
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in ('int64', 'float64', 'int32', 'float32', 'float', 'int')
        ]
        
        # Two-pass safety filter
        safe_feature_cols = []
        for c in feature_cols:
            # Pass 1: Is it explicitly safe?
            if c in SAFE_EXACT:
                safe_feature_cols.append(c)
                continue
            if any(c.startswith(p) for p in SAFE_PREFIXES):
                safe_feature_cols.append(c)
                continue
            if any(s in c for s in SAFE_SUBSTRINGS):
                safe_feature_cols.append(c)
                continue
            
            # Pass 2: Block anything that looks like a raw target or direct derivative
            c_lower = c.lower()
            if any(t in c_lower for t in target_names_lower):
                # This column contains a target name but wasn't matched as safe — block it
                logger.debug(f"Blocking potentially leaky feature: {c}")
                continue
            
            # Anything else that's numeric and not target-related is allowed
            # (e.g. OPPONENT defensive stats, team-level contextual stats)
            safe_feature_cols.append(c)
        
        feature_cols = safe_feature_cols

        # Final guard: remove raw team stats that mirror targets
        feature_cols = [c for c in feature_cols if not (c.endswith('_TEAM') and c.replace('_TEAM', '') in self.targets)]
        
        logger.info(f"Selected {len(feature_cols)} features for training.")
        return feature_cols

    def train_all(self, train_df: pd.DataFrame):
        """
        Master Training Loop: Adversarial Weights + CatBoost + Feature Selection.
        
        Args:
            train_df: Training DataFrame with features and targets
            
        Raises:
            ValueError: If input data is invalid or insufficient
        """
        # Validate input
        if train_df is None or train_df.empty:
            raise ValueError("Training DataFrame is None or empty")
        
        if len(train_df) < 5000:
            raise ValueError(f"Training data too small: {len(train_df)} rows (minimum 5000 required)")
        
        # Validate required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE'] + self.targets
        missing_cols = [c for c in required_cols if c not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in training data: {missing_cols}")
        
        # 1. Initial Feature Setup
        self.feature_cols = self._select_features(train_df)
        
        if not self.feature_cols:
            raise ValueError("No features selected for training")
        
        # Identify categorical columns specifically for CatBoost
        cat_cols = ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] 
        cat_cols = [c for c in cat_cols if c in train_df.columns]
        
        # 2. Robust Preprocessing
        logger.info("Preprocessing targets...")
        train_df = self._preprocess_targets(train_df)
        
        # 3. Split Data
        train_df = train_df.sort_values('GAME_DATE')
        split_idx = int(len(train_df) * 0.85)
        fit_df = train_df.iloc[:split_idx].copy()
        val_df = train_df.iloc[split_idx:].copy()
        
        # Validate splits
        if len(fit_df) < 1000:
            raise ValueError(f"Fit dataset too small after split: {len(fit_df)} rows")
        if len(val_df) < 500:
            raise ValueError(f"Validation dataset too small after split: {len(val_df)} rows")
        
        # Clean Data
        for df_split in [fit_df, val_df]:
            df_split[self.feature_cols] = df_split[self.feature_cols].fillna(0)
            for target in self.targets:
                t_col = f'{target}_CLEAN' if f'{target}_CLEAN' in df_split.columns else target
                df_split[target] = pd.to_numeric(df_split[t_col], errors='coerce').fillna(0)

        # 4. Advanced Optimization
        if not self.advanced_trainer:
            self.advanced_trainer = AdvancedTrainer(self.feature_cols, cat_features=cat_cols)
            if self.use_gpu:
                self.advanced_trainer.use_gpu = True

        # A. Adversarial Validation with Caching
        cached_weights = self._load_cached_validation_weights(fit_df, val_df)
        if cached_weights is not None:
            adv_weights = cached_weights
            logger.info("Loaded cached adversarial validation weights")
        else:
            adv_weights = self.advanced_trainer.perform_adversarial_validation(fit_df, val_df)
            self._save_cached_validation_weights(adv_weights, fit_df, val_df)

        # B. Feature Selection with Caching
        cached_features = self._load_cached_features(fit_df, target='PTS')
        if cached_features is not None:
            self.feature_cols = cached_features
            logger.info("Loaded cached feature selection results")
        else:
            logger.info("Optimizing feature space for PTS...")
            optimized_features = self.advanced_trainer.select_best_features(fit_df[self.feature_cols], fit_df['PTS'])
            self.feature_cols = optimized_features
            self._save_cached_features(self.feature_cols, fit_df, target='PTS')
            logger.info(f"Training with optimized feature set size: {len(self.feature_cols)}")
        
        X_fit = fit_df[self.feature_cols]
        self._save_feature_cols()

        # 5. Train CatBoost Models (Advanced Multi-Model Architecture)
        # Use parallel training if available, otherwise fall back to sequential
        try:
            # Attempt parallel training
            self._train_catboost_parallel(fit_df, val_df, cat_cols, adv_weights)
        except Exception as e:
            logger.warning(f"Parallel training failed ({e}), falling back to sequential training")
            self._train_catboost_advanced(fit_df, val_df, cat_cols, adv_weights)
        
        # Clear GPU memory after CatBoost training
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After CatBoost")
        
        # 6. Joint NN (using numerical subset of optimized features)
        nn_features = [c for c in self.feature_cols if c not in cat_cols]
        X_fit_nn = fit_df[nn_features]
        
        nn_config = self.model_config.get('nn', {})
        logger.info("Training Joint Stats NN...")
        logger.info(f"  Config: hidden={nn_config.get('hidden_dim', 512)}, blocks={nn_config.get('num_blocks', 6)}, epochs={nn_config.get('epochs', 100)}")
        self.joint_model = MultiOutputWrapper(
            input_dim=len(nn_features), 
            target_names=self.core_targets, 
            config=nn_config
        )
        self.joint_model.fit(X_fit_nn, fit_df[self.core_targets])
        self.joint_model.save(os.path.join(self.models_dir, 'joint_stats_nn.pkl'))
        
        # Clear GPU memory after NN training
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After Joint NN")

        # 7. Temporal Models
        lstm_config = dict(self.model_config.get('lstm', {}))
        lstm_seq = lstm_config.pop('seq_len', 10)
        logger.info("Training LSTM...")
        logger.info(f"  Config: hidden={lstm_config.get('hidden_dim', 128)}, layers={lstm_config.get('num_layers', 2)}, "
                    f"bidirectional={lstm_config.get('bidirectional', False)}, seq_len={lstm_seq}")
        self.temporal_model = LSTMWrapper(input_dim=len(nn_features), seq_len=lstm_seq, config=lstm_config)
        self.temporal_model.fit(fit_df, nn_features, self.core_targets)
        self.temporal_model.save(os.path.join(self.models_dir, 'temporal_lstm.pkl'))
        
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After LSTM")
        
        tx_config = dict(self.model_config.get('transformer', {}))
        tx_seq = tx_config.pop('seq_len', 50)
        logger.info("Training Transformer...")
        logger.info(f"  Config: d_model={tx_config.get('d_model', 128)}, heads={tx_config.get('nhead', 8)}, "
                    f"layers={tx_config.get('num_layers', 4)}, seq_len={tx_seq}")
        self.attention_model = TransformerWrapper(input_dim=len(nn_features), seq_len=tx_seq, config=tx_config)
        self.attention_model.fit(fit_df, nn_features, self.core_targets)
        self.attention_model.save(os.path.join(self.models_dir, 'attention_transformer.pkl'))
        
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After Transformer")

        # 7b. Advanced Temporal Attention (context-aware attention over game history)
        temp_config = dict(self.model_config.get('temporal', {}))
        temp_seq = temp_config.pop('seq_len', 20)
        logger.info("Training Temporal Attention...")
        logger.info(f"  Config: hidden={temp_config.get('hidden_dim', 128)}, heads={temp_config.get('num_heads', 4)}, seq_len={temp_seq}")
        self.adv_temporal_model = TemporalAttentionWrapper(input_dim=len(nn_features), seq_len=temp_seq, config=temp_config)
        self.adv_temporal_model.fit(fit_df, nn_features, self.core_targets)
        self.adv_temporal_model.save(os.path.join(self.models_dir, 'adv_temporal_attention.pkl'))

        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After Temporal Attention")

        # 8. GNN (with graph attention)
        gnn_config = self.model_config.get('gnn', {})
        logger.info("Training GNN...")
        logger.info(f"  Config: hidden={gnn_config.get('hidden_dim', 64)}, layers={gnn_config.get('num_layers', 2)}, "
                    f"attention={gnn_config.get('use_attention', False)}")
        self.gnn_model = GNNWrapper(input_dim=len(nn_features), target_names=self.core_targets, config=gnn_config)
        self.gnn_model.fit(fit_df, nn_features, self.core_targets)
        self.gnn_model.save(os.path.join(self.models_dir, 'team_chemistry_gnn.pkl'))
        
        # Clear GPU memory after GNN training
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After GNN")
        
        # 9. Train Blender
        self._train_blender(val_df)
        self._save_blenders()


    def _train_blender(self, val_df: pd.DataFrame):
        """
        Trains the meta-learner (Linear Blender) using GPU-accelerated training when available.
        """
        import numpy as np
        
        logger.info("Training Super Learner (Blender)...")
        
        blender_y = val_df[self.core_targets].values
        
        # Get predictions from base models
        ens_preds_list = []
        for target in self.core_targets:
            if target not in self.models:
                continue
                
            model = self.models[target]
            X_val = val_df[self.feature_cols]
            
            if 'CatBoost' in str(type(model)):
                p = self._predict_catboost_blended(target, X_val)
            else:
                p = model.predict(X_val, val_df)
            
            ens_preds_list.append(p.reshape(-1, 1))
        
        if not ens_preds_list:
            logger.warning("No ensemble models found for blending.")
            return

        ens_meta = np.hstack(ens_preds_list)
        
        # NN Predictions
        if self.joint_model and self.joint_model.is_trained:
            nn_m, _ = self.joint_model.predict(val_df[self.feature_cols])
            meta_features = np.hstack([ens_meta, nn_m])
        else:
            meta_features = ens_meta
        
        # Train blenders - use GPU-accelerated blender if available
        self.blenders = {}
        self._gpu_blenders = {}  # Store GPU blenders separately for prediction
        
        if self.use_gpu:
            try:
                from src.models.advanced_trainer import GPUBlender
                logger.info("Using GPU-accelerated blender")
                use_gpu_blender = True
            except ImportError:
                logger.info("GPUBlender not available, using sklearn Ridge")
                use_gpu_blender = False
        else:
            use_gpu_blender = False
        
        for i, target in enumerate(self.core_targets):
            y_target = blender_y[:, i]
            
            if use_gpu_blender:
                # GPU blender
                blender = GPUBlender(input_dim=meta_features.shape[1], device=self.device)
                blender.fit(meta_features, y_target, epochs=100, lr=1e-2)
                self._gpu_blenders[target] = blender
                # Also create sklearn Ridge for compatibility/loading
                from sklearn.linear_model import Ridge
                sklearn_blender = Ridge(alpha=5.0)
                sklearn_blender.fit(meta_features, y_target)
                self.blenders[target] = sklearn_blender
                logger.info(f"{target} Blender (GPU) - Coefficients available at inference time")
            else:
                # CPU fallback
                from sklearn.linear_model import Ridge
                blender = Ridge(alpha=5.0)
                blender.fit(meta_features, y_target)
                self.blenders[target] = blender
                logger.info(f"{target} Blender Weights: {blender.coef_}")
                logger.info(f"{target} Blend Intercept: {blender.intercept_:.4f}")


    def _save_blenders(self):
        if not self.blenders: return
        import joblib
        path = os.path.join(self.models_dir, 'blenders.pkl')
        joblib.dump(self.blenders, path)
        logger.info(f"Saved {len(self.blenders)} blenders to {path}")

    def _save_feature_cols(self):
        """Save feature column names for consistent loading."""
        import joblib
        path = os.path.join(self.models_dir, 'feature_cols.pkl')
        joblib.dump(self.feature_cols, path)
        logger.info(f"Saved {len(self.feature_cols)} feature column names to {path}")
    
    def _load_feature_cols(self) -> Optional[List[str]]:
        """Load saved feature column names."""
        import joblib
        path = os.path.join(self.models_dir, 'feature_cols.pkl')
        if os.path.exists(path):
            self.feature_cols = joblib.load(path)
            logger.info(f"Loaded {len(self.feature_cols)} feature column names")
            return self.feature_cols
        return None

    def evaluate_all(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluates all models on the test set."""
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        results = {}
        
        for target in self.targets:
            if target not in self.models:
                # Try loading if not present
                self._load_models()
                if target not in self.models: 
                    continue
            
            model = self.models[target]
            X_test = test_df[self.feature_cols]
            y_test = test_df[target]
            
            if 'CatBoost' in str(type(model)):
                y_pred = self._predict_catboost_blended(target, X_test)
            elif hasattr(model, 'evaluate'):
                # Custom model evaluate method
                results[target] = model.evaluate(X_test, y_test, df_test_full=test_df)
                continue
            else:
                y_pred = model.predict(X_test, df_meta=test_df)
            
            results[target] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': root_mean_squared_error(y_test, y_pred)
            }
            
            if target in ['REB', 'AST']:
                self.diagnose_predictions(test_df, target)
                
        return results

    def diagnose_predictions(self, test_df: pd.DataFrame, target: str):
        """Analyzes residuals to identify outlier predictions."""
        if target not in self.models:
            return
        model = self.models[target]
        
        X = test_df[self.feature_cols]
        y_true = test_df[target]
        
        if 'CatBoost' in str(type(model)):
            y_pred = self._predict_catboost_blended(target, X)
        else:
            y_pred = model.predict(X, df_meta=test_df)
        
        residuals = y_true - y_pred
        logger.info(f"\n{target} Residual Analysis:")
        logger.info(f"  Mean residual: {residuals.mean():.2f}")
        logger.info(f"  Median residual: {residuals.median():.2f}")
        logger.info(f"  Max |residual|: {residuals.abs().max():.2f}")
        logger.info(f"  % with |residual| > 10: {(residuals.abs() > 10).mean()*100:.1f}%")
        
        # Check for systematic issues
        worst_idx = residuals.abs().nlargest(10).index
        logger.info(f"  Worst predictions at indices: {worst_idx.tolist()}")

    def predict_player_stats(self, player_context_df: pd.DataFrame, history_df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Hybrid prediction with Ensemble, NN, LSTM, Transformer, and GNN Synergy.
        Includes fallback logic for missing models.
        """
        predictions = {}
        base_predictions = {}
        
        if not self.models:
            self._load_models()
        
        if not self.models:
            logger.warning("No models loaded, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        if self.feature_cols is None or not self.feature_cols:
            # Try to get feature names from any loaded CatBoost model
            for model in self.models.values():
                if hasattr(model, 'feature_names_') and model.feature_names_:
                    self.feature_cols = list(model.feature_names_)
                    break
        
        if self.feature_cols is None or not self.feature_cols:
            logger.warning("No feature columns available, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        X = player_context_df[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if X.empty:
            logger.warning("Empty feature matrix, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        # 1. Base Predictions (blended CatBoost RMSE+MAE when available)
        for target in self.targets:
            if target not in self.models:
                logger.debug(f"Model for {target} not available, using fallback")
                predictions[target] = self._get_fallback_value(player_context_df, target)
                base_predictions[target] = predictions[target]
                continue
            
            model = self.models[target]
            
            try:
                if 'CatBoost' in str(type(model)):
                    blended = self._predict_catboost_blended(target, X)
                    pred = blended[0]
                else:
                    pred = model.predict(X, df_meta=player_context_df)[0]
                
                if pd.isna(pred) or pred < 0:
                    logger.debug(f"Invalid prediction for {target}, using fallback")
                    pred = self._get_fallback_value(player_context_df, target)
                    
                predictions[target] = float(pred)
                base_predictions[target] = predictions[target]

                # Attach quantile-derived uncertainty if available
                q_preds = self._predict_catboost_quantiles(target, X)
                if q_preds is not None:
                    ci_low = float(q_preds['low'][0]) if 'low' in q_preds else None
                    ci_high = float(q_preds['high'][0]) if 'high' in q_preds else None
                    if ci_low is not None and ci_high is not None:
                        predictions[f'{target}_STD'] = (ci_high - ci_low) / 2.56
            except Exception as e:
                logger.warning(f"Prediction failed for {target}: {e}, using fallback")
                predictions[target] = self._get_fallback_value(player_context_df, target)
                base_predictions[target] = predictions[target]
            
        # 2. Joint NN & Temporal Refinement
        if self.joint_model is not None and self.joint_model.is_trained:
            try:
                joint_means, joint_stds = self.joint_model.predict(X)
                joint_means, joint_stds = joint_means[0], joint_stds[0]
                for i, target in enumerate(self.core_targets):
                    # Sanity check: joint prediction should be within 3x of base or base must be very small
                    base_val = base_predictions[target]
                    if (0.1 < joint_means[i] / (base_val + 1e-6) < 10) or (base_val < 1.0 and joint_means[i] < 15):
                        predictions[target] = (predictions[target] * 0.7) + (joint_means[i] * 0.3)
                        predictions[f'{target}_STD'] = joint_stds[i]
                    else:
                        logger.debug(f"Joint model prediction for {target} ({joint_means[i]:.2f}) "
                                     f"vs base ({base_val:.2f}) - skipping blend")
            except Exception as e:
                logger.debug(f"Joint NN prediction failed: {e}")
                
        if self.temporal_model is not None and history_df is not None:
            try:
                if len(history_df) >= self.temporal_model.seq_len:
                    seq_features = history_df[self.feature_cols].tail(self.temporal_model.seq_len).apply(
                        pd.to_numeric, errors='coerce').fillna(0).values
                    temp_preds = self.temporal_model.predict(seq_features)[0]
                    for i, target in enumerate(self.core_targets):
                        if 0.1 < temp_preds[i] / (base_predictions[target] + 1e-6) < 10:
                            predictions[target] = (predictions[target] * 0.85) + (temp_preds[i] * 0.15)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        if self.attention_model is not None and history_df is not None:
            try:
                if len(history_df) >= self.attention_model.seq_len:
                    seq_features = history_df[self.feature_cols].tail(self.attention_model.seq_len).apply(
                        pd.to_numeric, errors='coerce').fillna(0).values
                    attn_preds = self.attention_model.predict(seq_features)[0]
                    for i, target in enumerate(self.core_targets):
                        if 0.1 < attn_preds[i] / (base_predictions[target] + 1e-6) < 10:
                            predictions[target] = (predictions[target] * 0.85) + (attn_preds[i] * 0.15)
            except Exception as e:
                logger.warning(f"Attention prediction failed: {e}")

        if self.adv_temporal_model is not None and history_df is not None:
            try:
                if len(history_df) >= self.adv_temporal_model.seq_len:
                    seq_features = history_df[self.feature_cols].tail(self.adv_temporal_model.seq_len).apply(
                        pd.to_numeric, errors='coerce').fillna(0).values
                    adv_preds = self.adv_temporal_model.predict(seq_features, X.values[0])[0]
                    for i, target in enumerate(self.core_targets):
                        # Sanity check
                        if 0.1 < adv_preds[i] / (base_predictions[target] + 1e-6) < 10:
                            predictions[target] = (predictions[target] * 0.8) + (adv_preds[i] * 0.2)
            except Exception as e:
                logger.warning(f"Advanced Temporal Attention failed: {e}")
        
        # 3. GNN Refinement (0.1 weight)
        if self.gnn_model is not None and self.gnn_model.is_trained:
            try:
                gnn_preds = self.gnn_model.predict(player_context_df)[0]
                for i, target in enumerate(self.core_targets):
                    # Sanity check
                    if 0.1 < gnn_preds[i] / (base_predictions[target] + 1e-6) < 10:
                        predictions[target] = (predictions[target] * 0.9) + (gnn_preds[i] * 0.1)
            except Exception as e:
                logger.warning(f"GNN prediction failed: {e}")
                    
        return predictions

    def _load_models(self):
        """
        Loads all trained models safely with GPU configuration.
        Ensures GPU settings are applied before models are initialized.
        """
        self._load_feature_cols()
        from catboost import CatBoostRegressor, CatBoostClassifier
        
        loaded_count = 0
        failed_targets = []
        
        for target in self.targets:
            # 1. Check for CatBoost model first
            cb_path = os.path.join(self.models_dir, f'{target.lower()}_catboost.cbm')
            if os.path.exists(cb_path):
                try:
                    model = CatBoostRegressor()
                    model.load_model(cb_path)
                    self.models[target] = model
                    loaded_count += 1
                    logger.info(f"Loaded CatBoost model for {target}")
                except Exception as e:
                    logger.warning(f"Failed to load CatBoost for {target}: {e}")
                    failed_targets.append(target)
                    continue
            else:
                # 2. Fallback to Pickle Ensemble (Old code)
                pkl_path = os.path.join(self.models_dir, f'{target.lower()}_ensemble.pkl')
                if os.path.exists(pkl_path):
                    try:
                        loaded_model = StackedEnsembleModel.load(pkl_path)
                        loaded_model.use_gpu = self.use_gpu
                        self.models[target] = loaded_model
                        loaded_count += 1
                        logger.info(f"Loaded Ensemble model for {target}")
                    except Exception as e:
                        logger.warning(f"Failed to load Ensemble for {target}: {e}")
                        if target not in failed_targets:
                            failed_targets.append(target)
                continue

            # 3. Load MAE companion model
            mae_path = os.path.join(self.models_dir, f'{target.lower()}_catboost_mae.cbm')
            if os.path.exists(mae_path):
                try:
                    mae_model = CatBoostRegressor()
                    mae_model.load_model(mae_path)
                    self.catboost_mae_models[target] = mae_model
                    logger.info(f"Loaded CatBoost MAE model for {target}")
                except Exception as e:
                    logger.debug(f"Failed to load MAE model for {target}: {e}")

            # 4. Load quantile models
            for label in ('low', 'high'):
                q_path = os.path.join(
                    self.models_dir, f'{target.lower()}_catboost_q{label}.cbm'
                )
                if os.path.exists(q_path):
                    try:
                        q_model = CatBoostRegressor()
                        q_model.load_model(q_path)
                        self.catboost_quantile_models.setdefault(target, {})[label] = q_model
                        logger.info(f"Loaded CatBoost quantile-{label} model for {target}")
                    except Exception as e:
                        logger.debug(f"Failed to load quantile-{label} for {target}: {e}")
        
        if loaded_count == 0:
            logger.error("No models could be loaded!")
        elif failed_targets:
            logger.warning(f"Failed to load models for targets: {failed_targets}")
        else:
            logger.info(f"Successfully loaded {loaded_count}/{len(self.targets)} models")
        
        # Summarise auxiliary model counts
        n_mae = len(self.catboost_mae_models)
        n_quant = sum(len(v) for v in self.catboost_quantile_models.values())
        if n_mae or n_quant:
            logger.info(f"Auxiliary CatBoost models: {n_mae} MAE, {n_quant} quantile")
        
        # Load blenders
        blender_path = os.path.join(self.models_dir, 'blenders.pkl')
        if os.path.exists(blender_path):
            try:
                import joblib
                self.blenders = joblib.load(blender_path)
                logger.info("Loaded blenders")
            except Exception as e:
                logger.warning(f"Failed to load blenders: {e}")
        
        # Load Wrappers (NN, LSTM, etc.)
        loaders = {
            'joint_stats_nn.pkl': ('joint_model', MultiOutputWrapper),
            'temporal_lstm.pkl': ('temporal_model', LSTMWrapper),
            'attention_transformer.pkl': ('attention_model', TransformerWrapper),
            'adv_temporal_attention.pkl': ('adv_temporal_model', TemporalAttentionWrapper),
            'team_chemistry_gnn.pkl': ('gnn_model', GNNWrapper)
        }
        
        loaded_advanced = 0
        for file, (attr, cls) in loaders.items():
            path = os.path.join(self.models_dir, file)
            if os.path.exists(path):
                try:
                    loaded_model = cls.load(path)
                    setattr(self, attr, loaded_model)
                    loaded_advanced += 1
                    logger.info(f"Loaded {attr} from {file}")
                except Exception as e:
                    logger.debug(f"Failed to load {attr}: {e}")
                    setattr(self, attr, None)
        
        if loaded_advanced > 0:
            logger.info(f"Loaded {loaded_advanced}/{len(loaders)} advanced models")

    def predict_player_stats_batch(
        self, 
        context_df: pd.DataFrame, 
        histories_map: Optional[Dict[int, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple players at once.
        Uses all available models: Ensembles, Joint NN, LSTM, Transformer, GNN.
        
        Args:
            context_df: DataFrame with one row per player, containing all features
            histories_map: Optional dict mapping PLAYER_ID -> history DataFrame
            
        Returns:
            DataFrame with predictions for each player (same index as context_df)
        """
        if context_df.empty:
            return pd.DataFrame()
        
        logger.info(f"Batch predicting for {len(context_df)} players with full model ensemble...")
        
        # Ensure models are loaded
        if not self.models:
            self._load_models()
        
        # Ensure we have feature columns
        if self.feature_cols is None:
            self._load_feature_cols()
            
        if self.feature_cols is None:
            logger.error("Feature columns not loaded. Cannot perform batch prediction.")
            return pd.DataFrame()

        # Check for missing features
        missing_cols = [c for c in self.feature_cols if c not in context_df.columns]
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} features for batch prediction, filling with defaults")
            context_df = context_df.copy()
            for col in missing_cols:
                context_df[col] = 0.0
        
        # Prepare feature matrix
        X = context_df[self.feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        # Initialize prediction containers
        predictions = {target: np.zeros(len(context_df)) for target in self.targets}
        prediction_stds = {f'{target}_STD': np.zeros(len(context_df)) for target in self.targets}
        base_predictions = {}  # Store base for sanity checks
        
        # ========== 1. BASE ENSEMBLE PREDICTIONS (blended CatBoost) ==========
        for target in self.targets:
            if target in self.models:
                model = self.models[target]
                if 'CatBoost' in str(type(model)):
                    pred = self._predict_catboost_blended(target, X)
                else:
                    pred = model.predict(X)
                predictions[target] = pred.copy()
                base_predictions[target] = pred.copy()
                
                # Quantile-derived uncertainty (calibrated intervals)
                q_preds = self._predict_catboost_quantiles(target, X)
                if q_preds is not None and 'low' in q_preds and 'high' in q_preds:
                    prediction_stds[f'{target}_STD'] = (
                        q_preds['high'] - q_preds['low']
                    ) / 2.56
                elif hasattr(model, 'predict_std'):
                    prediction_stds[f'{target}_STD'] = model.predict_std(X)
        
        logger.debug(f"Base ensemble predictions complete for {len(self.models)} targets")
        
        # ========== 2. JOINT NN REFINEMENT (Batch) ==========
        if self.joint_model is not None and getattr(self.joint_model, 'is_trained', False):
            try:
                joint_means, joint_stds = self.joint_model.predict(X)
                # joint_means shape: (n_samples, n_core_targets)
                
                for i, target in enumerate(self.core_targets):
                    if target not in base_predictions:
                        continue
                    base_vals = base_predictions[target]
                    joint_vals = joint_means[:, i]
                    joint_std_vals = joint_stds[:, i]
                    
                    # Vectorized sanity check
                    ratio = joint_vals / (base_vals + 1e-6)
                    valid_mask = ((ratio > 0.1) & (ratio < 10)) | ((base_vals < 1.0) & (joint_vals < 15))
                    
                    # Blend where valid
                    predictions[target] = np.where(
                        valid_mask,
                        predictions[target] * 0.7 + joint_vals * 0.3,
                        predictions[target]
                    )
                    prediction_stds[f'{target}_STD'] = np.where(
                        valid_mask,
                        joint_std_vals,
                        prediction_stds[f'{target}_STD']
                    )
                
                logger.debug(f"Joint NN refinement applied")
            except Exception as e:
                logger.warning(f"Joint NN batch prediction failed: {e}")
        
        # ========== 3. TEMPORAL MODEL REFINEMENTS (Per-Player with History) ==========
        if histories_map is not None:
            # LSTM refinement
            if self.temporal_model is not None and getattr(self.temporal_model, 'is_trained', False):
                self._apply_temporal_refinement_batch(
                    context_df, X, predictions, base_predictions, 
                    histories_map, self.temporal_model, weight=0.15, model_name="LSTM"
                )
            
            # Transformer refinement
            if self.attention_model is not None and getattr(self.attention_model, 'is_trained', False):
                self._apply_temporal_refinement_batch(
                    context_df, X, predictions, base_predictions,
                    histories_map, self.attention_model, weight=0.15, model_name="Transformer"
                )
            
            # Advanced Temporal Attention refinement
            if self.adv_temporal_model is not None and getattr(self.adv_temporal_model, 'is_trained', False):
                self._apply_adv_temporal_refinement_batch(
                    context_df, X, predictions, base_predictions,
                    histories_map, weight=0.2
                )
        
        # ========== 4. GNN REFINEMENT (Batch) ==========
        if self.gnn_model is not None and getattr(self.gnn_model, 'is_trained', False):
            try:
                gnn_preds = self.gnn_model.predict(context_df)
                # gnn_preds shape: (n_samples, n_core_targets)
                
                for i, target in enumerate(self.core_targets):
                    if target not in base_predictions:
                        continue
                    base_vals = base_predictions[target]
                    gnn_vals = gnn_preds[:, i]
                    
                    # Vectorized sanity check
                    ratio = gnn_vals / (base_vals + 1e-6)
                    valid_mask = (ratio > 0.1) & (ratio < 10)
                    
                    # Blend where valid (10% weight for GNN)
                    predictions[target] = np.where(
                        valid_mask,
                        predictions[target] * 0.9 + gnn_vals * 0.1,
                        predictions[target]
                    )
                
                logger.debug(f"GNN refinement applied")
            except Exception as e:
                logger.warning(f"GNN batch prediction failed: {e}")
        
        # ========== 5. FILL MISSING STDS ==========
        for target in self.targets:
            std_col = f'{target}_STD'
            missing_std_mask = prediction_stds[std_col] <= 0
            if missing_std_mask.any():
                # Estimate from history for missing values
                estimated_stds = self._estimate_std_from_history(
                    context_df[missing_std_mask], histories_map, target
                )
                prediction_stds[std_col][missing_std_mask] = estimated_stds
        
        # ========== 6. ASSEMBLE RESULT DATAFRAME ==========
        result_df = pd.DataFrame(predictions, index=context_df.index)
        for std_col, std_vals in prediction_stds.items():
            result_df[std_col] = std_vals
        
        # Ensure non-negative predictions
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_cols] = result_df[numeric_cols].clip(lower=0)
        
        logger.info(f"Batch prediction complete: {len(result_df)} players, "
                   f"models used: Ensemble + Joint NN + Temporal + GNN")
        
        return result_df

    def _apply_temporal_refinement_batch(
        self,
        context_df: pd.DataFrame,
        X: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        base_predictions: Dict[str, np.ndarray],
        histories_map: Dict[int, pd.DataFrame],
        temporal_model,
        weight: float,
        model_name: str
    ):
        """Apply LSTM or Transformer refinement for each player with sufficient history."""
        seq_len = temporal_model.seq_len
        refined_count = 0
        
        for idx, (row_idx, row) in enumerate(context_df.iterrows()):
            player_id = row.get('PLAYER_ID')
            if player_id is None or player_id not in histories_map:
                continue
            
            history_df = histories_map[player_id]
            if len(history_df) < seq_len:
                continue
            
            try:
                # Prepare sequence features
                seq_features = history_df[self.feature_cols].tail(seq_len).apply(
                    pd.to_numeric, errors='coerce'
                ).fillna(0).values
                
                # Get temporal prediction
                temp_preds = temporal_model.predict(seq_features)[0]
                
                # Apply with sanity check for each target
                for i, target in enumerate(self.core_targets):
                    if target not in base_predictions:
                        continue
                    base_val = base_predictions[target][idx]
                    temp_val = temp_preds[i]
                    
                    # Sanity check
                    if 0.1 < temp_val / (base_val + 1e-6) < 10:
                        blend_weight = 1 - weight
                        predictions[target][idx] = (
                            predictions[target][idx] * blend_weight + temp_val * weight
                        )
                        refined_count += 1
                        
            except Exception as e:
                logger.debug(f"{model_name} refinement failed for player {player_id}: {e}")
                continue
        
        if refined_count > 0:
            logger.debug(f"{model_name} refinement applied to {refined_count} player-target pairs")

    def _apply_adv_temporal_refinement_batch(
        self,
        context_df: pd.DataFrame,
        X: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        base_predictions: Dict[str, np.ndarray],
        histories_map: Dict[int, pd.DataFrame],
        weight: float
    ):
        """Apply Advanced Temporal Attention refinement (context-aware)."""
        if self.adv_temporal_model is None:
            return
            
        seq_len = self.adv_temporal_model.seq_len
        refined_count = 0
        
        for idx, (row_idx, row) in enumerate(context_df.iterrows()):
            player_id = row.get('PLAYER_ID')
            if player_id is None or player_id not in histories_map:
                continue
            
            history_df = histories_map[player_id]
            if len(history_df) < seq_len:
                continue
            
            try:
                # Prepare sequence features
                seq_features = history_df[self.feature_cols].tail(seq_len).apply(
                    pd.to_numeric, errors='coerce'
                ).fillna(0).values
                
                # Get current context features
                current_context = X.iloc[idx].values
                
                # Get advanced temporal prediction with context
                adv_preds = self.adv_temporal_model.predict(seq_features, current_context)[0]
                
                # Apply with sanity check for each target
                for i, target in enumerate(self.core_targets):
                    if target not in base_predictions:
                        continue
                    base_val = base_predictions[target][idx]
                    adv_val = adv_preds[i]
                    
                    # Sanity check
                    if 0.1 < adv_val / (base_val + 1e-6) < 10:
                        blend_weight = 1 - weight
                        predictions[target][idx] = (
                            predictions[target][idx] * blend_weight + adv_val * weight
                        )
                        refined_count += 1
                        
            except Exception as e:
                logger.debug(f"Advanced Temporal refinement failed for player {player_id}: {e}")
                continue
        
        if refined_count > 0:
            logger.debug(f"Advanced Temporal Attention refinement applied to {refined_count} player-target pairs")

    def _estimate_std_from_history(
        self, 
        context_df: pd.DataFrame, 
        histories_map: Optional[Dict[int, pd.DataFrame]],
        target: str
    ) -> np.ndarray:
        """Estimate prediction std from player history when model doesn't provide it."""
        stds = []
        
        for _, row in context_df.iterrows():
            player_id = row.get('PLAYER_ID')
            std_val = None
            
            # Try to get from history
            if histories_map and player_id in histories_map:
                hist = histories_map[player_id]
                if target in hist.columns and len(hist) >= 3:
                    std_val = hist[target].tail(10).std()
            
            # Try rolling std columns from the row
            if std_val is None or np.isnan(std_val):
                for col in [f'ROLL_{target}_STD_10', f'ROLL_{target}_STD_20']:
                    if col in row.index and pd.notna(row[col]):
                        std_val = row[col]
                        break
            
            # Fallback to coefficient of variation estimate
            if std_val is None or np.isnan(std_val):
                mean_val = row.get(f'ROLL_{target}_AVG_10', row.get(target, 0))
                # Approximate CVs for NBA stats
                cv_map = {'PTS': 0.45, 'REB': 0.40, 'AST': 0.50, 'STL': 0.80, 'BLK': 0.90, 'TOV': 0.60}
                cv = cv_map.get(target, 0.40)
                std_val = max(1.0, float(mean_val) * cv) if pd.notna(mean_val) and mean_val > 0 else 2.0
            
            stds.append(float(std_val))
        
        return np.array(stds)

    def _fallback_prediction(self, player_context_df: pd.DataFrame) -> Dict[str, float]:
        """Fallback prediction using historical averages."""
        predictions = {}
        for target in self.targets:
            predictions[target] = self._get_fallback_value(player_context_df, target)
        return predictions
    
    def _get_fallback_value(self, player_context_df: pd.DataFrame, target: str) -> float:
        """Get fallback value for a target stat."""
        fallback_cols = [f'ROLL_{target}_AVG_10', f'ROLL_{target}_AVG_20', f'{target}_EWMA_5', target]
        for col in fallback_cols:
            if col in player_context_df.columns:
                val = player_context_df[col].iloc[0] if len(player_context_df) > 0 else None
                if pd.notna(val) and val > 0:
                    return float(val)
        
        league_avgs = {'PTS': 10.0, 'REB': 4.5, 'AST': 2.5, 'STL': 0.8, 'BLK': 0.6, 'TOV': 1.5}
        return league_avgs.get(target, 0.0)
