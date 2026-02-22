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
        
        self.models: Dict[str, StackedEnsembleModel] = {}
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
        
        split_date = pd.to_datetime('2016-03-01')
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

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features, avoiding leakage."""
        # Dynamically select all numerical features, excluding IDs, dates, and targets
        exclude_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'OPPONENT_ID', 'OPPONENT_ABBR', 
            'WL', 'SEASON_ID', 'VIDEO_AVAILABLE'
        ] + self.targets + self.secondary_targets
        
        # Initial selection of candidate features
        feature_cols = [
            c for c in df.columns 
            if c not in exclude_cols and 
            (df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'float', 'int'])
        ]
        
        # DIAGNOSTIC: Check for leakage
        leaky_suspects = [c for c in feature_cols if any(t.lower() in c.lower() for t in self.targets)]
        if leaky_suspects:
            logger.warning(f"Potential leaky features detected: {leaky_suspects[:10]}...")
        
        # Explicitly exclude targets and their direct derivatives
        # We only keep features that are explicitly time-shifted (Rolling/EWMA), 
        # contextual, or forecast-based to ensure no leakage.
        safe_feature_cols = [
            c for c in feature_cols 
            if c.startswith('ROLL') or c.startswith('EWMA_') or c.startswith('VS_OPP_')
            or 'TREND' in c or 'BAYESIAN' in c or 'PROJ_' in c or 'PACE' in c
            or c in ['IS_HOME', 'REST_DAYS', 'IS_B2B', 'FATIGUE_SCORE', 'MONTH', 'DAY_OF_WEEK', 
                     'EXP_PACE', 'EXP_TEAM_PTS', 'EXP_GAME_TOTAL', 'BLOWOUT_RISK', 'CLOSE_GAME', 'EXP_MARGIN']
            or '_TE' in c or '_SHARE_' in c or 'ROLE_INDEX' in c
        ]
        feature_cols = safe_feature_cols

        # Filter out raw team stats if they exist (leaks)
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

        # A. Adversarial Validation
        adv_weights = self.advanced_trainer.perform_adversarial_validation(fit_df, val_df)
        
        # B. Feature Selection (using PTS as proxy)
        logger.info("Optimizing feature space for PTS...")
        optimized_features = self.advanced_trainer.select_best_features(fit_df[self.feature_cols], fit_df['PTS'])
        self.feature_cols = optimized_features
        logger.info(f"Training with optimized feature set size: {len(self.feature_cols)}")
        
        X_fit = fit_df[self.feature_cols]
        self._save_feature_cols()

        # 5. Train CatBoost Models
        from catboost import CatBoostRegressor
        cat_config = self.model_config.get('catboost', {})
        cat_iterations = cat_config.get('iterations', 2000)
        cat_depth = cat_config.get('depth', 8)
        cat_lr = cat_config.get('learning_rate', 0.03)
        cat_l2 = cat_config.get('l2_leaf_reg', 3)
        
        logger.info(f"CatBoost config: iterations={cat_iterations}, depth={cat_depth}, lr={cat_lr}")
        
        for target in self.targets:
            logger.info(f"Training Advanced CatBoost for: {target}")
            y = fit_df[target]
            
            task_type = "GPU" if self.use_gpu else "CPU"
            
            try:
                model = CatBoostRegressor(
                    iterations=cat_iterations,
                    learning_rate=cat_lr,
                    depth=cat_depth,
                    l2_leaf_reg=cat_l2,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    cat_features=[c for c in cat_cols if c in self.feature_cols],
                    verbose=200,
                    early_stopping_rounds=100,
                    task_type=task_type,
                    devices='0'
                )
                
                model.fit(
                    X_fit, y, 
                    sample_weight=adv_weights,
                    eval_set=(val_df[self.feature_cols], val_df[target]),
                    use_best_model=True
                )
            except Exception as e:
                logger.error(f"CatBoost GPU training failed for {target}: {e}. Falling back to CPU.")
                model = CatBoostRegressor(
                    iterations=cat_iterations,
                    learning_rate=cat_lr,
                    depth=cat_depth,
                    l2_leaf_reg=cat_l2,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    cat_features=[c for c in cat_cols if c in self.feature_cols],
                    verbose=200,
                    early_stopping_rounds=100,
                    task_type="CPU"
                )
                model.fit(
                    X_fit, y, 
                    eval_set=(val_df[self.feature_cols], val_df[target]),
                    use_best_model=True
                )
            
        self.models[target] = model 
        model.save_model(os.path.join(self.models_dir, f'{target.lower()}_catboost.cbm'))
        
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
        lstm_config = self.model_config.get('lstm', {})
        logger.info("Training LSTM...")
        logger.info(f"  Config: hidden={lstm_config.get('hidden_dim', 128)}, layers={lstm_config.get('num_layers', 2)}, bidirectional={lstm_config.get('bidirectional', False)}")
        self.temporal_model = LSTMWrapper(input_dim=len(nn_features), seq_len=10, config=lstm_config)
        self.temporal_model.fit(fit_df, nn_features, self.core_targets)
        self.temporal_model.save(os.path.join(self.models_dir, 'temporal_lstm.pkl'))
        
        # Clear GPU memory after LSTM training
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After LSTM")
        
        tx_config = self.model_config.get('transformer', {})
        logger.info("Training Transformer...")
        logger.info(f"  Config: d_model={tx_config.get('d_model', 128)}, heads={tx_config.get('nhead', 8)}, layers={tx_config.get('num_layers', 4)}")
        self.attention_model = TransformerWrapper(input_dim=len(nn_features), seq_len=50, config=tx_config)
        self.attention_model.fit(fit_df, nn_features, self.core_targets)
        self.attention_model.save(os.path.join(self.models_dir, 'attention_transformer.pkl'))
        
        # Clear GPU memory after Transformer training
        clear_gpu_memory()
        if self.use_gpu:
            log_gpu_memory("After Transformer")

        # 8. GNN
        gnn_config = self.model_config.get('gnn', {})
        logger.info("Training GNN...")
        logger.info(f"  Config: hidden={gnn_config.get('hidden_dim', 64)}, layers={gnn_config.get('num_layers', 2)}")
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
            
            # Check if the model is CatBoost. CatBoost does not accept 'df_meta' as a second arg.
            if 'CatBoost' in str(type(model)):
                p = model.predict(X_val)
            else:
                # Fallback for custom StackedEnsemble models
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
            
            # --- FIX START ---
            if 'CatBoost' in str(type(model)):
                y_pred = model.predict(X_test)
            elif hasattr(model, 'evaluate'):
                # Custom model evaluate method
                results[target] = model.evaluate(X_test, y_test, df_test_full=test_df)
                continue
            else:
                # Fallback
                y_pred = model.predict(X_test, df_meta=test_df)
            # --- FIX END ---
            
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
        
        # --- FIX START ---
        # Use self.feature_cols instead of model.feature_names.
        # This works for both CatBoost and StackedEnsemble, and ensures
        # we use the columns that were actually selected/optimized.
        X = test_df[self.feature_cols]
        y_true = test_df[target]
        
        # Handle the different predict signatures (CatBoost vs Custom Ensemble)
        if 'CatBoost' in str(type(model)):
            y_pred = model.predict(X)
        else:
            y_pred = model.predict(X, df_meta=test_df)
        # --- FIX END ---
        
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
        
        if 'PTS' not in self.models or not self.models:
            self._load_models()
        
        if not self.models:
            logger.warning("No models loaded, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        feature_cols = self.models['PTS'].feature_names if hasattr(self.models['PTS'], 'feature_names') else self.feature_cols
        
        if feature_cols is None or not feature_cols:
            logger.warning("No feature columns available, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        X = player_context_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if X.empty:
            logger.warning("Empty feature matrix, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        # 1. Base Predictions
        for target in self.targets:
            if target not in self.models:
                logger.debug(f"Model for {target} not available, using fallback")
                predictions[target] = self._get_fallback_value(player_context_df, target)
                base_predictions[target] = predictions[target]
                continue
            
            model = self.models[target]
            
            try:
                if 'CatBoost' in str(type(model)):
                    pred = model.predict(X)[0]
                else:
                    pred = model.predict(X, df_meta=player_context_df)[0]
                
                if pd.isna(pred) or pred < 0:
                    logger.debug(f"Invalid prediction for {target}, using fallback")
                    pred = self._get_fallback_value(player_context_df, target)
                    
                predictions[target] = float(pred)
                base_predictions[target] = predictions[target]
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
                    seq_features = history_df[feature_cols].tail(self.temporal_model.seq_len).apply(
                        pd.to_numeric, errors='coerce').fillna(0).values
                    temp_preds = self.temporal_model.predict(seq_features)[0]
                    for i, target in enumerate(self.core_targets):
                        # Sanity check
                        if 0.1 < temp_preds[i] / (base_predictions[target] + 1e-6) < 10:
                            predictions[target] = (predictions[target] * 0.85) + (temp_preds[i] * 0.15)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        if self.attention_model is not None and history_df is not None:
            try:
                if len(history_df) >= self.attention_model.seq_len:
                    seq_features = history_df[feature_cols].tail(self.attention_model.seq_len).apply(
                        pd.to_numeric, errors='coerce').fillna(0).values
                    attn_preds = self.attention_model.predict(seq_features)[0]
                    for i, target in enumerate(self.core_targets):
                        # Sanity check
                        if 0.1 < attn_preds[i] / (base_predictions[target] + 1e-6) < 10:
                            predictions[target] = (predictions[target] * 0.85) + (attn_preds[i] * 0.15)
            except Exception as e:
                logger.warning(f"Attention prediction failed: {e}")

        if self.adv_temporal_model is not None and history_df is not None:
            try:
                if len(history_df) >= self.adv_temporal_model.seq_len:
                    seq_features = history_df[feature_cols].tail(self.adv_temporal_model.seq_len).apply(
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
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load CatBoost for {target}: {e}")
                    failed_targets.append(target)
                
            # 2. Fallback to Pickle Ensemble (Old code)
            pkl_path = os.path.join(self.models_dir, f'{target.lower()}_ensemble.pkl')
            if os.path.exists(pkl_path):
                try:
                    loaded_model = StackedEnsembleModel.load(pkl_path)
                    loaded_model.use_gpu = self.use_gpu
                    self.models[target] = loaded_model
                    loaded_count += 1
                    logger.info(f"Loaded Ensemble model for {target} (GPU={'enabled' if self.use_gpu else 'disabled'})")
                except Exception as e:
                    logger.warning(f"Failed to load Ensemble for {target}: {e}")
                    if target not in failed_targets:
                        failed_targets.append(target)
        
        if loaded_count == 0:
            logger.error("No models could be loaded!")
        elif failed_targets:
            logger.warning(f"Failed to load models for targets: {failed_targets}")
        else:
            logger.info(f"Successfully loaded {loaded_count}/{len(self.targets)} models")
        
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
        
        # ========== 1. BASE ENSEMBLE PREDICTIONS ==========
        for target in self.targets:
            if target in self.models:
                model = self.models[target]
                pred = model.predict(X)
                predictions[target] = pred.copy()
                base_predictions[target] = pred.copy()
                
                # Get uncertainty from ensemble if available
                if hasattr(model, 'predict_std'):
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
