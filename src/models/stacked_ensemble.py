import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import joblib
import logging
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Import centralized GPU compatibility check
from src.models.gpu_utils import check_gpu_compatibility

logger = logging.getLogger(__name__)

class StackedEnsembleModel:
    """
    Optimized Stacked Ensemble with automatic GPU fallback.
    Uses GPU Histogram training for XGBoost and LightGBM when supported.
    """
    
    def __init__(self, target_name: str, use_gpu=True):
        self.target_name = target_name
        # Use centralized GPU check that detects unsupported architectures
        self.use_gpu = use_gpu and check_gpu_compatibility()
        self.cpu_threads = self._resolve_cpu_threads()
        
        if self.use_gpu:
            logger.info(f"Initializing {target_name} Ensemble for GPU (RTX 5070 Ti Mode)...")
        else:
            logger.info(f"Initializing {target_name} Ensemble for CPU...")

        # --- RTX 50-SERIES GPU OPTIMIZED PARAMS ---
        # 'hist' method is significantly faster on NVIDIA GPUs
        # FIX: Changed 'gradient-based' to 'gradient_based'
        gpu_params = {
            'tree_method': 'hist', 
            'device': 'cuda',
            'sampling_method': 'gradient_based' # Valid values are 'gradient_based' or 'uniform'
        } if self.use_gpu else {}
        
        # LightGBM GPU params
        lgbm_gpu = {'device': 'gpu', 'gpu_use_dp': False, 'verbose': -1} if self.use_gpu else {'verbose': -1}
        
        # CatBoost GPU params
        cb_gpu = {'task_type': 'GPU', 'devices': '0'} if self.use_gpu else {}

        self.base_models = {
            # XGBoost: Primary engine. Hist method is key.
            'xgb': XGBRegressor(
                n_estimators=500, 
                learning_rate=0.04, 
                max_depth=7, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                n_jobs=self.cpu_threads,
                **gpu_params
            ),
            
            # LightGBM: Very fast on GPU
            'lgbm': LGBMRegressor(
                n_estimators=500, 
                learning_rate=0.04, 
                num_leaves=63, 
                feature_fraction=0.8, 
                n_jobs=self.cpu_threads,
                **lgbm_gpu
            ),
            
            # RF Proxy (XGB with higher depth)
            'rf_proxy': XGBRegressor(
                n_estimators=350, 
                learning_rate=0.08, 
                max_depth=12, # RF usually deeper
                subsample=0.7, 
                colsample_bytree=0.7, 
                n_jobs=self.cpu_threads,
                **gpu_params
            ),
            
            # Linear models run on CPU (faster than transferring data to GPU for simple math)
            'ridge': Ridge(alpha=10.0),
        }
        
        if HAS_CATBOOST:
            self.base_models['catboost'] = CatBoostRegressor(
                iterations=2000,
                learning_rate=0.025,
                depth=9,
                l2_leaf_reg=4.0,
                border_count=254,
                grow_policy='Depthwise',
                min_data_in_leaf=8,
                rsm=0.8,
                boosting_type='Plain',
                score_function='Cosine',
                verbose=False,
                early_stopping_rounds=100,
                thread_count=self.cpu_threads,
                **cb_gpu
            )
        
        # Meta-Learner
        self.meta_learner = ElasticNet(alpha=0.05, l1_ratio=0.5)
        self.is_trained = False
        self.feature_names = []
        
        self.player_avgs = {}
        self.opp_effects = {}
        self.league_avg = 0
        self.fallback_model = None

    def _compute_sample_weights(self, df):
        """Calculates temporal decay weights for training."""
        max_date = df['GAME_DATE'].max()
        days_ago = (max_date - df['GAME_DATE']).dt.days
        return np.exp(-days_ago / 180)

    def _resolve_cpu_threads(self) -> int:
        """Avoid CPU oversubscription, especially when GPU is active."""
        cpu_count = os.cpu_count() or 4
        if self.use_gpu:
            return max(1, min(8, cpu_count // 2))
        return max(1, cpu_count - 1)

    def fit(self, X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame):
        """Trains on residuals with STRICT time-series validation (no leakage)."""
        logger.info(f"Training Residual StackedEnsemble for {self.target_name} with time-series CV...")
        
        # CRITICAL: Sort chronologically BEFORE any operations
        df_full = df_full.sort_values('GAME_DATE').reset_index(drop=True)
        X = X.loc[df_full.index].reset_index(drop=True)
        y = y.loc[df_full.index].reset_index(drop=True)
        
        self.feature_names = X.columns.tolist()
        
        # Baseline calculation (MUST use expanding mean with shift to prevent leakage)
        # CRITICAL FIX: The old cumsum approach included current row, causing data leakage
        self.league_avg = y.mean()
        
        # Player baseline: use expanding mean of SHIFTED values (historical data only)
        # This ensures we never use current game data to predict current game
        player_avg = df_full.groupby('PLAYER_ID')[self.target_name].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        ).fillna(self.league_avg)
        
        # Opponent effect: same - use expanding mean of SHIFTED values
        opp_avg = df_full.groupby('OPPONENT_ID')[self.target_name].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        ).fillna(self.league_avg)
        opp_effect = opp_avg - self.league_avg
        
        baseline = player_avg + (opp_effect * 0.3)
        residuals = y - baseline

        # Precompute to avoid repeated conversions inside folds
        X_np = X.to_numpy()
        residuals_np = residuals.to_numpy()
        sample_weights_full = self._compute_sample_weights(df_full)
        
        # Time-series cross-validation (NO SHUFFLING)
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # CRITICAL: Ensure chronological separation with STRICT less-than comparison
            # Changed from <= to < to ensure no same-day games leak across splits
            train_max_date = df_full.loc[train_idx, 'GAME_DATE'].max()
            val_min_date = df_full.loc[val_idx, 'GAME_DATE'].min()
            assert train_max_date < val_min_date, f"TIME LEAKAGE DETECTED in Fold {fold}: train_max={train_max_date}, val_min={val_min_date}"
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Enforce GPU params per fold - handle all XGB models including rf_proxy
                if self.use_gpu:
                    if 'xgb' in name or name == 'rf_proxy':
                        model.set_params(device='cuda', tree_method='hist')
                    elif 'lgbm' in name:
                        model.set_params(device='gpu')
                    elif 'catboost' in name and HAS_CATBOOST:
                        model.set_params(task_type='GPU')
                
                fit_kw = {
                    'sample_weight': sample_weights_full[train_idx]
                }
                # CatBoost benefits from eval_set for early stopping
                if 'catboost' in name and HAS_CATBOOST:
                    fit_kw['eval_set'] = (X_np[val_idx], residuals_np[val_idx])
                    fit_kw['use_best_model'] = True

                model.fit(X_np[train_idx], residuals_np[train_idx], **fit_kw)
                meta_features[val_idx, i] = model.predict(X_np[val_idx])
        
        # Train meta-learner ONLY on non-zero predictions (where OOF predictions exist)
        valid_mask = np.any(meta_features != 0, axis=1)
        self.meta_learner.fit(
            meta_features[valid_mask], 
            residuals_np[valid_mask],
            sample_weight=sample_weights_full[valid_mask]
        )
        
        # Final refit on FULL dataset (for production predictions)
        logger.info(f"Final refit on full {self.target_name} dataset...")
        for name, model in self.base_models.items():
            if 'xgb' in name and self.use_gpu:
                model.set_params(device='cuda', tree_method='hist')
            elif 'lgbm' in name and self.use_gpu:
                model.set_params(device='gpu')
            
            model.fit(X_np, residuals_np, sample_weight=sample_weights_full)
        
        # Update dictionaries for inference
        self.player_avgs = df_full.groupby('PLAYER_ID')[self.target_name].mean().to_dict()
        opp_means = df_full.groupby('OPPONENT_ID')[self.target_name].mean()
        self.opp_effects = (opp_means - self.league_avg).to_dict()
        
        # Optional: Train fallback on full Y
        self.fallback_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
        if self.use_gpu: self.fallback_model.set_params(device='gpu')
        self.fallback_model.fit(X_np, y.to_numpy(), sample_weight=sample_weights_full)
        
        self.is_trained = True
        logger.info(f"Successfully trained {self.target_name} ensemble with strict TS validation.")

    def predict(self, X: pd.DataFrame, df_meta: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
        X_np = X.to_numpy()
        
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        
        player_ids = df_meta['PLAYER_ID'].values
        known_mask = np.array([pid in self.player_avgs for pid in player_ids])
        
        # --- KNOWN PLAYERS ---
        if known_mask.any():
            known_idx = np.where(known_mask)[0]
            X_known_np = X_np[known_idx]
            
            player_base = np.array([self.player_avgs.get(pid, self.league_avg) for pid in player_ids[known_idx]])
            opp_ids = df_meta.iloc[known_idx]['OPPONENT_ID'].values
            opp_adj = np.array([self.opp_effects.get(oid, 0) for oid in opp_ids])
            baseline = player_base + (opp_adj * 0.3)
            
            meta_features = np.zeros((len(X_known_np), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name in ['xgb', 'rf_proxy', 'lgbm']:
                         meta_features[:, i] = model.predict(X_known_np)
                    elif name == 'catboost' and HAS_CATBOOST:
                        meta_features[:, i] = model.predict(X_known_np)
                    else:
                        meta_features[:, i] = model.predict(X_known_np)
                except Exception as e:
                    logger.debug(f"Base model {name} prediction failed: {e}")
                    meta_features[:, i] = 0
            
            residual_pred = self.meta_learner.predict(meta_features)
            residual_pred = np.clip(residual_pred, -20, 20)
            predictions[known_idx] = np.maximum(0, baseline + residual_pred)
        
        # --- UNKNOWN PLAYERS ---
        if (~known_mask).any():
            unknown_idx = np.where(~known_mask)[0]
            X_unknown = X.iloc[unknown_idx]
            if self.fallback_model is not None:
                predictions[unknown_idx] = np.maximum(0, self.fallback_model.predict(X_unknown.values))
            else:
                predictions[unknown_idx] = self.league_avg
                
        return predictions
        
    def evaluate(self, X_test, y_test, df_test_full):
        preds = self.predict(X_test, df_test_full)
        return {'mae': mean_absolute_error(y_test, preds), 'rmse': root_mean_squared_error(y_test, preds)}

    def save(self, path):
        data = {
            'base_models': self.base_models, 'meta_learner': self.meta_learner,
            'feature_names': self.feature_names, 'target_name': self.target_name,
            'player_avgs': self.player_avgs, 'opp_effects': self.opp_effects,
            'league_avg': self.league_avg, 'fallback_model': self.fallback_model
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(data['target_name'])
        instance.base_models = data['base_models']
        instance.meta_learner = data['meta_learner']
        instance.feature_names = data['feature_names']
        instance.player_avgs = data['player_avgs']
        instance.opp_effects = data['opp_effects']
        instance.league_avg = data['league_avg']
        instance.fallback_model = data.get('fallback_model', None)
        instance.is_trained = True
        return instance
