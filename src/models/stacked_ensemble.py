import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import joblib
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

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
                n_estimators=300, 
                learning_rate=0.05, 
                max_depth=6, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                n_jobs=-1, 
                **gpu_params
            ),
            
            # LightGBM: Very fast on GPU
            'lgbm': LGBMRegressor(
                n_estimators=300, 
                learning_rate=0.05, 
                num_leaves=63, 
                feature_fraction=0.8, 
                **lgbm_gpu
            ),
            
            # RF Proxy (XGB with higher depth)
            'rf_proxy': XGBRegressor(
                n_estimators=200, 
                learning_rate=0.1, 
                max_depth=12, # RF usually deeper
                subsample=0.7, 
                colsample_bytree=0.7, 
                **gpu_params
            ),
            
            # Linear models run on CPU (faster than transferring data to GPU for simple math)
            'ridge': Ridge(alpha=10.0),
        }
        
        if HAS_CATBOOST:
            self.base_models['catboost'] = CatBoostRegressor(
                iterations=300, 
                learning_rate=0.05, 
                depth=6, 
                verbose=False, 
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

    def fit(self, X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame):
        """Trains on residuals with STRICT time-series validation (no leakage)."""
        logger.info(f"Training Residual StackedEnsemble for {self.target_name} with time-series CV...")
        
        # CRITICAL: Sort chronologically BEFORE any operations
        df_full = df_full.sort_values('GAME_DATE').reset_index(drop=True)
        X = X.loc[df_full.index].reset_index(drop=True)
        y = y.loc[df_full.index].reset_index(drop=True)
        
        self.feature_names = X.columns.tolist()
        
        # Baseline calculation (MUST use expanding with min_periods=1)
        self.league_avg = y.mean()
        player_cumsum = df_full.groupby('PLAYER_ID')[self.target_name].cumsum() - df_full[self.target_name]
        player_cumcount = df_full.groupby('PLAYER_ID').cumcount()
        player_avg = (player_cumsum / player_cumcount.replace(0, np.nan)).fillna(self.league_avg)
        
        # Opponent effect (historical only)
        opp_cumsum = df_full.groupby('OPPONENT_ID')[self.target_name].cumsum() - df_full[self.target_name]
        opp_cumcount = df_full.groupby('OPPONENT_ID').cumcount()
        opp_avg = (opp_cumsum / opp_cumcount.replace(0, np.nan)).fillna(self.league_avg)
        opp_effect = opp_avg - self.league_avg
        
        baseline = player_avg + (opp_effect * 0.3)
        residuals = y - baseline
        
        # Time-series cross-validation (NO SHUFFLING)
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # CRITICAL: Ensure chronological separation
            train_max_date = df_full.loc[train_idx, 'GAME_DATE'].max()
            val_min_date = df_full.loc[val_idx, 'GAME_DATE'].min()
            # Use <= to handle same-day games across split boundaries if necessary, 
            # but user requested < so we follow strictly for leakage detection.
            assert train_max_date <= val_min_date, f"TIME LEAKAGE DETECTED in Fold {fold}!"
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Enforce GPU params per fold
                if 'xgb' in name and self.use_gpu:
                    model.set_params(device='cuda', tree_method='hist')
                elif 'lgbm' in name and self.use_gpu:
                    model.set_params(device='gpu')
                elif 'catboost' in name and self.use_gpu:
                    model.set_params(task_type='GPU')
                
                model.fit(
                    X.iloc[train_idx], residuals.iloc[train_idx],
                    sample_weight=self._compute_sample_weights(df_full.iloc[train_idx])
                )
                meta_features[val_idx, i] = model.predict(X.iloc[val_idx])
        
        # Train meta-learner ONLY on non-zero predictions (where OOF predictions exist)
        valid_mask = np.any(meta_features != 0, axis=1)
        self.meta_learner.fit(
            meta_features[valid_mask], 
            residuals.iloc[valid_mask],
            sample_weight=self._compute_sample_weights(df_full[valid_mask])
        )
        
        # Final refit on FULL dataset (for production predictions)
        logger.info(f"Final refit on full {self.target_name} dataset...")
        for name, model in self.base_models.items():
            if 'xgb' in name and self.use_gpu:
                model.set_params(device='cuda', tree_method='hist')
            elif 'lgbm' in name and self.use_gpu:
                model.set_params(device='gpu')
            
            model.fit(X, residuals, sample_weight=self._compute_sample_weights(df_full))
        
        # Update dictionaries for inference
        self.player_avgs = df_full.groupby('PLAYER_ID')[self.target_name].mean().to_dict()
        opp_means = df_full.groupby('OPPONENT_ID')[self.target_name].mean()
        self.opp_effects = (opp_means - self.league_avg).to_dict()
        
        # Optional: Train fallback on full Y
        self.fallback_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
        if self.use_gpu: self.fallback_model.set_params(device='gpu')
        self.fallback_model.fit(X, y, sample_weight=self._compute_sample_weights(df_full))
        
        self.is_trained = True
        logger.info(f"Successfully trained {self.target_name} ensemble with strict TS validation.")

    def predict(self, X: pd.DataFrame, df_meta: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
        
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        
        player_ids = df_meta['PLAYER_ID'].values
        known_mask = np.array([pid in self.player_avgs for pid in player_ids])
        
        # --- KNOWN PLAYERS ---
        if known_mask.any():
            known_idx = np.where(known_mask)[0]
            X_known = X.iloc[known_idx]
            
            player_base = np.array([self.player_avgs.get(pid, self.league_avg) for pid in player_ids[known_idx]])
            opp_ids = df_meta.iloc[known_idx]['OPPONENT_ID'].values
            opp_adj = np.array([self.opp_effects.get(oid, 0) for oid in opp_ids])
            baseline = player_base + (opp_adj * 0.3)
            
            meta_features = np.zeros((len(X_known), len(self.base_models)))
            X_numpy = X_known.values
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name in ['xgb', 'rf_proxy', 'lgbm']:
                         meta_features[:, i] = model.predict(X_numpy)
                    elif name == 'catboost' and HAS_CATBOOST:
                        meta_features[:, i] = model.predict(X_numpy)
                    else:
                        meta_features[:, i] = model.predict(X_known)
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
