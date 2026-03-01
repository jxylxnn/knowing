"""Prediction service for NBA player stats."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from src.config import Config
from src.models.base import PredictionResult, ModelRegistry
from src.pipeline.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class PredictionService:
    """High-level prediction API with ensemble blending."""
    
    def __init__(
        self,
        config: Config,
        training_pipeline: Optional[TrainingPipeline] = None
    ):
        """Initialize prediction service."""
        self.config = config
        
        # Use provided pipeline or create new one
        if training_pipeline is not None:
            self.pipeline = training_pipeline
        else:
            from src.models.base import ModelRegistry
            registry = ModelRegistry(config.data.models_dir)
            self.pipeline = TrainingPipeline(config, registry)
            self.pipeline.load_models()
        
        # Get feature columns
        self.feature_cols = self.pipeline.feature_cols
        
        # Fallback values
        self._fallback_values = {
            'PTS': 10.0, 'REB': 4.0, 'AST': 2.0,
            'STL': 0.8, 'BLK': 0.5, 'TOV': 1.5
        }
    
    def predict_player(
        self,
        player_context_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None
    ) -> PredictionResult:
        """Predict stats for a single player."""
        if player_context_df is None or player_context_df.empty:
            raise ValueError("player_context_df must be a non-empty DataFrame")
        
        player_id = int(player_context_df['PLAYER_ID'].iloc[0])
        player_name = player_context_df.get('PLAYER_NAME', pd.Series(['Unknown'])).iloc[0]
        team = player_context_df.get('TEAM_ABBREVIATION', pd.Series(['UNK'])).iloc[0]
        opponent = player_context_df.get('OPPONENT_ABBR', pd.Series(['UNK'])).iloc[0]
        
        # Get predictions
        predictions_dict = self._predict_with_ensemble(player_context_df, history_df)
        
        # Extract uncertainties
        uncertainties = {}
        for target in self.config.training.targets:
            std_key = f'{target}_STD'
            if std_key in predictions_dict:
                uncertainties[target] = predictions_dict[std_key]
        
        # Build result
        result = PredictionResult(
            player_id=player_id,
            player_name=str(player_name),
            team=str(team),
            opponent=str(opponent),
            predictions={k: v for k, v in predictions_dict.items() 
                        if not k.endswith('_STD')},
            uncertainties=uncertainties
        )
        
        return result
    
    def predict_batch(
        self,
        contexts_df: pd.DataFrame,
        histories_map: Optional[Dict[int, pd.DataFrame]] = None
    ) -> List[PredictionResult]:
        """Predict stats for multiple players."""
        results = []
        
        for idx in range(len(contexts_df)):
            player_context = contexts_df.iloc[[idx]]
            player_id = int(player_context['PLAYER_ID'].iloc[0])
            
            # Get history if available
            history = None
            if histories_map and player_id in histories_map:
                history = histories_map[player_id]
            
            try:
                result = self.predict_player(player_context, history)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict for player {player_id}: {e}")
                results.append(self._create_fallback_result(player_context))
        
        return results
    
    def _predict_with_ensemble(
        self,
        player_context_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Predict using ensemble of models with blending."""
        predictions = {}
        base_predictions = {}
        
        # Check if models are loaded
        if not self.pipeline.models:
            logger.warning("No models loaded, using fallback predictions")
            return self._fallback_prediction(player_context_df)
        
        # Validate feature columns
        if self.feature_cols is None or not self.feature_cols:
            logger.warning("No feature columns available")
            return self._fallback_prediction(player_context_df)
        
        # Prepare features
        try:
            X = player_context_df[self.feature_cols].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(0)
            
            if X.empty:
                logger.warning("Empty feature matrix")
                return self._fallback_prediction(player_context_df)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return self._fallback_prediction(player_context_df)
        
        # 1. Base predictions from CatBoost models (blended RMSE+MAE)
        for target in self.config.training.targets:
            if target not in self.pipeline.models:
                pred = self._get_fallback_value(player_context_df, target)
                predictions[target] = pred
                base_predictions[target] = pred
                continue
            
            model = self.pipeline.models[target]
            
            try:
                # Use blended RMSE+MAE when MAE companion exists
                mae_models = getattr(self.pipeline, 'catboost_mae_models', {})
                if 'CatBoost' in str(type(model)) and target in mae_models:
                    rmse_pred = model.predict(X)
                    mae_pred = mae_models[target].predict(X)
                    w = self.config.catboost.multi_loss_rmse_weight
                    blended = rmse_pred * w + mae_pred * (1 - w)
                    pred = blended[0]
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)[0]
                else:
                    pred = model.predict(X, df_meta=player_context_df)[0]
                
                if pd.isna(pred) or pred < 0:
                    pred = self._get_fallback_value(player_context_df, target)
                
                predictions[target] = float(pred)
                base_predictions[target] = predictions[target]

                # Quantile-derived uncertainty
                q_models = getattr(self.pipeline, 'catboost_quantile_models', {})
                if target in q_models:
                    q = q_models[target]
                    if 'low' in q and 'high' in q:
                        ci_low = float(q['low'].predict(X)[0])
                        ci_high = float(q['high'].predict(X)[0])
                        predictions[f'{target}_STD'] = (ci_high - ci_low) / 2.56
                
            except Exception as e:
                logger.warning(f"Prediction failed for {target}: {e}")
                pred = self._get_fallback_value(player_context_df, target)
                predictions[target] = pred
                base_predictions[target] = pred
        
        # 2. Joint NN refinement
        if self.pipeline.joint_model is not None:
            try:
                if hasattr(self.pipeline.joint_model, 'is_trained'):
                    if self.pipeline.joint_model.is_trained:
                        joint_means, joint_stds = self.pipeline.joint_model.predict(X)
                        joint_means = joint_means[0]
                        joint_stds = joint_stds[0]
                        
                        core_targets = self.config.training.targets[:3]
                        for i, target in enumerate(core_targets):
                            base_val = base_predictions[target]
                            joint_val = joint_means[i]
                            
                            if (0.1 < joint_val / (base_val + 1e-6) < 10):
                                predictions[target] = base_val * 0.7 + joint_val * 0.3
                                predictions[f'{target}_STD'] = joint_stds[i]
            except Exception as e:
                logger.debug(f"Joint NN prediction failed: {e}")
        
        # 3. Temporal model refinement
        if self.pipeline.temporal_model is not None and history_df is not None:
            try:
                if hasattr(self.pipeline.temporal_model, 'seq_len'):
                    if len(history_df) >= self.pipeline.temporal_model.seq_len:
                        seq_features = history_df[self.feature_cols].tail(
                            self.pipeline.temporal_model.seq_len
                        ).apply(pd.to_numeric, errors='coerce').fillna(0).values
                        
                        temp_preds = self.pipeline.temporal_model.predict(seq_features)[0]
                        core_targets = self.config.training.targets[:3]
                        
                        for i, target in enumerate(core_targets):
                            base_val = base_predictions[target]
                            temp_val = temp_preds[i]
                            
                            if 0.1 < temp_val / (base_val + 1e-6) < 10:
                                predictions[target] = base_val * 0.85 + temp_val * 0.15
            except Exception as e:
                logger.debug(f"Temporal prediction failed: {e}")
        
        return predictions
    
    def _fallback_prediction(self, context_df: pd.DataFrame) -> Dict[str, float]:
        """Generate fallback predictions when models fail."""
        predictions = {}
        
        for target in self.config.training.targets:
            predictions[target] = self._get_fallback_value(context_df, target)
        
        return predictions
    
    def _get_fallback_value(self, context_df: pd.DataFrame, target: str) -> float:
        """Get fallback value for a target."""
        # Try rolling average
        rolling_col = f'ROLL_{target}_AVG_5'
        if rolling_col in context_df.columns:
            val = context_df[rolling_col].iloc[0]
            if pd.notna(val) and val > 0:
                return float(val)
        
        # Try season average
        season_col = f'SEASON_{target}'
        if season_col in context_df.columns:
            val = context_df[season_col].iloc[0]
            if pd.notna(val) and val > 0:
                return float(val)
        
        # Use default
        return self._fallback_values.get(target, 5.0)
    
    def _create_fallback_result(self, context_df: pd.DataFrame) -> PredictionResult:
        """Create a fallback PredictionResult."""
        player_id = int(context_df['PLAYER_ID'].iloc[0])
        player_name = context_df.get('PLAYER_NAME', pd.Series(['Unknown'])).iloc[0]
        team = context_df.get('TEAM_ABBREVIATION', pd.Series(['UNK'])).iloc[0]
        opponent = context_df.get('OPPONENT_ABBR', pd.Series(['UNK'])).iloc[0]
        
        predictions = self._fallback_prediction(context_df)
        
        return PredictionResult(
            player_id=player_id,
            player_name=str(player_name),
            team=str(team),
            opponent=str(opponent),
            predictions=predictions
        )
