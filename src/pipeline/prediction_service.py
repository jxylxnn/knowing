"""Prediction service for NBA player stats."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from src.config import Config
from src.models.base import PredictionResult, ModelRegistry
from src.training.pipeline import TrainingPipeline
from src.utils.prediction_utils import FeatureSelector, FeatureSchema

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
        self.feature_schema = getattr(self.pipeline, 'feature_schema', None)
        self.feature_selector = getattr(self.pipeline, 'feature_selector', FeatureSelector(config.training.targets))
        if self.feature_schema is None and self.feature_cols:
            self.feature_schema = FeatureSchema(
                feature_cols=list(self.feature_cols),
                categorical_cols=[c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] if c in self.feature_cols],
            )
            self.feature_selector.feature_schema = self.feature_schema
        self.blend_weights = getattr(self.pipeline, 'blend_weights', {})
        
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
        # Get player info
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
            X = self.feature_selector.transform(player_context_df, self.feature_schema, strict=False, fill_value=0.0)
            
            if X.empty:
                logger.warning("Empty feature matrix")
                return self._fallback_prediction(player_context_df)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return self._fallback_prediction(player_context_df)
        
        transformer_model = getattr(self.pipeline, 'transformer_model', None) or getattr(self.pipeline, 'attention_model', None)
        transformer_preds = None
        seq_features = None
        if transformer_model is not None and history_df is not None and self.feature_cols:
            seq_len = int(getattr(transformer_model, 'seq_len', 0) or 0)
            if seq_len > 0 and len(history_df) >= seq_len:
                seq_frame = self.feature_selector.transform(history_df, self.feature_schema, strict=False, fill_value=0.0)
                seq_features = seq_frame.tail(seq_len).values.astype(np.float32)
                try:
                    transformer_preds = transformer_model.predict(seq_features)[0]
                except Exception as e:
                    logger.debug(f"Transformer prediction failed: {e}")
                    transformer_preds = None

        # 1. Base predictions from CatBoost models
        for target in self.config.training.targets:
            if target not in self.pipeline.models:
                pred = self._get_fallback_value(player_context_df, target)
                predictions[target] = pred
                base_predictions[target] = pred
                continue
            
            model = self.pipeline.models[target]
            
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)[0]
                else:
                    pred = self._get_fallback_value(player_context_df, target)
                
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
                if transformer_preds is not None and target in self.config.training.targets:
                    blend_cfg = self.blend_weights.get(target, {})
                    cb_weight = float(blend_cfg.get('catboost', 1.0))
                    tx_weight = float(blend_cfg.get('transformer', 0.0))
                    tx_idx = self.config.training.targets.index(target)
                    tx_pred = float(transformer_preds[tx_idx])
                    predictions[target] = float(predictions[target] * cb_weight + tx_pred * tx_weight)
                    base_predictions[target] = predictions[target]
                    if f'{target}_STD' in predictions:
                        tx_uncertainty = float(blend_cfg.get('transformer_mae', predictions[f'{target}_STD']))
                        predictions[f'{target}_STD'] = float(
                            max(0.0, predictions[f'{target}_STD'] * cb_weight + tx_uncertainty * tx_weight)
                        )
                
            except Exception as e:
                logger.warning(f"Prediction failed for {target}: {e}")
                pred = self._get_fallback_value(player_context_df, target)
                predictions[target] = pred
                base_predictions[target] = pred
        
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
