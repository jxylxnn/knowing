"""Active model manager for the CatBoost + Transformer prediction stack.

This class is kept as the bridge for the simulator and query CLI, but the old
joint/LSTM/GNN/stacked-ensemble paths have been removed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config.model_config import get_model_config, normalize_model_size
from src.models.base import ModelRegistry
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.training.catboost_trainer import CatBoostTrainer

logger = logging.getLogger(__name__)


def _load_transformer_wrapper():
    """Import the Transformer wrapper lazily."""
    from src.models.transformer_model import TransformerWrapper

    return TransformerWrapper


class ModelManager:
    """Bridge between saved models and the live simulator/query code."""

    TARGETS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    CORE_TARGETS = ['PTS', 'REB', 'AST']

    _FALLBACK_VALUES = {
        'PTS': 10.0,
        'REB': 4.5,
        'AST': 2.5,
        'STL': 0.8,
        'BLK': 0.6,
        'TOV': 1.5,
    }

    def __init__(
        self,
        data_dir: str = 'data',
        models_dir: str = 'models',
        model_size: str = 'M',
        model_config: Optional[Dict[str, Any]] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        if not isinstance(data_dir, str) or not data_dir:
            raise ValueError(f"Invalid data_dir: {data_dir}")
        if not isinstance(models_dir, str) or not models_dir:
            raise ValueError(f"Invalid models_dir: {models_dir}")

        self.data_dir = data_dir
        self.models_dir = models_dir
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        self.core_targets = list(self.CORE_TARGETS)
        self.secondary_targets = ['STL', 'BLK', 'TOV']
        self.targets = list(self.TARGETS)

        self.models: Dict[str, Any] = {}
        self.catboost_mae_models: Dict[str, Any] = {}
        self.catboost_quantile_models: Dict[str, Dict[str, Any]] = {}
        self.transformer_model: Optional[Any] = None
        self.blend_weights: Dict[str, Dict[str, float]] = {}
        self.feature_cols: Optional[List[str]] = None
        self.cat_features: List[str] = []

        self.use_gpu = False
        self.device = None
        self.feature_engineer = FeatureEngineer(use_gpu=False)

        if model_config is not None:
            self.model_config = model_config
            self.hw_info = model_config.get('metadata', {})
        else:
            normalized_size = normalize_model_size(model_size)
            self.model_config, self.hw_info = get_model_config(
                force_size=normalized_size if normalized_size is not None else 'M'
            )

        self.registry = registry or ModelRegistry(Path(self.models_dir))
        logger.info(
            "ModelManager initialized (data_dir=%s, models_dir=%s, tier=%s)",
            self.data_dir,
            self.models_dir,
            self.hw_info.get('tier', model_size),
        )

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSVs and build the feature-engineered train/test split."""
        data_dir = Path(self.data_dir)
        players_file = data_dir / 'nba_players.csv'
        games_file = data_dir / 'nba_games.csv'

        if not players_file.exists():
            raise ValueError(f"Players file not found: {players_file}")
        if not games_file.exists():
            raise ValueError(f"Games file not found: {games_file}")

        loader = DataLoader(str(players_file), str(games_file))
        merged_df = loader.merge_datasets()
        if merged_df.empty:
            raise ValueError("Merged dataset is empty after loading")

        full_df = self.feature_engineer.create_features(merged_df)
        if full_df.empty:
            raise ValueError("Feature engineering resulted in empty dataset")

        required_cols = ['PLAYER_ID', 'GAME_DATE'] + self.core_targets
        missing_cols = [c for c in required_cols if c not in full_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after feature engineering: {missing_cols}")

        split_date_str = self.model_config.get('training', {}).get('test_split_date', '2024-03-01')
        split_date = pd.to_datetime(split_date_str)
        train_df = full_df[full_df['GAME_DATE'] < split_date].copy()
        test_df = full_df[full_df['GAME_DATE'] >= split_date].copy()

        if train_df.empty:
            raise ValueError("Training set is empty after split")
        if test_df.empty:
            raise ValueError("Test set is empty after split")

        logger.info("Train set: %s, Test set: %s", len(train_df), len(test_df))
        return train_df, test_df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select leakage-safe numeric features."""
        targets = set(self.targets)
        exclude = {
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'OPPONENT_ID', 'OPPONENT_ABBR',
            'WL', 'SEASON_ID', 'VIDEO_AVAILABLE', 'REST_BUCKET',
        }
        exclude.update(targets)

        safe_prefixes = ('ROLL_', 'EWMA_', 'VS_OPP_', 'PROJ_', 'LEAGUE_PCT_')
        safe_substrings = (
            'TREND', 'BAYESIAN', 'PACE', '_TE', '_SHARE_', 'ROLE_INDEX',
            'SEASON_AVG', 'SEASON_SIN', 'SEASON_COS', 'HOT_STREAK', 'COLD_STREAK',
            'POTENTIAL', 'B2B_IMPACT', 'FATIGUE', 'EFF_Z_SCORE', 'FANTASY',
            'SOS_', 'PACE_ADJ', 'DEF_MATCHUP', 'OPP_DEF',
        )
        safe_exact = {
            'IS_HOME', 'REST_DAYS', 'IS_B2B', 'FATIGUE_SCORE', 'MONTH', 'DAY_OF_WEEK',
            'EXP_PACE', 'EXP_TEAM_PTS', 'EXP_GAME_TOTAL', 'BLOWOUT_RISK',
            'CLOSE_GAME', 'EXP_MARGIN', 'DAYS_SINCE_LAST', 'MINS_LAST_3',
            'MINS_LAST_7', 'EST_POSS', 'TEAM_PACE_10', 'PACE_FACTOR',
            'STAR_TEAMMATE_OUT',
        }

        features: List[str] = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype not in ('int64', 'float64', 'int32', 'float32'):
                continue
            if col in safe_exact or any(col.startswith(p) for p in safe_prefixes) or any(s in col for s in safe_substrings):
                features.append(col)
                continue
            if (
                col.startswith('STATS_') or col.startswith('ADV_')
                or col.startswith('MATCHUP_') or col.startswith('PACE_')
                or col.startswith('CONTEXT_')
            ):
                features.append(col)

        if not features:
            numeric_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            features = numeric_cols

        return features

    def _load_feature_cols(self) -> Optional[List[str]]:
        """Load saved feature column names from disk."""
        path = Path(self.models_dir) / 'feature_cols.pkl'
        if not path.exists():
            return None

        try:
            self.feature_cols = joblib.load(path)
            self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] if c in self.feature_cols]
            logger.info("Loaded %s feature columns", len(self.feature_cols))
        except Exception as exc:
            logger.warning("Failed to load feature columns: %s", exc)
            self.feature_cols = None

        return self.feature_cols

    def _load_models(self) -> Dict[str, int]:
        """Load CatBoost and Transformer models from disk."""
        self._load_feature_cols()
        self.models = {}
        self.catboost_mae_models = {}
        self.catboost_quantile_models = {}
        self.transformer_model = None

        counts = {'catboost': 0, 'transformer': 0, 'failed': 0}

        for target in self.targets:
            try:
                trainer = CatBoostTrainer.load(self.models_dir, target)
            except Exception as exc:
                logger.debug("Failed to load CatBoost trainer for %s: %s", target, exc)
                counts['failed'] += 1
                continue

            if trainer.primary_model is not None:
                self.models[target] = trainer.primary_model
                counts['catboost'] += 1

            if trainer.mae_model is not None:
                self.catboost_mae_models[target] = trainer.mae_model

            if trainer.quantile_low_model is not None or trainer.quantile_high_model is not None:
                self.catboost_quantile_models[target] = {
                    k: v for k, v in {
                        'low': trainer.quantile_low_model,
                        'high': trainer.quantile_high_model,
                    }.items() if v is not None
                }

        if self.feature_cols is None and self.models:
            for model in self.models.values():
                if hasattr(model, 'feature_names_') and model.feature_names_:
                    self.feature_cols = list(model.feature_names_)
                    self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] if c in self.feature_cols]
                    break

        transformer_path = Path(self.models_dir) / 'attention_transformer.pkl'
        if transformer_path.exists():
            try:
                TransformerWrapper = _load_transformer_wrapper()
                self.transformer_model = TransformerWrapper.load(str(transformer_path))
                counts['transformer'] = 1
                logger.info("Loaded Transformer model")
            except Exception as exc:
                logger.warning("Failed to load Transformer model: %s", exc)

        blend_path = Path(self.models_dir) / 'blend_weights.pkl'
        if blend_path.exists():
            try:
                self.blend_weights = joblib.load(blend_path)
            except Exception as exc:
                logger.warning("Failed to load blend weights: %s", exc)

        logger.info(
            "Loaded %s CatBoost models, %s MAE companions, %s quantile sets",
            counts['catboost'],
            len(self.catboost_mae_models),
            len(self.catboost_quantile_models),
        )
        return counts

    def load_models(self) -> Dict[str, int]:
        """Public wrapper for compatibility with older callers."""
        return self._load_models()

    def _predict_catboost_quantiles(self, target: str, X: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        """Predict quantile bounds when available."""
        if target not in self.catboost_quantile_models:
            return None

        q_models = self.catboost_quantile_models[target]
        preds: Dict[str, np.ndarray] = {}
        if 'low' in q_models:
            preds['low'] = np.asarray(q_models['low'].predict(X), dtype=float)
        if 'high' in q_models:
            preds['high'] = np.asarray(q_models['high'].predict(X), dtype=float)
        return preds or None

    def _predict_catboost_target(self, target: str, X: pd.DataFrame) -> np.ndarray:
        """Predict a single target with optional MAE companion blending."""
        model = self.models[target]
        primary = np.asarray(model.predict(X), dtype=float)

        mae_model = self.catboost_mae_models.get(target)
        if mae_model is None:
            return np.clip(primary, 0.0, None)

        mae_pred = np.asarray(mae_model.predict(X), dtype=float)
        blended = 0.7 * primary + 0.3 * mae_pred
        return np.clip(blended, 0.0, None)

    def _predict_transformer_target(
        self,
        target: str,
        history_df: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Predict one target from the sequence model, if it is loaded."""
        if self.transformer_model is None or history_df is None:
            return None
        if self.feature_cols is None or not self.feature_cols:
            return None

        seq_len = int(getattr(self.transformer_model, 'seq_len', 0) or 0)
        if seq_len <= 0 or len(history_df) < seq_len:
            return None

        seq_df = history_df.reindex(columns=self.feature_cols, fill_value=0)
        seq = seq_df.tail(seq_len).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

        try:
            preds = self.transformer_model.predict(seq)
        except Exception as exc:
            logger.debug("Transformer prediction failed for %s: %s", target, exc)
            return None

        if preds is None:
            return None

        preds = np.asarray(preds, dtype=float)
        if preds.ndim == 2:
            preds = preds[0]

        target_idx = self.targets.index(target) if target in self.targets else None
        if target_idx is None or target_idx >= len(preds):
            return None
        return float(preds[target_idx])

    def predict_player_stats(self, player_context_df: pd.DataFrame, history_df: pd.DataFrame = None) -> Dict[str, float]:
        """Predict a single player's stat line using the active model stack."""
        if player_context_df is None or player_context_df.empty:
            return self._fallback_prediction(pd.DataFrame())

        if not self.models:
            self._load_models()

        if self.feature_cols is None or not self.feature_cols:
            self._load_feature_cols()

        if self.feature_cols is None or not self.feature_cols:
            return self._fallback_prediction(player_context_df)

        context = player_context_df.reindex(columns=self.feature_cols, fill_value=0)
        context = context.apply(pd.to_numeric, errors='coerce').fillna(0)

        predictions: Dict[str, float] = {}
        base_predictions: Dict[str, float] = {}

        for target in self.targets:
            if target not in self.models:
                value = self._get_fallback_value(player_context_df, target)
                predictions[target] = value
                base_predictions[target] = value
                continue

            try:
                base = float(self._predict_catboost_target(target, context)[0])
            except Exception as exc:
                logger.warning("CatBoost prediction failed for %s: %s", target, exc)
                base = self._get_fallback_value(player_context_df, target)

            base_predictions[target] = base
            final_pred = base

            transformer_pred = self._predict_transformer_target(target, history_df)
            if transformer_pred is not None:
                blend_cfg = self.blend_weights.get(target, {})
                cb_weight = float(blend_cfg.get('catboost', 0.7))
                tx_weight = float(blend_cfg.get('transformer', 0.3))
                final_pred = (base * cb_weight) + (transformer_pred * tx_weight)

            predictions[target] = float(max(0.0, final_pred))

            q_preds = self._predict_catboost_quantiles(target, context)
            if q_preds and 'low' in q_preds and 'high' in q_preds:
                low = float(q_preds['low'][0])
                high = float(q_preds['high'][0])
                predictions[f'{target}_STD'] = max(0.0, (high - low) / 2.56)

        return predictions

    def predict_player_stats_batch(
        self,
        context_df: pd.DataFrame,
        histories_map: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Predict stats for multiple players and return a DataFrame."""
        if context_df is None or context_df.empty:
            return pd.DataFrame()

        rows: List[Dict[str, float]] = []
        for idx in range(len(context_df)):
            row = context_df.iloc[[idx]]
            player_id = int(row['PLAYER_ID'].iloc[0]) if 'PLAYER_ID' in row.columns else idx
            history_df = histories_map.get(player_id) if histories_map else None
            try:
                rows.append(self.predict_player_stats(row, history_df))
            except Exception as exc:
                logger.warning("Prediction failed for player %s: %s", player_id, exc)
                rows.append(self._fallback_prediction(row))

        return pd.DataFrame(rows, index=context_df.index)

    def _fallback_prediction(self, player_context_df: pd.DataFrame) -> Dict[str, float]:
        """Fallback prediction using historical averages or league defaults."""
        predictions: Dict[str, float] = {}
        for target in self.targets:
            predictions[target] = self._get_fallback_value(player_context_df, target)
        return predictions

    def _get_fallback_value(self, player_context_df: pd.DataFrame, target: str) -> float:
        """Get a fallback value for a single stat."""
        fallback_cols = [f'ROLL_{target}_AVG_10', f'ROLL_{target}_AVG_20', f'{target}_EWMA_5', target]
        for col in fallback_cols:
            if col in player_context_df.columns and len(player_context_df) > 0:
                val = player_context_df[col].iloc[0]
                if pd.notna(val) and float(val) > 0:
                    return float(val)
        return float(self._FALLBACK_VALUES.get(target, 0.0))
