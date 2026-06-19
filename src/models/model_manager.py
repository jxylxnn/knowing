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
from src.contracts.artifacts import ArtifactContract, validate_runtime_artifacts
from src.contracts.features import align_feature_frame, load_expected_feature_cols
from src.correction.confidence_scorer import ConfidenceScorer
from src.correction.correction_applier import CorrectionApplier
from src.correction.correction_features import CorrectionFeatureBuilder
from src.correction.interval_store import CalibrationIntervalStore
from src.correction.residual_model import ResidualCorrectionModel
from src.evaluation.weight_store import EnsembleWeights, TargetBlend, WeightStore
from src.models.base import (
    ModelRegistry,
    collect_quantile_dict,
    load_blend_weights_from_disk,
    load_transformer_from_disk,
    validate_blend_contract,
)
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.training.catboost_trainer import CatBoostTrainer
from src.utils.prediction_utils import FeatureSelector, FeatureSchema

logger = logging.getLogger(__name__)


class ModelManager:
    """Bridge between saved models and the live simulator/query code."""

    TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    CORE_TARGETS = ["PTS", "REB", "AST"]

    _FALLBACK_VALUES = {
        "PTS": 10.0,
        "REB": 4.5,
        "AST": 2.5,
        "STL": 0.8,
        "BLK": 0.6,
        "TOV": 1.5,
    }

    def __init__(
        self,
        data_dir: str = "data",
        models_dir: str = "models",
        model_size: str = "M",
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
        self.secondary_targets = ["STL", "BLK", "TOV"]
        self.targets = list(self.TARGETS)

        self.models: Dict[str, Any] = {}
        self.catboost_mae_models: Dict[str, Any] = {}
        self.catboost_quantile_models: Dict[str, Dict[str, Any]] = {}
        self.transformer_model: Optional[Any] = None
        self.blend_weights: Dict[str, Dict[str, float]] = {}
        self.ensemble_weights: Optional[EnsembleWeights] = None  # v2 versioned weights
        self.feature_cols: Optional[List[str]] = None
        self.cat_features: List[str] = []
        self.feature_schema: Optional[FeatureSchema] = None

        self.use_gpu = False
        self.device = None
        self.feature_engineer = FeatureEngineer(cache_dir="cache/training")
        self.feature_selector = FeatureSelector(self.targets)

        self.residual_correction_model: Optional[ResidualCorrectionModel] = None
        self.correction_applier: Optional[CorrectionApplier] = None
        self.residual_corrections_enabled: bool = False
        self.calibration_interval_store: Optional[CalibrationIntervalStore] = None
        self.confidence_scorer = ConfidenceScorer()

        if model_config is not None:
            self.model_config = model_config
            self.hw_info = model_config.get("metadata", {})
        else:
            normalized_size = normalize_model_size(model_size)
            self.model_config, self.hw_info = get_model_config(
                force_size=normalized_size if normalized_size is not None else "M"
            )

        self.registry = registry or ModelRegistry(Path(self.models_dir))
        logger.info(
            "ModelManager initialized (data_dir=%s, models_dir=%s, tier=%s)",
            self.data_dir,
            self.models_dir,
            self.hw_info.get("tier", model_size),
        )

    def _blend_requires_transformer(self) -> bool:
        """Return True when the persisted blend weights expect Transformer output."""
        weights = self.blend_weights or {}
        for target_cfg in weights.values():
            if isinstance(target_cfg, dict) and float(target_cfg.get("transformer", 0.0)) > 0.0:
                return True
        return False

    def validate_runtime_artifacts(self) -> None:
        """Raise when the on-disk runtime artifact contract is incomplete."""
        validate_runtime_artifacts(
            ArtifactContract(
                models_dir=Path(self.models_dir),
                transformer_required=self._blend_requires_transformer(),
            )
        )

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSVs and build the feature-engineered train/test split."""
        data_dir = Path(self.data_dir)
        players_file = data_dir / "nba_players.csv"
        games_file = data_dir / "nba_games.csv"

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

        required_cols = ["PLAYER_ID", "GAME_DATE"] + self.core_targets
        missing_cols = [c for c in required_cols if c not in full_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns after feature engineering: {missing_cols}"
            )

        self.feature_schema = self.feature_selector.fit(
            full_df, group_columns=self.feature_engineer.get_group_columns()
        )
        self.feature_cols = self.feature_schema.feature_cols
        self.cat_features = list(self.feature_schema.categorical_cols)

        split_date_str = self.model_config.get("training", {}).get(
            "test_split_date", "2025-01-01"
        )
        split_date = pd.to_datetime(split_date_str)
        train_df = full_df[full_df["GAME_DATE"] < split_date].copy()
        test_df = full_df[full_df["GAME_DATE"] >= split_date].copy()

        if train_df.empty:
            raise ValueError("Training set is empty after split")
        if test_df.empty:
            raise ValueError("Test set is empty after split")

        logger.info("Train set: %s, Test set: %s", len(train_df), len(test_df))
        return train_df, test_df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select leakage-safe numeric features."""
        return self.feature_selector.fit(df).feature_cols

    def _load_feature_cols(self) -> Optional[List[str]]:
        """Load saved feature column names from disk."""
        schema_path = Path(self.models_dir) / "feature_schema.pkl"
        legacy_path = Path(self.models_dir) / "feature_cols.pkl"

        schema = self.feature_selector.load_schema(schema_path)
        if schema is None and legacy_path.exists():
            legacy_cols = joblib.load(legacy_path)
            schema = FeatureSchema(feature_cols=list(legacy_cols))
            self.feature_selector.feature_schema = schema

        if schema is None:
            return None

        self.feature_schema = schema
        self.feature_cols = schema.feature_cols
        self.cat_features = list(schema.categorical_cols)
        logger.info("Loaded %s feature columns", len(self.feature_cols))
        return self.feature_cols

    def _load_models(self) -> Dict[str, int]:
        """Load CatBoost and Transformer models from disk."""
        self.validate_runtime_artifacts()
        self._load_feature_cols()
        self.models = {}
        self.catboost_mae_models = {}
        self.catboost_quantile_models = {}
        self.transformer_model = None

        counts = {"catboost": 0, "transformer": 0, "failed": 0}

        for target in self.targets:
            try:
                trainer = CatBoostTrainer.load(self.models_dir, target)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load CatBoost runtime artifacts for {target} from {self.models_dir}"
                ) from exc

            if trainer.primary_model is not None:
                self.models[target] = trainer.primary_model
                counts["catboost"] += 1

            if trainer.mae_model is not None:
                self.catboost_mae_models[target] = trainer.mae_model

            quantile_dict = collect_quantile_dict(trainer)
            if quantile_dict:
                self.catboost_quantile_models[target] = quantile_dict

        if self.feature_cols is None and self.models:
            for model in self.models.values():
                if hasattr(model, "feature_names_") and model.feature_names_:
                    self.feature_cols = list(model.feature_names_)
                    self.cat_features = [
                        c
                        for c in ["PLAYER_ID", "TEAM_ID", "OPPONENT_ID"]
                        if c in self.feature_cols
                    ]
                    self.feature_schema = FeatureSchema(
                        feature_cols=self.feature_cols,
                        categorical_cols=list(self.cat_features),
                    )
                    self.feature_selector.feature_schema = self.feature_schema
                    break

        self.transformer_model = load_transformer_from_disk(self.models_dir)
        if self.transformer_model is not None:
            counts["transformer"] = 1

        self.blend_weights = load_blend_weights_from_disk(self.models_dir)

        # Bootstrap versioned EnsembleWeights from WeightStore (preferred over
        # legacy blend_weights.pkl).  This ensures predict_player_stats()
        # follows data-driven weights — not arbitrary defaults.
        try:
            store = WeightStore(str(Path(self.models_dir) / "blend_weights"))
            current = store.load_current()
            if current is not None:
                self.use_ensemble_weights(current)
                logger.info(
                    "Bootstrapped ensemble weights v%d from WeightStore (score=%s)",
                    current.version,
                    current.backtest_score or "N/A",
                )
        except Exception as exc:
            logger.debug("WeightStore bootstrap skipped: %s", exc)

        self._validate_blend_contract()

        logger.info(
            "Loaded %s CatBoost models, %s MAE companions, %s quantile sets",
            counts["catboost"],
            len(self.catboost_mae_models),
            len(self.catboost_quantile_models),
        )

        if counts["catboost"] != len(self.targets):
            raise RuntimeError(
                f"Expected {len(self.targets)} CatBoost targets but loaded {counts['catboost']}"
            )

        self._load_residual_corrections()
        self._load_interval_calibration()

        return counts

    def _load_residual_corrections(self) -> None:
        """Attempt to load residual correction models.

        This is a best-effort operation — the main prediction pipeline
        continues to work if residual artifacts are missing or broken.
        """
        residual_dir = Path(self.models_dir) / "residual"
        if not residual_dir.exists():
            logger.debug("No residual correction directory at %s", residual_dir)
            return

        try:
            self.residual_correction_model = ResidualCorrectionModel()
            self.residual_correction_model.load(str(residual_dir))
            self.correction_applier = CorrectionApplier(
                residual_model=self.residual_correction_model,
                feature_builder=CorrectionFeatureBuilder(),
            )
            self.residual_corrections_enabled = True
            loaded_stats = self.residual_correction_model.loaded_stats
            logger.info(
                "Residual corrections enabled for: %s",
                loaded_stats if loaded_stats else "none",
            )
        except Exception as exc:
            logger.warning("Residual corrections disabled: %s", exc)
            self.residual_corrections_enabled = False

    def _load_interval_calibration(self) -> None:
        """Attempt to load residual interval calibration artifacts."""
        store = CalibrationIntervalStore(str(Path(self.models_dir) / "calibration")).load()
        if store.enabled:
            self.calibration_interval_store = store
            logger.info(
                "Residual interval calibration enabled for: %s",
                sorted(store.intervals),
            )
        else:
            self.calibration_interval_store = None

    def _validate_blend_contract(self) -> None:
        """Raise when blend weights require a model that is not loaded."""
        validate_blend_contract(self.blend_weights, self.transformer_model, self.models_dir)

    def load_models(self) -> Dict[str, int]:
        """Public wrapper for compatibility with older callers."""
        return self._load_models()

    def use_ensemble_weights(self, weights: EnsembleWeights) -> None:
        """Hot-reload ensemble blend weights from a versioned EnsembleWeights object.

        This replaces the legacy blend_weights dict with the new versioned
        format and updates all blend coefficients immediately without
        reloading models.

        Args:
            weights: An EnsembleWeights object from WeightStore.
        """
        self.ensemble_weights = weights
        # Sync legacy dict for backward compatibility
        self.blend_weights = {}
        for target, tb in weights.per_target.items():
            self.blend_weights[target] = {
                "catboost": tb.catboost,
                "transformer": tb.transformer,
                "intercept": tb.intercept,
            }
        logger.info(
            "Hot-reloaded ensemble weights v%d (score=%.3f)",
            weights.version,
            weights.backtest_score or float("nan"),
        )

    def reload_weights(self, store_dir: str = "models/blend_weights") -> bool:
        """Reload ensemble weights from the versioned weight store.

        Args:
            store_dir: Path to the WeightStore directory.

        Returns:
            True if weights were successfully reloaded, False otherwise.
        """
        store = WeightStore(store_dir)
        weights = store.load_current()
        if weights is None:
            logger.warning("No current weights found in %s", store_dir)
            return False
        self.use_ensemble_weights(weights)
        return True

    def _predict_catboost_quantiles(
        self, target: str, X: pd.DataFrame
    ) -> Optional[Dict[str, np.ndarray]]:
        """Predict quantile bounds when available."""
        if target not in self.catboost_quantile_models:
            return None

        q_models = self.catboost_quantile_models[target]
        preds: Dict[str, np.ndarray] = {}
        if "low" in q_models:
            preds["low"] = np.asarray(q_models["low"].predict(X), dtype=float)
        if "high" in q_models:
            preds["high"] = np.asarray(q_models["high"].predict(X), dtype=float)
        return preds or None

    def _predict_catboost_target(self, target: str, X: pd.DataFrame) -> np.ndarray:
        """Predict a single target with optional MAE companion blending."""
        model = self.models[target]
        primary = np.asarray(model.predict(X), dtype=float)

        mae_model = self.catboost_mae_models.get(target)
        if mae_model is None:
            return np.clip(primary, 0.0, None)

        # Use versioned ensemble weight if available, else hardcoded 0.7
        cb_mae_weight = 0.7
        if self.ensemble_weights is not None:
            target_blend = self.ensemble_weights.per_target.get(target)
            if target_blend is not None:
                cb_mae_weight = target_blend.catboost_mae_blend

        mae_pred = np.asarray(mae_model.predict(X), dtype=float)
        blended = cb_mae_weight * primary + (1.0 - cb_mae_weight) * mae_pred
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

        seq_len = int(getattr(self.transformer_model, "seq_len", 0) or 0)
        if seq_len <= 0 or len(history_df) < seq_len:
            return None

        cat_features = getattr(self, "cat_features", []) or []
        nn_features = [c for c in self.feature_cols if c not in cat_features]
        selector = getattr(self, "feature_selector", None)
        schema = getattr(self, "feature_schema", None)
        if selector is not None and schema is not None:
            seq_df = selector.transform(
                history_df, schema, strict=False, fill_value=0.0
            )
            seq_df = seq_df.reindex(columns=nn_features, fill_value=0)
        else:
            seq_df = (
                history_df.reindex(columns=nn_features, fill_value=0)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
        seq = seq_df.tail(seq_len).values.astype(np.float32)

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

    def predict_player_stats(
        self,
        player_context_df: pd.DataFrame,
        history_df: pd.DataFrame = None,
        include_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Predict a single player's stat line using the active model stack."""
        if player_context_df is None or player_context_df.empty:
            return self._fallback_prediction(pd.DataFrame())

        if not self.models:
            self._load_models()

        if self.feature_cols is None or not self.feature_cols:
            self._load_feature_cols()

        if self.feature_cols is None or not self.feature_cols:
            return self._fallback_prediction(player_context_df)

        expected_cols = load_expected_feature_cols(self.models_dir)
        player_context_df = align_feature_frame(player_context_df, expected_cols)

        selector = getattr(self, "feature_selector", None)
        schema = getattr(self, "feature_schema", None)
        if selector is not None and schema is not None:
            context = selector.transform(
                player_context_df, schema, strict=False, fill_value=0.0
            )
        else:
            context = (
                player_context_df.reindex(columns=self.feature_cols, fill_value=0)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )

        predictions: Dict[str, Any] = {}
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
                # Prefer versioned EnsembleWeights over legacy blend_weights
                if self.ensemble_weights is not None:
                    tb = self.ensemble_weights.per_target.get(target)
                    if tb is not None:
                        cb_weight = tb.catboost
                        tx_weight = tb.transformer
                        intercept = tb.intercept
                    else:
                        cb_weight, tx_weight, intercept = 1.0, 0.0, 0.0
                else:
                    blend_cfg = self.blend_weights.get(target, {})
                    cb_weight = float(blend_cfg.get("catboost", 1.0))
                    tx_weight = float(blend_cfg.get("transformer", 0.0))
                    intercept = float(blend_cfg.get("intercept", 0.0))
                final_pred = (base * cb_weight) + (transformer_pred * tx_weight) + intercept

            predictions[target] = float(max(0.0, final_pred))

            q_preds = self._predict_catboost_quantiles(target, context)
            if q_preds and "low" in q_preds and "high" in q_preds:
                low = float(q_preds["low"][0])
                high = float(q_preds["high"][0])
                predictions[f"{target}_STD"] = max(0.0, (high - low) / 2.56)

        if self.residual_corrections_enabled and self.correction_applier is not None:
            predictions, correction_meta = self._apply_residual_corrections(
                predictions, player_context_df
            )
        else:
            correction_meta = {}

        if include_confidence:
            predictions = self._add_confidence_intervals(
                predictions,
                context_row=player_context_df,
                correction_meta=correction_meta,
            )

        return predictions

    def _apply_residual_corrections(
        self,
        base_predictions: Dict[str, Any],
        context_row: pd.DataFrame,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Apply residual corrections to base predictions.

        Non-target keys (e.g. ``PTS_STD``) are passed through unchanged.
        """
        stat_keys = {t for t in self.targets if t in base_predictions}
        stat_preds = {t: base_predictions[t] for t in stat_keys}

        try:
            corrected, _meta = self.correction_applier.apply(
                base_predictions=stat_preds,
                context_row=context_row,
            )
        except Exception as exc:
            logger.warning("Residual correction failed, returning base predictions: %s", exc)
            return base_predictions, {}

        result = dict(base_predictions)
        for stat, value in corrected.items():
            result[stat] = value
        return result, _meta

    def _add_confidence_intervals(
        self,
        predictions: Dict[str, Any],
        context_row: pd.DataFrame,
        correction_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Append calibrated intervals and confidence labels when available."""
        result = dict(predictions)
        store = self.calibration_interval_store
        if store is None or not store.enabled:
            return result

        bucket = self.confidence_scorer.bucket_from_context(context_row)
        data_quality = self.confidence_scorer.data_quality_from_context(context_row)
        minutes_confidence = self.confidence_scorer.minutes_confidence_from_context(context_row)
        correction_meta = correction_meta or {}

        for stat in self.targets:
            if stat not in result or not store.has_stat(stat):
                continue

            prediction = float(result[stat])
            interval_80 = store.make_interval(
                stat, prediction, confidence=0.8, bucket=bucket
            )
            interval_90 = store.make_interval(
                stat, prediction, confidence=0.9, bucket=bucket
            )
            if interval_80 is not None:
                result[f"{stat}_INTERVAL_80_LOW"] = interval_80.low
                result[f"{stat}_INTERVAL_80_HIGH"] = interval_80.high
            if interval_90 is not None:
                result[f"{stat}_INTERVAL_90_LOW"] = interval_90.low
                result[f"{stat}_INTERVAL_90_HIGH"] = interval_90.high

            width = store.get_interval_width(stat, confidence=0.9, bucket=bucket)
            meta = correction_meta.get(stat, {})
            confidence = self.confidence_scorer.score(
                stat=stat,
                interval_width=width,
                data_quality=data_quality,
                minutes_confidence=minutes_confidence,
                residual_applied=bool(meta.get("residual_applied", False)),
                residual_model_enabled=self.residual_corrections_enabled
                and self.residual_correction_model is not None
                and self.residual_correction_model.is_enabled(stat),
            )
            result[f"{stat}_CONFIDENCE"] = confidence.label
            result[f"{stat}_CONFIDENCE_SCORE"] = confidence.score

        return result

    def predict_player_stats_batch(
        self,
        context_df: pd.DataFrame,
        histories_map: Optional[Dict[int, pd.DataFrame]] = None,
        include_confidence: bool = False,
    ) -> pd.DataFrame:
        """Predict stats for multiple players and return a DataFrame."""
        if context_df is None or context_df.empty:
            return pd.DataFrame()

        rows: List[Dict[str, float]] = []
        for idx in range(len(context_df)):
            row = context_df.iloc[[idx]]
            player_id = (
                int(row["PLAYER_ID"].iloc[0]) if "PLAYER_ID" in row.columns else idx
            )
            history_df = histories_map.get(player_id) if histories_map else None
            try:
                rows.append(
                    self.predict_player_stats(
                        row,
                        history_df,
                        include_confidence=include_confidence,
                    )
                )
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

    def _get_fallback_value(
        self, player_context_df: pd.DataFrame, target: str
    ) -> float:
        """Get a fallback value for a single stat."""
        fallback_cols = [
            f"ROLL_{target}_AVG_10",
            f"ROLL_{target}_AVG_20",
            f"{target}_EWMA_5",
            target,
        ]
        for col in fallback_cols:
            if col in player_context_df.columns and len(player_context_df) > 0:
                val = player_context_df[col].iloc[0]
                if pd.notna(val) and float(val) > 0:
                    return float(val)
        return float(self._FALLBACK_VALUES.get(target, 0.0))
