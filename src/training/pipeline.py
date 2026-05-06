"""Main training pipeline orchestrator for the simplified NBA model stack.

The active path is:
- per-target CatBoost regressors
- one Transformer sequence model
- inverse-MAE blending between the two
- quantile models for uncertainty
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the project env
    torch = None

from src.config import Config
from src.config.model_config import get_model_config, normalize_model_size
from src.models.base import (
    ModelRegistry,
    ModelMetadata,
    collect_quantile_dict,
    load_blend_weights_from_disk,
    load_transformer_from_disk,
    validate_blend_contract,
)
from src.models.gpu_utils import (
    check_gpu_compatibility,
    clear_gpu_memory,
    initialize_gpu_optimizations,
)
from src.training.catboost_trainer import (
    CatBoostTrainer,
    ConstantRegressor,
    train_catboost_target,
)
from src.training.experiment import ExperimentTracker
from src.training.trainer import TrainResult
from src.utils.prediction_utils import FeatureSelector, FeatureSchema

logger = logging.getLogger(__name__)


def _load_transformer_wrapper():
    """Import the Transformer wrapper lazily to avoid pytest shim issues."""
    from src.models.transformer_model import TransformerWrapper

    return TransformerWrapper


class TrainingPipeline:
    """Orchestrates training for the active CatBoost + Transformer stack."""

    TRAINING_MODES = {
        "quick": {
            "catboost_iterations": 500,
            "transformer_epochs": 20,
            "description": "Fast training for development/testing",
        },
        "standard": {
            "catboost_iterations": 1500,
            "transformer_epochs": 60,
            "description": "Default training with the HTML M-tier stack",
        },
        "full": {
            "catboost_iterations": 5000,
            "transformer_epochs": 120,
            "description": "Extended training for maximum accuracy",
        },
    }

    TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

    def __init__(
        self,
        data_dir: Union[str, Path, Config] = "data",
        models_dir: Union[str, Path, ModelRegistry] = "models",
        cache_dir: Union[str, Path] = "cache/training",
        experiments_dir: Union[str, Path] = "experiments",
        mode: str = "standard",
        model_size: str = "M",
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        experiment_name: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        legacy_config: Optional[Config] = (
            data_dir if isinstance(data_dir, Config) else None
        )
        if legacy_config is not None:
            self.config = legacy_config
            self.training_config = legacy_config.training
            self.data_config = legacy_config.data
            data_dir = legacy_config.data.data_dir

            if isinstance(models_dir, ModelRegistry) and registry is None:
                registry = models_dir
                models_dir = legacy_config.data.models_dir
            elif models_dir == "models":
                models_dir = legacy_config.data.models_dir

            if cache_dir == "cache/training":
                cache_dir = legacy_config.data.cache_dir
        else:
            self.config = None
            self.training_config = None
            self.data_config = None

        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if mode not in self.TRAINING_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from {list(self.TRAINING_MODES)}"
            )
        self.mode = mode
        self.mode_config = self.TRAINING_MODES[mode]

        requested_gpu = True if use_gpu is None else bool(use_gpu)
        gpu_available = check_gpu_compatibility() if requested_gpu else False
        if requested_gpu and not gpu_available:
            logger.info("GPU requested but unavailable; falling back to CPU training.")
        self.use_gpu = requested_gpu and gpu_available
        self.gpu_settings = (
            initialize_gpu_optimizations(log_summary=False)
            if self.use_gpu
            else {
                "gpu_available": False,
                "tf32_enabled": False,
                "bf16_available": False,
                "flash_attention_available": False,
                "cudnn_benchmark": False,
                "optimal_workers": 0,
            }
        )

        default_workers = 1 if self.use_gpu else max(1, min(4, os.cpu_count() or 1))
        self.parallel = parallel
        self.max_workers = max_workers or (1 if self.use_gpu else default_workers)

        normalized_size = normalize_model_size(model_size)
        if normalized_size is None:
            normalized_size = "M"
        if normalized_size == "auto":
            self.model_config, self.hw_info = get_model_config(force_size=None)
        else:
            self.model_config, self.hw_info = get_model_config(
                force_size=normalized_size
            )

        self._apply_mode_config()

        self.registry = registry or ModelRegistry(self.models_dir)
        self.experiment = ExperimentTracker(experiments_dir, experiment_name)

        self.feature_cols: Optional[List[str]] = None
        self.cat_features: List[str] = []
        self.feature_schema: Optional[FeatureSchema] = None
        self.feature_selector = FeatureSelector(self.TARGETS)
        self.models: Dict[str, Any] = {}
        self.catboost_mae_models: Dict[str, Any] = {}
        self.catboost_quantile_models: Dict[str, Dict[str, Any]] = {}
        self.transformer_model: Optional[TransformerWrapper] = None
        self.attention_model: Optional[TransformerWrapper] = None
        self.temporal_model: Optional[TransformerWrapper] = None
        self.blend_weights: Dict[str, Dict[str, float]] = {}
        self.trainers: Dict[str, Any] = {}

        logger.info(
            "TrainingPipeline initialized (mode=%s, tier=%s, gpu=%s, parallel=%s)",
            mode,
            self.hw_info.get("tier", normalized_size),
            self.use_gpu,
            parallel,
        )

    def _apply_mode_config(self) -> None:
        """Apply quick/standard/full overrides without changing the tier shape."""
        catboost_cfg = self.model_config["catboost"]
        transformer_cfg = self.model_config["transformer"]
        training_cfg = self.model_config["training"]

        catboost_cfg["iterations"] = self.mode_config["catboost_iterations"]
        catboost_cfg["use_multi_loss"] = False
        catboost_cfg["use_quantile_models"] = True

        transformer_cfg["epochs"] = self.mode_config["transformer_epochs"]
        training_cfg["early_stop_patience"] = max(
            8, training_cfg.get("early_stop_patience", 12)
        )

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_date: Optional[str] = None,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the engineered dataset into chronological fit/val/test sets."""
        if "GAME_DATE" not in train_df.columns:
            raise ValueError("Training data must include a GAME_DATE column")
        if not 0 < val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        df = train_df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")

        if len(df) < 3:
            raise ValueError(
                "Need at least 3 dated rows to build train/validation/test splits"
            )

        split_date_str = test_date or self.model_config.get("training", {}).get(
            "test_split_date", "2025-01-01"
        )
        split_date = pd.to_datetime(split_date_str)

        test_df = df[df["GAME_DATE"] >= split_date].copy()
        train_before = df[df["GAME_DATE"] < split_date].copy()

        if train_before.empty or test_df.empty:
            fallback_test_size = min(max(1, int(len(df) * 0.15)), len(df) - 2)
            train_before = df.iloc[:-fallback_test_size].copy()
            test_df = df.iloc[-fallback_test_size:].copy()
            logger.warning(
                "Configured split date %s produced an empty partition; using chronological fallback.",
                split_date.strftime("%Y-%m-%d"),
            )

        if len(train_before) < 2:
            raise ValueError(
                "Need more historical rows before the split date to create fit/validation sets"
            )

        split_idx = int(len(train_before) * (1 - val_ratio))
        split_idx = min(max(split_idx, 1), len(train_before) - 1)
        fit_df = train_before.iloc[:split_idx].copy()
        val_df = train_before.iloc[split_idx:].copy()

        if fit_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "Temporal split produced an empty fit, validation, or test partition"
            )

        self.feature_schema = self.feature_selector.fit(fit_df)
        self.feature_cols = self.feature_schema.feature_cols
        self.cat_features = list(self.feature_schema.categorical_cols)

        if not self.feature_cols:
            raise ValueError(
                "No leakage-safe features found after feature selection. "
                "Check that input data has numeric columns matching safe prefixes "
                "(ROLL_, EWMA_, VS_OPP_, etc.)"
            )

        logger.info(
            "Data prepared: fit=%s, val=%s, test=%s, features=%s",
            len(fit_df),
            len(val_df),
            len(test_df),
            len(self.feature_cols or []),
        )
        return fit_df, val_df, test_df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select canonical leakage-safe features via the shared selector."""
        return self.feature_selector.fit(df).feature_cols

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill feature NaNs and coerce targets to numeric."""
        df = df.copy()
        if self.feature_schema is not None:
            aligned = self.feature_selector.transform(
                df, self.feature_schema, strict=False, fill_value=0.0
            )
            for col in aligned.columns:
                df[col] = aligned[col].values

        return df

    def _catboost_train_config(self) -> Dict[str, Any]:
        cfg = dict(self.model_config["catboost"])
        cfg["use_multi_loss"] = False
        cfg["use_quantile_models"] = True
        cfg["use_per_target_tuning"] = True
        return cfg

    def _train_catboost_parallel(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, TrainResult]:
        """Train one CatBoost regressor per stat."""
        cat_config = self._catboost_train_config()
        cpu_count = os.cpu_count() or 1
        requested_workers = self.max_workers if self.parallel else 1
        max_workers = max(1, requested_workers)
        if self.use_gpu and max_workers > 1:
            logger.info("Reducing CatBoost workers to 1 to avoid GPU contention")
            max_workers = 1

        if self.parallel and max_workers > 1:
            cat_config = {**cat_config, "thread_count": 1}
        else:
            thread_count_per_model = max(1, cpu_count // max_workers)
            cat_config = {**cat_config, "thread_count": thread_count_per_model}

        logger.info(
            "CatBoost setup: cores=%s workers=%s thread_count=%s",
            cpu_count,
            max_workers,
            cat_config.get("thread_count", 1),
        )

        if not self.feature_cols:
            if self.feature_schema is None:
                self.feature_schema = self.feature_selector.fit(fit_df)
            self.feature_cols = self.feature_schema.feature_cols
            self.cat_features = list(self.feature_schema.categorical_cols)
            if not self.feature_cols:
                raise ValueError(
                    "No feature columns available for CatBoost training. "
                    "Ensure prepare_data() has been called and feature selection succeeded."
                )

        filtered_frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        results: Dict[str, TrainResult] = {}
        for target in self.TARGETS:
            # Guard against targets whose column is entirely missing from the data.
            if target not in fit_df.columns or target not in val_df.columns:
                missing_from = []
                if target not in fit_df.columns:
                    missing_from.append("fit")
                if target not in val_df.columns:
                    missing_from.append("val")
                logger.warning(
                    "Target column '%s' missing from %s DataFrame(s); "
                    "using zero constant fallback.",
                    target, " and ".join(missing_from),
                )
                trainer = CatBoostTrainer(
                    model_name=f"catboost_{target}",
                    target=target,
                    config=cat_config,
                    use_gpu=False,
                    use_multi_loss=cat_config.get("use_multi_loss", True),
                    use_quantile=cat_config.get(
                        "use_quantile_models", cat_config.get("use_quantile", True)
                    ),
                )
                trainer.primary_model = ConstantRegressor(0.0)
                if trainer.use_multi_loss:
                    trainer.mae_model = ConstantRegressor(0.0)
                if trainer.use_quantile:
                    trainer.quantile_low_model = ConstantRegressor(0.0)
                    trainer.quantile_high_model = ConstantRegressor(0.0)
                trainer.feature_cols = list(self.feature_cols or [])
                trainer.cat_features = list(self.cat_features)
                trainer.is_trained = True
                self.models[target] = trainer.primary_model
                self.trainers[f"catboost_{target}"] = trainer
                results[target] = TrainResult(
                    model=trainer, metrics={}, training_time=0.0
                )
                continue

            fit_target = (
                fit_df[fit_df[target].notna()].copy()
                if target in fit_df.columns
                else fit_df.copy()
            )
            val_target = (
                val_df[val_df[target].notna()].copy()
                if target in val_df.columns
                else val_df.copy()
            )
            if fit_target.empty or val_target.empty:
                logger.warning(
                    "Target %s has too little labeled data after filtering; using constant fallback.",
                    target,
                )
                fallback_value = (
                    float(fit_df[target].dropna().mean())
                    if target in fit_df.columns and fit_df[target].notna().any()
                    else 0.0
                )
                trainer = CatBoostTrainer(
                    model_name=f"catboost_{target}",
                    target=target,
                    config=cat_config,
                    use_gpu=False,
                    use_multi_loss=cat_config.get("use_multi_loss", True),
                    use_quantile=cat_config.get(
                        "use_quantile_models", cat_config.get("use_quantile", True)
                    ),
                )
                trainer.primary_model = ConstantRegressor(fallback_value)
                if trainer.use_multi_loss:
                    trainer.mae_model = ConstantRegressor(fallback_value)
                if trainer.use_quantile:
                    trainer.quantile_low_model = ConstantRegressor(fallback_value)
                    trainer.quantile_high_model = ConstantRegressor(fallback_value)
                trainer.feature_cols = list(self.feature_cols or [])
                trainer.cat_features = list(self.cat_features)
                trainer.is_trained = True
                self.models[target] = trainer.primary_model
                self.trainers[f"catboost_{target}"] = trainer
                results[target] = TrainResult(
                    model=trainer, metrics={}, training_time=0.0
                )
                continue
            filtered_frames[target] = (fit_target, val_target)

        if self.parallel and len(self.TARGETS) > 1:
            results_list = Parallel(n_jobs=max_workers, prefer="threads")(
                delayed(train_catboost_target)(
                    target=target,
                    X_train=filtered_frames[target][0][self.feature_cols],
                    y_train=filtered_frames[target][0][target],
                    X_val=filtered_frames[target][1][self.feature_cols],
                    y_val=filtered_frames[target][1][target],
                    config=cat_config,
                    cat_features=self.cat_features,
                    sample_weight=None,
                    use_gpu=self.use_gpu,
                )
                for target in self.TARGETS
                if target in filtered_frames
            )
            results.update(dict(results_list))
        else:
            for target in self.TARGETS:
                if target not in filtered_frames:
                    if target in self.models:
                        results[target] = TrainResult(
                            model=self.models[target], metrics={}, training_time=0.0
                        )
                    continue
                _, result = train_catboost_target(
                    target=target,
                    X_train=filtered_frames[target][0][self.feature_cols],
                    y_train=filtered_frames[target][0][target],
                    X_val=filtered_frames[target][1][self.feature_cols],
                    y_val=filtered_frames[target][1][target],
                    config=cat_config,
                    cat_features=self.cat_features,
                    sample_weight=None,
                    use_gpu=self.use_gpu,
                )
                results[target] = result
                if self.use_gpu:
                    clear_gpu_memory()

        for target, result in results.items():
            self.experiment.log_model_metrics("catboost", result.metrics, target)
            trainer = result.model
            if isinstance(trainer, CatBoostTrainer):
                self.models[target] = trainer.primary_model
                if trainer.mae_model is not None:
                    self.catboost_mae_models[target] = trainer.mae_model
                if (
                    trainer.quantile_low_model is not None
                    or trainer.quantile_high_model is not None
                ):
                    self.catboost_quantile_models[target] = {
                        k: v
                        for k, v in {
                            "low": trainer.quantile_low_model,
                            "high": trainer.quantile_high_model,
                        }.items()
                        if v is not None
                    }
                self.trainers[f"catboost_{target}"] = trainer
            else:
                self.models[target] = trainer

        return results

    def _save_catboost_artifacts(
        self, catboost_results: Dict[str, TrainResult]
    ) -> None:
        """Persist every per-target CatBoost runtime artifact to disk."""
        missing_targets: List[str] = []
        artifact_errors: Dict[str, List[str]] = {}

        for target in self.TARGETS:
            result = catboost_results.get(target)
            trainer = (
                result.model
                if result and isinstance(result.model, CatBoostTrainer)
                else self.trainers.get(f"catboost_{target}")
            )
            if not isinstance(trainer, CatBoostTrainer):
                missing_targets.append(target)
                continue

            trainer.save(self.models_dir)
            missing_files = trainer.validate_saved_artifacts(self.models_dir)
            if missing_files:
                artifact_errors[target] = missing_files

        if missing_targets or artifact_errors:
            details: List[str] = []
            if missing_targets:
                details.append(
                    f"missing trainers for targets: {', '.join(missing_targets)}"
                )
            if artifact_errors:
                details.extend(
                    f"{target}: {', '.join(paths)}"
                    for target, paths in sorted(artifact_errors.items())
                )
            logger.error(
                "Training finished with incomplete CatBoost runtime artifacts: %s",
                " | ".join(details),
            )

    def _validate_runtime_artifact_contract(self, *, require_transformer: bool) -> None:
        """Fail if the training run did not produce the runtime artifacts loaders depend on."""
        missing: List[str] = []

        required_files = [
            self.models_dir / "feature_schema.pkl",
            self.models_dir / "feature_cols.pkl",
            self.models_dir / "blend_weights.pkl",
            self.models_dir / "model_stack_metadata.pkl",
        ]
        if require_transformer:
            required_files.append(self.models_dir / "attention_transformer.pkl")

        missing.extend(str(path) for path in required_files if not path.exists())

        per_target_missing: Dict[str, List[str]] = {}
        for target in self.TARGETS:
            target_missing = CatBoostTrainer.missing_runtime_artifacts(
                self.models_dir, target
            )
            if target_missing:
                per_target_missing[target] = target_missing

        if per_target_missing:
            missing.extend(
                f"{target}: {', '.join(paths)}"
                for target, paths in sorted(per_target_missing.items())
            )

        if missing:
            raise RuntimeError(
                "Training completed without all required runtime artifacts. Missing: "
                + " | ".join(missing)
            )

    def _build_sequence_batch(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        seq_len: int,
        target_index_set: Optional[set] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Build sliding window sequences for a set of player rows.

        Players with fewer than seq_len + 1 games are no longer skipped.
        Instead, early games use zero-padding for the missing context
        timesteps so that every player with at least 1 game contributes
        training samples.
        """
        sequences: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        indices: List[int] = []

        df_sorted = df.sort_values(["PLAYER_ID", "GAME_DATE"])
        for _, group in df_sorted.groupby("PLAYER_ID", sort=False):
            n_games = len(group)
            if n_games < 1:
                continue

            group_features = (
                group[feature_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .values.astype(np.float32)
            )
            group_targets_raw = group[target_cols].apply(pd.to_numeric, errors="coerce")
            valid_target_rows = group_targets_raw.notna().all(axis=1).values
            group_targets = group_targets_raw.fillna(0).values.astype(np.float32)
            group_indices = list(group.index)
            n_features = group_features.shape[1]

            for idx in range(n_games):
                target_idx = group_indices[idx]
                if target_index_set is not None and target_idx not in target_index_set:
                    continue
                if not valid_target_rows[idx]:
                    continue

                if idx >= seq_len:
                    # Enough context — standard sliding window
                    seq = group_features[idx - seq_len : idx]
                else:
                    # Not enough context — zero-pad the beginning
                    context = group_features[:idx]  # games 0..idx-1
                    pad_len = seq_len - idx
                    pad = np.zeros((pad_len, n_features), dtype=np.float32)
                    seq = np.vstack([pad, context]) if idx > 0 else pad

                sequences.append(seq)
                targets.append(group_targets[idx])
                indices.append(target_idx)

        if not sequences:
            return (
                np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.empty((0, len(target_cols)), dtype=np.float32),
                [],
            )

        return (
            np.asarray(sequences, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            indices,
        )

    def _predict_transformer_batch(
        self, model: TransformerWrapper, sequences: np.ndarray
    ) -> np.ndarray:
        """Run a TransformerWrapper on a batch of sequences."""
        if sequences.size == 0:
            return np.empty((0, len(self.TARGETS)), dtype=np.float32)
        return model.predict_batch(sequences)

    def _evaluate_transformer_validation(
        self,
        model: TransformerWrapper,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute per-target validation MAE for the sequence model."""
        seq_len = int(model.seq_len)
        combined_df = pd.concat([fit_df, val_df], axis=0)
        sequences, targets, _ = self._build_sequence_batch(
            combined_df,
            self.feature_cols,
            self.TARGETS,
            seq_len,
            target_index_set=set(val_df.index),
        )

        if len(sequences) == 0:
            return {}

        preds = self._predict_transformer_batch(model, sequences)
        metrics: Dict[str, float] = {}
        for i, target in enumerate(self.TARGETS):
            metrics[target] = float(np.mean(np.abs(targets[:, i] - preds[:, i])))

        metrics["mean_mae"] = float(np.mean([metrics[t] for t in self.TARGETS]))
        return metrics

    def _train_transformer_model(
        self, fit_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> TrainResult:
        """Train the single sequence Transformer used in the ensemble."""
        logger.info("Training Transformer sequence model...")
        transformer_cfg = dict(self.model_config["transformer"])
        nn_features = [c for c in self.feature_cols if c not in self.cat_features]
        seq_len = int(
            transformer_cfg.get("seq_len", transformer_cfg.get("max_seq_length", 10))
        )

        TransformerWrapper = _load_transformer_wrapper()
        model = TransformerWrapper(
            input_dim=len(nn_features),
            seq_len=seq_len,
            config=transformer_cfg,
            output_dim=len(self.TARGETS),
        )

        model.fit(fit_df, nn_features, self.TARGETS)
        model_path = self.models_dir / "attention_transformer.pkl"
        if model.is_trained:
            model.save(str(model_path))
            logger.info("Transformer model saved to %s", model_path)
        else:
            logger.warning(
                "Transformer training produced no sequences; skipping model save. "
                "Blend weights will fall back to CatBoost-only."
            )

        self.transformer_model = model
        self.attention_model = model
        self.temporal_model = model

        metrics = self._evaluate_transformer_validation(model, fit_df, val_df)
        self.experiment.log_model_metrics("transformer", metrics)

        return TrainResult(model=model, metrics=metrics, training_time=0.0)

    def _build_inverse_mae_weights(
        self,
        catboost_results: Dict[str, TrainResult],
        transformer_result: TrainResult,
    ) -> Dict[str, Dict[str, float]]:
        """Create normalized inverse-MAE blend weights."""
        weights: Dict[str, Dict[str, float]] = {}
        tx_metrics = transformer_result.metrics or {}
        tx_mean_mae = float(tx_metrics.get("mean_mae", 0.0))

        for target in self.TARGETS:
            cb_mae = float(
                catboost_results.get(target, TrainResult(None, {}, 0.0)).metrics.get(
                    "mae", 0.0
                )
            )
            tx_mae = float(tx_metrics.get(target, tx_mean_mae))

            if cb_mae <= 0 and tx_mae <= 0:
                cb_weight, tx_weight = 1.0, 0.0
            elif cb_mae <= 0:
                cb_weight, tx_weight = 0.0, 1.0
            elif tx_mae <= 0:
                cb_weight, tx_weight = 1.0, 0.0
            else:
                inv_cb = 1.0 / cb_mae
                inv_tx = 1.0 / tx_mae
                total = inv_cb + inv_tx
                cb_weight = inv_cb / total
                tx_weight = inv_tx / total

            weights[target] = {
                "catboost": float(cb_weight),
                "transformer": float(tx_weight),
                "catboost_mae": float(cb_mae),
                "transformer_mae": float(tx_mae),
            }

        return weights

    def _build_ridge_blend_weights(
        self,
        catboost_results: Dict[str, TrainResult],
        transformer_result: TrainResult,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Learn per-target blend weights via Ridge regression on validation predictions.

        Falls back to inverse-MAE weighting when Ridge fails or produces
        degenerate coefficients (e.g. negative weights).
        """
        fallback_weights = self._build_inverse_mae_weights(
            catboost_results, transformer_result
        )

        transformer_model = getattr(self, "transformer_model", None) or getattr(
            self, "attention_model", None
        )
        if transformer_model is None or not self.feature_cols:
            logger.info("No Transformer model available; using inverse-MAE blending.")
            return {**fallback_weights, "_method": "inverse_mae"}

        seq_len = int(
            getattr(transformer_model, "seq_len", None)
            or self.model_config.get("transformer", {}).get("seq_len", 10)
        )

        nn_features = [c for c in self.feature_cols if c not in self.cat_features]
        X_val = val_df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        try:
            sequences, targets, _ = self._build_sequence_batch(
                pd.concat([fit_df, val_df], axis=0),
                self.feature_cols,
                self.TARGETS,
                seq_len,
                target_index_set=set(val_df.index),
            )
        except Exception as exc:
            logger.warning("Sequence batch failed for Ridge blending: %s", exc)
            return {**fallback_weights, "_method": "inverse_mae"}

        if len(sequences) == 0:
            logger.warning("Empty sequence batch; falling back to inverse-MAE blending.")
            return {**fallback_weights, "_method": "inverse_mae"}

        tx_preds_all = self._predict_transformer_batch(transformer_model, sequences)

        ridge_weights: Dict[str, Dict[str, float]] = {}
        ridge_improved = 0

        for i, target in enumerate(self.TARGETS):
            cb_mae = float(
                catboost_results.get(target, TrainResult(None, {}, 0.0)).metrics.get(
                    "mae", 0.0
                )
            )
            tx_mae = float(
                (transformer_result.metrics or {}).get(target, 0.0)
            )

            fallback = fallback_weights[target]

            try:
                y_val = pd.to_numeric(val_df[target], errors="coerce").dropna()
                X_val_target = X_val.loc[y_val.index]

                cb_preds = np.full(len(y_val), np.nan)
                model = self.models.get(target)
                if model is not None and hasattr(model, "predict"):
                    cb_preds = model.predict(X_val_target).ravel()[: len(y_val)]

                val_indices = set(val_df.index)
                if hasattr(y_val, "index"):
                    seq_mask = np.array([idx in val_indices for idx in y_val.index])
                    seq_map = {
                        idx: j for j, idx in enumerate(y_val.index) if idx in val_indices
                    }
                else:
                    seq_mask = np.ones(len(y_val), dtype=bool)
                    seq_map = {}

                tx_target = np.full(len(y_val), np.nan)
                if len(tx_preds_all) > 0 and len(seq_map) > 0:
                    val_idx_arr = list(val_df.index)
                    seq_idx_set = set(val_df.index)
                    tx_idx_map = {}
                    for si, orig_idx in enumerate(val_df.index):
                        if orig_idx in seq_map:
                            tx_idx_map[seq_map[orig_idx]] = si

                    for row_i, orig_idx in enumerate(y_val.index):
                        if orig_idx in tx_idx_map:
                            si = tx_idx_map[orig_idx]
                            if si < len(tx_preds_all):
                                tx_target[row_i] = tx_preds_all[si, i]

                valid = ~(np.isnan(cb_preds) | np.isnan(tx_target) | y_val.isna().values)
                if valid.sum() < 50:
                    ridge_weights[target] = fallback
                    continue

                X_stack = np.column_stack([cb_preds[valid], tx_target[valid]])
                y_true = y_val.values[valid]

                ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
                ridge.fit(X_stack, y_true)

                w_cb = float(ridge.coef_[0])
                w_tx = float(ridge.coef_[1])
                intercept = float(ridge.intercept_)

                if w_cb < 0 and w_tx < 0:
                    ridge_weights[target] = fallback
                    continue

                ridge_mae = float(
                    np.mean(np.abs(y_true - ridge.predict(X_stack)))
                )
                inv_mae_mae = float(
                    np.mean(
                        np.abs(
                            y_true
                            - (cb_preds[valid] * fallback["catboost"]
                               + tx_target[valid] * fallback["transformer"])
                        )
                    )
                )

                if ridge_mae < inv_mae_mae and w_cb >= 0 and w_tx >= 0:
                    ridge_weights[target] = {
                        "catboost": float(w_cb),
                        "transformer": float(w_tx),
                        "intercept": float(intercept),
                        "catboost_mae": float(cb_mae),
                        "transformer_mae": float(tx_mae),
                    }
                    ridge_improved += 1
                else:
                    ridge_weights[target] = fallback

            except Exception as exc:
                logger.warning(
                    "Ridge blending failed for %s (%s); using inverse-MAE", target, exc
                )
                ridge_weights[target] = fallback

        logger.info(
            "Ridge blending improved %d/%d targets vs inverse-MAE",
            ridge_improved,
            len(self.TARGETS),
        )
        ridge_weights["_method"] = "ridge"
        return ridge_weights

    def _save_feature_cols(self) -> None:
        if self.feature_schema:
            joblib.dump(self.feature_schema, self.models_dir / "feature_schema.pkl")
            joblib.dump(
                self.feature_schema.feature_cols, self.models_dir / "feature_cols.pkl"
            )
            logger.info(
                "Saved %s feature columns and schema hash %s",
                len(self.feature_schema.feature_cols),
                self.feature_schema.schema_hash,
            )

    def _save_blend_weights(self) -> None:
        if self.blend_weights:
            joblib.dump(self.blend_weights, self.models_dir / "blend_weights.pkl")
            logger.info("Saved blend weights for %s targets", len(self.blend_weights))

    def _save_model_stack_metadata(self) -> None:
        transformer_enabled = bool(self.model_config["transformer"]["enabled"])
        metadata = {
            "transformer_enabled": transformer_enabled,
            "model_count": 2 if transformer_enabled else 1,
        }
        training_preset = getattr(self, "training_preset", None)
        if training_preset:
            metadata["training_preset"] = training_preset
        feature_groups = getattr(self, "feature_group_selection", None)
        if feature_groups:
            metadata["feature_groups"] = list(feature_groups)
        blend_method = self.blend_weights.get("_method", "inverse_mae")
        metadata["blend_method"] = blend_method
        joblib.dump(metadata, self.models_dir / "model_stack_metadata.pkl")
        logger.info(
            "Saved model stack metadata (transformer=%s, model_count=%s, preset=%s)",
            transformer_enabled,
            metadata["model_count"],
            training_preset or "none",
        )

    def train(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Train all active models."""
        if fit_df is None or fit_df.empty:
            raise ValueError("Training DataFrame is None or empty")
        if len(fit_df) < 1000:
            raise ValueError(f"Training data too small: {len(fit_df)} rows")
        if val_df is None or val_df.empty:
            raise ValueError("Validation DataFrame is None or empty")

        if self.feature_schema is None:
            if self.feature_cols:
                self.feature_schema = FeatureSchema(
                    feature_cols=list(self.feature_cols),
                    categorical_cols=[
                        c
                        for c in ["PLAYER_ID", "TEAM_ID", "OPPONENT_ID"]
                        if c in self.feature_cols
                    ],
                )
                self.feature_selector.feature_schema = self.feature_schema
            else:
                self.feature_schema = self.feature_selector.fit(fit_df)
            self.feature_cols = self.feature_schema.feature_cols
            self.cat_features = list(self.feature_schema.categorical_cols)

        required_cols = ["PLAYER_ID", "GAME_DATE"] + self.TARGETS
        missing_cols = [c for c in required_cols if c not in fit_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        fit_df = self._clean_data(fit_df)
        val_df = self._clean_data(val_df)

        self.experiment.start_run(
            config={
                "mode": self.mode,
                "model_size": self.hw_info.get("tier", "M"),
                "model_config": self.model_config,
                "hw_info": self.hw_info,
            },
            notes=f"Training run in {self.mode} mode",
        )

        overall_start = time.time()
        results: Dict[str, Any] = {}

        logger.info("=== Training CatBoost Models ===")
        catboost_results = self._train_catboost_parallel(fit_df, val_df)
        results["catboost"] = catboost_results

        if self.model_config["transformer"]["enabled"]:
            logger.info("=== Training Transformer Model ===")
            transformer_result = self._train_transformer_model(fit_df, val_df)
            results["transformer"] = transformer_result
        else:
            transformer_result = TrainResult(model=None, metrics={}, training_time=0.0)

        transformer_trained = (
            bool(self.model_config["transformer"]["enabled"])
            and transformer_result.model is not None
            and getattr(transformer_result.model, 'is_trained', False)
        )
        self._save_catboost_artifacts(catboost_results)

        if transformer_trained and transformer_result.metrics:
            self.blend_weights = self._build_ridge_blend_weights(
                catboost_results, transformer_result, fit_df, val_df
            )
        else:
            if bool(self.model_config["transformer"]["enabled"]) and not transformer_trained:
                logger.warning(
                    "Transformer enabled but not trained; using inverse-MAE blending "
                    "with CatBoost-only weights."
                )
            self.blend_weights = self._build_inverse_mae_weights(
                catboost_results, transformer_result
            )

        self._save_blend_weights()
        self._save_feature_cols()
        self._save_model_stack_metadata()
        self._validate_runtime_artifact_contract(
            require_transformer=transformer_trained,
        )

        total_time = time.time() - overall_start
        self.experiment.log_params(
            {
                "total_training_time": total_time,
                "model_size": self.hw_info.get("tier", "M"),
            }
        )
        self.experiment.end_run("completed")

        logger.info("=== Training Complete: %.1fs ===", total_time)
        return results

    def load_models(self) -> None:
        """Load saved models and blend weights from disk."""
        logger.info("Loading models from disk...")

        feature_schema_path = self.models_dir / "feature_schema.pkl"
        legacy_feature_cols_path = self.models_dir / "feature_cols.pkl"
        schema = self.feature_selector.load_schema(feature_schema_path)
        if schema is None and legacy_feature_cols_path.exists():
            legacy_cols = joblib.load(legacy_feature_cols_path)
            schema = FeatureSchema(
                feature_cols=list(legacy_cols),
                categorical_cols=[
                    c
                    for c in ["PLAYER_ID", "TEAM_ID", "OPPONENT_ID"]
                    if c in legacy_cols
                ],
            )
            self.feature_selector.feature_schema = schema
        if schema is not None:
            self.feature_schema = schema
            self.feature_cols = schema.feature_cols
            self.cat_features = list(schema.categorical_cols)
            logger.info("Loaded %s feature columns", len(self.feature_cols))

        self.models = {}
        self.catboost_mae_models = {}
        self.catboost_quantile_models = {}

        for target in self.TARGETS:
            try:
                trainer = CatBoostTrainer.load(self.models_dir, target)
            except Exception as exc:
                logger.debug("Failed to load CatBoost trainer for %s: %s", target, exc)
                continue

            if trainer.primary_model is not None:
                self.models[target] = trainer.primary_model
            if trainer.mae_model is not None:
                self.catboost_mae_models[target] = trainer.mae_model

            quantile_dict = collect_quantile_dict(trainer)
            if quantile_dict:
                self.catboost_quantile_models[target] = quantile_dict

        self.transformer_model = load_transformer_from_disk(self.models_dir)
        self.attention_model = self.transformer_model
        self.temporal_model = self.transformer_model

        self.blend_weights = load_blend_weights_from_disk(self.models_dir)

        self._validate_blend_contract()

    def _validate_blend_contract(self) -> None:
        """Raise when blend weights require a model that is not loaded."""
        validate_blend_contract(self.blend_weights, self.transformer_model, self.models_dir)

    def get_summary(self) -> Dict[str, Any]:
        """Return a compact summary for CLI output."""
        return {
            "mode": self.mode,
            "model_size": self.hw_info.get("tier", "M"),
            "training_preset": getattr(self, "training_preset", None),
            "experiment_name": self.experiment.experiment_name,
            "models_trained": list(self.trainers.keys()),
            "feature_count": len(self.feature_cols) if self.feature_cols else 0,
            "experiment_summary": self.experiment.get_summary(),
        }


def create_pipeline(mode: str = "standard", **kwargs) -> TrainingPipeline:
    """Factory function used by train.py."""
    return TrainingPipeline(mode=mode, **kwargs)
