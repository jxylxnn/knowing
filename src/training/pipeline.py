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

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the project env
    torch = None

from src.config import Config
from src.config.model_config import get_model_config, normalize_model_size
from src.models.base import ModelRegistry, ModelMetadata
from src.models.gpu_utils import (
    check_gpu_compatibility,
    clear_gpu_memory,
    initialize_gpu_optimizations,
)
from src.training.catboost_trainer import CatBoostTrainer, train_catboost_target
from src.training.experiment import ExperimentTracker
from src.training.trainer import TrainResult

logger = logging.getLogger(__name__)


def _load_transformer_wrapper():
    """Import the Transformer wrapper lazily to avoid pytest shim issues."""
    from src.models.transformer_model import TransformerWrapper

    return TransformerWrapper


class TrainingPipeline:
    """Orchestrates training for the active CatBoost + Transformer stack."""

    TRAINING_MODES = {
        'quick': {
            'catboost_iterations': 500,
            'transformer_epochs': 20,
            'description': 'Fast training for development/testing',
        },
        'standard': {
            'catboost_iterations': 1500,
            'transformer_epochs': 60,
            'description': 'Default training with the HTML M-tier stack',
        },
        'full': {
            'catboost_iterations': 5000,
            'transformer_epochs': 120,
            'description': 'Extended training for maximum accuracy',
        },
    }

    TARGETS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']

    def __init__(
        self,
        data_dir: Union[str, Path, Config] = 'data',
        models_dir: Union[str, Path, ModelRegistry] = 'models',
        cache_dir: Union[str, Path] = 'cache/training',
        experiments_dir: Union[str, Path] = 'experiments',
        mode: str = 'standard',
        model_size: str = 'M',
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        experiment_name: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        legacy_config: Optional[Config] = data_dir if isinstance(data_dir, Config) else None
        if legacy_config is not None:
            self.config = legacy_config
            self.training_config = legacy_config.training
            self.data_config = legacy_config.data
            data_dir = legacy_config.data.data_dir

            if isinstance(models_dir, ModelRegistry) and registry is None:
                registry = models_dir
                models_dir = legacy_config.data.models_dir
            elif models_dir == 'models':
                models_dir = legacy_config.data.models_dir

            if cache_dir == 'cache/training':
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
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self.TRAINING_MODES)}")
        self.mode = mode
        self.mode_config = self.TRAINING_MODES[mode]

        requested_gpu = True if use_gpu is None else bool(use_gpu)
        gpu_available = check_gpu_compatibility() if requested_gpu else False
        if requested_gpu and not gpu_available:
            logger.info("GPU requested but unavailable; falling back to CPU training.")
        self.use_gpu = requested_gpu and gpu_available
        self.gpu_settings = initialize_gpu_optimizations(log_summary=False) if self.use_gpu else {
            'gpu_available': False,
            'tf32_enabled': False,
            'bf16_available': False,
            'flash_attention_available': False,
            'cudnn_benchmark': False,
            'optimal_workers': 0,
        }

        default_workers = 1 if self.use_gpu else max(1, min(4, os.cpu_count() or 1))
        self.parallel = parallel
        self.max_workers = max_workers or (1 if self.use_gpu else default_workers)

        normalized_size = normalize_model_size(model_size)
        if normalized_size is None:
            normalized_size = 'M'
        if normalized_size == 'auto':
            self.model_config, self.hw_info = get_model_config(force_size=None)
        else:
            self.model_config, self.hw_info = get_model_config(force_size=normalized_size)

        self._apply_mode_config()

        self.registry = registry or ModelRegistry(self.models_dir)
        self.experiment = ExperimentTracker(experiments_dir, experiment_name)

        self.feature_cols: Optional[List[str]] = None
        self.cat_features: List[str] = []
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
            self.hw_info.get('tier', normalized_size),
            self.use_gpu,
            parallel,
        )

    def _apply_mode_config(self) -> None:
        """Apply quick/standard/full overrides without changing the tier shape."""
        catboost_cfg = self.model_config['catboost']
        transformer_cfg = self.model_config['transformer']
        training_cfg = self.model_config['training']

        catboost_cfg['iterations'] = self.mode_config['catboost_iterations']
        catboost_cfg['use_multi_loss'] = False
        catboost_cfg['use_quantile_models'] = True

        transformer_cfg['epochs'] = self.mode_config['transformer_epochs']
        training_cfg['early_stop_patience'] = max(8, training_cfg.get('early_stop_patience', 12))

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_date: Optional[str] = None,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the engineered dataset into chronological fit/val/test sets."""
        if 'GAME_DATE' not in train_df.columns:
            raise ValueError("Training data must include a GAME_DATE column")
        if not 0 < val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        df = train_df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.dropna(subset=['GAME_DATE']).sort_values('GAME_DATE')

        if len(df) < 3:
            raise ValueError("Need at least 3 dated rows to build train/validation/test splits")

        split_date_str = test_date or self.model_config.get('training', {}).get('test_split_date', '2024-03-01')
        split_date = pd.to_datetime(split_date_str)

        test_df = df[df['GAME_DATE'] >= split_date].copy()
        train_before = df[df['GAME_DATE'] < split_date].copy()

        if train_before.empty or test_df.empty:
            fallback_test_size = min(max(1, int(len(df) * 0.15)), len(df) - 2)
            train_before = df.iloc[:-fallback_test_size].copy()
            test_df = df.iloc[-fallback_test_size:].copy()
            logger.warning(
                "Configured split date %s produced an empty partition; using chronological fallback.",
                split_date.strftime('%Y-%m-%d'),
            )

        if len(train_before) < 2:
            raise ValueError("Need more historical rows before the split date to create fit/validation sets")

        split_idx = int(len(train_before) * (1 - val_ratio))
        split_idx = min(max(split_idx, 1), len(train_before) - 1)
        fit_df = train_before.iloc[:split_idx].copy()
        val_df = train_before.iloc[split_idx:].copy()

        if fit_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Temporal split produced an empty fit, validation, or test partition")

        self.feature_cols = self._select_features(fit_df)
        self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] if c in self.feature_cols]

        logger.info(
            "Data prepared: fit=%s, val=%s, test=%s, features=%s",
            len(fit_df),
            len(val_df),
            len(test_df),
            len(self.feature_cols or []),
        )
        return fit_df, val_df, test_df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select leakage-safe numeric features."""
        targets = set(self.TARGETS)
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
            if not any(t.lower() in col.lower() for t in targets):
                features.append(col)

        return features

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill feature NaNs and coerce targets to numeric."""
        df = df.copy()
        if self.feature_cols is not None:
            df[self.feature_cols] = df[self.feature_cols].fillna(0)

        for target in self.TARGETS:
            clean_col = f'{target}_CLEAN' if f'{target}_CLEAN' in df.columns else target
            if clean_col in df.columns:
                df[target] = pd.to_numeric(df[clean_col], errors='coerce').fillna(0)

        return df

    def _catboost_train_config(self) -> Dict[str, Any]:
        cfg = dict(self.model_config['catboost'])
        cfg['use_multi_loss'] = False
        cfg['use_quantile_models'] = True
        cfg['use_per_target_tuning'] = True
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

        thread_count_per_model = max(1, cpu_count // max_workers)
        cat_config = {**cat_config, 'thread_count': thread_count_per_model}

        logger.info(
            "CatBoost setup: cores=%s workers=%s thread_count_per_model=%s",
            cpu_count,
            max_workers,
            thread_count_per_model,
        )

        if self.parallel and len(self.TARGETS) > 1:
            results_list = Parallel(n_jobs=max_workers, prefer='threads')(
                delayed(train_catboost_target)(
                    target=target,
                    X_train=fit_df[self.feature_cols],
                    y_train=fit_df[target],
                    X_val=val_df[self.feature_cols],
                    y_val=val_df[target],
                    config=cat_config,
                    cat_features=self.cat_features,
                    sample_weight=None,
                    use_gpu=self.use_gpu,
                )
                for target in self.TARGETS
            )
            results = dict(results_list)
        else:
            results = {}
            for target in self.TARGETS:
                _, result = train_catboost_target(
                    target=target,
                    X_train=fit_df[self.feature_cols],
                    y_train=fit_df[target],
                    X_val=val_df[self.feature_cols],
                    y_val=val_df[target],
                    config=cat_config,
                    cat_features=self.cat_features,
                    sample_weight=None,
                    use_gpu=self.use_gpu,
                )
                results[target] = result

        for target, result in results.items():
            self.experiment.log_model_metrics('catboost', result.metrics, target)
            trainer = result.model
            if isinstance(trainer, CatBoostTrainer):
                self.models[target] = trainer.primary_model
                if trainer.mae_model is not None:
                    self.catboost_mae_models[target] = trainer.mae_model
                if trainer.quantile_low_model is not None or trainer.quantile_high_model is not None:
                    self.catboost_quantile_models[target] = {
                        k: v for k, v in {
                            'low': trainer.quantile_low_model,
                            'high': trainer.quantile_high_model,
                        }.items() if v is not None
                    }
                self.trainers[f'catboost_{target}'] = trainer
            else:
                self.models[target] = trainer

        return results

    def _build_sequence_batch(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        seq_len: int,
        target_index_set: Optional[set] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Build sliding window sequences for a set of player rows."""
        sequences: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        indices: List[int] = []

        df_sorted = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        for _, group in df_sorted.groupby('PLAYER_ID', sort=False):
            if len(group) < seq_len + 1:
                continue

            group_features = group[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
            group_targets = group[target_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
            group_indices = list(group.index)

            for idx in range(seq_len, len(group)):
                target_idx = group_indices[idx]
                if target_index_set is not None and target_idx not in target_index_set:
                    continue
                sequences.append(group_features[idx - seq_len:idx])
                targets.append(group_targets[idx])
                indices.append(target_idx)

        if not sequences:
            return (
                np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.empty((0, len(target_cols)), dtype=np.float32),
                [],
            )

        return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), indices

    def _predict_transformer_batch(self, model: TransformerWrapper, sequences: np.ndarray) -> np.ndarray:
        """Run a TransformerWrapper on a batch of sequences."""
        if sequences.size == 0:
            return np.empty((0, len(self.TARGETS)), dtype=np.float32)

        seq = (sequences - model.feat_mean) / model.feat_std
        seq_tensor = torch.from_numpy(seq.astype(np.float32))
        seq_tensor = seq_tensor.to(model.device)

        device_str = model.device.type
        use_amp = device_str == 'cuda'
        amp_dtype = torch.bfloat16 if use_amp and hasattr(torch, 'bfloat16') else None
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
                preds = model.model(seq_tensor).detach().cpu().numpy()
        return preds

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

        metrics['mean_mae'] = float(np.mean([metrics[t] for t in self.TARGETS]))
        return metrics

    def _train_transformer_model(self, fit_df: pd.DataFrame, val_df: pd.DataFrame) -> TrainResult:
        """Train the single sequence Transformer used in the ensemble."""
        logger.info("Training Transformer sequence model...")
        transformer_cfg = dict(self.model_config['transformer'])
        nn_features = [c for c in self.feature_cols if c not in self.cat_features]
        seq_len = int(transformer_cfg.get('seq_len', transformer_cfg.get('max_seq_length', 10)))

        TransformerWrapper = _load_transformer_wrapper()
        model = TransformerWrapper(
            input_dim=len(nn_features),
            seq_len=seq_len,
            config=transformer_cfg,
            output_dim=len(self.TARGETS),
        )

        model.fit(fit_df, nn_features, self.TARGETS)
        model_path = self.models_dir / 'attention_transformer.pkl'
        model.save(str(model_path))

        self.transformer_model = model
        self.attention_model = model
        self.temporal_model = model

        metrics = self._evaluate_transformer_validation(model, fit_df, val_df)
        self.experiment.log_model_metrics('transformer', metrics)

        return TrainResult(model=model, metrics=metrics, training_time=0.0)

    def _build_inverse_mae_weights(
        self,
        catboost_results: Dict[str, TrainResult],
        transformer_result: TrainResult,
    ) -> Dict[str, Dict[str, float]]:
        """Create normalized inverse-MAE blend weights."""
        weights: Dict[str, Dict[str, float]] = {}
        tx_metrics = transformer_result.metrics or {}
        tx_mean_mae = float(tx_metrics.get('mean_mae', 0.0))

        for target in self.TARGETS:
            cb_mae = float(catboost_results.get(target, TrainResult(None, {}, 0.0)).metrics.get('mae', 0.0))
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
                'catboost': float(cb_weight),
                'transformer': float(tx_weight),
                'catboost_mae': float(cb_mae),
                'transformer_mae': float(tx_mae),
            }

        return weights

    def _save_feature_cols(self) -> None:
        if self.feature_cols:
            joblib.dump(self.feature_cols, self.models_dir / 'feature_cols.pkl')
            logger.info("Saved %s feature columns", len(self.feature_cols))

    def _save_blend_weights(self) -> None:
        if self.blend_weights:
            joblib.dump(self.blend_weights, self.models_dir / 'blend_weights.pkl')
            logger.info("Saved blend weights for %s targets", len(self.blend_weights))

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

        if self.feature_cols is None:
            self.feature_cols = self._select_features(fit_df)
            self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] if c in self.feature_cols]

        required_cols = ['PLAYER_ID', 'GAME_DATE'] + self.TARGETS
        missing_cols = [c for c in required_cols if c not in fit_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        fit_df = self._clean_data(fit_df)
        val_df = self._clean_data(val_df)

        self.experiment.start_run(
            config={
                'mode': self.mode,
                'model_size': self.hw_info.get('tier', 'M'),
                'model_config': self.model_config,
                'hw_info': self.hw_info,
            },
            notes=f"Training run in {self.mode} mode",
        )

        overall_start = time.time()
        results: Dict[str, Any] = {}

        logger.info("=== Training CatBoost Models ===")
        catboost_results = self._train_catboost_parallel(fit_df, val_df)
        results['catboost'] = catboost_results

        if self.model_config['transformer']['enabled']:
            logger.info("=== Training Transformer Model ===")
            transformer_result = self._train_transformer_model(fit_df, val_df)
            results['transformer'] = transformer_result
        else:
            transformer_result = TrainResult(model=None, metrics={}, training_time=0.0)

        self.blend_weights = self._build_inverse_mae_weights(catboost_results, transformer_result)
        self._save_blend_weights()
        self._save_feature_cols()

        total_time = time.time() - overall_start
        self.experiment.log_params({'total_training_time': total_time, 'model_size': self.hw_info.get('tier', 'M')})
        self.experiment.end_run('completed')

        logger.info("=== Training Complete: %.1fs ===", total_time)
        return results

    train_all_models = train

    def load_models(self) -> None:
        """Load saved models and blend weights from disk."""
        logger.info("Loading models from disk...")

        feature_cols_path = self.models_dir / 'feature_cols.pkl'
        if feature_cols_path.exists():
            self.feature_cols = joblib.load(feature_cols_path)
            self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] if c in self.feature_cols]
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
            if trainer.quantile_low_model is not None or trainer.quantile_high_model is not None:
                self.catboost_quantile_models[target] = {
                    k: v for k, v in {
                        'low': trainer.quantile_low_model,
                        'high': trainer.quantile_high_model,
                    }.items() if v is not None
                }

        transformer_path = self.models_dir / 'attention_transformer.pkl'
        if transformer_path.exists():
            try:
                TransformerWrapper = _load_transformer_wrapper()
                self.transformer_model = TransformerWrapper.load(str(transformer_path))
                self.attention_model = self.transformer_model
                self.temporal_model = self.transformer_model
                logger.info("Loaded Transformer model")
            except Exception as exc:
                logger.warning("Failed to load Transformer model: %s", exc)

        blend_path = self.models_dir / 'blend_weights.pkl'
        if blend_path.exists():
            try:
                self.blend_weights = joblib.load(blend_path)
                logger.info("Loaded blend weights for %s targets", len(self.blend_weights))
            except Exception as exc:
                logger.warning("Failed to load blend weights: %s", exc)

    def get_summary(self) -> Dict[str, Any]:
        """Return a compact summary for CLI output."""
        return {
            'mode': self.mode,
            'model_size': self.hw_info.get('tier', 'M'),
            'experiment_name': self.experiment.experiment_name,
            'models_trained': list(self.trainers.keys()),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'experiment_summary': self.experiment.get_summary(),
        }


def create_pipeline(mode: str = 'standard', **kwargs) -> TrainingPipeline:
    """Factory function used by train.py."""
    return TrainingPipeline(mode=mode, **kwargs)
