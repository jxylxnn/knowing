"""Main training pipeline orchestrator for NBA prediction models.

This module provides a clean, efficient training pipeline with:
- Parallel model training across targets
- Smart caching of expensive computations
- Experiment tracking and comparison
- Multiple training modes (quick, standard, full)
- GPU optimizations (TF32, BF16, torch.compile)
"""

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
    from src.training.trainer import TrainResult
    from src.training.catboost_trainer import CatBoostTrainer, train_catboost_target
    from src.training.feature_cache import FeatureCache, DataSplitCache
    from src.training.experiment import ExperimentTracker
    from src.training.training_logger import get_training_logger, RichTrainingLogger
    from src.models.gpu_utils import (
        check_gpu_compatibility,
        clear_gpu_memory,
        get_gpu_memory_usage,
        initialize_gpu_optimizations,
        get_optimal_dataloader_workers,
        is_bf16_supported,
        print_gpu_summary,
    )
    from src.config.model_config import get_model_config
except ModuleNotFoundError:
    import json
    from dataclasses import dataclass

    @dataclass
    class TrainResult:
        model: Any
        metrics: Dict[str, float]
        training_time: float
        best_iteration: Optional[int] = None
        feature_importance: Optional[Dict[str, float]] = None

    class FeatureCache:
        def __init__(self, cache_dir):
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    class DataSplitCache(FeatureCache):
        pass

    class ExperimentTracker:
        def __init__(self, experiments_dir='experiments', experiment_name=None):
            self.experiments_dir = Path(experiments_dir)
            self.experiments_dir.mkdir(parents=True, exist_ok=True)
            self.experiment_name = experiment_name or f'run_{int(time.time())}'
            self.params = {}
            self.model_metrics = {}
            self.status = 'initialized'
        def start_run(self, config=None, notes=None):
            self.status = 'running'
            if config is not None:
                self.params['config'] = config
            if notes is not None:
                self.params['notes'] = notes
        def log_model_metrics(self, model_name: str, metrics: Dict[str, Any], target: Optional[str] = None):
            key = f'{model_name}:{target}' if target else model_name
            self.model_metrics[key] = metrics
        def log_params(self, params: Dict[str, Any]):
            self.params.update(params)
        def end_run(self, status='completed'):
            self.status = status
            out = self.experiments_dir / f'{self.experiment_name}.json'
            out.write_text(json.dumps(self.get_summary(), indent=2, default=str))
        def get_summary(self):
            return {
                'experiment_name': self.experiment_name,
                'status': self.status,
                'params': self.params,
                'model_metrics': self.model_metrics,
            }

    class RichTrainingLogger:
        def __init__(self, use_rich: bool = False, log_gpu: bool = False):
            self.use_rich = use_rich
            self.log_gpu = log_gpu
        def log_iteration(self, metrics):
            if getattr(metrics, 'iteration', None) in (0, getattr(metrics, 'total_iterations', None)):
                logger.info('%s %s iter=%s/%s train=%.5f val=%.5f', metrics.target, metrics.model_type, metrics.iteration, metrics.total_iterations, metrics.train_loss, metrics.val_loss)

    _TRAINING_LOGGER = None
    def get_training_logger(use_rich: bool = False, log_gpu: bool = False):
        global _TRAINING_LOGGER
        if _TRAINING_LOGGER is None:
            _TRAINING_LOGGER = RichTrainingLogger(use_rich=use_rich, log_gpu=log_gpu)
        return _TRAINING_LOGGER

    from catboost_trainer import CatBoostTrainer, train_catboost_target
    from gpu_utils import (
        check_gpu_compatibility,
        clear_gpu_memory,
        get_gpu_memory_usage,
        initialize_gpu_optimizations,
        get_optimal_dataloader_workers,
        is_bf16_supported,
        print_gpu_summary,
    )
    from model_config import get_model_config

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates efficient training of all NBA prediction models."""
    
    # Training modes with their configurations
    TRAINING_MODES = {
        'quick': {
            'catboost_iterations': 500,
            'catboost_depth': 6,
            'nn_epochs': 20,
            'feature_selection': False,
            'adversarial_validation': False,
            'quantile_models': False,
            'multi_loss': False,
            'train_nn_models': False,
            'description': 'Fast training for development/testing',
        },
        'standard': {
            'catboost_iterations': 3000,
            'catboost_depth': 8,
            'nn_epochs': 100,
            'feature_selection': True,
            'adversarial_validation': True,
            'quantile_models': True,
            'multi_loss': True,
            'train_nn_models': True,
            'description': 'Full training with all optimizations',
        },
        'full': {
            'catboost_iterations': 5000,
            'catboost_depth': 10,
            'nn_epochs': 200,
            'feature_selection': True,
            'adversarial_validation': True,
            'quantile_models': True,
            'multi_loss': True,
            'train_nn_models': True,
            'description': 'Extended training for maximum accuracy',
        },
    }
    
    TARGETS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    CORE_TARGETS = ['PTS', 'REB', 'AST']
    
    def __init__(
        self,
        data_dir: Union[str, Path] = 'data',
        models_dir: Union[str, Path] = 'models',
        cache_dir: Union[str, Path] = 'cache/training',
        experiments_dir: Union[str, Path] = 'experiments',
        mode: str = 'standard',
        model_size: str = 'auto',
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize training pipeline.
        
        Args:
            data_dir: Directory containing training data
            models_dir: Directory to save trained models
            cache_dir: Directory for feature caching
            experiments_dir: Directory for experiment tracking
            mode: Training mode ('quick', 'standard', 'full')
            model_size: Model size preset ('auto', 'small', 'medium', 'large')
            parallel: Whether to train targets in parallel
            max_workers: Max parallel workers (None = auto)
            use_gpu: Whether to use GPU (None = auto-detect)
            experiment_name: Name for this experiment
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert cache_dir to Path immediately to avoid string division errors
        cache_dir = Path(cache_dir)
        
        # Training mode
        if mode not in self.TRAINING_MODES:
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self.TRAINING_MODES.keys())}")
        self.mode = mode
        self.mode_config = self.TRAINING_MODES[mode]
        
        # Hardware setup
        gpu_requested = True if use_gpu is None else bool(use_gpu)
        gpu_available = check_gpu_compatibility() if gpu_requested else False
        if gpu_requested and not gpu_available:
            logger.info("GPU requested but unavailable or incompatible. Falling back to CPU training.")
        self.use_gpu = gpu_requested and gpu_available
        self.parallel = parallel
        if self.use_gpu:
            self.gpu_settings = initialize_gpu_optimizations(log_summary=False)
            default_workers = 1
        else:
            self.gpu_settings = {
                'gpu_available': False,
                'tf32_enabled': False,
                'bf16_available': False,
                'flash_attention_available': False,
                'cudnn_benchmark': False,
                'optimal_workers': 0,
            }
            default_workers = max(1, min(4, os.cpu_count() or 1))
        self.max_workers = max_workers or (default_workers if parallel else 1)
        self.dataloader_workers = self.gpu_settings['optimal_workers']
        
        # Get model config
        force_size = 'small' if model_size == 'auto' else model_size
        self.model_config, self.hw_info = get_model_config(
            force_size=force_size
        )
        
        # Apply mode overrides to config
        self._apply_mode_config()
        
        # Caching
        self.feature_cache = FeatureCache(cache_dir)
        self.split_cache = DataSplitCache(cache_dir / 'splits')
        
        # Experiment tracking
        self.experiment = ExperimentTracker(experiments_dir, experiment_name)
        
        # State
        self.feature_cols: Optional[List[str]] = None
        self.cat_features: Optional[List[str]] = None
        self.trainers: Dict[str, Any] = {}
        
        logger.info(f"TrainingPipeline initialized (mode={mode}, gpu={self.use_gpu}, parallel={parallel})")
        logger.info(f"Mode: {self.mode_config['description']}")
        logger.info(
            f"GPU settings: tf32={self.gpu_settings['tf32_enabled']}, "
            f"bf16={self.gpu_settings['bf16_available']}, "
            f"workers={self.dataloader_workers}, max_workers={self.max_workers}"
        )
    
    def _apply_mode_config(self) -> None:
        """Apply mode-specific overrides to model config."""
        # Override CatBoost settings
        self.model_config['catboost']['iterations'] = self.mode_config['catboost_iterations']
        self.model_config['catboost']['depth'] = self.mode_config['catboost_depth']
        self.model_config['catboost']['use_multi_loss'] = self.mode_config['multi_loss']
        self.model_config['catboost']['use_quantile_models'] = self.mode_config['quantile_models']
        
        # Override NN settings
        self.model_config['nn']['epochs'] = self.mode_config['nn_epochs']
        self.model_config['lstm']['epochs'] = self.mode_config['nn_epochs']
        self.model_config['transformer']['epochs'] = self.mode_config['nn_epochs']
    
    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_date: Optional[str] = None,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare and split data for training.
        
        Args:
            train_df: Full training DataFrame with features
            test_date: Date to split test set (games >= this date)
            val_ratio: Ratio of training data to use for validation
            
        Returns:
            Tuple of (fit_df, val_df, test_df)
        """
        if 'GAME_DATE' not in train_df.columns:
            raise ValueError("Training data must include a GAME_DATE column")
        if not 0 < val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        df = train_df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.dropna(subset=['GAME_DATE']).sort_values('GAME_DATE').reset_index(drop=True)

        if len(df) < 3:
            raise ValueError(
                f"Need at least 3 dated rows to build train/validation/test splits, got {len(df)}"
            )

        split_date_str = test_date or self.model_config.get('training', {}).get('test_split_date', '2024-03-01')
        split_date = pd.to_datetime(split_date_str)

        # Temporal split for test
        test_df = df[df['GAME_DATE'] >= split_date].copy()
        train_before = df[df['GAME_DATE'] < split_date].copy()

        if train_before.empty or test_df.empty:
            fallback_test_size = min(max(1, int(len(df) * 0.15)), len(df) - 2)
            if fallback_test_size < 1:
                raise ValueError(
                    "Unable to create a non-empty temporal holdout split. "
                    "Add more historical rows before training."
                )

            logger.warning(
                "Configured test split date %s produced an empty train or test partition. "
                "Falling back to a chronological 85/15 split for this run.",
                split_date.strftime('%Y-%m-%d'),
            )
            train_before = df.iloc[:-fallback_test_size].copy()
            test_df = df.iloc[-fallback_test_size:].copy()

        # Split train into fit/val
        if len(train_before) < 2:
            raise ValueError(
                "Need at least 2 rows before the test split date to create fit/validation sets. "
                "Fetch more historical data or choose an earlier split."
            )

        split_idx = int(len(train_before) * (1 - val_ratio))
        split_idx = min(max(split_idx, 1), len(train_before) - 1)
        fit_df = train_before.iloc[:split_idx].copy()
        val_df = train_before.iloc[split_idx:].copy()

        if fit_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "Temporal split produced an empty fit, validation, or test partition. "
                "Adjust the split date or fetch more data."
            )
        
        # Select features
        self.feature_cols = self._select_features(fit_df)
        self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] 
                           if c in self.feature_cols]
        
        logger.info(
            f"Data prepared: fit={len(fit_df)}, val={len(val_df)}, test={len(test_df)} "
            f"(split_date={split_date.strftime('%Y-%m-%d')})"
        )
        logger.info(f"Features: {len(self.feature_cols)} (categorical: {len(self.cat_features)})")
        
        return fit_df, val_df, test_df
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select valid feature columns."""
        # Exclude non-feature columns
        exclude = {
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
            'GAME_ID', 'GAME_DATE', 'MATCHUP', 'OPPONENT_ID', 'OPPONENT_ABBR',
            'WL', 'SEASON_ID', 'VIDEO_AVAILABLE', 'REST_BUCKET',
        }
        exclude.update(self.TARGETS)
        
        # Safe prefixes and patterns
        safe_prefixes = ('ROLL_', 'EWMA_', 'VS_OPP_', 'PROJ_', 'LEAGUE_PCT_')
        safe_substrings = ('TREND', 'BAYESIAN', 'PACE', '_TE', '_SHARE_', 'ROLE_INDEX',
                         'SEASON_AVG', 'SEASON_SIN', 'SEASON_COS', 'HOT_STREAK', 'COLD_STREAK',
                         'POTENTIAL', 'B2B_IMPACT', 'FATIGUE', 'EFF_Z_SCORE', 'FANTASY',
                         'SOS_', 'PACE_ADJ', 'DEF_MATCHUP', 'OPP_DEF')
        safe_exact = {
            'IS_HOME', 'REST_DAYS', 'IS_B2B', 'FATIGUE_SCORE', 'MONTH', 'DAY_OF_WEEK',
            'EXP_PACE', 'EXP_TEAM_PTS', 'EXP_GAME_TOTAL', 'BLOWOUT_RISK',
            'CLOSE_GAME', 'EXP_MARGIN', 'DAYS_SINCE_LAST', 'MINS_LAST_3',
            'MINS_LAST_7', 'EST_POSS', 'TEAM_PACE_10', 'PACE_FACTOR',
            'STAR_TEAMMATE_OUT',
        }
        
        # Filter numeric columns
        features = []
        for col in df.columns:
            if col in exclude or df[col].dtype not in ('int64', 'float64', 'int32', 'float32'):
                continue
            
            # Check if safe
            if col in safe_exact:
                features.append(col)
            elif any(col.startswith(p) for p in safe_prefixes):
                features.append(col)
            elif any(s in col for s in safe_substrings):
                features.append(col)
            elif not any(t.lower() in col.lower() for t in self.TARGETS):
                features.append(col)
        
        return features
    
    def train(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Train all models.
        
        Args:
            fit_df: Training data
            val_df: Validation data
            
        Returns:
            Dictionary of training results
        """
        # Start experiment
        self.experiment.start_run(
            config={
                'mode': self.mode,
                'model_config': self.model_config,
                'hw_info': self.hw_info,
            },
            notes=f"Training run in {self.mode} mode"
        )
        
        overall_start = time.time()
        results = {}
        
        # Clean data
        fit_df = self._clean_data(fit_df)
        val_df = self._clean_data(val_df)
        
        # Extract features
        X_fit = fit_df[self.feature_cols]
        X_val = val_df[self.feature_cols]
        
        # Train CatBoost models (parallel)
        logger.info("=== Training CatBoost Models ===")
        catboost_results = self._train_catboost_parallel(X_fit, X_val, fit_df, val_df)
        results['catboost'] = catboost_results
        
        # Train NN models if enabled
        if self.mode_config['train_nn_models']:
            logger.info("=== Training Neural Network Models ===")
            
            # Joint NN
            nn_results = self._train_joint_nn(X_fit, X_val, fit_df, val_df)
            results['joint_nn'] = nn_results
            
            # Clear GPU memory between models
            clear_gpu_memory()
            
            # LSTM (if we have sequence data)
            if 'lstm' in self.model_config:
                lstm_results = self._train_lstm(fit_df, val_df)
                results['lstm'] = lstm_results
                clear_gpu_memory()
            
            # Transformer
            if 'transformer' in self.model_config:
                transformer_results = self._train_transformer(fit_df, val_df)
                results['transformer'] = transformer_results
                clear_gpu_memory()
        
        # End experiment
        total_time = time.time() - overall_start
        self.experiment.log_params({'total_training_time': total_time})
        self.experiment.end_run('completed')
        
        # Save feature columns for inference
        self._save_feature_cols()
        
        logger.info(f"=== Training Complete: {total_time:.1f}s ===")
        
        return results
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for training."""
        df = df.copy()
        
        # Fill NaN in features
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        # Clean targets
        for target in self.TARGETS:
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        
        return df
    
    def _train_catboost_parallel(
        self,
        X_fit: pd.DataFrame,
        X_val: pd.DataFrame,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, TrainResult]:
        """Train CatBoost models in parallel across targets."""
        cat_config = self.model_config['catboost']
        cpu_count = os.cpu_count() or 1
        requested_workers = self.max_workers if self.parallel else 1
        max_workers = max(1, requested_workers)

        if self.use_gpu and max_workers > 1:
            logger.info(
                f"GPU CatBoost detected; reducing max_workers from {max_workers} to 1 to avoid GPU contention"
            )
            max_workers = 1

        thread_count_per_model = max(1, cpu_count // max_workers)
        logger.info(
            f"CatBoost training setup: cores={cpu_count}, max_workers={max_workers}, "
            f"thread_count_per_model={thread_count_per_model}"
        )

        if self.parallel and len(self.TARGETS) > 1:
            # Parallel training
            logger.info(
                f"Training {len(self.TARGETS)} CatBoost targets in parallel "
                f"(workers={max_workers}, thread_count_per_model={thread_count_per_model})"
            )

            results_list = Parallel(n_jobs=max_workers, prefer='threads')(
                delayed(train_catboost_target)(
                    target=target,
                    X_train=X_fit,
                    y_train=fit_df[target],
                    X_val=X_val,
                    y_val=val_df[target],
                    config={**cat_config, 'thread_count': thread_count_per_model},
                    cat_features=self.cat_features,
                    sample_weight=None,
                    use_gpu=self.use_gpu,
                )
                for target in self.TARGETS
            )

            results = dict(results_list)
        else:
            # Sequential training
            results = {}
            for target in self.TARGETS:
                logger.info(
                    f"Training CatBoost for {target} "
                    f"(workers={max_workers}, thread_count={thread_count_per_model})"
                )
                _, result = train_catboost_target(
                    target=target,
                    X_train=X_fit,
                    y_train=fit_df[target],
                    X_val=X_val,
                    y_val=val_df[target],
                    config={**cat_config, 'thread_count': thread_count_per_model},
                    cat_features=self.cat_features,
                    sample_weight=None,
                    use_gpu=self.use_gpu,
                )
                results[target] = result
        
        # Log metrics
        for target, result in results.items():
            self.experiment.log_model_metrics('catboost', result.metrics, target)
        
        # Save models
        for target, result in results.items():
            trainer = result.model
            trainer.save(self.models_dir)
            self.trainers[f'catboost_{target}'] = trainer
        
        return results
    
    def _train_joint_nn(
        self,
        X_fit: pd.DataFrame,
        X_val: pd.DataFrame,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> TrainResult:
        """Train joint multi-output neural network."""
        try:
            from src.models.multi_output_nn import MultiOutputNN
            from src.training.nn_trainer import NeuralNetworkTrainer
        except ModuleNotFoundError:
            try:
                from multi_output_nn import MultiOutputNN
                from nn_trainer import NeuralNetworkTrainer
            except ModuleNotFoundError as exc:
                logger.warning(f'Skipping joint NN training because required model code is unavailable: {exc}')
                return TrainResult(model=None, metrics={'skipped_missing_dependency': 1.0}, training_time=0)
        
        nn_config = self.model_config['nn']
        
        # Get numeric features only
        numeric_features = [c for c in self.feature_cols if c not in self.cat_features]
        
        trainer = NeuralNetworkTrainer(
            model_name='joint_nn',
            config=nn_config,
            model_class=MultiOutputNN,
            model_kwargs={
                'input_dim': len(numeric_features),
                'output_dim': len(self.CORE_TARGETS),
                'hidden_dim': nn_config.get('hidden_dim', 512),
                'num_blocks': nn_config.get('num_blocks', 6),
                'dropout': nn_config.get('dropout', 0.3),
            },
            use_gpu=self.use_gpu,
            use_amp=nn_config.get('amp', True),
            use_compile=nn_config.get('use_compile', False),
        )
        
        X_fit_num = fit_df[numeric_features].values
        X_val_num = val_df[numeric_features].values
        y_fit = fit_df[self.CORE_TARGETS].values
        y_val = val_df[self.CORE_TARGETS].values
        
        result = trainer.fit(X_fit_num, y_fit, X_val_num, y_val)
        
        # Log metrics
        self.experiment.log_model_metrics('joint_nn', result.metrics)
        
        # Save model
        trainer.save(self.models_dir / 'joint_stats_nn.pkl')
        self.trainers['joint_nn'] = trainer
        
        return result
    
    def _train_lstm(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> TrainResult:
        """Train LSTM temporal model."""
        # LSTM requires sequential data - use existing wrapper
        try:
            from src.models.lstm_model import LSTMWrapper
        except ModuleNotFoundError:
            try:
                from lstm_model import LSTMWrapper
            except ModuleNotFoundError as exc:
                logger.warning(f'Skipping LSTM training because required model code is unavailable: {exc}')
                return TrainResult(model=None, metrics={'skipped_missing_dependency': 1.0}, training_time=0)
        
        lstm_config = self.model_config['lstm']
        numeric_features = [c for c in self.feature_cols if c not in self.cat_features]
        
        trainer = LSTMWrapper(
            input_dim=len(numeric_features),
            seq_len=lstm_config.get('seq_len', 10),
            config=lstm_config,
        )
        
        trainer.fit(fit_df, numeric_features, self.CORE_TARGETS)
        trainer.save(str(self.models_dir / 'temporal_lstm.pkl'))
        
        # Log basic info
        self.experiment.log_model_metrics('lstm', {'trained': 1.0})
        
        return TrainResult(model=trainer, metrics={}, training_time=0)
    
    def _train_transformer(
        self,
        fit_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> TrainResult:
        """Train Transformer attention model."""
        try:
            from src.models.transformer_model import TransformerWrapper
        except ModuleNotFoundError:
            try:
                from transformer_model import TransformerWrapper
            except ModuleNotFoundError as exc:
                logger.warning(f'Skipping Transformer training because required model code is unavailable: {exc}')
                return TrainResult(model=None, metrics={'skipped_missing_dependency': 1.0}, training_time=0)
        
        tx_config = self.model_config['transformer']
        numeric_features = [c for c in self.feature_cols if c not in self.cat_features]
        
        trainer = TransformerWrapper(
            input_dim=len(numeric_features),
            seq_len=tx_config.get('seq_len', 50),
            config=tx_config,
        )
        
        trainer.fit(fit_df, numeric_features, self.CORE_TARGETS)
        trainer.save(str(self.models_dir / 'attention_transformer.pkl'))
        
        self.experiment.log_model_metrics('transformer', {'trained': 1.0})
        
        return TrainResult(model=trainer, metrics={}, training_time=0)
    
    def _save_feature_cols(self) -> None:
        """Save feature column names."""
        if self.feature_cols:
            joblib.dump(self.feature_cols, self.models_dir / 'feature_cols.pkl')
            logger.info(f"Saved {len(self.feature_cols)} feature columns")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'mode': self.mode,
            'experiment_name': self.experiment.experiment_name,
            'models_trained': list(self.trainers.keys()),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'experiment_summary': self.experiment.get_summary(),
        }


def create_pipeline(
    mode: str = 'standard',
    **kwargs
) -> TrainingPipeline:
    """Factory function to create a training pipeline.
    
    Args:
        mode: Training mode ('quick', 'standard', 'full')
        **kwargs: Additional arguments for TrainingPipeline
        
    Returns:
        Configured TrainingPipeline
    """
    return TrainingPipeline(mode=mode, **kwargs)
