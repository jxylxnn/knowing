"""Main training pipeline orchestrator for NBA prediction models.

This module provides a clean, efficient training pipeline with:
- Parallel model training across targets
- Smart caching of expensive computations
- Experiment tracking and comparison
- Multiple training modes (quick, standard, full)
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

from src.training.trainer import TrainResult
from src.training.catboost_trainer import CatBoostTrainer, train_catboost_target
from src.training.nn_trainer import NeuralNetworkTrainer
from src.training.feature_cache import FeatureCache, DataSplitCache
from src.training.experiment import ExperimentTracker
from src.training.training_logger import get_training_logger, RichTrainingLogger
from src.models.gpu_utils import check_gpu_compatibility, clear_gpu_memory, get_gpu_memory_usage
from src.config.model_config import get_model_config

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
        self.use_gpu = use_gpu if use_gpu is not None else check_gpu_compatibility()
        self.parallel = parallel
        self.max_workers = max_workers or (4 if parallel else 1)
        
        # Get model config
        self.model_config, self.hw_info = get_model_config(
            force_size=None if model_size == 'auto' else model_size
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
        test_date: str = '2024-03-01',
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
        # Sort by date
        df = train_df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Temporal split for test
        test_df = df[df['GAME_DATE'] >= test_date].copy()
        train_before = df[df['GAME_DATE'] < test_date].copy()
        
        # Split train into fit/val
        split_idx = int(len(train_before) * (1 - val_ratio))
        fit_df = train_before.iloc[:split_idx].copy()
        val_df = train_before.iloc[split_idx:].copy()
        
        # Select features
        self.feature_cols = self._select_features(fit_df)
        self.cat_features = [c for c in ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID'] 
                           if c in self.feature_cols]
        
        logger.info(f"Data prepared: fit={len(fit_df)}, val={len(val_df)}, test={len(test_df)}")
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
        from src.models.multi_output_nn import MultiOutputNN
        
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
        from src.models.lstm_model import LSTMWrapper
        
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
        from src.models.transformer_model import TransformerWrapper
        
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