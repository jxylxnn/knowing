"""New training pipeline for NBA prediction models.

This module provides a clean, modular, and efficient training pipeline
with parallel training, smart caching, and experiment tracking.
"""

from src.training.pipeline import TrainingPipeline
from src.training.trainer import BaseTrainer, TrainResult
from src.training.catboost_trainer import CatBoostTrainer
from src.training.nn_trainer import NeuralNetworkTrainer
from src.training.feature_cache import FeatureCache
from src.training.experiment import ExperimentTracker

__all__ = [
    'TrainingPipeline',
    'BaseTrainer',
    'TrainResult',
    'CatBoostTrainer',
    'NeuralNetworkTrainer',
    'FeatureCache',
    'ExperimentTracker',
]