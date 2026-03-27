"""Training package for the NBA prediction models.

Import the heavy Torch-backed trainers lazily so pytest and light-weight
callers can import :mod:`src.training` without forcing a full Torch import
at package initialization time.
"""

from importlib import import_module
from typing import Any

from src.training.pipeline import TrainingPipeline

__all__ = [
    'TrainingPipeline',
    'BaseTrainer',
    'TrainResult',
    'CatBoostTrainer',
    'NeuralNetworkTrainer',
    'FeatureCache',
    'ExperimentTracker',
]


def __getattr__(name: str) -> Any:
    if name == 'BaseTrainer' or name == 'TrainResult':
        module = import_module('.trainer', __name__)
        return getattr(module, name)
    if name == 'CatBoostTrainer':
        return import_module('.catboost_trainer', __name__).CatBoostTrainer
    if name == 'NeuralNetworkTrainer':
        return import_module('.nn_trainer', __name__).NeuralNetworkTrainer
    if name == 'FeatureCache':
        return import_module('.feature_cache', __name__).FeatureCache
    if name == 'ExperimentTracker':
        return import_module('.experiment', __name__).ExperimentTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
