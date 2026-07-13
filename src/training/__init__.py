"""Training package for the NBA prediction models.

Import optional helpers lazily so light-weight callers can import
:mod:`src.training` without loading every training dependency at package
initialization time.
"""

from importlib import import_module
from typing import Any

from src.training.pipeline import TrainingPipeline

__all__ = [
    'TrainingPipeline',
    'BaseTrainer',
    'TrainResult',
    'CatBoostTrainer',
    'ExperimentTracker',
]


def __getattr__(name: str) -> Any:
    if name == 'BaseTrainer' or name == 'TrainResult':
        module = import_module('.trainer', __name__)
        return getattr(module, name)
    if name == 'CatBoostTrainer':
        return import_module('.catboost_trainer', __name__).CatBoostTrainer
    if name == 'ExperimentTracker':
        return import_module('.experiment', __name__).ExperimentTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
