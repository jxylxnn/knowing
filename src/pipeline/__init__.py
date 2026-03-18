"""Pipeline module for NBA prediction system.

The training and prediction modules pull in heavy dependencies, including
PyTorch-based model wrappers. Import them lazily so lightweight imports such
as ``from src.pipeline.data_pipeline import DataPipeline`` do not crash during
test collection on environments where Torch initialization is fragile.
"""

from importlib import import_module
from typing import Any

from .data_pipeline import DataPipeline

__all__ = ['DataPipeline', 'TrainingPipeline', 'PredictionService']


def __getattr__(name: str) -> Any:
    if name == 'TrainingPipeline':
        return import_module('.training_pipeline', __name__).TrainingPipeline
    if name == 'PredictionService':
        return import_module('.prediction_service', __name__).PredictionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
