"""Pipeline package.

Provides DataPipeline, TrainingPipeline (delegated to src.training.pipeline),
and PredictionService. This package serves as the primary import surface for
pipeline classes.
"""

from importlib import import_module
from typing import Any

__all__ = ['DataPipeline', 'TrainingPipeline', 'PredictionService']


def __getattr__(name: str) -> Any:
    if name == 'DataPipeline':
        return import_module('.data_pipeline', __name__).DataPipeline
    if name == 'TrainingPipeline':
        return import_module('src.training.pipeline').TrainingPipeline
    if name == 'PredictionService':
        return import_module('.prediction_service', __name__).PredictionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
