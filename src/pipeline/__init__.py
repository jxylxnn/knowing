"""Legacy pipeline package.

The supported training path lives in :mod:`src.training.pipeline` and keeps the
CatBoost + Transformer-only flow from the diagram. This package remains as a
compatibility surface for older imports, but it should not be the place new
code reaches for training logic.
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
