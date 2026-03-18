"""Utility functions and helpers."""

from importlib import import_module
from typing import Any

from src.utils.logging_config import setup_logging, set_log_level

__all__ = [
    'setup_logging',
    'set_log_level',
    'set_global_seed',
    'normalize_team',
    'TEAM_MAPPINGS',
    'TemporalWeightCalculator',
    'FallbackPredictor',
    'FeatureSelector',
    'TargetPreprocessor',
]


def __getattr__(name: str) -> Any:
    if name == 'set_global_seed':
        return import_module('.reproducibility', __name__).set_global_seed
    if name in {'normalize_team', 'TEAM_MAPPINGS'}:
        module = import_module('.team_mappings', __name__)
        return getattr(module, name)
    if name in {'TemporalWeightCalculator', 'FallbackPredictor', 'FeatureSelector', 'TargetPreprocessor'}:
        module = import_module('.prediction_utils', __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
