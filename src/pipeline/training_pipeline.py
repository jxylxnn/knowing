"""Compatibility shim for the legacy pipeline import path.

The supported training pipeline lives in :mod:`src.training.pipeline` and
implements the CatBoost + Transformer workflow from the diagram. This module
exists only so older imports keep working without routing back into the legacy
joint-NN / LSTM / GNN training stack.
"""

from src.training.pipeline import TrainingPipeline, create_pipeline

__all__ = ["TrainingPipeline", "create_pipeline"]

