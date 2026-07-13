"""Backward-compatible facade for the canonical :class:`ForecastService`."""

from __future__ import annotations

from typing import Any, Optional

from src.pipeline.forecast_service import ForecastService


class PredictionService(ForecastService):
    """Compatibility name retained for older integrations.

    New code should import ``ForecastService``.  The old constructor accepting
    ``Config`` and an optional ``TrainingPipeline`` remains supported.
    """

    def __init__(self, config: Any, training_pipeline: Optional[Any] = None):
        self.config = config
        if training_pipeline is None:
            data = getattr(config, "data", None)
            super().__init__(
                config=config,
                data_dir=str(getattr(data, "data_dir", "data")),
                models_dir=str(getattr(data, "models_dir", "models")),
            )
            self.pipeline = self.backend
        else:
            super().__init__(
                model_backend=training_pipeline,
                config=config,
            )
            self.pipeline = training_pipeline
