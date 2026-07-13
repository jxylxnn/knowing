"""Compatibility import surface for canonical forecast frame contracts."""

from src.contracts.forecast import (
    CANONICAL_FORECAST_COLUMNS,
    FORECAST_HORIZONS,
    ForecastRequest,
    forecast_request_from_mapping,
    validate_forecast_frame,
)

__all__ = [
    "CANONICAL_FORECAST_COLUMNS",
    "FORECAST_HORIZONS",
    "ForecastRequest",
    "forecast_request_from_mapping",
    "validate_forecast_frame",
]

