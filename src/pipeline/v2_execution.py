"""Shared live/replay execution for a persisted scheduled forecast request."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.scheduled import materialize_scheduled_rows
from src.features.snapshot_inputs import (
    assert_snapshot_game, load_official_roster, load_request_history, load_request_status,
)
from src.operations.forecast_runs import ForecastRunStore


def execute_request(service, request, *, data_dir, simulations=1000, seed=42,
                    persist=False):
    """Resolve only request-bound inputs and produce one recoverable law."""
    if service.backend.model_version != request.model_bundle_id:
        raise ValueError("Execution request does not match loaded candidate")
    training_cutoff = getattr(service.backend, "learning_cutoff",
                              getattr(service.backend, "training_cutoff", None))
    if training_cutoff is not None and pd.Timestamp(training_cutoff).date() >= request.game_date:
        raise ValueError("Candidate training overlaps the requested game")

    def compute():
        assert_snapshot_game(data_dir, request)
        history = load_request_history(data_dir, request)
        if request.scenario == "degraded_observed_roster":
            from src.simulation.v2_runner import latest_observed_roster

            roster = latest_observed_roster(history, request.game_date)
            quality = "DEGRADED_OBSERVED_ROSTER"
        else:
            roster = load_official_roster(data_dir, request)
            quality = "OFFICIAL_ROSTER"
        statuses = load_request_status(data_dir, request, roster)
        contexts = materialize_scheduled_rows(request, roster, history, statuses=statuses)
        contexts["DATA_QUALITY"] = quality
        return service.predict_game_distribution(
            request, contexts, simulations=simulations, seed=seed
        )

    if persist:
        return ForecastRunStore(Path(data_dir) / "forecast_runs").get_or_create(request, compute)
    return compute()
