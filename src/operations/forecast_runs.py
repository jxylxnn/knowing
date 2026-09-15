"""Request-level immutable forecast recovery, including the original timestamp."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.contracts.forecast import ForecastRequest, validate_forecast_frame
from src.forecasting.distribution import GameDistribution
from src.operations.forecast_export import export_forecast_run, _digest
from src.utils.file_lock import exclusive_file_lock


def _hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ForecastRunStore:
    def __init__(self, root):
        self.root = Path(root)

    def load(self, request: ForecastRequest):
        path = self.root / "requests" / f"{request.request_id}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        digest = payload.pop("record_hash", None)
        if digest != _hash(payload) or payload["request"] != request.to_dict():
            raise ValueError("Persisted forecast request identity mismatch")
        artifact_id = payload["artifact_id"]
        if not isinstance(artifact_id, str) or len(artifact_id) != 64 or any(
                char not in "0123456789abcdef" for char in artifact_id):
            raise ValueError("Invalid persisted forecast artifact identity")
        artifact = self.root / "artifacts" / artifact_id
        for name, expected in payload["files"].items():
            if name not in ("forecasts.parquet", "samples.parquet"):
                raise ValueError("Unexpected forecast artifact file")
            if _digest(artifact / name) != expected:
                raise ValueError("Persisted forecast artifact checksum mismatch")
        if set(payload["files"]) != {"forecasts.parquet", "samples.parquet"}:
            raise ValueError("Incomplete persisted forecast artifact")
        forecasts = pd.read_parquet(artifact / "forecasts.parquet")
        samples = pd.read_parquet(artifact / "samples.parquet")
        validate_forecast_frame(forecasts)
        for frame in (forecasts, samples):
            if not frame.REQUEST_ID.eq(request.request_id).all():
                raise ValueError("Persisted forecast rows belong to another request")
        return GameDistribution(request, forecasts, samples,
                                int(payload["seed"]), int(payload["simulations"]))

    def load_request_id(self, request_id):
        if (not isinstance(request_id, str) or len(request_id) != 24
                or any(char not in "0123456789abcdef" for char in request_id)):
            raise ValueError("Invalid persisted request ID")
        path = self.root / "requests" / f"{request_id}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        request = ForecastRequest(**payload["request"])
        if request.request_id != request_id:
            raise ValueError("Persisted request file identity mismatch")
        return self.load(request)

    def get_or_create(self, request: ForecastRequest, compute):
        """A retry returns original bytes without rerunning the model."""
        index = self.root / "requests"
        index.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(index / f".{request.request_id}.lock"):
            existing = self.load(request)
            if existing is not None:
                return existing
            result = compute()
            if result.request != request:
                raise ValueError("Computed distribution belongs to another request")
            forecast_path, sample_path = export_forecast_run(
                self.root / "artifacts", result.forecasts, result.samples
            )
            payload = {"request": request.to_dict(), "seed": result.seed,
                       "simulations": result.simulations,
                       "artifact_id": forecast_path.parent.name,
                       "files": {forecast_path.name: _digest(forecast_path),
                                 sample_path.name: _digest(sample_path)}}
            payload["record_hash"] = _hash(payload)
            path = index / f"{request.request_id}.json"
            temporary = path.with_suffix(".pending")
            try:
                temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return result
