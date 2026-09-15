"""Historical execution through the same request service as live forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.evaluation.chronology import digest_payload
from src.forecasting.baseline_backend import V2BaselineBackend
from src.pipeline.forecast_service import ForecastService
from src.pipeline.v2_execution import execute_request


@dataclass(frozen=True)
class RequestReplay:
    forecasts: pd.DataFrame
    samples: pd.DataFrame
    requests: tuple[dict, ...]


def request_record(request, *, seed=42, simulations=1000):
    return {"request": request.to_dict(), "request_id": request.request_id,
            "seed": seed, "simulations": simulations}


def write_requests(path, records):
    payload = {"schema_version": "forecast_requests_v1", "requests": list(records)}
    payload["content_id"] = digest_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an earlier request definition.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return path


def read_requests(path):
    payload = json.loads(Path(path).read_text())
    content_id = payload.pop("content_id", None)
    if (payload.get("schema_version") != "forecast_requests_v1"
            or digest_payload(payload) != content_id):
        raise ValueError("Historical request manifest hash/schema mismatch")
    records = payload["requests"]
    seen = set()
    for record in records:
        request = ForecastRequest(**record["request"])
        if request.request_id != record["request_id"] or request.request_id in seen:
            raise ValueError("Historical request identity mismatch or duplicate")
        seen.add(request.request_id)
        if type(record["seed"]) is not int or type(record["simulations"]) is not int:
            raise ValueError("Historical sampling controls must be integers")
        if record["simulations"] < 1:
            raise ValueError("Historical simulation count must be positive")
    if not records:
        raise ValueError("Historical request manifest is empty")
    return records


def replay_requests(records, *, bundle_dir, data_dir, persist=False):
    """Regenerate immutable requests; supplied outcome labels are never inputs."""
    service = ForecastService(model_backend=V2BaselineBackend(bundle_dir))
    forecasts, samples, requests = [], [], []
    seen = set()
    for record in records:
        request = ForecastRequest(**record["request"])
        if request.request_id != record["request_id"] or request.request_id in seen:
            raise ValueError("Historical request identity mismatch or duplicate")
        seen.add(request.request_id)
        distribution = execute_request(
            service, request, data_dir=data_dir, seed=record["seed"],
            simulations=record["simulations"], persist=persist,
        )
        forecasts.append(distribution.forecasts)
        samples.append(distribution.samples)
        requests.append(record)
    if not forecasts:
        raise ValueError("Replay requires at least one historical request")
    return RequestReplay(pd.concat(forecasts, ignore_index=True),
                         pd.concat(samples, ignore_index=True), tuple(requests))
