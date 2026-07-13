"""Point-in-time feature materialization for scheduled-game forecasts.

This module intentionally has a small adapter surface.  A snapshot store can
be backed by CSV, Parquet, or an in-memory test fixture as long as it exposes
one of the read methods documented in :func:`_read_snapshot`.  Feature output
is selected by exact registered ``FeatureSpec.name`` values; names are never
used as a leakage allow-list.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd

from src.contracts.forecast import ForecastRequest

logger = logging.getLogger(__name__)

FeatureProducer = Callable[..., Any]


class FeatureMaterializationError(ValueError):
    """Raised when a point-in-time feature contract cannot be satisfied."""


@dataclass(frozen=True)
class FeatureSpec:
    """Provenance contract for one materialized feature column."""

    name: str
    group: str
    dtype: str
    entity_keys: tuple[str, ...]
    event_time_column: str
    max_lookback_days: int | None
    required_sources: tuple[str, ...]
    allowed_horizons: tuple[str, ...]
    missing_policy: str
    version: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise FeatureMaterializationError("FeatureSpec.name is required")
        if not self.group.strip():
            raise FeatureMaterializationError("FeatureSpec.group is required")
        if self.max_lookback_days is not None and self.max_lookback_days < 0:
            raise FeatureMaterializationError("max_lookback_days cannot be negative")
        if not self.allowed_horizons:
            raise FeatureMaterializationError(
                f"FeatureSpec {self.name} must declare allowed_horizons"
            )


class FeatureRegistry:
    """Registry of exact feature outputs and their producing callables."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}
        self._producers: dict[str, FeatureProducer] = {}

    def register(self, spec: FeatureSpec, producer: FeatureProducer | None = None) -> None:
        if spec.name in self._specs:
            raise FeatureMaterializationError(f"Feature already registered: {spec.name}")
        self._specs[spec.name] = spec
        if producer is not None:
            self._producers[spec.name] = producer

    def add(self, spec: FeatureSpec, producer: FeatureProducer | None = None) -> None:
        """Alias for ``register`` for small extension modules."""
        self.register(spec, producer)

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise FeatureMaterializationError(f"Unregistered feature output: {name}") from exc

    def specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._specs.values())

    def producer(self, name: str) -> FeatureProducer | None:
        return self._producers.get(name)


@dataclass
class ForecastFrame:
    """Small typed wrapper around a materialized point-in-time DataFrame."""

    data: pd.DataFrame
    request: ForecastRequest

    def to_pandas(self) -> pd.DataFrame:
        return self.data.copy()

    @property
    def columns(self):
        return self.data.columns

    def __len__(self) -> int:
        return len(self.data)


class FeatureMaterializer:
    """Materialize exact registered features as of a forecast cutoff."""

    REQUEST_COLUMNS = {
        "REQUEST_ID", "GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID",
        "FORECAST_CUTOFF", "HORIZON", "SOURCE_SNAPSHOT_ID", "MODEL_BUNDLE_ID",
        "SCENARIO",
    }

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        *,
        strict_point_in_time: bool = True,
    ) -> None:
        self.registry = registry or FeatureRegistry()
        self.strict_point_in_time = strict_point_in_time

    def materialize_training_examples(
        self,
        requests: Iterable[ForecastRequest],
        snapshot_store: Any,
    ) -> pd.DataFrame:
        """Materialize one point-in-time feature frame per training request."""
        frames = [
            self.materialize_forecast(request, snapshot_store)
            for request in requests
        ]
        if not frames:
            return pd.DataFrame(columns=sorted(self.REQUEST_COLUMNS))
        return pd.concat(frames, ignore_index=True, sort=False)

    def materialize_forecast(
        self,
        request: ForecastRequest,
        snapshot_store: Any,
    ) -> pd.DataFrame:
        """Materialize registered features for one immutable request."""
        if not isinstance(request, ForecastRequest):
            raise FeatureMaterializationError("request must be a ForecastRequest")

        source = _read_snapshot(snapshot_store, request)
        source = _as_frame(source)
        source = self._filter_point_in_time(source, request)

        rows: dict[str, Any] = {}
        for spec in self.registry.specs():
            if request.horizon not in spec.allowed_horizons:
                continue
            value = self._materialize_spec(spec, source, request, snapshot_store)
            rows[spec.name] = _coerce_feature(value, spec, len(source))

        # A snapshot may already contain a registered feature.  This path is
        # useful for Parquet feature partitions and still remains exact-name
        # safe because only registry entries are copied.
        frame = pd.DataFrame(index=source.index)
        for name, values in rows.items():
            frame[name] = values

        request_values = {
            "REQUEST_ID": request.request_id,
            "GAME_ID": request.game_id,
            "GAME_DATE": request.game_date.isoformat(),
            "HOME_TEAM_ID": request.home_team_id,
            "AWAY_TEAM_ID": request.away_team_id,
            "FORECAST_CUTOFF": request.forecast_cutoff.isoformat(),
            "HORIZON": request.horizon,
            "SOURCE_SNAPSHOT_ID": request.source_snapshot_id,
            "MODEL_BUNDLE_ID": request.model_bundle_id,
            "SCENARIO": request.scenario,
        }
        if frame.empty:
            frame = pd.DataFrame([request_values])
        else:
            for column, value in request_values.items():
                frame[column] = value
        return frame.reset_index(drop=True)

    def _materialize_spec(
        self,
        spec: FeatureSpec,
        source: pd.DataFrame,
        request: ForecastRequest,
        snapshot_store: Any,
    ) -> Any:
        producer = self.registry.producer(spec.name)
        if producer is not None:
            try:
                signature = inspect.signature(producer)
                if len(signature.parameters) >= 3:
                    value = producer(request, source, snapshot_store)
                elif len(signature.parameters) == 2:
                    value = producer(request, source)
                else:
                    value = producer(source)
            except (TypeError, ValueError):
                value = producer(request, source)
        elif spec.name in source.columns:
            value = source[spec.name]
        elif spec.missing_policy in {"zero", "fill_zero"}:
            value = 0
        elif spec.missing_policy in {"null", "nullable"}:
            value = None
        else:
            raise FeatureMaterializationError(
                f"No producer or snapshot column for registered feature {spec.name}"
            )
        return value

    def _filter_point_in_time(
        self,
        source: pd.DataFrame,
        request: ForecastRequest,
    ) -> pd.DataFrame:
        if source.empty:
            return source.copy()
        result = source.copy()
        cutoff = pd.Timestamp(request.forecast_cutoff)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)

        if "available_at" in result.columns:
            available = pd.to_datetime(result["available_at"], errors="coerce")
            if available.isna().any():
                raise FeatureMaterializationError("available_at contains malformed timestamps")
            if getattr(available.dt, "tz", None) is not None:
                available = available.dt.tz_convert(None)
            result = result.loc[available <= cutoff].copy()
        elif self.strict_point_in_time and "event_time" in result.columns:
            raise FeatureMaterializationError(
                "Snapshot rows with event_time must declare available_at for point-in-time joins"
            )

        # No feature may use a row whose event time is after the scheduled
        # game.  available_at controls observability; event_time controls the
        # event itself.
        for spec in self.registry.specs():
            column = spec.event_time_column
            if column and column in result.columns:
                event_time = pd.to_datetime(result[column], errors="coerce")
                if event_time.isna().any():
                    raise FeatureMaterializationError(
                        f"{spec.name} has malformed {column} timestamps"
                    )
                if getattr(event_time.dt, "tz", None) is not None:
                    event_time = event_time.dt.tz_convert(None)
                result = result.loc[event_time < pd.Timestamp(request.game_date)].copy()

        return result


def _read_snapshot(snapshot_store: Any, request: ForecastRequest) -> Any:
    """Read a snapshot from common local-store adapters.

    Supported adapters expose ``read_snapshot``, ``load_snapshot``, ``read``,
    or ``load``.  A mapping may be keyed by snapshot id, and a DataFrame is
    accepted for tests and small local workflows.
    """
    if isinstance(snapshot_store, pd.DataFrame):
        return snapshot_store
    if isinstance(snapshot_store, Mapping):
        for key in (request.source_snapshot_id, "forecast", "data"):
            if key in snapshot_store:
                return snapshot_store[key]
        raise FeatureMaterializationError(
            f"Snapshot {request.source_snapshot_id!r} not found in snapshot mapping"
        )

    for method_name in ("read_snapshot", "load_snapshot", "read", "load"):
        method = getattr(snapshot_store, method_name, None)
        if method is None:
            continue
        for args in ((request.source_snapshot_id,), (request,)):
            try:
                return method(*args)
            except TypeError:
                continue

    raise FeatureMaterializationError(
        "snapshot_store must be a DataFrame, mapping, or expose a snapshot read method"
    )


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        return pd.DataFrame(value)
    raise FeatureMaterializationError(
        f"Snapshot reader returned unsupported type: {type(value).__name__}"
    )


def _coerce_feature(value: Any, spec: FeatureSpec, size: int) -> Any:
    if isinstance(value, pd.Series):
        if len(value) == size:
            return value.reset_index(drop=True)
        if len(value) == 1:
            return [value.iloc[0]] * size
        raise FeatureMaterializationError(
            f"Producer for {spec.name} returned {len(value)} rows; expected {size}"
        )
    if isinstance(value, pd.DataFrame):
        if spec.name not in value.columns:
            raise FeatureMaterializationError(
                f"Producer for {spec.name} did not return its registered column"
            )
        return _coerce_feature(value[spec.name], spec, size)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        if len(value) not in (1, size):
            raise FeatureMaterializationError(
                f"Producer for {spec.name} returned {len(value)} values; expected {size}"
            )
        if len(value) == 1:
            return [value[0]] * size
        return value
    return [value] * max(1, size)


_DEFAULT_MATERIALIZER = FeatureMaterializer(strict_point_in_time=False)


def materialize_training_examples(requests, snapshot_store) -> pd.DataFrame:
    """Module-level training entry point required by the architecture plan."""
    return _DEFAULT_MATERIALIZER.materialize_training_examples(requests, snapshot_store)


def materialize_forecast(request: ForecastRequest, snapshot_store) -> pd.DataFrame:
    """Module-level forecast entry point required by the architecture plan."""
    return _DEFAULT_MATERIALIZER.materialize_forecast(request, snapshot_store)

