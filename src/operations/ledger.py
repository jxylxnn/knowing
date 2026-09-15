"""Append-only official forecast and reconciliation storage for Model v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.contracts.forecast import validate_forecast_frame
from src.evaluation.replay import score_replay
from src.utils.file_lock import exclusive_file_lock


class OfficialForecastLedger:
    def __init__(self, root: str | Path = "data/ledger") -> None:
        self.root = Path(root)

    def write_forecast(self, frame: pd.DataFrame) -> Path:
        """Persist one request exactly once, rejecting contradictory retries."""

        validate_forecast_frame(frame)
        if not frame["SCENARIO"].astype(str).eq("official").all():
            raise ValueError("Only strict official forecasts may enter the official ledger")
        request_ids = tuple(frame["REQUEST_ID"].astype(str).drop_duplicates())
        if len(request_ids) != 1:
            raise ValueError("One ledger payload must contain exactly one REQUEST_ID")
        request_id = request_ids[0]
        destination = self.root / "forecasts" / f"{request_id}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = destination.with_name(f".{destination.name}.lock")
        with exclusive_file_lock(lock_path):
            if destination.is_file():
                existing = pd.read_parquet(destination)
                try:
                    assert_frame_equal(
                        existing.reset_index(drop=True),
                        frame.reset_index(drop=True),
                        check_dtype=False,
                        check_exact=True,
                    )
                except AssertionError as exc:
                    raise ValueError(
                        f"Conflicting immutable forecast payload for REQUEST_ID {request_id}"
                    ) from exc
                return destination
            _atomic_parquet(destination, frame)
        return destination

    def load_forecasts(self) -> pd.DataFrame:
        files = sorted((self.root / "forecasts").glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(path) for path in files]
        for frame in frames:
            validate_forecast_frame(frame)
        return pd.concat(frames, ignore_index=True)

    def reconcile(self, actuals: pd.DataFrame, *, game_date: str) -> Path:
        """Write a content-addressed reconciliation revision for one date."""

        forecasts = self.load_forecasts()
        if forecasts.empty:
            raise FileNotFoundError("No official forecasts are available to reconcile")
        if "GAME_DATE" in actuals:
            dates = pd.to_datetime(actuals["GAME_DATE"], errors="raise").dt.date.astype(str)
            actuals = actuals.loc[dates.eq(str(game_date))].copy()
        if actuals.empty:
            raise ValueError(f"No actual rows found for {game_date}")
        # The ledger defines the expected population, including games for which
        # actuals are completely absent. Actual-file coverage cannot shrink it.
        dates = pd.to_datetime(forecasts["GAME_DATE"], errors="raise").dt.date.astype(str)
        selected = forecasts.loc[dates.eq(str(game_date))].copy()
        if selected.empty:
            raise ValueError(f"No ledger forecasts match actuals for {game_date}")
        normalized_actuals = _normalized_actuals(actuals, selected)
        score = score_replay(selected, normalized_actuals).to_dict()
        actual_digest = _frame_digest(normalized_actuals)
        forecast_payloads = []
        for request_id, request_frame in selected.groupby(
            "REQUEST_ID", sort=True
        ):
            forecast_payloads.append({
                "request_id": str(request_id),
                "digest": _frame_digest(request_frame),
                "rows": int(len(request_frame)),
            })
        payload = {
            "schema_version": "reconciliation_v2",
            "game_date": str(game_date),
            "score": score,
            "forecast_request_ids": sorted(selected["REQUEST_ID"].astype(str).unique()),
            "forecast_payloads": forecast_payloads,
            "actuals_digest": actual_digest,
            "actual_source_digest": actual_digest,
            "actuals": {
                "schema_version": "normalized_actuals_v1",
                "digest": actual_digest,
                "columns": list(normalized_actuals.columns),
                "rows": _frame_records(normalized_actuals),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        revision = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        directory = self.root / "reconciliations" / f"date={game_date}"
        destination = directory / f"{revision}.json"
        if destination.is_file():
            return destination
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return destination


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".ledger_", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".reconciliation_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _normalized_actuals(actuals: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return the exact wide actual rows used by replay in stable form."""

    stats = sorted(set(forecasts["STAT"].astype(str).str.upper()) & set(actuals.columns))
    columns = ["GAME_ID", "PLAYER_ID", *stats]
    if "GAME_DATE" in actuals:
        columns.append("GAME_DATE")
    normalized = actuals.loc[:, columns].copy()
    normalized["GAME_ID"] = normalized["GAME_ID"].astype(str)
    normalized["PLAYER_ID"] = normalized["PLAYER_ID"].astype(str)
    for stat in stats:
        numeric = pd.to_numeric(normalized[stat], errors="coerce")
        observed = normalized[stat].notna()
        observed_values = numeric.loc[observed]
        if (
            observed_values.isna().any()
            or not np.isfinite(observed_values.to_numpy(dtype=float)).all()
            or (observed_values < 0).any()
        ):
            raise ValueError(
                f"Actual {stat} counts must be finite and nonnegative or missing"
            )
        normalized[stat] = numeric.astype(float)
    if "GAME_DATE" in normalized:
        normalized["GAME_DATE"] = pd.to_datetime(
            normalized["GAME_DATE"], errors="raise", utc=True
        ).dt.date.astype(str)

    selected_keys = forecasts[["GAME_ID", "PLAYER_ID"]].copy()
    selected_keys["GAME_ID"] = selected_keys["GAME_ID"].astype(str)
    selected_keys["PLAYER_ID"] = selected_keys["PLAYER_ID"].astype(str)
    normalized = normalized.merge(
        selected_keys.drop_duplicates(),
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
        validate="many_to_many",
    )
    normalized = normalized.sort_values(
        ["GAME_ID", "PLAYER_ID", *stats], kind="stable"
    ).reset_index(drop=True)
    return normalized


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, row in frame.iterrows():
        records.append({column: _json_value(value) for column, value in row.items()})
    return sorted(
        records,
        key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":")),
    )


def _frame_digest(frame: pd.DataFrame) -> str:
    encoded = json.dumps(
        {"columns": list(frame.columns), "rows": _frame_records(frame)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        return value.isoformat()
    return value


__all__ = ["OfficialForecastLedger"]
