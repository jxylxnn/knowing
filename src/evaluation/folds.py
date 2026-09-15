"""Rolling-origin fold construction with game-level isolation.

The public helpers in this module operate on calendar boundaries, but all
frame splitting is performed after collapsing rows to a single date per game.
That makes it impossible for two player rows from the same NBA game to land in
different partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Literal

import pandas as pd


WindowPolicy = Literal["expanding", "bounded"]


@dataclass(frozen=True)
class RollingOriginFold:
    """One chronological fit/validation/test definition."""

    fold_id: str
    fit_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    train_start: date | None = None
    window_policy: WindowPolicy = "expanding"

    def __post_init__(self) -> None:
        if self.window_policy not in {"expanding", "bounded"}:
            raise ValueError("window_policy must be 'expanding' or 'bounded'")
        if not (
            self.fit_end < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.test_end
        ):
            raise ValueError("Fold boundaries must be strictly chronological")
        if self.window_policy == "bounded" and self.train_start is None:
            raise ValueError("A bounded fold requires train_start")
        if self.train_start is not None and self.train_start > self.fit_end:
            raise ValueError("train_start cannot be after fit_end")

    def to_dict(self) -> dict[str, str | None]:
        return {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class FoldPartitions:
    """Indices assigned to a fold without splitting a game."""

    fit: tuple[object, ...]
    validation: tuple[object, ...]
    test: tuple[object, ...]


def rolling_origin_folds(
    dates: Iterable[object] | pd.DataFrame,
    *,
    validation_days: int = 14,
    test_days: int = 28,
    min_train_days: int = 90,
    step_days: int | None = None,
    window_policy: WindowPolicy = "expanding",
    train_window_days: int | None = None,
    date_column: str = "GAME_DATE",
    game_column: str = "GAME_ID",
) -> tuple[RollingOriginFold, ...]:
    """Create chronological outer folds.

    ``dates`` may be an iterable or a player-game frame. Passing a frame is
    preferred because inconsistent dates for a ``GAME_ID`` are rejected up
    front. A bounded policy requires ``train_window_days`` and emits an
    explicit ``train_start`` for every fold.
    """

    _validate_positive("validation_days", validation_days)
    _validate_positive("test_days", test_days)
    _validate_positive("min_train_days", min_train_days)
    if step_days is not None:
        _validate_positive("step_days", step_days)
    if window_policy not in {"expanding", "bounded"}:
        raise ValueError("window_policy must be 'expanding' or 'bounded'")
    if window_policy == "bounded":
        if train_window_days is None:
            raise ValueError("bounded folds require train_window_days")
        _validate_positive("train_window_days", train_window_days)

    values = _game_dates(dates, date_column=date_column, game_column=game_column)
    if values.empty:
        return ()
    start, end = values.min(), values.max()
    step = step_days or test_days
    cursor = start + timedelta(days=min_train_days - 1)
    folds: list[RollingOriginFold] = []
    while cursor + timedelta(days=validation_days + test_days) <= end:
        validation_start = cursor + timedelta(days=1)
        validation_end = validation_start + timedelta(days=validation_days - 1)
        test_start = validation_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        train_start = None
        if window_policy == "bounded":
            train_start = cursor - timedelta(days=int(train_window_days) - 1)
            train_start = max(train_start, start)
        folds.append(
            RollingOriginFold(
                fold_id=f"fold_{len(folds) + 1:02d}",
                train_start=train_start,
                fit_end=cursor,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
                window_policy=window_policy,
            )
        )
        cursor += timedelta(days=step)
    return tuple(folds)


def partition_frame(
    frame: pd.DataFrame,
    fold: RollingOriginFold,
    *,
    date_column: str = "GAME_DATE",
    game_column: str = "GAME_ID",
) -> FoldPartitions:
    """Return row indices for one fold, assigning each game atomically."""

    game_dates = _game_date_frame(
        frame, date_column=date_column, game_column=game_column
    )
    fit_start = fold.train_start or game_dates[date_column].min()
    assignments: dict[str, set[object]] = {
        "fit": set(
            game_dates.loc[
                game_dates[date_column].between(fit_start, fold.fit_end), game_column
            ]
        ),
        "validation": set(
            game_dates.loc[
                game_dates[date_column].between(
                    fold.validation_start, fold.validation_end
                ),
                game_column,
            ]
        ),
        "test": set(
            game_dates.loc[
                game_dates[date_column].between(fold.test_start, fold.test_end),
                game_column,
            ]
        ),
    }
    if any(
        assignments[left] & assignments[right]
        for left, right in (
            ("fit", "validation"),
            ("fit", "test"),
            ("validation", "test"),
        )
    ):
        raise AssertionError("A GAME_ID was assigned to more than one fold partition")

    return FoldPartitions(
        fit=tuple(frame.index[frame[game_column].isin(assignments["fit"])]),
        validation=tuple(
            frame.index[frame[game_column].isin(assignments["validation"])]
        ),
        test=tuple(frame.index[frame[game_column].isin(assignments["test"])]),
    )


def fold_manifest(
    folds: Iterable[RollingOriginFold],
    *,
    source_hash: str | None = None,
) -> dict[str, object]:
    """Return a stable, hash-addressable fold manifest payload."""

    records = [fold.to_dict() for fold in folds]
    body: dict[str, object] = {
        "schema_version": "rolling_origin_folds_v1",
        "source_hash": source_hash,
        "folds": records,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["manifest_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return body


def write_fold_manifest(
    path: str | Path,
    folds: Iterable[RollingOriginFold],
    *,
    source_hash: str | None = None,
) -> Path:
    """Write a fold manifest once; existing content is never replaced."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Fold manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        fold_manifest(folds, source_hash=source_hash), indent=2, sort_keys=True
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=".fold_manifest_", suffix=".json", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _game_dates(
    values: Iterable[object] | pd.DataFrame,
    *,
    date_column: str,
    game_column: str,
) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        return _game_date_frame(
            values, date_column=date_column, game_column=game_column
        )[date_column]
    return pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna().dt.date


def _game_date_frame(
    frame: pd.DataFrame,
    *,
    date_column: str,
    game_column: str,
) -> pd.DataFrame:
    missing = {date_column, game_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Fold frame missing columns: {sorted(missing)}")
    values = frame[[game_column, date_column]].copy()
    values[date_column] = pd.to_datetime(values[date_column], errors="coerce").dt.date
    if values[date_column].isna().any() or values[game_column].isna().any():
        raise ValueError("GAME_ID and GAME_DATE must be populated for fold construction")
    counts = values.groupby(game_column, dropna=False)[date_column].nunique()
    inconsistent = counts[counts != 1]
    if not inconsistent.empty:
        examples = list(inconsistent.index[:10])
        raise ValueError(f"GAME_ID values span multiple dates: {examples}")
    return values.drop_duplicates(game_column).sort_values(
        [date_column, game_column], kind="stable"
    )


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
