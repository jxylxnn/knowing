"""Immutable, checksummed, multi-source snapshots for point-in-time forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping
from uuid import uuid4

from src.contracts.errors import ContractError
from src.contracts.sources import (
    SOURCE_SNAPSHOT_SCHEMA_VERSION,
    SnapshotFile,
    SourceSnapshotManifest,
    parse_aware_datetime,
)


DEFAULT_SNAPSHOT_FILES = (
    "nba_players.csv",
    "nba_games.csv",
    "schedule.csv",
    "nba_schedule.csv",
    "roster_membership.csv",
    "rosters.csv",
    "player_game_eligibility.csv",
    "player_status_snapshots.csv",
    "injury_history.csv",
    "lineup_snapshots.csv",
    "lineups.csv",
    "odds_snapshots.csv",
    "odds.csv",
    "player_bios.csv",
    "advanced_tracking.csv",
)


def create_source_snapshot(
    data_dir: str | Path,
    *,
    files: Iterable[str] = DEFAULT_SNAPSHOT_FILES,
    source: str = "core",
    sources: Mapping[str, Iterable[str]] | None = None,
    snapshot_id: str | None = None,
    created_at: datetime | None = None,
    fetched_at: Mapping[str, datetime | str] | None = None,
    input_dir: str | Path | None = None,
) -> SourceSnapshotManifest:
    """Copy present working files into a new append-only snapshot.

    ``files``/``source`` preserves the legacy single-source interface.
    ``sources`` enables a snapshot to contain independently named source
    collections, for example ``{"core": (...), "injuries": (...)}``.
    Source entries use paths relative to ``data_dir``; absent optional files
    are omitted and never synthesized.
    """

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    input_root = Path(input_dir).resolve() if input_dir is not None else root
    if not input_root.is_dir():
        raise FileNotFoundError(f"Snapshot input directory does not exist: {input_root}")
    timestamp = created_at or datetime.now(timezone.utc)
    timestamp = parse_aware_datetime(timestamp, field="created_at").astimezone(
        timezone.utc
    )
    resolved_id = snapshot_id or _new_snapshot_id(timestamp)

    requested_sources = _normalize_sources(files, source, sources)
    core_files = set(requested_sources.get("core", ()))
    if {"nba_players.csv", "nba_games.csv"}.issubset(core_files):
        missing_core = sorted(
            name
            for name in ("nba_players.csv", "nba_games.csv")
            if not (input_root / name).is_file()
        )
        if missing_core:
            raise FileNotFoundError(
                "Cannot create source snapshot; missing required core file(s): "
                + ", ".join(missing_core)
            )

    manifest_path = snapshot_manifest_path(root, resolved_id)
    target_dirs = {
        source_name: root / "raw" / source_name / resolved_id
        for source_name in requested_sources
    }
    if manifest_path.exists() or any(path.exists() for path in target_dirs.values()):
        raise FileExistsError(f"Source snapshot already exists: {resolved_id}")

    (root / "raw").mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".snapshot_stage_", dir=root / "raw"))
    committed: list[Path] = []
    try:
        records: list[SnapshotFile] = []
        for source_name, relative_paths in requested_sources.items():
            source_stage = stage_root / source_name
            for relative_path in relative_paths:
                origin = _safe_source_path(input_root, relative_path)
                if not origin.is_file():
                    continue
                destination = source_stage / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
                observed = _file_fetched_at(
                    fetched_at or {}, source_name, relative_path, timestamp
                )
                records.append(
                    SnapshotFile(
                        relative_path=relative_path,
                        source=source_name,
                        size_bytes=destination.stat().st_size,
                        sha256=_sha256(destination),
                        fetched_at=observed.isoformat(),
                    )
                )
        if not records:
            raise FileNotFoundError("No requested source files exist to snapshot")

        manifest = SourceSnapshotManifest(
            snapshot_id=resolved_id,
            created_at=timestamp.isoformat(),
            schema_version=SOURCE_SNAPSHOT_SCHEMA_VERSION,
            files=tuple(records),
        )
        for source_name in manifest.sources:
            staged = stage_root / source_name
            target = target_dirs[source_name]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
            committed.append(target)
        _atomic_write_json(manifest_path, manifest.to_dict())
        return manifest
    except Exception:
        # Without a manifest, a partial directory is not a valid snapshot.
        # Clean only directories created by this invocation.
        for path in committed:
            if path.exists():
                shutil.rmtree(path)
        raise
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


def load_source_snapshot_manifest(
    data_dir: str | Path,
    snapshot_id: str,
) -> SourceSnapshotManifest:
    """Load a named manifest; absence is always a hard failure."""

    path = snapshot_manifest_path(data_dir, snapshot_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Source snapshot manifest does not exist: {snapshot_id}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Source snapshot manifest is unreadable: {path}") from exc
    manifest = SourceSnapshotManifest.from_dict(payload)
    if manifest.snapshot_id != str(snapshot_id):
        raise ContractError("Source snapshot manifest ID does not match its file name")
    return manifest


def validate_source_snapshot(
    data_dir: str | Path,
    snapshot_id: str,
    *,
    source: str | None = None,
    forecast_cutoff: datetime | str | None = None,
) -> SourceSnapshotManifest:
    """Checksum-validate a named snapshot and optionally enforce a cutoff."""

    root = Path(data_dir).resolve()
    manifest = load_source_snapshot_manifest(root, snapshot_id)
    if forecast_cutoff is not None:
        manifest.assert_available_at(forecast_cutoff)
    selected = [
        record for record in manifest.files
        if source is None or record.source == source
    ]
    if source is not None and not selected:
        raise ContractError(
            f"Source snapshot {snapshot_id!r} does not contain source {source!r}"
        )

    by_source: dict[str, set[str]] = {}
    for record in selected:
        path = snapshot_file_path(root, snapshot_id, record)
        if not path.is_file():
            raise ContractError(
                f"Source snapshot file is missing: "
                f"{record.source}/{record.relative_path}"
            )
        if path.stat().st_size != record.size_bytes or _sha256(path) != record.sha256:
            raise ValueError(
                f"Source snapshot file checksum mismatch: "
                f"{record.source}/{record.relative_path}"
            )
        by_source.setdefault(record.source, set()).add(record.relative_path)

    # Extra files would make the on-disk snapshot differ from its manifest.
    for source_name, expected in by_source.items():
        source_dir = root / "raw" / source_name / str(snapshot_id)
        actual = {
            path.relative_to(source_dir).as_posix()
            for path in source_dir.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            details = []
            if extra:
                details.append("unmanifested=" + ",".join(extra))
            if missing:
                details.append("missing=" + ",".join(missing))
            raise ContractError(
                f"Source snapshot directory differs from manifest for {source_name}: "
                + "; ".join(details)
            )
    return manifest


def snapshot_manifest_path(data_dir: str | Path, snapshot_id: str) -> Path:
    return (
        Path(data_dir).resolve()
        / "manifests"
        / f"{SourceSnapshotManifest.FILE_PREFIX}{snapshot_id}.json"
    )


def snapshot_file_path(
    data_dir: str | Path,
    snapshot_id: str,
    record: SnapshotFile,
) -> Path:
    root = Path(data_dir).resolve()
    path = root / "raw" / record.source / str(snapshot_id) / record.relative_path
    resolved = path.resolve()
    expected_root = (root / "raw" / record.source / str(snapshot_id)).resolve()
    if not resolved.is_relative_to(expected_root):
        raise ContractError("Snapshot manifest contains an unsafe path")
    return resolved


def find_snapshot_file(
    data_dir: str | Path,
    snapshot_id: str,
    relative_names: Iterable[str],
    *,
    required: bool = False,
) -> Path | None:
    """Find one explicitly named file in a validated multi-source manifest."""

    manifest = validate_source_snapshot(data_dir, snapshot_id)
    names = tuple(relative_names)
    matches = [record for record in manifest.files if record.relative_path in names]
    if len(matches) > 1:
        locations = ", ".join(
            f"{record.source}/{record.relative_path}" for record in matches
        )
        raise ContractError(f"Ambiguous snapshot source file: {locations}")
    if not matches:
        if required:
            raise FileNotFoundError(
                f"Snapshot {snapshot_id!r} lacks required file; expected one of {names}"
            )
        return None
    return snapshot_file_path(data_dir, snapshot_id, matches[0])


def _normalize_sources(
    files: Iterable[str],
    source: str,
    sources: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    raw = sources if sources is not None else {source: files}
    normalized: dict[str, tuple[str, ...]] = {}
    for source_name, source_files in raw.items():
        name = str(source_name).strip()
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid snapshot source name: {source_name!r}")
        paths = tuple(dict.fromkeys(str(path) for path in source_files))
        _validate_relative_paths(paths)
        normalized[name] = paths
    if not normalized:
        raise ValueError("At least one snapshot source must be configured")
    return normalized


def _safe_source_path(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError(f"Snapshot path escapes the data directory: {relative_path!r}")
    return resolved


def _file_fetched_at(
    configured: Mapping[str, datetime | str],
    source: str,
    relative_path: str,
    default: datetime,
) -> datetime:
    raw = configured.get(f"{source}/{relative_path}", configured.get(source, default))
    parsed = parse_aware_datetime(raw, field="fetched_at").astimezone(timezone.utc)
    if parsed > default:
        raise ContractError("Source fetched_at cannot be later than snapshot created_at")
    return parsed


def _new_snapshot_id(timestamp: datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]


def _validate_relative_paths(paths: Iterable[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts or raw_path in {"", "."}:
            raise ValueError(f"Snapshot paths must be safe relative paths: {raw_path!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=".snapshot_manifest_", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


__all__ = [
    "DEFAULT_SNAPSHOT_FILES",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "SnapshotFile",
    "SourceSnapshotManifest",
    "create_source_snapshot",
    "find_snapshot_file",
    "load_source_snapshot_manifest",
    "snapshot_file_path",
    "snapshot_manifest_path",
    "validate_source_snapshot",
]
