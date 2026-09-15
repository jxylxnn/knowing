"""Rebuild a new source snapshot from verified archives and prospective captures."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import tempfile

from src.data.official_capture import REQUIRED
from src.data.snapshots import (
    create_source_snapshot, snapshot_file_path, validate_source_snapshot,
    snapshot_manifest_path, SnapshotFile,
)


def rebuild_snapshot(data_dir, *, base_snapshot_id, captures=(), import_v1=False,
                     historical_schedules=()):
    """Preserve the base snapshot and replace only explicitly captured tables."""
    root = Path(data_dir)
    import_receipt = None
    if import_v1:
        entries, import_receipt = _verified_v1_entries(root, base_snapshot_id)
    else:
        base = validate_source_snapshot(root, base_snapshot_id)
        entries = {}
        for record in base.files:
            if record.relative_path in entries:
                raise ValueError("Base snapshot has ambiguous file names")
            entries[record.relative_path] = (
                snapshot_file_path(root, base_snapshot_id, record), record.source,
                record.fetched_at or base.created_at,
            )
    replaced = set()
    for capture in captures:
        directory = Path(capture)
        manifest = validate_source_snapshot(directory, directory.name)
        tables = [record for record in manifest.files if record.relative_path in REQUIRED]
        if len(tables) != 1 or tables[0].relative_path in replaced:
            raise ValueError("Each capture must supply one distinct official table")
        table = tables[0].relative_path
        replaced.add(table)
        for record in manifest.files:
            name = (record.relative_path if record.relative_path == table
                    else f"capture_{manifest.snapshot_id}_{record.relative_path}")
            entries[name] = (snapshot_file_path(directory, manifest.snapshot_id, record),
                             "official", record.fetched_at or manifest.created_at)
    for capture in historical_schedules:
        directory = Path(capture)
        manifest = validate_source_snapshot(directory, directory.name)
        if sum(record.relative_path == "schedule.csv" for record in manifest.files) != 1:
            raise ValueError("Historical schedule capture requires one schedule.csv")
        for record in manifest.files:
            name = (f"historical_schedule_{manifest.snapshot_id}.csv"
                    if record.relative_path == "schedule.csv"
                    else f"capture_{manifest.snapshot_id}_{record.relative_path}")
            if name in entries:
                raise ValueError("Historical capture is already present in base snapshot")
            entries[name] = (snapshot_file_path(directory, manifest.snapshot_id, record),
                             "official", record.fetched_at or manifest.created_at)
    with tempfile.TemporaryDirectory(prefix=".rebuild_inputs_", dir=root) as temporary:
        stage = Path(temporary)
        sources, fetched = {}, {}
        for name, (path, source, observed) in entries.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            sources.setdefault(source, []).append(name)
            fetched[f"{source}/{name}"] = observed
        if import_receipt is not None:
            (stage / "legacy_import_receipt.json").write_text(
                json.dumps(import_receipt, sort_keys=True, indent=2) + "\n"
            )
            sources.setdefault("import", []).append("legacy_import_receipt.json")
        return create_source_snapshot(root, input_dir=stage, sources=sources, fetched_at=fetched)


def _verified_v1_entries(root, snapshot_id):
    """Explicit forward import only; runtime readers continue to reject v1."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", snapshot_id):
        raise ValueError("Invalid legacy snapshot identity")
    path = snapshot_manifest_path(root, snapshot_id)
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (manifest.get("schema_version") != "source_snapshot_v1"
            or manifest.get("snapshot_id") != snapshot_id or not manifest.get("files")):
        raise ValueError("Explicit import requires a named v1 source manifest")
    observed = datetime.now(timezone.utc).isoformat()
    entries = {}
    by_source = {}
    for item in manifest["files"]:
        record = SnapshotFile(**item)
        source_path = snapshot_file_path(root, snapshot_id, record)
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if source_path.stat().st_size != record.size_bytes or digest.hexdigest() != record.sha256:
            raise ValueError("Legacy source checksum mismatch; import refused")
        if record.relative_path in entries:
            raise ValueError("Legacy source contains ambiguous file names")
        entries[record.relative_path] = (source_path, record.source, observed)
        by_source.setdefault(record.source, set()).add(record.relative_path)
    for source, expected in by_source.items():
        directory = root / "raw" / source / snapshot_id
        actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError("Legacy source directory differs from its manifest")
    return entries, {"schema_version": "explicit_source_import_v1",
                     "original_snapshot_id": snapshot_id,
                     "original_manifest_sha256": hashlib.sha256(payload).hexdigest(),
                     "original_created_at": manifest.get("created_at"),
                     "imported_at": observed, "historical_availability_claim": False}
