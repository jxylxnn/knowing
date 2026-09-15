"""Contracts for immutable, point-in-time source snapshots.

Snapshots are the audit boundary between mutable working files and a forecast.
The contract intentionally records one checksum and observation timestamp per
source file so a request can prove that every named input existed by its
forecast cutoff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import re
from typing import Any, Mapping

from src.contracts.errors import ContractError


SOURCE_SNAPSHOT_SCHEMA_VERSION = "source_snapshot_v2"
_SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_aware_datetime(value: Any, *, field: str) -> datetime:
    """Parse *value* as a timezone-aware :class:`datetime`.

    This helper lives in the contract module so snapshot creation, request
    validation, and replay all apply identical timezone rules.
    """

    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{field} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{field} must be timezone-aware")
    return parsed


@dataclass(frozen=True)
class SnapshotFile:
    """One immutable file in a multi-source snapshot."""

    relative_path: str
    source: str
    size_bytes: int
    sha256: str
    fetched_at: str | None = None

    def __post_init__(self) -> None:
        from pathlib import PurePosixPath

        path = PurePosixPath(str(self.relative_path))
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ContractError(
                f"Snapshot file path must be safe and relative: {self.relative_path!r}"
            )
        source = str(self.source).strip()
        if not _SNAPSHOT_ID.fullmatch(source):
            raise ContractError(f"Invalid snapshot source name: {self.source!r}")
        if int(self.size_bytes) < 0:
            raise ContractError("Snapshot file size cannot be negative")
        digest = str(self.sha256).lower()
        if not _SHA256.fullmatch(digest):
            raise ContractError("Snapshot file sha256 must be a 64-character hex digest")
        if self.fetched_at is not None:
            parsed = parse_aware_datetime(self.fetched_at, field="fetched_at")
            object.__setattr__(self, "fetched_at", parsed.isoformat())
        object.__setattr__(self, "relative_path", path.as_posix())
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class SourceSnapshotManifest:
    """Portable identity and provenance for one immutable source snapshot."""

    snapshot_id: str
    created_at: str
    schema_version: str
    files: tuple[SnapshotFile, ...]

    FILE_PREFIX = "source_snapshot_"

    def __post_init__(self) -> None:
        snapshot_id = str(self.snapshot_id).strip()
        if not _SNAPSHOT_ID.fullmatch(snapshot_id):
            raise ContractError(f"Invalid source snapshot id: {self.snapshot_id!r}")
        if self.schema_version != SOURCE_SNAPSHOT_SCHEMA_VERSION:
            raise ContractError(
                f"Unsupported source snapshot schema: {self.schema_version!r}"
            )
        created = parse_aware_datetime(self.created_at, field="created_at")
        files = tuple(self.files)
        if not files:
            raise ContractError("A source snapshot must contain at least one file")
        keys = [(record.source, record.relative_path) for record in files]
        if len(keys) != len(set(keys)):
            raise ContractError("Source snapshot contains duplicate source/file entries")
        for record in files:
            fetched = parse_aware_datetime(
                record.fetched_at or created,
                field=f"fetched_at for {record.source}/{record.relative_path}",
            )
            if fetched > created:
                raise ContractError(
                    "Snapshot file fetched_at cannot be later than manifest created_at"
                )
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "created_at", created.isoformat())
        object.__setattr__(self, "files", files)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(record.source for record in self.files))

    @property
    def created_datetime(self) -> datetime:
        return parse_aware_datetime(self.created_at, field="created_at")

    def assert_available_at(self, forecast_cutoff: datetime | str) -> None:
        """Reject a snapshot that did not yet exist at *forecast_cutoff*."""

        cutoff = parse_aware_datetime(forecast_cutoff, field="forecast_cutoff")
        if self.created_datetime > cutoff:
            raise ContractError(
                f"Source snapshot {self.snapshot_id!r} was created at "
                f"{self.created_at}, after forecast cutoff {cutoff.isoformat()}"
            )
        late_files = [
            f"{record.source}/{record.relative_path}"
            for record in self.files
            if parse_aware_datetime(
                record.fetched_at or self.created_at,
                field="fetched_at",
            )
            > cutoff
        ]
        if late_files:
            raise ContractError(
                "Snapshot contains files fetched after the forecast cutoff: "
                + ", ".join(sorted(late_files))
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["files"] = [asdict(record) for record in self.files]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceSnapshotManifest":
        required = {"snapshot_id", "created_at", "schema_version", "files"}
        missing = sorted(required - set(payload))
        if missing:
            raise ContractError(
                "Source snapshot manifest is missing fields: " + ", ".join(missing)
            )
        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            raise ContractError("Source snapshot manifest files must be a list")
        try:
            files = tuple(SnapshotFile(**dict(record)) for record in raw_files)
        except (TypeError, ValueError) as exc:
            raise ContractError("Source snapshot manifest contains malformed files") from exc
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            created_at=str(payload["created_at"]),
            schema_version=str(payload["schema_version"]),
            files=files,
        )


__all__ = [
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "SnapshotFile",
    "SourceSnapshotManifest",
    "parse_aware_datetime",
]
