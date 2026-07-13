"""Atomic champion/challenger artifact versioning for runtime models."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ChampionManifest:
    """Pointer to the artifact bundle currently used by inference."""

    version: str
    path: str
    promoted_at: str
    data_cutoff: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    bundle_id: Optional[str] = None
    bundle_manifest: Optional[str] = None
    schema_hash: Optional[str] = None
    config_hash: Optional[str] = None
    code_version: Optional[str] = None
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelBundleManifest:
    """Immutable manifest for every file used by a runtime model bundle."""

    bundle_id: str
    created_at: str
    data_cutoff: Optional[str]
    schema_hash: str
    config_hash: str
    code_version: str
    components: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any] = field(default_factory=dict)
    weighting_policy: Dict[str, Any] = field(default_factory=dict)

    FILE_NAME = "bundle_manifest.json"

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        bundle_id: Optional[str] = None,
        data_cutoff: Optional[str] = None,
        config: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
        weighting_policy: Optional[Dict[str, Any]] = None,
        code_version: Optional[str] = None,
    ) -> "ModelBundleManifest":
        root = Path(directory)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Model bundle directory does not exist: {root}")

        components: Dict[str, Dict[str, Any]] = {}
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            if relative == cls.FILE_NAME or relative.startswith("."):
                continue
            components[relative] = {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

        schema_hash = _schema_hash(root, components)
        config_hash = _payload_hash(config) if config is not None else ""
        resolved_code_version = code_version or os.environ.get("GIT_COMMIT", "unknown")
        resolved_id = bundle_id or _bundle_id(
            components, schema_hash, config_hash, data_cutoff, resolved_code_version
        )
        return cls(
            bundle_id=resolved_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            data_cutoff=data_cutoff,
            schema_hash=schema_hash,
            config_hash=config_hash,
            code_version=resolved_code_version,
            components=components,
            metrics=metrics or {},
            weighting_policy=weighting_policy or {},
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundleManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, directory: str | Path) -> Path:
        """Write once; an existing manifest can never be silently replaced."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / self.FILE_NAME
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != payload:
                raise FileExistsError(f"Immutable bundle manifest already exists: {path}")
            return path
        fd, temp_name = tempfile.mkstemp(
            prefix=".bundle_manifest_", suffix=".json", dir=str(root)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path

    def validate(self, directory: str | Path) -> None:
        """Validate component presence and content hashes."""
        root = Path(directory)
        for name, component in self.components.items():
            relative = str(component.get("path", name))
            path = root / relative
            if not path.exists() or not path.is_file():
                raise ValueError(f"Bundle component is missing: {relative}")
            expected = str(component.get("sha256", ""))
            actual = _sha256_file(path)
            if expected != actual:
                raise ValueError(
                    f"Bundle component hash mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )

        expected_schema = _schema_hash(root, self.components)
        if self.schema_hash and expected_schema != self.schema_hash:
            raise ValueError(
                f"Bundle schema hash mismatch: expected {self.schema_hash}, got {expected_schema}"
            )


class ModelVersionRegistry:
    """Keep immutable candidate bundles and atomically switch the champion."""

    MANIFEST_NAME = "champion.json"

    def __init__(self, models_root: str | Path = "models") -> None:
        self.root = Path(models_root)
        self.versions_dir = self.root / "versions"
        self.manifest_path = self.root / self.MANIFEST_NAME

    def read_manifest(self) -> Optional[ChampionManifest]:
        if not self.manifest_path.exists():
            return None
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return ChampionManifest(**payload)
        except (OSError, TypeError, ValueError, KeyError):
            return None

    def active_dir(self) -> Path:
        """Resolve the active artifact directory, preserving legacy flat models."""
        manifest = self.read_manifest()
        if manifest is None:
            return self.root
        candidate = Path(manifest.path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate if candidate.exists() else self.root

    def create_candidate(self, run_id: Optional[str] = None) -> Path:
        stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.versions_dir / stamp
        path.mkdir(parents=True, exist_ok=False)
        return path

    def create_bundle_manifest(self, directory: str | Path, **kwargs: Any) -> ModelBundleManifest:
        """Create and persist the immutable manifest for a candidate bundle."""
        manifest = ModelBundleManifest.from_directory(directory, **kwargs)
        manifest.write(directory)
        return manifest

    def promote(
        self,
        candidate_dir: str | Path,
        *,
        version: Optional[str] = None,
        data_cutoff: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ChampionManifest:
        candidate = Path(candidate_dir).resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Candidate model directory does not exist: {candidate}")
        bundle_path = candidate / ModelBundleManifest.FILE_NAME
        bundle_manifest = None
        if bundle_path.exists():
            bundle_manifest = ModelBundleManifest.load(bundle_path)
            bundle_manifest.validate(candidate)
        version = version or candidate.name
        try:
            relative_path = os.path.relpath(candidate, self.root.resolve())
        except ValueError:
            relative_path = str(candidate)
        manifest = ChampionManifest(
            version=version,
            path=relative_path,
            promoted_at=datetime.now(timezone.utc).isoformat(),
            data_cutoff=data_cutoff,
            metrics=metrics or {},
            bundle_id=bundle_manifest.bundle_id if bundle_manifest else None,
            bundle_manifest=(
                os.path.relpath(bundle_path, self.root.resolve())
                if bundle_manifest else None
            ),
            schema_hash=bundle_manifest.schema_hash if bundle_manifest else None,
            config_hash=bundle_manifest.config_hash if bundle_manifest else None,
            code_version=bundle_manifest.code_version if bundle_manifest else None,
            components=bundle_manifest.components if bundle_manifest else {},
        )
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".champion_", suffix=".json", dir=str(self.root)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_name, self.manifest_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return manifest

    def rollback(self, version: str) -> ChampionManifest:
        candidate = self.versions_dir / version
        if not candidate.exists():
            raise FileNotFoundError(f"Model version not found: {version}")
        return self.promote(candidate, version=version)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _schema_hash(root: Path, components: Dict[str, Dict[str, Any]]) -> str:
    schema_path = root / "feature_schema.pkl"
    if schema_path.exists():
        try:
            import joblib

            schema = joblib.load(schema_path)
            value = getattr(schema, "schema_hash", None)
            if value:
                return str(value)
        except Exception:
            pass
    schema_files = {
        name: component.get("sha256")
        for name, component in components.items()
        if "schema" in name.lower() or "feature" in name.lower()
    }
    return _payload_hash(schema_files) if schema_files else ""


def _bundle_id(
    components: Dict[str, Dict[str, Any]],
    schema_hash: str,
    config_hash: str,
    data_cutoff: Optional[str],
    code_version: str,
) -> str:
    return _payload_hash({
        "components": components,
        "schema_hash": schema_hash,
        "config_hash": config_hash,
        "data_cutoff": data_cutoff,
        "code_version": code_version,
    })[:24]
