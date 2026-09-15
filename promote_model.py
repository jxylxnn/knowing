#!/usr/bin/env python3
"""Validate and atomically promote an immutable Model v2 bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

from src.models.bundle import validate_v2_bundle
from src.models.versioning import ModelBundleManifest, ModelVersionRegistry


def resolve_candidate(value: str, models_dir: str | Path) -> Path:
    """Resolve either an explicit directory or a bundle/version identifier."""

    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()
    version = Path(models_dir).expanduser() / "versions" / value
    if version.is_dir():
        return version.resolve()
    raise FileNotFoundError(f"Candidate bundle does not exist: {value}")


def validate_candidate(candidate: str | Path, *, promotion: bool = True) -> dict:
    """Validate the appropriate artifact contract without changing champion."""

    path = Path(candidate)
    manifest_path = path / ModelBundleManifest.FILE_NAME
    if not manifest_path.is_file():
        raise ValueError("Candidate is missing bundle_manifest.json")
    manifest = ModelBundleManifest.load(manifest_path)
    if manifest.architecture != "v2":
        raise ValueError("Only sealed Model v2 candidates may enter production")
    return validate_v2_bundle(path, promotion=promotion).to_dict()


def _clean_process_validate(candidate: Path) -> None:
    """Prove that the sealed bundle contract loads in a fresh interpreter."""

    project_root = Path(__file__).resolve().parent
    code = (
        "from src.forecasting.baseline_backend import V2BaselineBackend\n"
        f"V2BaselineBackend({str(candidate)!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Post-promotion clean-process validation failed: {detail}")


def promote_with_rollback(
    registry: ModelVersionRegistry,
    candidate: str | Path,
    *,
    post_check: Callable[[Path], None] = _clean_process_validate,
):
    """Promote atomically and restore the prior pointer if post-check fails."""

    candidate_path = Path(candidate).resolve()
    previous = (
        registry.manifest_path.read_bytes()
        if registry.manifest_path.is_file()
        else None
    )
    if previous is not None:
        validate_v2_bundle(registry.active_dir(), promotion=True)
    manifest = registry.promote(candidate_path)
    try:
        post_check(candidate_path)
    except Exception:
        if previous is None:
            registry.manifest_path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(registry.manifest_path, previous)
        raise
    return manifest


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".champion_restore_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_decision(models_dir: Path, payload: dict) -> Path:
    history = models_dir / "promotion_history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = history / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Candidate bundle directory")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = ModelVersionRegistry(args.models_dir)
    candidate = resolve_candidate(args.candidate, args.models_dir)
    try:
        validation = validate_candidate(candidate, promotion=True)
        if args.dry_run:
            payload = {
                "eligible": True,
                "dry_run": True,
                "candidate": str(candidate),
                **validation,
            }
            payload["decision_record"] = str(
                _record_decision(Path(args.models_dir), payload)
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        manifest = promote_with_rollback(registry, candidate)
        payload = {"eligible": True, "dry_run": False, **manifest.to_dict()}
        payload["decision_record"] = str(
            _record_decision(Path(args.models_dir), payload)
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        _record_decision(
            Path(args.models_dir),
            {
                "eligible": False,
                "dry_run": bool(args.dry_run),
                "candidate": str(candidate),
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
