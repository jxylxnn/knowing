"""Content identity for runtime/training source, including a dirty integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def source_tree_manifest(project_root="."):
    root = Path(project_root).resolve()
    files = set(root.glob("*.py"))
    files.update((root / "src").rglob("*.py"))
    files.update((root / "config").rglob("*.yaml"))
    files.update(root.glob("requirements*.txt"))
    files.update(root.glob("pyproject.toml"))
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(files) if path.is_file() and not path.is_symlink()}
    body = {"schema_version": "source_tree_v1", "files": hashes,
            "scope": "runtime_training_sources_and_configuration"}
    body["source_tree_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body
