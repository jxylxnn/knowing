"""Publish complete forecast/sample pairs under immutable content identities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.utils.file_lock import exclusive_file_lock


def export_forecast_run(root, forecasts: pd.DataFrame, samples: pd.DataFrame):
    """Stage both files before publishing; no previous export is overwritten."""
    if forecasts.empty or samples.empty:
        raise ValueError("Cannot export an empty forecast run")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".forecast_export_", dir=root))
    try:
        forecasts.to_parquet(stage / "forecasts.parquet", index=False)
        samples.to_parquet(stage / "samples.parquet", index=False)
        files = {
            name: _digest(stage / name)
            for name in ("forecasts.parquet", "samples.parquet")
        }
        identity = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest()
        destination = root / identity
        (stage / "manifest.json").write_text(
            json.dumps({"content_id": identity, "files": files},
                       sort_keys=True, indent=2) + "\n"
        )
        with exclusive_file_lock(root / f".{identity}.lock"):
            if destination.exists():
                for name, digest in files.items():
                    if not (destination / name).is_file() or _digest(destination / name) != digest:
                        raise ValueError("Existing immutable export is corrupt")
                if (destination / "manifest.json").read_bytes() != (stage / "manifest.json").read_bytes():
                    raise ValueError("Existing immutable export manifest is corrupt")
            else:
                os.rename(stage, destination)
        return destination / "forecasts.parquet", destination / "samples.parquet"
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
