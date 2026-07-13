"""JSONL persistence for optional player reasoning reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .engine import PlayerReasoningReport


def sidecar_path(projection_path: str | Path) -> Path:
    """Return the deterministic reasoning sidecar path for a projection CSV."""
    path = Path(projection_path)
    stem = path.stem.replace("player_projections_", "player_reasoning_", 1)
    return path.with_name(f"{stem}.jsonl")


def write_reasoning_sidecar(
    reports: Iterable[PlayerReasoningReport],
    path: str | Path,
) -> Path:
    """Write one complete versioned report per line using an atomic replace."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
    tmp.replace(output)
    return output


def load_reasoning_sidecar(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load reports indexed by normalized player name."""
    output: Dict[str, Dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return output
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            name = str(payload.get("player_name", "")).strip().lower()
            if name:
                output[name] = payload
    return output
