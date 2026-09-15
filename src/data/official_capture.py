"""Prospective official-source capture with receipt times that cannot be backdated."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from src.contracts.canonical_data import require_columns
from src.data.snapshots import create_source_snapshot


REQUIRED = {
    "roster_membership.csv": ("PLAYER_ID", "TEAM_ID", "START_DATE", "END_DATE"),
    "player_status_snapshots.csv": ("PLAYER_ID", "TEAM_ID", "STATUS"),
    "schedule.csv": ("GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "SCHEDULED_TIP"),
}


def _official_url(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == "nba.com" or host.endswith(".nba.com")):
        raise ValueError("Official capture requires an HTTPS nba.com source")


def capture_official_source(data_dir, *, url, filename, normalize=None):
    """Archive source bytes plus normalized rows; no historical timestamp option.

    An adapter may normalize official JSON into the declared table schema.
    Without an adapter the source must be CSV. Historical effective dates may
    be retained, but availability is always the actual local receipt time.
    """
    _official_url(url)
    if filename not in REQUIRED:
        raise ValueError("Unsupported official capture table")
    with urlopen(url, timeout=30) as response:
        _official_url(response.geturl())
        payload = response.read()
        final_url = response.geturl()
    observed = datetime.now(timezone.utc)
    frame = normalize(payload) if normalize else pd.read_csv(BytesIO(payload), dtype=str)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Official source returned no normalized records")
    require_columns(frame, REQUIRED[filename], source=filename)
    frame = frame.copy()
    if "AVAILABLE_AT" in frame:
        frame["UPSTREAM_AVAILABLE_AT"] = frame["AVAILABLE_AT"]
    frame["AVAILABLE_AT"] = observed.isoformat()
    frame["EVENT_TIME"] = frame.get("EVENT_TIME", observed.isoformat())
    frame["SOURCE"] = final_url
    if filename == "roster_membership.csv":
        frame["COVERAGE_STATUS"] = "official"
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Temporary receipt directory is not a mutable working-source replacement.
    with tempfile.TemporaryDirectory(prefix=".official_capture_", dir=root) as temporary:
        stage = Path(temporary)
        (stage / "response.bin").write_bytes(payload)
        frame.to_csv(stage / filename, index=False)
        receipt = {"source_url": final_url, "received_at": observed.isoformat(),
                   "raw_sha256": hashlib.sha256(payload).hexdigest(),
                   "normalizer": getattr(normalize, "__qualname__", "pandas_csv"),
                   "rows": len(frame), "table": filename}
        (stage / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        # The snapshot writer copies staged bytes into a separate immutable
        # capture root. Capture snapshots can later be included in full slates.
        manifest = create_source_snapshot(
            stage, files=(filename, "response.bin", "receipt.json"), source="official",
            created_at=observed,
        )
        destination = root / "official_captures" / manifest.snapshot_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        publication = stage / "publication"
        publication.mkdir()
        (stage / "raw").rename(publication / "raw")
        (stage / "manifests").rename(publication / "manifests")
        publication.rename(destination)
    return destination
