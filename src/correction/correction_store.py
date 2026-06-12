"""Correction artifact store for managing residual model metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CorrectionStore:
    """Manage residual model artifacts and metadata.

    Provides read/write access to ``residual_metadata.json`` and
    ``residual_feature_schema.json`` in the model output directory.
    """

    METADATA_FILE = "residual_metadata.json"
    SCHEMA_FILE = "residual_feature_schema.json"

    def __init__(self, store_dir: str = "models/residual"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load residual metadata, or None if not found."""
        path = self.store_dir / self.METADATA_FILE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load residual metadata: %s", exc)
            return None

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save residual metadata."""
        path = self.store_dir / self.METADATA_FILE
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved residual metadata to %s", path)

    def load_feature_schema(self) -> Optional[Dict[str, Any]]:
        """Load feature schema, or None if not found."""
        path = self.store_dir / self.SCHEMA_FILE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load feature schema: %s", exc)
            return None

    def get_accepted_stats(self) -> List[str]:
        """Return list of stats with accepted residual models."""
        metadata = self.load_metadata()
        if metadata is None:
            return []
        targets = metadata.get("targets", {})
        return [
            stat for stat, meta in targets.items()
            if meta.get("status") == "accepted"
        ]

    def get_rejected_stats(self) -> List[str]:
        """Return list of stats with rejected residual models."""
        metadata = self.load_metadata()
        if metadata is None:
            return []
        targets = metadata.get("targets", {})
        return [
            stat for stat, meta in targets.items()
            if meta.get("status") == "rejected"
        ]

    def list_artifacts(self) -> List[Path]:
        """List all artifact files in the store directory."""
        if not self.store_dir.exists():
            return []
        return sorted(self.store_dir.iterdir())
