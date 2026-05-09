"""Versioned ensemble weight storage with atomic writes and rollback.

Replaces the opaque binary blend_weights.pkl with a human-readable,
versioned JSON store. Each version is a numbered file; a pointer
(current.json) indicates the active version.

Atomic writes (temp file + rename) prevent corruption on crash.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TargetBlend:
    """Per-target ensemble blend coefficients."""
    catboost: float = 1.0
    transformer: float = 0.0
    intercept: float = 0.0
    catboost_mae_blend: float = 0.7   # primary vs MAE companion mix


@dataclass
class EnsembleWeights:
    """Complete set of ensemble blend weights for all targets.

    This is the tunable configuration that the self-optimizer modifies.
    It replaces the hardcoded constants in ModelManager._predict_catboost_target
    (0.7/0.3 split) and the loaded blend_weights.pkl.
    """
    version: int = 1
    created_at: str = ""
    description: str = ""
    backtest_score: Optional[float] = None
    backtest_date_range: str = ""

    # Per-target blend coefficients
    per_target: Dict[str, TargetBlend] = field(default_factory=dict)

    # Global CatBoost-MAE companion blend (shared across targets)
    catboost_mae_blend: float = 0.7

    # Metadata
    parent_version: Optional[int] = None
    optimizer_method: str = ""
    accept_margin: float = 0.01

    @classmethod
    def default_for_targets(cls, targets: List[str]) -> "EnsembleWeights":
        """Create default weights for a set of target stats."""
        per_target = {
            t: TargetBlend(catboost=0.7, transformer=0.3, intercept=0.0, catboost_mae_blend=0.7)
            for t in targets
        }
        return cls(
            version=1,
            created_at=datetime.now().isoformat(),
            description="Default ensemble weights",
            per_target=per_target,
            catboost_mae_blend=0.7,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "backtest_score": self.backtest_score,
            "backtest_date_range": self.backtest_date_range,
            "catboost_mae_blend": self.catboost_mae_blend,
            "per_target": {
                t: asdict(b) for t, b in self.per_target.items()
            },
            "parent_version": self.parent_version,
            "optimizer_method": self.optimizer_method,
            "accept_margin": self.accept_margin,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleWeights":
        """Deserialize from JSON dict."""
        per_target = {
            t: TargetBlend(**b) for t, b in data.get("per_target", {}).items()
        }
        return cls(
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
            backtest_score=data.get("backtest_score"),
            backtest_date_range=data.get("backtest_date_range", ""),
            per_target=per_target,
            catboost_mae_blend=data.get("catboost_mae_blend", 0.7),
            parent_version=data.get("parent_version"),
            optimizer_method=data.get("optimizer_method", ""),
            accept_margin=data.get("accept_margin", 0.01),
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"EnsembleWeights v{self.version} | {self.created_at[:19]}",
            f"  CatBoost/MAE blend: {self.catboost_mae_blend:.2f}",
            f"  Score: {self.backtest_score or 'N/A'}",
        ]
        for target, b in sorted(self.per_target.items()):
            lines.append(
                f"  {target:4s}: CB={b.catboost:.3f} TX={b.transformer:.3f} "
                f"INT={b.intercept:+.3f} MAE_blend={b.catboost_mae_blend:.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Weight store
# ---------------------------------------------------------------------------

class WeightStore:
    """Versioned storage for ensemble blend weights.

    Directory layout:
        models/blend_weights/
        ├── v0001.json
        ├── v0002.json
        ├── ...
        ├── current.json   (copy of active version)
        └── history.json   (log of all versions with scores)
    """

    CURRENT_FILE = "current.json"
    HISTORY_FILE = "history.json"
    VERSION_PREFIX = "v"

    def __init__(self, store_dir: str = "models/blend_weights"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _version_path(self, version: int) -> Path:
        return self.store_dir / f"{self.VERSION_PREFIX}{version:04d}.json"

    def _current_path(self) -> Path:
        return self.store_dir / self.CURRENT_FILE

    def _history_path(self) -> Path:
        return self.store_dir / self.HISTORY_FILE

    def _next_version(self) -> int:
        """Find the next available version number."""
        existing = sorted(self.store_dir.glob(f"{self.VERSION_PREFIX}*.json"))
        if not existing:
            return 1
        versions = []
        for p in existing:
            try:
                versions.append(int(p.stem[1:]))  # strip 'v' prefix
            except ValueError:
                pass
        return max(versions, default=0) + 1

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_current(self) -> Optional[EnsembleWeights]:
        """Load the currently active ensemble weights."""
        current_path = self._current_path()
        if not current_path.exists():
            # Fall back to legacy blend_weights.pkl migration
            return self._migrate_legacy_weights()
        try:
            with open(current_path) as f:
                data = json.load(f)
            return EnsembleWeights.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Failed to load current weights from %s: %s", current_path, exc)
            return None

    def load_version(self, version: int) -> Optional[EnsembleWeights]:
        """Load a specific version by number."""
        path = self._version_path(version)
        if not path.exists():
            logger.warning("Version %d not found at %s", version, path)
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return EnsembleWeights.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Failed to load version %d: %s", version, exc)
            return None

    def load_history(self) -> List[Dict[str, Any]]:
        """Load the version history log."""
        path = self._history_path()
        if not path.exists():
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            return []

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save(self, weights: EnsembleWeights, set_current: bool = True) -> int:
        """Save weights as a new version with atomic write.

        Writes to a temp file in the same directory, then atomically renames
        to the final version path. This prevents corruption on crash.

        Args:
            weights: The ensemble weights to save.
            set_current: If True, update current.json to point to this version.

        Returns:
            The version number assigned.
        """
        version = self._next_version()
        weights.version = version
        weights.created_at = datetime.now().isoformat()

        version_path = self._version_path(version)

        # Atomic write: temp file → rename
        data = weights.to_dict()
        json_text = json.dumps(data, indent=2, sort_keys=True)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=f".tmp_{self.VERSION_PREFIX}{version:04d}_",
            dir=str(self.store_dir),
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json_text)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(version_path))
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info("Saved weights v%d to %s", version, version_path)

        # Update current pointer
        if set_current:
            self._write_current(weights)

        # Append to history log
        self._append_history(weights)

        return version

    def _write_current(self, weights: EnsembleWeights) -> None:
        """Atomically write current.json."""
        current_path = self._current_path()
        data = weights.to_dict()
        json_text = json.dumps(data, indent=2, sort_keys=True)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=".tmp_current_",
            dir=str(self.store_dir),
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json_text)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(current_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _append_history(self, weights: EnsembleWeights) -> None:
        """Append an entry to the history log."""
        history_path = self._history_path()
        history = self.load_history()

        entry = {
            "version": weights.version,
            "created_at": weights.created_at,
            "description": weights.description,
            "backtest_score": weights.backtest_score,
            "backtest_date_range": weights.backtest_date_range,
            "parent_version": weights.parent_version,
            "optimizer_method": weights.optimizer_method,
        }
        history.append(entry)

        json_text = json.dumps(history, indent=2, sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=".tmp_history_",
            dir=str(self.store_dir),
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json_text)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(history_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, target_version: int) -> Optional[EnsembleWeights]:
        """Roll back current weights to a specific version.

        Args:
            target_version: The version number to roll back to.

        Returns:
            The restored EnsembleWeights, or None if the version doesn't exist.
        """
        weights = self.load_version(target_version)
        if weights is None:
            logger.error("Cannot rollback: version %d not found", target_version)
            return None

        # Create a new version pointing back to the rolled-back version
        weights.version = self._next_version()
        weights.parent_version = target_version
        weights.description = f"ROLLBACK to v{target_version:04d}"
        weights.backtest_score = None

        self.save(weights, set_current=True)
        logger.info("Rolled back to v%d (saved as v%d)", target_version, weights.version)
        return weights

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    def _migrate_legacy_weights(self) -> Optional[EnsembleWeights]:
        """Migrate from legacy blend_weights.pkl to versioned JSON store.

        This is called when current.json doesn't exist but the legacy
        binary file might. We import it, save as v0001, and set current.

        Returns:
            Migrated EnsembleWeights, or None if no legacy weights exist.
        """
        import joblib

        # Try standard models/blend_weights.pkl location
        legacy_path = self.store_dir.parent / "blend_weights.pkl"
        if not legacy_path.exists():
            return None

        try:
            legacy_data = joblib.load(legacy_path)
        except Exception as exc:
            logger.warning("Failed to load legacy blend weights: %s", exc)
            return None

        if not isinstance(legacy_data, dict):
            logger.warning("Legacy blend weights are not a dict, skipping migration")
            return None

        # Convert legacy format: {"PTS": {"catboost": 0.7, "transformer": 0.3,
        # "intercept": 0.0}, ...}
        per_target: Dict[str, TargetBlend] = {}
        for target, cfg in legacy_data.items():
            if isinstance(cfg, dict) and target not in ("_method",):
                per_target[target] = TargetBlend(
                    catboost=float(cfg.get("catboost", 1.0)),
                    transformer=float(cfg.get("transformer", 0.0)),
                    intercept=float(cfg.get("intercept", 0.0)),
                    catboost_mae_blend=float(cfg.get("catboost_mae_blend", 0.7)),
                )

        if not per_target:
            logger.warning("No per-target weights found in legacy file")
            return None

        weights = EnsembleWeights(
            version=1,
            created_at=datetime.now().isoformat(),
            description="Migrated from legacy blend_weights.pkl",
            per_target=per_target,
            catboost_mae_blend=0.7,
        )

        logger.info("Migrating legacy blend weights → v0001")
        self.save(weights, set_current=True)

        # Don't delete the legacy file — keep as backup
        backup_path = legacy_path.with_suffix(".pkl.bak")
        shutil.copy2(legacy_path, backup_path)
        logger.info("Legacy weights backed up to %s", backup_path)

        return weights

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all saved versions with metadata."""
        versions = []
        for p in sorted(self.store_dir.glob(f"{self.VERSION_PREFIX}*.json")):
            try:
                ver = int(p.stem[1:])
                with open(p) as f:
                    data = json.load(f)
                versions.append({
                    "version": ver,
                    "created_at": data.get("created_at", ""),
                    "description": data.get("description", ""),
                    "backtest_score": data.get("backtest_score"),
                    "optimizer_method": data.get("optimizer_method", ""),
                    "is_current": self._is_current(ver),
                })
            except (ValueError, json.JSONDecodeError):
                pass
        return sorted(versions, key=lambda v: v["version"])

    def _is_current(self, version: int) -> bool:
        """Check if a version is the currently active one."""
        current = self.load_current()
        return current is not None and current.version == version
