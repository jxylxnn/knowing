"""Read-only inspection of deployed runtime model artifacts.

Answers the question "what is actually deployed right now?" by reading
artifact files (``model_stack_metadata.pkl``, ``blend_weights/current.json``,
and the champion manifest) without loading inference models and without
mutating anything.

This module intentionally never instantiates ``WeightStore``: its constructor
creates directories and its loader may migrate legacy data. All weight
information is read straight from ``current.json`` with the ``json`` module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.contracts.artifacts import CANONICAL_TARGETS
from src.models.versioning import ModelVersionRegistry

# Value used for metadata-derived fields when the metadata file is missing.
UNKNOWN = "unknown"


class RuntimeStatusError(Exception):
    """Raised when a present artifact file cannot be interpreted."""


def resolve_active_models_dir(models_dir: Path) -> Path:
    """Resolve the active artifact directory for a models root.

    When ``models_dir/champion.json`` exists, the active directory is the
    champion version directory resolved via
    ``ModelVersionRegistry.active_dir()``. Otherwise the supplied directory
    is inspected directly, so a candidate/version directory continues to
    validate directly.

    This is strictly read-only: it never creates files.
    """
    root = Path(models_dir)
    if (root / ModelVersionRegistry.MANIFEST_NAME).exists():
        return ModelVersionRegistry(root).active_dir()
    return root


def inspect_runtime_status(models_dir: Path) -> dict:
    """Inspect deployed runtime artifacts and return a status dictionary.

    The models root is resolved through the champion manifest when one
    exists. Missing optional data is reported as ``None`` (absent field in a
    present file) or ``"unknown"`` (file missing entirely); nothing is
    invented. Contradictions between metadata and files are reported in the
    ``warnings`` list. Present-but-malformed files raise
    :class:`RuntimeStatusError` instead of guessing.
    """
    root = Path(models_dir)
    warnings: List[str] = []

    registry = ModelVersionRegistry(root)
    manifest = registry.read_manifest()
    champion_path = root / ModelVersionRegistry.MANIFEST_NAME
    if champion_path.exists() and manifest is None:
        warnings.append(
            f"{champion_path} exists but could not be parsed; "
            "inspecting the supplied directory directly"
        )
    champion_version = manifest.version if manifest is not None else None
    active_models_dir = resolve_active_models_dir(root)

    metadata = _load_metadata(active_models_dir / "model_stack_metadata.pkl")
    weights = _load_current_weights(
        active_models_dir / "blend_weights" / "current.json"
    )

    # --- metadata-derived fields -----------------------------------------
    if metadata is None:
        training_preset: Any = UNKNOWN
        transformer_enabled: Any = UNKNOWN
        model_count: Any = UNKNOWN
        feature_groups: Any = UNKNOWN
        feature_group_count: Any = UNKNOWN
        training_blend_method: Any = UNKNOWN
    else:
        training_preset = metadata.get("training_preset")
        transformer_enabled = metadata.get("transformer_enabled")
        model_count = metadata.get("model_count")
        feature_groups = metadata.get("feature_groups")
        feature_group_count = (
            len(feature_groups)
            if isinstance(feature_groups, (list, tuple))
            else None
        )
        training_blend_method = metadata.get("blend_method") or metadata.get(
            "training_blend_method"
        )

    transformer_artifact_present = (
        active_models_dir / "attention_transformer.pkl"
    ).exists()
    if transformer_enabled is True and not transformer_artifact_present:
        warnings.append(
            "metadata enables the Transformer but attention_transformer.pkl "
            "is absent from the active models directory"
        )
    if transformer_enabled is False and transformer_artifact_present:
        warnings.append(
            "metadata disables the Transformer but attention_transformer.pkl "
            "is present in the active models directory"
        )

    mae_companion_targets = sorted(
        target
        for target in CANONICAL_TARGETS
        if (active_models_dir / f"{target.lower()}_catboost_mae.cbm").exists()
    )

    # --- weight-derived fields -------------------------------------------
    if weights is None:
        weight_version = None
        weight_description = None
        optimizer_method = None
        backtest_score = None
        backtest_date_range = None
    else:
        weight_version = weights.get("version")
        weight_description = weights.get("description")
        optimizer_method = weights.get("optimizer_method")
        backtest_score = weights.get("backtest_score")
        backtest_date_range = weights.get("backtest_date_range")

    optimizer_produced_current_weights = (
        _is_finite_number(backtest_score)
        and isinstance(optimizer_method, str)
        and optimizer_method.strip() != ""
        and isinstance(backtest_date_range, str)
        and backtest_date_range.strip() != ""
    )

    return {
        "models_root": str(root),
        "active_models_dir": str(active_models_dir),
        "champion_version": champion_version,
        "training_preset": training_preset,
        "transformer_enabled": transformer_enabled,
        "transformer_artifact_present": transformer_artifact_present,
        "mae_companion_targets": mae_companion_targets,
        "model_count": model_count,
        "feature_groups": feature_groups,
        "feature_group_count": feature_group_count,
        "training_blend_method": training_blend_method,
        "weight_version": weight_version,
        "weight_description": weight_description,
        "optimizer_method": optimizer_method,
        "backtest_score": backtest_score,
        "backtest_date_range": backtest_date_range,
        "optimizer_produced_current_weights": optimizer_produced_current_weights,
        "warnings": warnings,
    }


def _load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Load the model stack metadata pickle, or ``None`` when absent."""
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise RuntimeStatusError(
            f"Could not load model stack metadata: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeStatusError(
            f"model_stack_metadata.pkl must contain a dictionary, "
            f"got {type(payload).__name__}: {path}"
        )
    return payload


def _load_current_weights(path: Path) -> Optional[Dict[str, Any]]:
    """Load ``blend_weights/current.json``, or ``None`` when absent."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeStatusError(
            f"Could not parse blend weights JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeStatusError(
            f"blend_weights/current.json must contain a JSON object, "
            f"got {type(payload).__name__}: {path}"
        )
    return payload


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` is a real finite number (not bool/str/None)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
