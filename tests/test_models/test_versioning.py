import json
import pickle

import pytest

from src.contracts.features import FEATURE_SCHEMA_VERSION
from src.models.versioning import ModelVersionRegistry


def _write_runtime_artifacts(models_dir, feature_cols):
    models_dir.mkdir(parents=True, exist_ok=True)
    for target in ("PTS", "REB", "AST", "STL", "BLK", "TOV"):
        (models_dir / f"{target.lower()}_catboost.cbm").write_bytes(b"model")
        (models_dir / f"{target.lower()}_metadata.joblib").write_bytes(b"metadata")
    (models_dir / "blend_weights.pkl").write_bytes(b"weights")
    (models_dir / "model_stack_metadata.pkl").write_bytes(
        pickle.dumps({"targets": ["PTS", "REB", "AST", "STL", "BLK", "TOV"]})
    )
    (models_dir / "feature_cols.pkl").write_bytes(pickle.dumps(feature_cols))
    (models_dir / "feature_schema.pkl").write_bytes(
        pickle.dumps({"feature_cols": feature_cols, "version": FEATURE_SCHEMA_VERSION})
    )
    from src.models.versioning import ModelBundleManifest

    ModelBundleManifest.from_directory(
        models_dir,
        code_version="test-commit",
        dirty_worktree=False,
        source_snapshot_id="test-snapshot",
    ).write(models_dir)


def test_model_version_registry_rejects_legacy_candidate_promotion(tmp_path):
    registry = ModelVersionRegistry(tmp_path / "models")
    candidate = registry.create_candidate("legacy")
    _write_runtime_artifacts(candidate, ["ROLL_PTS_AVG_5"])

    with pytest.raises(ValueError, match="Only Model v2 bundles"):
        registry.promote(candidate, version="legacy")

    assert not registry.manifest_path.exists()


def test_model_version_registry_rejects_legacy_candidate_before_schema_checks(tmp_path):
    registry = ModelVersionRegistry(tmp_path / "models")
    candidate = registry.create_candidate("unsafe")
    _write_runtime_artifacts(candidate, ["PTS_TEAM"])

    with pytest.raises(ValueError, match="Only Model v2 bundles"):
        registry.promote(candidate, version="unsafe")

    assert not (tmp_path / "models" / "champion.json").exists()


def test_model_version_registry_requires_a_configured_champion(tmp_path):
    registry = ModelVersionRegistry(tmp_path / "models")

    with pytest.raises(FileNotFoundError, match="No valid model champion"):
        registry.active_dir()


def test_model_version_registry_does_not_fall_back_to_root_for_missing_champion_path(
    tmp_path,
):
    registry = ModelVersionRegistry(tmp_path / "models")
    registry.root.mkdir(parents=True)
    (registry.root / "feature_schema.pkl").write_bytes(b"stale root artifact")
    registry.manifest_path.write_text(
        json.dumps({
            "version": "missing",
            "path": "versions/missing",
            "promoted_at": "2026-09-09T00:00:00+00:00",
        }),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="champion directory does not exist"):
        registry.active_dir()
