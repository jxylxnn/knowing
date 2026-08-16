"""Tests for the read-only runtime artifact status inspection.

All fixtures live in ``tmp_path``; these tests never read or write the real
``models/`` directory.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

from src.contracts.artifacts import ArtifactContractError
from src.contracts.runtime_status import (
    RuntimeStatusError,
    inspect_runtime_status,
    resolve_active_models_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")

SMALL_METADATA = {
    "transformer_enabled": False,
    "model_count": 1,
    "training_preset": "small",
    "feature_groups": [
        "rolling",
        "efficiency",
        "momentum",
        "pace",
        "opponent_strength",
        "archetype",
    ],
    "blend_method": "inverse_mae",
    "feature_selection_enabled": False,
}

FULL_METADATA = {
    "transformer_enabled": True,
    "model_count": 2,
    "training_preset": "full",
    "feature_groups": ["rolling", "efficiency", "momentum", "archetype"],
    "blend_method": "inverse_mae",
}

TRAINING_WEIGHTS = {
    "version": 1,
    "description": "Training-time ridge blend",
    "optimizer_method": "",
    "backtest_score": None,
    "backtest_date_range": "",
    "parent_version": None,
    "per_target": {
        target: {"catboost": 1.0, "transformer": 0.0, "intercept": 0.0}
        for target in CANONICAL_TARGETS
    },
}

OPTIMIZED_WEIGHTS = {
    "version": 2,
    "description": "Optimizer promotion",
    "optimizer_method": "scipy_differential_evolution",
    "backtest_score": 0.4123,
    "backtest_date_range": "2026-01-01:2026-01-31",
    "parent_version": 1,
    "per_target": {
        target: {"catboost": 0.9, "transformer": 0.1, "intercept": 0.05}
        for target in CANONICAL_TARGETS
    },
}


def write_metadata(models_dir: Path, payload) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, models_dir / "model_stack_metadata.pkl")


def write_current_weights(models_dir: Path, payload) -> None:
    weights_dir = models_dir / "blend_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "current.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_small_bundle(models_dir: Path) -> None:
    """Small/CatBoost-only fixture: metadata + training-initialized weights."""
    write_metadata(models_dir, SMALL_METADATA)
    write_current_weights(models_dir, TRAINING_WEIGHTS)


def write_valid_artifact_bundle(
    models_dir: Path,
    *,
    metadata: dict | None = None,
    transformer: bool = False,
) -> None:
    """Write a minimal artifact set that passes contract validation."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for target in CANONICAL_TARGETS:
        lower = target.lower()
        (models_dir / f"{lower}_catboost.cbm").write_bytes(b"")
        joblib.dump({"target": target}, models_dir / f"{lower}_metadata.joblib")
    feature_cols = ["ROLL_PTS_10", "OPP_DEF_RATING"]
    with (models_dir / "feature_cols.pkl").open("wb") as handle:
        pickle.dump(feature_cols, handle)
    with (models_dir / "feature_schema.pkl").open("wb") as handle:
        pickle.dump(
            {"feature_cols": feature_cols, "version": "feature_schema_v4"},
            handle,
        )
    (models_dir / "blend_weights.pkl").write_bytes(b"")
    payload = metadata if metadata is not None else {"targets": list(CANONICAL_TARGETS)}
    with (models_dir / "model_stack_metadata.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    if transformer:
        (models_dir / "attention_transformer.pkl").write_bytes(b"")


def write_champion(root: Path, *, version: str, path: str) -> None:
    (root / "champion.json").write_text(
        json.dumps(
            {
                "version": version,
                "path": path,
                "promoted_at": "2026-08-16T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Required scenario tests
# ---------------------------------------------------------------------------


def test_small_catboost_only_reports_disabled_and_not_optimizer_produced(tmp_path):
    write_small_bundle(tmp_path)

    status = inspect_runtime_status(tmp_path)

    assert status["models_root"] == str(tmp_path)
    assert status["active_models_dir"] == str(tmp_path)
    assert status["champion_version"] is None
    assert status["training_preset"] == "small"
    assert status["transformer_enabled"] is False
    assert status["transformer_artifact_present"] is False
    assert status["model_count"] == 1
    assert status["feature_groups"] == SMALL_METADATA["feature_groups"]
    assert status["feature_group_count"] == 6
    assert status["training_blend_method"] == "inverse_mae"
    assert status["weight_version"] == 1
    assert status["weight_description"] == "Training-time ridge blend"
    assert status["optimizer_method"] == ""
    assert status["backtest_score"] is None
    assert status["backtest_date_range"] == ""
    assert status["optimizer_produced_current_weights"] is False
    assert status["warnings"] == []


def test_full_fixture_with_transformer_file_reports_enabled(tmp_path):
    write_metadata(tmp_path, FULL_METADATA)
    write_current_weights(tmp_path, TRAINING_WEIGHTS)
    (tmp_path / "attention_transformer.pkl").write_bytes(b"")

    status = inspect_runtime_status(tmp_path)

    assert status["training_preset"] == "full"
    assert status["transformer_enabled"] is True
    assert status["transformer_artifact_present"] is True
    assert status["model_count"] == 2
    assert status["warnings"] == []


def test_optimized_weight_fixture_reports_optimizer_provenance(tmp_path):
    write_metadata(tmp_path, SMALL_METADATA)
    write_current_weights(tmp_path, OPTIMIZED_WEIGHTS)

    status = inspect_runtime_status(tmp_path)

    assert status["weight_version"] == 2
    assert status["optimizer_method"] == "scipy_differential_evolution"
    assert status["backtest_score"] == 0.4123
    assert status["backtest_date_range"] == "2026-01-01:2026-01-31"
    assert status["optimizer_produced_current_weights"] is True


def test_missing_metadata_and_missing_weights_report_unknown(tmp_path):
    status = inspect_runtime_status(tmp_path)

    assert status["training_preset"] == "unknown"
    assert status["transformer_enabled"] == "unknown"
    assert status["model_count"] == "unknown"
    assert status["feature_groups"] == "unknown"
    assert status["feature_group_count"] == "unknown"
    assert status["training_blend_method"] == "unknown"
    assert status["weight_version"] is None
    assert status["weight_description"] is None
    assert status["optimizer_method"] is None
    assert status["backtest_score"] is None
    assert status["backtest_date_range"] is None
    assert status["optimizer_produced_current_weights"] is False


def test_malformed_weight_json_raises_clear_error(tmp_path):
    write_metadata(tmp_path, SMALL_METADATA)
    weights_dir = tmp_path / "blend_weights"
    weights_dir.mkdir()
    (weights_dir / "current.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeStatusError, match="current.json"):
        inspect_runtime_status(tmp_path)


def test_non_dictionary_metadata_pickle_raises_clear_error(tmp_path):
    joblib.dump(["not", "a", "dict"], tmp_path / "model_stack_metadata.pkl")

    with pytest.raises(RuntimeStatusError, match="dictionary"):
        inspect_runtime_status(tmp_path)


def test_malformed_artifact_cli_exits_nonzero(tmp_path):
    weights_dir = tmp_path / "blend_weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "current.json").write_text("{not valid json", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "inspect_artifacts.py",
            "--models-dir",
            str(tmp_path),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Error" in result.stderr


def test_transformer_metadata_file_contradiction_produces_warning(tmp_path):
    enabled_dir = tmp_path / "enabled"
    write_metadata(enabled_dir, FULL_METADATA)  # enabled, no artifact file

    status = inspect_runtime_status(enabled_dir)
    assert status["transformer_enabled"] is True
    assert status["transformer_artifact_present"] is False
    assert any("transformer" in w.lower() and "absent" in w.lower() for w in status["warnings"])

    disabled_dir = tmp_path / "disabled"
    write_metadata(disabled_dir, SMALL_METADATA)  # disabled, artifact present
    (disabled_dir / "attention_transformer.pkl").write_bytes(b"")

    status = inspect_runtime_status(disabled_dir)
    assert status["transformer_enabled"] is False
    assert status["transformer_artifact_present"] is True
    assert any("transformer" in w.lower() and "present" in w.lower() for w in status["warnings"])


def test_champion_root_inspects_active_version_not_stale_flat_files(tmp_path):
    root = tmp_path / "models"
    version_dir = root / "versions" / "v1"
    write_small_bundle(version_dir)
    # Stale flat files at the root describe a different deployment.
    write_metadata(root, {**SMALL_METADATA, "training_preset": "stale"})
    write_current_weights(root, OPTIMIZED_WEIGHTS)
    write_champion(root, version="v1", path="versions/v1")

    status = inspect_runtime_status(root)

    assert status["models_root"] == str(root)
    assert status["active_models_dir"] == str(version_dir)
    assert status["champion_version"] == "v1"
    assert status["training_preset"] == "small"
    assert status["weight_version"] == 1
    assert status["optimizer_produced_current_weights"] is False


def test_resolve_active_models_dir_uses_champion_only_when_manifest_exists(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    assert resolve_active_models_dir(candidate) == candidate

    root = tmp_path / "championed"
    version_dir = root / "versions" / "v1"
    write_small_bundle(version_dir)
    write_champion(root, version="v1", path="versions/v1")
    assert resolve_active_models_dir(root) == version_dir


# ---------------------------------------------------------------------------
# check_contracts.py champion resolution
# ---------------------------------------------------------------------------


def test_check_contracts_resolves_champion_directory(tmp_path, monkeypatch, capsys):
    import check_contracts

    root = tmp_path / "models"
    version_dir = root / "versions" / "v1"
    write_valid_artifact_bundle(version_dir)
    write_champion(root, version="v1", path="versions/v1")

    monkeypatch.setattr(
        sys, "argv", ["check_contracts.py", "--models-dir", str(root)]
    )
    check_contracts.main()

    out = capsys.readouterr().out
    assert "Resolved active champion directory" in out
    assert "Contracts passed" in out


def test_check_contracts_validates_candidate_directory_directly(tmp_path, monkeypatch):
    import check_contracts

    candidate = tmp_path / "candidate"
    write_valid_artifact_bundle(candidate)

    monkeypatch.setattr(
        sys, "argv", ["check_contracts.py", "--models-dir", str(candidate)]
    )
    check_contracts.main()  # must not raise


def test_check_contracts_fails_when_champion_version_is_broken(tmp_path, monkeypatch):
    """Proves validation targets the champion dir, not valid flat root files."""
    import check_contracts

    root = tmp_path / "models"
    version_dir = root / "versions" / "v1"
    version_dir.mkdir(parents=True)  # champion dir missing required artifacts
    write_valid_artifact_bundle(root)  # stale flat root IS valid
    write_champion(root, version="v1", path="versions/v1")

    monkeypatch.setattr(
        sys, "argv", ["check_contracts.py", "--models-dir", str(root)]
    )
    with pytest.raises(ArtifactContractError):
        check_contracts.main()
