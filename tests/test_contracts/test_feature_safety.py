from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src.contracts.artifacts import ArtifactContract, validate_runtime_artifacts
from src.contracts.errors import ArtifactContractError, FeatureSchemaContractError
from src.contracts.features import (
    CURRENT_GAME_TEAM_OUTCOMES,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMNS,
    is_forbidden_feature,
    load_feature_schema,
    validate_feature_frame,
    validate_feature_names,
)


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


def test_feature_policy_forbids_current_game_outcomes_but_allows_identifiers():
    assert TARGET_COLUMNS.issubset({"PTS", "REB", "AST", "STL", "BLK", "TOV"})
    assert CURRENT_GAME_TEAM_OUTCOMES.issuperset(
        {"PTS_TEAM", "REB_TEAM", "AST_TEAM", "FGA_TEAM", "FTA_TEAM", "OREB_TEAM", "DREB_TEAM", "TOV_TEAM"}
    )
    for name in ("PTS", "MIN", "FGA", "PTS_TEAM", "BOX_PTS_TEAM", "PTS_TEAM_CURRENT"):
        assert is_forbidden_feature(name)
    for name in ("PLAYER_ID", "TEAM_ID", "OPPONENT_ID", "TEAM_PTS_ROLL_10", "PTS_PLAYER_TE"):
        assert not is_forbidden_feature(name)


def test_validate_feature_names_lists_forbidden_columns():
    with pytest.raises(FeatureSchemaContractError, match="PTS_TEAM"):
        validate_feature_names(["ROLL_PTS_AVG_5", "PTS_TEAM"])


def test_runtime_contract_rejects_unsafe_feature_schema(tmp_path):
    leaking = [
        "AST_TEAM",
        "DREB_TEAM",
        "FGA_TEAM",
        "FTA_TEAM",
        "OREB_TEAM",
        "PTS_TEAM",
        "REB_TEAM",
        "TOV_TEAM",
    ]
    _write_runtime_artifacts(tmp_path, leaking)

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_runtime_artifacts(ArtifactContract(models_dir=tmp_path))

    message = str(exc_info.value)
    for name in leaking:
        assert name in message


def test_runtime_contract_requires_schema_pair_and_version(tmp_path):
    _write_runtime_artifacts(tmp_path, ["ROLL_PTS_AVG_5"])
    (tmp_path / "feature_cols.pkl").write_bytes(pickle.dumps(["ROLL_REB_AVG_5"]))

    with pytest.raises(ArtifactContractError, match="same ordered feature list"):
        validate_runtime_artifacts(ArtifactContract(models_dir=tmp_path))


def test_legacy_mode_is_explicit_and_does_not_make_it_strict(tmp_path):
    _write_runtime_artifacts(tmp_path, ["PTS_TEAM"])

    validate_runtime_artifacts(
        ArtifactContract(models_dir=tmp_path, allow_legacy_artifacts=True)
    )


def test_feature_frame_contract_rejects_forbidden_expected_schema():
    with pytest.raises(FeatureSchemaContractError, match="PTS_TEAM"):
        validate_feature_frame(pd.DataFrame({"PTS_TEAM": [1.0]}), ["PTS_TEAM"])


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"feature_cols": {"unexpected": "mapping"}},
    ],
)
def test_feature_schema_loader_rejects_malformed_payloads_without_recursing(tmp_path, payload):
    path = tmp_path / "feature_schema.pkl"
    path.write_bytes(pickle.dumps(payload))

    with pytest.raises(FeatureSchemaContractError, match="feature_cols list or schema object"):
        load_feature_schema(path)


def test_feature_schema_loader_rejects_self_referential_payload_without_recursing(tmp_path):
    payload = {}
    payload["feature_cols"] = payload
    path = tmp_path / "feature_schema.pkl"
    path.write_bytes(pickle.dumps(payload))

    with pytest.raises(FeatureSchemaContractError, match="feature_cols list or schema object"):
        load_feature_schema(path)
