import json

from src.models.versioning import ModelVersionRegistry


def test_model_version_registry_promotes_and_rolls_back(tmp_path):
    registry = ModelVersionRegistry(tmp_path / "models")
    first = registry.create_candidate("v1")
    (first / "feature_schema.pkl").write_bytes(b"ok")
    manifest = registry.promote(first, version="v1", metrics={"score": 1.0})
    assert manifest.version == "v1"
    assert registry.active_dir() == first
    second = registry.create_candidate("v2")
    manifest = registry.promote(second, version="v2")
    assert manifest.version == "v2"
    assert registry.active_dir() == second
    assert registry.rollback("v1").version == "v1"
    payload = json.loads((tmp_path / "models" / "champion.json").read_text())
    assert payload["version"] == "v1"
