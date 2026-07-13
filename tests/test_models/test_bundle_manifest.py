import pytest

from src.models.versioning import ModelBundleManifest


def test_bundle_manifest_hashes_and_rejects_tampering(tmp_path):
    (tmp_path / "model.cbm").write_bytes(b"model")
    (tmp_path / "feature_schema.pkl").write_bytes(b"schema")
    manifest = ModelBundleManifest.from_directory(
        tmp_path,
        data_cutoff="2026-04-10",
        config={"preset": "small"},
        code_version="test",
    )
    manifest.write(tmp_path)
    manifest.validate(tmp_path)

    (tmp_path / "model.cbm").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest.validate(tmp_path)

