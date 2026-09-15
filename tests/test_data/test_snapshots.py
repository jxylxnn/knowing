from datetime import datetime, timezone

import pytest

from src.data.snapshots import (
    create_source_snapshot,
    validate_source_snapshot,
)


def _write_core_files(data_dir):
    (data_dir / "nba_players.csv").write_text("PLAYER_ID,GAME_ID\n1,10\n", encoding="utf-8")
    (data_dir / "nba_games.csv").write_text("TEAM_ID,GAME_ID\n2,10\n", encoding="utf-8")


def test_snapshot_is_immutable_checksums_present_files_and_ignores_absent_optional_files(tmp_path):
    _write_core_files(tmp_path)
    (tmp_path / "player_bios.csv").write_text("PLAYER_ID\n1\n", encoding="utf-8")

    manifest = create_source_snapshot(
        tmp_path,
        snapshot_id="snapshot-a",
        created_at=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
    )

    assert manifest.snapshot_id == "snapshot-a"
    assert [record.relative_path for record in manifest.files] == [
        "nba_players.csv",
        "nba_games.csv",
        "player_bios.csv",
    ]
    assert (tmp_path / "raw" / "core" / "snapshot-a" / "nba_players.csv").is_file()
    assert (tmp_path / "manifests" / "source_snapshot_snapshot-a.json").is_file()
    assert validate_source_snapshot(tmp_path, "snapshot-a") == manifest

    with pytest.raises(FileExistsError, match="already exists"):
        create_source_snapshot(tmp_path, snapshot_id="snapshot-a")


def test_snapshot_validation_detects_tampering(tmp_path):
    _write_core_files(tmp_path)
    create_source_snapshot(tmp_path, snapshot_id="snapshot-a")
    (tmp_path / "raw" / "core" / "snapshot-a" / "nba_games.csv").write_text(
        "modified", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_snapshot(tmp_path, "snapshot-a")


def test_snapshot_requires_the_two_core_files_and_safe_paths(tmp_path):
    (tmp_path / "nba_players.csv").write_text("PLAYER_ID\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nba_games.csv"):
        create_source_snapshot(tmp_path)

    _write_core_files(tmp_path)
    with pytest.raises(ValueError, match="safe relative paths"):
        create_source_snapshot(tmp_path, files=("../secret.csv",))
