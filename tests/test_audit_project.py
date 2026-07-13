from __future__ import annotations

import hashlib
import json
import pickle

import audit_project


def _write_csv(path, contents):
    path.write_text(contents, encoding="utf-8")


def test_audit_reports_legacy_schema_and_missing_optional_sources(tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()

    _write_csv(
        data_dir / "nba_players.csv",
        "PLAYER_ID,GAME_ID,GAME_DATE,PTS\n1,100,2026-01-01,20\n",
    )
    _write_csv(
        data_dir / "nba_games.csv",
        "GAME_ID,GAME_DATE,PTS\n100,2026-01-01,110\n",
    )
    feature_columns = [
        "ROLLING_PTS_5",
        "PTS_TEAM",
        "REB_TEAM",
        "AST_TEAM",
        "FGA_TEAM",
        "FTA_TEAM",
        "OREB_TEAM",
        "DREB_TEAM",
        "TOV_TEAM",
    ]
    for filename in ("feature_cols.pkl", "feature_schema.pkl"):
        with (models_dir / filename).open("wb") as handle:
            pickle.dump(feature_columns, handle)

    before = hashlib.sha256((models_dir / "feature_cols.pkl").read_bytes()).hexdigest()
    report = audit_project.build_report(data_dir, models_dir, repo_root=tmp_path)
    after = hashlib.sha256((models_dir / "feature_cols.pkl").read_bytes()).hexdigest()

    assert before == after
    assert report["repository"]["commit"] is None if "commit" in report["repository"] else True
    assert report["data"]["csv_tables"][0]["row_count"] == 1
    assert report["data"]["missing_optional_files"] == [
        "player_bios.csv",
        "injury_history.csv",
        "advanced_tracking.csv",
    ]
    assert report["models"]["manifests"] == {
        "champion_json": False,
        "bundle_manifest_json": False,
    }
    assert report["models"]["feature_schema"]["feature_cols_count"] == 9
    assert report["models"]["feature_schema"]["forbidden_features"] == [
        "AST_TEAM",
        "DREB_TEAM",
        "FGA_TEAM",
        "FTA_TEAM",
        "OREB_TEAM",
        "PTS_TEAM",
        "REB_TEAM",
        "TOV_TEAM",
    ]


def test_audit_cli_emits_json_and_returns_zero_for_unsafe_artifacts(tmp_path, capsys):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    _write_csv(data_dir / "nba_players.csv", "PLAYER_ID\n1\n")
    _write_csv(data_dir / "nba_games.csv", "GAME_ID\n100\n")
    with (models_dir / "feature_cols.pkl").open("wb") as handle:
        pickle.dump(["PTS_TEAM"], handle)

    exit_code = audit_project.main(
        ["--data-dir", str(data_dir), "--models-dir", str(models_dir)]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_schema_version"] == "audit_report_v1"
    assert payload["models"]["feature_schema"]["forbidden_features"] == ["PTS_TEAM"]


def test_audit_rejects_unreadable_directories(tmp_path):
    exit_code = audit_project.main(
        ["--data-dir", str(tmp_path / "missing-data"), "--models-dir", str(tmp_path)]
    )

    assert exit_code == 2
