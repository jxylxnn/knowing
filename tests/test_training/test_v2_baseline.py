from datetime import date, timedelta

import pandas as pd

from src.forecasting.baseline_backend import V2BaselineBackend
from src.models.bundle import validate_v2_bundle
from src.training.v2_baseline import train_v2_baseline


def test_v2_baseline_training_creates_loadable_immutable_bundle(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    player_rows = []
    team_rows = []
    start = date(2025, 1, 1)
    for index in range(16):
        game_date = (start + timedelta(days=index * 10)).isoformat()
        game_id = str(index + 1)
        team_rows.extend([
            {"GAME_ID": game_id, "GAME_DATE": game_date, "TEAM_ID": "10", "MATCHUP": "AAA vs. BBB"},
            {"GAME_ID": game_id, "GAME_DATE": game_date, "TEAM_ID": "20", "MATCHUP": "BBB @ AAA"},
        ])
        for player_id, team_id in ((100, 10), (200, 20)):
            player_rows.append({
                "GAME_ID": game_id,
                "GAME_DATE": game_date,
                "PLAYER_ID": player_id,
                "TEAM_ID": team_id,
                "MIN": 30,
                "PTS": 15,
                "REB": 5,
                "AST": 4,
                "STL": 1,
                "BLK": 1,
                "TOV": 2,
            })
    pd.DataFrame(player_rows).to_csv(data_dir / "nba_players.csv", index=False)
    pd.DataFrame(team_rows).to_csv(data_dir / "nba_games.csv", index=False)
    config = tmp_path / "v2.yaml"
    config.write_text(
        "architecture: v2\n"
        "evaluation:\n"
        "  validation_days: 10\n"
        "  test_days: 10\n"
        "  min_train_days: 30\n",
        encoding="utf-8",
    )

    from src.data.snapshots import create_source_snapshot

    create_source_snapshot(data_dir, snapshot_id="snapshot-test")
    candidate = train_v2_baseline(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        config_path=config,
        snapshot_id="snapshot-test",
        run_id="candidate-test",
    )

    report = validate_v2_bundle(candidate)
    assert report.source_snapshot_id == "snapshot-test"
    backend = V2BaselineBackend(candidate)
    contexts = backend.prepare_contexts(pd.DataFrame([
        {"PLAYER_ID": team * 10 + offset, "TEAM_ID": team}
        for team in (10, 20) for offset in range(6)
    ]))
    assert contexts.groupby("TEAM_ID")["UNCONDITIONAL_MINUTES"].sum().round(8).eq(240).all()
    prediction = backend.predict_player_stats(contexts.iloc[[0]], include_confidence=True)
    assert prediction["PTS"] > 0

    import json

    score = json.loads((candidate / "candidate_scorecard.json").read_text())
    assert score["status"] == "not_evaluated"
    assert not score["aggregate"]
    provenance = json.loads((candidate / "training_provenance.json").read_text())
    assert provenance["rows"] == len(player_rows)
    assert provenance["training_end"] == game_date
    assert provenance["canonical_content_id"]


def test_explicit_missing_snapshot_is_not_created(tmp_path):
    import pytest

    config = tmp_path / "v2.yaml"
    config.write_text("architecture: v2\n")
    with pytest.raises(FileNotFoundError, match="Requested source snapshot"):
        train_v2_baseline(
            data_dir=tmp_path, models_dir=tmp_path / "models",
            config_path=config, snapshot_id="nonexistent",
        )
    assert not (tmp_path / "raw").exists()
