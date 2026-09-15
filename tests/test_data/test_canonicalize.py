from datetime import datetime, timezone

import pandas as pd

from src.data.canonicalize import canonicalize_snapshot, load_canonical_table
from src.data.coverage import build_coverage_report
from src.contracts.errors import ContractError
from src.data.snapshots import create_source_snapshot


def test_canonicalize_snapshot_builds_reconciled_core_tables(tmp_path):
    players = pd.DataFrame([
        {"GAME_ID": "1", "TEAM_ID": "10", "PLAYER_ID": "100", "GAME_DATE": "2026-10-20"},
        {"GAME_ID": "1", "TEAM_ID": "20", "PLAYER_ID": "200", "GAME_DATE": "2026-10-20"},
    ])
    teams = pd.DataFrame([
        {"GAME_ID": "1", "TEAM_ID": "10", "GAME_DATE": "2026-10-20", "MATCHUP": "AAA vs. BBB"},
        {"GAME_ID": "1", "TEAM_ID": "20", "GAME_DATE": "2026-10-20", "MATCHUP": "BBB @ AAA"},
    ])
    players.to_csv(tmp_path / "nba_players.csv", index=False)
    teams.to_csv(tmp_path / "nba_games.csv", index=False)
    create_source_snapshot(tmp_path, snapshot_id="s1", created_at=datetime(2026, 8, 27, tzinfo=timezone.utc))

    report = canonicalize_snapshot(tmp_path, "s1")

    assert report.invalid_games == 0
    assert report.missing_player_team_game == 0
    assert len(load_canonical_table(tmp_path, "s1", "games")) == 1
    coverage = build_coverage_report(tmp_path, "s1")
    assert coverage.core_reconciled_fraction == 1.0
    assert coverage.observed_roster_only
    import json

    json.dumps(coverage.to_dict(), allow_nan=False)


def test_canonical_bytes_and_manifest_are_verified(tmp_path):
    import json
    import pytest

    test_canonicalize_snapshot_builds_reconciled_core_tables(tmp_path)
    directory = tmp_path / "canonical/snapshot=s1"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["transformation_sha256"]
    assert manifest["content_id"]
    table = directory / "player_games.parquet"
    frame = pd.read_parquet(table)
    frame["PTS"] = 999
    frame.to_parquet(table, index=False)
    with pytest.raises(ContractError, match="checksum"):
        load_canonical_table(tmp_path, "s1", "player_games")
    manifest["source_snapshot_id"] = "other"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="hash mismatch"):
        load_canonical_table(tmp_path, "s1", "games")


def test_neutral_venue_requires_matching_official_fixture():
    import pytest
    from src.data.canonicalize import _bind_historical_schedule, _team_games, _games

    raw = pd.DataFrame([
        {"GAME_ID": "22400147", "GAME_DATE": "2024-11-02", "TEAM_ID": team,
         "MATCHUP": "neutral @ neutral"} for team in ("10", "20")
    ])
    teams = _team_games(raw, "2026-09-07T14:00:00+00:00")
    assert _games(teams, "2026-09-07T14:00:00+00:00")[1] == 1
    schedule = pd.DataFrame([{
        "GAME_ID": "0022400147", "GAME_DATE": "2024-11-02",
        "HOME_TEAM_ID": "20", "AWAY_TEAM_ID": "10", "SOURCE": "nba_official",
        "AVAILABLE_AT": "2026-09-07T14:00:00+00:00",
        "SCHEDULED_TIP": "2024-11-03T00:00:00+00:00",
    }])
    repaired = _bind_historical_schedule(teams, schedule)
    games, invalid = _games(repaired, "2026-09-07T14:00:00+00:00")
    assert invalid == 0
    assert games.HOME_TEAM_ID.tolist() == ["20"]
    assert set(repaired.SCHEDULE_SOURCE_GAME_ID) == {"0022400147"}
    with pytest.raises(ContractError, match="conflicts"):
        _bind_historical_schedule(teams, schedule.assign(AWAY_TEAM_ID="30"))
