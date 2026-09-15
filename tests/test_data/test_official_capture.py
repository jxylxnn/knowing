from datetime import datetime, timezone

from src.data.official_capture import capture_official_source
from src.data.rebuild_snapshot import rebuild_snapshot
from src.data.snapshots import create_source_snapshot, validate_source_snapshot


def test_prospective_capture_and_rebuild_preserve_base(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def geturl(self):
            return "https://www.nba.com/fixture.csv"
        def read(self):
            return b"PLAYER_ID,TEAM_ID,START_DATE,END_DATE,AVAILABLE_AT\n1,10,2020-01-01,,2020-01-01T00:00:00Z\n"
    monkeypatch.setattr("src.data.official_capture.urlopen", lambda *args, **kwargs: Response())
    (tmp_path / "nba_players.csv").write_text("PLAYER_ID,TEAM_ID\n1,10\n")
    (tmp_path / "nba_games.csv").write_text("GAME_ID,TEAM_ID\n1,10\n")
    create_source_snapshot(tmp_path, snapshot_id="base",
                           created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    captured = capture_official_source(tmp_path, url="https://www.nba.com/fixture.csv",
                                       filename="roster_membership.csv")
    before = validate_source_snapshot(tmp_path, "base").to_dict()
    rebuilt = rebuild_snapshot(tmp_path, base_snapshot_id="base", captures=[captured])
    assert rebuilt.snapshot_id != "base"
    assert validate_source_snapshot(tmp_path, "base").to_dict() == before
    from src.features.snapshot_inputs import read_snapshot_csv

    roster = read_snapshot_csv(tmp_path, rebuilt.snapshot_id, ("roster_membership.csv",))
    assert roster.AVAILABLE_AT.iloc[0] != roster.UPSTREAM_AVAILABLE_AT.iloc[0]
    assert "2020-01-01" == roster.START_DATE.iloc[0]
