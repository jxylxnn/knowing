import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from src.data.nba_capture import capture_nba_rosters, capture_nba_schedule
from src.data.official_capture import capture_official_source
from src.data.rebuild_snapshot import rebuild_snapshot
from src.data.snapshots import create_source_snapshot, validate_source_snapshot


def _schedule_payload(season="2026-27"):
    return {
        "leagueSchedule": {
            "seasonYear": season,
            "gameDates": [
                {
                    "gameDate": "2026-10-21",
                    "games": [
                        {
                            "gameId": "0022600001",
                            "gameDateTimeUTC": "2026-10-22T00:30:00Z",
                            "gameStatus": 1,
                            "homeTeam": {"teamId": 1610612738},
                            "awayTeam": {"teamId": 1610612752},
                        },
                        {
                            "gameId": "0022600002",
                            "gameDateTimeUTC": None,
                            "gameStatus": 1,
                            "homeTeam": {"teamId": 0},
                            "awayTeam": {"teamId": 1610612747},
                        },
                    ],
                }
            ],
        }
    }


class _ScheduleResponse:
    def __init__(self, payload):
        self.content = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.url = (
            "https://cdn.nba.com/static/json/staticData/"
            "scheduleLeagueV2.json"
        )

    def raise_for_status(self):
        return None


def test_native_schedule_capture_uses_cdn_and_preserves_raw_response(
    tmp_path, monkeypatch
):
    response = _ScheduleResponse(_schedule_payload())
    monkeypatch.setattr(
        "src.data.nba_capture.requests.get", lambda *args, **kwargs: response
    )

    captured = capture_nba_schedule(tmp_path, season="2026-27")

    schedule = pd.read_csv(next(captured.rglob("schedule.csv")), dtype=str)
    quarantine = pd.read_csv(next(captured.rglob("quarantine.csv")), dtype=str)
    raw = next(captured.rglob("schedule_response.json")).read_bytes()
    receipt = json.loads(next(captured.rglob("receipt.json")).read_text())
    assert schedule.GAME_ID.tolist() == ["0022600001"]
    assert schedule.GAME_DATE.tolist() == ["2026-10-21"]
    assert schedule.SCHEDULED_TIP.tolist() == ["2026-10-22T00:30:00+00:00"]
    assert quarantine.GAME_ID.tolist() == ["0022600002"]
    assert raw == response.content
    assert receipt["receipts"][0]["url"] == response.url


def test_native_schedule_capture_retries_a_timeout(tmp_path, monkeypatch):
    response = _ScheduleResponse(_schedule_payload())
    outcomes = iter([requests.Timeout("slow NBA response"), response])
    sleeps = []
    timeouts = []

    def fake_get(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.data.nba_capture.requests.get", fake_get)
    monkeypatch.setattr("src.data.nba_capture.time.sleep", sleeps.append)

    captured = capture_nba_schedule(
        tmp_path,
        season="2026-27",
        max_attempts=2,
        retry_delay=0.25,
    )

    assert captured.is_dir()
    assert sleeps == [0.25]
    assert timeouts == [(10.0, 30.0), (10.0, 30.0)]


def test_native_schedule_capture_exhaustion_publishes_nothing(
    tmp_path, monkeypatch
):
    def always_timeout(*args, **kwargs):
        raise requests.Timeout("slow NBA response")

    monkeypatch.setattr("src.data.nba_capture.requests.get", always_timeout)

    with pytest.raises(
        RuntimeError, match=r"failed after 2 attempt\(s\).*Timeout.*slow NBA"
    ):
        capture_nba_schedule(
            tmp_path,
            season="2026-27",
            max_attempts=2,
            retry_delay=0,
        )

    assert not (tmp_path / "official_captures").exists()


def test_native_schedule_capture_rejects_another_season(tmp_path, monkeypatch):
    response = _ScheduleResponse(_schedule_payload(season="2025-26"))
    monkeypatch.setattr(
        "src.data.nba_capture.requests.get", lambda *args, **kwargs: response
    )

    with pytest.raises(ValueError, match="2025-26.*not 2026-27"):
        capture_nba_schedule(tmp_path, season="2026-27")


def test_native_schedule_capture_reports_schema_drift(tmp_path, monkeypatch):
    payload = _schedule_payload()
    del payload["leagueSchedule"]["gameDates"][0]["games"][0][
        "gameDateTimeUTC"
    ]
    response = _ScheduleResponse(payload)
    monkeypatch.setattr(
        "src.data.nba_capture.requests.get", lambda *args, **kwargs: response
    )

    with pytest.raises(ValueError, match="schema is missing: gameDateTimeUTC"):
        capture_nba_schedule(tmp_path, season="2026-27")


def test_native_roster_capture_retries_each_team(tmp_path, monkeypatch):
    team_id = 1610612738
    calls = []

    class NbaResponse:
        @staticmethod
        def get_url():
            return "https://stats.nba.com/stats/commonteamroster"

        @staticmethod
        def get_response():
            return '{"resultSets": []}'

    class RosterResult:
        @staticmethod
        def get_data_frame():
            return pd.DataFrame({"PLAYER_ID": [1], "TeamID": [team_id]})

    class Endpoint:
        nba_response = NbaResponse()
        common_team_roster = RosterResult()

    def fake_roster(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.Timeout("slow roster response")
        return Endpoint()

    monkeypatch.setattr(
        "nba_api.stats.endpoints.commonteamroster.CommonTeamRoster", fake_roster
    )

    captured = capture_nba_rosters(
        tmp_path,
        season="2026-27",
        team_ids=[team_id],
        max_attempts=2,
        retry_delay=0,
    )

    roster = pd.read_csv(next(captured.rglob("roster_membership.csv")))
    assert roster.PLAYER_ID.tolist() == [1]
    assert [call["timeout"] for call in calls] == [
        (10.0, 30.0),
        (10.0, 30.0),
    ]


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
