"""Small offline fixtures for request/source identity and roster safety."""

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.contracts.forecast import ForecastRequest
from src.contracts.errors import ContractError
from src.data.snapshots import create_source_snapshot
from src.features.scheduled import materialize_scheduled_rows
from src.features.snapshot_inputs import (
    assert_snapshot_game, load_official_roster, load_request_history,
)


def request():
    return ForecastRequest(
        game_id="001", game_date="2026-10-20",
        scheduled_tip="2026-10-20T23:00:00+00:00", home_team_id=1, away_team_id=2,
        forecast_cutoff="2026-10-20T13:00:00+00:00", horizon="morning",
        source_snapshot_id="s1", model_bundle_id="b1",
    )


def snapshot(root, *, available="2026-10-19T12:00:00+00:00", coverage="official"):
    pd.DataFrame([
        {"GAME_ID": "000", "GAME_DATE": "2026-10-18", "PLAYER_ID": 10,
         "TEAM_ID": 1, "PTS": 22},
        {"GAME_ID": "001", "GAME_DATE": "2026-10-20", "PLAYER_ID": 10,
         "TEAM_ID": 1, "PTS": 999},
    ]).to_csv(root / "nba_players.csv", index=False)
    pd.DataFrame([{
        "GAME_ID": "001", "GAME_DATE": "2026-10-20",
        "SCHEDULED_TIP": "2026-10-20T23:00:00+00:00",
        "HOME_TEAM_ID": 1, "AWAY_TEAM_ID": 2,
    }]).to_csv(root / "schedule.csv", index=False)
    pd.DataFrame([
        {"PLAYER_ID": player, "TEAM_ID": team, "START_DATE": "2026-10-01",
         "END_DATE": None, "AVAILABLE_AT": available, "SOURCE": "nba_official",
         "COVERAGE_STATUS": coverage}
        for player, team in [(10, 1), (11, 1), (20, 2)]
    ]).to_csv(root / "roster_membership.csv", index=False)
    create_source_snapshot(
        root, files=("nba_players.csv", "schedule.csv", "roster_membership.csv"),
        snapshot_id="s1", created_at=datetime(2026, 10, 20, 12, tzinfo=timezone.utc),
    )


def test_request_reads_snapshot_not_mutable_history_and_binds_game(tmp_path):
    snapshot(tmp_path)
    (tmp_path / "nba_players.csv").write_text("wrong,data\n")
    assert_snapshot_game(tmp_path, request())
    history = load_request_history(tmp_path, request())
    assert history.GAME_ID.tolist() == ["000"]
    assert history.PTS.tolist() == ["22"]
    with pytest.raises(ContractError, match="differs"):
        assert_snapshot_game(tmp_path, replace(request(), home_team_id=3))
    roster = load_official_roster(tmp_path, request())
    assert "11" in set(roster.PLAYER_ID)  # No appearance history required.
    rows = materialize_scheduled_rows(request(), roster.assign(PTS=999), history)
    assert "PTS" not in rows


@pytest.mark.parametrize("available,coverage", [
    ("2026-10-21T00:00:00+00:00", "official"),
    ("2026-10-19T00:00:00", "official"),
    ("2026-10-19T00:00:00+00:00", "observed_appearances_only"),
])
def test_unavailable_or_undated_or_derived_roster_fails(tmp_path, available, coverage):
    snapshot(tmp_path, available=available, coverage=coverage)
    with pytest.raises(ContractError):
        load_official_roster(tmp_path, request())


def test_corrupt_snapshot_fails_before_materialization(tmp_path):
    snapshot(tmp_path)
    (tmp_path / "raw/core/s1/nba_players.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_request_history(tmp_path, request())


def test_official_roster_rejects_stale_capture(tmp_path):
    snapshot(tmp_path)
    with pytest.raises(ContractError, match="current-day"):
        load_official_roster(tmp_path, replace(
            request(), game_date="2026-10-21",
            scheduled_tip="2026-10-21T23:00:00+00:00",
            forecast_cutoff="2026-10-21T13:00:00+00:00",
        ))
