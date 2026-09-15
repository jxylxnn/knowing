from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.pipeline.v2_execution as v2_execution
from src.pipeline.v2_execution import execute_request
from src.simulation.v2_runner import latest_observed_roster
from src.simulation.v2_runner import horizon_cutoff


def test_horizon_cutoffs_are_timezone_aware_and_before_tip():
    tip = datetime(2026, 10, 20, 19, 30, tzinfo=ZoneInfo("America/New_York"))
    for horizon in ("previous_night", "morning", "pregame_90m", "pregame_30m"):
        cutoff = horizon_cutoff(tip, horizon)
        assert cutoff.tzinfo is not None
        assert cutoff < tip


def test_latest_observed_roster_uses_only_prior_games_and_latest_team():
    history = pd.DataFrame({
        "GAME_ID": ["older", "newer", "future", "same_day"],
        "GAME_DATE": ["2026-10-18", "2026-10-19", "2026-10-21", "2026-10-20"],
        "PLAYER_ID": [1, 1, 1, 2],
        "TEAM_ID": [10, 11, 12, 20],
    })

    roster = latest_observed_roster(history, date(2026, 10, 20))

    assert roster.set_index("PLAYER_ID")["TEAM_ID"].to_dict() == {1: 11}
    assert roster.set_index("PLAYER_ID")["GAME_ID"].to_dict() == {1: "newer"}


def test_latest_observed_roster_requires_game_id_for_stable_latest_selection():
    history = pd.DataFrame({
        "GAME_DATE": ["2026-10-19"],
        "PLAYER_ID": [1],
        "TEAM_ID": [10],
    })

    with pytest.raises(ValueError, match="GAME_ID"):
        latest_observed_roster(history, date(2026, 10, 20))


def test_execute_request_uses_observed_roster_only_for_degraded_scenario(monkeypatch):
    history = pd.DataFrame({
        "GAME_ID": ["prior"],
        "GAME_DATE": ["2026-10-19"],
        "PLAYER_ID": [1],
        "TEAM_ID": [10],
    })
    official_roster = pd.DataFrame({"PLAYER_ID": [2], "TEAM_ID": [20]})
    routes = {"observed": [], "official": []}

    monkeypatch.setattr(v2_execution, "assert_snapshot_game", lambda *_: None)
    monkeypatch.setattr(v2_execution, "load_request_history", lambda *_: history)
    monkeypatch.setattr(
        v2_execution,
        "load_request_status",
        lambda *_: pd.DataFrame({"PLAYER_ID": [1], "TEAM_ID": [10], "STATUS": ["AVAILABLE"]}),
    )
    monkeypatch.setattr(
        v2_execution,
        "materialize_scheduled_rows",
        lambda request, roster, history, *, statuses: pd.DataFrame({"PLAYER_ID": [1]}),
    )

    def observed_route(history_arg, game_date):
        routes["observed"].append((history_arg, game_date))
        return latest_observed_roster(history_arg, game_date)

    def official_route(*args):
        routes["official"].append(args)
        return official_roster

    monkeypatch.setattr("src.simulation.v2_runner.latest_observed_roster", observed_route)
    monkeypatch.setattr(v2_execution, "load_official_roster", official_route)

    class Service:
        backend = SimpleNamespace(model_version="bundle")

        def predict_game_distribution(self, request, contexts, **kwargs):
            return SimpleNamespace(request=request, contexts=contexts)

    def request(scenario):
        from src.contracts.forecast import ForecastRequest

        return ForecastRequest(
            game_id="g1",
            game_date=date(2026, 10, 20),
            scheduled_tip=datetime(2026, 10, 20, 19, tzinfo=ZoneInfo("America/New_York")),
            home_team_id=10,
            away_team_id=20,
            forecast_cutoff=datetime(2026, 10, 20, 9, tzinfo=ZoneInfo("America/New_York")),
            horizon="morning",
            source_snapshot_id="snapshot",
            model_bundle_id="bundle",
            scenario=scenario,
        )

    execute_request(
        Service(), request("degraded_observed_roster"), data_dir="unused"
    )
    execute_request(Service(), request("official"), data_dir="unused")

    assert len(routes["observed"]) == 1
    assert routes["observed"][0][0].equals(history)
    assert routes["observed"][0][1] == date(2026, 10, 20)
    assert len(routes["official"]) == 1
