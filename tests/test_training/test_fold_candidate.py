"""Tiny sealed-candidate/live/replay fixtures; no real training data."""

from datetime import datetime, timezone
import json

import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.snapshots import create_source_snapshot
from src.evaluation.chronology import EvaluationPolicy
from src.evaluation.request_replay import replay_requests, request_record
from src.forecasting.baseline_backend import V2BaselineBackend
from src.models.versioning import ModelBundleManifest
from src.operations.forecast_runs import ForecastRunStore
from src.pipeline.forecast_service import ForecastService
from src.pipeline.v2_execution import execute_request
from src.training.fold_candidate import fit_fold_candidate


def test_fold_fit_ignores_future_outcomes_and_replay_recovers_identical_law(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    history = pd.DataFrame([
        {"GAME_ID": f"g{day}", "GAME_DATE": f"2025-01-0{day}",
         "PLAYER_ID": team * 10 + player, "TEAM_ID": team, "MIN": 30,
         "PTS": 15, "REB": 5, "AST": 4, "STL": 1, "BLK": 1, "TOV": 2}
        for day in (1, 2, 4) for team in (1, 2) for player in range(6)
    ])
    history.loc[history.GAME_ID.eq("g4"), "PTS"] = 99999
    history.to_csv(data / "nba_players.csv", index=False)
    pd.DataFrame([{"GAME_ID": "g3", "GAME_DATE": "2025-01-03",
                   "HOME_TEAM_ID": 1, "AWAY_TEAM_ID": 2,
                   "SCHEDULED_TIP": "2025-01-03T23:00:00+00:00"}]).to_csv(data / "schedule.csv", index=False)
    pd.DataFrame([{"PLAYER_ID": team * 10 + player, "TEAM_ID": team,
                   "START_DATE": "2025-01-01", "END_DATE": None,
                   "SOURCE": "nba_official", "COVERAGE_STATUS": "official",
                   "AVAILABLE_AT": "2025-01-01T00:00:00+00:00"}
                  for team in (1, 2) for player in range(6)]).to_csv(data / "rosters.csv", index=False)
    create_source_snapshot(data, snapshot_id="s", files=("nba_players.csv", "schedule.csv", "rosters.csv"),
                           created_at=datetime(2025, 1, 3, 12, tzinfo=timezone.utc))
    request = ForecastRequest(game_id="g3", game_date="2025-01-03",
        scheduled_tip="2025-01-03T23:00:00+00:00", forecast_cutoff="2025-01-03T13:00:00+00:00",
        home_team_id=1, away_team_id=2, horizon="morning", source_snapshot_id="s",
        model_bundle_id="template", scenario="diagnostic")
    policy = EvaluationPolicy(simulations=2)
    candidate = fit_fold_candidate(boundary_request=request, fit_game_ids=["g1", "g2"],
        models_dir=tmp_path / "models", data_dir=data, settings={"window": 5}, policy=policy, fold_id="f")
    rates = json.loads((candidate / "stat_rate_models/baseline.json").read_text())
    assert rates["players"]["10"]["PTS"]["mean"] == .5
    provenance = json.loads((candidate / "training_provenance.json").read_text())
    assert provenance["fit_game_ids"] == ["g1", "g2"]
    from dataclasses import replace

    request = replace(request, model_bundle_id=ModelBundleManifest.load(candidate / "bundle_manifest.json").bundle_id)
    service = ForecastService(model_backend=V2BaselineBackend(candidate))
    live = execute_request(service, request, data_dir=data, simulations=2, seed=4)
    replay = replay_requests([request_record(request, simulations=2, seed=4)], bundle_dir=candidate, data_dir=data)
    pd.testing.assert_frame_equal(live.samples, replay.samples)
    pd.testing.assert_frame_equal(live.forecasts.drop(columns="GENERATED_AT"),
                                  replay.forecasts.drop(columns="GENERATED_AT"))
    store = ForecastRunStore(tmp_path / "runs")
    first = store.get_or_create(request, lambda: live)
    def must_not_run():
        raise AssertionError("Retry must not rerun the model")
    second = store.get_or_create(request, must_not_run)
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)
