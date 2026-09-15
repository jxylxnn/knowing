"""Two tiny folds/two draws: integration wiring, not a modeling benchmark."""

from datetime import datetime, timezone
import json

import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.snapshots import create_source_snapshot
from src.evaluation.chronology import EvaluationPolicy
from src.evaluation.request_replay import request_record
from src.training.fold_campaign import run_fold_campaign


def test_two_small_folds_persist_actual_candidate_evidence(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {"GAME_ID": f"g{day}", "GAME_DATE": f"2025-01-{day:02d}", "PLAYER_ID": team * 10 + player,
         "TEAM_ID": team, "MIN": 30, "PTS": 15, "REB": 5, "AST": 4,
         "STL": 1, "BLK": 1, "TOV": 2, "ACTIVE": 1, "APPEARED": 1}
        for day in range(1, 9) for team in (1, 2) for player in range(6)
    ]
    truth = pd.DataFrame(rows)
    schedule = pd.DataFrame([{"GAME_ID": f"g{day}", "GAME_DATE": f"2025-01-{day:02d}",
        "HOME_TEAM_ID": 1, "AWAY_TEAM_ID": 2,
        "SCHEDULED_TIP": f"2025-01-{day:02d}T23:00:00+00:00"} for day in range(1, 9)])
    schedule.to_csv(data / "schedule.csv", index=False)
    pd.DataFrame([{"PLAYER_ID": team * 10 + player, "TEAM_ID": team,
                   "START_DATE": "2025-01-01", "END_DATE": None,
                   "SOURCE": "nba_official", "COVERAGE_STATUS": "official",
                   "AVAILABLE_AT": "2025-01-01T00:00:00+00:00"}
                  for team in (1, 2) for player in range(6)]).to_csv(data / "rosters.csv", index=False)
    records = []
    for day in range(1, 10):
        prior = truth.loc[truth.GAME_DATE < f"2025-01-{day:02d}"]
        prior.to_csv(data / "nba_players.csv", index=False)
        prior.to_csv(data / "player_game_eligibility.csv", index=False)
        create_source_snapshot(data, snapshot_id=f"s{day}",
            files=("nba_players.csv", "player_game_eligibility.csv", "schedule.csv", "rosters.csv"),
            created_at=datetime(2025, 1, day, 12, tzinfo=timezone.utc))
        if day < 9:
            req = ForecastRequest(game_id=f"g{day}", game_date=f"2025-01-{day:02d}",
                scheduled_tip=f"2025-01-{day:02d}T23:00:00+00:00",
                forecast_cutoff=f"2025-01-{day:02d}T13:00:00+00:00", home_team_id=1, away_team_id=2,
                horizon="morning", source_snapshot_id=f"s{day}", model_bundle_id="template",
                scenario="diagnostic")
            records.append(request_record(req, simulations=2))
    policy = EvaluationPolicy(min_fit_games=1, min_tune_games=1, min_calibrate_games=1,
                              min_outer_games=1, min_slice_rows=1, simulations=2, bootstrap_samples=100)
    output = run_fold_campaign(records, data_dir=data, models_dir=tmp_path / "models",
        output_dir=tmp_path / "campaigns", actuals_snapshot_id="s9", policy=policy,
        trials=({"window": 5},), windows={"min_train_days": 2, "tune_days": 1,
                                       "calibrate_days": 1, "outer_days": 2})
    result = json.loads((output / "summary.json").read_text())
    assert len(result["folds"]) == 2
    assert not result["promotion_eligible"]
    assert all(row["candidate_scorecard"]["known_rows"] == 144 for row in result["folds"])
    assert all(len(row["baseline_comparisons"]) == 6 for row in result["folds"])
