from dataclasses import replace

import pandas as pd

from src.evaluation.chronology import EvaluationPolicy, four_role_folds


def test_four_roles_are_game_atomic_and_policy_is_frozen():
    dates = pd.date_range("2025-01-01", periods=20)
    frame = pd.DataFrame([
        {"GAME_ID": str(i), "GAME_DATE": date, "REQUEST_ID": f"{i}-{h}",
         "SOURCE_SNAPSHOT_ID": f"s{i}"} for i, date in enumerate(dates) for h in range(2)
    ])
    policy = EvaluationPolicy(min_fit_games=2, min_tune_games=2,
                              min_calibrate_games=2, min_outer_games=2)
    result = four_role_folds(frame, policy=policy, min_train_days=4,
                             tune_days=2, calibrate_days=2, outer_days=4)
    assert result["sufficient_folds"]
    for fold in result["folds"]:
        seen = set()
        for role in fold["roles"].values():
            games = set(role["game_ids"])
            assert not seen & games
            seen |= games
            assert len(role["request_ids"]) == 2 * len(games)
    assert replace(policy, min_fit_games=3).policy_id != policy.policy_id
    sparse = four_role_folds(frame.iloc[::8], policy=policy, min_train_days=4,
                             tune_days=2, calibrate_days=2, outer_days=4)
    assert not sparse["sufficient_folds"]
    assert sparse["skipped"]
