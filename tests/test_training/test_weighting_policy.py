import pandas as pd

from src.training.weighting import TrainingWeightPolicy


def test_weight_policy_is_reproducible_and_non_uniform():
    policy = TrainingWeightPolicy(
        policy_id="recency_exp_test",
        lambda_decay=0.1,
        min_weight=0.1,
    )
    frame = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-10"])})
    weights = policy.calculate(frame)
    assert weights[0] < weights[1]
    assert policy.policy_hash == TrainingWeightPolicy.from_dict(policy.to_dict()).policy_hash
    assert policy.summarize(frame)["count"] == 2
