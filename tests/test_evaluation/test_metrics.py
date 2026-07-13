import numpy as np

from src.evaluation.metrics import compute_target_metrics


def test_target_metrics_mape_is_stable_for_zero_box_scores():
    metrics = compute_target_metrics(
        "STL",
        actuals=np.array([0.0, 2.0]),
        predictions=np.array([1.0, 3.0]),
    )

    # The zero-actual row is scaled by one stat unit instead of an arbitrary
    # epsilon, so the metric remains interpretable: mean(1 / 1, 1 / 2).
    assert metrics.mape == 0.75
    assert np.isfinite(metrics.mape)
