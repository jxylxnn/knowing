import numpy as np

from src.evaluation.continual_learning import (
    PromotionPolicy,
    evaluate_promotion,
    paired_bootstrap_improvement,
)


def _metrics(value):
    return {stat: value for stat in ("PTS", "REB", "AST", "STL", "BLK", "TOV")}


def _errors(scale):
    return {stat: np.full(1200, scale, dtype=float) for stat in _metrics(1)}


def test_paired_bootstrap_and_promotion_gate_accept_real_improvement():
    candidate = np.linspace(0.7, 0.9, 1200)
    low, high = paired_bootstrap_improvement(np.ones(1200), candidate)
    assert low > 0
    assert high > low
    decision = evaluate_promotion(
        _metrics(1.0),
        _metrics(0.8),
        champion_errors=_errors(1.0),
        candidate_errors=_errors(0.8),
        champion_rows={stat: 1200 for stat in _metrics(1)},
        candidate_rows={stat: 1200 for stat in _metrics(1)},
        policy=PromotionPolicy(bootstrap_samples=100),
    )
    assert decision.eligible


def test_promotion_rejects_regression_and_insufficient_rows():
    decision = evaluate_promotion(
        _metrics(1.0),
        _metrics(1.1),
        champion_errors=_errors(1.0),
        candidate_errors=_errors(1.1),
        champion_rows={stat: 10 for stat in _metrics(1)},
        candidate_rows={stat: 10 for stat in _metrics(1)},
        policy=PromotionPolicy(bootstrap_samples=100),
    )
    assert not decision.eligible
    assert any("minimum" in reason for reason in decision.reasons)
    assert any("below" in reason or "lower bound" in reason for reason in decision.reasons)
