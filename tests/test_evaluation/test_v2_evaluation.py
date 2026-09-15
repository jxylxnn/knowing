import pandas as pd
import pytest

from src.evaluation.baselines import add_lagged_baselines
from src.evaluation.folds import rolling_origin_folds
from src.evaluation.promotion import evaluate_v2_promotion
from src.evaluation.significance import paired_game_bootstrap


def test_baselines_are_prior_only():
    frame = pd.DataFrame({"PLAYER_ID": [1, 1, 1], "GAME_ID": [1, 2, 3], "GAME_DATE": ["2026-01-01", "2026-01-02", "2026-01-03"], "PTS": [10, 20, 30]})
    result = add_lagged_baselines(frame, targets=("PTS",))
    assert pd.isna(result.iloc[0]["BASELINE_LAST_PTS"])
    assert result.iloc[2]["BASELINE_ROLL_5_PTS"] == 15


def test_rolling_folds_and_bootstrap_use_game_groups():
    assert rolling_origin_folds(pd.date_range("2025-01-01", periods=200), min_train_days=90)
    rows = pd.DataFrame({"GAME_ID": [1, 1, 2, 2], "ACTUAL": [10, 10, 10, 10], "candidate": [9, 9, 9, 9], "baseline": [12, 12, 12, 12]})
    assert paired_game_bootstrap(rows, candidate_column="candidate", baseline_column="baseline")["delta_mae"] < 0


def test_promotion_requires_all_quality_gates():
    decision = evaluate_v2_promotion(normalized_mae_improvement=.02, target_improvements={"PTS": .01}, participation_beats_baseline=True, minutes_beats_baseline=True, coverage_80=.80, coverage_90=.90, point_in_time_valid=True)
    assert decision.eligible


def _complete_promotion_kwargs():
    targets = {target: 0.02 for target in ("PTS", "REB", "AST", "STL", "BLK", "TOV")}
    return {
        "normalized_mae_improvement": 0.02,
        "target_improvements": targets,
        "participation_beats_baseline": True,
        "minutes_beats_baseline": True,
        "coverage_80": 0.80,
        "coverage_90": 0.90,
        "point_in_time_valid": True,
        "bootstrap_ci": (-0.10, -0.01),
        "contract_flags": {"artifact_contract": True, "calibration": True},
        "live_replay_parity": True,
        "fallback_rate": 0.01,
        "degraded_rate": 0.02,
        "major_slice_reconciliation": {"all": 0.99},
        "major_slice_coverage_80": {"all": 0.80},
        "major_slice_coverage_90": {"all": 0.90},
        "core_reconciliation": 1.0,
        "require_complete_evidence": True,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalized_mae_improvement", float("nan")),
        ("normalized_mae_improvement", float("inf")),
        ("target_improvements", {"PTS": float("nan")}),
        ("target_improvements", {"PTS": float("-inf")}),
        ("contract_flags", {}),
        ("major_slice_reconciliation", {}),
        ("major_slice_coverage_80", {"all": float("nan")}),
        ("major_slice_coverage_90", {"all": float("inf")}),
        ("fallback_rate", 1.1),
        ("degraded_rate", -0.1),
    ],
)
def test_complete_promotion_evidence_rejects_invalid_values(field, value):
    kwargs = _complete_promotion_kwargs()
    kwargs[field] = value
    decision = evaluate_v2_promotion(**kwargs)
    assert not decision.eligible
    assert decision.reasons


def test_complete_promotion_evidence_requires_all_targets_and_contracts():
    kwargs = _complete_promotion_kwargs()
    kwargs["target_improvements"] = {"PTS": 0.02}
    kwargs["contract_flags"] = {"artifact_contract": True}
    decision = evaluate_v2_promotion(**kwargs)
    assert not decision.eligible
    assert any("missing" in reason for reason in decision.reasons)


def test_complete_promotion_evidence_accepts_threshold_boundaries():
    kwargs = _complete_promotion_kwargs()
    kwargs.update({
        "normalized_mae_improvement": 0.01,
        "target_improvements": {
            "PTS": -0.01, "REB": -0.01, "AST": -0.01,
            "STL": -0.02, "BLK": -0.02, "TOV": -0.02,
        },
        "coverage_80": 0.77,
        "coverage_90": 0.87,
        "fallback_rate": 0.05,
        "degraded_rate": 0.10,
        "major_slice_reconciliation": {"all": 0.95},
        "major_slice_coverage_80": {"all": 0.75},
        "major_slice_coverage_90": {"all": 0.85},
    })
    assert evaluate_v2_promotion(**kwargs).eligible
