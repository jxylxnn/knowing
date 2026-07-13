import pandas as pd

from src.reasoning import PredictionTrace, ReasoningEngine


def test_reasoning_ranks_support_risk_and_sensitivity():
    trace = PredictionTrace(
        stat="PTS",
        catboost_prediction=25.0,
        transformer_prediction=30.0,
        pre_correction_prediction=27.0,
        final_prediction=28.0,
        residual_correction=1.0,
        confidence="MEDIUM",
        confidence_score=0.72,
        feature_contributions={
            "ROLL_PTS_5": 2.5,
            "MATCHUP_DEFENSE_PTS": -1.5,
            "PACE_FACTOR": 0.8,
        },
        model_version="v-test",
        data_quality="FULL",
    )
    context = pd.DataFrame([{"MINUTES_PRED": 30.0, "USAGE_RATE": 0.25}])
    report = ReasoningEngine().build_report(
        {"PTS": trace}, context=context, player_name="Test Player"
    )

    item = report.stats["PTS"]
    assert item.projection == 28.0
    assert item.supporting_evidence[0].label == "ROLL_PTS_5"
    assert any(e.label == "MATCHUP_DEFENSE_PTS" for e in item.opposing_evidence)
    assert any(s.change == "+3 minutes" for s in item.scenarios)
    assert "Transformer" in item.conflicts[0]
    assert report.to_dict()["schema_version"] == 1


def test_projection_cache_reasoning_is_honest_about_missing_model_attribution():
    engine = ReasoningEngine()
    report = engine.build_projection_report(
        {
            "pts_mean": 24.0,
            "PTS_INTERVAL_90_LOW": 17.0,
            "PTS_INTERVAL_90_HIGH": 31.0,
            "PTS_CONFIDENCE": "HIGH",
            "DATA_QUALITY": "FULL",
        },
        {
            "recent_avg": {"pts": 26.0},
            "recent_sample_size": 5,
            "matchup_avg": {"pts": 21.0},
            "matchup_sample_size": 3,
        },
        player_name="Cached Player",
    )
    item = report.stats["PTS"]
    assert item.projection == 24.0
    assert item.interval == (17.0, 31.0)
    assert any(e.label == "RECENT_FORM" for e in item.supporting_evidence)
    assert any(e.label == "MATCHUP_HISTORY" for e in item.opposing_evidence)
    assert item.trace.feature_contributions
