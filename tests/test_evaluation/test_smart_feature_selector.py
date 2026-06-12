"""Tests for the smart feature selector subsystem.

The selector combines group ablation, per-target pruning, and shadow
filtering.  These tests exercise each piece in isolation and validate the
end-to-end manifest contract that training and inference share.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.feature_group_ablation import (
    DEFAULT_TARGETS,
    FeatureGroupAblator,
    filter_group_columns,
)
from src.evaluation.shadow_feature_filter import (
    SHADOW_COLUMNS,
    ShadowFeatureFilter,
)
from src.evaluation.smart_feature_selector import (
    ProfileConfig,
    SelectionManifest,
    SelectorConfig,
    SmartFeatureSelector,
    TargetSelection,
    load_manifest,
)
from src.training.pipeline import TrainingPipeline
from src.utils.prediction_utils import FeatureSchema, FeatureSelector


# -----------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------


def _build_engineered_frame(rows: int = 600) -> pd.DataFrame:
    """Generate a feature-engineered-looking DataFrame for selector tests.

    The frame contains three signal groups (rolling, defense, pace) plus
    a strong per-target signal so the selector has something to keep.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-10-01", periods=rows, freq="D")
    player_ids = rng.integers(1000, 1010, size=rows)
    pts_signal = rng.normal(20, 5, size=rows)
    reb_signal = rng.normal(6, 2, size=rows)
    ast_signal = rng.normal(4, 1.5, size=rows)
    frame = pd.DataFrame(
        {
            "PLAYER_ID": player_ids,
            "TEAM_ID": rng.integers(1610612737, 1610612767, size=rows),
            "OPPONENT_ID": rng.integers(1610612737, 1610612767, size=rows),
            "GAME_DATE": dates,
            "PTS": pts_signal + rng.normal(0, 1.5, size=rows),
            "REB": reb_signal + rng.normal(0, 1.0, size=rows),
            "AST": ast_signal + rng.normal(0, 0.8, size=rows),
            "STL": rng.normal(0.8, 0.4, size=rows),
            "BLK": rng.normal(0.6, 0.3, size=rows),
            "TOV": rng.normal(1.5, 0.5, size=rows),
            # Rolling group — strong predictor of every target
            "ROLL_PTS_AVG_5": pts_signal + rng.normal(0, 0.5, size=rows),
            "ROLL_REB_AVG_5": reb_signal + rng.normal(0, 0.4, size=rows),
            "ROLL_AST_AVG_5": ast_signal + rng.normal(0, 0.3, size=rows),
            "ROLL_STL_AVG_5": rng.normal(0.8, 0.2, size=rows),
            "ROLL_BLK_AVG_5": rng.normal(0.6, 0.2, size=rows),
            "ROLL_TOV_AVG_5": rng.normal(1.5, 0.2, size=rows),
            # Defense group — moderate signal
            "DEF_PTS_ALLOWED": rng.normal(110, 5, size=rows),
            "DEF_REB_ALLOWED": rng.normal(45, 3, size=rows),
            "DEF_AST_ALLOWED": rng.normal(25, 2, size=rows),
            "DEF_STL_ALLOWED": rng.normal(8, 1, size=rows),
            "DEF_BLK_ALLOWED": rng.normal(5, 1, size=rows),
            "DEF_TOV_FORCED": rng.normal(13, 1.5, size=rows),
            # Pace group — pure noise
            "PACE_FACTOR": rng.normal(100, 1.5, size=rows),
            "TEAM_PACE_10": rng.normal(99, 1.4, size=rows),
            "OPP_PACE_10": rng.normal(101, 1.6, size=rows),
            # Context — random
            "REST_DAYS": rng.integers(0, 4, size=rows),
            "IS_HOME": rng.integers(0, 2, size=rows),
        }
    )
    return frame


SAMPLE_GROUP_COLUMNS = {
    "rolling": [
        "ROLL_PTS_AVG_5",
        "ROLL_REB_AVG_5",
        "ROLL_AST_AVG_5",
        "ROLL_STL_AVG_5",
        "ROLL_BLK_AVG_5",
        "ROLL_TOV_AVG_5",
    ],
    "defense": [
        "DEF_PTS_ALLOWED",
        "DEF_REB_ALLOWED",
        "DEF_AST_ALLOWED",
        "DEF_STL_ALLOWED",
        "DEF_BLK_ALLOWED",
        "DEF_TOV_FORCED",
    ],
    "pace": ["PACE_FACTOR", "TEAM_PACE_10", "OPP_PACE_10"],
    "context": ["REST_DAYS", "IS_HOME"],
}


# -----------------------------------------------------------------------
# FeatureGroupAblator
# -----------------------------------------------------------------------


def test_feature_group_ablation_returns_per_target_scores():
    frame = _build_engineered_frame()
    feature_cols = [c for cols in SAMPLE_GROUP_COLUMNS.values() for c in cols]
    ablator = FeatureGroupAblator(targets=("PTS", "REB", "AST"), random_state=7)
    report = ablator.run(
        full_df=frame,
        feature_cols=feature_cols,
        group_columns=SAMPLE_GROUP_COLUMNS,
        targets=("PTS", "REB", "AST"),
    )

    assert report.baseline_metrics, "Baseline metrics should be populated"
    assert any(score.target == "PTS" for score in report.group_scores)
    assert any(score.target == "REB" for score in report.group_scores)
    assert any(score.target == "AST" for score in report.group_scores)

    # Each GroupScore records the MAE delta when the group is removed.
    for score in report.group_scores:
        assert score.ablated_feature_count > 0
        assert score.n_train > 0 and score.n_val > 0
        assert np.isfinite(score.baseline_mae)
        assert np.isfinite(score.ablated_mae)


def test_feature_group_ablation_handles_missing_target():
    frame = _build_engineered_frame().drop(columns=["BLK"])
    feature_cols = [c for cols in SAMPLE_GROUP_COLUMNS.values() for c in cols]
    ablator = FeatureGroupAblator()
    report = ablator.run(
        full_df=frame,
        feature_cols=feature_cols,
        group_columns=SAMPLE_GROUP_COLUMNS,
        targets=("PTS", "BLK"),
    )
    assert "PTS" in report.baseline_metrics
    assert "BLK" not in report.baseline_metrics


def test_filter_group_columns_drops_disallowed_groups():
    filtered = filter_group_columns(SAMPLE_GROUP_COLUMNS, ["rolling", "defense"])
    assert set(filtered) == {"rolling", "defense"}
    assert "pace" not in filtered
    assert "context" not in filtered


# -----------------------------------------------------------------------
# ShadowFeatureFilter
# -----------------------------------------------------------------------


def test_shadow_filter_drops_noise_columns():
    frame = _build_engineered_frame()
    feature_cols = [
        "ROLL_PTS_AVG_5",      # strong signal
        "ROLL_REB_AVG_5",
        "PACE_FACTOR",         # pure noise
        "OPP_PACE_10",         # pure noise
        "TEAM_PACE_10",        # pure noise
    ]
    flt = ShadowFeatureFilter(
        targets=("PTS", "REB"),
        random_state=11,
        min_keep=2,
    )
    result = flt.run(frame, feature_cols, "PTS")
    # Shadow filter must record importance values for every column.
    importance_features = {i.feature for i in result.importances}
    assert importance_features.issuperset(set(feature_cols) | set(SHADOW_COLUMNS))
    # At least one of the noisy pace features should be ranked below the
    # shadow floor — the filter may or may not drop it, but the
    # importance vector must be populated.
    assert any(i.feature in {"PACE_FACTOR", "OPP_PACE_10", "TEAM_PACE_10"}
               and i.below_shadow_median
               for i in result.importances)
    # The min_keep invariant should never reduce the kept set below 2.
    assert len(result.kept_features) >= 2


def test_shadow_filter_keeps_all_when_signal_is_constant():
    frame = _build_engineered_frame()
    feature_cols = ["ROLL_PTS_AVG_5", "ROLL_REB_AVG_5"]
    flt = ShadowFeatureFilter(targets=("STL",), random_state=4)
    # Force the target to a constant so the model cannot learn anything.
    frame["STL"] = 1.0
    result = flt.run(frame, feature_cols, "STL")
    assert result.kept_features == list(feature_cols)
    assert result.dropped_features == []


# -----------------------------------------------------------------------
# SmartFeatureSelector
# -----------------------------------------------------------------------


def _make_selector(profile: str = "balanced") -> SmartFeatureSelector:
    cfg = SelectorConfig(enabled=True, profile=profile, target_specific=True)
    return SmartFeatureSelector(config=cfg, profile=ProfileConfig(name=profile))


def test_smart_selector_produces_per_target_manifest(tmp_path):
    frame = _build_engineered_frame()
    feature_cols = [c for cols in SAMPLE_GROUP_COLUMNS.values() for c in cols]
    selector = _make_selector("balanced")
    manifest_path = tmp_path / "manifest.json"
    selector.config.output_path = str(manifest_path)

    manifest = selector.run(
        full_df=frame,
        feature_cols=feature_cols,
        group_columns=SAMPLE_GROUP_COLUMNS,
        targets=("PTS", "REB", "AST"),
    )

    # Manifest contract
    assert manifest.enabled is True
    assert manifest.profile == "balanced"
    assert manifest.target_specific is True
    assert set(manifest.targets) == {"PTS", "REB", "AST"}
    assert set(manifest.selected_features_by_target) == {"PTS", "REB", "AST"}

    # Each target must have at least one selected feature and they must
    # be drawn from the input feature pool.
    for target, cols in manifest.selected_features_by_target.items():
        assert cols, f"No features selected for {target}"
        for col in cols:
            assert col in feature_cols, f"{col} not in input features"

    # File was written
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["enabled"] is True
    assert "selected_features_by_target" in payload
    assert "created_at" in payload


def test_smart_selector_keeps_global_list_when_target_specific_disabled(tmp_path):
    frame = _build_engineered_frame()
    feature_cols = [c for cols in SAMPLE_GROUP_COLUMNS.values() for c in cols]
    selector = _make_selector("fast")
    selector.config.target_specific = False
    selector.config.output_path = str(tmp_path / "manifest.json")
    manifest = selector.run(
        full_df=frame,
        feature_cols=feature_cols,
        group_columns=SAMPLE_GROUP_COLUMNS,
        targets=("PTS", "REB"),
    )
    assert manifest.target_specific is False
    for target in ("PTS", "REB"):
        assert manifest.selected_features_by_target[target] == (
            manifest.selected_features_global
        )


def test_smart_selector_records_ablation_report_metadata(tmp_path):
    frame = _build_engineered_frame()
    feature_cols = [c for cols in SAMPLE_GROUP_COLUMNS.values() for c in cols]
    selector = _make_selector("balanced")
    selector.config.output_path = str(tmp_path / "manifest.json")
    manifest = selector.run(
        full_df=frame,
        feature_cols=feature_cols,
        group_columns=SAMPLE_GROUP_COLUMNS,
        targets=("PTS",),
    )
    metadata = manifest.metadata
    assert "weights" in metadata
    assert metadata["weights"]["backtest_gain"] == pytest.approx(0.40, abs=1e-6)
    assert metadata["n_input_features"] == len(feature_cols)
    assert metadata["n_selected_features_global"] >= 1
    assert "ablation_baseline" in metadata


def test_smart_selector_load_round_trip(tmp_path):
    frame = _build_engineered_frame()
    feature_cols = [c for cols in SAMPLE_GROUP_COLUMNS.values() for c in cols]
    selector = _make_selector("fast")
    manifest_path = tmp_path / "manifest.json"
    selector.config.output_path = str(manifest_path)
    selector.run(
        full_df=frame,
        feature_cols=feature_cols,
        group_columns=SAMPLE_GROUP_COLUMNS,
        targets=("PTS", "REB"),
    )

    loaded = load_manifest(manifest_path)
    assert loaded.profile == "fast"
    assert loaded.targets == ["PTS", "REB"]
    for target, cols in loaded.selected_features_by_target.items():
        assert cols
        for col in cols:
            assert col in feature_cols


# -----------------------------------------------------------------------
# Profile + config helpers
# -----------------------------------------------------------------------


def test_profile_config_resolve_uses_overrides():
    profile = ProfileConfig.resolve(
        "balanced",
        profiles_cfg={
            "balanced": {
                "run_group_ablation": True,
                "run_individual_pruning": False,
                "run_shadow_filter": True,
            }
        },
    )
    assert profile.run_individual_pruning is False
    assert profile.run_group_ablation is True
    assert profile.run_shadow_filter is True


def test_selector_config_from_dict_uses_defaults_when_missing():
    cfg = SelectorConfig.from_config({})
    assert cfg.enabled is False
    assert cfg.profile == "balanced"
    assert cfg.target_specific is True


# -----------------------------------------------------------------------
# Pipeline integration
# -----------------------------------------------------------------------


def _make_fake_pipeline(tmp_path, monkeypatch) -> TrainingPipeline:
    """Create a TrainingPipeline that does not require CatBoost."""
    from src.training import catboost_trainer as catboost_module

    class _Stub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.feature_names_ = []
            self.is_trained_ = False

    monkeypatch.setattr(catboost_module, "CatBoostRegressor", _Stub)
    pipeline = TrainingPipeline(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        parallel=False,
        use_gpu=False,
    )
    pipeline.model_config["transformer"]["enabled"] = False
    return pipeline


def test_pipeline_feature_cols_for_target_falls_back_to_master(pipeline_stub):
    pipeline = pipeline_stub
    pipeline.feature_cols = ["A", "B", "C"]
    pipeline.target_feature_cols = {}
    assert pipeline._feature_cols_for_target("PTS") == ["A", "B", "C"]


def test_pipeline_feature_cols_for_target_uses_per_target_subset(pipeline_stub):
    pipeline = pipeline_stub
    pipeline.feature_cols = ["A", "B", "C", "D"]
    pipeline.target_feature_cols = {
        "PTS": ["A", "B"],
        "REB": ["C", "D"],
    }
    assert pipeline._feature_cols_for_target("PTS") == ["A", "B"]
    assert pipeline._feature_cols_for_target("REB") == ["C", "D"]
    # Unknown target falls back to the master list.
    assert pipeline._feature_cols_for_target("AST") == ["A", "B", "C", "D"]


def test_pipeline_apply_manifest_records_per_target_lists(pipeline_stub):
    pipeline = pipeline_stub
    pipeline.feature_cols = ["A", "B", "C", "D", "E"]
    pipeline.TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    manifest = {
        "target_specific": True,
        "selected_features_by_target": {
            "PTS": ["A", "B"],
            "REB": ["C", "D"],
            "AST": ["B", "E"],
        },
        "selected_features_global": ["A", "B", "C", "D", "E"],
    }
    pipeline.apply_feature_selection_manifest(manifest)
    assert pipeline.target_feature_cols["PTS"] == ["A", "B"]
    assert pipeline.target_feature_cols["REB"] == ["C", "D"]
    assert pipeline.target_feature_cols["AST"] == ["B", "E"]
    # Targets not covered by the manifest keep the master list via the
    # default in _feature_cols_for_target.
    assert pipeline.target_feature_cols.get("STL") is None


def test_pipeline_apply_manifest_filters_unknown_columns(pipeline_stub):
    pipeline = pipeline_stub
    pipeline.feature_cols = ["A", "B", "C"]
    pipeline.TARGETS = ["PTS"]
    manifest = {
        "target_specific": True,
        "selected_features_by_target": {"PTS": ["A", "B", "GHOST"]},
    }
    pipeline.apply_feature_selection_manifest(manifest)
    assert pipeline.target_feature_cols["PTS"] == ["A", "B"]


def test_pipeline_model_stack_metadata_includes_selection_when_enabled(
    pipeline_stub, monkeypatch
):
    pipeline = pipeline_stub
    pipeline.training_preset = "small"
    pipeline.feature_group_selection = ["rolling"]
    pipeline.feature_cols = ["A", "B", "C"]
    pipeline.target_feature_cols = {"PTS": ["A", "B"], "REB": ["B", "C"]}
    pipeline.feature_selection_manifest = {
        "target_specific": True,
        "profile": "balanced",
        "output_path": "models/feature_selection_manifest.json",
    }
    pipeline.blend_weights = {"PTS": {"catboost": 1.0, "transformer": 0.0}}
    pipeline._save_model_stack_metadata()

    import joblib

    metadata = joblib.load(pipeline.models_dir / "model_stack_metadata.pkl")
    assert metadata["feature_selection_enabled"] is True
    assert metadata["feature_selection_target_specific"] is True
    assert metadata["feature_selection_profile"] == "balanced"
    assert metadata["selected_features_by_target"]["PTS"] == ["A", "B"]
    assert metadata["selected_features_by_target"]["REB"] == ["B", "C"]
    assert metadata["feature_selection_manifest_path"] == (
        "models/feature_selection_manifest.json"
    )


def test_pipeline_model_stack_metadata_without_selection_is_disabled(
    pipeline_stub,
):
    pipeline = pipeline_stub
    pipeline.feature_cols = ["A", "B"]
    pipeline.target_feature_cols = {}
    pipeline.blend_weights = {}
    pipeline._save_model_stack_metadata()
    import joblib

    metadata = joblib.load(pipeline.models_dir / "model_stack_metadata.pkl")
    assert metadata["feature_selection_enabled"] is False
    assert "selected_features_by_target" not in metadata


# -----------------------------------------------------------------------
# FeatureSelector target-specific schema
# -----------------------------------------------------------------------


def test_feature_selector_select_for_target_respects_whitelist():
    df = pd.DataFrame(
        {
            "ROLL_PTS_AVG_5": [10.0, 12.0, 15.0, 11.0],
            "ROLL_REB_AVG_5": [4.0, 5.0, 6.0, 4.5],
            "PACE_FACTOR": [100.0, 101.0, 99.0, 102.0],
            "PTS": [10, 12, 15, 11],
        }
    )
    selector = FeatureSelector(["PTS"])
    cols = selector.select_features_for_target(
        df, target="PTS", allowed_features=["ROLL_PTS_AVG_5", "PACE_FACTOR", "GHOST"]
    )
    assert cols == ["ROLL_PTS_AVG_5", "PACE_FACTOR"]
    assert selector.feature_schema is not None
    assert isinstance(selector.feature_schema, FeatureSchema)


def test_feature_selector_select_for_target_without_whitelist_matches_default():
    df = pd.DataFrame(
        {
            "ROLL_PTS_AVG_5": [10.0, 12.0, 15.0],
            "PACE_FACTOR": [100.0, 101.0, 99.0],
            "PTS": [10, 12, 15],
        }
    )
    selector = FeatureSelector(["PTS"])
    cols = selector.select_features_for_target(df, target="PTS")
    assert "ROLL_PTS_AVG_5" in cols
    assert "PTS" not in cols


# -----------------------------------------------------------------------
# Fixtures for pipeline tests
# -----------------------------------------------------------------------


@pytest.fixture
def pipeline_stub(tmp_path, monkeypatch):
    return _make_fake_pipeline(tmp_path, monkeypatch)
