import pandas as pd

from src.training.presets import (
    CANONICAL_TARGETS,
    apply_recent_history_window,
    resolve_training_preset,
)


def test_resolve_training_preset_small_uses_reduced_stack():
    preset = resolve_training_preset("small")

    assert preset.name == "small"
    assert preset.transformer_enabled is False
    assert preset.default_mode == "quick"
    assert preset.default_model_size == "S"
    assert preset.recent_seasons == 2
    assert preset.enable_groups == (
        "rolling",
        "efficiency",
        "momentum",
        "pace",
        "opponent_strength",
        "archetype",
    )


def test_resolve_training_preset_laptop_quality_sits_between_small_and_full():
    preset = resolve_training_preset("laptop_quality")

    assert preset.name == "laptop_quality"
    # Transformer must be off for laptop training.
    assert preset.transformer_enabled is False
    # Between small (quick) and full (standard): standard mode, small model.
    assert preset.default_mode == "standard"
    assert preset.default_model_size == "S"
    # More history than small (2), less than full (all seasons).
    assert preset.recent_seasons == 3
    # Must still train the canonical six targets.
    assert preset.targets == CANONICAL_TARGETS
    # Broader feature set than small but lighter than full.
    small = resolve_training_preset("small")
    full = resolve_training_preset("full")
    assert len(small.enable_groups) < len(preset.enable_groups) < len(full.enable_groups)
    # Smart feature selection is on by default for this preset.
    assert preset.feature_selection is not None
    assert preset.feature_selection.get("enabled") is True
    assert preset.feature_selection_profile == "balanced"
    # Uses canonical feature-group names (not the ticket's aliases).
    expected_groups = (
        "rolling",
        "efficiency",
        "momentum",
        "context",
        "fatigue",
        "minutes_confidence",
        "rest_density",
        "matchup",
        "opponent_strength",
        "pace",
        "team_role",
        "recency_form",
        "archetype",
    )
    assert preset.enable_groups == expected_groups


def test_resolve_training_preset_laptop_quality_applies_config_overrides():
    overrides = {
        "laptop_quality": {
            "default_mode": "quick",
            "recent_seasons": 1,
            "feature_engineer": {
                "enable_groups": ["rolling", "efficiency"],
            },
        }
    }

    preset = resolve_training_preset("laptop_quality", overrides)

    assert preset.default_mode == "quick"
    assert preset.recent_seasons == 1
    assert preset.enable_groups == ("rolling", "efficiency")
    # Transformer stays off even when other fields are overridden.
    assert preset.transformer_enabled is False


def test_resolve_training_preset_laptop_quality_targets_all_six_stats():
    preset = resolve_training_preset("laptop_quality")

    assert list(preset.targets) == ["PTS", "REB", "AST", "STL", "BLK", "TOV"]


def test_resolve_training_preset_applies_config_overrides():
    overrides = {
        "small": {
            "default_mode": "standard",
            "default_model_size": "M",
            "transformer_enabled": False,
            "recent_seasons": 1,
            "feature_engineer": {
                "rolling_windows": [3, 5],
                "enable_groups": ["rolling", "efficiency"],
            },
        }
    }

    preset = resolve_training_preset("small", overrides)

    assert preset.default_mode == "standard"
    assert preset.default_model_size == "M"
    assert preset.recent_seasons == 1
    assert preset.rolling_windows == (3, 5)
    assert preset.enable_groups == ("rolling", "efficiency")


def test_apply_recent_history_window_keeps_most_recent_seasons():
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-02-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2024-01-01",
                ]
            ),
            "SEASON_ID": ["22022", "22022", "22023", "22023", "22024"],
            "PTS": [10, 11, 12, 13, 14],
        }
    )

    filtered = apply_recent_history_window(df, 2)

    assert list(filtered["SEASON_ID"].unique()) == ["22023", "22024"]
    assert len(filtered) == 3


def test_apply_recent_history_window_noops_without_season_id():
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "PTS": [10, 11],
        }
    )

    filtered = apply_recent_history_window(df, 2)

    pd.testing.assert_frame_equal(filtered, df)


def test_apply_recent_history_window_uses_season_year_when_season_id_is_absent():
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
            ),
            "SEASON_YEAR": ["2021-22", "2022-23", "2023-24", "2024-25"],
            "PTS": [10, 11, 12, 13],
        }
    )

    filtered = apply_recent_history_window(df, 2)

    assert list(filtered["SEASON_YEAR"]) == ["2023-24", "2024-25"]
