import pandas as pd

from src.training.presets import (
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
