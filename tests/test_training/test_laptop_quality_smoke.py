"""End-to-end smoke test for the ``laptop_quality`` training preset.

This test proves that ``python train.py --preset laptop_quality`` can run
through the **real** CatBoost training path and produce the required runtime
artifacts — without needing the full 52k-row dataset. It uses a tiny checked-in
fixture (``tests/fixtures/laptop_quality/data/``) and ``--mode quick`` to keep
training fast.

The fixture has ~2160 player-game rows across 3 seasons, 12 players, and 4
teams. The pipeline's ``len(fit_df) < 1000`` guard requires at least 1000
training rows, so the fixture is larger than the ticket's "30-60 row" minimum
suggestion but still trivial compared to the real dataset.

Smart feature selection is disabled (``--feature-selection off``) because this
ticket is about proving the **preset + artifact path**, not feature-selection
quality.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

from src.contracts.artifacts import ArtifactContract, validate_runtime_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DATA_DIR = PROJECT_ROOT / "tests" / "fixtures" / "laptop_quality" / "data"

REQUIRED_ARTIFACTS = [
    "pts_catboost.cbm",
    "reb_catboost.cbm",
    "ast_catboost.cbm",
    "stl_catboost.cbm",
    "blk_catboost.cbm",
    "tov_catboost.cbm",
    "feature_schema.pkl",
    "feature_cols.pkl",
    "blend_weights.pkl",
    "model_stack_metadata.pkl",
]

CANONICAL_TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
LAPTOP_QUALITY_GROUPS = [
    "rolling", "efficiency", "momentum", "context", "fatigue",
    "minutes_confidence", "rest_density", "matchup", "opponent_strength",
    "pace", "team_role", "recency_form", "archetype",
]


@pytest.mark.slow
@pytest.mark.integration
def test_laptop_quality_preset_reaches_artifact_generation(tmp_path):
    # Copy the checked-in fixture into the temp data dir so the DataLoader
    # finds nba_players.csv and nba_games.csv at the expected paths.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy2(FIXTURE_DATA_DIR / "nba_players.csv", data_dir / "nba_players.csv")
    shutil.copy2(FIXTURE_DATA_DIR / "nba_games.csv", data_dir / "nba_games.csv")

    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--preset", "laptop_quality",
        "--mode", "quick",
        "--feature-selection", "off",
        "--no-gpu",
        "--data-dir", str(data_dir),
        "--models-dir", str(models_dir),
        "--cache-dir", str(cache_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )

    assert result.returncode == 0, (
        f"train.py exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )

    # --- Assert all required runtime artifacts exist ---
    for name in REQUIRED_ARTIFACTS:
        assert (models_dir / name).exists(), f"Missing required artifact: {name}"

    # --- Transformer artifact must NOT be required ---
    # It may or may not exist; the contract just must not require it.
    # For laptop_quality (transformer_enabled=False) it should not exist.
    assert not (models_dir / "attention_transformer.pkl").exists(), (
        "Transformer artifact should not be produced when transformer_enabled=False"
    )

    # --- Check model_stack_metadata.pkl ---
    metadata = joblib.load(models_dir / "model_stack_metadata.pkl")
    assert metadata["training_preset"] == "laptop_quality"
    assert metadata["transformer_enabled"] is False
    assert metadata["model_count"] == 1
    assert metadata["feature_groups"] == LAPTOP_QUALITY_GROUPS
    assert metadata.get("feature_selection_enabled") is False

    # --- Artifact contract validation (transformer not required) ---
    validate_runtime_artifacts(
        ArtifactContract(
            models_dir=models_dir,
            transformer_required=False,
        )
    )
    # If we reach here, no exception was raised — the contract is satisfied.


@pytest.mark.slow
@pytest.mark.integration
def test_laptop_quality_fixture_data_is_valid():
    """Sanity-check the fixture data loads and has the required columns."""
    from src.preprocessing.data_loader import DataLoader

    loader = DataLoader(
        str(FIXTURE_DATA_DIR / "nba_players.csv"),
        str(FIXTURE_DATA_DIR / "nba_games.csv"),
    )
    merged = loader.merge_datasets()

    required = {"PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON_ID"}
    required |= set(CANONICAL_TARGETS)
    assert required.issubset(set(merged.columns)), (
        f"Fixture missing columns: {required - set(merged.columns)}"
    )

    assert len(merged) >= 1000, (
        f"Fixture too small for the pipeline's 1000-row minimum: {len(merged)} rows"
    )
    assert merged["SEASON_ID"].nunique() >= 2, "Fixture needs >=2 seasons"
    assert merged["PLAYER_ID"].nunique() >= 8, "Fixture needs >=8 players"
    assert merged["TEAM_ID"].nunique() >= 4, "Fixture needs >=4 teams"

    for target in CANONICAL_TARGETS:
        assert target in merged.columns
        assert merged[target].notna().any(), f"Target {target} is all-NaN in fixture"
