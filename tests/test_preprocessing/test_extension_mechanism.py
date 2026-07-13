"""Tests for the pluggable data-extension mechanism.

Covers:
* FeatureGroupRegistry discovery (built-ins + the bundled example extension)
* FeatureSelector auto-folds extension-declared safe prefixes (no silent drop)
* FeatureEngineer includes extension groups while preserving built-in order
* ScraperRegistry discovery, enable-gating, and core-output collision guard
* presets ``"all"`` sentinel resolves to the registry truth (drift fix)
* end-to-end: example extension features are produced AND kept by the selector
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features.registry import (
    all_feature_groups,
    get_registry,
)
from src.data.base_scraper import (
    BaseScraper,
    CORE_PROTECTED_FILES,
    ScraperRegistry,
)


# ---------------------------------------------------------------------------
# FeatureGroupRegistry
# ---------------------------------------------------------------------------
def test_registry_discovers_builtin_groups():
    registry = get_registry()
    names = registry.builtin_group_names()
    assert "rolling" in names
    # The four lifecycle groups that were previously missing from the
    # hardcoded presets tuple must now be present (drift fix).
    for g in ("injury_risk", "aging_curve", "kan_aging", "skill_development"):
        assert g in names, f"{g} missing from registry builtins"


def test_registry_discovers_example_extension():
    names = get_registry().all_group_names()
    assert "advanced_tracking" in names


def test_all_feature_groups_includes_extensions():
    groups = all_feature_groups()
    assert "rolling" in groups
    assert "advanced_tracking" in groups


def test_registry_build_groups_preserves_builtin_order():
    registry = get_registry()
    groups = registry.build_groups(include_extensions=True)
    names = [g.name for g in groups]
    # Built-ins come first, in canonical order.
    assert names[:3] == ["rolling", "efficiency", "momentum"]
    # Extensions are appended after.
    assert names[-1] == "advanced_tracking"
    assert len(names) == len(set(names))  # no duplicates


def test_registry_build_groups_without_extensions():
    registry = get_registry()
    groups = registry.build_groups(include_extensions=False)
    names = [g.name for g in groups]
    assert "advanced_tracking" not in names
    assert "rolling" in names


def test_registry_safe_prefixes_include_extension_declarations():
    prefixes = get_registry().safe_prefixes()
    assert "ADVTRACK_" in prefixes
    # A built-in prefix is still present.
    assert "ROLL_" in prefixes


def test_registry_safe_keywords_include_builtin_set():
    keywords = get_registry().safe_keywords()
    assert "TREND" in keywords
    assert "FATIGUE" in keywords


def test_discover_extension_groups_isolates_import_errors(tmp_path):
    # Build a fake extensions package with one good and one bad module.
    import importlib

    pkg_name = "_extpkg_test_isolation"
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text(
        "from src.preprocessing.features.base import FeatureGroup\n"
        "class GoodGroup(FeatureGroup):\n"
        "    @property\n"
        "    def name(self): return 'good_ext'\n"
        "    def create(self, df, *, diagnostics=None, context=None): return df\n"
    )
    (pkg / "bad.py").write_text("raise RuntimeError('boom')\n")

    sys.path.insert(0, str(tmp_path))
    try:
        importlib.import_module(pkg_name)
        # Re-invoke discovery scoped at the temp package by pointing both the
        # directory and the dotted package name at the temp tree.
        from src.preprocessing.features import registry as reg_mod

        found = reg_mod.discover_extension_groups(str(pkg), package_name=pkg_name)
        names = [cls().name for cls in found]
        assert "good_ext" in names
        # The bad module is skipped, not fatal — good_ext still discovered.
        assert len(found) >= 1
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(pkg_name, None)
        for m in list(sys.modules):
            if m.startswith(pkg_name + "."):
                sys.modules.pop(m, None)


# ---------------------------------------------------------------------------
# FeatureSelector auto-prefix fold (the silent-drop fix)
# ---------------------------------------------------------------------------
def test_selector_keeps_extension_prefixed_features():
    from src.utils.prediction_utils import FeatureSelector

    sel = FeatureSelector()
    assert "ADVTRACK_" in sel.SAFE_PREFIXES
    # Historical prefixes preserved.
    assert "ROLL_" in sel.SAFE_PREFIXES


def test_selector_warns_on_unsafe_numeric_columns(caplog):
    from src.utils.prediction_utils import FeatureSelector

    df = pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "PTS": [10, 20],
        "WEIRD_UNDECLARED_FEATURE": [0.1, 0.2],  # matches no safe rule
    })
    sel = FeatureSelector()
    with caplog.at_level("WARNING"):
        sel.fit(df)
    assert any("dropped" in rec.message and "leakage-safe" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# FeatureEngineer integration
# ---------------------------------------------------------------------------
def test_feature_engineer_includes_extension_group():
    from src.preprocessing.feature_engineer import FeatureEngineer

    fe = FeatureEngineer(enable_groups=["rolling", "advanced_tracking"])
    names = [g.name for g in fe.feature_groups]
    assert "advanced_tracking" in names
    assert "rolling" in names
    # Order preserved.
    assert names[0] == "rolling"


def test_feature_engineer_disable_groups_skips_extension():
    from src.preprocessing.feature_engineer import FeatureEngineer

    fe = FeatureEngineer(
        enable_groups=None,
        disable_groups=["advanced_tracking"],
    )
    names = [g.name for g in fe.feature_groups]
    assert "advanced_tracking" in names  # registered but...
    assert not fe._should_run_group(
        next(g for g in fe.feature_groups if g.name == "advanced_tracking")
    )


# ---------------------------------------------------------------------------
# ScraperRegistry
# ---------------------------------------------------------------------------
def test_scraper_registry_discovers_example():
    registry = ScraperRegistry()
    assert "advanced_tracking" in registry.all_names()


def test_scraper_registry_enable_gating():
    registry = ScraperRegistry()
    enabled = registry.build_enabled({"advanced_tracking": {"enabled": True}})
    assert [n for n, _ in enabled] == ["advanced_tracking"]

    disabled = registry.build_enabled({"advanced_tracking": {"enabled": False}})
    assert disabled == []

    none_cfg = registry.build_enabled(None)
    assert none_cfg == []


def test_scraper_registry_collision_guard():
    class CollidingScraper(BaseScraper):
        @property
        def name(self): return "colliding"

        def output_files(self): return ["nba_players.csv"]

        def fetch(self, config=None): return {}

    registry = ScraperRegistry()
    ok, collisions = registry.validate_outputs(CollidingScraper())
    assert not ok
    assert "nba_players.csv" in collisions


def test_core_protected_files_defined():
    assert "nba_players.csv" in CORE_PROTECTED_FILES
    assert "nba_games.csv" in CORE_PROTECTED_FILES


# ---------------------------------------------------------------------------
# presets "all" sentinel (drift fix)
# ---------------------------------------------------------------------------
def test_preset_all_sentinel_resolves_to_registry():
    from src.training.presets import resolve_training_preset

    preset = resolve_training_preset(
        "full",
        {"full": {"feature_engineer": {"enable_groups": ["all"]}}},
    )
    assert "rolling" in preset.enable_groups
    assert "advanced_tracking" in preset.enable_groups
    # The previously-missing lifecycle groups are present.
    assert "injury_risk" in preset.enable_groups


def test_preset_all_feature_groups_no_drift():
    from src.training.presets import ALL_FEATURE_GROUPS

    for g in ("injury_risk", "aging_curve", "kan_aging", "skill_development"):
        assert g in ALL_FEATURE_GROUPS


# ---------------------------------------------------------------------------
# End-to-end: example extension produces AND keeps features
# ---------------------------------------------------------------------------
@pytest.fixture
def core_game_log():
    rows = []
    for pid in [1, 2, 3]:
        for i in range(25):
            rows.append({
                "PLAYER_ID": pid,
                "GAME_ID": 1000 + i,
                "GAME_DATE": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "TEAM_ID": 10 + pid,
                "OPPONENT_ID": 99,
                "MIN": 30.0,
                "PTS": 20,
                "REB": 5,
                "AST": 4,
                "STL": 1,
                "BLK": 0,
                "TOV": 2,
                "FGA": 10,
                "FGM": 5,
                "FTA": 4,
                "FTM": 3,
                "FG3A": 6,
                "FG3M": 2,
                "OREB": 1,
                "DREB": 4,
                "WL": "W",
                "SEASON_ID": "2023-24",
                "PLUS_MINUS": 5,
                "VIDEO_AVAILABLE": 1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tracking_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 75
    df = pd.DataFrame({
        "PLAYER_ID": [1, 2, 3] * 25,
        "GAME_DATE": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(25)] * 3,
        "AVG_SPEED_MPS": rng.uniform(3.6, 4.8, n),
        "DIST_MILES": rng.uniform(1.2, 1.6, n),
        "TOUCHES": rng.uniform(30, 90, n),
        "ELBOW_TOUCHES": rng.uniform(5, 15, n),
        "POST_TOUCHES": rng.uniform(2, 8, n),
    })
    df.to_csv(os.path.join(str(tmp_path), "advanced_tracking.csv"), index=False)
    return tmp_path


def test_example_extension_end_to_end(core_game_log, tracking_data):
    from src.preprocessing.feature_engineer import FeatureEngineer
    from src.preprocessing.features.extensions.advanced_tracking_features import (
        OUTPUT_COLUMNS,
    )
    from src.utils.prediction_utils import FeatureSelector

    fe = FeatureEngineer(enable_groups=["rolling", "advanced_tracking"])
    for g in fe.feature_groups:
        if g.name == "advanced_tracking":
            g.data_dir = str(tracking_data)

    out = fe.create_features(core_game_log, is_training=False)
    for col in OUTPUT_COLUMNS:
        assert col in out.columns, f"{col} not produced"

    # The selector keeps the extension features (no silent drop).
    sel = FeatureSelector()
    schema = sel.fit(out)
    kept = [c for c in schema.feature_cols if c.startswith("ADVTRACK_")]
    assert set(kept) == set(OUTPUT_COLUMNS)


def test_example_extension_degrades_without_data(core_game_log, tmp_path):
    from src.preprocessing.feature_engineer import FeatureEngineer
    from src.preprocessing.features.extensions.advanced_tracking_features import (
        OUTPUT_COLUMNS,
    )

    fe = FeatureEngineer(enable_groups=["rolling", "advanced_tracking"])
    for g in fe.feature_groups:
        if g.name == "advanced_tracking":
            g.data_dir = str(tmp_path)  # no advanced_tracking.csv present

    out = fe.create_features(core_game_log, is_training=False)
    for col in OUTPUT_COLUMNS:
        assert col in out.columns
        # Zero-filled fallback, not NaN.
        assert out[col].isna().sum() == 0


def test_example_extension_is_leakage_safe(core_game_log, tracking_data):
    from src.preprocessing.feature_engineer import FeatureEngineer

    fe = FeatureEngineer(enable_groups=["advanced_tracking"])
    for g in fe.feature_groups:
        if g.name == "advanced_tracking":
            g.data_dir = str(tracking_data)

    out = fe.create_features(core_game_log, is_training=False)
    # The first game per player has no history -> rolling features are 0
    # (shift(1) ensures the current game's tracking value never leaks in).
    first_rows = out.groupby("PLAYER_ID").first().reset_index()
    assert (first_rows["ADVTRACK_TOUCHES_10"] == 0.0).all()