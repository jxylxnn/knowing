"""Training preset definitions and helpers.

This module keeps the feature-stack/preset logic separate from the CLI so the
same preset semantics can be reused by tests and future config loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


CANONICAL_TARGETS: Tuple[str, ...] = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


def _registry_group_names() -> Tuple[str, ...]:
    """Return the registry-derived canonical group-name set.

    Imported lazily to avoid a circular import (the registry's builtin factory
    imports the feature-group classes via ``features/__init__``, which does not
    import presets, so this is safe at call time). Falls back to a static list
    only if the registry is unavailable during very early bootstrap.
    """
    try:
        from src.preprocessing.features.registry import all_feature_groups

        names = all_feature_groups()
        if names:
            return names
    except Exception:  # pragma: no cover - defensive bootstrap path
        pass
    return _FALLBACK_FEATURE_GROUPS


# Static fallback kept in sync with the built-in groups so preset resolution
# still works if the registry cannot be imported for any reason.
_FALLBACK_FEATURE_GROUPS: Tuple[str, ...] = (
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
    "lineup_stability",
    "injury_opportunity",
    "teammate_usage",
    "recency_form",
    "archetype",
    "defense_position",
    "target_encoding",
    "league_rank",
    "injury_risk",
    "aging_curve",
    "kan_aging",
    "skill_development",
    "season_phase",
    "team_motivation",
    "postseason_context",
)

# Public alias: the canonical set of feature-group names. Derived from the
# registry so it can never drift out of sync with the groups that are
# actually registered (the original hardcoded tuple was missing four
# lifecycle groups that config/default.yaml enabled).
ALL_FEATURE_GROUPS: Tuple[str, ...] = _registry_group_names()


@dataclass(frozen=True)
class TrainingPreset:
    """Resolved training preset used by the CLI."""

    name: str
    description: str
    default_mode: str
    default_model_size: str
    transformer_enabled: bool
    recent_seasons: Optional[int]
    rolling_windows: Tuple[int, ...]
    enable_groups: Tuple[str, ...]
    disable_groups: Tuple[str, ...] = ()
    targets: Tuple[str, ...] = CANONICAL_TARGETS
    feature_selection: Optional[Dict[str, Any]] = None
    feature_selection_profile: Optional[str] = None

    def feature_engineer_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for build_feature_engineer(...)."""
        return {
            "rolling_windows": list(self.rolling_windows),
            "enable_groups": list(self.enable_groups),
            "disable_groups": list(self.disable_groups),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable representation for logging/debugging."""
        return {
            "name": self.name,
            "description": self.description,
            "default_mode": self.default_mode,
            "default_model_size": self.default_model_size,
            "transformer_enabled": self.transformer_enabled,
            "recent_seasons": self.recent_seasons,
            "rolling_windows": list(self.rolling_windows),
            "enable_groups": list(self.enable_groups),
            "disable_groups": list(self.disable_groups),
            "targets": list(self.targets),
            "feature_selection": dict(self.feature_selection) if self.feature_selection else None,
            "feature_selection_profile": self.feature_selection_profile,
        }


BUILTIN_TRAINING_PRESETS: Dict[str, TrainingPreset] = {
    "baseline": TrainingPreset(
        name="baseline",
        description=(
            "Model-v2 CatBoost champion candidate with only the priority, "
            "point-in-time feature families and no neural default blend."
        ),
        default_mode="standard",
        default_model_size="S",
        transformer_enabled=False,
        recent_seasons=4,
        rolling_windows=(3, 5, 10, 20),
        enable_groups=(
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
            "lineup_stability",
            "injury_opportunity",
            "teammate_usage",
            "recency_form",
            "defense_position",
        ),
    ),
    "full": TrainingPreset(
        name="full",
        description="Full CatBoost + Transformer stack with the complete feature set.",
        default_mode="standard",
        default_model_size="M",
        transformer_enabled=True,
        recent_seasons=None,
        rolling_windows=(3, 5, 10, 20, 50),
        enable_groups=ALL_FEATURE_GROUPS,
    ),
    "small": TrainingPreset(
        name="small",
        description=(
            "Fast CatBoost-first preset with a reduced feature set and no Transformer."
        ),
        default_mode="quick",
        default_model_size="S",
        transformer_enabled=False,
        recent_seasons=2,
        rolling_windows=(3, 5, 10, 20),
        enable_groups=(
            "rolling",
            "efficiency",
            "momentum",
            "pace",
            "opponent_strength",
            "archetype",
        ),
    ),
    "laptop_quality": TrainingPreset(
        name="laptop_quality",
        description=(
            "Laptop-friendly quality preset between small and full training: "
            "CatBoost-first with smart feature selection, no Transformer, and a "
            "mid-size feature-group set."
        ),
        default_mode="standard",
        default_model_size="S",
        transformer_enabled=False,
        recent_seasons=3,
        rolling_windows=(3, 5, 10, 20),
        enable_groups=(
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
        ),
        feature_selection={"enabled": True, "profile": "balanced"},
        feature_selection_profile="balanced",
    ),
}


def _coerce_sequence(values: Optional[Iterable[Any]], *, item_type: type) -> Tuple[Any, ...]:
    if values is None:
        return ()
    coerced: List[Any] = []
    for value in values:
        if item_type is int:
            coerced.append(int(value))
        else:
            coerced.append(str(value))
    return tuple(coerced)


def _resolve_enable_groups(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Resolve an enable_groups list, expanding the ``"all"`` sentinel.

    ``enable_groups: ["all"]`` resolves to the registry's canonical set of
    every available feature-group name (built-ins + enabled extensions), so a
    preset never has to hand-maintain the full list and cannot drift out of
    sync as new groups are registered.
    """
    if values is None:
        return ()
    items = [str(v) for v in values]
    if "all" in items:
        # Registry truth: every group including extensions.
        return _registry_group_names()
    return tuple(items)


def _merge_preset_definition(
    base: TrainingPreset,
    override: Optional[Dict[str, Any]],
) -> TrainingPreset:
    if not override:
        return base

    feature_engineer = dict(override.get("feature_engineer", {}))
    if "rolling_windows" in override:
        feature_engineer["rolling_windows"] = override["rolling_windows"]
    if "enable_groups" in override:
        feature_engineer["enable_groups"] = override["enable_groups"]
    if "disable_groups" in override:
        feature_engineer["disable_groups"] = override["disable_groups"]

    merged = replace(
        base,
        description=str(override.get("description", base.description)),
        default_mode=str(override.get("default_mode", base.default_mode)),
        default_model_size=str(
            override.get("default_model_size", base.default_model_size)
        ),
        transformer_enabled=bool(
            override.get("transformer_enabled", base.transformer_enabled)
        ),
        recent_seasons=override.get("recent_seasons", base.recent_seasons),
        rolling_windows=_coerce_sequence(
            feature_engineer.get("rolling_windows", base.rolling_windows),
            item_type=int,
        ),
        enable_groups=_resolve_enable_groups(
            feature_engineer.get("enable_groups", base.enable_groups),
        ),
        disable_groups=_coerce_sequence(
            feature_engineer.get("disable_groups", base.disable_groups),
            item_type=str,
        ),
        targets=_coerce_sequence(override.get("targets", base.targets), item_type=str)
        or base.targets,
        feature_selection=override.get("feature_selection", base.feature_selection),
        feature_selection_profile=override.get(
            "feature_selection_profile", base.feature_selection_profile
        ),
    )
    return merged


def resolve_training_preset(
    preset_name: str,
    preset_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TrainingPreset:
    """Resolve a named preset, optionally applying config-file overrides."""
    normalized = str(preset_name).strip().lower()
    if normalized not in BUILTIN_TRAINING_PRESETS:
        raise ValueError(
            f"Unsupported training preset '{preset_name}'. "
            f"Expected one of {sorted(BUILTIN_TRAINING_PRESETS)}."
        )

    preset = BUILTIN_TRAINING_PRESETS[normalized]
    override = (preset_overrides or {}).get(normalized)
    return _merge_preset_definition(preset, override)


def apply_recent_history_window(
    df: pd.DataFrame,
    recent_seasons: Optional[int],
    *,
    season_column: str = "SEASON_ID",
    date_column: str = "GAME_DATE",
) -> pd.DataFrame:
    """Keep only the most recent ``recent_seasons`` seasons when possible.

    ``SEASON_ID`` is the canonical internal name, while NBA exports commonly
    provide the equivalent ``SEASON_YEAR`` (for example, ``"2024-25"``).
    Accept either representation so preset history limits remain effective
    across both data formats. If neither is available, return the input
    unchanged rather than inventing a brittle date-based heuristic.
    """
    if df is None or df.empty or recent_seasons is None:
        return df
    if recent_seasons <= 0:
        raise ValueError("recent_seasons must be positive when provided")
    resolved_season_column = season_column
    if resolved_season_column not in df.columns:
        if season_column == "SEASON_ID" and "SEASON_YEAR" in df.columns:
            resolved_season_column = "SEASON_YEAR"
        else:
            return df.copy()

    ordered = df
    if date_column in df.columns:
        ordered = df.sort_values(date_column, kind="mergesort")

    season_series = ordered[resolved_season_column].astype(str)
    unique_seasons = list(dict.fromkeys(season_series.tolist()))
    if len(unique_seasons) <= recent_seasons:
        return df.copy()

    keep = set(unique_seasons[-recent_seasons:])
    filtered = df[df[resolved_season_column].astype(str).isin(keep)].copy()
    if date_column in filtered.columns:
        filtered = filtered.sort_values(date_column, kind="mergesort")
    return filtered
