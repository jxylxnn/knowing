"""Training preset definitions and helpers.

This module keeps the feature-stack/preset logic separate from the CLI so the
same preset semantics can be reused by tests and future config loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


CANONICAL_TARGETS: Tuple[str, ...] = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
ALL_FEATURE_GROUPS: Tuple[str, ...] = (
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
    "season_phase",
    "team_motivation",
    "postseason_context",
)


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
        enable_groups=_coerce_sequence(
            feature_engineer.get("enable_groups", base.enable_groups),
            item_type=str,
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

    The training data already carries ``SEASON_ID`` in the current loader path.
    If that field is missing, this helper returns the input unchanged rather than
    inventing a brittle date-based heuristic.
    """
    if df is None or df.empty or recent_seasons is None:
        return df
    if recent_seasons <= 0:
        raise ValueError("recent_seasons must be positive when provided")
    if season_column not in df.columns:
        return df.copy()

    ordered = df
    if date_column in df.columns:
        ordered = df.sort_values(date_column, kind="mergesort")

    season_series = ordered[season_column].astype(str)
    unique_seasons = list(dict.fromkeys(season_series.tolist()))
    if len(unique_seasons) <= recent_seasons:
        return df.copy()

    keep = set(unique_seasons[-recent_seasons:])
    filtered = df[df[season_column].astype(str).isin(keep)].copy()
    if date_column in filtered.columns:
        filtered = filtered.sort_values(date_column, kind="mergesort")
    return filtered
