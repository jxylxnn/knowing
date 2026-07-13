"""Feature-group registry: discovery + registry-derived leak-safe prefixes.

This module is the backbone of the data-extension mechanism. It lets a new
data source + feature group be added by dropping a single self-contained
module into ``src/preprocessing/features/extensions/`` *without* editing the
four historical sync points (the group module, ``features/__init__.py``,
``FeatureEngineer._build_groups``, and ``config``/``presets`` enable-lists).

What the registry provides
--------------------------
* ``discover_extension_groups()`` — import every ``*.py`` in the extensions
  directory and collect concrete ``FeatureGroup`` subclasses found in module
  scope. Import failures are isolated and logged so one broken extension can
  never break the core pipeline.
* ``FeatureGroupRegistry`` — a single source of truth for the set of feature
  group names, the factory that instantiates built-in groups (preserving the
  historical column order so existing model contracts stay stable), and the
  leak-safe prefixes/keywords each group emits. The ``FeatureSelector`` pulls
  its safe-prefix set from the registry so an extension group's features are
  never silently dropped.
* ``ALL_FEATURE_GROUPS`` — the canonical, registry-derived tuple of every
  group name (built-ins + enabled extensions). ``presets.py`` and config can
  reference this instead of hardcoding lists, which eliminates the drift bug
  where ``presets.ALL_FEATURE_GROUPS`` previously omitted four lifecycle
  groups that ``config/default.yaml`` enabled.

Backward compatibility
-----------------------
With no extensions present and no config changes, behaviour is identical to
before: the registry returns exactly the built-in groups in the same order,
and the selector's safe-prefix set is the union of the historical
``SAFE_PREFIXES``/``SAFE_KEYWORDS`` plus anything extensions declare.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from src.preprocessing.features.base import FeatureGroup

logger = logging.getLogger(__name__)

# A factory that returns the list of built-in feature group instances.
BuiltinFactory = Callable[[], List[FeatureGroup]]

# Directory scanned for plug-in feature groups. Resolved relative to this
# file so it works regardless of the caller's working directory.
FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSIONS_DIR = os.path.join(FEATURES_DIR, "extensions")


# ---------------------------------------------------------------------------
# Leak-safe prefix/keyword metadata for the built-in groups.
#
# Each entry maps a built-in group's ``name`` to the prefixes and keywords
# its emitted feature columns use. This mirrors the historical
# ``FeatureSelector.SAFE_PREFIXES`` / ``SAFE_KEYWORDS`` tables but is keyed by
# group so the registry can assemble the full safe set without a feature
# author having to edit the selector. Extension groups declare their own
# prefixes via the ``FeatureGroup.feature_prefixes`` / ``feature_keywords``
# class attributes (see ``base.py``).
# ---------------------------------------------------------------------------
BUILTIN_SAFE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "rolling": ("ROLL_", "EWMA_"),
    "efficiency": ("EFF_Z_SCORE",),
    "momentum": (),
    "context": (),
    "fatigue": (),
    "minutes_confidence": ("MIN_CONF_",),
    "rest_density": ("DAYS_SINCE_", "GAMES_WITH_"),
    "matchup": ("VS_OPP_",),
    "opponent_strength": ("VS_OPP_",),
    "pace": ("TEAM_PACE_", "PACE_FACTOR", "EST_POSS"),
    "team_role": ("ROLE_INDEX",),
    "lineup_stability": ("LINEUP_",),
    "injury_opportunity": ("INJURY_OPP_",),
    "teammate_usage": ("TEAMMATE_",),
    "recency_form": ("RECENCY_", "IS_RECENT_"),
    "archetype": ("ARCHETYPE_", "SIMILARITY_TO_"),
    "defense_position": ("DEF_POS_", "DEF_MATCHUP", "OPP_DEF"),
    # ``_TE`` is intentionally not a broad safe prefix/keyword. The selector
    # admits only the exact target-encoding output shape.
    "target_encoding": ("_SHARE",),
    "league_rank": ("LEAGUE_PCT_", "TEAM_CUMULATIVE_"),
    "injury_risk": ("INJURY_RISK_",),
    "aging_curve": (),
    "kan_aging": (),
    "skill_development": ("POTENTIAL",),
    "season_phase": ("IS_SEASON_", "SCHED_"),
    "team_motivation": ("IS_LATE_", "IS_TANKING_"),
    "postseason_context": ("IS_PLAYOFF_", "PLAYOFF_PACE_"),
}

# Keywords that are shared across groups (kept as a flat supplemental set so
# the selector keeps its historical keyword behaviour even for built-ins that
# did not declare per-group keywords above).
BUILTIN_SAFE_KEYWORDS: Tuple[str, ...] = (
    "TREND",
    "BAYESIAN",
    "PACE",
    "FATIGUE",
    "HOT_STREAK",
    "COLD_STREAK",
    "B2B_IMPACT",
    "USG_PCT",
    "REB_OPPORTUNITY",
    "FT_RATE",
    "TS_PCT_MOMENTUM",
    "COLD_START",
    "MISSING_",
    "IMPUTED_",
)


def _is_concrete_feature_group(obj: object) -> bool:
    return (
        isinstance(obj, type)
        and issubclass(obj, FeatureGroup)
        and obj is not FeatureGroup
        and not getattr(obj, "_is_registry_base", False)
    )


def discover_extension_groups(
    extensions_dir: Optional[str] = None,
    *,
    package_name: Optional[str] = None,
) -> List[Type[FeatureGroup]]:
    """Import every extension module and return concrete FeatureGroup classes.

    Each module in the extensions directory is imported once. Any
    ``FeatureGroup`` subclass defined at module scope is collected. Import
    errors for a single module are caught and logged so they cannot break the
    rest of the pipeline — a broken extension is simply skipped.

    Args:
        extensions_dir: Directory to scan. Defaults to the bundled
            ``src/preprocessing/features/extensions``.
        package_name: Dotted package path corresponding to ``extensions_dir``.
            Defaults to ``src.preprocessing.features.extensions``. Accepting
            this explicitly lets the discoverer be pointed at an arbitrary
            on-disk package (used by tests and future alternate locations).
    """
    scan_dir = extensions_dir or EXTENSIONS_DIR
    pkg_path = package_name or "src.preprocessing.features.extensions"
    discovered: List[Type[FeatureGroup]] = []
    if not os.path.isdir(scan_dir):
        return discovered

    try:
        pkg = importlib.import_module(pkg_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not import extensions package %s: %s", pkg_path, exc)
        return discovered

    # ``iter_modules`` needs an iterable __path__; fall back to scanning the
    # directory directly when the imported object has no package path (e.g. a
    # namespace package resolved from an absolute dir in tests).
    module_iter: Iterable
    if hasattr(pkg, "__path__"):
        module_iter = pkgutil.iter_modules(pkg.__path__)
    else:  # pragma: no cover - defensive
        module_iter = [
            type("M", (), {"name": os.path.splitext(f)[0]})
            for f in os.listdir(scan_dir)
            if f.endswith(".py") and not f.startswith("_")
        ]

    for mod_info in module_iter:
        mod_name = f"{pkg_path}.{mod_info.name}"
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:
            logger.warning("Skipping feature extension %s (import failed): %s", mod_name, exc)
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if _is_concrete_feature_group(obj):
                discovered.append(obj)  # type: ignore[arg-type]

    return discovered


class FeatureGroupRegistry:
    """Single source of truth for available feature groups.

    Built-in groups are constructed via the provided factory so the historical
    instantiation arguments and ordering are preserved exactly. Extension
    groups are discovered on demand and instantiated with no arguments (they
    may override ``__init__`` to read config/external files themselves).
    """

    def __init__(
        self,
        builtin_factory: Optional[BuiltinFactory] = None,
        extensions_dir: Optional[str] = None,
    ) -> None:
        self._builtin_factory = builtin_factory
        self._extensions_dir = extensions_dir
        self._extensions: Optional[List[Type[FeatureGroup]]] = None

    # -- discovery -------------------------------------------------------
    def _load_extensions(self) -> List[Type[FeatureGroup]]:
        if self._extensions is None:
            self._extensions = discover_extension_groups(self._extensions_dir)
        return self._extensions

    def extension_group_names(self) -> List[str]:
        names: List[str] = []
        for cls in self._load_extensions():
            try:
                names.append(cls().name)
            except Exception as exc:
                logger.warning("Could not read name from extension %s: %s", cls.__name__, exc)
        return names

    # -- canonical name set ---------------------------------------------
    def builtin_group_names(self) -> List[str]:
        return [g.name for g in (self._builtin_factory() if self._builtin_factory else [])]

    def all_group_names(self) -> List[str]:
        """Every group name: built-ins first (stable order), then extensions."""
        names = list(self.builtin_group_names())
        for ext in self.extension_group_names():
            if ext not in names:
                names.append(ext)
        return names

    # -- instantiation ---------------------------------------------------
    def build_groups(
        self,
        *,
        include_extensions: bool = True,
    ) -> List[FeatureGroup]:
        """Instantiate built-in groups (via the factory) then extension groups.

        Built-in order is preserved for contract stability; extensions are
        appended in discovery order. Duplicate names are skipped so an
        extension cannot shadow a built-in.
        """
        groups: List[FeatureGroup] = []
        seen: set = set()
        builtins = self._builtin_factory() if self._builtin_factory else []
        for g in builtins:
            if g.name not in seen:
                groups.append(g)
                seen.add(g.name)

        if include_extensions:
            for cls in self._load_extensions():
                try:
                    inst = cls()
                except Exception as exc:
                    logger.warning("Could not instantiate extension %s: %s", cls.__name__, exc)
                    continue
                if inst.name in seen:
                    logger.warning(
                        "Extension %s duplicates group name '%s'; skipping.",
                        cls.__name__,
                        inst.name,
                    )
                    continue
                groups.append(inst)
                seen.add(inst.name)
        return groups

    # -- leak-safe metadata ---------------------------------------------
    def safe_prefixes(self) -> Tuple[str, ...]:
        """All leak-safe column prefixes: built-in map + extension declarations."""
        prefixes: List[str] = []
        for name in self.builtin_group_names():
            prefixes.extend(BUILTIN_SAFE_PREFIXES.get(name, ()))
        for cls in self._load_extensions():
            for p in getattr(cls, "feature_prefixes", ()) or ():
                prefixes.append(p)
        return tuple(dict.fromkeys(prefixes))  # dedupe, keep order

    def safe_keywords(self) -> Tuple[str, ...]:
        keywords: List[str] = list(BUILTIN_SAFE_KEYWORDS)
        for cls in self._load_extensions():
            for k in getattr(cls, "feature_keywords", ()) or ():
                keywords.append(k)
        return tuple(dict.fromkeys(keywords))


# Module-level singleton used by the rest of the codebase. The builtin factory
# is wired lazily to avoid an import cycle (FeatureEngineer imports this
# module, and the factory imports FeatureEngineer's group classes via
# features/__init__).
_default_registry: Optional[FeatureGroupRegistry] = None


def _default_builtin_factory() -> List[FeatureGroup]:
    """Construct the built-in feature groups in their canonical order.

    Mirrors the historical ``FeatureEngineer._build_groups`` list so existing
    feature-column ordering — and therefore existing model contracts — is
    preserved exactly when no extensions are present.
    """
    from src.preprocessing.features import (  # noqa: WPS235 - many imports
        AgingCurveFeatureGroup,
        ContextualFeatureGroup,
        DefensePositionFeatureGroup,
        EfficiencyFeatureGroup,
        FatigueFeatureGroup,
        InjuryAdjustedOpportunityFeatureGroup,
        InjuryRiskFeatureGroup,
        KANAgingFeatureGroup,
        LeagueRankingFeatureGroup,
        LineupStabilityFeatureGroup,
        MatchupFeatureGroup,
        MinutesConfidenceFeatureGroup,
        MomentumFeatureGroup,
        OpponentStrengthFeatureGroup,
        PaceFeatureGroup,
        PlayerArchetypeFeatureGroup,
        PostseasonContextFeatureGroup,
        RecencyFormFeatureGroup,
        RestGameDensityFeatureGroup,
        RollingFeatureGroup,
        SeasonPhaseFeatureGroup,
        SkillDevelopmentFeatureGroup,
        TargetEncodingFeatureGroup,
        TeamMotivationFeatureGroup,
        TeamRoleFeatureGroup,
        TeammateUsageFeatureGroup,
    )

    target_cols = ["PTS", "REB", "AST"]
    return [
        RollingFeatureGroup(windows=[3, 5, 10, 20, 50], target_cols=target_cols),
        EfficiencyFeatureGroup(windows=[5, 10, 20]),
        MomentumFeatureGroup(target_cols=target_cols),
        ContextualFeatureGroup(),
        FatigueFeatureGroup(),
        MinutesConfidenceFeatureGroup(target_cols=target_cols),
        RestGameDensityFeatureGroup(),
        MatchupFeatureGroup(target_cols=target_cols, recent_window=5),
        OpponentStrengthFeatureGroup(target_cols=target_cols),
        PaceFeatureGroup(),
        TeamRoleFeatureGroup(),
        LineupStabilityFeatureGroup(),
        InjuryAdjustedOpportunityFeatureGroup(),
        TeammateUsageFeatureGroup(),
        RecencyFormFeatureGroup(target_cols=target_cols),
        PlayerArchetypeFeatureGroup(),
        DefensePositionFeatureGroup(),
        TargetEncodingFeatureGroup(target_cols=target_cols, smoothing=20),
        LeagueRankingFeatureGroup(target_cols=target_cols, window=2000, min_periods=500),
        InjuryRiskFeatureGroup(),
        AgingCurveFeatureGroup(),
        KANAgingFeatureGroup(),
        SkillDevelopmentFeatureGroup(),
        SeasonPhaseFeatureGroup(),
        TeamMotivationFeatureGroup(),
        PostseasonContextFeatureGroup(),
    ]


def get_registry() -> FeatureGroupRegistry:
    """Return the process-wide default registry (built lazily)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = FeatureGroupRegistry(
            builtin_factory=_default_builtin_factory,
        )
    return _default_registry


def all_feature_groups() -> Tuple[str, ...]:
    """Canonical tuple of every available feature-group name.

    Used by ``presets.py`` and config resolution so the ``full`` preset and
    any ``enable_groups: "all"`` sentinel resolve to the registry truth
    rather than a hand-maintained list that can drift out of sync.
    """
    return tuple(get_registry().all_group_names())
