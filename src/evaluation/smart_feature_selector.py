"""Smart per-target feature selector.

Combines four signals into a per-target feature score and keeps different
feature lists per stat.  The signal mix follows the ticket spec:

    final_score = (
          0.40 * backtest_gain
        + 0.25 * stability_score
        + 0.20 * catboost_importance
        + 0.10 * permutation_importance
        - 0.05 * missingness_penalty
    )

The selector consumes:

* an :class:`AblationReport` (from :mod:`feature_group_ablation`) for the
  backtest_gain signal (per group, broadcast to the columns that live in
  that group);
* a :class:`ShadowFilterResult` (from :mod:`shadow_feature_filter`) to
  prune features that score below the random-control noise floor;
* a quick ``HistGradientBoostingRegressor`` fit to estimate CatBoost-style
  gain importances without spinning up the full CatBoost trainer;
* a permutation importance pass on the validation split;
* a simple stability score computed by comparing the importances of two
  temporal sub-splits (first half vs. second half of training data);
* a missingness penalty proportional to the share of NaN rows in the
  training frame.

The selected feature lists are written to
``models/feature_selection_manifest.json`` so training and inference
share the same contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation.feature_group_ablation import (
    AblationReport,
    DEFAULT_TARGETS,
    FeatureGroupAblator,
    GroupScore,
    filter_group_columns,
)
from src.evaluation.shadow_feature_filter import (
    SHADOW_COLUMNS,
    ShadowFeatureFilter,
    ShadowFilterResult,
)

logger = logging.getLogger(__name__)


# Signal weights — keep in sync with the ticket.
WEIGHTS: Dict[str, float] = {
    "backtest_gain": 0.40,
    "stability": 0.25,
    "catboost_importance": 0.20,
    "permutation_importance": 0.10,
    "missingness_penalty": 0.05,
}


@dataclass
class ProfileConfig:
    """Resolved view of the ``feature_selection_profiles`` block."""

    name: str
    run_group_ablation: bool = True
    run_individual_pruning: bool = True
    run_shadow_filter: bool = True
    run_time_stability_check: bool = False
    description: str = ""

    @classmethod
    def resolve(
        cls,
        profile_name: str,
        profiles_cfg: Optional[Mapping[str, Mapping[str, Any]]],
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> "ProfileConfig":
        cfg = dict(defaults or {})
        if profiles_cfg and profile_name in profiles_cfg:
            cfg.update(profiles_cfg[profile_name])
        return cls(
            name=profile_name,
            run_group_ablation=bool(cfg.get("run_group_ablation", True)),
            run_individual_pruning=bool(cfg.get("run_individual_pruning", True)),
            run_shadow_filter=bool(cfg.get("run_shadow_filter", True)),
            run_time_stability_check=bool(cfg.get("run_time_stability_check", False)),
            description=str(cfg.get("description", "")),
        )


@dataclass
class SelectorConfig:
    """Resolved view of the ``feature_selection`` config block."""

    enabled: bool = False
    mode: str = "smart"
    profile: str = "balanced"
    min_backtest_gain: float = 0.0
    use_shadow_features: bool = True
    target_specific: bool = True
    output_path: str = "models/feature_selection_manifest.json"
    random_state: int = 42

    @classmethod
    def from_config(
        cls,
        feature_selection_cfg: Optional[Mapping[str, Any]] = None,
    ) -> "SelectorConfig":
        cfg = dict(feature_selection_cfg or {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            mode=str(cfg.get("mode", "smart")),
            profile=str(cfg.get("profile", "balanced")),
            min_backtest_gain=float(cfg.get("min_backtest_gain", 0.0)),
            use_shadow_features=bool(cfg.get("use_shadow_features", True)),
            target_specific=bool(cfg.get("target_specific", True)),
            output_path=str(cfg.get("output_path", "models/feature_selection_manifest.json")),
            random_state=int(cfg.get("random_state", 42)),
        )


@dataclass
class TargetSelection:
    """Selection record for a single target stat."""

    target: str
    selected_features: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    dropped_features: List[str] = field(default_factory=list)
    shadow_dropped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "selected_features": list(self.selected_features),
            "dropped_features": list(self.dropped_features),
            "shadow_dropped": list(self.shadow_dropped),
            "scores": {k: float(v) for k, v in self.scores.items()},
        }


@dataclass
class SelectionManifest:
    """Top-level manifest saved to disk."""

    enabled: bool
    profile: str
    target_specific: bool
    targets: List[str]
    selected_features_by_target: Dict[str, List[str]]
    selected_features_global: List[str]
    dropped_feature_groups: List[str]
    kept_feature_groups: List[str]
    target_selections: List[TargetSelection]
    created_at: str
    output_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "profile": self.profile,
            "target_specific": bool(self.target_specific),
            "targets": list(self.targets),
            "selected_features_by_target": {
                t: list(cols) for t, cols in self.selected_features_by_target.items()
            },
            "selected_features_global": list(self.selected_features_global),
            "dropped_feature_groups": list(self.dropped_feature_groups),
            "kept_feature_groups": list(self.kept_feature_groups),
            "target_selections": [s.to_dict() for s in self.target_selections],
            "created_at": self.created_at,
            "output_path": self.output_path,
            "metadata": dict(self.metadata),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        target_path = Path(path) if path is not None else Path(self.output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return target_path

    @classmethod
    def load(cls, path: Path) -> "SelectionManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        targets = list(data.get("targets", []))
        target_selections: List[TargetSelection] = []
        for entry in data.get("target_selections", []):
            target_selections.append(
                TargetSelection(
                    target=str(entry["target"]),
                    selected_features=list(entry.get("selected_features", [])),
                    dropped_features=list(entry.get("dropped_features", [])),
                    shadow_dropped=list(entry.get("shadow_dropped", [])),
                    scores={k: float(v) for k, v in entry.get("scores", {}).items()},
                )
            )
        return cls(
            enabled=bool(data.get("enabled", False)),
            profile=str(data.get("profile", "balanced")),
            target_specific=bool(data.get("target_specific", True)),
            targets=targets,
            selected_features_by_target={
                str(t): list(cols)
                for t, cols in data.get("selected_features_by_target", {}).items()
            },
            selected_features_global=list(data.get("selected_features_global", [])),
            dropped_feature_groups=list(data.get("dropped_feature_groups", [])),
            kept_feature_groups=list(data.get("kept_feature_groups", [])),
            target_selections=target_selections,
            created_at=str(data.get("created_at", "")),
            output_path=str(data.get("output_path", "")),
            metadata=dict(data.get("metadata", {})),
        )


def _safe_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    return df[list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 1,
    random_state: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(X) == 0:
        return np.zeros(X.shape[1])
    base = model.predict(X)
    base_score = float(np.mean((base - y) ** 2))
    X_arr = X.to_numpy()
    cols = X.shape[1]
    out = np.zeros(cols, dtype=np.float64)
    for c in range(cols):
        scores = 0.0
        for _ in range(max(1, n_repeats)):
            shuffled = X_arr.copy()
            shuffled[:, c] = rng.permutation(shuffled[:, c])
            pred = model.predict(shuffled)
            scores += float(np.mean((pred - y) ** 2)) - base_score
        out[c] = scores / max(1, n_repeats)
    return out


def _gain_importance_from_model(model, feature_cols: Sequence[str]) -> np.ndarray:
    """Approximate CatBoost-style gain importance from a sklearn model.

    ``HistGradientBoostingRegressor`` doesn't expose ``feature_importances_``
    on all sklearn versions, so fall back to a uniform importance vector
    if necessary — the selector still works, just with one fewer signal.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return np.full(len(feature_cols), 1.0 / max(1, len(feature_cols)))
    return np.asarray(importances, dtype=np.float64)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores)
    out = np.zeros_like(scores, dtype=np.float64)
    out[finite] = scores[finite]
    lo, hi = float(out[finite].min()), float(out[finite].max())
    if hi - lo < 1e-9:
        out[finite] = 1.0
    else:
        out[finite] = (out[finite] - lo) / (hi - lo)
    return out


def _missingness_penalty(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if len(df) == 0:
        return np.zeros(len(feature_cols))
    penalties = []
    n = len(df)
    for col in feature_cols:
        if col not in df.columns:
            penalties.append(1.0)
            continue
        missing = float(df[col].isna().sum() + (df[col] == 0).sum())
        penalties.append(missing / max(1, n))
    return np.asarray(penalties, dtype=np.float64)


def _backtest_gain_per_feature(
    feature_cols: Sequence[str],
    group_columns: Mapping[str, Sequence[str]],
    ablation_report: Optional[AblationReport],
) -> np.ndarray:
    """Broadcast the ablation group score to per-feature importance."""
    out = np.zeros(len(feature_cols), dtype=np.float64)
    if not ablation_report or not ablation_report.group_scores:
        return out
    avg_by_group = ablation_report.average_score_by_group()
    for i, col in enumerate(feature_cols):
        groups_for_col = [g for g, cols in group_columns.items() if col in cols]
        if not groups_for_col:
            continue
        scores = [avg_by_group[g] for g in groups_for_col if g in avg_by_group]
        if scores:
            out[i] = float(np.mean(scores))
    return out


def _stability_score(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    random_state: int,
) -> np.ndarray:
    """Compare feature importances on two temporal sub-splits of training data."""
    if target not in df.columns or len(df) < 60:
        return np.zeros(len(feature_cols))
    ordered = df.sort_values("GAME_DATE", kind="mergesort") if "GAME_DATE" in df.columns else df.reset_index(drop=True)
    mid = len(ordered) // 2
    halves = (ordered.iloc[:mid], ordered.iloc[mid:])
    gains: List[np.ndarray] = []
    for sub in halves:
        if len(sub) < 30:
            continue
        y = pd.to_numeric(sub[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        X = _safe_feature_frame(sub, feature_cols)
        if X.shape[0] < 30:
            continue
        model = HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=60,
            max_depth=4,
            learning_rate=0.05,
        )
        model.fit(X, y)
        gains.append(_gain_importance_from_model(model, feature_cols))
    if len(gains) < 2:
        return np.zeros(len(feature_cols))
    a, b = gains[0], gains[1]
    if a.size != b.size or a.size == 0:
        return np.zeros(len(feature_cols))
    corr = np.zeros(a.size, dtype=np.float64)
    for i in range(a.size):
        if a[i] == 0 and b[i] == 0:
            corr[i] = 0.0
        else:
            denom = float(np.sqrt(a[i] * b[i]))
            if denom < 1e-12:
                corr[i] = 0.0
            else:
                corr[i] = max(0.0, min(1.0, float(2 * min(a[i], b[i]) / (a[i] + b[i] + 1e-12))))
    return corr


def _per_target_signals(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_columns: Mapping[str, Sequence[str]],
    target: str,
    ablation_report: Optional[AblationReport],
    random_state: int,
) -> Dict[str, np.ndarray]:
    """Compute the four scoring signals for one target."""
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    X = _safe_feature_frame(df, feature_cols)
    n_features = len(feature_cols)

    # Default: zeros if anything fails
    catboost_importance = np.zeros(n_features, dtype=np.float64)
    permutation_importance = np.zeros(n_features, dtype=np.float64)
    backtest_gain = _backtest_gain_per_feature(feature_cols, group_columns, ablation_report)
    stability = _stability_score(df, feature_cols, target, random_state)

    if len(X) >= 30 and not np.allclose(y, y[0]):
        # Temporal validation split — last 20% is the holdout.
        split_idx = max(1, int(len(X) * 0.8))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        if len(X_val) >= 5:
            try:
                model = HistGradientBoostingRegressor(
                    random_state=random_state,
                    max_iter=100,
                    max_depth=4,
                    learning_rate=0.05,
                )
                model.fit(X_train, y_train)
                catboost_importance = _gain_importance_from_model(model, feature_cols)
                permutation_importance = _permutation_importance(
                    model, X_val, y_val, n_repeats=1, random_state=random_state
                )
            except Exception as exc:
                logger.warning("Per-target signal fit failed for %s: %s", target, exc)

    missingness = _missingness_penalty(df, feature_cols)
    return {
        "backtest_gain": backtest_gain,
        "stability": stability,
        "catboost_importance": catboost_importance,
        "permutation_importance": permutation_importance,
        "missingness_penalty": missingness,
    }


def _final_score(signals: Mapping[str, np.ndarray]) -> np.ndarray:
    keys = ["backtest_gain", "stability", "catboost_importance", "permutation_importance"]
    components = []
    for key in keys:
        arr = signals.get(key, np.zeros(0))
        if arr.size == 0:
            return np.zeros(0)
        components.append(WEIGHTS[key] * _normalize_scores(arr))
    if not components:
        return np.zeros(0)
    score = sum(components)
    penalty_key = "missingness_penalty"
    if penalty_key in signals and signals[penalty_key].size == score.size:
        score = score - WEIGHTS[penalty_key] * np.clip(signals[penalty_key], 0.0, 1.0)
    return score


class SmartFeatureSelector:
    """Top-level orchestrator: group ablation + per-target scoring + manifest."""

    def __init__(
        self,
        config: Optional[SelectorConfig] = None,
        profile: Optional[ProfileConfig] = None,
    ):
        self.config = config or SelectorConfig()
        self.profile = profile or ProfileConfig(name=self.config.profile)
        self._ablation: Optional[AblationReport] = None
        self._shadow_results: Dict[str, ShadowFilterResult] = {}
        self._manifest: Optional[SelectionManifest] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ablation_report(self) -> Optional[AblationReport]:
        return self._ablation

    @property
    def manifest(self) -> Optional[SelectionManifest]:
        return self._manifest

    def run(
        self,
        full_df: pd.DataFrame,
        feature_cols: Sequence[str],
        group_columns: Mapping[str, Sequence[str]],
        targets: Sequence[str] = DEFAULT_TARGETS,
    ) -> SelectionManifest:
        """Run the full smart selection pipeline."""
        feature_cols = [c for c in feature_cols if c in full_df.columns]
        if not feature_cols:
            raise ValueError("No feature columns available for smart selection")

        targets = list(targets)
        ablation_report: Optional[AblationReport] = None
        if self.profile.run_group_ablation:
            ablator = FeatureGroupAblator(
                targets=targets,
                random_state=self.config.random_state,
            )
            try:
                ablation_report = ablator.run(
                    full_df=full_df,
                    feature_cols=feature_cols,
                    group_columns=group_columns,
                    targets=targets,
                    min_gain=self.config.min_backtest_gain,
                )
            except Exception as exc:
                logger.warning("Group ablation failed: %s", exc)
                ablation_report = None
        self._ablation = ablation_report

        if ablation_report and ablation_report.dropped_groups:
            working_groups = filter_group_columns(
                group_columns, ablation_report.selected_groups or list(group_columns.keys())
            )
        else:
            working_groups = {g: list(cols) for g, cols in group_columns.items()}

        # Restrict to columns that belong to a kept group, when group
        # ablation produced a non-empty allow-list.
        if ablation_report and ablation_report.selected_groups:
            allowed_cols = {c for cols in working_groups.values() for c in cols}
            working_features = [c for c in feature_cols if c in allowed_cols]
        else:
            working_features = list(feature_cols)

        shadow_results: Dict[str, ShadowFilterResult] = {}
        if self.profile.run_shadow_filter and self.config.use_shadow_features:
            filter_ = ShadowFeatureFilter(
                targets=targets,
                random_state=self.config.random_state,
            )
            try:
                shadow_results = filter_.run_all(full_df, working_features, targets=targets)
            except Exception as exc:
                logger.warning("Shadow filter failed: %s", exc)
                shadow_results = {}
        self._shadow_results = shadow_results

        target_selections: List[TargetSelection] = []
        per_target_features: Dict[str, List[str]] = {}
        all_selected: set[str] = set()

        for target in targets:
            if target not in full_df.columns:
                continue
            signals = _per_target_signals(
                full_df,
                working_features,
                working_groups,
                target,
                ablation_report,
                self.config.random_state,
            )
            scores = _final_score(signals)
            if scores.size != len(working_features):
                logger.debug("Skipping target %s (no signal scores)", target)
                continue

            # Shadow filter overrides the lower-tier drop decision.
            shadow = shadow_results.get(target)
            shadow_drop = set(shadow.dropped_features) if shadow else set()
            shadow_kept = set(shadow.kept_features) if shadow else None

            feature_score_map = {
                col: float(scores[i]) for i, col in enumerate(working_features)
            }
            sorted_features = sorted(
                feature_score_map.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )

            # 1) Start with the shadow-kept subset (or all if no shadow).
            if shadow_kept is not None:
                candidate_set = shadow_kept
            else:
                candidate_set = set(working_features)

            # 2) Always keep the top scoring feature for the target so the
            #    model isn't accidentally fed an empty column list.
            if sorted_features:
                candidate_set.add(sorted_features[0][0])

            # 3) Drop features that score below the median if individual
            #    pruning is enabled.
            if self.profile.run_individual_pruning and len(sorted_features) >= 4:
                finite_scores = np.array([s for _, s in sorted_features if np.isfinite(s)])
                if finite_scores.size > 0:
                    cutoff = float(np.median(finite_scores))
                else:
                    cutoff = 0.0
                below_cutoff = {f for f, s in feature_score_map.items() if s < cutoff}
                # Don't drop the top feature even if it's below the median.
                if sorted_features:
                    below_cutoff.discard(sorted_features[0][0])
                # Don't drop more than 75% of the features — keep enough
                # signal for the model to learn.
                max_drop = max(0, int(0.75 * len(feature_score_map)))
                if len(below_cutoff) > max_drop:
                    worst = sorted(below_cutoff, key=lambda f: feature_score_map[f])[:max_drop]
                    below_cutoff = set(worst)
                selected = [f for f in working_features if f in candidate_set and f not in below_cutoff]
            else:
                selected = [f for f in working_features if f in candidate_set]

            # 4) Always retain the top 3 features per target even if a
            #    different filter is asking us to drop them.
            top_keep = [f for f, _ in sorted_features[:3] if f in feature_score_map]
            selected_set = set(selected) | set(top_keep)
            selected = [f for f in working_features if f in selected_set]

            per_target_features[target] = list(selected)
            all_selected.update(selected)

            target_selections.append(
                TargetSelection(
                    target=target,
                    selected_features=list(selected),
                    dropped_features=[f for f in working_features if f not in selected_set],
                    shadow_dropped=sorted(shadow_drop),
                    scores=feature_score_map,
                )
            )

        if not self.config.target_specific:
            # Use the union of per-target selections as a single global list.
            union_features = sorted(all_selected) if all_selected else list(working_features)
            per_target_features = {t: list(union_features) for t in targets}
            global_features = list(union_features)
        else:
            global_features = sorted(all_selected) if all_selected else list(working_features)

        manifest = SelectionManifest(
            enabled=self.config.enabled,
            profile=self.profile.name,
            target_specific=self.config.target_specific,
            targets=targets,
            selected_features_by_target=per_target_features,
            selected_features_global=global_features,
            dropped_feature_groups=list(ablation_report.dropped_groups) if ablation_report else [],
            kept_feature_groups=list(ablation_report.selected_groups) if ablation_report else [],
            target_selections=target_selections,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            output_path=self.config.output_path,
            metadata={
                "weights": dict(WEIGHTS),
                "min_backtest_gain": self.config.min_backtest_gain,
                "use_shadow_features": self.config.use_shadow_features,
                "n_input_features": int(len(feature_cols)),
                "n_selected_features_global": int(len(global_features)),
                "ablation_baseline": ablation_report.baseline_metrics if ablation_report else {},
                "shadow_targets_run": sorted(self._shadow_results.keys()),
            },
        )

        try:
            manifest.save(Path(self.config.output_path))
            logger.info("Saved feature selection manifest to %s", self.config.output_path)
        except Exception as exc:
            logger.warning("Failed to persist feature selection manifest: %s", exc)

        self._manifest = manifest
        return manifest

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def features_for_target(self, target: str) -> List[str]:
        if self._manifest is None:
            return []
        if self.config.target_specific:
            return list(self._manifest.selected_features_by_target.get(target, []))
        return list(self._manifest.selected_features_global)

    def all_features(self) -> List[str]:
        if self._manifest is None:
            return []
        return list(self._manifest.selected_features_global)


def load_manifest(path: Path) -> SelectionManifest:
    """Convenience wrapper around :meth:`SelectionManifest.load`."""
    return SelectionManifest.load(path)


__all__ = [
    "ProfileConfig",
    "SelectorConfig",
    "SelectionManifest",
    "SmartFeatureSelector",
    "TargetSelection",
    "WEIGHTS",
    "load_manifest",
]
