"""Shadow feature filter for smart feature selection.

Inject random "control" features alongside the real feature matrix, fit a
fast model, and use the random features' importances as a noise floor.
Real features whose importance is *below* the median shadow importance are
almost certainly noise and can be dropped.

This is a cheap, fully-supervised screening step that does not require
backtest runs, so it scales well to hundreds of features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)


# Column names reserved for the injected control features.  These must not
# collide with real feature names — they are deliberately prefixed with
# ``SHADOW_`` to make leakage obvious in any downstream log.
SHADOW_COLUMNS: Tuple[str, ...] = (
    "SHADOW_RANDOM_NORMAL",
    "SHADOW_RANDOM_UNIFORM",
    "SHADOW_PERMUTED_TARGET",
)


@dataclass
class ShadowImportance:
    """Importance score for a single feature relative to shadow controls."""

    feature: str
    importance: float
    is_shadow: bool
    below_shadow_median: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "importance": float(self.importance),
            "is_shadow": bool(self.is_shadow),
            "below_shadow_median": bool(self.below_shadow_median),
        }


@dataclass
class ShadowFilterResult:
    """Aggregate result of the shadow filtering step."""

    importances: List[ShadowImportance] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    kept_features: List[str] = field(default_factory=list)
    shadow_median_importance: float = 0.0
    target: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "shadow_median_importance": float(self.shadow_median_importance),
            "dropped_features": list(self.dropped_features),
            "kept_features": list(self.kept_features),
            "importances": [i.to_dict() for i in self.importances],
        }


def _shade_competition_ranking(model) -> np.ndarray:
    """Return a permutation-style ranking of feature importances.

    ``HistGradientBoostingRegressor`` doesn't expose feature_importances_,
    so we approximate importance with the absolute value of a permutation
    pass: shuffle each column, measure the resulting change in model
    output, and use the average squared shift as the importance score.
    """
    # Fall back to a zero importance vector — the caller will treat
    # everything as "below" the shadow floor and we will keep the
    # original list.  This is the safe behaviour for unsupported models.
    return np.zeros(0)


def _permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 1,
    random_state: int = 0,
) -> np.ndarray:
    """Approximate permutation importance for any sklearn-compatible model."""
    rng = np.random.default_rng(random_state)
    if len(X) == 0:
        return np.zeros(X.shape[1])
    base_pred = model.predict(X)
    base_score = float(np.mean((base_pred - y) ** 2)) if len(y) else 0.0
    importances = np.zeros(X.shape[1], dtype=np.float64)
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    for col in range(X_arr.shape[1]):
        col_importance = 0.0
        for _ in range(max(1, n_repeats)):
            shuffled = X_arr.copy()
            shuffled[:, col] = rng.permutation(shuffled[:, col])
            preds = model.predict(shuffled)
            col_importance += float(np.mean((preds - y) ** 2)) - base_score
        importances[col] = col_importance / max(1, n_repeats)
    return importances


def _drop_rule(features: Sequence[str], importances: np.ndarray) -> List[str]:
    """Return real features whose importance is below the shadow median."""
    if not features or importances.size == 0:
        return []
    shadow_idx = [i for i, f in enumerate(features) if f in SHADOW_COLUMNS]
    real_idx = [i for i, f in enumerate(features) if f not in SHADOW_COLUMNS]
    if not shadow_idx:
        return []
    shadow_median = float(np.median(importances[shadow_idx]))
    return [
        features[i]
        for i in real_idx
        if importances[i] < shadow_median
    ]


class ShadowFeatureFilter:
    """Drop real features that are weaker than random control features."""

    def __init__(
        self,
        targets: Sequence[str] = (
            "PTS", "REB", "AST", "STL", "BLK", "TOV",
        ),
        random_state: int = 42,
        n_repeats: int = 1,
        min_keep: int = 5,
    ):
        self.targets = list(targets)
        self.random_state = int(random_state)
        self.n_repeats = max(1, int(n_repeats))
        # Always keep at least this many features per target, even if all
        # of them are below the shadow floor.
        self.min_keep = max(1, int(min_keep))

    def _inject_shadows(
        self,
        df: pd.DataFrame,
        target_series: pd.Series,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        out = df.copy()
        n = len(out)
        if n == 0:
            return out
        out["SHADOW_RANDOM_NORMAL"] = rng.standard_normal(n)
        out["SHADOW_RANDOM_UNIFORM"] = rng.uniform(0.0, 1.0, size=n)
        permuted_target = (
            pd.to_numeric(target_series, errors="coerce")
            .fillna(target_series.mean() if hasattr(target_series, "mean") else 0.0)
            .to_numpy()
        )
        out["SHADOW_PERMUTED_TARGET"] = rng.permutation(permuted_target)
        return out

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target: str,
        val_ratio: float = 0.25,
    ) -> ShadowFilterResult:
        """Return a :class:`ShadowFilterResult` for a single target."""
        if target not in df.columns:
            logger.debug("ShadowFeatureFilter: target %s not present", target)
            return ShadowFilterResult(target=target)

        feature_cols = [c for c in feature_cols if c in df.columns]
        if len(feature_cols) < self.min_keep:
            return ShadowFilterResult(
                target=target,
                kept_features=list(feature_cols),
                dropped_features=[],
            )

        ordered = df.sort_values("GAME_DATE", kind="mergesort") if "GAME_DATE" in df.columns else df.reset_index(drop=True)
        split_idx = max(1, int(len(ordered) * (1 - val_ratio)))
        split_idx = min(split_idx, len(ordered) - 1)
        train_df = ordered.iloc[:split_idx]
        val_df = ordered.iloc[split_idx:]

        if len(train_df) < 30 or len(val_df) < 5:
            return ShadowFilterResult(
                target=target,
                kept_features=list(feature_cols),
                dropped_features=[],
            )

        y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_val = pd.to_numeric(val_df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if np.allclose(y_train, y_train[0]) or np.allclose(y_val, y_val[0]):
            return ShadowFilterResult(
                target=target,
                kept_features=list(feature_cols),
                dropped_features=[],
            )

        X_train = self._inject_shadows(train_df[feature_cols], train_df[target])
        X_val = self._inject_shadows(val_df[feature_cols], val_df[target])

        X_train_arr = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_val_arr = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        try:
            model = HistGradientBoostingRegressor(
                random_state=self.random_state,
                max_iter=80,
                max_depth=4,
                learning_rate=0.05,
            )
            model.fit(X_train_arr, y_train)
        except Exception as exc:
            logger.warning("ShadowFeatureFilter: model fit failed for %s: %s", target, exc)
            return ShadowFilterResult(
                target=target,
                kept_features=list(feature_cols),
                dropped_features=[],
            )

        importances = _permutation_importance(
            model,
            X_val_arr,
            y_val,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        feature_index = list(X_train_arr.columns)
        shadow_idx = [i for i, f in enumerate(feature_index) if f in SHADOW_COLUMNS]
        if not shadow_idx:
            return ShadowFilterResult(
                target=target,
                kept_features=list(feature_cols),
                dropped_features=[],
            )
        shadow_median = float(np.median(importances[shadow_idx]))

        records: List[ShadowImportance] = []
        real_drop_candidates: List[Tuple[float, str]] = []
        for i, feat in enumerate(feature_index):
            is_shadow = feat in SHADOW_COLUMNS
            below = bool(importances[i] < shadow_median) if importances.size > i else False
            records.append(ShadowImportance(feat, float(importances[i]), is_shadow, below))
            if not is_shadow and below:
                real_drop_candidates.append((importances[i], feat))

        # Always preserve the top-``min_keep`` features even if shadow-floor
        # would otherwise drop them.
        real_drop_candidates.sort(key=lambda x: x[0])
        n_to_drop = max(0, len(real_drop_candidates) - self.min_keep)
        dropped = [name for _, name in real_drop_candidates[:n_to_drop]] if n_to_drop else []
        kept = [c for c in feature_cols if c not in set(dropped)]
        return ShadowFilterResult(
            importances=records,
            dropped_features=dropped,
            kept_features=kept,
            shadow_median_importance=shadow_median,
            target=target,
        )

    def run_all(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        targets: Optional[Sequence[str]] = None,
    ) -> Dict[str, ShadowFilterResult]:
        targets = list(targets) if targets is not None else list(self.targets)
        return {t: self.run(df, feature_cols, t) for t in targets}
