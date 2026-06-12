"""Feature-group ablation runner for smart feature selection.

The group ablation step trains a baseline model on the full feature set and a
leave-one-out (LOO) model for each feature group, then compares the
validation MAE per target.  A positive group score (``ablated_mae -
baseline_mae``) means removing the group *hurts* prediction quality, so the
group helps.  Negative scores mean the group adds noise.

The runner is intentionally lightweight — it uses a small gradient-boosted
regressor per target so that the smart selector can iterate quickly even on
modest hardware.  The output feeds into ``SmartFeatureSelector``, which
combines these scores with per-feature importances, stability, and shadow
filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# Default target list — kept in sync with TrainingPipeline.TARGETS.
DEFAULT_TARGETS: tuple[str, ...] = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


@dataclass
class GroupScore:
    """Per-target MAE delta when a feature group is removed."""

    group: str
    target: str
    baseline_mae: float
    ablated_mae: float
    score: float  # ablated_mae - baseline_mae  (positive = group helps)
    baseline_rmse: float
    ablated_rmse: float
    ablated_feature_count: int
    n_train: int
    n_val: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "group": self.group,
            "target": self.target,
            "baseline_mae": float(self.baseline_mae),
            "ablated_mae": float(self.ablated_mae),
            "score": float(self.score),
            "baseline_rmse": float(self.baseline_rmse),
            "ablated_rmse": float(self.ablated_rmse),
            "ablated_feature_count": int(self.ablated_feature_count),
            "n_train": int(self.n_train),
            "n_val": int(self.n_val),
        }


@dataclass
class AblationReport:
    """Full ablation report covering all (group, target) pairs."""

    baseline_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    group_scores: List[GroupScore] = field(default_factory=list)
    selected_groups: List[str] = field(default_factory=list)
    dropped_groups: List[str] = field(default_factory=list)
    min_gain: float = 0.0

    def by_target(self) -> Dict[str, List[GroupScore]]:
        out: Dict[str, List[GroupScore]] = {t: [] for t in DEFAULT_TARGETS}
        for score in self.group_scores:
            out.setdefault(score.target, []).append(score)
        return out

    def average_score_by_group(self) -> Dict[str, float]:
        agg: Dict[str, List[float]] = {}
        for score in self.group_scores:
            agg.setdefault(score.group, []).append(score.score)
        return {g: float(np.mean(v)) for g, v in agg.items()}

    def to_dict(self) -> Dict:
        return {
            "baseline_metrics": self.baseline_metrics,
            "group_scores": [s.to_dict() for s in self.group_scores],
            "selected_groups": list(self.selected_groups),
            "dropped_groups": list(self.dropped_groups),
            "min_gain": float(self.min_gain),
        }


class FeatureGroupAblator:
    """Train-and-compare loop that scores feature groups by MAE delta.

    The ablator expects pre-engineered features with stable group
    attribution (a ``group_columns`` mapping).  If the mapping is missing
    or a column is in multiple groups, the column is kept in every group
    (so it only disappears when *all* of its groups are dropped).
    """

    def __init__(
        self,
        targets: Sequence[str] = DEFAULT_TARGETS,
        model_factory: Optional[callable] = None,
        random_state: int = 42,
    ):
        self.targets = list(targets)
        self.random_state = int(random_state)
        self._model_factory = model_factory or self._default_factory

    def _default_factory(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            random_state=self.random_state,
            max_depth=5,
            max_iter=120,
            learning_rate=0.05,
        )

    # ------------------------------------------------------------------
    # Validation split
    # ------------------------------------------------------------------

    def _temporal_split(
        self,
        df: pd.DataFrame,
        val_ratio: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "GAME_DATE" in df.columns:
            ordered = df.sort_values("GAME_DATE", kind="mergesort")
        else:
            ordered = df.reset_index(drop=True)
        split_idx = max(1, int(len(ordered) * (1 - val_ratio)))
        split_idx = min(split_idx, len(ordered) - 1)
        return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()

    def _safe_y(self, series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    def _score_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[float, float]:
        return (
            float(mean_absolute_error(y_true, y_pred)),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
        )

    def _fit_and_score(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> tuple[float, float]:
        if len(X_train) < 20 or len(X_val) < 5:
            return float("nan"), float("nan")
        model = self._model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return self._score_regression(y_val, preds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        full_df: pd.DataFrame,
        feature_cols: Sequence[str],
        group_columns: Mapping[str, Sequence[str]],
        targets: Optional[Sequence[str]] = None,
        val_ratio: float = 0.2,
        min_gain: float = 0.0,
    ) -> AblationReport:
        """Run the ablation.  ``group_columns`` maps group name -> list of columns.

        Returns an :class:`AblationReport` with per-target group scores and a
        simple average-then-threshold summary of which groups to keep.
        """
        targets = list(targets) if targets is not None else list(self.targets)
        feature_cols = [c for c in feature_cols if c in full_df.columns]
        if not feature_cols:
            logger.warning("FeatureGroupAblator: no feature columns available — skipping")
            return AblationReport(min_gain=min_gain)

        # Keep only groups that have at least one column in the feature set.
        active_groups: Dict[str, List[str]] = {
            g: [c for c in cols if c in feature_cols]
            for g, cols in group_columns.items()
        }
        active_groups = {g: cols for g, cols in active_groups.items() if cols}
        if not active_groups:
            logger.warning("FeatureGroupAblator: no active groups — skipping")
            return AblationReport(min_gain=min_gain)

        train_df, val_df = self._temporal_split(full_df, val_ratio=val_ratio)
        if train_df.empty or val_df.empty:
            logger.warning("FeatureGroupAblator: empty train/val split — skipping")
            return AblationReport(min_gain=min_gain)

        report = AblationReport(min_gain=min_gain)

        for target in targets:
            if target not in train_df.columns:
                logger.debug("Ablation: target %s not in data; skipping", target)
                continue

            y_train = self._safe_y(train_df[target])
            y_val = self._safe_y(val_df[target])
            train_mask = np.isfinite(y_train)
            val_mask = np.isfinite(y_val)
            if train_mask.sum() < 20 or val_mask.sum() < 5:
                logger.debug("Ablation: too few labelled rows for %s", target)
                continue

            X_train_full = train_df.loc[train_mask, feature_cols].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0.0)
            X_val_full = val_df.loc[val_mask, feature_cols].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0.0)

            base_mae, base_rmse = self._fit_and_score(
                X_train_full, y_train[train_mask], X_val_full, y_val[val_mask]
            )
            report.baseline_metrics[target] = {
                "mae": base_mae,
                "rmse": base_rmse,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
            }

            for group, cols in active_groups.items():
                ablated_cols = [c for c in feature_cols if c not in set(cols)]
                if not ablated_cols:
                    continue
                X_train_abl = train_df.loc[train_mask, ablated_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0.0)
                X_val_abl = val_df.loc[val_mask, ablated_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0.0)
                ablated_mae, ablated_rmse = self._fit_and_score(
                    X_train_abl, y_train[train_mask], X_val_abl, y_val[val_mask]
                )
                score = float(ablated_mae - base_mae) if np.isfinite(base_mae) else 0.0
                report.group_scores.append(
                    GroupScore(
                        group=group,
                        target=target,
                        baseline_mae=base_mae,
                        ablated_mae=ablated_mae,
                        score=score,
                        baseline_rmse=base_rmse,
                        ablated_rmse=ablated_rmse,
                        ablated_feature_count=len(ablated_cols),
                        n_train=int(train_mask.sum()),
                        n_val=int(val_mask.sum()),
                    )
                )

        avg_by_group = report.average_score_by_group()
        report.selected_groups = sorted(
            g for g, score in avg_by_group.items() if score >= min_gain
        )
        report.dropped_groups = sorted(
            g for g, score in avg_by_group.items() if score < min_gain
        )
        logger.info(
            "FeatureGroupAblator: %d groups evaluated, %d kept, %d dropped",
            len(active_groups),
            len(report.selected_groups),
            len(report.dropped_groups),
        )
        return report


def filter_group_columns(
    group_columns: Mapping[str, Sequence[str]],
    allowed_groups: Iterable[str],
) -> Dict[str, List[str]]:
    """Return a copy of ``group_columns`` keeping only the allowed groups."""
    allowed = set(allowed_groups)
    return {g: list(cols) for g, cols in group_columns.items() if g in allowed}
