"""Residual model trainer — learns base-model mistakes per target."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None
    Pool = None

from src.correction.correction_features import CorrectionFeatureBuilder

logger = logging.getLogger(__name__)

CATBOOST_AVAILABLE = CatBoostRegressor is not None


@dataclass
class ResidualTargetResult:
    """Training result for a single target stat."""

    stat: str
    rows: int
    base_mae: float
    corrected_mae: float
    mae_improvement: float
    mae_improvement_pct: float
    status: str
    reason: str = ""
    training_time: float = 0.0
    best_iteration: Optional[int] = None


@dataclass
class ResidualTrainingResult:
    """Aggregate result across all targets."""

    targets: Dict[str, ResidualTargetResult] = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=list)
    total_time: float = 0.0

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize to metadata dict for residual_metadata.json."""
        return {
            "trained_at": datetime.now().isoformat(),
            "total_time": self.total_time,
            "feature_cols": self.feature_cols,
            "targets": {
                stat: {
                    "rows": r.rows,
                    "base_mae": round(r.base_mae, 4),
                    "corrected_mae": round(r.corrected_mae, 4),
                    "mae_improvement": round(r.mae_improvement, 4),
                    "mae_improvement_pct": round(r.mae_improvement_pct, 2),
                    "status": r.status,
                    "reason": r.reason,
                    "training_time": round(r.training_time, 2),
                    "best_iteration": r.best_iteration,
                }
                for stat, r in self.targets.items()
            },
        }


class ResidualModelTrainer:
    """Train one CatBoost residual model per stat target."""

    TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")

    def __init__(
        self,
        min_rows: int = 1000,
        acceptance_min_improvement: float = 0.0,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.2,
        random_seed: int = 42,
    ):
        self.min_rows = min_rows
        self.acceptance_min_improvement = acceptance_min_improvement
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_seed = random_seed

    def train_all(
        self,
        residual_path: str,
        output_dir: str,
        targets: Optional[List[str]] = None,
    ) -> ResidualTrainingResult:
        """Train residual models for all targets.

        Args:
            residual_path: Path to residual_training.parquet.
            output_dir: Directory to save model artifacts.
            targets: Override list of targets (defaults to TARGETS).

        Returns:
            ResidualTrainingResult with per-target metrics.
        """
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not available. Install catboost to train residual models.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        residual_df = pd.read_parquet(residual_path)
        logger.info("Loaded residual dataset: %d rows", len(residual_df))

        builder = CorrectionFeatureBuilder()
        featured_df, feature_cols = builder.build(residual_df)

        target_list = targets or list(self.TARGETS)
        result = ResidualTrainingResult(feature_cols=feature_cols)
        start = time.time()

        for stat in target_list:
            stat_result = self._train_target(stat, featured_df, feature_cols, output_path)
            result.targets[stat] = stat_result

        result.total_time = time.time() - start

        self._save_metadata(result, output_path, feature_cols)
        self._save_feature_schema(feature_cols, output_path)

        logger.info(
            "Residual training complete: %d targets in %.1fs",
            len(result.targets),
            result.total_time,
        )
        return result

    def _train_target(
        self,
        stat: str,
        featured_df: pd.DataFrame,
        feature_cols: List[str],
        output_path: Path,
    ) -> ResidualTargetResult:
        """Train a residual model for a single stat."""
        stat_df = featured_df[featured_df["STAT"] == stat].copy()

        if len(stat_df) < self.min_rows:
            logger.warning(
                "Skipping %s: only %d rows (min_rows=%d)",
                stat, len(stat_df), self.min_rows,
            )
            return ResidualTargetResult(
                stat=stat,
                rows=len(stat_df),
                base_mae=0.0,
                corrected_mae=0.0,
                mae_improvement=0.0,
                mae_improvement_pct=0.0,
                status="rejected",
                reason=f"insufficient rows ({len(stat_df)} < {self.min_rows})",
            )

        stat_df = stat_df.sort_values("GAME_DATE").reset_index(drop=True)
        split_idx = int(len(stat_df) * (1.0 - self.val_fraction))
        train_df = stat_df.iloc[:split_idx]
        val_df = stat_df.iloc[split_idx:]

        X_train = train_df[feature_cols].values
        y_train = train_df["ERROR"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["ERROR"].values

        train_start = time.time()

        model = CatBoostRegressor(
            loss_function="MAE",
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
            random_seed=self.random_seed,
            verbose=False,
        )

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, verbose=False)
        training_time = time.time() - train_start

        best_iteration = int(model.get_best_iteration()) if hasattr(model, "get_best_iteration") else None

        val_pred = model.predict(X_val)

        base_mae = float(np.mean(np.abs(val_df["ACTUAL"].values - val_df["BASE_PREDICTION"].values)))
        corrected_pred = val_df["BASE_PREDICTION"].values + val_pred
        corrected_mae = float(np.mean(np.abs(val_df["ACTUAL"].values - corrected_pred)))

        mae_improvement = base_mae - corrected_mae
        mae_improvement_pct = (mae_improvement / base_mae * 100.0) if base_mae > 0 else 0.0

        if corrected_mae > base_mae:
            status = "rejected"
            reason = "corrected_mae worse than base_mae"
        elif mae_improvement < self.acceptance_min_improvement:
            status = "rejected"
            reason = f"improvement {mae_improvement:.4f} below threshold {self.acceptance_min_improvement}"
        else:
            status = "accepted"
            reason = ""

        model_file = output_path / f"{stat.lower()}_residual.cbm"
        model.save_model(str(model_file))
        logger.info(
            "%s: base_mae=%.4f corrected_mae=%.4f improvement=%.4f (%.2f%%) [%s]",
            stat, base_mae, corrected_mae, mae_improvement, mae_improvement_pct, status,
        )

        return ResidualTargetResult(
            stat=stat,
            rows=len(stat_df),
            base_mae=base_mae,
            corrected_mae=corrected_mae,
            mae_improvement=mae_improvement,
            mae_improvement_pct=mae_improvement_pct,
            status=status,
            reason=reason,
            training_time=training_time,
            best_iteration=best_iteration,
        )

    def _save_metadata(
        self,
        result: ResidualTrainingResult,
        output_path: Path,
        feature_cols: List[str],
    ) -> None:
        """Save residual_metadata.json."""
        metadata = result.to_metadata()
        metadata_path = output_path / "residual_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved metadata to %s", metadata_path)

    def _save_feature_schema(
        self,
        feature_cols: List[str],
        output_path: Path,
    ) -> None:
        """Save residual_feature_schema.json."""
        schema = {
            "version": 1,
            "feature_cols": feature_cols,
            "created_at": datetime.now().isoformat(),
        }
        schema_path = output_path / "residual_feature_schema.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        logger.info("Saved feature schema to %s", schema_path)
