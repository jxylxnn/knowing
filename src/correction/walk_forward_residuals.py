"""Walk-forward residual builder for honest mistake-learning."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer, build_feature_engineer
from src.training.pipeline import create_pipeline
from src.training.presets import (
    CANONICAL_TARGETS,
    apply_recent_history_window,
    resolve_training_preset,
)
from src.correction.residual_dataset import ResidualTrainingRow, build_residual_dataframe

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """A single walk-forward fold."""

    name: str
    holdout_season: str
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame


class WalkForwardResidualBuilder:
    """Build a residual training dataset via chronological walk-forward folds."""

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        output_path: str = "data/evaluation/residual_training.parquet",
        summary_path: str = "data/evaluation/residual_training_summary.json",
        min_train_seasons: int = 3,
        preset: str = "full",
        mode: Optional[str] = None,
        model_size: Optional[str] = None,
        start_season: Optional[str] = None,
        end_season: Optional[str] = None,
        targets: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        parallel: bool = False,
        use_gpu: Optional[bool] = None,
    ):
        self.config_path = Path(config_path).expanduser()
        self.output_path = Path(output_path)
        self.summary_path = Path(summary_path)
        self.min_train_seasons = max(1, int(min_train_seasons))
        self.preset_name = str(preset).strip().lower()
        self.mode = mode
        self.model_size = model_size
        self.start_season = start_season
        self.end_season = end_season
        self.targets = list(targets) if targets else list(CANONICAL_TARGETS)
        self.max_workers = max_workers
        self.parallel = parallel
        self.use_gpu = use_gpu

        # Resolve preset
        self.runtime_config = load_config(self.config_path)
        self.preset = resolve_training_preset(
            self.preset_name,
            getattr(self.runtime_config, "training_presets", {}),
        )
        self.resolved_mode = self.mode or self.preset.default_mode
        self.resolved_model_size = self.model_size or self.preset.default_model_size

    @staticmethod
    def _season_column(df: pd.DataFrame) -> str:
        """Return the column to use for season-based splitting."""
        if "SEASON_ID" in df.columns:
            return "SEASON_ID"
        # Fallback: derive a proxy season from the calendar year
        if "GAME_DATE" in df.columns:
            return "_WALK_FORWARD_SEASON_YEAR"
        raise ValueError("DataFrame must contain 'SEASON_ID' or 'GAME_DATE' to build folds")

    def _ensure_season_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a proxy season column if needed."""
        col = self._season_column(df)
        if col == "_WALK_FORWARD_SEASON_YEAR":
            df = df.copy()
            df[col] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.year.astype(str)
        return df

    def build_folds(self, df: pd.DataFrame) -> List[Fold]:
        """Create chronological walk-forward folds."""
        df = self._ensure_season_column(df)
        season_col = self._season_column(df)
        seasons = sorted(df[season_col].astype(str).unique())

        if self.start_season:
            seasons = [s for s in seasons if s >= self.start_season]
        if self.end_season:
            seasons = [s for s in seasons if s <= self.end_season]

        if len(seasons) <= self.min_train_seasons:
            raise ValueError(
                f"Not enough seasons ({len(seasons)}) for min_train_seasons={self.min_train_seasons}"
            )

        folds: List[Fold] = []
        for i in range(self.min_train_seasons, len(seasons)):
            holdout_season = seasons[i]
            train_seasons = seasons[:i]
            train_df = df[df[season_col].astype(str).isin(train_seasons)].copy()
            holdout_df = df[df[season_col].astype(str) == holdout_season].copy()

            if train_df.empty or holdout_df.empty:
                logger.warning("Skipping fold %s due to empty partition", holdout_season)
                continue

            folds.append(
                Fold(
                    name=f"fold_{holdout_season}",
                    holdout_season=holdout_season,
                    train_df=train_df,
                    holdout_df=holdout_df,
                )
            )

        logger.info("Built %d walk-forward folds", len(folds))
        return folds

    def _load_and_engineer_data(self) -> pd.DataFrame:
        """Load raw CSVs and engineer features."""
        data_dir = Path("data").resolve()
        players_file = data_dir / "nba_players.csv"
        games_file = data_dir / "nba_games.csv"

        if not players_file.exists() or not games_file.exists():
            raise FileNotFoundError(
                f"Missing required data files in {data_dir}: "
                f"{'nba_players.csv' if not players_file.exists() else ''}"
                f"{'nba_games.csv' if not games_file.exists() else ''}"
            )

        loader = DataLoader(str(players_file), str(games_file))
        merged_df = loader.merge_datasets()

        # Apply recent-history window from preset
        if self.preset.recent_seasons is not None:
            trimmed_df = apply_recent_history_window(merged_df, self.preset.recent_seasons)
            if len(trimmed_df) != len(merged_df):
                logger.info(
                    "Preset recent-history window trimmed data from %d to %d rows",
                    len(merged_df),
                    len(trimmed_df),
                )
                merged_df = trimmed_df

        fe_kwargs = self.preset.feature_engineer_kwargs()
        feature_engineer = build_feature_engineer(
            rolling_windows=fe_kwargs.get("rolling_windows"),
            enable_groups=fe_kwargs.get("enable_groups"),
            disable_groups=fe_kwargs.get("disable_groups"),
            cache_dir="cache/training",
        )
        full_df = feature_engineer.create_features(merged_df)
        logger.info("Feature engineering complete: %d rows, %d columns", len(full_df), len(full_df.columns))
        return full_df

    def _predict_holdout(
        self, pipeline, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate base-model predictions for a holdout set."""
        if test_df.empty:
            return pd.DataFrame()

        X_test = pipeline.feature_selector.transform(
            test_df, pipeline.feature_schema, strict=False, fill_value=0.0
        )

        preds: Dict[str, np.ndarray] = {}
        for target in self.targets:
            model = pipeline.models.get(target)
            if model is None:
                preds[target] = np.full(len(test_df), np.nan)
                continue

            # Determine the exact feature columns this target was trained on
            trainer = pipeline.trainers.get(f"catboost_{target}")
            if trainer and hasattr(trainer, "feature_cols") and trainer.feature_cols:
                feat_cols = trainer.feature_cols
                X_target = X_test[feat_cols] if feat_cols else X_test
            else:
                X_target = X_test

            try:
                pred = np.asarray(model.predict(X_target), dtype=float)
            except Exception as exc:
                logger.warning("Prediction failed for target %s: %s", target, exc)
                pred = np.full(len(test_df), np.nan)
            preds[target] = pred

        return pd.DataFrame(preds, index=test_df.index)

    def _build_residual_rows(
        self,
        test_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        fold_name: str,
        cutoff_date: str,
    ) -> List[ResidualTrainingRow]:
        """Compare predictions to actuals and return residual rows."""
        rows: List[ResidualTrainingRow] = []
        for target in self.targets:
            if target not in preds_df.columns:
                continue
            actuals = test_df[target].values
            predictions = preds_df[target].values
            for i in range(len(test_df)):
                actual = actuals[i]
                if pd.isna(actual):
                    continue
                pred = predictions[i]
                row = test_df.iloc[i]
                rows.append(
                    ResidualTrainingRow(
                        game_id=str(row.get("GAME_ID", "")),
                        game_date=str(row.get("GAME_DATE", "")),
                        player_id=str(row.get("PLAYER_ID", "")),
                        player_name=str(row.get("PLAYER_NAME", "")),
                        team_id=str(row.get("TEAM_ID", "")),
                        opponent=str(row.get("OPPONENT_ABBR", row.get("OPPONENT_ID", ""))),
                        stat=target,
                        base_prediction=float(pred),
                        actual=float(actual),
                        error=float(actual) - float(pred),
                        model_fold=fold_name,
                        model_version=None,
                        data_quality=str(row.get("DATA_QUALITY", None))
                        if "DATA_QUALITY" in test_df.columns
                        else None,
                        feature_cutoff_date=cutoff_date,
                    )
                )
        return rows

    def _process_fold(self, fold: Fold) -> List[ResidualTrainingRow]:
        """Train a base model on the fold and return residual rows."""
        logger.info("--- Processing fold %s (holdout %s) ---", fold.name, fold.holdout_season)

        with tempfile.TemporaryDirectory(prefix="residual_fold_") as tmpdir:
            models_dir = Path(tmpdir) / "models"
            cache_dir = Path(tmpdir) / "cache"
            models_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            pipeline = create_pipeline(
                mode=self.resolved_mode,
                data_dir="data",
                models_dir=str(models_dir),
                cache_dir=str(cache_dir),
                model_size=self.resolved_model_size,
                parallel=self.parallel,
                max_workers=self.max_workers,
                use_gpu=self.use_gpu,
            )
            pipeline.training_preset = self.preset.name
            pipeline.feature_group_selection = list(self.preset.enable_groups)
            pipeline.model_config["transformer"]["enabled"] = bool(
                self.preset.transformer_enabled
            )

            # Combine train + holdout for pipeline.prepare_data
            fold_df = pd.concat([fold.train_df, fold.holdout_df], ignore_index=True)
            fold_df["GAME_DATE"] = pd.to_datetime(fold_df["GAME_DATE"], errors="coerce")
            fold_df = fold_df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)

            if fold_df.empty:
                logger.warning("Fold %s has no valid rows after date cleaning", fold.name)
                return []

            holdout_start = fold.holdout_df["GAME_DATE"].min()
            test_date_str = (
                holdout_start.strftime("%Y-%m-%d")
                if pd.notna(holdout_start)
                else None
            )

            try:
                fit_df, val_df, test_df = pipeline.prepare_data(
                    fold_df, test_date=test_date_str
                )
            except Exception as exc:
                logger.warning("Fold %s prepare_data failed: %s", fold.name, exc)
                return []

            if test_df.empty:
                logger.warning("Fold %s produced empty test/holdout partition", fold.name)
                return []

            try:
                pipeline.train(fit_df, val_df)
            except Exception as exc:
                logger.warning("Fold %s training failed: %s", fold.name, exc)
                return []

            preds_df = self._predict_holdout(pipeline, test_df)
            rows = self._build_residual_rows(
                test_df, preds_df, fold.name, str(test_date_str or "")
            )
            logger.info(
                "Fold %s complete: %d residual rows", fold.name, len(rows)
            )
            return rows

    def run(self) -> pd.DataFrame:
        """Run the full walk-forward residual build and save outputs."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

        full_df = self._load_and_engineer_data()
        folds = self.build_folds(full_df)

        all_rows: List[ResidualTrainingRow] = []
        for fold in folds:
            rows = self._process_fold(fold)
            all_rows.extend(rows)

        residual_df = build_residual_dataframe(all_rows)
        if residual_df.empty:
            logger.warning("Residual dataset is empty — no rows were produced")
        else:
            residual_df.to_parquet(self.output_path, index=False)
            logger.info("Saved residual dataset to %s (%d rows)", self.output_path, len(residual_df))

        summary = self._build_summary(folds, residual_df)
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Saved residual summary to %s", self.summary_path)

        return residual_df

    def _build_summary(
        self, folds: List[Fold], residual_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build a JSON summary of the residual dataset."""
        summary: Dict[str, Any] = {
            "num_folds": len(folds),
            "num_residual_rows": len(residual_df),
            "targets": self.targets,
        }
        if not residual_df.empty:
            summary["date_range"] = {
                "min": str(residual_df["GAME_DATE"].min()),
                "max": str(residual_df["GAME_DATE"].max()),
            }
            summary["mean_absolute_error_by_stat"] = {
                target: float(
                    residual_df[residual_df["STAT"] == target]["ERROR"].abs().mean()
                )
                for target in self.targets
                if target in residual_df["STAT"].values
            }
        else:
            summary["date_range"] = None
            summary["mean_absolute_error_by_stat"] = {}
        return summary
