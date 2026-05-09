"""Backtest runner for evaluating NBA prediction accuracy on historical data.

Runs model predictions against completed games and compares to actual box
scores, producing per-stat metrics that feed the self-optimization loop.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    BacktestResult,
    TargetMetrics,
    compute_target_metrics,
)
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Evaluate model predictions against historical game outcomes.

    Usage:
        runner = BacktestRunner(manager, data_dir="data")
        result = runner.run("2026-04-01", "2026-04-15")
        print(result.summary())
    """

    # Columns used as identifiers (excluded from feature columns during prediction)
    ID_COLUMNS: List[str] = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
        "OPPONENT_ID", "OPPONENT_ABBR", "SEASON_YEAR",
    ]

    def __init__(
        self,
        manager,  # ModelManager instance (avoid circular import)
        data_dir: str = "data",
        models_dir: str = "models",
        cache_dir: Optional[str] = None,
    ):
        """Initialize the backtest runner.

        Args:
            manager: A ModelManager instance with loaded models and blend weights.
            data_dir: Path to raw data (CSV files).
            models_dir: Path to trained model artifacts.
            cache_dir: Optional path for caching feature-engineered DataFrames.
        """
        self._manager = manager
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self._feature_df: Optional[pd.DataFrame] = None
        self._feature_df_hash: str = ""

    @property
    def targets(self) -> List[str]:
        """Target stats being predicted."""
        if hasattr(self._manager, "TARGETS"):
            return list(self._manager.TARGETS)
        return ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _compute_data_hash(self) -> str:
        """Compute a fast hash of the raw data files for cache invalidation."""
        players_file = self.data_dir / "nba_players.csv"
        games_file = self.data_dir / "nba_games.csv"

        hasher = hashlib.md5()
        for fpath in (players_file, games_file):
            if fpath.exists():
                stat = fpath.stat()
                hasher.update(f"{fpath.name}:{stat.st_size}:{stat.st_mtime}".encode())
        return hasher.hexdigest()[:12]

    def load_feature_df(self, force_recompute: bool = False) -> pd.DataFrame:
        """Load or compute the feature-engineered DataFrame.

        The result is cached to disk (pickle) keyed by a hash of the raw data
        files, so subsequent backtest runs avoid recomputing all features.

        Args:
            force_recompute: If True, skip cache and recompute features.

        Returns:
            Feature-engineered DataFrame with all 150+ columns.
        """
        data_hash = self._compute_data_hash()
        cache_path = self.cache_dir / f"backtest_features_{data_hash}.pkl"

        if not force_recompute and cache_path.exists():
            logger.info("Loading cached feature DataFrame from %s", cache_path)
            self._feature_df = pd.read_pickle(cache_path)
            self._feature_df_hash = data_hash
            return self._feature_df

        logger.info("Computing feature DataFrame (this may take a while)...")
        players_file = self.data_dir / "nba_players.csv"
        games_file = self.data_dir / "nba_games.csv"

        if not players_file.exists():
            raise FileNotFoundError(f"Players file not found: {players_file}")
        if not games_file.exists():
            raise FileNotFoundError(f"Games file not found: {games_file}")

        loader = DataLoader(str(players_file), str(games_file))
        merged_df = loader.merge_datasets()
        if merged_df.empty:
            raise ValueError("Merged dataset is empty")

        feature_df = self.feature_engineer.create_features(merged_df)
        if feature_df.empty:
            raise ValueError("Feature engineering produced empty DataFrame")

        # Cache for future runs
        try:
            feature_df.to_pickle(cache_path)
            logger.info("Cached feature DataFrame to %s", cache_path)
        except Exception as exc:
            logger.warning("Failed to cache feature DataFrame: %s", exc)

        self._feature_df = feature_df
        self._feature_df_hash = data_hash
        return feature_df

    # ------------------------------------------------------------------
    # Backtest execution
    # ------------------------------------------------------------------

    def run(
        self,
        date_start: str,
        date_end: str,
        *,
        feature_df: Optional[pd.DataFrame] = None,
        force_recompute: bool = False,
        progress: bool = True,
    ) -> BacktestResult:
        """Run backtest over a date range.

        Args:
            date_start: Start date (YYYY-MM-DD), inclusive.
            date_end: End date (YYYY-MM-DD), inclusive.
            feature_df: Optional pre-computed feature DataFrame. If not
                        provided, loaded from disk or computed fresh.
            progress: If True, log progress every 100 predictions.

        Returns:
            BacktestResult with per-stat metrics and aggregate scores.
        """
        t0 = time.monotonic()

        # --- 1. Load and filter data ---
        if feature_df is not None:
            df = feature_df.copy()
        else:
            df = self.load_feature_df(force_recompute=force_recompute).copy()

        # Ensure GAME_DATE is datetime
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

        start = pd.Timestamp(date_start)
        end = pd.Timestamp(date_end)
        mask = (df["GAME_DATE"] >= start) & (df["GAME_DATE"] <= end)
        backtest_df = df[mask].copy()

        if backtest_df.empty:
            logger.warning(
                "No data found between %s and %s. Available range: %s → %s",
                date_start,
                date_end,
                df["GAME_DATE"].min().strftime("%Y-%m-%d"),
                df["GAME_DATE"].max().strftime("%Y-%m-%d"),
            )
            return BacktestResult(
                date_start=date_start,
                date_end=date_end,
                num_games=0,
                num_players=0,
                timestamp=datetime.now().isoformat(),
                data_hash=self._feature_df_hash,
            )

        # --- 2. Ensure models are loaded ---
        if not getattr(self._manager, "models", None):
            logger.info("Loading models...")
            self._manager._load_models()

        feature_cols = getattr(self._manager, "feature_cols", None)
        if feature_cols is None:
            self._manager._load_feature_cols()
            feature_cols = getattr(self._manager, "feature_cols", None)

        if not feature_cols:
            raise RuntimeError("No feature columns available — cannot run backtest")

        # --- 3. Run predictions ---
        targets = self.targets
        target_actuals: Dict[str, List[float]] = {t: [] for t in targets}
        target_preds: Dict[str, List[float]] = {t: [] for t in targets}
        target_stds: Dict[str, List[float]] = {t: [] for t in targets}

        n_rows = len(backtest_df)
        game_ids = backtest_df["GAME_ID"].nunique() if "GAME_ID" in backtest_df.columns else 0

        for i, (_, row) in enumerate(backtest_df.iterrows()):
            if progress and (i + 1) % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Backtest progress: %d/%d rows (%.0f rows/sec)",
                    i + 1, n_rows, rate,
                )

            row_df = row.to_frame().T

            try:
                preds = self._manager.predict_player_stats(row_df, history_df=None)
            except Exception as exc:
                logger.debug("Prediction failed for row %d: %s", i, exc)
                continue

            for target in targets:
                actual = row.get(target)
                pred = preds.get(target)
                std_key = f"{target}_STD"

                if actual is not None and pred is not None and pd.notna(actual) and pd.notna(pred):
                    target_actuals[target].append(float(actual))
                    target_preds[target].append(float(pred))
                    if std_key in preds and preds[std_key] is not None:
                        target_stds[target].append(float(preds[std_key]))

        # --- 4. Compute metrics ---
        per_target: Dict[str, TargetMetrics] = {}
        all_actuals: List[float] = []
        all_preds: List[float] = []

        for target in targets:
            actuals_arr = np.array(target_actuals[target], dtype=float)
            preds_arr = np.array(target_preds[target], dtype=float)
            stds_arr = (
                np.array(target_stds[target], dtype=float)
                if target_stds[target]
                else None
            )

            metrics = compute_target_metrics(target, actuals_arr, preds_arr, stds_arr)
            per_target[target] = metrics

            all_actuals.extend(target_actuals[target])
            all_preds.extend(target_preds[target])

        # Aggregates
        all_actuals_arr = np.array(all_actuals, dtype=float)
        all_preds_arr = np.array(all_preds, dtype=float)
        residuals = all_actuals_arr - all_preds_arr

        overall_mae = float(np.mean(np.abs(residuals))) if len(residuals) > 0 else float("inf")
        overall_rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else float("inf")
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((all_actuals_arr - np.mean(all_actuals_arr)) ** 2))
        overall_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        # Weighted score: core targets (PTS, REB, AST) weighted 2x
        core_weight = 2.0
        secondary_weight = 1.0
        weighted_sum = 0.0
        weight_total = 0.0
        for target, metrics in per_target.items():
            w = core_weight if target in ("PTS", "REB", "AST") else secondary_weight
            if np.isfinite(metrics.mae):
                weighted_sum += w * metrics.mae
                weight_total += w
        weighted_score = weighted_sum / weight_total if weight_total > 0 else float("inf")

        elapsed = time.monotonic() - t0
        result = BacktestResult(
            date_start=date_start,
            date_end=date_end,
            num_games=game_ids,
            num_players=backtest_df["PLAYER_ID"].nunique() if "PLAYER_ID" in backtest_df.columns else 0,
            per_target=per_target,
            overall_mae=overall_mae,
            overall_rmse=overall_rmse,
            overall_r2=overall_r2,
            weighted_score=weighted_score,
            timestamp=datetime.now().isoformat(),
            data_hash=self._feature_df_hash,
        )

        logger.info(
            "Backtest complete in %.1fs: %d rows, %.1f rows/sec | Overall MAE=%.3f",
            elapsed, n_rows, n_rows / elapsed if elapsed > 0 else 0, overall_mae,
        )

        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run_recent(self, days: int = 14) -> BacktestResult:
        """Run backtest on the most recent N days of completed games.

        Args:
            days: Number of days to look back.

        Returns:
            BacktestResult for the recent window.
        """
        end = datetime.now()
        start = end - pd.Timedelta(days=days)
        return self.run(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    def compare_results(
        self,
        baseline: BacktestResult,
        candidate: BacktestResult,
        core_only: bool = True,
    ) -> Dict[str, float]:
        """Compare two backtest results and return per-target deltas.

        Positive delta = candidate is WORSE (higher MAE).
        Negative delta = candidate is BETTER (lower MAE).

        Args:
            baseline: The current/reference result.
            candidate: The new/candidate result.
            core_only: If True, only compare core targets (PTS/REB/AST).

        Returns:
            Dict mapping target name to MAE delta.
        """
        deltas: Dict[str, float] = {}
        compare_targets = (
            ["PTS", "REB", "AST"] if core_only
            else list(baseline.per_target.keys())
        )
        for target in compare_targets:
            base_m = baseline.per_target.get(target)
            cand_m = candidate.per_target.get(target)
            if base_m and cand_m and np.isfinite(base_m.mae) and np.isfinite(cand_m.mae):
                deltas[target] = cand_m.mae - base_m.mae
            else:
                deltas[target] = float("nan")
        return deltas
