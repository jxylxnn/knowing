"""Self-optimizing ensemble weight tuner.

Takes a backtest runner, evaluates candidate blend weights against a holdout
set of recently completed games, and uses scipy.optimize to find weights that
minimize prediction error.  Accept/verify gates prevent regressions from being
deployed.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics import BacktestResult, compute_target_metrics
from src.evaluation.weight_store import EnsembleWeights, TargetBlend, WeightStore

logger = logging.getLogger(__name__)


# Number of tunable parameters:
#   6 per-target CatBoost/Transformer blend ratios
#   6 per-target intercepts
#   1 global CatBoost-MAE blend ratio
#   Total: 13 parameters
_TUNABLE_DIMS = 13


@dataclass
class OptimizationResult:
    """Result of a single optimization run."""

    accepted: bool
    weights: EnsembleWeights
    baseline_score: float
    candidate_score: float
    improvement_pct: float
    num_iterations: int
    optimizer_message: str
    holdout_result: Optional[BacktestResult] = None
    verification_result: Optional[BacktestResult] = None
    rejection_reason: str = ""


class EnsembleOptimizer:
    """Optimize ensemble blend weights using holdout backtesting.

    Usage:
        runner = BacktestRunner(manager)
        store = WeightStore("models/blend_weights")
        optimizer = EnsembleOptimizer(runner, store)

        result = optimizer.optimize(
            holdout_start="2026-04-15",
            holdout_end="2026-05-01",
        )
        if result.accepted:
            print(f"Deployed v{result.weights.version}")
    """

    # Per-target blend ratio bounds: catboost+transformer ≈ 1.0 constraint
    # is handled by the optimizer normalizing to sum-to-1 internally.
    BLEND_BOUNDS = (0.1, 0.9)       # catboost fraction ∈ [0.1, 0.9]
    INTERCEPT_BOUNDS = (-3.0, 3.0)  # per-stat intercept ∈ [-3, 3]
    MAE_BLEND_BOUNDS = (0.5, 0.95)  # catboost fraction ∈ [0.5, 0.95]

    def __init__(
        self,
        backtest_runner,   # BacktestRunner (avoid circular import)
        weight_store: WeightStore,
        *,
        accept_margin: float = 0.01,
        verification_margin: float = 0.02,
        max_iterations: int = 100,
    ):
        self._runner = backtest_runner
        self._store = weight_store
        self.accept_margin = accept_margin
        self.verification_margin = verification_margin
        self.max_iterations = max_iterations

        self._targets: List[str] = self._runner.targets
        self._n_targets = len(self._targets)

    # ------------------------------------------------------------------
    # Parameter encoding
    # ------------------------------------------------------------------

    def _weights_to_vector(self, weights: EnsembleWeights) -> np.ndarray:
        """Encode EnsembleWeights into a flat parameter vector.

        Layout: [cb_ratio_t0, ..., cb_ratio_t5, intercept_t0, ..., intercept_t5, cb_mae_blend]
        """
        vec = np.zeros(_TUNABLE_DIMS, dtype=float)
        for i, target in enumerate(self._targets):
            tb = weights.per_target.get(target, TargetBlend())
            vec[i] = tb.catboost  # transformer weight = 1.0 - catboost
            vec[self._n_targets + i] = tb.intercept
        vec[-1] = weights.catboost_mae_blend
        return vec

    def _vector_to_weights(
        self, vec: np.ndarray, base_weights: Optional[EnsembleWeights] = None
    ) -> EnsembleWeights:
        """Decode a flat parameter vector into EnsembleWeights.

        The CatBoost fraction is clamped to [0.1, 0.9]; the Transformer
        fraction is inferred as 1.0 - catboost (so they sum to 1.0).

        Args:
            vec: Flat parameter vector.
            base_weights: Optional base to copy metadata from.

        Returns:
            New EnsembleWeights with decoded parameters.
        """
        per_target: Dict[str, TargetBlend] = {}
        for i, target in enumerate(self._targets):
            cb = float(np.clip(vec[i], *self.BLEND_BOUNDS))
            tx = 1.0 - cb  # sum-to-1 constraint
            intercept = float(np.clip(vec[self._n_targets + i], *self.INTERCEPT_BOUNDS))
            per_target[target] = TargetBlend(
                catboost=cb,
                transformer=tx,
                intercept=intercept,
                catboost_mae_blend=float(np.clip(vec[-1], *self.MAE_BLEND_BOUNDS)),
            )

        weights = EnsembleWeights(
            per_target=per_target,
            catboost_mae_blend=float(np.clip(vec[-1], *self.MAE_BLEND_BOUNDS)),
            created_at=datetime.now().isoformat(),
            description="Optimizer candidate",
        )

        if base_weights is not None:
            weights.parent_version = base_weights.version

        return weights

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _build_objective(
        self,
        holdout_start: str,
        holdout_end: str,
        feature_df,
        baseline_score: float,
    ) -> Callable[[np.ndarray], float]:
        """Build the objective function for scipy.optimize.

        Returns a callable that takes a parameter vector and returns the
        weighted MAE on the holdout set.  Lower is better.

        The feature_df is captured in the closure to avoid recomputing
        features on every iteration.
        """

        def objective(vec: np.ndarray) -> float:
            # Decode to weights
            candidate = self._vector_to_weights(vec)

            # Apply to model manager
            manager = self._runner._manager
            manager.use_ensemble_weights(candidate)

            # Run backtest on holdout
            result = self._runner.run(
                holdout_start,
                holdout_end,
                feature_df=feature_df,
                progress=False,
            )

            # Weighted score: core targets (PTS, REB, AST) weighted 2x
            core_weight = 2.0
            secondary_weight = 1.0
            total_weight = 0.0
            weighted_sum = 0.0

            for target, metrics in result.per_target.items():
                w = core_weight if target in ("PTS", "REB", "AST") else secondary_weight
                if np.isfinite(metrics.mae):
                    weighted_sum += w * metrics.mae
                    total_weight += w

            if total_weight == 0:
                return float("inf")

            return weighted_sum / total_weight

        return objective

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        holdout_start: str,
        holdout_end: str,
        *,
        verification_start: Optional[str] = None,
        verification_end: Optional[str] = None,
        progress: bool = True,
    ) -> OptimizationResult:
        """Run the self-optimization loop.

        1. Baseline: backtest current weights on holdout
        2. Optimize: find candidate weights that minimize holdout MAE
        3. Accept gate: candidate must improve by ≥ accept_margin
        4. Verify gate: candidate must not degrade verification set by > verification_margin
        5. Deploy: atomically save new weights if both gates pass

        Args:
            holdout_start: Start of holdout period (YYYY-MM-DD).
            holdout_end: End of holdout period (YYYY-MM-DD).
            verification_start: Optional start of verification period.
                                Defaults to moving the holdout window back by
                                the same duration.
            verification_end: Optional end of verification period.
            progress: If True, log optimization progress.

        Returns:
            OptimizationResult with acceptance status and details.
        """
        # --- 0. Load feature DataFrame once (expensive) ---
        feature_df = self._runner.load_feature_df()
        manager = self._runner._manager

        # Load current weights
        current = self._store.load_current()
        if current is None:
            current = EnsembleWeights.default_for_targets(self._targets)
            logger.info("No current weights found; using defaults")

        # Ensure models are loaded
        if not getattr(manager, "models", None):
            manager._load_models()

        # --- 1. Baseline ---
        logger.info("Computing baseline on %s → %s...", holdout_start, holdout_end)
        manager.use_ensemble_weights(current)
        baseline_result = self._runner.run(
            holdout_start, holdout_end,
            feature_df=feature_df, progress=progress,
        )
        baseline_score = baseline_result.weighted_score

        if not np.isfinite(baseline_score):
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=float("inf"),
                candidate_score=float("inf"),
                improvement_pct=0.0,
                num_iterations=0,
                optimizer_message="Baseline score is invalid (no holdout data?)",
                rejection_reason="invalid_baseline",
            )

        logger.info("Baseline weighted MAE: %.4f", baseline_score)

        # --- 2. Optimize ---
        x0 = self._weights_to_vector(current)
        objective_fn = self._build_objective(
            holdout_start, holdout_end, feature_df, baseline_score,
        )

        logger.info("Starting Nelder-Mead optimization (%d iterations max)...", self.max_iterations)

        try:
            from scipy.optimize import minimize

            # Build bounds: [(blend_low, blend_high) * n_targets, (int_low, int_high) * n_targets, (mae_low, mae_high)]
            bounds = (
                [self.BLEND_BOUNDS] * self._n_targets
                + [self.INTERCEPT_BOUNDS] * self._n_targets
                + [self.MAE_BLEND_BOUNDS]
            )

            opt_result = minimize(
                objective_fn,
                x0,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "xatol": 1e-4,
                    "fatol": 1e-4,
                    "adaptive": True,
                },
            )
        except ImportError:
            logger.error("scipy is required for optimization. Install with: pip install scipy")
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=baseline_score,
                candidate_score=float("inf"),
                improvement_pct=0.0,
                num_iterations=0,
                optimizer_message="scipy not installed",
                rejection_reason="missing_dependency",
            )

        candidate_score = float(opt_result.fun)
        improvement = baseline_score - candidate_score
        improvement_pct = (improvement / baseline_score * 100.0) if baseline_score > 0 else 0.0

        logger.info(
            "Optimization complete: %.4f → %.4f (Δ=%.4f, %.2f%%) after %d iters",
            baseline_score, candidate_score, improvement, improvement_pct, opt_result.nit,
        )

        # --- 3. Accept gate ---
        if improvement < self.accept_margin * baseline_score:
            reason = (
                f"Improvement {improvement:.4f} < accept_margin "
                f"{self.accept_margin * baseline_score:.4f}"
            )
            logger.info("ACCEPT GATE FAILED: %s", reason)
            # Restore current weights
            manager.use_ensemble_weights(current)
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                improvement_pct=improvement_pct,
                num_iterations=opt_result.nit,
                optimizer_message=opt_result.message,
                holdout_result=baseline_result,
                rejection_reason=reason,
            )

        # Decode candidate weights
        candidate_weights = self._vector_to_weights(opt_result.x, base_weights=current)
        candidate_weights.backtest_score = candidate_score
        candidate_weights.backtest_date_range = f"{holdout_start}→{holdout_end}"

        manager.use_ensemble_weights(candidate_weights)

        # --- 4. Verify gate ---
        if verification_start is None:
            # Default: use a window before the holdout of equal duration
            holdout_start_dt = datetime.strptime(holdout_start, "%Y-%m-%d")
            holdout_end_dt = datetime.strptime(holdout_end, "%Y-%m-%d")
            duration = (holdout_end_dt - holdout_start_dt).days
            ver_end = holdout_start_dt - timedelta(days=1)
            ver_start = ver_end - timedelta(days=duration)
            verification_start = ver_start.strftime("%Y-%m-%d")
            verification_end = ver_end.strftime("%Y-%m-%d")

        logger.info("Verification backtest on %s → %s...", verification_start, verification_end)
        verify_result = self._runner.run(
            verification_start, verification_end,
            feature_df=feature_df, progress=progress,
        )
        verify_score = verify_result.weighted_score

        if not np.isfinite(verify_score):
            logger.warning("Verification score is invalid — skipping verify gate")
        else:
            verify_degradation = verify_score - baseline_score
            verify_degradation_pct = (
                (verify_degradation / baseline_score * 100.0) if baseline_score > 0 else 0.0
            )

            if verify_degradation > self.verification_margin * baseline_score:
                reason = (
                    f"Verification degradation {verify_degradation:.4f} "
                    f"({verify_degradation_pct:.1f}%) > margin "
                    f"{self.verification_margin * baseline_score:.4f}"
                )
                logger.info("VERIFY GATE FAILED: %s", reason)
                # Restore current weights
                manager.use_ensemble_weights(current)
                return OptimizationResult(
                    accepted=False,
                    weights=current,
                    baseline_score=baseline_score,
                    candidate_score=candidate_score,
                    improvement_pct=improvement_pct,
                    num_iterations=opt_result.nit,
                    optimizer_message=opt_result.message,
                    holdout_result=baseline_result,
                    verification_result=verify_result,
                    rejection_reason=reason,
                )

        # --- 5. Deploy ---
        candidate_weights.description = (
            f"Optimized: MAE {baseline_score:.4f}→{candidate_score:.4f} "
            f"({improvement_pct:+.1f}%) on {holdout_start}→{holdout_end}"
        )
        candidate_weights.optimizer_method = "Nelder-Mead"
        candidate_weights.accept_margin = self.accept_margin

        version = self._store.save(candidate_weights, set_current=True)
        logger.info("DEPLOYED v%d: ΔMAE = %.4f (%.2f%%)", version, improvement, improvement_pct)

        return OptimizationResult(
            accepted=True,
            weights=candidate_weights,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            improvement_pct=improvement_pct,
            num_iterations=opt_result.nit,
            optimizer_message=opt_result.message,
            holdout_result=baseline_result,
            verification_result=verify_result if np.isfinite(verify_score) else None,
        )
