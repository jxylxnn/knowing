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
    dry_run: bool = False
    deployed: bool = False
    current_verification_score: Optional[float] = None
    candidate_verification_score: Optional[float] = None


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
        dry_run: bool = False,
    ) -> OptimizationResult:
        """Run the self-optimization loop.

        1. Baseline: backtest current weights on holdout
        2. Optimize: find candidate weights that minimize holdout MAE
        3. Accept gate: candidate must improve by ≥ accept_margin
        4. Verify gate: candidate must not degrade the verification window
           relative to the CURRENT weights on the exact same window
        5. Deploy: atomically save new weights if both gates pass (never in
           dry-run mode)

        Args:
            holdout_start: Start of holdout period (YYYY-MM-DD).
            holdout_end: End of holdout period (YYYY-MM-DD).
            verification_start: Optional start of verification period.
                                Defaults to moving the holdout window back by
                                the same duration.
            verification_end: Optional end of verification period.
            progress: If True, log optimization progress.
            dry_run: If True, evaluate everything but never save, never
                     update current.json, and never append promotion history.
                     An audit record is still appended.

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

        # Resolve the verification window up front so every audit record
        # carries both ranges, even for runs that fail before verification.
        if verification_start is None:
            # Default: use a window before the holdout of equal duration
            holdout_start_dt = datetime.strptime(holdout_start, "%Y-%m-%d")
            holdout_end_dt = datetime.strptime(holdout_end, "%Y-%m-%d")
            duration = (holdout_end_dt - holdout_start_dt).days
            ver_end = holdout_start_dt - timedelta(days=1)
            ver_start = ver_end - timedelta(days=duration)
            verification_start = ver_start.strftime("%Y-%m-%d")
            verification_end = ver_end.strftime("%Y-%m-%d")

        # Apply the starting weights. They are restored on every rejection,
        # exception, and dry-run exit (see finally below).
        manager.use_ensemble_weights(current)

        result: Optional[OptimizationResult] = None
        try:
            result = self._optimize_inner(
                manager=manager,
                current=current,
                feature_df=feature_df,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
                verification_start=verification_start,
                verification_end=verification_end,
                progress=progress,
                dry_run=dry_run,
            )
            return result
        finally:
            # Only a real (non-dry-run) promotion keeps the candidate weights
            # hot-loaded on the manager. Everything else — rejection,
            # exception, dry-run exit — restores the original weights.
            if result is None or not result.deployed:
                manager.use_ensemble_weights(current)

    # ------------------------------------------------------------------
    # Optimization body
    # ------------------------------------------------------------------

    def _optimize_inner(
        self,
        *,
        manager,
        current: EnsembleWeights,
        feature_df,
        holdout_start: str,
        holdout_end: str,
        verification_start: str,
        verification_end: str,
        progress: bool,
        dry_run: bool,
    ) -> OptimizationResult:
        """Run gates, verification, and (optionally) promotion.

        The caller restores the manager's original weights unless the result
        reports ``deployed=True``.
        """
        optimizer_method = "Nelder-Mead"

        # --- 1. Baseline ---
        logger.info("Computing baseline on %s → %s...", holdout_start, holdout_end)
        manager.use_ensemble_weights(current)
        baseline_result = self._runner.run(
            holdout_start, holdout_end,
            feature_df=feature_df, progress=progress,
        )
        baseline_score = baseline_result.weighted_score

        if not np.isfinite(baseline_score):
            reason = "invalid_baseline"
            self._record_run(
                dry_run=dry_run, accepted=False, reason=reason,
                optimizer_method="",
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=baseline_score, candidate_holdout_score=None,
                current_verification_score=None, candidate_verification_score=None,
                iterations=0, source_version=current.version, promoted_version=None,
            )
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=float("inf"),
                candidate_score=float("inf"),
                improvement_pct=0.0,
                num_iterations=0,
                optimizer_message="Baseline score is invalid (no holdout data?)",
                rejection_reason=reason,
                dry_run=dry_run,
                holdout_result=baseline_result,
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
            reason = "missing_dependency"
            self._record_run(
                dry_run=dry_run, accepted=False, reason=reason,
                optimizer_method="",
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=baseline_score, candidate_holdout_score=None,
                current_verification_score=None, candidate_verification_score=None,
                iterations=0, source_version=current.version, promoted_version=None,
            )
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=baseline_score,
                candidate_score=float("inf"),
                improvement_pct=0.0,
                num_iterations=0,
                optimizer_message="scipy not installed",
                rejection_reason=reason,
                dry_run=dry_run,
                holdout_result=baseline_result,
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
            self._record_run(
                dry_run=dry_run, accepted=False, reason=reason,
                optimizer_method=optimizer_method,
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=baseline_score, candidate_holdout_score=candidate_score,
                current_verification_score=None, candidate_verification_score=None,
                iterations=opt_result.nit, source_version=current.version, promoted_version=None,
            )
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
                dry_run=dry_run,
            )

        # Decode candidate weights
        candidate_weights = self._vector_to_weights(opt_result.x, base_weights=current)
        candidate_weights.backtest_score = candidate_score
        candidate_weights.backtest_date_range = f"{holdout_start}→{holdout_end}"

        # --- 4. Verify gate: candidate vs current on the SAME window ---
        logger.info(
            "Verification backtest (current weights) on %s → %s...",
            verification_start, verification_end,
        )
        manager.use_ensemble_weights(current)
        current_verify_result = self._runner.run(
            verification_start, verification_end,
            feature_df=feature_df, progress=progress,
        )
        current_verify_score = current_verify_result.weighted_score

        logger.info(
            "Verification backtest (candidate weights) on %s → %s...",
            verification_start, verification_end,
        )
        manager.use_ensemble_weights(candidate_weights)
        candidate_verify_result = self._runner.run(
            verification_start, verification_end,
            feature_df=feature_df, progress=progress,
        )
        candidate_verify_score = candidate_verify_result.weighted_score

        if not (np.isfinite(current_verify_score) and np.isfinite(candidate_verify_score)):
            # An invalid verification score is a FAILED GATE: never skip
            # verification and promote.
            if not np.isfinite(current_verify_score):
                reason = "invalid_verification: current verification score is not finite"
            else:
                reason = "invalid_verification: candidate verification score is not finite"
            logger.warning("VERIFY GATE FAILED: %s", reason)
            self._record_run(
                dry_run=dry_run, accepted=False, reason=reason,
                optimizer_method=optimizer_method,
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=baseline_score, candidate_holdout_score=candidate_score,
                current_verification_score=current_verify_score,
                candidate_verification_score=candidate_verify_score,
                iterations=opt_result.nit, source_version=current.version, promoted_version=None,
            )
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                improvement_pct=improvement_pct,
                num_iterations=opt_result.nit,
                optimizer_message=opt_result.message,
                holdout_result=baseline_result,
                verification_result=candidate_verify_result,
                rejection_reason=reason,
                dry_run=dry_run,
                current_verification_score=current_verify_score,
                candidate_verification_score=candidate_verify_score,
            )

        verify_degradation = candidate_verify_score - current_verify_score
        verify_degradation_pct = (
            (verify_degradation / current_verify_score * 100.0)
            if current_verify_score > 0 else 0.0
        )

        if verify_degradation > self.verification_margin * current_verify_score:
            reason = (
                f"Verification degradation {verify_degradation:.4f} "
                f"({verify_degradation_pct:.1f}%) > margin "
                f"{self.verification_margin * current_verify_score:.4f}"
            )
            logger.info("VERIFY GATE FAILED: %s", reason)
            self._record_run(
                dry_run=dry_run, accepted=False, reason=reason,
                optimizer_method=optimizer_method,
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=baseline_score, candidate_holdout_score=candidate_score,
                current_verification_score=current_verify_score,
                candidate_verification_score=candidate_verify_score,
                iterations=opt_result.nit, source_version=current.version, promoted_version=None,
            )
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                improvement_pct=improvement_pct,
                num_iterations=opt_result.nit,
                optimizer_message=opt_result.message,
                holdout_result=baseline_result,
                verification_result=candidate_verify_result,
                rejection_reason=reason,
                dry_run=dry_run,
                current_verification_score=current_verify_score,
                candidate_verification_score=candidate_verify_score,
            )

        # --- 5. Promote: exactly once, only after both gates pass ---
        candidate_weights.description = (
            f"Optimized: MAE {baseline_score:.4f}→{candidate_score:.4f} "
            f"({improvement_pct:+.1f}%) on {holdout_start}→{holdout_end}"
        )
        candidate_weights.optimizer_method = optimizer_method
        candidate_weights.accept_margin = self.accept_margin

        if dry_run:
            logger.info("DRY RUN: candidate accepted but nothing is saved or deployed")
            self._record_run(
                dry_run=True, accepted=True, reason="",
                optimizer_method=optimizer_method,
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=baseline_score, candidate_holdout_score=candidate_score,
                current_verification_score=current_verify_score,
                candidate_verification_score=candidate_verify_score,
                iterations=opt_result.nit, source_version=current.version, promoted_version=None,
            )
            return OptimizationResult(
                accepted=True,
                weights=candidate_weights,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                improvement_pct=improvement_pct,
                num_iterations=opt_result.nit,
                optimizer_message=opt_result.message,
                holdout_result=baseline_result,
                verification_result=candidate_verify_result,
                dry_run=True,
                deployed=False,
                current_verification_score=current_verify_score,
                candidate_verification_score=candidate_verify_score,
            )

        version = self._store.save(candidate_weights, set_current=True)
        logger.info("DEPLOYED v%d: ΔMAE = %.4f (%.2f%%)", version, improvement, improvement_pct)

        self._record_run(
            dry_run=False, accepted=True, reason="",
            optimizer_method=optimizer_method,
            holdout_start=holdout_start, holdout_end=holdout_end,
            verification_start=verification_start, verification_end=verification_end,
            current_holdout_score=baseline_score, candidate_holdout_score=candidate_score,
            current_verification_score=current_verify_score,
            candidate_verification_score=candidate_verify_score,
            iterations=opt_result.nit, source_version=current.version, promoted_version=version,
        )

        return OptimizationResult(
            accepted=True,
            weights=candidate_weights,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            improvement_pct=improvement_pct,
            num_iterations=opt_result.nit,
            optimizer_message=opt_result.message,
            holdout_result=baseline_result,
            verification_result=candidate_verify_result,
            dry_run=False,
            deployed=True,
            current_verification_score=current_verify_score,
            candidate_verification_score=candidate_verify_score,
        )

    # ------------------------------------------------------------------
    # Audit records
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe_score(value: Optional[float]) -> Optional[float]:
        """Return a JSON-serializable score (None for missing/non-finite)."""
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _record_run(
        self,
        *,
        dry_run: bool,
        accepted: bool,
        reason: str,
        optimizer_method: str,
        holdout_start: str,
        holdout_end: str,
        verification_start: str,
        verification_end: str,
        current_holdout_score: Optional[float],
        candidate_holdout_score: Optional[float],
        current_verification_score: Optional[float],
        candidate_verification_score: Optional[float],
        iterations: int,
        source_version: int,
        promoted_version: Optional[int],
    ) -> None:
        """Append one append-only audit record for a completed attempt.

        Records every completed attempt — accepted, rejected, and dry-run —
        in a log separate from promoted version history. Rejected and
        dry-run attempts always carry ``promoted_version=None``.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": bool(dry_run),
            "status": "accepted" if accepted else "rejected",
            "reason": reason or "",
            "optimizer_method": optimizer_method or "",
            "holdout_range": f"{holdout_start}→{holdout_end}",
            "verification_range": f"{verification_start}→{verification_end}",
            "current_holdout_score": self._json_safe_score(current_holdout_score),
            "candidate_holdout_score": self._json_safe_score(candidate_holdout_score),
            "current_verification_score": self._json_safe_score(current_verification_score),
            "candidate_verification_score": self._json_safe_score(candidate_verification_score),
            "iterations": int(iterations),
            "source_weight_version": int(source_version),
            "promoted_version": promoted_version,
        }
        self._store.record_run(record)
