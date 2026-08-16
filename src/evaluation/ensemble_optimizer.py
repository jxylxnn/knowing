"""Self-optimizing ensemble weight tuner.

Takes a backtest runner, evaluates candidate blend weights against a holdout
set of recently completed games, and uses scipy.optimize to find weights that
minimize prediction error.  Accept/verify gates prevent regressions from being
deployed.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics import BacktestResult, compute_target_metrics
from src.evaluation.weight_store import EnsembleWeights, TargetBlend, WeightStore

logger = logging.getLogger(__name__)


@dataclass
class ParameterLayout:
    """Component-derived optimizer parameter layout.

    Built AFTER models are loaded, so the tunable parameter set always
    matches the available model components:

    - Per-target CatBoost/Transformer ratios and intercepts exist only when a
      usable Transformer artifact is loaded.
    - A single global CatBoost/MAE-companion blend parameter exists only when
      at least one MAE companion model is loaded. It only affects targets
      that actually have a companion; other targets keep their base value.
    - With neither component, the layout is empty and optimization is
      rejected up front with ``no_tunable_components``.

    A six-target Transformer bundle without MAE companions therefore has
    12 parameters (6 ratios + 6 intercepts); it has 13 only when the global
    MAE-companion blend is also usable. Neither number is hard-coded.
    """

    names: List[str]
    bounds: List[Tuple[float, float]]
    x0: np.ndarray
    has_transformer: bool
    ratio_index: Dict[str, int] = field(default_factory=dict)
    intercept_index: Dict[str, int] = field(default_factory=dict)
    mae_blend_index: Optional[int] = None
    mae_targets: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.names)


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

    # ------------------------------------------------------------------
    # Parameter encoding
    # ------------------------------------------------------------------

    def _weights_to_vector(
        self, weights: EnsembleWeights, layout: ParameterLayout
    ) -> np.ndarray:
        """Encode EnsembleWeights into the flat parameter vector of a layout.

        Only parameters present in the layout are encoded: per-target
        CatBoost ratios and intercepts when a Transformer is loaded, and the
        global CatBoost/MAE blend when MAE companions are loaded.
        """
        vec = np.zeros(len(layout), dtype=float)
        for target in self._targets:
            tb = weights.per_target.get(target, TargetBlend())
            if target in layout.ratio_index:
                vec[layout.ratio_index[target]] = tb.catboost  # transformer = 1 - catboost
            if target in layout.intercept_index:
                vec[layout.intercept_index[target]] = tb.intercept
        if layout.mae_blend_index is not None:
            vec[layout.mae_blend_index] = weights.catboost_mae_blend
        return vec

    def _vector_to_weights(
        self,
        vec: np.ndarray,
        layout: ParameterLayout,
        base_weights: Optional[EnsembleWeights] = None,
    ) -> EnsembleWeights:
        """Decode a flat parameter vector into EnsembleWeights.

        The CatBoost fraction is clamped to [0.1, 0.9]; the Transformer
        fraction is inferred as 1.0 - catboost (so they sum to 1.0).

        When no Transformer is loaded the layout has no ratio/intercept
        parameters, so decoded targets keep catboost=1.0, transformer=0.0,
        and intercept=0.0 — a Transformer contribution is never manufactured.

        The global CatBoost/MAE blend parameter only affects targets that
        have a loaded MAE companion; other targets keep their base value.

        Args:
            vec: Flat parameter vector.
            layout: The component-derived ParameterLayout for this manager.
            base_weights: Optional base to copy metadata from.

        Returns:
            New EnsembleWeights with decoded parameters.
        """
        vec = np.asarray(vec, dtype=float)
        if vec.shape[0] != len(layout):
            raise ValueError(
                f"parameter vector length {vec.shape[0]} does not match "
                f"layout dimension {len(layout)}"
            )

        base_blend = (
            base_weights.catboost_mae_blend if base_weights is not None else 0.7
        )
        if layout.mae_blend_index is not None:
            global_mae_blend = float(np.clip(vec[layout.mae_blend_index], *self.MAE_BLEND_BOUNDS))
        else:
            global_mae_blend = base_blend

        per_target: Dict[str, TargetBlend] = {}
        for target in self._targets:
            base_tb = (
                base_weights.per_target.get(target)
                if base_weights is not None else None
            )
            if layout.has_transformer:
                cb = float(np.clip(vec[layout.ratio_index[target]], *self.BLEND_BOUNDS))
                tx = 1.0 - cb  # sum-to-1 constraint
                intercept = float(
                    np.clip(vec[layout.intercept_index[target]], *self.INTERCEPT_BOUNDS)
                )
            else:
                cb, tx, intercept = 1.0, 0.0, 0.0
            if layout.mae_blend_index is not None and target in layout.mae_targets:
                mae_blend = global_mae_blend
            else:
                mae_blend = base_tb.catboost_mae_blend if base_tb is not None else 0.7
            per_target[target] = TargetBlend(
                catboost=cb,
                transformer=tx,
                intercept=intercept,
                catboost_mae_blend=mae_blend,
            )

        weights = EnsembleWeights(
            per_target=per_target,
            catboost_mae_blend=global_mae_blend,
            created_at=datetime.now().isoformat(),
            description="Optimizer candidate",
        )

        if base_weights is not None:
            weights.parent_version = base_weights.version

        return weights

    # ------------------------------------------------------------------
    # Component-derived layout
    # ------------------------------------------------------------------

    def _build_parameter_layout(self) -> ParameterLayout:
        """Derive the tunable parameter layout from the loaded components.

        Reads the manager's loaded models: ``models`` (CatBoost),
        ``transformer_model``, and ``catboost_mae_models`` (MAE companions).
        """
        manager = self._runner._manager
        has_transformer = getattr(manager, "transformer_model", None) is not None
        mae_models = getattr(manager, "catboost_mae_models", {}) or {}
        mae_targets = tuple(
            target for target in self._targets
            if mae_models.get(target) is not None
        )

        names: List[str] = []
        bounds: List[Tuple[float, float]] = []
        ratio_index: Dict[str, int] = {}
        intercept_index: Dict[str, int] = {}

        if has_transformer:
            for target in self._targets:
                ratio_index[target] = len(names)
                names.append(f"catboost_ratio_{target}")
                bounds.append(self.BLEND_BOUNDS)
                intercept_index[target] = len(names)
                names.append(f"intercept_{target}")
                bounds.append(self.INTERCEPT_BOUNDS)

        mae_blend_index: Optional[int] = None
        if mae_targets:
            mae_blend_index = len(names)
            names.append("catboost_mae_blend")
            bounds.append(self.MAE_BLEND_BOUNDS)

        return ParameterLayout(
            names=names,
            bounds=bounds,
            x0=np.zeros(len(names), dtype=float),
            has_transformer=has_transformer,
            ratio_index=ratio_index,
            intercept_index=intercept_index,
            mae_blend_index=mae_blend_index,
            mae_targets=mae_targets,
        )

    def _default_weights_for_components(self, layout: ParameterLayout) -> EnsembleWeights:
        """Build default weights from the loaded components.

        A CatBoost-only manager gets catboost=1.0, transformer=0.0, and
        intercept=0.0 — never the legacy ``default_for_targets()``
        Transformer weight of 0.3.
        """
        if layout.has_transformer:
            catboost, transformer = 0.7, 0.3
        else:
            catboost, transformer = 1.0, 0.0
        per_target = {
            target: TargetBlend(
                catboost=catboost,
                transformer=transformer,
                intercept=0.0,
                catboost_mae_blend=0.7,
            )
            for target in self._targets
        }
        return EnsembleWeights(
            version=1,
            created_at=datetime.now().isoformat(),
            description="Component-derived default weights",
            per_target=per_target,
            catboost_mae_blend=0.7,
        )

    def validate_candidate_weights(self, weights: EnsembleWeights) -> None:
        """Validate candidate weights against the loaded components.

        Raises ValueError when a candidate requires a component that is not
        loaded (e.g. a non-zero Transformer weight without a Transformer
        artifact). Called before every candidate backtest.
        """
        manager = self._runner._manager
        has_transformer = getattr(manager, "transformer_model", None) is not None
        if has_transformer:
            return
        for target, tb in weights.per_target.items():
            if abs(tb.transformer) > 1e-9:
                raise ValueError(
                    f"target {target}: transformer weight {tb.transformer} requires "
                    "a Transformer model that is not loaded"
                )

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _build_objective(
        self,
        holdout_start: str,
        holdout_end: str,
        feature_df,
        baseline_score: float,
        layout: ParameterLayout,
    ) -> Callable[[np.ndarray], float]:
        """Build the objective function for scipy.optimize.

        Returns a callable that takes a parameter vector and returns the
        weighted MAE on the holdout set.  Lower is better.

        The feature_df is captured in the closure to avoid recomputing
        features on every iteration. Candidate weights are validated against
        the loaded components before every backtest; invalid candidates
        score ``inf``.
        """

        def objective(vec: np.ndarray) -> float:
            # Decode to weights
            candidate = self._vector_to_weights(vec, layout)

            # Validate against loaded components before the backtest.
            try:
                self.validate_candidate_weights(candidate)
            except ValueError as exc:
                logger.warning("Skipping invalid candidate: %s", exc)
                return float("inf")

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

        # Ensure models are loaded FIRST: the parameter layout is derived
        # from the loaded components.
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

        # Derive the tunable layout from the loaded components.
        layout = self._build_parameter_layout()

        # Load current weights; when none exist, construct defaults from the
        # loaded components (never the legacy 0.3 Transformer default for a
        # CatBoost-only manager).
        current = self._store.load_current()
        if current is None:
            current = self._default_weights_for_components(layout)
            logger.info("No current weights found; using component-derived defaults")

        # Apply the starting weights. They are restored on every rejection,
        # exception, and dry-run exit (see finally below).
        manager.use_ensemble_weights(current)

        if len(layout) == 0:
            # Nothing is tunable for the loaded components. Reject cleanly
            # BEFORE importing/calling scipy; no save, no history, but the
            # audit record is still appended.
            reason = "no_tunable_components"
            self._record_run(
                dry_run=dry_run, accepted=False, reason=reason,
                optimizer_method="",
                holdout_start=holdout_start, holdout_end=holdout_end,
                verification_start=verification_start, verification_end=verification_end,
                current_holdout_score=None, candidate_holdout_score=None,
                current_verification_score=None, candidate_verification_score=None,
                iterations=0, source_version=current.version, promoted_version=None,
                parameter_names=[], parameter_count=0,
            )
            return OptimizationResult(
                accepted=False,
                weights=current,
                baseline_score=float("inf"),
                candidate_score=float("inf"),
                improvement_pct=0.0,
                num_iterations=0,
                optimizer_message="No tunable parameters for the loaded model components",
                rejection_reason=reason,
                dry_run=dry_run,
            )

        result: Optional[OptimizationResult] = None
        try:
            result = self._optimize_inner(
                manager=manager,
                current=current,
                feature_df=feature_df,
                layout=layout,
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
        layout: ParameterLayout,
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
                parameter_names=list(layout.names), parameter_count=len(layout),
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
        x0 = self._weights_to_vector(current, layout)
        objective_fn = self._build_objective(
            holdout_start, holdout_end, feature_df, baseline_score, layout,
        )

        logger.info(
            "Starting Nelder-Mead optimization over %d parameters (%d iterations max)...",
            len(layout), self.max_iterations,
        )

        try:
            from scipy.optimize import minimize

            opt_result = minimize(
                objective_fn,
                x0,
                method="Nelder-Mead",
                bounds=list(layout.bounds),
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
                parameter_names=list(layout.names), parameter_count=len(layout),
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
                parameter_names=list(layout.names), parameter_count=len(layout),
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

        # Decode candidate weights and validate against the loaded components
        # before the verification backtest.
        candidate_weights = self._vector_to_weights(
            opt_result.x, layout, base_weights=current,
        )
        self.validate_candidate_weights(candidate_weights)
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
                parameter_names=list(layout.names), parameter_count=len(layout),
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
                parameter_names=list(layout.names), parameter_count=len(layout),
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
                parameter_names=list(layout.names), parameter_count=len(layout),
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
            parameter_names=list(layout.names), parameter_count=len(layout),
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
        parameter_names: List[str],
        parameter_count: int,
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
            "parameter_names": list(parameter_names),
            "parameter_count": int(parameter_count),
        }
        self._store.record_run(record)
