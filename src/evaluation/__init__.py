"""Evaluation module for backtesting, optimization, drift detection,
and smart feature selection."""

from src.evaluation.metrics import BacktestResult, TargetMetrics
from src.evaluation.backtest_runner import BacktestRunner
from src.evaluation.feature_group_ablation import (
    AblationReport,
    FeatureGroupAblator,
    GroupScore,
)
from src.evaluation.shadow_feature_filter import (
    SHADOW_COLUMNS,
    ShadowFeatureFilter,
    ShadowFilterResult,
)
from src.evaluation.smart_feature_selector import (
    ProfileConfig,
    SelectionManifest,
    SelectorConfig,
    SmartFeatureSelector,
    TargetSelection,
    load_manifest,
)
from src.evaluation.continual_learning import (
    PromotionDecision,
    PromotionPolicy,
    evaluate_promotion,
    paired_bootstrap_improvement,
)
from src.evaluation.prediction_ledger import PredictionLedger

__all__ = [
    # Backtest
    "BacktestResult",
    "TargetMetrics",
    "BacktestRunner",
    # Smart feature selection
    "AblationReport",
    "FeatureGroupAblator",
    "GroupScore",
    "SHADOW_COLUMNS",
    "ShadowFeatureFilter",
    "ShadowFilterResult",
    "ProfileConfig",
    "SelectionManifest",
    "SelectorConfig",
    "SmartFeatureSelector",
    "TargetSelection",
    "load_manifest",
    "PromotionDecision",
    "PromotionPolicy",
    "evaluate_promotion",
    "paired_bootstrap_improvement",
    "PredictionLedger",
]
