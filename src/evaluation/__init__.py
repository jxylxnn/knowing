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
]
