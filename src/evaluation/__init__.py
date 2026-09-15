"""Leak-safe Model v2 evaluation and promotion APIs."""

from src.evaluation.baselines import TARGETS, add_lagged_baselines
from src.evaluation.folds import (
    FoldPartitions,
    RollingOriginFold,
    partition_frame,
    rolling_origin_folds,
)
from src.evaluation.promotion import (
    PROMOTION_EVIDENCE_SCHEMA_VERSION,
    PromotionDecision,
    PromotionThresholds,
    REQUIRED_CONTRACT_FLAGS,
    REQUIRED_TARGETS,
    evaluate_v2_promotion,
)
from src.evaluation.replay import ReplayScore, score_replay
from src.evaluation.significance import (
    paired_game_bootstrap,
    paired_game_bootstrap_many,
)

__all__ = [
    "TARGETS",
    "FoldPartitions",
    "PromotionDecision",
    "PromotionThresholds",
    "PROMOTION_EVIDENCE_SCHEMA_VERSION",
    "REQUIRED_CONTRACT_FLAGS",
    "REQUIRED_TARGETS",
    "ReplayScore",
    "RollingOriginFold",
    "add_lagged_baselines",
    "evaluate_v2_promotion",
    "paired_game_bootstrap",
    "paired_game_bootstrap_many",
    "partition_frame",
    "rolling_origin_folds",
    "score_replay",
]
