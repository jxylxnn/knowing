"""Extract distribution parameters from quantile model outputs (P10/P50/P90).

Moves from point-estimates (Mean) to full distribution parameters
(Mean, Std, Skew, Zero-Prob) using already-trained quantile models,
without needing the Nexus model or its Copula Head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StatDistribution:
    """Distribution parameters for a single stat in a single player projection."""

    mean: float = 0.0
    std: float = 0.0
    skew: float = 0.0
    zero_prob: float = 0.0
    lambda_param: float = 0.0


class DistributionFitter:
    """Derives distribution parameters from CatBoost/Quantile predictions.

    Uses P10, P50, and P90 quantile predictions to reverse-engineer the
    shape of the distribution for each stat without requiring a full
    neural copula.
    """

    # P90 - P10 = 2.563 * Std for a normal distribution
    QUANTILE_SPREAD_FACTOR = 2.563

    # Stats that are approximately continuous with possible skew
    CONTINUOUS_STATS = frozenset({"PTS", "REB", "AST", "MIN"})

    # Stats that are zero-inflated count distributions
    COUNT_STATS = frozenset({"STL", "BLK", "TOV"})

    @staticmethod
    def fit_from_quantiles(
        p50: float,
        p10: float,
        p90: float,
        stat: str,
        historical_zero_rate: float = 0.0,
    ) -> StatDistribution:
        """Derive distribution parameters from quantile predictions.

        Args:
            p50: Predicted 50th percentile (median) from quantile model.
            p10: Predicted 10th percentile.
            p90: Predicted 90th percentile.
            stat: Target stat name (e.g. 'PTS', 'STL').
            historical_zero_rate: Rate of zero outcomes for count stats (0-1).

        Returns:
            StatDistribution with mean, std, skew, zero_prob, lambda_param.
        """
        std = max((p90 - p10) / DistributionFitter.QUANTILE_SPREAD_FACTOR, 0.1)
        mean = p50

        if stat.upper() in DistributionFitter.CONTINUOUS_STATS:
            # Estimate skew: if P90 is further from median than P10, right-skewed
            right_tail = p90 - p50
            left_tail = p50 - p10
            skew_val = (right_tail - left_tail) / std if std > 0 else 0.0
            return StatDistribution(mean=mean, std=std, skew=skew_val)

        if stat.upper() in DistributionFitter.COUNT_STATS:
            # Zero-Inflated Poisson (ZIP) approximation
            zero_prob = historical_zero_rate if p10 <= 0 else 0.0
            # Adjust lambda for the non-zero portion
            denom = 1 - zero_prob
            lambda_param = mean / denom if denom > 0 else mean
            return StatDistribution(
                mean=mean,
                std=std,
                zero_prob=zero_prob,
                lambda_param=max(lambda_param, 0.01),
            )

        return StatDistribution(mean=mean, std=std)
