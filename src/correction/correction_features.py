"""Correction feature builder for residual model training and runtime inference."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "STAT",
    "BASE_PREDICTION",
    "ACTUAL",
    "ERROR",
    "GAME_DATE",
    "PLAYER_ID",
]

OPTIONAL_FEATURE_COLUMNS = [
    "MINUTES_CONFIDENCE",
    "USAGE_TREND",
    "ROLLING_MINUTES",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "IS_HOME",
    "IS_PLAYOFF_GAME",
    "OPP_DEFENSE_RANK",
    "TEAM_PACE",
    "PLAYER_ARCHETYPE",
    "TEAM_ID",
    "OPPONENT",
    "MODEL_FOLD",
    "DATA_QUALITY",
]

DATA_QUALITY_MAP = {
    "FULL": 1.0,
    "PARTIAL": 0.6,
    "DEGRADED": 0.3,
    "MISSING": 0.0,
}

DEFAULT_DATA_QUALITY_SCORE = 0.5

# Default feature values used at runtime when historical error context is unavailable.
_RUNTIME_DEFAULTS: Dict[str, float] = {
    "BASE_PREDICTION": 0.0,
    "DATA_QUALITY_SCORE": 0.5,
    "RECENT_PLAYER_ERROR_MEAN": 0.0,
    "RECENT_PLAYER_ERROR_ABS_MEAN": 0.0,
    "RECENT_STAT_ERROR_MEAN": 0.0,
    "RECENT_STAT_ERROR_ABS_MEAN": 0.0,
    "MINUTES_CONFIDENCE": 0.0,
    "USAGE_TREND": 0.0,
    "ROLLING_MINUTES": 0.0,
    "REST_DAYS": 0.0,
    "IS_BACK_TO_BACK": 0,
    "IS_HOME": 0,
    "IS_PLAYOFF_GAME": 0,
    "OPP_DEFENSE_RANK": 0.0,
    "TEAM_PACE": 0.0,
    "PLAYER_ARCHETYPE": 0.0,
    "TEAM_ID": 0.0,
    "OPPONENT": 0.0,
    "MODEL_FOLD": 0.0,
}


class CorrectionFeatureBuilder:
    """Build features for the residual correction model.

    Rolling error features use ``shift(1)`` to prevent leakage from the
    current row's actual result.
    """

    ROLLING_WINDOW = 10
    ROLLING_MIN_PERIODS = 3

    def build(
        self,
        residual_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Build correction features from a residual DataFrame.

        Args:
            residual_df: DataFrame with at least the required columns.

        Returns:
            Tuple of (feature DataFrame, list of feature column names).
        """
        df = residual_df.copy()

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values(["PLAYER_ID", "STAT", "GAME_DATE"]).reset_index(drop=True)

        df["DATA_QUALITY_SCORE"] = df["DATA_QUALITY"].map(
            lambda v: DATA_QUALITY_MAP.get(
                str(v).upper() if pd.notna(v) else "",
                DEFAULT_DATA_QUALITY_SCORE,
            )
        ) if "DATA_QUALITY" in df.columns else DEFAULT_DATA_QUALITY_SCORE

        df["RECENT_PLAYER_ERROR_MEAN"] = (
            df.groupby(["PLAYER_ID", "STAT"])["ERROR"]
            .transform(
                lambda s: s.shift(1)
                .rolling(self.ROLLING_WINDOW, min_periods=self.ROLLING_MIN_PERIODS)
                .mean()
            )
        )
        df["RECENT_PLAYER_ERROR_ABS_MEAN"] = (
            df.groupby(["PLAYER_ID", "STAT"])["ERROR"]
            .transform(
                lambda s: s.abs().shift(1)
                .rolling(self.ROLLING_WINDOW, min_periods=self.ROLLING_MIN_PERIODS)
                .mean()
            )
        )
        df["RECENT_STAT_ERROR_MEAN"] = (
            df.groupby("STAT")["ERROR"]
            .transform(
                lambda s: s.shift(1)
                .rolling(self.ROLLING_WINDOW, min_periods=self.ROLLING_MIN_PERIODS)
                .mean()
            )
        )
        df["RECENT_STAT_ERROR_ABS_MEAN"] = (
            df.groupby("STAT")["ERROR"]
            .transform(
                lambda s: s.abs().shift(1)
                .rolling(self.ROLLING_WINDOW, min_periods=self.ROLLING_MIN_PERIODS)
                .mean()
            )
        )

        feature_cols: List[str] = [
            "BASE_PREDICTION",
            "DATA_QUALITY_SCORE",
            "RECENT_PLAYER_ERROR_MEAN",
            "RECENT_PLAYER_ERROR_ABS_MEAN",
            "RECENT_STAT_ERROR_MEAN",
            "RECENT_STAT_ERROR_ABS_MEAN",
        ]

        for col in OPTIONAL_FEATURE_COLUMNS:
            if col in df.columns:
                if col == "DATA_QUALITY":
                    continue
                feature_cols.append(col)
            else:
                if col in ("IS_BACK_TO_BACK", "IS_HOME", "IS_PLAYOFF_GAME"):
                    df[col] = 0
                elif col == "PLAYER_ARCHETYPE":
                    df[col] = "UNKNOWN"
                else:
                    df[col] = 0.0
                feature_cols.append(col)

        for col in feature_cols:
            if df[col].dtype == object:
                df[col] = df[col].astype("category").cat.codes.astype(float)
            df[col] = df[col].fillna(0.0)

        logger.info(
            "Built %d correction features for %d rows", len(feature_cols), len(df)
        )
        return df, feature_cols

    def build_runtime_row(
        self,
        stat: str,
        base_prediction: float,
        context_row: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame for runtime correction prediction.

        At inference time we do not have the actual outcome, so rolling error
        features default to ``0.0``.  Optional context columns from
        *context_row* are pulled through when available.

        Args:
            stat: Target stat (e.g. ``"PTS"``).
            base_prediction: The base model's predicted value.
            context_row: Optional single-row DataFrame with player/game context.
            feature_cols: Ordered list of feature column names the residual
                model expects.  When ``None``, the default feature set is used.

        Returns:
            Single-row DataFrame ready for ``ResidualCorrectionModel.predict_correction``.
        """
        if feature_cols is None:
            feature_cols = list(_RUNTIME_DEFAULTS.keys())

        row: Dict[str, float] = {}
        for col in feature_cols:
            if col == "BASE_PREDICTION":
                row[col] = float(base_prediction)
            elif context_row is not None and col in context_row.columns:
                val = context_row[col].iloc[0] if len(context_row) > 0 else 0.0
                if pd.isna(val):
                    row[col] = _RUNTIME_DEFAULTS.get(col, 0.0)
                else:
                    row[col] = float(val) if not isinstance(val, str) else float(
                        DATA_QUALITY_MAP.get(str(val).upper(), DEFAULT_DATA_QUALITY_SCORE)
                    )
            else:
                row[col] = _RUNTIME_DEFAULTS.get(col, 0.0)

        return pd.DataFrame([row], columns=feature_cols)
