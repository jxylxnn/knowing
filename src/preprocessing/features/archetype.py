"""Player archetype and similarity feature group."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.features.base import FeatureContext, FeatureDiagnostics, FeatureGroup


class PlayerArchetypeFeatureGroup(FeatureGroup):
    """Deterministic player-style buckets and soft similarity features.

    The group consumes only past-only rolling/context features produced by the
    earlier preprocessing groups. It emits one hard archetype label plus soft
    similarity scores to a small set of stable playstyle templates.
    """

    ARCHETYPE_NAMES: Tuple[str, ...] = (
        "playmaker",
        "shot_creator",
        "rebound_big",
        "three_and_d_wing",
        "rim_runner",
        "bench_scorer",
    )

    PROFILE_FEATURES: Tuple[str, ...] = (
        "minutes",
        "usage",
        "points_rate",
        "rebounds_rate",
        "assists_rate",
        "steals_rate",
        "blocks_rate",
        "turnovers_rate",
        "three_point_attempt_rate",
        "field_goal_efficiency",
        "free_throw_rate",
        "pace_adjusted_production",
        "on_ball_index",
        "off_ball_index",
    )

    PROFILE_PRIORS: Dict[str, float] = {
        "minutes": 24.0,
        "usage": 0.18,
        "points_rate": 10.0 / 24.0,
        "rebounds_rate": 4.5 / 24.0,
        "assists_rate": 2.5 / 24.0,
        "steals_rate": 0.8 / 24.0,
        "blocks_rate": 0.6 / 24.0,
        "turnovers_rate": 1.5 / 24.0,
        "three_point_attempt_rate": 4.0 / 9.0,
        "field_goal_efficiency": 0.52,
        "free_throw_rate": 3.0 / 9.0,
        "pace_adjusted_production": 1.0,
        "on_ball_index": 0.22,
        "off_ball_index": 0.24,
    }

    FEATURE_WEIGHTS: Dict[str, float] = {
        "minutes": 0.80,
        "usage": 1.30,
        "points_rate": 1.35,
        "rebounds_rate": 1.15,
        "assists_rate": 1.35,
        "steals_rate": 0.75,
        "blocks_rate": 0.75,
        "turnovers_rate": 0.85,
        "three_point_attempt_rate": 1.00,
        "field_goal_efficiency": 0.90,
        "free_throw_rate": 0.70,
        "pace_adjusted_production": 1.00,
        "on_ball_index": 1.25,
        "off_ball_index": 1.00,
    }

    ARCHETYPE_PROFILES: Dict[str, Dict[str, float]] = {
        "playmaker": {
            "minutes": 1.10,
            "usage": 1.25,
            "points_rate": 1.00,
            "rebounds_rate": 0.75,
            "assists_rate": 2.40,
            "steals_rate": 1.00,
            "blocks_rate": 0.60,
            "turnovers_rate": 1.35,
            "three_point_attempt_rate": 0.80,
            "field_goal_efficiency": 0.98,
            "free_throw_rate": 1.10,
            "pace_adjusted_production": 1.05,
            "on_ball_index": 1.85,
            "off_ball_index": 0.75,
        },
        "shot_creator": {
            "minutes": 1.00,
            "usage": 1.40,
            "points_rate": 1.45,
            "rebounds_rate": 0.80,
            "assists_rate": 0.85,
            "steals_rate": 0.95,
            "blocks_rate": 0.55,
            "turnovers_rate": 1.20,
            "three_point_attempt_rate": 1.45,
            "field_goal_efficiency": 0.98,
            "free_throw_rate": 1.20,
            "pace_adjusted_production": 1.25,
            "on_ball_index": 1.20,
            "off_ball_index": 0.80,
        },
        "rebound_big": {
            "minutes": 1.05,
            "usage": 0.80,
            "points_rate": 0.85,
            "rebounds_rate": 1.95,
            "assists_rate": 0.55,
            "steals_rate": 0.85,
            "blocks_rate": 1.80,
            "turnovers_rate": 0.75,
            "three_point_attempt_rate": 0.20,
            "field_goal_efficiency": 1.02,
            "free_throw_rate": 0.85,
            "pace_adjusted_production": 0.95,
            "on_ball_index": 0.45,
            "off_ball_index": 1.20,
        },
        "three_and_d_wing": {
            "minutes": 1.00,
            "usage": 0.90,
            "points_rate": 0.95,
            "rebounds_rate": 0.85,
            "assists_rate": 0.80,
            "steals_rate": 1.45,
            "blocks_rate": 0.95,
            "turnovers_rate": 0.70,
            "three_point_attempt_rate": 1.60,
            "field_goal_efficiency": 0.95,
            "free_throw_rate": 0.75,
            "pace_adjusted_production": 0.90,
            "on_ball_index": 0.75,
            "off_ball_index": 1.35,
        },
        "rim_runner": {
            "minutes": 1.00,
            "usage": 0.95,
            "points_rate": 1.00,
            "rebounds_rate": 1.45,
            "assists_rate": 0.60,
            "steals_rate": 0.80,
            "blocks_rate": 1.35,
            "turnovers_rate": 0.80,
            "three_point_attempt_rate": 0.15,
            "field_goal_efficiency": 1.08,
            "free_throw_rate": 1.35,
            "pace_adjusted_production": 1.00,
            "on_ball_index": 0.55,
            "off_ball_index": 1.05,
        },
        "bench_scorer": {
            "minutes": 0.70,
            "usage": 1.10,
            "points_rate": 1.25,
            "rebounds_rate": 0.75,
            "assists_rate": 0.85,
            "steals_rate": 0.90,
            "blocks_rate": 0.55,
            "turnovers_rate": 1.05,
            "three_point_attempt_rate": 1.15,
            "field_goal_efficiency": 0.92,
            "free_throw_rate": 1.05,
            "pace_adjusted_production": 1.10,
            "on_ball_index": 0.95,
            "off_ball_index": 0.90,
        },
    }

    def __init__(self, min_history_window: int = 10):
        self.min_history_window = min_history_window

    @property
    def name(self) -> str:
        return "archetype"

    @property
    def required_columns(self) -> List[str]:
        return ["PLAYER_ID"]

    @property
    def optional_columns(self) -> List[str]:
        return [
            "ROLL_MIN_AVG_10",
            "ROLL_MIN_AVG_20",
            "ROLL_PTS_PER_MIN_10",
            "ROLL_PTS_PER_MIN_20",
            "ROLL_REB_PER_MIN_10",
            "ROLL_REB_PER_MIN_20",
            "ROLL_AST_PER_MIN_10",
            "ROLL_AST_PER_MIN_20",
            "ROLL_STL_AVG_10",
            "ROLL_STL_AVG_20",
            "ROLL_BLK_AVG_10",
            "ROLL_BLK_AVG_20",
            "ROLL_TOV_AVG_10",
            "ROLL_TOV_AVG_20",
            "ROLL_3PT_FREQ_10",
            "ROLL_3PT_FREQ_20",
            "ROLL_EFG_PCT_10",
            "ROLL_TS_PCT_10",
            "ROLL_FT_RATE_10",
            "PACE_ADJ_USAGE",
            "ROLL_USG_PCT_10",
            "RAW_USAGE",
            "TEAM_PACE_10",
            "PACE_FACTOR",
            "RAW_PTS_SHARE",
            "RAW_REB_OPPORTUNITY",
        ]

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        return [
            c
            for c in df.columns
            if c.startswith("ARCHETYPE_") or c.startswith("SIMILARITY_TO_")
        ]

    def _select_series(
        self,
        df: pd.DataFrame,
        candidates: Sequence[str],
        fallback: float,
    ) -> pd.Series:
        for col in candidates:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series.astype(float)
        return pd.Series(float(fallback), index=df.index, dtype=float)

    def _derive_rate(
        self,
        df: pd.DataFrame,
        direct_candidates: Sequence[str],
        numerator_candidates: Sequence[str],
        denominator_candidates: Sequence[str],
        fallback: float,
    ) -> pd.Series:
        direct = self._select_series(df, direct_candidates, np.nan)
        if direct.notna().any():
            return direct.fillna(float(fallback)).astype(float)

        numerator = self._select_series(df, numerator_candidates, np.nan)
        denominator = self._select_series(df, denominator_candidates, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator.replace(0, np.nan)
        if ratio.notna().any():
            return ratio.fillna(float(fallback)).astype(float)
        return pd.Series(float(fallback), index=df.index, dtype=float)

    def _normalized(self, raw: pd.Series, prior: float) -> pd.Series:
        prior = float(prior) if prior else 1.0
        values = pd.to_numeric(raw, errors="coerce") / prior
        return values.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 4.0)

    def _build_raw_profile(
        self,
        df: pd.DataFrame,
        context: FeatureContext,
    ) -> Dict[str, pd.Series]:
        priors = context.league_priors
        eps = 1e-6

        minutes = self._select_series(
            df,
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20", "ROLL_MIN_AVG_5"],
            priors.get("MIN", 24.0),
        )
        usage = self._select_series(
            df,
            ["PACE_ADJ_USAGE", "ROLL_USG_PCT_10", "ROLL_USG_PCT_5", "RAW_USAGE"],
            priors.get("USAGE", 0.18),
        )
        points_rate = self._derive_rate(
            df,
            ["ROLL_PTS_PER_MIN_10", "ROLL_PTS_PER_MIN_20"],
            ["ROLL_PTS_AVG_10", "ROLL_PTS_AVG_20"],
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20"],
            priors.get("PTS", 10.0) / max(priors.get("MIN", 24.0), eps),
        )
        rebounds_rate = self._derive_rate(
            df,
            ["ROLL_REB_PER_MIN_10", "ROLL_REB_PER_MIN_20"],
            ["ROLL_REB_AVG_10", "ROLL_REB_AVG_20"],
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20"],
            priors.get("REB", 4.5) / max(priors.get("MIN", 24.0), eps),
        )
        assists_rate = self._derive_rate(
            df,
            ["ROLL_AST_PER_MIN_10", "ROLL_AST_PER_MIN_20"],
            ["ROLL_AST_AVG_10", "ROLL_AST_AVG_20"],
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20"],
            priors.get("AST", 2.5) / max(priors.get("MIN", 24.0), eps),
        )
        steals_rate = self._derive_rate(
            df,
            ["ROLL_STL_PER_MIN_10", "ROLL_STL_PER_MIN_20"],
            ["ROLL_STL_AVG_10", "ROLL_STL_AVG_20"],
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20"],
            priors.get("STL", 0.8) / max(priors.get("MIN", 24.0), eps),
        )
        blocks_rate = self._derive_rate(
            df,
            ["ROLL_BLK_PER_MIN_10", "ROLL_BLK_PER_MIN_20"],
            ["ROLL_BLK_AVG_10", "ROLL_BLK_AVG_20"],
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20"],
            priors.get("BLK", 0.6) / max(priors.get("MIN", 24.0), eps),
        )
        turnovers_rate = self._derive_rate(
            df,
            ["ROLL_TOV_PER_MIN_10", "ROLL_TOV_PER_MIN_20"],
            ["ROLL_TOV_AVG_10", "ROLL_TOV_AVG_20"],
            ["ROLL_MIN_AVG_10", "ROLL_MIN_AVG_20"],
            priors.get("TOV", 1.5) / max(priors.get("MIN", 24.0), eps),
        )
        three_point_rate = self._select_series(
            df,
            ["ROLL_3PT_FREQ_10", "ROLL_3PT_FREQ_20", "RAW_3PT_FREQ"],
            priors.get("FG3A", 4.0) / max(priors.get("FGA", 9.0), eps),
        )
        field_goal_efficiency = self._select_series(
            df,
            ["ROLL_EFG_PCT_10", "ROLL_TS_PCT_10"],
            priors.get("EFG_PCT", 0.52),
        )
        free_throw_rate = self._select_series(
            df,
            ["ROLL_FT_RATE_10", "ROLL_FT_RATE_5"],
            priors.get("FTA", 3.0) / max(priors.get("FGA", 9.0), eps),
        )

        pace_factor = self._select_series(df, ["PACE_FACTOR"], 1.0)
        if "TEAM_PACE_10" in df.columns:
            team_pace = pd.to_numeric(df["TEAM_PACE_10"], errors="coerce")
            pace_factor = team_pace / max(priors.get("TEAM_PACE", 100.0), eps)
            pace_factor = pace_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        pace_adjusted_production = points_rate * pace_factor
        assist_share = assists_rate / (points_rate + assists_rate + eps)
        on_ball_index = (0.60 * usage) + (0.30 * assist_share) + (0.10 * turnovers_rate)
        off_ball_index = (
            (0.55 * three_point_rate)
            + (0.25 * field_goal_efficiency)
            + (0.20 * (1.0 - assist_share))
        )

        return {
            "minutes": minutes,
            "usage": usage,
            "points_rate": points_rate,
            "rebounds_rate": rebounds_rate,
            "assists_rate": assists_rate,
            "steals_rate": steals_rate,
            "blocks_rate": blocks_rate,
            "turnovers_rate": turnovers_rate,
            "three_point_attempt_rate": three_point_rate,
            "field_goal_efficiency": field_goal_efficiency,
            "free_throw_rate": free_throw_rate,
            "pace_adjusted_production": pace_adjusted_production,
            "on_ball_index": on_ball_index,
            "off_ball_index": off_ball_index,
        }

    def _build_profile_matrix(
        self,
        df: pd.DataFrame,
        context: FeatureContext,
    ) -> pd.DataFrame:
        raw_profile = self._build_raw_profile(df, context)
        profile = pd.DataFrame(index=df.index)
        for feature in self.PROFILE_FEATURES:
            profile[feature] = self._normalized(
                raw_profile[feature], self.PROFILE_PRIORS[feature]
            )
        return profile

    def _score_archetypes(self, profile: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_order = list(self.PROFILE_FEATURES)
        weights = np.asarray(
            [self.FEATURE_WEIGHTS.get(feature, 1.0) for feature in feature_order],
            dtype=float,
        )
        prototype_matrix = np.asarray(
            [
                [self.ARCHETYPE_PROFILES[name][feature] for feature in feature_order]
                for name in self.ARCHETYPE_NAMES
            ],
            dtype=float,
        )

        profile_matrix = profile[feature_order].to_numpy(dtype=float)
        diff = profile_matrix[:, None, :] - prototype_matrix[None, :, :]
        weighted_distance = np.sqrt((diff**2 * weights[None, None, :]).sum(axis=2))
        similarity = np.exp(-weighted_distance)
        similarity = similarity / similarity.sum(axis=1, keepdims=True)
        return similarity, weighted_distance

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()

        self._check_columns(df, diagnostics)
        df = df.copy()
        context = context or FeatureContext()

        profile = self._build_profile_matrix(df, context)
        similarity, _ = self._score_archetypes(profile)

        top_idx = similarity.argmax(axis=1)
        top_scores = similarity[np.arange(len(df)), top_idx]
        sorted_scores = np.sort(similarity, axis=1)[:, ::-1]
        second_scores = sorted_scores[:, 1] if similarity.shape[1] > 1 else sorted_scores[:, 0]

        df["ARCHETYPE_ID"] = top_idx.astype(int)
        df["ARCHETYPE_CONFIDENCE"] = top_scores.astype(float)
        df["ARCHETYPE_SIMILARITY_PRIMARY"] = top_scores.astype(float)
        df["ARCHETYPE_SIMILARITY_SECONDARY"] = second_scores.astype(float)

        for idx, archetype in enumerate(self.ARCHETYPE_NAMES):
            df[f"SIMILARITY_TO_{archetype.upper()}"] = similarity[:, idx].astype(float)

        if diagnostics is not None:
            for col in [
                "ARCHETYPE_ID",
                "ARCHETYPE_CONFIDENCE",
                "ARCHETYPE_SIMILARITY_PRIMARY",
                "ARCHETYPE_SIMILARITY_SECONDARY",
            ]:
                diagnostics.record_imputation(col, int(pd.isna(df[col]).sum()))

        return df
