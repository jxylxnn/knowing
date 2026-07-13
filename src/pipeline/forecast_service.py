"""Single production prediction path for CLIs, simulation, and evaluation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.contracts.forecast import ForecastRequest, validate_forecast_frame
from src.models.base import PredictionResult
from src.utils.prediction_utils import FeatureSchema, FeatureSelector

logger = logging.getLogger(__name__)


class ForecastService:
    """Own the runtime model path and its compatibility fallbacks.

    ``ModelManager`` remains the model/artifact bridge.  All callers use this
    service for prediction so feature alignment, blending, quantiles, and
    fallback behavior have one public implementation.  A legacy
    ``TrainingPipeline`` can be supplied during migration; its model objects
    are handled by the same service methods.
    """

    DEFAULT_TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    FALLBACK_VALUES = {
        "PTS": 10.0, "REB": 4.5, "AST": 2.5,
        "STL": 0.8, "BLK": 0.6, "TOV": 1.5,
    }

    def __init__(
        self,
        manager: Any | None = None,
        *,
        model_backend: Any | None = None,
        config: Any | None = None,
        data_dir: str = "data",
        models_dir: str = "models",
        model_size: str = "M",
    ) -> None:
        if manager is not None and model_backend is not None:
            raise ValueError("Pass manager or model_backend, not both")
        self.backend = manager or model_backend
        if self.backend is None:
            from src.models.model_manager import ModelManager

            self.backend = ModelManager(
                data_dir=data_dir,
                models_dir=models_dir,
                model_size=model_size,
            )
        self.config = config
        self._loaded = bool(getattr(self.backend, "models", None))

    @property
    def targets(self) -> List[str]:
        targets = getattr(self.backend, "targets", None)
        if targets:
            return list(targets)
        targets = getattr(self.backend, "TARGETS", None)
        if targets:
            return list(targets)
        training = getattr(self.config, "training", None)
        configured = getattr(training, "targets", None) if training else None
        return list(configured or self.DEFAULT_TARGETS)

    def _ensure_loaded(self) -> None:
        if self._loaded and getattr(self.backend, "models", None):
            return
        loader = getattr(self.backend, "load_models", None)
        if loader is None:
            loader = getattr(self.backend, "_load_models", None)
        if loader is not None:
            loader()
        self._loaded = True

    def predict_player_stats(
        self,
        player_context_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        *,
        include_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Return the canonical runtime stat prediction dictionary."""
        if player_context_df is None or player_context_df.empty:
            return self._fallback_prediction(player_context_df)
        self._ensure_loaded()

        # ModelManager is the active backend and already owns residual and
        # interval calibration.  Keeping this one delegation here avoids a
        # second implementation in simulation/query code.
        method = getattr(self.backend, "predict_player_stats", None)
        if method is not None:
            return method(
                player_context_df,
                history_df=history_df,
                include_confidence=include_confidence,
            )

        return self._predict_legacy_pipeline(player_context_df, history_df)

    def predict_player_stats_batch(
        self,
        context_df: pd.DataFrame,
        histories_map: Optional[Dict[int, pd.DataFrame]] = None,
        *,
        include_confidence: bool = False,
    ) -> pd.DataFrame:
        if context_df is None or context_df.empty:
            return pd.DataFrame()
        rows = []
        for index in range(len(context_df)):
            row = context_df.iloc[[index]]
            player_id = row.iloc[0].get("PLAYER_ID", index)
            try:
                player_id = int(player_id)
            except (TypeError, ValueError):
                pass
            history = histories_map.get(player_id) if histories_map else None
            try:
                rows.append(
                    self.predict_player_stats(
                        row,
                        history_df=history,
                        include_confidence=include_confidence,
                    )
                )
            except Exception as exc:
                logger.warning("Prediction failed for player %s: %s", player_id, exc)
                rows.append(self._fallback_prediction(row))
        return pd.DataFrame(rows, index=context_df.index)

    def predict_player(
        self,
        player_context_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        if player_context_df is None or player_context_df.empty:
            raise ValueError("player_context_df must contain one player row")
        row = player_context_df.iloc[0]
        predictions = self.predict_player_stats(
            player_context_df,
            history_df=history_df,
            include_confidence=True,
        )
        uncertainties = {
            target: float(predictions[f"{target}_STD"])
            for target in self.targets
            if f"{target}_STD" in predictions and predictions[f"{target}_STD"] is not None
        }
        return PredictionResult(
            player_id=int(row.get("PLAYER_ID", 0)),
            player_name=str(row.get("PLAYER_NAME", "Unknown")),
            team=str(row.get("TEAM_ABBREVIATION", row.get("TEAM", "UNK"))),
            opponent=str(row.get("OPPONENT_ABBR", row.get("OPPONENT", "UNK"))),
            predictions={
                key: value for key, value in predictions.items()
                if not key.endswith("_STD")
            },
            uncertainties=uncertainties,
        )

    def predict_forecast_frame(
        self,
        request: ForecastRequest,
        contexts_df: pd.DataFrame,
        histories_map: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Create canonical long-form rows for a scheduled-game request."""
        if contexts_df is None or contexts_df.empty:
            return pd.DataFrame()
        predictions = self.predict_player_stats_batch(
            contexts_df,
            histories_map=histories_map,
            include_confidence=True,
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []
        for idx, context in contexts_df.iterrows():
            predicted = predictions.loc[idx] if idx in predictions.index else pd.Series(dtype=float)
            player_id = context.get("PLAYER_ID", idx)
            team_id = context.get("TEAM_ID", context.get("TEAM_ABBREVIATION", ""))
            opponent_id = context.get("OPPONENT_ID", context.get("OPPONENT_ABBR", ""))
            play_prob = _probability(context.get("PLAY_PROB", context.get("P_PLAY", 1.0)))
            minutes = _number(
                context.get("EXPECTED_MINUTES", context.get("MINUTES_PRED", 0.0)),
                0.0,
            )
            min_std = _number(context.get("MINUTES_STD", 0.0), 0.0)
            min_p10, min_p50, min_p90 = _bounds(minutes, min_std, nonnegative=True)
            quality = str(context.get("DATA_QUALITY", "FULL"))
            for stat in self.targets:
                mean = _number(predicted.get(stat, self.FALLBACK_VALUES.get(stat, 0.0)))
                std = _number(predicted.get(f"{stat}_STD", 0.0), 0.0)
                p10 = _number(predicted.get(f"{stat}_P10", mean - 1.28155 * std), mean)
                p25 = _number(predicted.get(f"{stat}_P25", mean - 0.67449 * std), mean)
                p50 = _number(predicted.get(f"{stat}_P50", mean), mean)
                p75 = _number(predicted.get(f"{stat}_P75", mean + 0.67449 * std), mean)
                p90 = _number(predicted.get(f"{stat}_P90", mean + 1.28155 * std), mean)
                bounds = sorted((max(0.0, p10), max(0.0, p25), max(0.0, p50), max(0.0, p75), max(0.0, p90)))
                rows.append({
                    "MODEL_BUNDLE_ID": request.model_bundle_id,
                    "SNAPSHOT_ID": request.source_snapshot_id,
                    "GENERATED_AT": generated_at,
                    "FORECAST_CUTOFF": request.forecast_cutoff.isoformat(),
                    "GAME_ID": request.game_id,
                    "PLAYER_ID": player_id,
                    "TEAM_ID": team_id,
                    "OPPONENT_ID": opponent_id,
                    "HORIZON": request.horizon,
                    "PLAY_PROB": play_prob,
                    "EXPECTED_MINUTES": minutes,
                    "MIN_P10": min_p10,
                    "MIN_P50": min_p50,
                    "MIN_P90": min_p90,
                    "STAT": stat,
                    "MEAN": mean,
                    "P10": bounds[0],
                    "P25": bounds[1],
                    "P50": bounds[2],
                    "P75": bounds[3],
                    "P90": bounds[4],
                    "ZERO_PROB": _probability(predicted.get(f"{stat}_ZERO_PROB", 0.0)),
                    "DATA_QUALITY": quality,
                    "CALIBRATION_VERSION": str(getattr(self.backend, "model_version", "uncalibrated")),
                    "SCENARIO": request.scenario,
                })
        frame = pd.DataFrame(rows)
        validate_forecast_frame(frame)
        return frame

    def explain_player_stats(self, *args, **kwargs):
        method = getattr(self.backend, "explain_player_stats", None)
        if method is None:
            raise NotImplementedError("The selected backend does not support explanations")
        return method(*args, **kwargs)

    def _predict_legacy_pipeline(
        self,
        context_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        models = getattr(self.backend, "models", {}) or {}
        feature_cols = getattr(self.backend, "feature_cols", None)
        if not models or not feature_cols:
            return self._fallback_prediction(context_df)
        schema = getattr(self.backend, "feature_schema", None)
        selector = getattr(self.backend, "feature_selector", None)
        if selector is None:
            selector = FeatureSelector(self.targets)
        if schema is None:
            schema = FeatureSchema(
                feature_cols=list(feature_cols),
                categorical_cols=[
                    c for c in ("PLAYER_ID", "TEAM_ID", "OPPONENT_ID")
                    if c in feature_cols
                ],
            )
        X = selector.transform(context_df, schema, strict=False, fill_value=0.0)
        predictions: Dict[str, Any] = {}
        for target in self.targets:
            model = models.get(target)
            if model is None or not hasattr(model, "predict"):
                predictions[target] = self._fallback_value(context_df, target)
                continue
            try:
                predictions[target] = max(0.0, float(np.asarray(model.predict(X)).reshape(-1)[0]))
            except Exception:
                predictions[target] = self._fallback_value(context_df, target)
        return predictions

    def _fallback_prediction(self, context_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        return {
            target: self._fallback_value(context_df, target)
            for target in self.targets
        }

    def _fallback_value(self, context_df: Optional[pd.DataFrame], target: str) -> float:
        if context_df is not None and not context_df.empty:
            for column in (
                f"ROLL_{target}_AVG_5", f"ROLL_{target}_AVG_10",
                f"SEASON_{target}", f"{target}_EWMA_5",
            ):
                if column in context_df and pd.notna(context_df.iloc[0][column]):
                    value = float(context_df.iloc[0][column])
                    if value >= 0:
                        return value
        return float(self.FALLBACK_VALUES.get(target, 5.0))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _probability(value: Any) -> float:
    return float(np.clip(_number(value, 0.0), 0.0, 1.0))


def _bounds(mean: float, std: float, *, nonnegative: bool = False) -> tuple[float, float, float]:
    values = [mean - 1.28155 * std, mean, mean + 1.28155 * std]
    if nonnegative:
        values = [max(0.0, value) for value in values]
    return values[0], values[1], values[2]
