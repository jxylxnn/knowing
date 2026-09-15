"""Single production prediction path for CLIs, simulation, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.contracts.forecast import ForecastRequest, validate_forecast_frame
from src.models.base import PredictionResult


class ForecastService:
    """Own the supported Model v2 runtime prediction path.

    All callers use this service so artifact loading, prediction, sampling,
    and canonical forecast construction have one fail-closed implementation.
    """

    DEFAULT_TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

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
            from src.forecasting.baseline_backend import V2BaselineBackend
            from src.models.versioning import ModelBundleManifest, ModelVersionRegistry

            registry = ModelVersionRegistry(models_dir)
            champion = registry.read_manifest()
            if champion is None:
                raise FileNotFoundError(
                    "No Model v2 champion is configured. Train and promote an "
                    "immutable v2 bundle first."
                )
            active = registry.active_dir()
            bundle = ModelBundleManifest.load(
                active / ModelBundleManifest.FILE_NAME
            )
            if bundle.architecture != "v2":
                raise ValueError("The configured champion is not a Model v2 bundle")
            self.backend = V2BaselineBackend(active)
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
            raise ValueError("player_context_df must contain one player row")
        self._ensure_loaded()

        method = getattr(self.backend, "predict_player_stats", None)
        if not callable(method):
            raise TypeError(
                "Forecast backend must implement predict_player_stats"
            )
        predictions = method(
            player_context_df,
            history_df=history_df,
            include_confidence=include_confidence,
        )
        if not isinstance(predictions, Mapping):
            raise TypeError("Forecast backend predictions must be a mapping")

        validated = dict(predictions)
        for target in self.targets:
            validated[target] = _required_prediction_value(predictions, target)
        return validated

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
                raise RuntimeError(
                    f"Prediction failed for player {player_id}: {exc}"
                ) from exc
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
        *,
        simulations: int = 1000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create canonical long-form rows for a scheduled-game request."""
        if contexts_df is None or contexts_df.empty:
            return pd.DataFrame()
        if hasattr(self.backend, "prepare_sampling_frame"):
            return self.predict_game_distribution(
                request, contexts_df, simulations=simulations, seed=seed
            ).forecasts
        prepare = getattr(self.backend, "prepare_contexts", None)
        if prepare is not None:
            contexts_df = prepare(contexts_df)
        predictions = self.predict_player_stats_batch(
            contexts_df, histories_map=histories_map, include_confidence=True,
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []
        for idx, context in contexts_df.iterrows():
            predicted = predictions.loc[idx] if idx in predictions.index else pd.Series(dtype=float)
            player_id = context.get("PLAYER_ID", idx)
            team_id = context.get("TEAM_ID", context.get("TEAM_ABBREVIATION", ""))
            opponent_id = context.get("OPPONENT_ID", context.get("OPPONENT_ABBR", ""))
            p_active = _probability(context.get("P_ACTIVE", 1.0))
            p_play_given_active = _probability(
                context.get("P_PLAY_GIVEN_ACTIVE", context.get("PLAY_PROB", context.get("P_PLAY", 1.0)))
            )
            play_prob = _probability(p_active * p_play_given_active)
            minutes = _number(
                context.get("EXPECTED_MINUTES", context.get("MINUTES_PRED", 0.0)),
                0.0,
            )
            min_std = _number(context.get("MINUTES_STD", 0.0), 0.0)
            min_p10, min_p50, min_p90 = _bounds(minutes, min_std, nonnegative=True)
            quality = str(context.get("DATA_QUALITY", "FULL"))
            for stat in self.targets:
                conditional_mean = _required_prediction_value(predicted, stat)
                conditional_std = max(
                    0.0, _number(predicted.get(f"{stat}_STD", 0.0), 0.0)
                )
                mean = play_prob * conditional_mean
                bounds = _mixture_quantiles(
                    play_prob,
                    conditional_mean,
                    conditional_std,
                    probabilities=(0.10, 0.25, 0.50, 0.75, 0.90),
                )
                conditional_zero = _probability(
                    predicted.get(f"{stat}_ZERO_PROB", 0.0)
                )
                rows.append({
                    "REQUEST_ID": request.request_id,
                    "MODEL_BUNDLE_ID": request.model_bundle_id,
                    "SOURCE_SNAPSHOT_ID": request.source_snapshot_id,
                    "GENERATED_AT": generated_at,
                    "FORECAST_CUTOFF": request.forecast_cutoff.isoformat(),
                    "GAME_ID": request.game_id,
                    "GAME_DATE": request.game_date.isoformat(),
                    "SCHEDULE_VERSION": request.schedule_version,
                    "PLAYER_ID": player_id,
                    "TEAM_ID": team_id,
                    "OPPONENT_ID": opponent_id,
                    "HORIZON": request.horizon,
                    "P_ACTIVE": p_active,
                    "P_PLAY_GIVEN_ACTIVE": p_play_given_active,
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
                    "ZERO_PROB": _probability(
                        (1.0 - play_prob) + play_prob * conditional_zero
                    ),
                    "DATA_QUALITY": quality,
                    "CALIBRATION_VERSION": str(getattr(self.backend, "model_version", "uncalibrated")),
                    "SCENARIO": request.scenario,
                })
        frame = pd.DataFrame(rows)
        validate_forecast_frame(frame)
        return frame

    def predict_game_distribution(
        self, request: ForecastRequest, contexts: pd.DataFrame,
        *, simulations: int = 1000, seed: int = 42,
    ):
        """Sample once and return the exact law used for all consumer outputs."""
        from src.forecasting.distribution import summarize_game
        from src.simulation.joint_sampler import sample_game

        if request.model_bundle_id != self.backend.model_version:
            raise ValueError("Request model bundle does not match loaded backend")
        if request.scenario == "official" and not getattr(self.backend, "supports_official", False):
            raise ValueError(
                "Regulation-only or unqualified baseline cannot produce official full-game forecasts"
            )
        prepared = self.backend.prepare_sampling_frame(contexts)
        overtime = getattr(self.backend, "overtime_probabilities", None)
        samples = sample_game(
            prepared, targets=tuple(self.targets), simulations=simulations,
            seed=seed, overtime_probabilities=overtime,
        )
        return summarize_game(
            request, prepared, samples, targets=self.targets, seed=seed,
            simulations=simulations, full_game=overtime is not None,
        )

    def explain_player_stats(self, *args, **kwargs):
        method = getattr(self.backend, "explain_player_stats", None)
        if method is None:
            raise NotImplementedError("The selected backend does not support explanations")
        return method(*args, **kwargs)


def _required_prediction_value(
    predictions: Mapping[str, Any] | pd.Series,
    target: str,
) -> float:
    if target not in predictions:
        raise ValueError(f"Prediction is missing required target {target}")
    try:
        value = float(predictions[target])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Prediction for required target {target} must be numeric and finite"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Prediction for required target {target} must be numeric and finite"
        )
    if value < 0:
        raise ValueError(
            f"Prediction for required target {target} must be nonnegative"
        )
    return value


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


def _mixture_quantiles(
    play_probability: float,
    conditional_mean: float,
    conditional_std: float,
    *,
    probabilities: tuple[float, ...],
) -> tuple[float, ...]:
    """Quantiles for a zero-inflated, nonnegative conditional Normal model."""

    play_probability = _probability(play_probability)
    if play_probability <= 0:
        return tuple(0.0 for _ in probabilities)
    values = []
    for probability in probabilities:
        if probability <= 1.0 - play_probability:
            values.append(0.0)
            continue
        conditional_probability = (
            probability - (1.0 - play_probability)
        ) / play_probability
        conditional_probability = float(
            np.clip(conditional_probability, 1e-9, 1.0 - 1e-9)
        )
        if conditional_std <= 0:
            value = conditional_mean
        else:
            value = conditional_mean + conditional_std * float(
                norm.ppf(conditional_probability)
            )
        values.append(max(0.0, value))
    return tuple(sorted(values))
