"""Explicit, versioned sample-weighting policies for model training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingWeightPolicy:
    """A reproducible policy for assigning weights to historical examples."""

    policy_id: str = "recency_exp_v1"
    method: str = "exponential_decay"
    enabled: bool = True
    lambda_decay: float = 0.023
    min_weight: float = 0.1
    reference_date: str = "max_training_date"
    version: int = 1

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id is required")
        if self.method not in {"exponential_decay", "uniform"}:
            raise ValueError(f"Unsupported weighting method: {self.method}")
        if self.lambda_decay < 0:
            raise ValueError("lambda_decay must be non-negative")
        if not 0 < self.min_weight <= 1:
            raise ValueError("min_weight must be in (0, 1]")
        if self.version < 1:
            raise ValueError("version must be positive")

    @classmethod
    def from_config(cls, config: Any | None = None) -> "TrainingWeightPolicy":
        values: dict[str, Any] = {}
        if config is not None:
            configured = getattr(config, "weighting", None)
            if isinstance(configured, Mapping):
                values.update(configured)
            training = getattr(config, "training", None)
            if training is not None:
                values.setdefault("enabled", getattr(training, "use_sample_weights", True))
                values.setdefault("lambda_decay", getattr(training, "temporal_decay_lambda", 0.023))
        return cls(**{key: values[key] for key in values if key in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingWeightPolicy":
        """Deserialize a policy while ignoring the derived policy hash."""
        return cls(**{
            key: value for key, value in payload.items()
            if key in cls.__dataclass_fields__
        })

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def calculate(self, frame: pd.DataFrame) -> np.ndarray:
        """Calculate one finite, positive weight per training row."""
        if not self.enabled or self.method == "uniform":
            return np.ones(len(frame), dtype=float)
        if "GAME_DATE" not in frame.columns:
            raise ValueError("Recency weighting requires GAME_DATE")

        dates = pd.to_datetime(frame["GAME_DATE"], errors="coerce")
        if dates.isna().any():
            raise ValueError("GAME_DATE contains malformed values for weighting")
        reference = dates.max()
        days_ago = (reference - dates).dt.total_seconds() / 86400.0
        weights = np.exp(-self.lambda_decay * np.maximum(days_ago, 0.0))
        weights = np.clip(weights, self.min_weight, 1.0)
        return np.asarray(weights, dtype=float)

    def summarize(self, frame: pd.DataFrame) -> dict[str, Any]:
        weights = self.calculate(frame)
        return {
            "count": int(len(weights)),
            "min": float(np.min(weights)) if len(weights) else None,
            "max": float(np.max(weights)) if len(weights) else None,
            "mean": float(np.mean(weights)) if len(weights) else None,
            "sum": float(np.sum(weights)) if len(weights) else 0.0,
        }

    def evaluate(
        self,
        validation_metrics: Mapping[str, Mapping[str, Any] | Any],
        *,
        train_frame: pd.DataFrame,
        scored_at: str | None = None,
    ) -> "WeightPolicyEvaluation":
        """Record validation scores produced by this policy's trained models."""
        normalized: dict[str, dict[str, float]] = {}
        for target, metrics in validation_metrics.items():
            if hasattr(metrics, "metrics"):
                metrics = metrics.metrics
            if not isinstance(metrics, Mapping):
                continue
            normalized[str(target)] = {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and np.isfinite(value)
            }
        return WeightPolicyEvaluation(
            policy=self,
            train_weight_summary=self.summarize(train_frame),
            validation_metrics=normalized,
            scored_at=scored_at or datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_hash"] = self.policy_hash
        return payload


@dataclass(frozen=True)
class WeightPolicyEvaluation:
    policy: TrainingWeightPolicy
    train_weight_summary: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    scored_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "train_weight_summary": self.train_weight_summary,
            "validation_metrics": self.validation_metrics,
            "scored_at": self.scored_at,
            "evaluation_status": "scored_on_validation",
        }
