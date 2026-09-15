"""Participation baselines and probability calibration for Model v2.

The module deliberately models two different events:

``active``
    The player is on the eligible/active game-day roster.
``plays_given_active``
    The player records an appearance, conditional on being active.

Collapsing the two labels during training makes injury/status information hard
to audit and was the source of an earlier ``P(active) ** 2`` implementation.
All public prediction helpers therefore retain both probabilities and derive
``PLAY_PROB`` exactly once at the output boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.contracts.sources import parse_aware_datetime


_EPSILON = 1e-7
CALIBRATION_EVIDENCE_KINDS = frozenset({"heldout", "oof"})

# These are declared rules, not learned injury opinions. The source status is
# retained in output so replay scorecards can evaluate every rule separately.
DEFAULT_STATUS_RULES: dict[str, tuple[float, float]] = {
    "ACTIVE": (0.995, 0.995),
    "AVAILABLE": (0.995, 0.995),
    "PROBABLE": (0.95, 0.97),
    "QUESTIONABLE": (0.55, 0.90),
    "DOUBTFUL": (0.15, 0.75),
    "OUT": (0.0, 0.0),
    "INACTIVE": (0.0, 0.0),
    "SUSPENDED": (0.0, 0.0),
    "NOT_WITH_TEAM": (0.0, 0.0),
}


@dataclass(frozen=True)
class ReliabilityBin:
    """One non-empty bin in a probability reliability curve."""

    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_rate: float


@dataclass(frozen=True)
class BinaryProbabilityMetrics:
    """Proper scores and calibration error for a binary forecast."""

    rows: int
    brier: float
    log_loss: float
    ece: float
    reliability: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reliability"] = [asdict(item) for item in self.reliability]
        return payload


def _probability_array(values: Iterable[float]) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    return np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)


def _binary_array(values: Iterable[float]) -> np.ndarray:
    labels = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("Binary labels must contain only finite zeroes and ones")
    return labels.astype(int)


def reliability_curve(
    y_true: Iterable[float],
    probabilities: Iterable[float],
    *,
    bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    """Return equal-width reliability bins without fabricating empty bins."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    labels = _binary_array(y_true)
    predicted = _probability_array(probabilities)
    if len(labels) != len(predicted):
        raise ValueError("Labels and probabilities must have the same length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.minimum(
        np.searchsorted(edges, predicted, side="right") - 1,
        bins - 1,
    )
    output: list[ReliabilityBin] = []
    for index in range(bins):
        mask = membership == index
        if not mask.any():
            continue
        output.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=int(mask.sum()),
                mean_probability=float(predicted[mask].mean()),
                observed_rate=float(labels[mask].mean()),
            )
        )
    return tuple(output)


def binary_probability_metrics(
    y_true: Iterable[float],
    probabilities: Iterable[float],
    *,
    bins: int = 10,
) -> BinaryProbabilityMetrics:
    """Compute Brier score, log loss, ECE, and reliability evidence."""
    labels = _binary_array(y_true)
    predicted = _probability_array(probabilities)
    if not len(labels):
        raise ValueError("At least one probability observation is required")
    if len(labels) != len(predicted):
        raise ValueError("Labels and probabilities must have the same length")
    curve = reliability_curve(labels, predicted, bins=bins)
    ece = sum(
        item.count / len(labels)
        * abs(item.mean_probability - item.observed_rate)
        for item in curve
    )
    return BinaryProbabilityMetrics(
        rows=len(labels),
        brier=float(np.mean(np.square(predicted - labels))),
        log_loss=float(log_loss(labels, predicted, labels=[0, 1])),
        ece=float(ece),
        reliability=curve,
    )


def _normalise_status(value: object) -> str:
    status = str("UNKNOWN" if pd.isna(value) else value).strip().upper().replace(" ", "_")
    aliases = {
        "GTD": "QUESTIONABLE",
        "GAME_TIME_DECISION": "QUESTIONABLE",
        "DAY_TO_DAY": "QUESTIONABLE",
        "DNP": "INACTIVE",
        "HEALTHY": "AVAILABLE",
    }
    return aliases.get(status, status)


def _latest_as_of(
    frame: pd.DataFrame,
    *,
    cutoff: object | None,
    player_column: str,
    available_at_column: str,
) -> pd.DataFrame:
    source = frame.copy()
    if cutoff is not None and available_at_column not in source:
        raise ValueError("Status evidence requires AVAILABLE_AT with a cutoff")
    if available_at_column in source.columns:
        if cutoff is not None:
            parse_aware_datetime(cutoff, field="status cutoff")
            for value in source[available_at_column]:
                parse_aware_datetime(value, field="status AVAILABLE_AT")
        times = pd.to_datetime(
            source[available_at_column],
            utc=True,
            errors="coerce",
        )
        if cutoff is not None:
            boundary = pd.Timestamp(cutoff)
            boundary = (
                boundary.tz_localize("UTC")
                if boundary.tzinfo is None
                else boundary.tz_convert("UTC")
            )
            source = source.loc[times.notna() & times.le(boundary)].copy()
            times = times.loc[source.index]
        source["__AVAILABLE_AT"] = times
        source = source.sort_values("__AVAILABLE_AT", kind="stable")
    return source.drop_duplicates(player_column, keep="last")


def status_rule_baseline(
    candidates: pd.DataFrame,
    status_snapshots: pd.DataFrame | None = None,
    *,
    cutoff: object | None = None,
    status_rules: Mapping[str, tuple[float, float]] | None = None,
    default_active_probability: float = 0.90,
    default_play_given_active: float = 0.95,
    player_column: str = "PLAYER_ID",
    status_column: str = "STATUS",
    available_at_column: str = "AVAILABLE_AT",
) -> pd.DataFrame:
    """Apply auditable last-known-status rules as of ``cutoff``.

    If snapshots are supplied, undated snapshots are excluded when a cutoff is
    requested: treating an unknown publication time as historical evidence is
    a look-ahead leak. Candidates may themselves contain a status for already
    point-in-time materialized use cases.
    """
    if player_column not in candidates:
        raise ValueError(f"Candidates missing column: {player_column}")
    rules = {
        key.upper(): value
        for key, value in (status_rules or DEFAULT_STATUS_RULES).items()
    }
    if not 0 <= default_active_probability <= 1:
        raise ValueError("default_active_probability must be in [0, 1]")
    if not 0 <= default_play_given_active <= 1:
        raise ValueError("default_play_given_active must be in [0, 1]")

    result = candidates.copy()
    if status_snapshots is not None and not status_snapshots.empty:
        required = {player_column, status_column}
        if missing := required - set(status_snapshots.columns):
            raise ValueError(f"Status snapshots missing columns: {sorted(missing)}")
        latest = _latest_as_of(
            status_snapshots,
            cutoff=cutoff,
            player_column=player_column,
            available_at_column=available_at_column,
        )
        status_map = latest.set_index(player_column)[status_column]
        candidate_status = result.get(
            status_column,
            pd.Series(index=result.index, dtype=object),
        )
        result[status_column] = result[player_column].map(status_map).combine_first(
            candidate_status
        )
    elif status_column not in result:
        result[status_column] = "UNKNOWN"

    normalised = result[status_column].map(_normalise_status)
    pairs = normalised.map(rules)
    result["P_ACTIVE"] = pairs.map(
        lambda value: (
            value[0] if isinstance(value, tuple) else default_active_probability
        )
    ).astype(float)
    result["P_PLAY_GIVEN_ACTIVE"] = pairs.map(
        lambda value: (
            value[1] if isinstance(value, tuple) else default_play_given_active
        )
    ).astype(float)
    result["PLAY_PROB"] = result["P_ACTIVE"] * result["P_PLAY_GIVEN_ACTIVE"]
    result["AVAILABILITY_STATUS"] = normalised
    result["AVAILABILITY_SOURCE"] = np.where(
        normalised.isin(rules),
        "last_known_status_rule",
        "unknown_status_prior",
    )
    return result


def appearance_participation_baseline(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    window: int = 10,
    active_prior: float = 0.90,
    play_given_active_prior: float = 0.95,
    prior_strength: float = 2.0,
    player_column: str = "PLAYER_ID",
) -> pd.DataFrame:
    """Estimate the two participation events from their honest denominators.

    ``P_ACTIVE`` uses eligible games as its denominator. ``P_PLAY_GIVEN_ACTIVE``
    uses only known-active games. When historical ``ACTIVE`` labels are absent,
    the active probability stays at its declared prior instead of pretending
    that every appearance row proves an active-roster event.
    """
    if window < 1 or prior_strength < 0:
        raise ValueError("window must be positive and prior_strength non-negative")
    if player_column not in candidates:
        raise ValueError(f"Candidates missing column: {player_column}")
    for value, name in (
        (active_prior, "active_prior"),
        (play_given_active_prior, "play_given_active_prior"),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    source = (
        history.copy()
        if history is not None
        else pd.DataFrame(columns=[player_column])
    )
    if not source.empty and player_column not in source:
        raise ValueError(f"History missing column: {player_column}")
    if "GAME_DATE" in source:
        source = source.sort_values("GAME_DATE", kind="stable")

    estimates: dict[object, tuple[float, float, str]] = {}
    for player_id, group in source.groupby(player_column, sort=False):
        recent = group.tail(window)
        if "APPEARED" in recent:
            appeared = pd.to_numeric(recent["APPEARED"], errors="coerce")
        elif "MIN" in recent:
            minutes = pd.to_numeric(recent["MIN"], errors="coerce")
            appeared = pd.Series(np.nan, index=recent.index)
            appeared.loc[minutes.gt(0)] = 1.0
        else:
            appeared = pd.Series(np.nan, index=recent.index)
        if not appeared.dropna().isin([0, 1]).all():
            raise ValueError("Known appearance labels must be binary")
        eligible_column = next((name for name in ("ROSTER_ELIGIBLE", "ELIGIBLE") if name in recent), None)
        eligible = (
            pd.to_numeric(recent[eligible_column], errors="raise").eq(1)
            if eligible_column else pd.Series(True, index=recent.index)
        )
        active_column = next(
            (name for name in ("ACTIVE", "IS_ACTIVE") if name in recent),
            None,
        )
        if active_column is None:
            p_active = float(active_prior)
            active = pd.Series(False, index=recent.index)
            source_name = "unknown_active_and_conditional_play_priors"
        else:
            active_values = pd.to_numeric(
                recent[active_column], errors="coerce"
            )
            if not active_values.dropna().isin([0, 1]).all():
                raise ValueError("Known active labels must be binary")
            active_known = eligible & active_values.notna()
            active = active_values.fillna(0).gt(0)
            active_successes = float(active.loc[active_known].sum())
            active_count = int(active_known.sum())
            denominator = active_count + prior_strength
            p_active = (
                (active_successes + prior_strength * active_prior) / denominator
                if denominator
                else active_prior
            )
            source_name = "active_and_appearance_frequency"

        play_known = eligible & active & appeared.notna()
        play_successes = float(appeared.loc[play_known].sum())
        play_count = int(play_known.sum())
        denominator = play_count + prior_strength
        p_play = (
            (play_successes + prior_strength * play_given_active_prior)
            / denominator
            if denominator
            else play_given_active_prior
        )
        estimates[player_id] = (float(p_active), float(p_play), source_name)

    result = candidates[[player_column]].copy()
    mapped = result[player_column].map(estimates)
    result["P_ACTIVE"] = mapped.map(
        lambda value: value[0] if isinstance(value, tuple) else active_prior
    )
    result["P_PLAY_GIVEN_ACTIVE"] = mapped.map(
        lambda value: (
            value[1] if isinstance(value, tuple) else play_given_active_prior
        )
    )
    result["PLAY_PROB"] = result["P_ACTIVE"] * result["P_PLAY_GIVEN_ACTIVE"]
    result["AVAILABILITY_SOURCE"] = mapped.map(
        lambda value: value[2] if isinstance(value, tuple) else "cold_start_prior"
    )
    return result


def participation_baseline(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    window: int = 10,
) -> pd.DataFrame:
    """Backward-compatible alias for the honest appearance baseline."""
    return appearance_participation_baseline(history, candidates, window=window)


class ProbabilityCalibrator:
    """Select identity, Platt, or isotonic calibration from valid evidence.

    Candidate methods are selected on the final chronological portion of the
    supplied held-out/OOF evidence, then the winning calibrator is refit on all
    evidence. This avoids selecting isotonic from its own training fit.
    """

    def __init__(
        self,
        *,
        methods: Sequence[str] = ("identity", "platt", "isotonic"),
        selection_fraction: float = 0.25,
        bins: int = 10,
        minimum_rows: int = 24,
    ) -> None:
        unknown = set(methods) - {"identity", "platt", "isotonic"}
        if unknown:
            raise ValueError(f"Unknown calibration methods: {sorted(unknown)}")
        if not methods:
            raise ValueError("At least one calibration method is required")
        if not 0.1 <= selection_fraction <= 0.5:
            raise ValueError("selection_fraction must be between 0.1 and 0.5")
        self.methods = tuple(dict.fromkeys(methods))
        self.selection_fraction = float(selection_fraction)
        self.bins = int(bins)
        self.minimum_rows = int(minimum_rows)
        self.method_: str | None = None
        self.model_: Any = None
        self.method_scores_: dict[str, dict[str, float]] = {}
        self.evidence_kind_: str | None = None

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, _EPSILON, 1 - _EPSILON)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    @classmethod
    def _fit_method(
        cls,
        method: str,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> Any:
        if method == "identity":
            return None
        if method == "platt":
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            model.fit(cls._logit(probabilities), labels)
            return model
        model = IsotonicRegression(
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        model.fit(probabilities, labels)
        return model

    @classmethod
    def _apply_method(
        cls,
        method: str,
        model: Any,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if method == "identity":
            return probabilities
        if method == "platt":
            return model.predict_proba(cls._logit(probabilities))[:, 1]
        return np.asarray(model.predict(probabilities), dtype=float)

    def fit(
        self,
        probabilities: Iterable[float],
        y_true: Iterable[float],
        *,
        evidence_kind: str,
    ) -> "ProbabilityCalibrator":
        self.method_ = None
        self.model_ = None
        self.method_scores_ = {}
        self.evidence_kind_ = None
        self.evidence_rows_ = 0
        self.insufficient_evidence_ = False
        if evidence_kind not in CALIBRATION_EVIDENCE_KINDS:
            raise ValueError(
                "Calibration requires explicit 'heldout' or 'oof' evidence"
            )
        predicted = _probability_array(probabilities)
        labels = _binary_array(y_true)
        if len(predicted) != len(labels):
            raise ValueError("Labels and probabilities must have the same length")
        if len(predicted) < 4:
            raise ValueError("At least four calibration observations are required")
        self.evidence_kind_ = evidence_kind
        self.evidence_rows_ = len(labels)
        self.insufficient_evidence_ = len(labels) < self.minimum_rows

        split = max(
            2,
            int(np.floor(len(predicted) * (1 - self.selection_fraction))),
        )
        split = min(split, len(predicted) - 2)
        fit_p, score_p = predicted[:split], predicted[split:]
        fit_y, score_y = labels[:split], labels[split:]
        usable = (
            ["identity"]
            if len(predicted) < self.minimum_rows
            else list(self.methods)
        )
        if len(np.unique(fit_y)) < 2:
            usable = [
                method
                for method in usable
                if method in {"identity", "isotonic"}
            ]

        if not usable:
            usable = ["identity"]
            self.insufficient_evidence_ = True

        for method in usable:
            try:
                model = self._fit_method(method, fit_p, fit_y)
                calibrated = np.clip(
                    self._apply_method(method, model, score_p),
                    _EPSILON,
                    1 - _EPSILON,
                )
                metrics = binary_probability_metrics(
                    score_y,
                    calibrated,
                    bins=self.bins,
                )
                objective = metrics.brier + 0.25 * metrics.log_loss + metrics.ece
                self.method_scores_[method] = {
                    "brier": metrics.brier,
                    "log_loss": metrics.log_loss,
                    "ece": metrics.ece,
                    "objective": float(objective),
                }
            except (ValueError, FloatingPointError):
                continue
        if not self.method_scores_:
            raise ValueError("No probability calibration method could be fit")
        self.method_ = min(
            self.method_scores_,
            key=lambda name: (
                self.method_scores_[name]["objective"],
                usable.index(name),
            ),
        )
        self.model_ = self._fit_method(self.method_, predicted, labels)
        return self

    def predict(self, probabilities: Iterable[float]) -> np.ndarray:
        if self.method_ is None:
            raise RuntimeError("ProbabilityCalibrator must be fit before predict")
        predicted = _probability_array(probabilities)
        return np.clip(
            self._apply_method(self.method_, self.model_, predicted),
            0.0,
            1.0,
        )

    def metadata(self) -> dict[str, Any]:
        if self.method_ is None:
            raise RuntimeError("ProbabilityCalibrator has not been fit")
        return {
            "method": self.method_,
            "evidence_kind": self.evidence_kind_,
            "evidence_rows": self.evidence_rows_,
            "insufficient_evidence": self.insufficient_evidence_,
            "selection_scores": self.method_scores_,
        }


class CalibratedBinaryClassifier:
    """Generic classifier plus calibration fit from OOF/held-out scores."""

    def __init__(
        self,
        estimator: Any,
        *,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> None:
        self.estimator = clone(estimator)
        self.calibrator = calibrator or ProbabilityCalibrator()
        self.feature_columns_: tuple[str, ...] | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[float],
        *,
        calibration_probabilities: Iterable[float],
        calibration_y: Iterable[float],
        evidence_kind: str,
    ) -> "CalibratedBinaryClassifier":
        labels = _binary_array(y)
        self.feature_columns_ = (
            tuple(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        self.estimator.fit(X, labels)
        self.calibrator.fit(
            calibration_probabilities,
            calibration_y,
            evidence_kind=evidence_kind,
        )
        return self

    def predict_uncalibrated(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            probabilities = np.asarray(
                self.estimator.predict_proba(X),
                dtype=float,
            )
            return (
                probabilities[:, -1]
                if probabilities.ndim == 2
                else probabilities
            )
        if hasattr(self.estimator, "decision_function"):
            score = np.asarray(
                self.estimator.decision_function(X),
                dtype=float,
            )
            return 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
        raise TypeError(
            "Classifier must implement predict_proba or decision_function"
        )

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        calibrated = self.calibrator.predict(self.predict_uncalibrated(X))
        return np.column_stack((1.0 - calibrated, calibrated))


class ParticipationClassifier:
    """Two calibrated classifiers with distinct active and play labels."""

    def __init__(
        self,
        active_classifier: CalibratedBinaryClassifier,
        play_given_active_classifier: CalibratedBinaryClassifier,
    ) -> None:
        self.active_classifier = active_classifier
        self.play_given_active_classifier = play_given_active_classifier

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        p_active = self.active_classifier.predict_proba(X)[:, 1]
        p_play_active = self.play_given_active_classifier.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "P_ACTIVE": p_active,
                "P_PLAY_GIVEN_ACTIVE": p_play_active,
                "PLAY_PROB": p_active * p_play_active,
            },
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )


__all__ = [
    "BinaryProbabilityMetrics",
    "CALIBRATION_EVIDENCE_KINDS",
    "CalibratedBinaryClassifier",
    "DEFAULT_STATUS_RULES",
    "ParticipationClassifier",
    "ProbabilityCalibrator",
    "ReliabilityBin",
    "appearance_participation_baseline",
    "binary_probability_metrics",
    "participation_baseline",
    "reliability_curve",
    "status_rule_baseline",
]
