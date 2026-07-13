"""Build transparent evidence reports from prediction traces.

This module is intentionally deterministic and numerical.  It does not claim
that a feature is causal; it reports the evidence that moved the model output
and clearly separates model disagreement from feature evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class EvidenceItem:
    """One signed piece of evidence supporting or opposing a projection."""

    label: str
    family: str
    direction: str
    contribution: float
    source: str = "model"
    value: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioResult:
    """Sensitivity result, explicitly not a causal counterfactual."""

    name: str
    change: str
    projection: float
    delta: float
    available: bool = True
    note: str = "model sensitivity; not a causal estimate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionTrace:
    """Runtime trace used to reconstruct and explain one stat prediction."""

    stat: str
    catboost_prediction: float
    transformer_prediction: Optional[float] = None
    opportunity_prediction: Optional[float] = None
    catboost_weight: float = 1.0
    transformer_weight: float = 0.0
    opportunity_weight: float = 0.0
    intercept: float = 0.0
    pre_correction_prediction: float = 0.0
    residual_correction: float = 0.0
    final_prediction: float = 0.0
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    model_version: str = "unknown"
    data_as_of: Optional[str] = None
    data_quality: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatReasoning:
    """Explanation for one player/stat pair."""

    stat: str
    projection: float
    interval: Optional[Tuple[float, float]]
    confidence: Optional[str]
    confidence_score: Optional[float]
    summary: str
    supporting_evidence: List[EvidenceItem] = field(default_factory=list)
    opposing_evidence: List[EvidenceItem] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    scenarios: List[ScenarioResult] = field(default_factory=list)
    trace: Optional[PredictionTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["supporting_evidence"] = [x.to_dict() for x in self.supporting_evidence]
        payload["opposing_evidence"] = [x.to_dict() for x in self.opposing_evidence]
        payload["scenarios"] = [x.to_dict() for x in self.scenarios]
        payload["trace"] = self.trace.to_dict() if self.trace else None
        return payload


@dataclass
class PlayerReasoningReport:
    """Complete explanation for a player's stat line."""

    player_name: str = ""
    player_id: Optional[Any] = None
    team: str = ""
    opponent: str = ""
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_version: str = "unknown"
    stats: Dict[str, StatReasoning] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "player_name": self.player_name,
            "player_id": self.player_id,
            "team": self.team,
            "opponent": self.opponent,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "stats": {stat: report.to_dict() for stat, report in self.stats.items()},
        }


class ReasoningEngine:
    """Convert traces and safe context into ranked evidence reports."""

    FAMILY_RULES: Sequence[Tuple[str, Tuple[str, ...]]] = (
        ("minutes/opportunity", ("MIN", "ROLE", "USAGE", "LINEUP", "INJURY")),
        ("recent form", ("ROLL", "EWMA", "MOMENTUM", "RECENCY", "TREND", "FORM")),
        ("matchup", ("MATCHUP", "OPP", "DEFENSE", "POSITION")),
        ("pace/environment", ("PACE", "POSS", "HOME", "REST", "FATIGUE", "CONTEXT")),
        ("team context", ("TEAM", "SEASON", "PHASE", "MOTIVATION")),
        ("player profile", ("AGE", "ARCHETYPE", "SKILL", "EFFICIENCY")),
    )

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = max(1, int(top_k))

    @classmethod
    def family_for_feature(cls, name: str) -> str:
        upper = str(name).upper()
        for family, keywords in cls.FAMILY_RULES:
            if any(keyword in upper for keyword in keywords):
                return family
        return "other model evidence"

    @staticmethod
    def _safe_number(value: Any) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    @staticmethod
    def _context_value(context: Optional[pd.DataFrame], names: Iterable[str]) -> Optional[float]:
        if context is None or context.empty:
            return None
        row = context.iloc[0]
        for name in names:
            if name in row.index:
                value = ReasoningEngine._safe_number(row[name])
                if value is not None:
                    return value
        upper = {str(col).upper(): col for col in row.index}
        for name in names:
            for candidate, original in upper.items():
                if name.upper() in candidate:
                    value = ReasoningEngine._safe_number(row[original])
                    if value is not None:
                        return value
        return None

    def _feature_evidence(self, trace: PredictionTrace) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        for name, contribution in trace.feature_contributions.items():
            value = self._safe_number(contribution)
            if value is None or abs(value) < 1e-8:
                continue
            evidence.append(
                EvidenceItem(
                    label=str(name),
                    family=self.family_for_feature(str(name)),
                    direction="supports" if value > 0 else "opposes",
                    contribution=float(value),
                    source="CatBoost local attribution",
                )
            )
        return evidence

    def _context_evidence(
        self,
        stat: str,
        projection: float,
        context: Optional[pd.DataFrame],
    ) -> List[EvidenceItem]:
        if context is None or context.empty:
            return []
        items: List[EvidenceItem] = []
        stat_upper = stat.upper()
        for label, names, family in (
            ("recent form", (f"{stat_upper}_ROLLING_AVG", f"ROLLING_{stat_upper}", f"{stat_upper}_RECENT_AVG"), "recent form"),
            ("matchup history", (f"{stat_upper}_MATCHUP_AVG", "MATCHUP_AVG"), "matchup"),
            ("projected minutes", ("MINUTES_PRED", "PREDICTED_MINUTES", "MIN_PRED", "MIN"), "minutes/opportunity"),
        ):
            value = self._context_value(context, names)
            if value is None:
                continue
            contribution = value - projection
            if abs(contribution) < 1e-8:
                continue
            items.append(
                EvidenceItem(
                    label=label,
                    family=family,
                    direction="supports" if contribution > 0 else "opposes",
                    contribution=float(contribution),
                    source="safe context comparison",
                    value=value,
                    detail=f"context={value:.2f}, projection={projection:.2f}",
                )
            )
        return items

    def _scenarios(
        self,
        projection: float,
        context: Optional[pd.DataFrame],
    ) -> List[ScenarioResult]:
        scenarios: List[ScenarioResult] = []
        minutes = self._context_value(context, ("MINUTES_PRED", "PREDICTED_MINUTES", "MIN_PRED", "MIN"))
        if minutes is not None and minutes > 0:
            delta = projection * (3.0 / minutes)
            scenarios.extend(
                [
                    ScenarioResult("minutes", "+3 minutes", max(0.0, projection + delta), delta),
                    ScenarioResult("minutes", "-3 minutes", max(0.0, projection - delta), -delta),
                ]
            )
        pace = self._context_value(context, ("PACE", "PACE_FACTOR", "OPP_PACE"))
        if pace is not None:
            delta = projection * 0.05
            scenarios.extend(
                [
                    ScenarioResult("pace", "+5% pace", max(0.0, projection + delta), delta),
                    ScenarioResult("pace", "-5% pace", max(0.0, projection - delta), -delta),
                ]
            )
        usage = self._context_value(context, ("USAGE", "USG_PCT", "USAGE_RATE"))
        if usage is not None:
            delta = projection * 0.10
            scenarios.extend(
                [
                    ScenarioResult("usage", "+10% usage", max(0.0, projection + delta), delta),
                    ScenarioResult("usage", "-10% usage", max(0.0, projection - delta), -delta),
                ]
            )
        return scenarios

    def build_report(
        self,
        traces: Mapping[str, PredictionTrace],
        *,
        context: Optional[pd.DataFrame] = None,
        player_name: str = "",
        player_id: Optional[Any] = None,
        team: str = "",
        opponent: str = "",
    ) -> PlayerReasoningReport:
        report = PlayerReasoningReport(
            player_name=player_name,
            player_id=player_id,
            team=team,
            opponent=opponent,
            model_version=next(iter(traces.values())).model_version if traces else "unknown",
        )
        for stat, trace in traces.items():
            all_evidence = self._feature_evidence(trace) + self._context_evidence(
                stat, trace.final_prediction, context
            )
            all_evidence.sort(key=lambda item: abs(item.contribution), reverse=True)
            supporting = [item for item in all_evidence if item.direction == "supports"][: self.top_k]
            opposing = [item for item in all_evidence if item.direction == "opposes"][: self.top_k]

            conflicts: List[str] = []
            expert_values = [
                ("CatBoost", trace.catboost_prediction),
                ("Transformer", trace.transformer_prediction),
                ("opportunity", trace.opportunity_prediction),
            ]
            available = [(name, value) for name, value in expert_values if value is not None]
            if len(available) >= 2:
                values = [value for _, value in available]
                if max(values) - min(values) > max(1.0, 0.15 * max(values)):
                    conflicts.append(
                        "Model experts disagree: "
                        + ", ".join(f"{name}={value:.1f}" for name, value in available)
                    )
            if trace.residual_correction and abs(trace.residual_correction) > 1e-8:
                conflicts.append(f"Residual correction adjusted the blend by {trace.residual_correction:+.2f}")
            if trace.data_quality not in ("FULL", "UNKNOWN"):
                conflicts.append(f"Input quality is {trace.data_quality}")

            interval = None
            if trace.interval_low is not None and trace.interval_high is not None:
                interval = (trace.interval_low, trace.interval_high)
            summary = f"{stat}: {trace.final_prediction:.1f}"
            if trace.confidence:
                summary += f" ({trace.confidence.lower()} confidence)"
            if conflicts:
                summary += "; review model/context disagreement"
            elif supporting:
                summary += f"; strongest evidence is {supporting[0].label}"

            report.stats[stat] = StatReasoning(
                stat=stat,
                projection=float(trace.final_prediction),
                interval=interval,
                confidence=trace.confidence,
                confidence_score=trace.confidence_score,
                summary=summary,
                supporting_evidence=supporting,
                opposing_evidence=opposing,
                conflicts=conflicts,
                scenarios=self._scenarios(trace.final_prediction, context),
                trace=trace,
            )
        return report

    def build_projection_report(
        self,
        projection: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        *,
        player_name: str = "",
        player_id: Optional[Any] = None,
        team: str = "",
        opponent: str = "",
        model_version: str = "projection-cache",
    ) -> PlayerReasoningReport:
        """Explain a cached projection when runtime model artifacts are absent.

        This path is intentionally honest: it compares safe recent and
        matchup summaries but does not invent local model attributions.
        """
        context = context or {}
        recent = context.get("recent_avg", {}) if isinstance(context, Mapping) else {}
        matchup = context.get("matchup_avg", {}) if isinstance(context, Mapping) else {}
        recent_available = int(context.get("recent_sample_size", 0) or 0) > 0 if isinstance(context, Mapping) else False
        matchup_available = int(context.get("matchup_sample_size", 0) or 0) > 0 if isinstance(context, Mapping) else False
        traces: Dict[str, PredictionTrace] = {}
        for stat in ("PTS", "REB", "AST", "STL", "BLK", "TOV"):
            key = stat.lower()
            value = self._safe_number(
                projection.get(f"PROJ_{stat}_MEAN", projection.get(f"{key}_mean", projection.get(stat, 0.0)))
            ) or 0.0
            contributions: Dict[str, float] = {}
            recent_value = self._safe_number(recent.get(key)) if recent_available and isinstance(recent, Mapping) else None
            matchup_value = self._safe_number(matchup.get(key)) if matchup_available and isinstance(matchup, Mapping) else None
            if recent_value is not None:
                contributions["RECENT_FORM"] = recent_value - value
            if matchup_value is not None:
                contributions["MATCHUP_HISTORY"] = matchup_value - value
            low = self._safe_number(projection.get(f"{stat}_INTERVAL_90_LOW", projection.get(f"{stat}_CI_LOW")))
            high = self._safe_number(projection.get(f"{stat}_INTERVAL_90_HIGH", projection.get(f"{stat}_CI_HIGH")))
            traces[stat] = PredictionTrace(
                stat=stat,
                catboost_prediction=value,
                pre_correction_prediction=value,
                final_prediction=value,
                interval_low=low,
                interval_high=high,
                confidence=projection.get(f"{stat}_CONFIDENCE"),
                confidence_score=self._safe_number(projection.get(f"{stat}_CONFIDENCE_SCORE")),
                feature_contributions=contributions,
                model_version=model_version,
                data_quality=str(projection.get("DATA_QUALITY", "UNKNOWN")),
            )
        return self.build_report(
            traces,
            player_name=player_name,
            player_id=player_id,
            team=team,
            opponent=opponent,
        )

    @staticmethod
    def format_concise(report: PlayerReasoningReport, stat: Optional[str] = None) -> str:
        """Render the report for a human without hiding uncertainty."""
        selected = [stat.upper()] if stat else list(report.stats)
        lines = [f"Reasoning for {report.player_name or 'player'} ({report.model_version})"]
        for target in selected:
            item = report.stats.get(target)
            if item is None:
                continue
            interval = ""
            if item.interval:
                interval = f" | interval {item.interval[0]:.1f}–{item.interval[1]:.1f}"
            lines.append(f"\n{item.summary}{interval}")
            if item.supporting_evidence:
                lines.append("  Supports: " + "; ".join(
                    f"{x.label} ({x.contribution:+.2f})" for x in item.supporting_evidence
                ))
            if item.opposing_evidence:
                lines.append("  Risks: " + "; ".join(
                    f"{x.label} ({x.contribution:+.2f})" for x in item.opposing_evidence
                ))
            for conflict in item.conflicts:
                lines.append(f"  Note: {conflict}")
            if item.scenarios:
                lines.append("  Sensitivity: " + "; ".join(
                    f"{scenario.change} → {scenario.projection:.1f}" for scenario in item.scenarios
                ))
        return "\n".join(lines)
