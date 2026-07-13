"""Auditable, evidence-based prediction reasoning.

The reasoning package deliberately contains no language-model dependency.  It
turns model traces into structured evidence, uncertainty, and sensitivity
reports that can be rendered by the CLI or saved as JSON.
"""

from .engine import (
    EvidenceItem,
    PredictionTrace,
    ReasoningEngine,
    ScenarioResult,
    StatReasoning,
    PlayerReasoningReport,
)
from .sidecar import load_reasoning_sidecar, sidecar_path, write_reasoning_sidecar

__all__ = [
    "EvidenceItem",
    "PredictionTrace",
    "ReasoningEngine",
    "ScenarioResult",
    "StatReasoning",
    "PlayerReasoningReport",
    "load_reasoning_sidecar",
    "sidecar_path",
    "write_reasoning_sidecar",
]
