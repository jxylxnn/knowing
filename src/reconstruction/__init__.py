"""Latent game and possession reconstruction package.

Implements the design in ``plans/game_reconstruction_mega_plan.md``: turning
aggregate box scores into constraint-satisfying plausible possession histories,
weighting them into a posterior, and materializing leak-safe summaries.

This package is added incrementally. PR 01 contributes only the typed domain
schema (``schema.py``); later PRs add canonicalization, the event ledger,
possession templates, the sampler, summaries, and the service/CLI.
"""

from src.reconstruction.schema import (
    RECONSTRUCTION_METHOD_VERSION,
    RECONSTRUCTION_SCHEMA_VERSION,
    EventType,
    FiniteNumberError,
    GameBoxScore,
    LatentEvent,
    LatentPossession,
    LineupError,
    PlayerBoxLine,
    PossessionEnd,
    ReconstructionPosterior,
    ReconstructionQuality,
    ReconstructionSample,
    RotationSample,
    RotationSlot,
    TeamBoxLine,
    assert_finite_numbers,
)
from src.reconstruction.contracts import (
    GameValidationReport,
    validate_game_box_score,
)
from src.reconstruction.canonicalize import (
    CanonicalizationError,
    canonicalize_game,
    canonicalize_games,
    group_games,
    hash_source_rows,
    parse_minutes_value,
)

__all__ = [
    "RECONSTRUCTION_SCHEMA_VERSION",
    "RECONSTRUCTION_METHOD_VERSION",
    "EventType",
    "PossessionEnd",
    "ReconstructionQuality",
    "PlayerBoxLine",
    "TeamBoxLine",
    "GameBoxScore",
    "LatentEvent",
    "LatentPossession",
    "RotationSlot",
    "RotationSample",
    "ReconstructionSample",
    "ReconstructionPosterior",
    "FiniteNumberError",
    "LineupError",
    "assert_finite_numbers",
    "GameValidationReport",
    "validate_game_box_score",
    "CanonicalizationError",
    "canonicalize_game",
    "canonicalize_games",
    "group_games",
    "hash_source_rows",
    "parse_minutes_value",
]
