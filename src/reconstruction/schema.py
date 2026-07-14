"""Versioned domain schema for latent game reconstruction.

This module defines the immutable typed objects that flow through the
reconstruction pipeline described in
``plans/game_reconstruction_mega_plan.md``. It is the first pull request of the
initiative and deliberately contains *no* sampling logic — only the data
contracts, enums, and JSON serialization rules that later modules
(``canonicalize``, ``ledger``, ``sampler``, ``posterior``, ...) depend on.

Design rules enforced here (Section 8 of the plan):

* Two version constants gate compatibility. Changing field meaning, event
  semantics, or hard constraints requires bumping one of them.
* Dataclasses are frozen so a constructed box score / event / sample cannot be
  silently mutated after the fact; later stages rebuild objects via
  ``dataclasses.replace`` rather than in-place edits.
* Collection fields are tuples (not lists) for hashability and deterministic
  serialization.
* ``to_dict()`` produces JSON-primitive payloads (enums -> ``.value``,
  ``date`` -> ISO string, tuples -> lists) and runs the recursive
  finite-number validator so NaN/Infinity can never reach a persisted file.
* Lineups carry an explicit invariant: a non-empty lineup must hold exactly
  five *unique* player IDs. Empty lineups are permitted only when rotation
  inference is disabled (``rotation_mode="disabled"``).

No random sampling, no file I/O, no pandas. Pure typed contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import Any, Dict, Optional, Tuple

__all__ = [
    # versions
    "RECONSTRUCTION_SCHEMA_VERSION",
    "RECONSTRUCTION_METHOD_VERSION",
    # validation
    "assert_finite_numbers",
    "FiniteNumberError",
    "LineupError",
    "validate_lineup",
    # enums
    "EventType",
    "PossessionEnd",
    "ReconstructionQuality",
    # box score
    "PlayerBoxLine",
    "TeamBoxLine",
    "GameBoxScore",
    # events / possessions
    "LatentEvent",
    "LatentPossession",
    # rotations
    "RotationSlot",
    "RotationSample",
    # posterior
    "ReconstructionSample",
    "ReconstructionPosterior",
]


# ---------------------------------------------------------------------------
# Version constants (Section 8.1)
# ---------------------------------------------------------------------------
RECONSTRUCTION_SCHEMA_VERSION = "reconstruction_schema_v1"
RECONSTRUCTION_METHOD_VERSION = "boxscore_posterior_v1"


# ---------------------------------------------------------------------------
# Finite-number validation (Section 8.6 / PR 01 step 4)
# ---------------------------------------------------------------------------
class FiniteNumberError(ValueError):
    """Raised when a value that would serialize as NaN/Infinity is encountered."""


class LineupError(ValueError):
    """Raised when a lineup violates the five-unique-player invariant."""


def _assert_finite(value: Any, path: str) -> None:
    """Validate a single scalar; recurse is handled by ``assert_finite_numbers``."""
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; it is always finite. Check first so
        # the numeric branch below does not re-evaluate it.
        return
    if isinstance(value, (int,)):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise FiniteNumberError(
                f"non-finite float at {path}: {value!r}"
            )
        return


def assert_finite_numbers(obj: Any, path: str = "root") -> None:
    """Recursively reject NaN/Infinity anywhere in a nested structure.

    Walks dataclasses, dicts, tuples/lists, ``date`` objects, and scalars so a
    persisted payload can never contain a value that JSON cannot represent
    strictly. Called by the top-level ``to_dict()`` methods before return.
    """
    if hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            assert_finite_numbers(getattr(obj, name), f"{path}.{name}")
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            assert_finite_numbers(val, f"{path}[{key!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            assert_finite_numbers(item, f"{path}[{index}]")
        return
    _assert_finite(obj, path)


# ---------------------------------------------------------------------------
# Enums (Section 8.2)
# ---------------------------------------------------------------------------
class EventType(str, Enum):
    """Observed-stat event vocabulary shared by inference and forward simulation.

    ``is_observed_stat`` on ``LatentEvent`` distinguishes a token that must be
    conserved exactly (made field goal, recorded steal, ...) from an inferred
    boundary (``UNKNOWN_BOUNDARY`` / ``PERIOD_END``).
    """

    FG2_MADE = "fg2_made"
    FG2_MISSED = "fg2_missed"
    FG3_MADE = "fg3_made"
    FG3_MISSED = "fg3_missed"
    FREE_THROW = "free_throw"
    TURNOVER = "turnover"
    OFFENSIVE_REBOUND = "offensive_rebound"
    DEFENSIVE_REBOUND = "defensive_rebound"
    ASSIST = "assist"
    STEAL = "steal"
    BLOCK = "block"
    PERSONAL_FOUL = "personal_foul"
    FOUL_DRAWN = "foul_drawn"
    PERIOD_END = "period_end"
    UNKNOWN_BOUNDARY = "unknown_boundary"


class PossessionEnd(str, Enum):
    """Reason a possession terminated.

    ``UNKNOWN`` is allowed only to absorb possession-count uncertainty; it
    carries no observed stat and receives a strong soft penalty at scoring time.
    """

    MADE_FIELD_GOAL = "made_field_goal"
    DEFENSIVE_REBOUND = "defensive_rebound"
    TURNOVER = "turnover"
    FREE_THROW_TRIP = "free_throw_trip"
    PERIOD_END = "period_end"
    UNKNOWN = "unknown"


class ReconstructionQuality(str, Enum):
    """Quality tier assigned to a reconstructed game (Section 15.3).

    The tier governs whether summaries from the game may feed downstream
    features and how much trust reasoning surfaces attach to them.
    """

    VALID_HIGH = "valid_high"
    VALID_MEDIUM = "valid_medium"
    VALID_LOW = "valid_low"
    UNUSABLE = "unusable"


# ---------------------------------------------------------------------------
# Box-score dataclasses (Section 8.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlayerBoxLine:
    """One player's box-score row for one completed game.

    All count fields are the raw aggregate totals from ``nba_players.csv``.
    Computed two-point quantities are exposed as read-only properties so the
    rest of the code never re-derives ``FG2M``/``FG2A`` inconsistently.
    """

    game_id: str
    team_id: int
    player_id: int
    player_name: str
    minutes: float
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    reb: int
    ast: int
    tov: int
    stl: int
    blk: int
    blka: int
    pf: int
    pfd: int
    pts: int
    plus_minus: float

    # -- computed two-point quantities -----------------------------------
    @property
    def fg2m(self) -> int:
        return self.fgm - self.fg3m

    @property
    def fg2a(self) -> int:
        return self.fga - self.fg3a

    @property
    def fg2_missed(self) -> int:
        return self.fg2a - self.fg2m

    @property
    def fg3_missed(self) -> int:
        return self.fg3a - self.fg3m

    @property
    def fg_missed(self) -> int:
        return self.fga - self.fgm

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "team_id": self.team_id,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "minutes": self.minutes,
            "fgm": self.fgm,
            "fga": self.fga,
            "fg3m": self.fg3m,
            "fg3a": self.fg3a,
            "ftm": self.ftm,
            "fta": self.fta,
            "oreb": self.oreb,
            "dreb": self.dreb,
            "reb": self.reb,
            "ast": self.ast,
            "tov": self.tov,
            "stl": self.stl,
            "blk": self.blk,
            "blka": self.blka,
            "pf": self.pf,
            "pfd": self.pfd,
            "pts": self.pts,
            "plus_minus": self.plus_minus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerBoxLine":
        return cls(
            game_id=str(data["game_id"]),
            team_id=int(data["team_id"]),
            player_id=int(data["player_id"]),
            player_name=str(data["player_name"]),
            minutes=float(data["minutes"]),
            fgm=int(data["fgm"]),
            fga=int(data["fga"]),
            fg3m=int(data["fg3m"]),
            fg3a=int(data["fg3a"]),
            ftm=int(data["ftm"]),
            fta=int(data["fta"]),
            oreb=int(data["oreb"]),
            dreb=int(data["dreb"]),
            reb=int(data["reb"]),
            ast=int(data["ast"]),
            tov=int(data["tov"]),
            stl=int(data["stl"]),
            blk=int(data["blk"]),
            blka=int(data["blka"]),
            pf=int(data["pf"]),
            pfd=int(data["pfd"]),
            pts=int(data["pts"]),
            plus_minus=float(data["plus_minus"]),
        )


@dataclass(frozen=True)
class TeamBoxLine:
    """One team's box-score totals for one completed game.

    Holds the same statistical columns as ``PlayerBoxLine`` plus the team
    context fields. Team totals are the conservation target that player sums
    must reproduce (except turnovers, which carry a team residual).
    """

    game_id: str
    team_id: int
    team_abbreviation: str
    game_date: date
    matchup: str
    minutes: float
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    reb: int
    ast: int
    tov: int
    stl: int
    blk: int
    blka: int
    pf: int
    pfd: int
    pts: int
    team_plus_minus: float

    @property
    def fg2m(self) -> int:
        return self.fgm - self.fg3m

    @property
    def fg2a(self) -> int:
        return self.fga - self.fg3a

    @property
    def fg2_missed(self) -> int:
        return self.fg2a - self.fg2m

    @property
    def fg3_missed(self) -> int:
        return self.fg3a - self.fg3m

    @property
    def fg_missed(self) -> int:
        return self.fga - self.fgm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "team_id": self.team_id,
            "team_abbreviation": self.team_abbreviation,
            "game_date": self.game_date.isoformat(),
            "matchup": self.matchup,
            "minutes": self.minutes,
            "fgm": self.fgm,
            "fga": self.fga,
            "fg3m": self.fg3m,
            "fg3a": self.fg3a,
            "ftm": self.ftm,
            "fta": self.fta,
            "oreb": self.oreb,
            "dreb": self.dreb,
            "reb": self.reb,
            "ast": self.ast,
            "tov": self.tov,
            "stl": self.stl,
            "blk": self.blk,
            "blka": self.blka,
            "pf": self.pf,
            "pfd": self.pfd,
            "pts": self.pts,
            "team_plus_minus": self.team_plus_minus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamBoxLine":
        return cls(
            game_id=str(data["game_id"]),
            team_id=int(data["team_id"]),
            team_abbreviation=str(data["team_abbreviation"]),
            game_date=date.fromisoformat(str(data["game_date"])),
            matchup=str(data["matchup"]),
            minutes=float(data["minutes"]),
            fgm=int(data["fgm"]),
            fga=int(data["fga"]),
            fg3m=int(data["fg3m"]),
            fg3a=int(data["fg3a"]),
            ftm=int(data["ftm"]),
            fta=int(data["fta"]),
            oreb=int(data["oreb"]),
            dreb=int(data["dreb"]),
            reb=int(data["reb"]),
            ast=int(data["ast"]),
            tov=int(data["tov"]),
            stl=int(data["stl"]),
            blk=int(data["blk"]),
            blka=int(data["blka"]),
            pf=int(data["pf"]),
            pfd=int(data["pfd"]),
            pts=int(data["pts"]),
            team_plus_minus=float(data["team_plus_minus"]),
        )


@dataclass(frozen=True)
class GameBoxScore:
    """Canonical, validated box score for one completed game.

    ``team_ids`` is stored sorted ascending and ``players_by_team`` values are
    sorted by player ID so serialization is deterministic regardless of the
    input row order. Home/away identity (when parseable from ``MATCHUP``) is
    preserved on each ``TeamBoxLine`` via its ``matchup`` field, never inferred
    from the sort order.
    """

    game_id: str
    game_date: date
    season_year: str
    team_ids: Tuple[int, int]
    teams: Dict[int, TeamBoxLine]
    players_by_team: Dict[int, Tuple[PlayerBoxLine, ...]]
    overtime_periods: int
    source_player_hash: str
    source_team_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "game_date": self.game_date.isoformat(),
            "season_year": self.season_year,
            "team_ids": list(self.team_ids),
            "teams": {
                str(tid): self.teams[tid].to_dict()
                for tid in sorted(self.teams)
            },
            "players_by_team": {
                str(tid): [p.to_dict() for p in self.players_by_team[tid]]
                for tid in sorted(self.players_by_team)
            },
            "overtime_periods": self.overtime_periods,
            "source_player_hash": self.source_player_hash,
            "source_team_hash": self.source_team_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameBoxScore":
        teams = {
            int(tid): TeamBoxLine.from_dict(td)
            for tid, td in data["teams"].items()
        }
        players = {
            int(tid): tuple(
                PlayerBoxLine.from_dict(pd) for pd in plist
            )
            for tid, plist in data["players_by_team"].items()
        }
        return cls(
            game_id=str(data["game_id"]),
            game_date=date.fromisoformat(str(data["game_date"])),
            season_year=str(data["season_year"]),
            team_ids=tuple(int(t) for t in data["team_ids"]),
            teams=teams,
            players_by_team=players,
            overtime_periods=int(data["overtime_periods"]),
            source_player_hash=str(data["source_player_hash"]),
            source_team_hash=str(data["source_team_hash"]),
        )


# ---------------------------------------------------------------------------
# Lineup invariant helpers (Section 8.4)
# ---------------------------------------------------------------------------
def validate_lineup(lineup: Tuple[int, ...]) -> None:
    """Enforce the five-unique-player lineup invariant.

    A non-empty lineup must contain exactly five distinct player IDs. Empty
    lineups are permitted and signal that rotation inference was disabled for
    the reconstruction that produced this possession.
    """
    if len(lineup) == 0:
        return
    if len(lineup) != 5:
        raise LineupError(
            f"lineup must have exactly 5 players when non-empty, got {len(lineup)}"
        )
    if len(set(lineup)) != 5:
        raise LineupError(f"lineup must have 5 unique players: {lineup}")


# ---------------------------------------------------------------------------
# Event and possession dataclasses (Section 8.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatentEvent:
    """One inferred event in a plausible possession history.

    ``is_observed_stat`` marks events that correspond to a ledger token and
    therefore must be conserved exactly by aggregation. ``UNKNOWN_BOUNDARY``
    events set ``is_observed_stat=False`` and carry no consumable stat.
    """

    event_id: str
    event_type: EventType
    offense_team_id: int
    defense_team_id: int
    actor_player_id: Optional[int]
    secondary_player_id: Optional[int]
    points: int
    made: Optional[bool]
    shot_value: Optional[int]
    free_throw_number: Optional[int]
    free_throw_total: Optional[int]
    is_team_event: bool
    is_observed_stat: bool
    source_stat: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "offense_team_id": self.offense_team_id,
            "defense_team_id": self.defense_team_id,
            "actor_player_id": self.actor_player_id,
            "secondary_player_id": self.secondary_player_id,
            "points": self.points,
            "made": self.made,
            "shot_value": self.shot_value,
            "free_throw_number": self.free_throw_number,
            "free_throw_total": self.free_throw_total,
            "is_team_event": self.is_team_event,
            "is_observed_stat": self.is_observed_stat,
            "source_stat": self.source_stat,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentEvent":
        return cls(
            event_id=str(data["event_id"]),
            event_type=EventType(str(data["event_type"])),
            offense_team_id=int(data["offense_team_id"]),
            defense_team_id=int(data["defense_team_id"]),
            actor_player_id=(
                None if data["actor_player_id"] is None
                else int(data["actor_player_id"])
            ),
            secondary_player_id=(
                None if data["secondary_player_id"] is None
                else int(data["secondary_player_id"])
            ),
            points=int(data["points"]),
            made=None if data["made"] is None else bool(data["made"]),
            shot_value=(
                None if data["shot_value"] is None else int(data["shot_value"])
            ),
            free_throw_number=(
                None if data["free_throw_number"] is None
                else int(data["free_throw_number"])
            ),
            free_throw_total=(
                None if data["free_throw_total"] is None
                else int(data["free_throw_total"])
            ),
            is_team_event=bool(data["is_team_event"]),
            is_observed_stat=bool(data["is_observed_stat"]),
            source_stat=str(data["source_stat"]),
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class LatentPossession:
    """One inferred possession: a sequence of events with a terminal reason."""

    possession_index: int
    period: int
    seconds_remaining_estimate: float
    offense_team_id: int
    defense_team_id: int
    offense_lineup: Tuple[int, ...]
    defense_lineup: Tuple[int, ...]
    events: Tuple[LatentEvent, ...]
    end_reason: PossessionEnd
    points: int
    duration_seconds: float
    boundary_confidence: float

    def __post_init__(self) -> None:
        validate_lineup(self.offense_lineup)
        validate_lineup(self.defense_lineup)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "possession_index": self.possession_index,
            "period": self.period,
            "seconds_remaining_estimate": self.seconds_remaining_estimate,
            "offense_team_id": self.offense_team_id,
            "defense_team_id": self.defense_team_id,
            "offense_lineup": list(self.offense_lineup),
            "defense_lineup": list(self.defense_lineup),
            "events": [e.to_dict() for e in self.events],
            "end_reason": self.end_reason.value,
            "points": self.points,
            "duration_seconds": self.duration_seconds,
            "boundary_confidence": self.boundary_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentPossession":
        return cls(
            possession_index=int(data["possession_index"]),
            period=int(data["period"]),
            seconds_remaining_estimate=float(data["seconds_remaining_estimate"]),
            offense_team_id=int(data["offense_team_id"]),
            defense_team_id=int(data["defense_team_id"]),
            offense_lineup=tuple(int(p) for p in data["offense_lineup"]),
            defense_lineup=tuple(int(p) for p in data["defense_lineup"]),
            events=tuple(LatentEvent.from_dict(e) for e in data["events"]),
            end_reason=PossessionEnd(str(data["end_reason"])),
            points=int(data["points"]),
            duration_seconds=float(data["duration_seconds"]),
            boundary_confidence=float(data["boundary_confidence"]),
        )


# ---------------------------------------------------------------------------
# Rotation dataclasses (Section 8.5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RotationSlot:
    """A 30-second window with the five players on the floor for one team."""

    period: int
    slot_index: int
    start_second: float
    end_second: float
    team_id: int
    player_ids: Tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.player_ids) != 5:
            raise LineupError(
                f"RotationSlot must have exactly 5 players, got {len(self.player_ids)}"
            )
        if len(set(self.player_ids)) != 5:
            raise LineupError(
                f"RotationSlot must have 5 unique players: {self.player_ids}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "slot_index": self.slot_index,
            "start_second": self.start_second,
            "end_second": self.end_second,
            "team_id": self.team_id,
            "player_ids": list(self.player_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RotationSlot":
        return cls(
            period=int(data["period"]),
            slot_index=int(data["slot_index"]),
            start_second=float(data["start_second"]),
            end_second=float(data["end_second"]),
            team_id=int(data["team_id"]),
            player_ids=tuple(int(p) for p in data["player_ids"]),
        )


@dataclass(frozen=True)
class RotationSample:
    """A full-team rotation sample fitting minutes (and softly plus/minus)."""

    team_id: int
    slots: Tuple[RotationSlot, ...]
    player_minutes: Dict[int, float]
    target_minutes: Dict[int, float]
    max_abs_minute_error: float
    plus_minus_rmse: float
    starter_prior_score: float
    rotation_prior_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "slots": [s.to_dict() for s in self.slots],
            "player_minutes": {str(k): v for k, v in self.player_minutes.items()},
            "target_minutes": {str(k): v for k, v in self.target_minutes.items()},
            "max_abs_minute_error": self.max_abs_minute_error,
            "plus_minus_rmse": self.plus_minus_rmse,
            "starter_prior_score": self.starter_prior_score,
            "rotation_prior_score": self.rotation_prior_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RotationSample":
        return cls(
            team_id=int(data["team_id"]),
            slots=tuple(RotationSlot.from_dict(s) for s in data["slots"]),
            player_minutes={int(k): float(v) for k, v in data["player_minutes"].items()},
            target_minutes={int(k): float(v) for k, v in data["target_minutes"].items()},
            max_abs_minute_error=float(data["max_abs_minute_error"]),
            plus_minus_rmse=float(data["plus_minus_rmse"]),
            starter_prior_score=float(data["starter_prior_score"]),
            rotation_prior_score=float(data["rotation_prior_score"]),
        )


# ---------------------------------------------------------------------------
# Candidate and posterior dataclasses (Section 8.6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReconstructionSample:
    """One constraint-satisfying plausible history for a completed game."""

    game_id: str
    sample_id: str
    seed: int
    possessions: Tuple[LatentPossession, ...]
    home_rotation: Optional[RotationSample]
    away_rotation: Optional[RotationSample]
    hard_errors: Tuple[str, ...]
    soft_penalties: Dict[str, float]
    log_prior: float
    log_likelihood: float
    log_weight: float
    box_score_exact: bool

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "method_version": RECONSTRUCTION_METHOD_VERSION,
            "game_id": self.game_id,
            "sample_id": self.sample_id,
            "seed": self.seed,
            "possessions": [p.to_dict() for p in self.possessions],
            "home_rotation": (
                None if self.home_rotation is None
                else self.home_rotation.to_dict()
            ),
            "away_rotation": (
                None if self.away_rotation is None
                else self.away_rotation.to_dict()
            ),
            "hard_errors": list(self.hard_errors),
            "soft_penalties": dict(self.soft_penalties),
            "log_prior": self.log_prior,
            "log_likelihood": self.log_likelihood,
            "log_weight": self.log_weight,
            "box_score_exact": self.box_score_exact,
        }
        assert_finite_numbers(payload, path=self.sample_id)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructionSample":
        return cls(
            game_id=str(data["game_id"]),
            sample_id=str(data["sample_id"]),
            seed=int(data["seed"]),
            possessions=tuple(
                LatentPossession.from_dict(p) for p in data["possessions"]
            ),
            home_rotation=(
                None if data["home_rotation"] is None
                else RotationSample.from_dict(data["home_rotation"])
            ),
            away_rotation=(
                None if data["away_rotation"] is None
                else RotationSample.from_dict(data["away_rotation"])
            ),
            hard_errors=tuple(str(e) for e in data["hard_errors"]),
            soft_penalties={
                str(k): float(v) for k, v in data["soft_penalties"].items()
            },
            log_prior=float(data["log_prior"]),
            log_likelihood=float(data["log_likelihood"]),
            log_weight=float(data["log_weight"]),
            box_score_exact=bool(data["box_score_exact"]),
        )


@dataclass(frozen=True)
class ReconstructionPosterior:
    """Weighted posterior ensemble of plausible histories for one game."""

    game_id: str
    samples: Tuple[ReconstructionSample, ...]
    normalized_weights: Tuple[float, ...]
    candidate_count: int
    valid_candidate_count: int
    effective_sample_size: float
    quality: ReconstructionQuality
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "method_version": RECONSTRUCTION_METHOD_VERSION,
            "observed_play_by_play": False,
            "interpretation": "latent reconstruction constrained by aggregate box scores",
            "game_id": self.game_id,
            "samples": [s.to_dict() for s in self.samples],
            "normalized_weights": list(self.normalized_weights),
            "candidate_count": self.candidate_count,
            "valid_candidate_count": self.valid_candidate_count,
            "effective_sample_size": self.effective_sample_size,
            "quality": self.quality.value,
            "diagnostics": dict(self.diagnostics),
        }
        assert_finite_numbers(payload, path=f"posterior:{self.game_id}")
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructionPosterior":
        return cls(
            game_id=str(data["game_id"]),
            samples=tuple(
                ReconstructionSample.from_dict(s) for s in data["samples"]
            ),
            normalized_weights=tuple(
                float(w) for w in data["normalized_weights"]
            ),
            candidate_count=int(data["candidate_count"]),
            valid_candidate_count=int(data["valid_candidate_count"]),
            effective_sample_size=float(data["effective_sample_size"]),
            quality=ReconstructionQuality(str(data["quality"])),
            diagnostics=dict(data.get("diagnostics", {})),
        )


# Re-export ``replace`` so downstream modules construct modified copies of the
# frozen dataclasses without importing dataclasses directly.
__all__.append("replace")
