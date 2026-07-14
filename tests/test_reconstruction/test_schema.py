"""Tests for the reconstruction domain schema (PR 01).

Covers the PR 01 exit gate:
* Round-trip every persisted dataclass through strict JSON.
* NaN/infinity are rejected before serialization.
* Possession lineups enforce the five-unique-player invariant.
* Schema/method versions appear in top-level persisted payloads.

No sampling logic exists yet — these tests exercise only ``schema.py``.
"""

from __future__ import annotations

import json
import math
from datetime import date

import pytest

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


# ---------------------------------------------------------------------------
# Fixture builders — small synthetic objects used across the tests.
# ---------------------------------------------------------------------------
def make_player(player_id=101, **overrides) -> PlayerBoxLine:
    defaults = dict(
        game_id="0022400001",
        team_id=1610612737,
        player_id=player_id,
        player_name="Test Player",
        minutes=30.0,
        fgm=5,
        fga=10,
        fg3m=2,
        fg3a=5,
        ftm=3,
        fta=4,
        oreb=1,
        dreb=4,
        reb=5,
        ast=6,
        tov=2,
        stl=1,
        blk=0,
        blka=1,
        pf=3,
        pfd=2,
        pts=15,
        plus_minus=4.0,
    )
    defaults.update(overrides)
    return PlayerBoxLine(**defaults)


def make_team(team_id=1610612737, **overrides) -> TeamBoxLine:
    defaults = dict(
        game_id="0022400001",
        team_id=team_id,
        team_abbreviation="ATL",
        game_date=date(2025, 1, 1),
        matchup="ATL vs. BOS",
        minutes=240.0,
        fgm=40,
        fga=85,
        fg3m=12,
        fg3a=32,
        ftm=18,
        fta=24,
        oreb=8,
        dreb=30,
        reb=38,
        ast=24,
        tov=14,
        stl=7,
        blk=4,
        blka=6,
        pf=20,
        pfd=22,
        pts=110,
        team_plus_minus=5.0,
    )
    defaults.update(overrides)
    return TeamBoxLine(**defaults)


def make_game() -> GameBoxScore:
    home_id, away_id = 1610612737, 1610612738
    players = {
        home_id: (make_player(101), make_player(102, minutes=20.0, pts=8)),
        away_id: (make_player(201, team_id=away_id), make_player(202, team_id=away_id, minutes=20.0)),
    }
    return GameBoxScore(
        game_id="0022400001",
        game_date=date(2025, 1, 1),
        season_year="2024-25",
        team_ids=(home_id, away_id),
        teams={home_id: make_team(home_id), away_id: make_team(away_id, team_abbreviation="BOS", team_plus_minus=-5.0)},
        players_by_team=players,
        overtime_periods=0,
        source_player_hash="abc",
        source_team_hash="def",
    )


def make_event(event_id="evt1") -> LatentEvent:
    return LatentEvent(
        event_id=event_id,
        event_type=EventType.FG2_MADE,
        offense_team_id=1610612737,
        defense_team_id=1610612738,
        actor_player_id=101,
        secondary_player_id=None,
        points=2,
        made=True,
        shot_value=2,
        free_throw_number=None,
        free_throw_total=None,
        is_team_event=False,
        is_observed_stat=True,
        source_stat="fg2_made",
        confidence=1.0,
    )


def make_possession(lineup=(101, 102, 103, 104, 105)) -> LatentPossession:
    return LatentPossession(
        possession_index=0,
        period=1,
        seconds_remaining_estimate=700.0,
        offense_team_id=1610612737,
        defense_team_id=1610612738,
        offense_lineup=lineup,
        defense_lineup=(201, 202, 203, 204, 205),
        events=(make_event(),),
        end_reason=PossessionEnd.MADE_FIELD_GOAL,
        points=2,
        duration_seconds=14.5,
        boundary_confidence=0.9,
    )


def make_slot(slot_index=0) -> RotationSlot:
    return RotationSlot(
        period=1,
        slot_index=slot_index,
        start_second=0.0,
        end_second=30.0,
        team_id=1610612737,
        player_ids=(101, 102, 103, 104, 105),
    )


def make_rotation() -> RotationSample:
    return RotationSample(
        team_id=1610612737,
        slots=(make_slot(0), make_slot(1)),
        player_minutes={101: 30.0, 102: 30.0},
        target_minutes={101: 30.0, 102: 30.0},
        max_abs_minute_error=0.5,
        plus_minus_rmse=2.0,
        starter_prior_score=-1.0,
        rotation_prior_score=-2.0,
    )


def make_sample() -> ReconstructionSample:
    return ReconstructionSample(
        game_id="0022400001",
        sample_id="s0",
        seed=42,
        possessions=(make_possession(),),
        home_rotation=make_rotation(),
        away_rotation=None,
        hard_errors=(),
        soft_penalties={"plus_minus_penalty": 0.5},
        log_prior=-1.0,
        log_likelihood=-2.0,
        log_weight=-3.0,
        box_score_exact=True,
    )


def make_posterior() -> ReconstructionPosterior:
    sample = make_sample()
    return ReconstructionPosterior(
        game_id="0022400001",
        samples=(sample,),
        normalized_weights=(1.0,),
        candidate_count=8,
        valid_candidate_count=8,
        effective_sample_size=1.0,
        quality=ReconstructionQuality.VALID_HIGH,
        diagnostics={"unknown_boundary_rate": 0.0},
    )


# ---------------------------------------------------------------------------
# Strict JSON round-trip helpers
# ---------------------------------------------------------------------------
def strict_roundtrip(obj):
    """to_dict -> strict json (allow_nan=False) -> from_dict and compare equality."""
    payload = obj.to_dict()
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    restored_payload = json.loads(text)
    rebuilt = type(obj).from_dict(restored_payload)
    assert rebuilt == obj, f"round-trip changed object: {rebuilt!r} != {obj!r}"
    return payload


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------
def test_version_constants_present():
    assert RECONSTRUCTION_SCHEMA_VERSION == "reconstruction_schema_v1"
    assert RECONSTRUCTION_METHOD_VERSION == "boxscore_posterior_v1"


def test_enums_serialize_as_their_string_values():
    assert EventType.FG2_MADE.value == "fg2_made"
    assert EventType.UNKNOWN_BOUNDARY.value == "unknown_boundary"
    assert PossessionEnd.MADE_FIELD_GOAL.value == "made_field_goal"
    assert ReconstructionQuality.UNUSABLE.value == "unusable"
    # String-enum identity so ``EventType("fg2_made")`` round-trips.
    assert EventType(EventType.FG2_MADE.value) is EventType.FG2_MADE


# ---------------------------------------------------------------------------
# Round-trip every persisted dataclass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "builder",
    [
        make_player,
        make_team,
        make_game,
        make_event,
        make_possession,
        make_slot,
        make_rotation,
        make_sample,
        make_posterior,
    ],
)
def test_dataclass_strict_json_roundtrip(builder):
    obj = builder()
    strict_roundtrip(obj)


def test_tuples_serialize_as_json_lists():
    """Tuples must become JSON lists (not raise) and round-trip back to tuples."""
    payload = make_possession().to_dict()
    assert isinstance(payload["offense_lineup"], list)
    assert isinstance(payload["events"], list)
    # game box score team ids tuple
    game_payload = make_game().to_dict()
    assert isinstance(game_payload["team_ids"], list)


def test_date_serializes_as_iso_string():
    payload = make_team().to_dict()
    assert payload["game_date"] == "2025-01-01"
    assert TeamBoxLine.from_dict(payload).game_date == date(2025, 1, 1)


def test_optional_fields_roundtrip_with_none():
    evt = make_event()
    assert evt.secondary_player_id is None
    payload = evt.to_dict()
    assert payload["secondary_player_id"] is None
    rebuilt = LatentEvent.from_dict(payload)
    assert rebuilt.secondary_player_id is None
    assert rebuilt == evt


def test_computed_two_point_properties():
    p = make_player(fgm=7, fga=15, fg3m=2, fg3a=6)
    assert p.fg2m == 5
    assert p.fg2a == 9
    assert p.fg2_missed == 4
    assert p.fg3_missed == 4
    assert p.fg_missed == 8


# ---------------------------------------------------------------------------
# NaN / Infinity rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_posterior_rejects_non_finite_weights(bad_value):
    posterior = ReconstructionPosterior(
        game_id="g",
        samples=(),
        normalized_weights=(bad_value,),
        candidate_count=1,
        valid_candidate_count=1,
        effective_sample_size=1.0,
        quality=ReconstructionQuality.VALID_LOW,
    )
    with pytest.raises(FiniteNumberError):
        posterior.to_dict()


def test_sample_rejects_non_finite_log_weight():
    sample = make_sample()
    sample = ReconstructionSample(
        game_id=sample.game_id,
        sample_id=sample.sample_id,
        seed=sample.seed,
        possessions=sample.possessions,
        home_rotation=sample.home_rotation,
        away_rotation=sample.away_rotation,
        hard_errors=sample.hard_errors,
        soft_penalties=sample.soft_penalties,
        log_prior=sample.log_prior,
        log_likelihood=sample.log_likelihood,
        log_weight=float("nan"),
        box_score_exact=True,
    )
    with pytest.raises(FiniteNumberError):
        sample.to_dict()


def test_assert_finite_numbers_rejects_nested_nan():
    nested = {"a": [1, 2, {"b": float("inf")}]}
    with pytest.raises(FiniteNumberError):
        assert_finite_numbers(nested)


def test_assert_finite_numbers_accepts_clean_nested():
    nested = {"a": [1, 2, {"b": 3.5}], "c": (4, 5)}
    assert_finite_numbers(nested)  # no raise


def test_assert_finite_numbers_walks_dataclass_fields():
    p = make_player()
    # Clean object passes.
    assert_finite_numbers(p)
    # Mutate via rebuild with a NaN minute.
    bad = PlayerBoxLine(
        game_id=p.game_id, team_id=p.team_id, player_id=p.player_id,
        player_name=p.player_name, minutes=float("nan"),
        fgm=p.fgm, fga=p.fga, fg3m=p.fg3m, fg3a=p.fg3a,
        ftm=p.ftm, fta=p.fta, oreb=p.oreb, dreb=p.dreb, reb=p.reb,
        ast=p.ast, tov=p.tov, stl=p.stl, blk=p.blk, blka=p.blka,
        pf=p.pf, pfd=p.pfd, pts=p.pts, plus_minus=p.plus_minus,
    )
    with pytest.raises(FiniteNumberError):
        assert_finite_numbers(bad)


def test_strict_json_itself_rejects_nan_payload():
    """Even if to_dict somehow produced NaN, strict json.dumps must fail."""
    payload = {"x": float("nan")}
    with pytest.raises(ValueError):
        json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# Lineup invariants
# ---------------------------------------------------------------------------
def test_empty_lineup_allowed():
    # rotation_mode="disabled" case.
    poss = LatentPossession(
        possession_index=0, period=1, seconds_remaining_estimate=700.0,
        offense_team_id=1, defense_team_id=2,
        offense_lineup=(), defense_lineup=(),
        events=(), end_reason=PossessionEnd.UNKNOWN,
        points=0, duration_seconds=0.0, boundary_confidence=0.0,
    )
    assert poss.offense_lineup == ()


def test_lineup_with_wrong_length_rejected():
    with pytest.raises(LineupError):
        LatentPossession(
            possession_index=0, period=1, seconds_remaining_estimate=700.0,
            offense_team_id=1, defense_team_id=2,
            offense_lineup=(1, 2, 3, 4),  # only four
            defense_lineup=(5, 6, 7, 8, 9),
            events=(), end_reason=PossessionEnd.UNKNOWN,
            points=0, duration_seconds=0.0, boundary_confidence=0.0,
        )


def test_lineup_with_duplicates_rejected():
    with pytest.raises(LineupError):
        LatentPossession(
            possession_index=0, period=1, seconds_remaining_estimate=700.0,
            offense_team_id=1, defense_team_id=2,
            offense_lineup=(1, 1, 2, 3, 4),  # duplicate
            defense_lineup=(5, 6, 7, 8, 9),
            events=(), end_reason=PossessionEnd.UNKNOWN,
            points=0, duration_seconds=0.0, boundary_confidence=0.0,
        )


def test_rotation_slot_requires_five_unique():
    with pytest.raises(LineupError):
        RotationSlot(period=1, slot_index=0, start_second=0.0, end_second=30.0,
                     team_id=1, player_ids=(1, 2, 3, 4, 4))
    with pytest.raises(LineupError):
        RotationSlot(period=1, slot_index=0, start_second=0.0, end_second=30.0,
                     team_id=1, player_ids=(1, 2, 3, 4))


# ---------------------------------------------------------------------------
# Version presence in persisted payloads
# ---------------------------------------------------------------------------
def test_posterior_payload_carries_versions_and_disclaimer():
    payload = make_posterior().to_dict()
    assert payload["schema_version"] == RECONSTRUCTION_SCHEMA_VERSION
    assert payload["method_version"] == RECONSTRUCTION_METHOD_VERSION
    assert payload["observed_play_by_play"] is False
    assert payload["interpretation"] == "latent reconstruction constrained by aggregate box scores"


def test_sample_payload_carries_versions():
    payload = make_sample().to_dict()
    assert payload["schema_version"] == RECONSTRUCTION_SCHEMA_VERSION
    assert payload["method_version"] == RECONSTRUCTION_METHOD_VERSION


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------
def test_dataclasses_are_frozen():
    p = make_player()
    with pytest.raises(Exception):
        p.pts = 99  # type: ignore[misc]
