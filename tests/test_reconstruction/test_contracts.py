"""Tests for reconstruction hard-contract validation (PR 02).

Exercises ``validate_game_box_score`` and the individual constraint helpers
directly on synthetic ``GameBoxScore`` / ``PlayerBoxLine`` / ``TeamBoxLine``
objects built via the schema, independent of CSV canonicalization. This
isolates the arithmetic and cross-team logic from row parsing.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.reconstruction.contracts import (
    MAX_OT_PERIODS,
    GameValidationReport,
    infer_overtime,
    validate_game_box_score,
    validate_player_line,
    validate_team_line,
)
from src.reconstruction.schema import (
    GameBoxScore,
    PlayerBoxLine,
    TeamBoxLine,
)


# ---------------------------------------------------------------------------
# Builders — minimal valid objects, mutated per test.
# ---------------------------------------------------------------------------
def _player(**kw) -> PlayerBoxLine:
    base = dict(
        game_id="g1", team_id=1001, player_id=11, player_name="P",
        minutes=30.0, fgm=5, fga=10, fg3m=2, fg3a=5, ftm=3, fta=4,
        oreb=1, dreb=4, reb=5, ast=6, tov=2, stl=1, blk=0, blka=1,
        pf=3, pfd=2, pts=15, plus_minus=4.0,
    )
    base.update(kw)
    return PlayerBoxLine(**base)


def _team(team_id=1001, **kw) -> TeamBoxLine:
    base = dict(
        game_id="g1", team_id=team_id, team_abbreviation="HOM",
        game_date=date(2025, 1, 1), matchup="HOM vs. AWY",
        minutes=240.0, fgm=40, fga=85, fg3m=12, fg3a=32, ftm=18, fta=24,
        oreb=8, dreb=30, reb=38, ast=24, tov=14, stl=7, blk=4, blka=6,
        pf=20, pfd=22, pts=110, team_plus_minus=5.0,
    )
    base.update(kw)
    return TeamBoxLine(**base)


def _team_from_players(players, team_id, abbr, matchup, team_pm, tov_residual=0):
    """Build a team row whose stat totals equal the summed player totals.

    ``team_pm`` is set explicitly by the caller so that
    ``team_plus_minus == pts - opp_pts`` and ``sum(player pm) == 5*team_pm``
    both hold (the caller distributes player pm accordingly).
    """
    return TeamBoxLine(
        game_id="g1", team_id=team_id, team_abbreviation=abbr,
        game_date=date(2025, 1, 1), matchup=matchup,
        minutes=240.0,
        fgm=sum(p.fgm for p in players),
        fga=sum(p.fga for p in players),
        fg3m=sum(p.fg3m for p in players),
        fg3a=sum(p.fg3a for p in players),
        ftm=sum(p.ftm for p in players),
        fta=sum(p.fta for p in players),
        oreb=sum(p.oreb for p in players),
        dreb=sum(p.dreb for p in players),
        reb=sum(p.reb for p in players),
        ast=sum(p.ast for p in players),
        tov=sum(p.tov for p in players) + tov_residual,
        stl=sum(p.stl for p in players),
        blk=sum(p.blk for p in players),
        blka=sum(p.blka for p in players),
        pf=sum(p.pf for p in players),
        pfd=sum(p.pfd for p in players),
        pts=sum(p.pts for p in players),
        team_plus_minus=team_pm,
    )


def _set_player_pm(players, target_team_pm):
    """Return new player tuples whose plus/minus sums to ``5 * target_team_pm``.

    The delta is applied to the first player so every other value is preserved.
    """
    target = 5.0 * target_team_pm
    current = sum(p.plus_minus for p in players)
    delta = target - current
    first = players[0]
    new_first = PlayerBoxLine(
        game_id=first.game_id, team_id=first.team_id, player_id=first.player_id,
        player_name=first.player_name, minutes=first.minutes, fgm=first.fgm,
        fga=first.fga, fg3m=first.fg3m, fg3a=first.fg3a, ftm=first.ftm,
        fta=first.fta, oreb=first.oreb, dreb=first.dreb, reb=first.reb,
        ast=first.ast, tov=first.tov, stl=first.stl, blk=first.blk,
        blka=first.blka, pf=first.pf, pfd=first.pfd, pts=first.pts,
        plus_minus=first.plus_minus + delta,
    )
    return (new_first,) + tuple(players[1:])


def _valid_game() -> GameBoxScore:
    """A fully-consistent game: player sums == team totals, 240 min/team, 0 OT.

    Built so every hard constraint holds by construction:
    * each player's PTS == 2*FG2M + 3*FG3M + FTM
    * team rows == summed player rows
    * sum(player pm) == 5 * team_pm == 5 * (pts - opp_pts)
    * steals/blocks/assists within bounds
    * BLKA set to the opponent's BLK
    """
    pa = [
        _player(player_id=11, minutes=200.0, fgm=5, fga=10, fg3m=2, fg3a=5,
                ftm=3, fta=4, oreb=1, dreb=4, reb=5, ast=6, tov=2, stl=1,
                blk=0, blka=1, pf=3, pfd=2, pts=15, plus_minus=10.0),
        _player(player_id=12, team_id=1001, minutes=40.0, fgm=3, fga=6, fg3m=1,
                fg3a=2, ftm=1, fta=1, oreb=0, dreb=2, reb=2, ast=2, tov=1,
                stl=0, blk=1, blka=0, pf=2, pfd=1, pts=8, plus_minus=-5.0),
    ]
    pb = [
        _player(player_id=21, team_id=1002, minutes=200.0, fgm=4, fga=9, fg3m=1,
                fg3a=3, ftm=2, fta=3, oreb=1, dreb=5, reb=6, ast=5, tov=3,
                stl=2, blk=1, blka=0, pf=4, pfd=3, pts=11, plus_minus=-8.0),
        _player(player_id=22, team_id=1002, minutes=40.0, fgm=2, fga=5, fg3m=0,
                fg3a=1, ftm=0, fta=0, oreb=0, dreb=1, reb=1, ast=1, tov=0,
                stl=0, blk=0, blka=1, pf=1, pfd=0, pts=4, plus_minus=3.0),
    ]
    pts_a = sum(p.pts for p in pa)
    pts_b = sum(p.pts for p in pb)
    team_pm_a = float(pts_a - pts_b)
    team_pm_b = -team_pm_a
    # Distribute player pm so sum == 5 * team_pm.
    pa = _set_player_pm(pa, team_pm_a)
    pb = _set_player_pm(pb, team_pm_b)
    ta = _team_from_players(pa, 1001, "HOM", "HOM vs. AWY", team_pm_a)
    tb = _team_from_players(pb, 1002, "AWY", "AWY @ HOM", team_pm_b)
    # Cross-team BLKA consistency: each team's BLKA = opponent's BLK.
    ta = _team(team_id=1001, team_abbreviation="HOM", matchup="HOM vs. AWY",
               fgm=ta.fgm, fga=ta.fga, fg3m=ta.fg3m, fg3a=ta.fg3a, ftm=ta.ftm,
               fta=ta.fta, oreb=ta.oreb, dreb=ta.dreb, reb=ta.reb, ast=ta.ast,
               tov=ta.tov, stl=ta.stl, blk=ta.blk, blka=tb.blk, pf=ta.pf,
               pfd=ta.pfd, pts=ta.pts, team_plus_minus=ta.team_plus_minus)
    tb = _team(team_id=1002, team_abbreviation="AWY", matchup="AWY @ HOM",
               fgm=tb.fgm, fga=tb.fga, fg3m=tb.fg3m, fg3a=tb.fg3a, ftm=tb.ftm,
               fta=tb.fta, oreb=tb.oreb, dreb=tb.dreb, reb=tb.reb, ast=tb.ast,
               tov=tb.tov, stl=tb.stl, blk=tb.blk, blka=ta.blk, pf=tb.pf,
               pfd=tb.pfd, pts=tb.pts, team_plus_minus=tb.team_plus_minus)
    return GameBoxScore(
        game_id="g1", game_date=date(2025, 1, 1), season_year="2024-25",
        team_ids=(1001, 1002),
        teams={1001: ta, 1002: tb},
        players_by_team={1001: tuple(pa), 1002: tuple(pb)},
        overtime_periods=0,
        source_player_hash="p", source_team_hash="t",
    )


# ---------------------------------------------------------------------------
# GameValidationReport basics
# ---------------------------------------------------------------------------
def test_report_ok_when_no_errors():
    r = GameValidationReport(game_id="g")
    assert r.ok is True
    r.hard_errors.append("x")
    assert r.ok is False


def test_report_to_dict_roundtrip_shape():
    r = GameValidationReport(game_id="g", hard_errors=["e"], warnings=["w"], metrics={"m": 1.0})
    d = r.to_dict()
    assert d["game_id"] == "g"
    assert d["ok"] is False
    assert d["hard_errors"] == ["e"]
    assert d["warnings"] == ["w"]
    assert d["metrics"] == {"m": 1.0}


# ---------------------------------------------------------------------------
# Per-row arithmetic
# ---------------------------------------------------------------------------
def test_valid_player_line_no_errors():
    r = GameValidationReport(game_id="g")
    validate_player_line(_player(), r)
    assert r.hard_errors == []


def test_player_points_identity_violation():
    r = GameValidationReport(game_id="g")
    p = _player(pts=99)  # PTS doesn't match shooting
    validate_player_line(p, r)
    assert any("PTS" in e for e in r.hard_errors)


def test_player_rebound_identity_violation():
    r = GameValidationReport(game_id="g")
    p = _player(oreb=2, dreb=2, reb=99)
    validate_player_line(p, r)
    assert any("REB" in e for e in r.hard_errors)


def test_player_made_over_attempt():
    r = GameValidationReport(game_id="g")
    p = _player(fgm=10, fga=5)
    validate_player_line(p, r)
    assert any("FGM" in e for e in r.hard_errors)


def test_player_negative_count():
    r = GameValidationReport(game_id="g")
    p = _player(ast=-1)
    validate_player_line(p, r)
    assert any("nonnegative integer" in e for e in r.hard_errors)


def test_player_fg3m_exceeds_fgm():
    r = GameValidationReport(game_id="g")
    p = _player(fgm=1, fg3m=2)
    validate_player_line(p, r)
    assert any("FG3M<=FGM" in e for e in r.hard_errors)


def test_team_line_valid():
    r = GameValidationReport(game_id="g")
    validate_team_line(_team(), r)
    assert r.hard_errors == []


# ---------------------------------------------------------------------------
# Overtime inference
# ---------------------------------------------------------------------------
def test_infer_overtime_regulation():
    r = GameValidationReport(game_id="g")
    ot = infer_overtime(240.0, 240.0, r)
    assert ot == 0
    assert r.ok


def test_infer_overtime_one_period():
    r = GameValidationReport(game_id="g")
    ot = infer_overtime(265.0, 265.0, r)
    assert ot == 1
    assert r.ok


def test_infer_overtime_two_periods():
    r = GameValidationReport(game_id="g")
    ot = infer_overtime(290.0, 290.0, r)
    assert ot == 2
    assert r.ok


def test_infer_overtime_disagreement_rejected():
    r = GameValidationReport(game_id="g")
    ot = infer_overtime(240.0, 265.0, r)
    assert not r.ok
    assert "disagrees" in r.hard_errors[0]
    assert ot == 0


def test_infer_overtime_minute_error_rejected():
    r = GameValidationReport(game_id="g")
    # 250 min -> raw_ot = 0.4 -> rounds to 0 -> expected 240 -> err 10 > 1
    infer_overtime(250.0, 250.0, r)
    assert not r.ok


def test_infer_overtime_within_tolerance():
    r = GameValidationReport(game_id="g")
    # 240.5 -> raw_ot 0.02 -> round 0 -> expected 240 -> err 0.5 <= 1.0 OK
    ot = infer_overtime(240.5, 240.5, r)
    assert r.ok
    assert ot == 0


def test_infer_overtime_exceeds_max_rejected():
    r = GameValidationReport(game_id="g")
    big = 240.0 + 25.0 * (MAX_OT_PERIODS + 2)
    infer_overtime(big, big, r)
    assert not r.ok


# ---------------------------------------------------------------------------
# Cross-team constraints
# ---------------------------------------------------------------------------
def _swap_team(game: GameBoxScore, team_id: int, **changes) -> GameBoxScore:
    """Return a copy of ``game`` with one team's fields replaced.

    Keeps every other team stat identical so the only new hard errors are the
    ones the test intends to provoke.
    """
    from dataclasses import replace as _replace
    new_team = _replace(game.teams[team_id], **changes)
    return GameBoxScore(
        game_id=game.game_id, game_date=game.game_date, season_year=game.season_year,
        team_ids=game.team_ids, teams={**game.teams, team_id: new_team},
        players_by_team=game.players_by_team,
        overtime_periods=game.overtime_periods,
        source_player_hash=game.source_player_hash,
        source_team_hash=game.source_team_hash,
    )


def test_valid_game_passes_full_validation():
    game = _valid_game()
    report = validate_game_box_score(game)
    assert report.ok, f"unexpected errors: {report.hard_errors}"


def test_cross_team_steals_exceed_opponent_tov():
    game = _valid_game()
    # Inflate team A steals above team B turnovers (3).
    game2 = _swap_team(game, 1001, stl=99)
    report = validate_game_box_score(game2)
    assert any("STL" in e and "opponent TOV" in e for e in report.hard_errors)


def test_cross_team_blocks_exceed_opponent_misses():
    game = _valid_game()
    game2 = _swap_team(game, 1001, blk=99)
    report = validate_game_box_score(game2)
    assert any("BLK" in e and "missed FGA" in e for e in report.hard_errors)


def test_assists_exceed_made_field_goals():
    game = _valid_game()
    game2 = _swap_team(game, 1001, ast=99)
    report = validate_game_box_score(game2)
    assert any("AST" in e and "FGM" in e for e in report.hard_errors)


def test_plus_minus_not_antisymmetric():
    game = _valid_game()
    tb = game.teams[1002]
    # Flip team B's plus/minus sign so A+B != 0.
    game2 = _swap_team(game, 1002, team_plus_minus=abs(tb.team_plus_minus))
    report = validate_game_box_score(game2)
    assert any("antisymmetric" in e for e in report.hard_errors)


def test_large_turnover_residual_is_warning_not_error():
    game = _valid_game()
    # Team A player TOV sum is 3; set team TOV to 12 -> residual 9 (> 8).
    game2 = _swap_team(game, 1001, tov=12)
    report = validate_game_box_score(game2)
    assert any("residual" in w for w in report.warnings)
    # A large residual is only a warning, so no hard error is introduced.
    assert report.ok
