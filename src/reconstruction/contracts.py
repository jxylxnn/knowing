"""Hard validation contracts for canonicalized box scores.

Implements the arithmetic and structural guarantees from Section 9 of the
reconstruction plan. A ``GameBoxScore`` only reaches the reconstruction engine
after ``validate_game_box_score`` reports ``ok=True`` — i.e. every hard
constraint holds. Invalid games are quarantined with a structured report
rather than crashing the batch.

Constraint layers
-----------------
* per-row arithmetic (Section 9.3): shot/attempt ordering, points identity,
  rebound identity, nonnegativity, integer counts.
* player-vs-team sums (Section 9.3): every conserved stat except turnovers
  must match exactly; turnovers allow a nonnegative team residual.
* cross-team (Section 9.4): plus/minus identity, steals <= opponent turnovers,
  blocks <= opponent missed field goals, assists <= made field goals, and the
  BLKA-vs-opponent-blocks consistency warning.
* overtime (Section 9.5): both teams must infer the same overtime period count
  from summed player minutes, within a 1.0-minute tolerance.

All numeric comparisons on plus/minus use tolerance ``1e-6`` (Section 9.4).
Turnover residuals above 8 are a *warning* in schema v1, not an error, because
future source data may differ from the audited sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.reconstruction.schema import GameBoxScore, PlayerBoxLine, TeamBoxLine

__all__ = [
    "PLUS_MINUS_TOLERANCE",
    "MAX_OT_PERIODS",
    "TEAM_TURNOVER_RESIDUAL_WARN",
    "GameValidationReport",
    "validate_game_box_score",
    "validate_player_line",
    "validate_team_line",
    "validate_cross_team",
    "infer_overtime",
]

# Tolerance for plus/minus identity checks (Section 9.4).
PLUS_MINUS_TOLERANCE = 1e-6

# Maximum supported overtime periods in schema v1 (Section 9.5).
MAX_OT_PERIODS = 6

# A team turnover residual above this is a warning, not an error (Section 9.3).
TEAM_TURNOVER_RESIDUAL_WARN = 8

# Minutes per regulation game and per overtime period (Section 9.5 / 3.3).
_REGULATION_TEAM_MINUTES = 240.0
_OT_TEAM_MINUTES = 25.0
_MINUTE_TOLERANCE = 1.0

# Stats whose player sums must exactly equal the team row (Section 9.3).
_EXACT_SUM_STATS: Tuple[str, ...] = (
    "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "pf", "pfd",
)


@dataclass
class GameValidationReport:
    """Structured result of validating one canonicalized game.

    ``ok`` is True only when ``hard_errors`` is empty. ``warnings`` carry
    non-fatal anomalies (large turnover residual, BLKA mismatch) that the
    reconstruction engine and reports may still surface.
    """

    game_id: str
    hard_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.hard_errors

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "ok": self.ok,
            "hard_errors": list(self.hard_errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


# ---------------------------------------------------------------------------
# Per-row arithmetic (Section 9.3)
# ---------------------------------------------------------------------------
def _is_nonneg_int(value: object) -> bool:
    """True when ``value`` is an int (not bool) and >= 0."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float) and value.is_integer():
        return value >= 0
    return False


def validate_player_line(player: PlayerBoxLine, report: GameValidationReport) -> None:
    """Append hard errors for any per-player arithmetic violation."""
    pid = player.player_id
    prefix = f"player {pid}"

    # Nonnegative integer counts.
    for stat in (
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "oreb", "dreb", "reb", "ast", "tov", "stl", "blk",
        "blka", "pf", "pfd", "pts",
    ):
        val = getattr(player, stat)
        if not _is_nonneg_int(val):
            report.hard_errors.append(
                f"{prefix}: {stat}={val} must be a nonnegative integer"
            )

    if player.minutes < 0:
        report.hard_errors.append(f"{prefix}: MIN={player.minutes} must be >= 0")

    # Shot/attempt ordering.
    if not (0 <= player.fgm and player.fgm <= player.fga):
        report.hard_errors.append(
            f"{prefix}: require 0<=FGM<=FGA, got FGM={player.fgm} FGA={player.fga}"
        )
    if not (0 <= player.fg3m and player.fg3m <= player.fg3a):
        report.hard_errors.append(
            f"{prefix}: require 0<=FG3M<=FG3A, got FG3M={player.fg3m} FG3A={player.fg3a}"
        )
    if player.fg3m > player.fgm:
        report.hard_errors.append(
            f"{prefix}: require FG3M<=FGM, got FG3M={player.fg3m} FGM={player.fgm}"
        )
    if player.fg3a > player.fga:
        report.hard_errors.append(
            f"{prefix}: require FG3A<=FGA, got FG3A={player.fg3a} FGA={player.fga}"
        )
    if not (0 <= player.ftm <= player.fta):
        report.hard_errors.append(
            f"{prefix}: require 0<=FTM<=FTA, got FTM={player.ftm} FTA={player.fta}"
        )

    # Two-point derivation must stay nonnegative.
    if player.fg2m < 0:
        report.hard_errors.append(
            f"{prefix}: FG2M=FGM-FG3M={player.fg2m} must be >= 0"
        )
    if player.fg2a < player.fg2m:
        report.hard_errors.append(
            f"{prefix}: FG2A=FGA-FG3A={player.fg2a} must be >= FG2M={player.fg2m}"
        )

    # Points identity: PTS = 2*FG2M + 3*FG3M + FTM.
    expected_pts = 2 * player.fg2m + 3 * player.fg3m + player.ftm
    if player.pts != expected_pts:
        report.hard_errors.append(
            f"{prefix}: PTS={player.pts} != 2*FG2M+3*FG3M+FTM={expected_pts}"
        )

    # Rebound identity.
    if player.reb != player.oreb + player.dreb:
        report.hard_errors.append(
            f"{prefix}: REB={player.reb} != OREB+DREB={player.oreb + player.dreb}"
        )


def validate_team_line(team: TeamBoxLine, report: GameValidationReport) -> None:
    """Append hard errors for any per-team arithmetic violation (same rules)."""
    tid = team.team_id
    prefix = f"team {tid}"

    for stat in (
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "oreb", "dreb", "reb", "ast", "tov", "stl", "blk",
        "blka", "pf", "pfd", "pts",
    ):
        val = getattr(team, stat)
        if not _is_nonneg_int(val):
            report.hard_errors.append(
                f"{prefix}: {stat}={val} must be a nonnegative integer"
            )

    if team.minutes < 0:
        report.hard_errors.append(f"{prefix}: MIN={team.minutes} must be >= 0")

    if not (0 <= team.fgm <= team.fga):
        report.hard_errors.append(
            f"{prefix}: require 0<=FGM<=FGA, got FGM={team.fgm} FGA={team.fga}"
        )
    if not (0 <= team.fg3m <= team.fg3a):
        report.hard_errors.append(
            f"{prefix}: require 0<=FG3M<=FG3A, got FG3M={team.fg3m} FG3A={team.fg3a}"
        )
    if team.fg3m > team.fgm:
        report.hard_errors.append(
            f"{prefix}: require FG3M<=FGM, got FG3M={team.fg3m} FGM={team.fgm}"
        )
    if team.fg3a > team.fga:
        report.hard_errors.append(
            f"{prefix}: require FG3A<=FGA, got FG3A={team.fg3a} FGA={team.fga}"
        )
    if not (0 <= team.ftm <= team.fta):
        report.hard_errors.append(
            f"{prefix}: require 0<=FTM<=FTA, got FTM={team.ftm} FTA={team.fta}"
        )
    if team.fg2m < 0:
        report.hard_errors.append(
            f"{prefix}: FG2M=FGM-FG3M={team.fg2m} must be >= 0"
        )
    if team.fg2a < team.fg2m:
        report.hard_errors.append(
            f"{prefix}: FG2A=FGA-FG3A={team.fg2a} must be >= FG2M={team.fg2m}"
        )

    expected_pts = 2 * team.fg2m + 3 * team.fg3m + team.ftm
    if team.pts != expected_pts:
        report.hard_errors.append(
            f"{prefix}: PTS={team.pts} != 2*FG2M+3*FG3M+FTM={expected_pts}"
        )
    if team.reb != team.oreb + team.dreb:
        report.hard_errors.append(
            f"{prefix}: REB={team.reb} != OREB+DREB={team.oreb + team.dreb}"
        )


# ---------------------------------------------------------------------------
# Player-vs-team sums (Section 9.3) and cross-team (Section 9.4)
# ---------------------------------------------------------------------------
def _player_sums(players: Tuple[PlayerBoxLine, ...]) -> Dict[str, int]:
    sums: Dict[str, int] = {}
    for stat in _EXACT_SUM_STATS + ("tov",):
        sums[stat] = sum(getattr(p, stat) for p in players)
    sums["minutes"] = sum(p.minutes for p in players)
    sums["plus_minus"] = sum(p.plus_minus for p in players)
    return sums


def validate_cross_team(game: GameBoxScore, report: GameValidationReport) -> None:
    """Append errors for sum mismatches and cross-team invariant violations."""
    if len(game.team_ids) != 2:
        report.hard_errors.append(
            f"expected exactly 2 team ids, got {list(game.team_ids)}"
        )
        return

    id_a, id_b = game.team_ids
    team_a, team_b = game.teams[id_a], game.teams[id_b]
    players_a = game.players_by_team.get(id_a, ())
    players_b = game.players_by_team.get(id_b, ())
    sum_a = _player_sums(players_a)
    sum_b = _player_sums(players_b)

    # Exact-sum stats (turnovers handled separately).
    for stat in _EXACT_SUM_STATS:
        team_val = getattr(team_a, stat)
        if sum_a[stat] != team_val:
            report.hard_errors.append(
                f"team {id_a}: summed player {stat.upper()}={sum_a[stat]} "
                f"!= team {stat.upper()}={team_val}"
            )
        team_val = getattr(team_b, stat)
        if sum_b[stat] != team_val:
            report.hard_errors.append(
                f"team {id_b}: summed player {stat.upper()}={sum_b[stat]} "
                f"!= team {stat.upper()}={team_val}"
            )

    # Turnovers: team >= player sum; negative residual is a hard error.
    for tid, team, psum in ((id_a, team_a, sum_a), (id_b, team_b, sum_b)):
        residual = team.tov - psum["tov"]
        report.metrics[f"team_{tid}_turnover_residual"] = float(residual)
        if residual < 0:
            report.hard_errors.append(
                f"team {tid}: team TOV={team.tov} < summed player TOV={psum['tov']} "
                f"(residual {residual})"
            )
        elif residual > TEAM_TURNOVER_RESIDUAL_WARN:
            report.warnings.append(
                f"team {tid}: team turnover residual {residual} exceeds "
                f"{TEAM_TURNOVER_RESIDUAL_WARN}"
            )

    # Plus/minus identity (tolerance 1e-6).
    for tid, team, psum in ((id_a, team_a, sum_a), (id_b, team_b, sum_b)):
        expected = 5.0 * team.team_plus_minus
        if abs(psum["plus_minus"] - expected) > PLUS_MINUS_TOLERANCE:
            report.hard_errors.append(
                f"team {tid}: summed player PLUS_MINUS={psum['plus_minus']} "
                f"!= 5*team PLUS_MINUS={expected}"
            )

    # team PLUS_MINUS = pts - opp pts, and antisymmetry.
    if abs(team_a.team_plus_minus - (team_a.pts - team_b.pts)) > PLUS_MINUS_TOLERANCE:
        report.hard_errors.append(
            f"team {id_a}: PLUS_MINUS={team_a.team_plus_minus} "
            f"!= PTS-opp_PTS={team_a.pts - team_b.pts}"
        )
    if abs(team_b.team_plus_minus - (team_b.pts - team_a.pts)) > PLUS_MINUS_TOLERANCE:
        report.hard_errors.append(
            f"team {id_b}: PLUS_MINUS={team_b.team_plus_minus} "
            f"!= PTS-opp_PTS={team_b.pts - team_a.pts}"
        )
    if abs(team_a.team_plus_minus + team_b.team_plus_minus) > PLUS_MINUS_TOLERANCE:
        report.hard_errors.append(
            f"team plus/minus not antisymmetric: {team_a.team_plus_minus} "
            f"+ {team_b.team_plus_minus} != 0"
        )

    # Steals <= opponent turnovers.
    if team_a.stl > team_b.tov:
        report.hard_errors.append(
            f"team {id_a}: STL={team_a.stl} > opponent TOV={team_b.tov}"
        )
    if team_b.stl > team_a.tov:
        report.hard_errors.append(
            f"team {id_b}: STL={team_b.stl} > opponent TOV={team_a.tov}"
        )

    # Blocks <= opponent missed field goals.
    a_missed = team_a.fga - team_a.fgm
    b_missed = team_b.fga - team_b.fgm
    if team_a.blk > b_missed:
        report.hard_errors.append(
            f"team {id_a}: BLK={team_a.blk} > opponent missed FGA={b_missed}"
        )
    if team_b.blk > a_missed:
        report.hard_errors.append(
            f"team {id_b}: BLK={team_b.blk} > opponent missed FGA={a_missed}"
        )

    # Assists <= own made field goals.
    if team_a.ast > team_a.fgm:
        report.hard_errors.append(
            f"team {id_a}: AST={team_a.ast} > FGM={team_a.fgm}"
        )
    if team_b.ast > team_b.fgm:
        report.hard_errors.append(
            f"team {id_b}: AST={team_b.ast} > FGM={team_b.fgm}"
        )

    # BLKA should equal opponent blocks (warning, not error).
    if team_a.blka != team_b.blk:
        report.warnings.append(
            f"team {id_a}: BLKA={team_a.blka} != opponent BLK={team_b.blk}"
        )
    if team_b.blka != team_a.blk:
        report.warnings.append(
            f"team {id_b}: BLKA={team_b.blka} != opponent BLK={team_a.blk}"
        )


# ---------------------------------------------------------------------------
# Overtime inference (Section 9.5)
# ---------------------------------------------------------------------------
def infer_overtime(
    sum_minutes_a: float,
    sum_minutes_b: float,
    report: GameValidationReport,
) -> int:
    """Infer the shared overtime period count from both teams' player minutes.

    Returns the agreed OT count and records the inference error metric. A
    negative total, OT disagreement between teams, or count above
    ``MAX_OT_PERIODS`` is a hard error.
    """
    raw_ot_a = (sum_minutes_a - _REGULATION_TEAM_MINUTES) / _OT_TEAM_MINUTES
    raw_ot_b = (sum_minutes_b - _REGULATION_TEAM_MINUTES) / _OT_TEAM_MINUTES
    ot_a = max(0, round(raw_ot_a))
    ot_b = max(0, round(raw_ot_b))

    expected_a = _REGULATION_TEAM_MINUTES + _OT_TEAM_MINUTES * ot_a
    expected_b = _REGULATION_TEAM_MINUTES + _OT_TEAM_MINUTES * ot_b
    err_a = abs(sum_minutes_a - expected_a)
    err_b = abs(sum_minutes_b - expected_b)
    report.metrics["team_a_minute_sum"] = float(sum_minutes_a)
    report.metrics["team_b_minute_sum"] = float(sum_minutes_b)
    report.metrics["team_a_minute_error"] = float(err_a)
    report.metrics["team_b_minute_error"] = float(err_b)

    if sum_minutes_a < 0 or sum_minutes_b < 0:
        report.hard_errors.append(
            f"negative team minute sum: a={sum_minutes_a} b={sum_minutes_b}"
        )
        return 0
    if err_a > _MINUTE_TOLERANCE:
        report.hard_errors.append(
            f"team A minute sum {sum_minutes_a} not within {_MINUTE_TOLERANCE} of "
            f"expected {expected_a} for {ot_a} OT"
        )
    if err_b > _MINUTE_TOLERANCE:
        report.hard_errors.append(
            f"team B minute sum {sum_minutes_b} not within {_MINUTE_TOLERANCE} of "
            f"expected {expected_b} for {ot_b} OT"
        )
    if ot_a != ot_b:
        report.hard_errors.append(
            f"overtime inference disagrees: team A -> {ot_a}, team B -> {ot_b}"
        )
        return 0
    if ot_a > MAX_OT_PERIODS:
        report.hard_errors.append(
            f"inferred {ot_a} overtime periods exceed max {MAX_OT_PERIODS}"
        )
        return MAX_OT_PERIODS
    return ot_a


# ---------------------------------------------------------------------------
# Top-level entry point (Section 9.1)
# ---------------------------------------------------------------------------
def validate_game_box_score(game: GameBoxScore) -> GameValidationReport:
    """Run every hard-constraint layer and return a structured report.

    A game is ``ok`` only when no hard error was raised across per-row
    arithmetic, player/team sum conservation, cross-team invariants, and
    overtime inference. Warnings never affect ``ok``.
    """
    report = GameValidationReport(game_id=game.game_id)

    for team_id, players in game.players_by_team.items():
        for player in players:
            validate_player_line(player, report)
    for team in game.teams.values():
        validate_team_line(team, report)

    validate_cross_team(game, report)

    if len(game.team_ids) == 2:
        id_a, id_b = game.team_ids
        sum_a = sum(p.minutes for p in game.players_by_team.get(id_a, ()))
        sum_b = sum(p.minutes for p in game.players_by_team.get(id_b, ()))
        # ``infer_overtime`` mutates ``report`` with metrics/errors but the
        # already-stored ``game.overtime_periods`` is the value canonicalization
        # chose; we re-infer here purely to validate consistency.
        infer_overtime(sum_a, sum_b, report)

    return report
