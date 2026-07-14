"""Exact event ledger: turns box-score counts into consumable tokens.

Implements Section 10 of the reconstruction plan. The ledger is the main
safeguard against a plausible-looking possession sequence that does not
reproduce the input box score: every observed count becomes a token that must
be consumed exactly once by a valid reconstruction.

Design
------
* ``PlayerEventInventory`` holds the 15 token pools for one player (made/missed
  field goals split by shot value, made/missed free throws, rebounds, assists,
  turnovers, steals, blocks, blocked attempts, fouls, fouls drawn).
* ``TeamResidualInventory`` holds team-only turnovers (the gap between team TOV
  and summed player TOV that has no player assignment).
* ``EventLedger`` is the accounting book: ``from_game`` builds it from a
  ``GameBoxScore``; ``consume``/``release_for_repair``/``remaining``/
  ``assert_fully_consumed`` enforce exact conservation.
* Every token has a deterministic ID
  ``{game_id}:{team_id}:{player_or_TEAM}:{stat}:{index}`` so results are
  reproducible regardless of worker count.
* ``aggregate_sample`` counts the events in a reconstruction sample back to
  player/team totals, which must match the source box score exactly.

The ledger deliberately does **not** order events in time and does **not**
include minutes or plus/minus (those come from rotations/scoring and are
checked separately with tolerances).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.reconstruction.schema import (
    EventType,
    GameBoxScore,
    LatentEvent,
    PlayerBoxLine,
    ReconstructionSample,
    TeamBoxLine,
)

__all__ = [
    "TOKEN_STAT_NAMES",
    "PLAYER_TOKEN_STATS",
    "TEAM_TOKEN_STATS",
    "LedgerToken",
    "PlayerEventInventory",
    "TeamResidualInventory",
    "EventLedger",
    "LedgerError",
    "AggregatedPlayerCounts",
    "AggregatedTeamCounts",
    "AggregatedBoxScore",
    "aggregate_sample",
    "events_to_player_counts",
]


class LedgerError(ValueError):
    """Raised on double consumption, unknown token, or incomplete consumption."""


# ---------------------------------------------------------------------------
# Token stat names and box-score column mapping (Section 10.2)
# ---------------------------------------------------------------------------
# The 15 per-player token pools. Each maps to a derived box-score count.
# Note: fg2_missed and blocked_attempt overlap semantically (a blocked shot is
# a kind of miss) but are independent conservation pools because BLKA is a
# distinct box-score column tracked on the shooter.
TOKEN_STAT_NAMES: Tuple[str, ...] = (
    "fg2_made",
    "fg2_missed",
    "fg3_made",
    "fg3_missed",
    "free_throw_made",
    "free_throw_missed",
    "offensive_rebound",
    "defensive_rebound",
    "assist",
    "turnover",
    "steal",
    "block",
    "blocked_attempt",
    "personal_foul",
    "foul_drawn",
)

# Team-level token pool (only the turnover residual).
TEAM_TOKEN_STATS: Tuple[str, ...] = ("team_turnover",)

# Mapping from a token stat name to a callable that computes its count from
# a ``PlayerBoxLine``. Several pools are derived (missed = attempts - made)
# rather than stored directly on the dataclass.
def _player_token_counts(player: PlayerBoxLine) -> Dict[str, int]:
    return {
        "fg2_made": player.fg2m,
        "fg2_missed": player.fg2_missed,
        "fg3_made": player.fg3m,
        "fg3_missed": player.fg3_missed,
        "free_throw_made": player.ftm,
        "free_throw_missed": player.fta - player.ftm,
        "offensive_rebound": player.oreb,
        "defensive_rebound": player.dreb,
        "assist": player.ast,
        "turnover": player.tov,
        "steal": player.stl,
        "block": player.blk,
        "blocked_attempt": player.blka,
        "personal_foul": player.pf,
        "foul_drawn": player.pfd,
    }

# Convenience alias for the full set used by callers.
PLAYER_TOKEN_STATS: Tuple[str, ...] = TOKEN_STAT_NAMES


# ---------------------------------------------------------------------------
# LedgerToken
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerToken:
    """One unit of an observed stat that a reconstruction must consume.

    ``player_id`` is ``None`` only for team-residual tokens (unassigned team
    turnovers). The ``token_id`` is deterministic so consumption state is
    reproducible regardless of iteration order.
    """

    token_id: str
    game_id: str
    team_id: int
    stat: str
    player_id: Optional[int]
    index: int

    @classmethod
    def make(
        cls,
        game_id: str,
        team_id: int,
        stat: str,
        player_id: Optional[int],
        index: int,
    ) -> "LedgerToken":
        """Build a token with the canonical deterministic ID."""
        actor = str(player_id) if player_id is not None else "TEAM"
        token_id = f"{game_id}:{team_id}:{actor}:{stat}:{index}"
        return cls(
            token_id=token_id,
            game_id=game_id,
            team_id=team_id,
            stat=stat,
            player_id=player_id,
            index=index,
        )


# ---------------------------------------------------------------------------
# Inventories
# ---------------------------------------------------------------------------
@dataclass
class PlayerEventInventory:
    """Token pools for one player, keyed by stat name."""

    game_id: str
    team_id: int
    player_id: int
    # remaining[stat] -> count still available to consume
    remaining: Dict[str, int] = field(default_factory=dict)
    # total[stat] -> original count (for reset / reporting)
    total: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_player(cls, player: PlayerBoxLine) -> "PlayerEventInventory":
        inv = cls(
            game_id=player.game_id,
            team_id=player.team_id,
            player_id=player.player_id,
        )
        for stat, count in _player_token_counts(player).items():
            inv.total[stat] = int(count)
            inv.remaining[stat] = int(count)
        return inv

    def total_tokens(self) -> int:
        return sum(self.total.values())


@dataclass
class TeamResidualInventory:
    """Unassigned team turnovers for one team-game."""

    game_id: str
    team_id: int
    remaining: int
    total: int

    @classmethod
    def from_team(
        cls, team: TeamBoxLine, player_tov_sum: int
    ) -> "TeamResidualInventory":
        residual = int(team.tov) - int(player_tov_sum)
        return cls(
            game_id=team.game_id,
            team_id=team.team_id,
            remaining=residual,
            total=residual,
        )


# ---------------------------------------------------------------------------
# EventLedger
# ---------------------------------------------------------------------------
class EventLedger:
    """Accounting book for all observed-stat tokens in one game.

    Built from a ``GameBoxScore`` via ``from_game``. The ledger tracks which
    tokens have been consumed so a reconstruction candidate can be validated:
    every token must be consumed exactly once, with no double-consumption or
    invented counts.
    """

    def __init__(
        self,
        game_id: str,
        player_inventories: Dict[int, PlayerEventInventory],
        team_residuals: Dict[int, TeamResidualInventory],
    ) -> None:
        self.game_id = game_id
        self._player_inventories = player_inventories
        self._team_residuals = team_residuals
        # Track consumed token IDs for double-consume detection.
        self._consumed: set = set()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_game(cls, game: GameBoxScore) -> "EventLedger":
        """Build the full ledger from a validated ``GameBoxScore``."""
        player_invs: Dict[int, PlayerEventInventory] = {}
        for team_id, players in game.players_by_team.items():
            for player in players:
                player_invs[player.player_id] = PlayerEventInventory.from_player(player)

        team_resids: Dict[int, TeamResidualInventory] = {}
        for team_id, team in game.teams.items():
            player_tov_sum = sum(
                p.tov for p in game.players_by_team.get(team_id, ())
            )
            team_resids[team_id] = TeamResidualInventory.from_team(
                team, player_tov_sum
            )

        return cls(
            game_id=game.game_id,
            player_inventories=player_invs,
            team_residuals=team_resids,
        )

    # -- consume / release -----------------------------------------------
    def consume(self, token: LedgerToken) -> None:
        """Mark one token as consumed.

        Raises ``LedgerError`` if the token is unknown, already consumed, or
        its pool is exhausted (over-consumption).
        """
        if token.token_id in self._consumed:
            raise LedgerError(f"token already consumed: {token.token_id}")
        if token.player_id is not None:
            inv = self._player_inventories.get(token.player_id)
            if inv is None:
                raise LedgerError(
                    f"unknown player {token.player_id} for token {token.token_id}"
                )
            if token.stat not in inv.remaining:
                raise LedgerError(
                    f"unknown stat '{token.stat}' for token {token.token_id}"
                )
            if inv.remaining[token.stat] <= 0:
                raise LedgerError(
                    f"over-consumption of {token.stat} for player "
                    f"{token.player_id} ({token.token_id})"
                )
            inv.remaining[token.stat] -= 1
        else:
            resid = self._team_residuals.get(token.team_id)
            if resid is None:
                raise LedgerError(
                    f"unknown team {token.team_id} for token {token.token_id}"
                )
            if token.stat != "team_turnover":
                raise LedgerError(
                    f"team token has wrong stat '{token.stat}' "
                    f"({token.token_id})"
                )
            if resid.remaining <= 0:
                raise LedgerError(
                    f"over-consumption of team_turnover for team "
                    f"{token.team_id} ({token.token_id})"
                )
            resid.remaining -= 1
        self._consumed.add(token.token_id)

    def release_for_repair(self, token: LedgerToken) -> None:
        """Release a previously-consumed token so a repair can reassign it.

        Raises ``LedgerError`` if the token was not consumed.
        """
        if token.token_id not in self._consumed:
            raise LedgerError(
                f"cannot release uncomsumed token: {token.token_id}"
            )
        if token.player_id is not None:
            inv = self._player_inventories.get(token.player_id)
            if inv is not None and token.stat in inv.remaining:
                inv.remaining[token.stat] += 1
        else:
            resid = self._team_residuals.get(token.team_id)
            if resid is not None:
                resid.remaining += 1
        self._consumed.discard(token.token_id)

    # -- inspection ------------------------------------------------------
    def remaining(self) -> Dict[str, int]:
        """Flat ``{token_id_fragment: count}`` of all unconsumed tokens.

        Keys are ``"{team_id}:{player_or_TEAM}:{stat}"`` and values are the
        remaining count. Zero-count entries are excluded.
        """
        result: Dict[str, int] = {}
        for pid, inv in self._player_inventories.items():
            for stat, count in inv.remaining.items():
                if count > 0:
                    result[f"{inv.team_id}:{pid}:{stat}"] = count
        for tid, resid in self._team_residuals.items():
            if resid.remaining > 0:
                result[f"{tid}:TEAM:team_turnover"] = resid.remaining
        return result

    def total_remaining(self) -> int:
        """Sum of all unconsumed token counts across the entire ledger."""
        return sum(self.remaining().values())

    def total_tokens(self) -> int:
        """Total token count (consumed + remaining) across the entire ledger."""
        total = sum(inv.total_tokens() for inv in self._player_inventories.values())
        total += sum(r.total for r in self._team_residuals.values())
        return total

    def assert_fully_consumed(self) -> None:
        """Raise ``LedgerError`` if any token remains unconsumed.

        The error message lists every pool with a positive remaining count.
        """
        leftovers = self.remaining()
        if leftovers:
            details = ", ".join(f"{k}={v}" for k, v in sorted(leftovers.items()))
            raise LedgerError(
                f"ledger not fully consumed ({len(leftovers)} pools): {details}"
            )

    # -- clone -----------------------------------------------------------
    def clone(self) -> "EventLedger":
        """Return a deep copy so a candidate can consume without mutation.

        Inventories are deep-copied (dict-level) so the clone is fully
        independent. Token IDs are immutable (frozen dataclasses) and shared
        by reference safely.
        """
        return EventLedger(
            game_id=self.game_id,
            player_inventories=copy.deepcopy(self._player_inventories),
            team_residuals=copy.deepcopy(self._team_residuals),
        )

    # -- token generation (for samplers) ---------------------------------
    def tokens_for(self, team_id: int, player_id: int, stat: str) -> List[LedgerToken]:
        """Return the ``LedgerToken`` objects for one player-stat pool.

        Useful for samplers that iterate tokens in deterministic order.
        """
        inv = self._player_inventories.get(player_id)
        if inv is None:
            raise LedgerError(f"unknown player {player_id}")
        if stat not in inv.total:
            raise LedgerError(f"unknown stat '{stat}' for player {player_id}")
        count = inv.total[stat]
        return [
            LedgerToken.make(self.game_id, team_id, stat, player_id, i)
            for i in range(count)
        ]

    def team_turnover_tokens(self, team_id: int) -> List[LedgerToken]:
        """Return the ``LedgerToken`` objects for a team's residual turnovers."""
        resid = self._team_residuals.get(team_id)
        if resid is None:
            raise LedgerError(f"unknown team {team_id}")
        return [
            LedgerToken.make(self.game_id, team_id, "team_turnover", None, i)
            for i in range(resid.total)
        ]


# ---------------------------------------------------------------------------
# Event aggregation (Section 10.4)
# ---------------------------------------------------------------------------
@dataclass
class AggregatedPlayerCounts:
    """Reconstructed box-score counts for one player from events."""

    player_id: int
    team_id: int
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    reb: int = 0
    ast: int = 0
    tov: int = 0
    stl: int = 0
    blk: int = 0
    blka: int = 0
    pf: int = 0
    pfd: int = 0
    pts: int = 0

    @property
    def fg2m(self) -> int:
        return self.fgm - self.fg3m

    @property
    def fg2a(self) -> int:
        return self.fga - self.fg3a


@dataclass
class AggregatedTeamCounts:
    """Reconstructed box-score counts for one team from events."""

    team_id: int
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    reb: int = 0
    ast: int = 0
    tov: int = 0
    stl: int = 0
    blk: int = 0
    blka: int = 0
    pf: int = 0
    pfd: int = 0
    pts: int = 0
    team_turnover_residual: int = 0


@dataclass
class AggregatedBoxScore:
    """Round-trip result: event-aggregated counts that must match the source."""

    game_id: str
    players: Dict[int, AggregatedPlayerCounts] = field(default_factory=dict)
    teams: Dict[int, AggregatedTeamCounts] = field(default_factory=dict)


def _empty_player_counts(player_id: int, team_id: int) -> AggregatedPlayerCounts:
    return AggregatedPlayerCounts(player_id=player_id, team_id=team_id)


def _empty_team_counts(team_id: int) -> AggregatedTeamCounts:
    return AggregatedTeamCounts(team_id=team_id)


def events_to_player_counts(
    events: Tuple[LatentEvent, ...],
) -> Dict[int, AggregatedPlayerCounts]:
    """Aggregate a flat event list into per-player counts.

    Walks every event and increments the actor's counts based on event type
    and properties. Team events (``is_team_event=True``) are skipped for player
    counts but contribute to team totals via ``aggregate_sample``.

    Mapping (Section 10.4 / 13):
    * ``FG2_MADE``  -> FGM, pts += 2
    * ``FG3_MADE``  -> FGM, FG3M, FG3A, pts += 3
    * ``FG2_MISSED`` -> FGA
    * ``FG3_MISSED`` -> FGA, FG3A
    * ``FREE_THROW`` (made)  -> FTM, FTA, pts += 1
    * ``FREE_THROW`` (missed) -> FTA
    * ``OFFENSIVE_REBOUND`` -> OREB, REB
    * ``DEFENSIVE_REBOUND`` -> DREB, REB
    * ``ASSIST`` -> AST
    * ``TURNOVER`` -> TOV
    * ``STEAL`` -> STL
    * ``BLOCK`` -> BLK
    * ``PERSONAL_FOUL`` -> PF
    * ``FOUL_DRAWN`` -> PFD

    ``BLOCKED`` attempts (BLKA) are counted from missed field-goal events whose
    ``secondary_player_id`` is set (the blocker), which signals the shooter's
    attempt was blocked.
    """
    counts: Dict[int, AggregatedPlayerCounts] = {}

    for evt in events:
        if evt.is_team_event or evt.actor_player_id is None:
            continue
        pid = evt.actor_player_id
        if pid not in counts:
            counts[pid] = _empty_player_counts(pid, evt.offense_team_id)
        pc = counts[pid]

        if evt.event_type == EventType.FG2_MADE:
            pc.fgm += 1
            pc.fga += 1
            pc.pts += 2
        elif evt.event_type == EventType.FG3_MADE:
            pc.fgm += 1
            pc.fg3m += 1
            pc.fga += 1
            pc.fg3a += 1
            pc.pts += 3
        elif evt.event_type == EventType.FG2_MISSED:
            pc.fga += 1
            if evt.secondary_player_id is not None:
                pc.blka += 1
        elif evt.event_type == EventType.FG3_MISSED:
            pc.fga += 1
            pc.fg3a += 1
            if evt.secondary_player_id is not None:
                pc.blka += 1
        elif evt.event_type == EventType.FREE_THROW:
            pc.fta += 1
            if evt.made:
                pc.ftm += 1
                pc.pts += 1
        elif evt.event_type == EventType.OFFENSIVE_REBOUND:
            pc.oreb += 1
            pc.reb += 1
        elif evt.event_type == EventType.DEFENSIVE_REBOUND:
            pc.dreb += 1
            pc.reb += 1
        elif evt.event_type == EventType.ASSIST:
            pc.ast += 1
        elif evt.event_type == EventType.TURNOVER:
            pc.tov += 1
        elif evt.event_type == EventType.STEAL:
            pc.stl += 1
        elif evt.event_type == EventType.BLOCK:
            pc.blk += 1
        elif evt.event_type == EventType.PERSONAL_FOUL:
            pc.pf += 1
        elif evt.event_type == EventType.FOUL_DRAWN:
            pc.pfd += 1
        # PERIOD_END / UNKNOWN_BOUNDARY carry no observed stat.

    return counts


def _aggregate_team_from_players(
    team_id: int,
    player_counts: Dict[int, AggregatedPlayerCounts],
    team_events: List[LatentEvent],
) -> AggregatedTeamCounts:
    """Sum player counts into a team total and add team-only turnovers."""
    tc = _empty_team_counts(team_id)
    for pc in player_counts.values():
        if pc.team_id != team_id:
            continue
        tc.fgm += pc.fgm
        tc.fga += pc.fga
        tc.fg3m += pc.fg3m
        tc.fg3a += pc.fg3a
        tc.ftm += pc.ftm
        tc.fta += pc.fta
        tc.oreb += pc.oreb
        tc.dreb += pc.dreb
        tc.reb += pc.reb
        tc.ast += pc.ast
        tc.tov += pc.tov
        tc.stl += pc.stl
        tc.blk += pc.blk
        tc.blka += pc.blka
        tc.pf += pc.pf
        tc.pfd += pc.pfd
        tc.pts += pc.pts

    # Team-only turnover events (is_team_event=True, event_type=TURNOVER).
    for evt in team_events:
        if (
            evt.is_team_event
            and evt.event_type == EventType.TURNOVER
            and evt.offense_team_id == team_id
        ):
            tc.tov += 1
            tc.team_turnover_residual += 1

    return tc


def aggregate_sample(sample: ReconstructionSample) -> AggregatedBoxScore:
    """Aggregate a reconstruction sample's events back to box-score counts.

    Walks every event in every possession, bins by actor and team, and returns
    ``AggregatedBoxScore`` with per-player and per-team totals. The result must
    reproduce every hard player and team count from the source box score
    exactly; minutes and plus/minus are excluded (they come from
    rotations/scoring and are checked separately with tolerances).
    """
    all_events: List[LatentEvent] = []
    for poss in sample.possessions:
        all_events.extend(poss.events)

    player_counts = events_to_player_counts(tuple(all_events))

    # Group players by team for team aggregation.
    teams_in_game: set = set()
    for pc in player_counts.values():
        teams_in_game.add(pc.team_id)
    # Also pick up teams from possession offense_team_id (for games where
    # one team has zero observed player events — unlikely but defensive).
    for poss in sample.possessions:
        teams_in_game.add(poss.offense_team_id)

    team_counts: Dict[int, AggregatedTeamCounts] = {}
    for team_id in sorted(teams_in_game):
        team_counts[team_id] = _aggregate_team_from_players(
            team_id, player_counts, all_events
        )

    return AggregatedBoxScore(
        game_id=sample.game_id,
        players=player_counts,
        teams=team_counts,
    )
