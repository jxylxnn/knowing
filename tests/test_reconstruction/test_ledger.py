"""Tests for the exact event ledger (PR 03).

Covers the PR 03 exit gate:
* Inventory counts equal source box counts.
* Team-only TOV tokens equal residual.
* Double consume raises.
* Unknown token raises.
* Fully consuming a synthetic ledger round-trips exactly.
* One missing token prevents candidate validity (assert_fully_consumed fails).

Also covers clone independence and aggregate_sample round-trip.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.reconstruction import (
    EventLedger,
    LedgerError,
    LedgerToken,
    canonicalize_game,
    aggregate_sample,
)
from src.reconstruction.ledger import (
    AggregatedBoxScore,
    PlayerEventInventory,
    TeamResidualInventory,
    TOKEN_STAT_NAMES,
    events_to_player_counts,
)
from src.reconstruction.schema import (
    EventType,
    GameBoxScore,
    LatentEvent,
    LatentPossession,
    PossessionEnd,
    PlayerBoxLine,
    ReconstructionSample,
    TeamBoxLine,
)
from datetime import date

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "reconstruction"


# ---------------------------------------------------------------------------
# Schema builders for synthetic games
# ---------------------------------------------------------------------------
def _player(pid=11, **kw) -> PlayerBoxLine:
    base = dict(
        game_id="g1", team_id=1001, player_id=pid, player_name="P",
        minutes=30.0, fgm=5, fga=10, fg3m=2, fg3a=5, ftm=3, fta=4,
        oreb=1, dreb=4, reb=5, ast=6, tov=2, stl=1, blk=0, blka=1,
        pf=3, pfd=2, pts=15, plus_minus=4.0,
    )
    base.update(kw)
    return PlayerBoxLine(**base)


def _team(team_id=1001, **kw) -> TeamBoxLine:
    base = dict(
        game_id="g1", team_id=team_id, team_abbreviation="AAA",
        game_date=date(2025, 1, 1), matchup="AAA vs. BBB",
        minutes=48.0, fgm=10, fga=20, fg3m=4, fg3a=10, ftm=6, fta=8,
        oreb=2, dreb=8, reb=10, ast=12, tov=5, stl=2, blk=1, blka=2,
        pf=6, pfd=4, pts=30, team_plus_minus=5.0,
    )
    base.update(kw)
    return TeamBoxLine(**base)


def _game_with_residual() -> GameBoxScore:
    """A small synthetic game where team 1001 has a turnover residual of 3.

    Two teams, one player each for simplicity. Player TOV sums are set below
    team TOV so the residual is non-zero.
    """
    pa = (_player(pid=11, tov=2),)
    pb = (_player(pid=21, team_id=1002, tov=1),)
    ta = _team(team_id=1001, tov=5)   # residual = 5 - 2 = 3
    tb = _team(team_id=1002, team_abbreviation="BBB", matchup="BBB @ AAA",
               tov=1)  # residual = 1 - 1 = 0
    return GameBoxScore(
        game_id="g1", game_date=date(2025, 1, 1), season_year="2024-25",
        team_ids=(1001, 1002),
        teams={1001: ta, 1002: tb},
        players_by_team={1001: pa, 1002: pb},
        overtime_periods=0,
        source_player_hash="p", source_team_hash="t",
    )


def _load_regulation_game() -> GameBoxScore:
    p = pd.read_csv(FIXTURES / "regulation_game_players.csv", dtype={"GAME_ID": str})
    t = pd.read_csv(FIXTURES / "regulation_game_teams.csv", dtype={"GAME_ID": str})
    return canonicalize_game(p["GAME_ID"].iloc[0], p, t)


# ---------------------------------------------------------------------------
# Token ID format
# ---------------------------------------------------------------------------
class TestTokenIds:
    def test_player_token_id_format(self):
        tok = LedgerToken.make("g1", 1001, "fg2_made", 11, 3)
        assert tok.token_id == "g1:1001:11:fg2_made:3"
        assert tok.player_id == 11
        assert tok.stat == "fg2_made"
        assert tok.index == 3

    def test_team_token_id_uses_TEAM_actor(self):
        tok = LedgerToken.make("g1", 1001, "team_turnover", None, 0)
        assert tok.token_id == "g1:1001:TEAM:team_turnover:0"
        assert tok.player_id is None

    def test_token_is_frozen(self):
        tok = LedgerToken.make("g1", 1001, "fg2_made", 11, 0)
        with pytest.raises(Exception):
            tok.stat = "assist"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Inventory counts equal source box counts
# ---------------------------------------------------------------------------
class TestInventoryCounts:
    def test_player_inventory_matches_box_score(self):
        p = _player(fgm=7, fga=15, fg3m=2, fg3a=5, ftm=4, fta=6,
                    oreb=3, dreb=5, ast=8, tov=3, stl=2, blk=1, blka=2,
                    pf=4, pfd=3)
        inv = PlayerEventInventory.from_player(p)
        # FG2: made = 7-2=5, missed = (15-5) - 5 = 5
        assert inv.total["fg2_made"] == 5
        assert inv.total["fg2_missed"] == 5
        # FG3: made=2, missed=5-2=3
        assert inv.total["fg3_made"] == 2
        assert inv.total["fg3_missed"] == 3
        # FT: made=4, missed=6-4=2
        assert inv.total["free_throw_made"] == 4
        assert inv.total["free_throw_missed"] == 2
        assert inv.total["offensive_rebound"] == 3
        assert inv.total["defensive_rebound"] == 5
        assert inv.total["assist"] == 8
        assert inv.total["turnover"] == 3
        assert inv.total["steal"] == 2
        assert inv.total["block"] == 1
        assert inv.total["blocked_attempt"] == 2
        assert inv.total["personal_foul"] == 4
        assert inv.total["foul_drawn"] == 3

    def test_player_inventory_zero_pools(self):
        p = _player(fgm=0, fga=0, fg3m=0, fg3a=0, ftm=0, fta=0, oreb=0, dreb=0,
                    reb=0, ast=0, tov=0, stl=0, blk=0, blka=0, pf=0, pfd=0,
                    pts=0)
        inv = PlayerEventInventory.from_player(p)
        assert all(v == 0 for v in inv.total.values())
        assert inv.total_tokens() == 0

    def test_remaining_equals_total_initially(self):
        inv = PlayerEventInventory.from_player(_player())
        for stat in TOKEN_STAT_NAMES:
            assert inv.remaining[stat] == inv.total[stat]


# ---------------------------------------------------------------------------
# Team residual tokens
# ---------------------------------------------------------------------------
class TestTeamResidual:
    def test_residual_equals_team_minus_player_tov(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        # Team 1001: team TOV 5, player TOV 2 -> residual 3
        assert ledger._team_residuals[1001].total == 3
        # Team 1002: team TOV 1, player TOV 1 -> residual 0
        assert ledger._team_residuals[1002].total == 0

    def test_team_turnover_token_ids(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        toks = ledger.team_turnover_tokens(1001)
        assert len(toks) == 3
        assert all(t.player_id is None for t in toks)
        assert all(t.token_id.startswith("g1:1001:TEAM:team_turnover:") for t in toks)
        assert [t.index for t in toks] == [0, 1, 2]

    def test_zero_residual_team_has_no_tokens(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        assert ledger.team_turnover_tokens(1002) == []


# ---------------------------------------------------------------------------
# Consume / double-consume / unknown token
# ---------------------------------------------------------------------------
class TestConsumeRejection:
    def test_consume_decrements_remaining(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = ledger.tokens_for(1001, 11, "fg2_made")[0]
        before = ledger._player_inventories[11].remaining["fg2_made"]
        ledger.consume(tok)
        after = ledger._player_inventories[11].remaining["fg2_made"]
        assert after == before - 1

    def test_double_consume_raises(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = ledger.tokens_for(1001, 11, "fg2_made")[0]
        ledger.consume(tok)
        with pytest.raises(LedgerError, match="already consumed"):
            ledger.consume(tok)

    def test_over_consumption_raises(self):
        """Consuming more tokens than exist in a pool raises."""
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        # Player 11 has 2 fg2_made tokens (fgm=5, fg3m=2 -> fg2m=3).
        toks = ledger.tokens_for(1001, 11, "fg2_made")
        for t in toks:
            ledger.consume(t)
        # Now pool is empty; consuming one more should raise.
        extra = LedgerToken.make("g1", 1001, "fg2_made", 11, len(toks))
        with pytest.raises(LedgerError, match="over-consumption"):
            ledger.consume(extra)

    def test_unknown_player_raises(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = LedgerToken.make("g1", 1001, "fg2_made", 9999, 0)
        with pytest.raises(LedgerError, match="unknown player"):
            ledger.consume(tok)

    def test_unknown_stat_raises(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = LedgerToken.make("g1", 1001, "bogus_stat", 11, 0)
        with pytest.raises(LedgerError, match="unknown stat"):
            ledger.consume(tok)

    def test_unknown_team_raises(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = LedgerToken.make("g1", 9999, "team_turnover", None, 0)
        with pytest.raises(LedgerError, match="unknown team"):
            ledger.consume(tok)


# ---------------------------------------------------------------------------
# Release for repair
# ---------------------------------------------------------------------------
class TestReleaseForRepair:
    def test_release_restores_count(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = ledger.tokens_for(1001, 11, "fg2_made")[0]
        before = ledger._player_inventories[11].remaining["fg2_made"]
        ledger.consume(tok)
        ledger.release_for_repair(tok)
        after = ledger._player_inventories[11].remaining["fg2_made"]
        assert after == before

    def test_release_unconsumed_raises(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = ledger.tokens_for(1001, 11, "fg2_made")[0]
        with pytest.raises(LedgerError, match="cannot release"):
            ledger.release_for_repair(tok)

    def test_release_allows_reconsume(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        tok = ledger.tokens_for(1001, 11, "fg2_made")[0]
        ledger.consume(tok)
        ledger.release_for_repair(tok)
        # Should be consumable again without error.
        ledger.consume(tok)


# ---------------------------------------------------------------------------
# assert_fully_consumed
# ---------------------------------------------------------------------------
class TestAssertFullyConsumed:
    def test_empty_ledger_passes(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        # Consume every token.
        _consume_all(ledger)
        ledger.assert_fully_consumed()  # no raise

    def test_unconsumed_token_fails(self):
        """One missing token prevents candidate validity."""
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        _consume_all(ledger)
        # Release one token so it's unconsumed.
        first_tok = ledger.tokens_for(1001, 11, "fg2_made")[0]
        ledger.release_for_repair(first_tok)
        with pytest.raises(LedgerError, match="not fully consumed"):
            ledger.assert_fully_consumed()

    def test_remaining_lists_unconsumed_pools(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        # Consume everything except player 11's assists.
        _consume_all_except(ledger, skip_player_stat={(1001, 11, "assist")})
        rem = ledger.remaining()
        assert "1001:11:assist" in rem
        assert rem["1001:11:assist"] == 6  # player 11 has ast=6


# ---------------------------------------------------------------------------
# Clone independence
# ---------------------------------------------------------------------------
class TestClone:
    def test_clone_is_independent(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        clone = ledger.clone()
        # Consume from clone, original unchanged.
        tok = clone.tokens_for(1001, 11, "fg2_made")[0]
        clone.consume(tok)
        assert clone.total_remaining() == ledger.total_remaining() - 1

    def test_clone_total_tokens_unchanged(self):
        game = _game_with_residual()
        ledger = EventLedger.from_game(game)
        clone = ledger.clone()
        assert clone.total_tokens() == ledger.total_tokens()


# ---------------------------------------------------------------------------
# Round-trip: aggregate_sample
# ---------------------------------------------------------------------------
def _make_event(event_id, et, offense, defense, actor, source_stat,
                made=None, secondary=None, is_team=False):
    pts = 0
    if et == EventType.FG2_MADE:
        pts = 2
    elif et == EventType.FG3_MADE:
        pts = 3
    elif et == EventType.FREE_THROW and made:
        pts = 1
    return LatentEvent(
        event_id=event_id, event_type=et, offense_team_id=offense,
        defense_team_id=defense, actor_player_id=actor,
        secondary_player_id=secondary, points=pts, made=made,
        shot_value=None, free_throw_number=None, free_throw_total=None,
        is_team_event=is_team, is_observed_stat=True,
        source_stat=source_stat, confidence=1.0,
    )


def _consume_all(ledger: EventLedger) -> None:
    """Consume every token in the ledger once."""
    game_id = ledger.game_id
    # Collect all tokens first (from total), then consume.
    all_player = {}
    for pid, inv in ledger._player_inventories.items():
        for stat in TOKEN_STAT_NAMES:
            toks = [LedgerToken.make(game_id, inv.team_id, stat, pid, i)
                    for i in range(inv.total[stat])]
            all_player[(inv.team_id, pid, stat)] = toks
    all_team = {}
    for tid, resid in ledger._team_residuals.items():
        all_team[tid] = [LedgerToken.make(game_id, tid, "team_turnover", None, i)
                         for i in range(resid.total)]
    for toks in all_player.values():
        for t in toks:
            ledger.consume(t)
    for toks in all_team.values():
        for t in toks:
            ledger.consume(t)


def _consume_all_except(ledger: EventLedger, skip_player_stat) -> None:
    """Consume every token except those in ``skip_player_stat``."""
    game_id = ledger.game_id
    for pid, inv in ledger._player_inventories.items():
        for stat in TOKEN_STAT_NAMES:
            count = inv.total[stat]
            for i in range(count):
                if (inv.team_id, pid, stat) in skip_player_stat:
                    continue
                ledger.consume(LedgerToken.make(game_id, inv.team_id, stat, pid, i))
    for tid, resid in ledger._team_residuals.items():
        for i in range(resid.total):
            ledger.consume(LedgerToken.make(game_id, tid, "team_turnover", None, i))


def _build_events_from_game(game: GameBoxScore):
    """Build one LatentEvent per ledger token, consuming the full ledger.

    Returns ``(events_list, ledger_after_consume)``. Each token maps to exactly
    one event with the matching event type. ``blocked_attempt`` tokens are
    paired with miss events via ``secondary_player_id``.
    """
    ledger = EventLedger.from_game(game)
    STAT_ET = {
        "fg2_made": EventType.FG2_MADE, "fg3_made": EventType.FG3_MADE,
        "fg2_missed": EventType.FG2_MISSED, "fg3_missed": EventType.FG3_MISSED,
        "free_throw_made": EventType.FREE_THROW,
        "free_throw_missed": EventType.FREE_THROW,
        "offensive_rebound": EventType.OFFENSIVE_REBOUND,
        "defensive_rebound": EventType.DEFENSIVE_REBOUND,
        "assist": EventType.ASSIST, "turnover": EventType.TURNOVER,
        "steal": EventType.STEAL, "block": EventType.BLOCK,
        "blocked_attempt": EventType.FG2_MISSED,
        "personal_foul": EventType.PERSONAL_FOUL,
        "foul_drawn": EventType.FOUL_DRAWN,
    }
    events = []
    eid = 0
    for tid in game.team_ids:
        opp = game.team_ids[0] if tid == game.team_ids[1] else game.team_ids[1]
        for pl in game.players_by_team[tid]:
            blka_remaining = ledger._player_inventories[pl.player_id].total["blocked_attempt"]
            for stat in TOKEN_STAT_NAMES:
                if stat == "blocked_attempt":
                    continue
                count = ledger._player_inventories[pl.player_id].total[stat]
                for i in range(count):
                    tok = LedgerToken.make(game.game_id, tid, stat, pl.player_id, i)
                    ledger.consume(tok)
                    sec = None
                    if stat in ("fg2_missed", "fg3_missed") and blka_remaining > 0:
                        btok = LedgerToken.make(
                            game.game_id, tid, "blocked_attempt", pl.player_id,
                            blka_remaining - 1,
                        )
                        ledger.consume(btok)
                        sec = 999
                        blka_remaining -= 1
                    made = True if stat == "free_throw_made" else (
                        False if stat == "free_throw_missed" else None
                    )
                    events.append(_make_event(
                        f"e{eid}", STAT_ET[stat], tid, opp, pl.player_id,
                        stat, made=made, secondary=sec,
                    ))
                    eid += 1
            # Consume any remaining blocked_attempt tokens (no miss to pair).
            for i in range(blka_remaining):
                btok = LedgerToken.make(
                    game.game_id, tid, "blocked_attempt", pl.player_id, i
                )
                ledger.consume(btok)
        for tok in ledger.team_turnover_tokens(tid):
            ledger.consume(tok)
            events.append(_make_event(
                f"e{eid}", EventType.TURNOVER, tid,
                game.team_ids[0] if tid == game.team_ids[1] else game.team_ids[1],
                None, "team_turnover", is_team=True,
            ))
            eid += 1
    return events, ledger


class TestAggregateSample:
    def test_fully_consumed_ledger_round_trips_player_counts(self):
        """Fully consuming a synthetic ledger round-trips exactly."""
        game = _game_with_residual()
        events, ledger = _build_events_from_game(game)
        ledger.assert_fully_consumed()

        poss = LatentPossession(
            possession_index=0, period=1, seconds_remaining_estimate=700.0,
            offense_team_id=1001, defense_team_id=1002,
            offense_lineup=(), defense_lineup=(),
            events=tuple(events), end_reason=PossessionEnd.UNKNOWN,
            points=0, duration_seconds=0.0, boundary_confidence=0.0,
        )
        sample = ReconstructionSample(
            game_id="g1", sample_id="s0", seed=42, possessions=(poss,),
            home_rotation=None, away_rotation=None, hard_errors=(),
            soft_penalties={}, log_prior=0.0, log_likelihood=0.0,
            log_weight=0.0, box_score_exact=True,
        )
        agg = aggregate_sample(sample)

        # Every player's hard count must reproduce the source exactly.
        for tid in game.team_ids:
            for pl in game.players_by_team[tid]:
                pc = agg.players[pl.player_id]
                for stat in ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                             "oreb", "dreb", "reb", "ast", "tov", "stl",
                             "blk", "pf", "pfd", "pts"):
                    assert getattr(pc, stat) == getattr(pl, stat), (
                        f"player {pl.player_id} {stat}: "
                        f"agg={getattr(pc, stat)} src={getattr(pl, stat)}"
                    )

    def test_team_turnover_residual_round_trips(self):
        """Team-only turnover events count toward team TOV."""
        game = _game_with_residual()
        events, ledger = _build_events_from_game(game)
        poss = LatentPossession(
            possession_index=0, period=1, seconds_remaining_estimate=700.0,
            offense_team_id=1001, defense_team_id=1002,
            offense_lineup=(), defense_lineup=(),
            events=tuple(events), end_reason=PossessionEnd.UNKNOWN,
            points=0, duration_seconds=0.0, boundary_confidence=0.0,
        )
        sample = ReconstructionSample(
            game_id="g1", sample_id="s0", seed=42, possessions=(poss,),
            home_rotation=None, away_rotation=None, hard_errors=(),
            soft_penalties={}, log_prior=0.0, log_likelihood=0.0,
            log_weight=0.0, box_score_exact=True,
        )
        agg = aggregate_sample(sample)
        # Team 1001: player TOV 2 + residual 3 = team TOV 5.
        assert agg.teams[1001].tov == 5
        assert agg.teams[1001].team_turnover_residual == 3
        # Team 1002: player TOV 1 + residual 0 = team TOV 1.
        assert agg.teams[1002].tov == 1
        assert agg.teams[1002].team_turnover_residual == 0

    def test_empty_sample_aggregates_to_empty(self):
        """A sample with no events produces zero counts."""
        poss = LatentPossession(
            possession_index=0, period=1, seconds_remaining_estimate=700.0,
            offense_team_id=1001, defense_team_id=1002,
            offense_lineup=(), defense_lineup=(),
            events=(), end_reason=PossessionEnd.UNKNOWN,
            points=0, duration_seconds=0.0, boundary_confidence=0.0,
        )
        sample = ReconstructionSample(
            game_id="g1", sample_id="s0", seed=42, possessions=(poss,),
            home_rotation=None, away_rotation=None, hard_errors=(),
            soft_penalties={}, log_prior=0.0, log_likelihood=0.0,
            log_weight=0.0, box_score_exact=True,
        )
        agg = aggregate_sample(sample)
        assert agg.players == {}
        # Teams still created (from possession offense_team_id).
        assert 1001 in agg.teams
        assert agg.teams[1001].pts == 0

    def test_events_to_player_counts_skips_team_events(self):
        """Team events (is_team_event=True) are not attributed to a player."""
        ev = _make_event("e0", EventType.TURNOVER, 1001, 1002, None,
                         "team_turnover", is_team=True)
        counts = events_to_player_counts((ev,))
        assert counts == {}

    def test_free_throw_made_and_missed_separate(self):
        made = _make_event("e0", EventType.FREE_THROW, 1001, 1002, 11,
                           "free_throw_made", made=True)
        missed = _make_event("e1", EventType.FREE_THROW, 1001, 1002, 11,
                             "free_throw_missed", made=False)
        counts = events_to_player_counts((made, missed))
        assert counts[11].ftm == 1
        assert counts[11].fta == 2
        assert counts[11].pts == 1

    def test_points_round_trip(self):
        """FG2=2, FG3=3, FT=1 points accumulate correctly."""
        events = (
            _make_event("e0", EventType.FG2_MADE, 1001, 1002, 11, "fg2_made"),
            _make_event("e1", EventType.FG3_MADE, 1001, 1002, 11, "fg3_made"),
            _make_event("e2", EventType.FREE_THROW, 1001, 1002, 11,
                        "free_throw_made", made=True),
        )
        counts = events_to_player_counts(events)
        assert counts[11].pts == 6  # 2 + 3 + 1


# ---------------------------------------------------------------------------
# Regulation fixture integration
# ---------------------------------------------------------------------------
class TestRegulationFixture:
    def test_total_token_count(self):
        game = _load_regulation_game()
        ledger = EventLedger.from_game(game)
        # Every box-score count contributes a token.
        expected = 0
        for tid in game.team_ids:
            for pl in game.players_by_team[tid]:
                expected += (
                    pl.fg2m + pl.fg2_missed + pl.fg3m + pl.fg3_missed
                    + pl.ftm + (pl.fta - pl.ftm) + pl.oreb + pl.dreb
                    + pl.ast + pl.tov + pl.stl + pl.blk + pl.blka + pl.pf + pl.pfd
                )
            # Team residual
            psum = sum(p.tov for p in game.players_by_team[tid])
            expected += game.teams[tid].tov - psum
        assert ledger.total_tokens() == expected

    def test_all_player_inventories_present(self):
        game = _load_regulation_game()
        ledger = EventLedger.from_game(game)
        for tid in game.team_ids:
            for pl in game.players_by_team[tid]:
                assert pl.player_id in ledger._player_inventories

    def test_full_round_trip_on_regulation_fixture(self):
        game = _load_regulation_game()
        events, ledger = _build_events_from_game(game)
        assert ledger.total_remaining() == 0
        ledger.assert_fully_consumed()

        poss = LatentPossession(
            possession_index=0, period=1, seconds_remaining_estimate=700.0,
            offense_team_id=1001, defense_team_id=1002,
            offense_lineup=(), defense_lineup=(),
            events=tuple(events), end_reason=PossessionEnd.UNKNOWN,
            points=0, duration_seconds=0.0, boundary_confidence=0.0,
        )
        sample = ReconstructionSample(
            game_id=game.game_id, sample_id="s0", seed=42, possessions=(poss,),
            home_rotation=None, away_rotation=None, hard_errors=(),
            soft_penalties={}, log_prior=0.0, log_likelihood=0.0,
            log_weight=0.0, box_score_exact=True,
        )
        agg = aggregate_sample(sample)

        # All hard player counts must match.
        for tid in game.team_ids:
            for pl in game.players_by_team[tid]:
                pc = agg.players[pl.player_id]
                for stat in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                             "oreb", "dreb", "reb", "ast", "tov", "stl",
                             "blk", "pf", "pfd"):
                    assert getattr(pc, stat) == getattr(pl, stat), (
                        f"player {pl.player_id} {stat}: "
                        f"agg={getattr(pc, stat)} src={getattr(pl, stat)}"
                    )
