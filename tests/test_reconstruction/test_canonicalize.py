"""Tests for box-score canonicalization (PR 02).

Covers the PR 02 exit gate:
* Regulation game infers zero overtime.
* Overtime fixture infers correct OT count.
* Team turnover residual is retained.
* Points-identity failure is rejected.
* Missing team row is quarantined.
* Negative count, made-over-attempt, duplicate player, wrong team, and
  plus/minus mismatch are reported as hard errors.
* Row order does not affect canonical output.
* Exact duplicate rows are dropped; conflicting duplicates are errors.
* Batch grouping + quarantine behavior.
* Source hashes are stable and order-independent within the hash column set.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reconstruction import (
    CanonicalizationError,
    canonicalize_game,
    canonicalize_games,
    group_games,
    hash_source_rows,
    parse_minutes_value,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "reconstruction"


def _load(name: str, *, game: bool = False):
    """Load a fixture pair. ``game=True`` uses the ``_game_`` infix names."""
    infix = "_game_" if game else "_"
    p = pd.read_csv(FIXTURES / f"{name}{infix}players.csv", dtype={"GAME_ID": str})
    t = pd.read_csv(FIXTURES / f"{name}{infix}teams.csv", dtype={"GAME_ID": str})
    return p, t


def _load_valid(name: str):
    p, t = _load(name, game=True)
    return p, t, p["GAME_ID"].iloc[0]


def _load_malformed(name: str):
    p, t = _load(name, game=False)
    return p, t, p["GAME_ID"].iloc[0]


# ---------------------------------------------------------------------------
# Valid fixtures
# ---------------------------------------------------------------------------
def test_regulation_game_canonicalizes_with_zero_ot():
    p, t, gid = _load_valid("regulation")
    game = canonicalize_game(gid, p, t)
    assert game.game_id == gid
    assert game.overtime_periods == 0
    assert game.team_ids == (1001, 1002)
    assert len(game.teams) == 2
    # Players sorted by id within each team.
    for tid, players in game.players_by_team.items():
        ids = [pl.player_id for pl in players]
        assert ids == sorted(ids)
    assert len(game.players_by_team[1001]) == 7
    assert len(game.players_by_team[1002]) == 7


def test_overtime_game_infers_one_ot():
    p, t, gid = _load_valid("overtime")
    game = canonicalize_game(gid, p, t)
    assert game.overtime_periods == 1
    assert game.team_ids == (2001, 2002)


def test_team_turnover_residual_is_retained_in_team_row():
    p, t, gid = _load_valid("regulation")
    game = canonicalize_game(gid, p, t)
    # Team 1001 has a residual of 2 (team TOV 7 - player sum 5).
    player_tov_a = sum(pl.tov for pl in game.players_by_team[1001])
    assert game.teams[1001].tov - player_tov_a == 2
    # Team 1002 has zero residual.
    player_tov_b = sum(pl.tov for pl in game.players_by_team[1002])
    assert game.teams[1002].tov - player_tov_b == 0


def test_game_date_and_season_parsed():
    p, t, gid = _load_valid("regulation")
    game = canonicalize_game(gid, p, t)
    assert game.game_date == date(2025, 1, 15)
    assert game.season_year == "2024-25"


def test_source_hashes_populated_and_stable():
    p, t, gid = _load_valid("regulation")
    game = canonicalize_game(gid, p, t)
    assert game.source_player_hash
    assert game.source_team_hash
    assert game.source_player_hash != game.source_team_hash
    # Re-canonicalizing identical rows yields identical hashes.
    game2 = canonicalize_game(gid, p, t)
    assert game2.source_player_hash == game.source_player_hash
    assert game2.source_team_hash == game.source_team_hash


# ---------------------------------------------------------------------------
# Row-order independence
# ---------------------------------------------------------------------------
def test_row_order_does_not_change_canonical_output():
    p, t, gid = _load_valid("regulation")
    # Shuffle player rows deterministically.
    rng = np.random.default_rng(0)
    shuffled_idx = rng.permutation(len(p))
    p_shuffled = p.iloc[shuffled_idx].reset_index(drop=True)
    g1 = canonicalize_game(gid, p, t)
    g2 = canonicalize_game(gid, p_shuffled, t)
    # All observable canonical fields must match.
    assert g1.team_ids == g2.team_ids
    assert g1.overtime_periods == g2.overtime_periods
    assert g1.source_player_hash == g2.source_player_hash
    for tid in g1.team_ids:
        ids1 = [pl.player_id for pl in g1.players_by_team[tid]]
        ids2 = [pl.player_id for pl in g2.players_by_team[tid]]
        assert ids1 == ids2  # both sorted


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def test_exact_duplicate_player_rows_are_dropped():
    p, t, gid = _load_malformed("malformed_duplicate_player_exact")
    game = canonicalize_game(gid, p, t)
    # The exact duplicate is silently dropped — 14 unique players remain.
    total = sum(len(v) for v in game.players_by_team.values())
    assert total == 14


def test_conflicting_duplicate_player_rows_rejected():
    p, t, gid = _load_malformed("malformed_duplicate_player")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("conflicting duplicate" in e for e in exc_info.value.report.hard_errors)


# ---------------------------------------------------------------------------
# Hard-contract rejections (one per failure type)
# ---------------------------------------------------------------------------
def test_points_identity_failure_rejected():
    p, t, gid = _load_malformed("malformed_points_identity")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    errs = exc_info.value.report.hard_errors
    assert any("PTS" in e and "FG2M" in e for e in errs)


def test_missing_team_row_quarantined():
    p, t, gid = _load_malformed("malformed_missing_team")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert "2 team rows" in str(exc_info.value)


def test_negative_count_rejected():
    p, t, gid = _load_malformed("malformed_negative_count")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("must be a nonnegative integer" in e for e in exc_info.value.report.hard_errors)


def test_made_over_attempt_rejected():
    p, t, gid = _load_malformed("malformed_made_over_attempt")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("0<=FGM<=FGA" in e for e in exc_info.value.report.hard_errors)


def test_wrong_team_player_rejected():
    p, t, gid = _load_malformed("malformed_wrong_team")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("not in game teams" in e for e in exc_info.value.report.hard_errors)


def test_plus_minus_mismatch_rejected():
    p, t, gid = _load_malformed("malformed_plus_minus")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("5*team PLUS_MINUS" in e for e in exc_info.value.report.hard_errors)


def test_team_sum_mismatch_rejected():
    p, t, gid = _load_malformed("malformed_team_sum")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("summed player" in e or "PTS" in e for e in exc_info.value.report.hard_errors)


def test_negative_turnover_residual_rejected():
    p, t, gid = _load_malformed("malformed_negative_tov_residual")
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_game(gid, p, t)
    assert any("residual" in e for e in exc_info.value.report.hard_errors)


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------
def test_wrong_number_of_team_rows_rejected():
    p, t, gid = _load_valid("regulation")
    # Add a third team row.
    extra = t.iloc[[0]].copy()
    extra["TEAM_ID"] = 9999
    t3 = pd.concat([t, extra], ignore_index=True)
    with pytest.raises(CanonicalizationError):
        canonicalize_game(gid, p, t3)


def test_same_team_id_twice_rejected():
    p, t, gid = _load_valid("regulation")
    t_dup = pd.concat([t[t.TEAM_ID == 1001], t[t.TEAM_ID == 1001]], ignore_index=True)
    with pytest.raises(CanonicalizationError):
        canonicalize_game(gid, p, t_dup)


def test_missing_core_column_raises():
    p, t, gid = _load_valid("regulation")
    p_bad = p.drop(columns=["PTS"])
    with pytest.raises(CanonicalizationError):
        canonicalize_game(gid, p_bad, t)


# ---------------------------------------------------------------------------
# Minutes parsing
# ---------------------------------------------------------------------------
def test_parse_minutes_value_decimal():
    assert parse_minutes_value(26.5) == 26.5
    assert parse_minutes_value("26.5") == 26.5


def test_parse_minutes_value_mmss():
    assert parse_minutes_value("26:30") == 26.5
    assert abs(parse_minutes_value("1:15") - 1.25) < 1e-9


def test_parse_minutes_value_missing():
    assert np.isnan(parse_minutes_value(np.nan))
    assert np.isnan(parse_minutes_value(""))
    assert np.isnan(parse_minutes_value(None))


# ---------------------------------------------------------------------------
# Optional BLKA/PFD
# ---------------------------------------------------------------------------
def test_missing_blka_defaults_to_zero_with_valid_game():
    p, t, gid = _load_valid("regulation")
    p_missing = p.drop(columns=["BLKA"])
    # BLKA missing -> players get blka=0; game still valid because BLKA does not
    # enter a hard constraint except the soft BLKA-vs-blocks warning.
    game = canonicalize_game(gid, p_missing, t)
    assert all(pl.blka == 0 for pl in game.players_by_team[1001])


# ---------------------------------------------------------------------------
# Batch grouping and quarantine
# ---------------------------------------------------------------------------
def test_group_games_preserves_first_appearance_order():
    p, t, gid = _load_valid("regulation")
    p2, t2, gid2 = _load_valid("overtime")
    big_p = pd.concat([p, p2], ignore_index=True)
    big_t = pd.concat([t, t2], ignore_index=True)
    groups = list(group_games(big_p, big_t))
    assert [g[0] for g in groups] == [gid, gid2]


def test_canonicalize_games_quarantines_invalid_and_continues():
    p_reg, t_reg, gid_reg = _load_valid("regulation")
    p_bad, t_bad, _ = _load_malformed("malformed_points_identity")
    # Relabel the bad game to a distinct GAME_ID so the batch sees two games.
    p_bad = p_bad.copy(); t_bad = t_bad.copy()
    gid_bad = "0099900001"
    p_bad["GAME_ID"] = gid_bad
    t_bad["GAME_ID"] = gid_bad
    big_p = pd.concat([p_reg, p_bad], ignore_index=True)
    big_t = pd.concat([t_reg, t_bad], ignore_index=True)
    valid, errors = canonicalize_games(big_p, big_t)
    assert len(valid) == 1
    assert valid[0].game_id == gid_reg
    assert len(errors) == 1
    assert errors[0].report.game_id == gid_bad


def test_canonicalize_games_strict_raises_on_first_invalid():
    p_reg, t_reg, _ = _load_valid("regulation")
    p_bad, t_bad, _ = _load_malformed("malformed_points_identity")
    p_bad = p_bad.copy(); t_bad = t_bad.copy()
    p_bad["GAME_ID"] = "0099900002"
    t_bad["GAME_ID"] = "0099900002"
    big_p = pd.concat([p_bad, p_reg], ignore_index=True)
    big_t = pd.concat([t_bad, t_reg], ignore_index=True)
    with pytest.raises(CanonicalizationError):
        canonicalize_games(big_p, big_t, strict=True)


# ---------------------------------------------------------------------------
# Source-row hashing
# ---------------------------------------------------------------------------
def test_hash_source_rows_deterministic():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    h1 = hash_source_rows(df, ("A", "B"))
    h2 = hash_source_rows(df, ("A", "B"))
    assert h1 == h2
    # Different column subset -> different hash.
    h3 = hash_source_rows(df, ("A",))
    assert h3 != h1


def test_hash_source_rows_empty():
    df = pd.DataFrame({"A": []})
    h = hash_source_rows(df, ("A",))
    assert isinstance(h, str) and len(h) == 64
