"""Canonicalization: turn raw CSV rows into validated ``GameBoxScore`` objects.

Implements Section 9 of the reconstruction plan. This is the bridge between the
existing aggregate logs (``nba_players.csv`` / ``nba_games.csv``) and the typed
reconstruction domain. It is intentionally strict: a game that fails any hard
arithmetic or structural constraint is rejected with a ``CanonicalizationError``
carrying a full ``GameValidationReport`` so the batch runner can quarantine it.

Normalization rules (Section 9.2)
---------------------------------
* ``GAME_ID`` is read as a string (leading zeroes preserved).
* ``GAME_DATE`` and numeric stats use ``errors="raise"`` in strict mode.
* ``MIN`` is parsed through ``_parse_minutes`` (decimal or ``MM:SS``), a minimal
  duplicate of ``DataLoader._parse_minutes`` so this module does not import the
  whole data pipeline.
* Missing optional ``BLKA`` / ``PFD`` default to zero with a warning; missing
  core fields raise.
* Exact-duplicate player rows are dropped; conflicting duplicates are a hard
  error.
* Exactly two distinct team IDs are required; every player's team must be one
  of them; players are sorted by player ID for deterministic output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.reconstruction.contracts import (
    GameValidationReport,
    infer_overtime,
    validate_game_box_score,
)
from src.reconstruction.schema import (
    GameBoxScore,
    PlayerBoxLine,
    TeamBoxLine,
)

__all__ = [
    "CanonicalizationError",
    "canonicalize_game",
    "validate_game_box_score",
    "group_games",
    "canonicalize_games",
    "parse_minutes_value",
    "hash_source_rows",
]

# Canonical columns read from the player source file.
PLAYER_STAT_COLUMNS: Tuple[str, ...] = (
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
)
# Core fields whose absence is fatal (Section 9.2 rule 5).
PLAYER_CORE_COLUMNS: Tuple[str, ...] = tuple(
    c for c in PLAYER_STAT_COLUMNS if c not in ("BLKA", "PFD")
) + ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE")
# Optional fields that may be missing and default to zero.
OPTIONAL_PLAYER_COLUMNS: Tuple[str, ...] = ("BLKA", "PFD")


class CanonicalizationError(ValueError):
    """Raised when a game cannot be canonicalized (hard contract failure)."""

    def __init__(self, message: str, report: GameValidationReport) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Minimal minutes parser (Section 9.2 rule 4)
# ---------------------------------------------------------------------------
def parse_minutes_value(val) -> float:
    """Parse ``MIN`` accepting decimal floats or ``MM:SS`` strings.

    A trimmed duplicate of ``DataLoader._parse_minutes`` so canonicalization
    does not pull in the full data pipeline. Returns ``NaN`` for missing.
    """
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return float("nan")
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Source-row hashing
# ---------------------------------------------------------------------------
# Columns included in a player source-row hash. The canonical stat columns plus
# identity/context keep the hash sensitive to any change that could affect the
# reconstruction, while ignoring derived rank/fantasy columns.
_PLAYER_HASH_COLUMNS: Tuple[str, ...] = (
    "PLAYER_ID", "TEAM_ID", "GAME_ID", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS",
    "PLUS_MINUS",
)
_TEAM_HASH_COLUMNS: Tuple[str, ...] = (
    "TEAM_ID", "GAME_ID", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS",
    "PLUS_MINUS",
)


def _row_payload(row: pd.Series, columns: Tuple[str, ...]) -> str:
    """Stable JSON string of the hash-relevant columns of one row."""
    payload = {col: row.get(col) for col in columns}
    # Normalize NaN -> null so a missing optional field does not destabilize
    # the hash against an explicit zero.
    clean = {}
    for key, val in payload.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            clean[key] = None
        elif isinstance(val, (np.integer,)):
            clean[key] = int(val)
        elif isinstance(val, (np.floating,)):
            clean[key] = float(val)
        elif isinstance(val, (np.bool_,)):
            clean[key] = bool(val)
        else:
            clean[key] = val
    return json.dumps(clean, sort_keys=True, default=str)


def hash_source_rows(rows: pd.DataFrame, columns: Tuple[str, ...]) -> str:
    """Deterministic SHA-256 over the hash-relevant columns of a row set.

    Rows are serialized in their current order and hashed as one JSON list so
    the digest reflects both values and row membership. Output is a hex string.
    """
    if columns is None:
        raise ValueError("columns must be provided")
    if len(rows) == 0:
        return hashlib.sha256(b"[]").hexdigest()
    payload = json.dumps(
        [_row_payload(rows.iloc[i], columns) for i in range(len(rows))],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _require_columns(
    df: pd.DataFrame, columns: Iterable[str], kind: str
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CanonicalizationError(
            f"missing {kind} columns: {missing}",
            GameValidationReport(game_id="?"),
        )


def _normalize_player_rows(player_rows: pd.DataFrame) -> pd.DataFrame:
    _require_columns(player_rows, PLAYER_CORE_COLUMNS, "player core")

    df = player_rows.copy()
    # GAME_ID as string (preserve leading zeroes).
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    # GAME_DATE strict parse.
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise").dt.date

    # Optional missing fields -> 0. The column may be entirely absent
    # (Section 9.2 rule 5) or present with NaNs. Added before numeric
    # conversion so the strict ``to_numeric`` loop always sees the column.
    for col in OPTIONAL_PLAYER_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        elif df[col].isna().any():
            df[col] = df[col].fillna(0)

    # Numeric stat columns strict conversion.
    for col in PLAYER_STAT_COLUMNS:
        if col == "MIN":
            continue
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["MIN"] = df["MIN"].apply(parse_minutes_value)

    # Integer coercions for count stats.
    for col in PLAYER_STAT_COLUMNS:
        if col in ("MIN", "PLUS_MINUS"):
            continue
        df[col] = df[col].astype(int)

    return df


def _normalize_team_rows(team_rows: pd.DataFrame) -> pd.DataFrame:
    team_core = tuple(
        c for c in PLAYER_STAT_COLUMNS if c not in ("BLKA", "PFD")
    ) + ("TEAM_ID", "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP")
    _require_columns(team_rows, team_core, "team core")

    df = team_rows.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise").dt.date

    # Optional columns first (same rationale as the player path).
    for col in OPTIONAL_PLAYER_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        elif df[col].isna().any():
            df[col] = df[col].fillna(0)

    for col in PLAYER_STAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in PLAYER_STAT_COLUMNS:
        if col in ("MIN", "PLUS_MINUS"):
            continue
        df[col] = df[col].astype(int)
    return df


def _build_player_line(row: pd.Series) -> PlayerBoxLine:
    return PlayerBoxLine(
        game_id=str(row["GAME_ID"]),
        team_id=int(row["TEAM_ID"]),
        player_id=int(row["PLAYER_ID"]),
        player_name=str(row["PLAYER_NAME"]),
        minutes=float(row["MIN"]),
        fgm=int(row["FGM"]),
        fga=int(row["FGA"]),
        fg3m=int(row["FG3M"]),
        fg3a=int(row["FG3A"]),
        ftm=int(row["FTM"]),
        fta=int(row["FTA"]),
        oreb=int(row["OREB"]),
        dreb=int(row["DREB"]),
        reb=int(row["REB"]),
        ast=int(row["AST"]),
        tov=int(row["TOV"]),
        stl=int(row["STL"]),
        blk=int(row["BLK"]),
        blka=int(row["BLKA"]),
        pf=int(row["PF"]),
        pfd=int(row["PFD"]),
        pts=int(row["PTS"]),
        plus_minus=float(row["PLUS_MINUS"]),
    )


def _build_team_line(row: pd.Series) -> TeamBoxLine:
    return TeamBoxLine(
        game_id=str(row["GAME_ID"]),
        team_id=int(row["TEAM_ID"]),
        team_abbreviation=str(row["TEAM_ABBREVIATION"]),
        game_date=row["GAME_DATE"],
        matchup=str(row.get("MATCHUP", "")),
        minutes=float(row["MIN"]),
        fgm=int(row["FGM"]),
        fga=int(row["FGA"]),
        fg3m=int(row["FG3M"]),
        fg3a=int(row["FG3A"]),
        ftm=int(row["FTM"]),
        fta=int(row["FTA"]),
        oreb=int(row["OREB"]),
        dreb=int(row["DREB"]),
        reb=int(row["REB"]),
        ast=int(row["AST"]),
        tov=int(row["TOV"]),
        stl=int(row["STL"]),
        blk=int(row["BLK"]),
        blka=int(row["BLKA"]),
        pf=int(row["PF"]),
        pfd=int(row["PFD"]),
        pts=int(row["PTS"]),
        team_plus_minus=float(row["PLUS_MINUS"]),
    )


def _dedupe_players(
    df: pd.DataFrame, report: GameValidationReport
) -> pd.DataFrame:
    """Drop exact duplicate player rows; reject conflicting duplicates.

    A conflict is two rows with the same ``(GAME_ID, TEAM_ID, PLAYER_ID)``
    that differ on any canonical field. Exact duplicates (every field equal)
    are collapsed to one (Section 9.2 rule 6).
    """
    key = ["GAME_ID", "TEAM_ID", "PLAYER_ID"]
    canonical_view = df[key + [c for c in df.columns if c not in key]]

    exact_dupes = df.duplicated(keep="first")
    if exact_dupes.any():
        df = df[~exact_dupes].copy()

    # Detect remaining key collisions -> conflicting duplicates.
    conflicts = df.duplicated(subset=key, keep=False)
    if conflicts.any():
        conflicting = df[conflicts]
        for (_gid, _tid, pid), _grp in conflicting.groupby(key):
            distinct_rows = _grp.drop_duplicates().shape[0]
            if distinct_rows > 1:
                report.hard_errors.append(
                    f"player {int(pid)}: conflicting duplicate rows "
                    f"(same GAME_ID/TEAM_ID/PLAYER_ID, different stats)"
                )
    return df


# ---------------------------------------------------------------------------
# Public API (Section 9.1)
# ---------------------------------------------------------------------------
def canonicalize_game(
    game_id: str,
    player_rows: pd.DataFrame,
    team_rows: pd.DataFrame,
) -> GameBoxScore:
    """Canonicalize one game's raw rows into a validated ``GameBoxScore``.

    Raises ``CanonicalizationError`` if any hard contract fails. The error
    carries the full ``GameValidationReport`` for quarantine reporting.
    """
    game_id = str(game_id)
    report = GameValidationReport(game_id=game_id)

    try:
        players = _normalize_player_rows(player_rows)
        teams = _normalize_team_rows(team_rows)
    except CanonicalizationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive strict-mode errors
        raise CanonicalizationError(
            f"normalization failed: {exc}", report
        ) from exc

    # Structure: exactly two team rows with distinct team IDs.
    if len(teams) != 2:
        raise CanonicalizationError(
            f"expected exactly 2 team rows, got {len(teams)}",
            report,
        )
    team_ids = tuple(sorted(int(t) for t in teams["TEAM_ID"].tolist()))
    if len(set(team_ids)) != 2:
        raise CanonicalizationError(
            f"team rows must have distinct team IDs, got {list(team_ids)}",
            report,
        )

    # Every player's team must be one of the two team IDs.
    bad_teams = players.loc[~players["TEAM_ID"].isin(team_ids), "TEAM_ID"]
    if len(bad_teams) > 0:
        for tid in bad_teams.unique():
            report.hard_errors.append(
                f"player row has TEAM_ID={int(tid)} not in game teams {list(team_ids)}"
            )

    # Deduplicate players (exact dupes dropped, conflicts flagged).
    players = _dedupe_players(players, report)

    # Sort players by player ID for deterministic serialization.
    players = players.sort_values("PLAYER_ID", kind="mergesort").reset_index(drop=True)

    # Build typed objects.
    team_lines: Dict[int, TeamBoxLine] = {}
    for _, row in teams.iterrows():
        tid = int(row["TEAM_ID"])
        team_lines[tid] = _build_team_line(row)

    players_by_team: Dict[int, Tuple[PlayerBoxLine, ...]] = {
        tid: () for tid in team_ids
    }
    for tid in team_ids:
        team_players = players[players["TEAM_ID"] == tid]
        players_by_team[tid] = tuple(
            _build_player_line(row) for _, row in team_players.iterrows()
        )

    # Overtime inference from summed player minutes.
    sum_a = float(sum(p.minutes for p in players_by_team[team_ids[0]]))
    sum_b = float(sum(p.minutes for p in players_by_team[team_ids[1]]))
    overtime_periods = infer_overtime(sum_a, sum_b, report)

    # Season year + game date from the first team row (both teams share them).
    first_team = team_lines[team_ids[0]]
    game_date = first_team.game_date
    season_year = ""
    if "SEASON_YEAR" in teams.columns:
        season_year = str(teams["SEASON_YEAR"].iloc[0])
    elif "SEASON_YEAR" in player_rows.columns:
        season_year = str(player_rows["SEASON_YEAR"].iloc[0])

    game = GameBoxScore(
        game_id=game_id,
        game_date=game_date,
        season_year=season_year,
        team_ids=team_ids,
        teams=team_lines,
        players_by_team=players_by_team,
        overtime_periods=overtime_periods,
        source_player_hash=hash_source_rows(players, _PLAYER_HASH_COLUMNS),
        source_team_hash=hash_source_rows(teams, _TEAM_HASH_COLUMNS),
    )

    # Run the full validation suite on the assembled object.
    full_report = validate_game_box_score(game)
    # Merge metrics/errors discovered during normalization (e.g. conflicting
    # duplicates) into the final report.
    full_report.hard_errors = report.hard_errors + full_report.hard_errors
    full_report.warnings = report.warnings + full_report.warnings

    if not full_report.ok:
        raise CanonicalizationError(
            f"game {game_id} failed hard contracts: "
            f"{len(full_report.hard_errors)} error(s)",
            full_report,
        )
    return game


def group_games(
    player_df: pd.DataFrame,
    team_df: pd.DataFrame,
) -> Iterable[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(game_id, player_rows, team_rows)`` tuples grouped by ``GAME_ID``.

    Iterates lazily so a very large source file does not require holding every
    candidate object in memory at once (Section 9.2 step 5). Only the game's
    own rows are yielded.
    """
    if "GAME_ID" not in player_df.columns or "GAME_ID" not in team_df.columns:
        raise ValueError("both frames must have a GAME_ID column")
    # Preserve first-appearance order for deterministic batch sequencing.
    seen: List[str] = []
    seen_set: set = set()
    for gid in player_df["GAME_ID"].astype(str):
        if gid not in seen_set:
            seen_set.add(gid)
            seen.append(gid)
    for gid in team_df["GAME_ID"].astype(str):
        if gid not in seen_set:
            seen_set.add(gid)
            seen.append(gid)
    for gid in seen:
        p_rows = player_df[player_df["GAME_ID"].astype(str) == gid]
        t_rows = team_df[team_df["GAME_ID"].astype(str) == gid]
        yield gid, p_rows, t_rows


def canonicalize_games(
    player_df: pd.DataFrame,
    team_df: pd.DataFrame,
    *,
    strict: bool = False,
) -> Tuple[List[GameBoxScore], List[CanonicalizationError]]:
    """Canonicalize every game in the source frames.

    Returns ``(valid_games, errors)``. By default an invalid game is appended
    to ``errors`` and the batch continues (Section 9.6 quarantine behavior).
    When ``strict`` is True the first invalid game raises immediately.
    """
    valid: List[GameBoxScore] = []
    errors: List[CanonicalizationError] = []
    for gid, p_rows, t_rows in group_games(player_df, team_df):
        try:
            valid.append(canonicalize_game(gid, p_rows, t_rows))
        except CanonicalizationError as exc:
            if strict:
                raise
            errors.append(exc)
    return valid, errors
