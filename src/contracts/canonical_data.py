"""Schema contracts for snapshot-scoped canonical basketball data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from src.contracts.errors import ContractError
from src.contracts.sources import parse_aware_datetime


CANONICAL_SCHEMA_VERSION = "canonical_data_v2"


@dataclass(frozen=True)
class CanonicalTableContract:
    """Minimal stable schema for one canonical table.

    Producers may append source-specific label columns, but the identity,
    provenance, and point-in-time columns below are always present.
    """

    name: str
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    id_columns: tuple[str, ...] = ()
    require_available_at: bool = True


CANONICAL_TABLE_CONTRACTS: dict[str, CanonicalTableContract] = {
    "games": CanonicalTableContract(
        "games",
        (
            "GAME_ID", "GAME_DATE", "SCHEDULED_TIP", "HOME_TEAM_ID",
            "AWAY_TEAM_ID", "STATUS", "EVENT_TIME", "AVAILABLE_AT", "SOURCE",
        ),
        ("GAME_ID",),
        ("GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"),
    ),
    "team_games": CanonicalTableContract(
        "team_games",
        (
            "GAME_ID", "GAME_DATE", "TEAM_ID", "OPPONENT_ID", "HOME_FLAG",
            "EVENT_TIME", "AVAILABLE_AT", "SOURCE",
        ),
        ("GAME_ID", "TEAM_ID"),
        ("GAME_ID", "TEAM_ID", "OPPONENT_ID"),
    ),
    "player_games": CanonicalTableContract(
        "player_games",
        (
            "GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM_ID", "EVENT_TIME",
            "AVAILABLE_AT", "SOURCE",
        ),
        ("GAME_ID", "TEAM_ID", "PLAYER_ID"),
        ("GAME_ID", "TEAM_ID", "PLAYER_ID"),
    ),
    "roster_membership": CanonicalTableContract(
        "roster_membership",
        (
            "PLAYER_ID", "TEAM_ID", "START_DATE", "END_DATE", "EVENT_TIME",
            "AVAILABLE_AT", "SOURCE", "COVERAGE_STATUS",
        ),
        ("PLAYER_ID", "TEAM_ID", "START_DATE"),
        ("PLAYER_ID", "TEAM_ID"),
    ),
    "player_game_eligibility": CanonicalTableContract(
        "player_game_eligibility",
        (
            "GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM_ID", "ROSTER_ELIGIBLE",
            "ACTIVE", "APPEARED", "STARTED", "DNP_REASON", "EVENT_TIME",
            "AVAILABLE_AT", "SOURCE",
        ),
        ("GAME_ID", "TEAM_ID", "PLAYER_ID"),
        ("GAME_ID", "TEAM_ID", "PLAYER_ID"),
    ),
    "player_status_snapshots": CanonicalTableContract(
        "player_status_snapshots",
        (
            "PLAYER_ID", "TEAM_ID", "STATUS_RAW", "STATUS_NORMALIZED",
            "EVENT_TIME", "AVAILABLE_AT", "SOURCE",
        ),
        ("PLAYER_ID", "TEAM_ID", "AVAILABLE_AT", "SOURCE"),
        ("PLAYER_ID",),
    ),
    "lineup_snapshots": CanonicalTableContract(
        "lineup_snapshots",
        (
            "GAME_ID", "PLAYER_ID", "TEAM_ID", "STARTER_STATE", "EVENT_TIME",
            "AVAILABLE_AT", "SOURCE",
        ),
        ("GAME_ID", "PLAYER_ID", "TEAM_ID", "AVAILABLE_AT", "SOURCE"),
        ("GAME_ID", "PLAYER_ID", "TEAM_ID"),
    ),
    "odds_snapshots": CanonicalTableContract(
        "odds_snapshots",
        (
            "GAME_ID", "SPORTSBOOK", "EVENT_TIME", "AVAILABLE_AT", "SOURCE",
        ),
        ("GAME_ID", "SPORTSBOOK", "AVAILABLE_AT", "SOURCE"),
        ("GAME_ID",),
    ),
    "player_bios": CanonicalTableContract(
        "player_bios",
        (
            "PLAYER_ID", "BIRTHDATE", "POSITION", "HEIGHT", "WEIGHT",
            "EXPERIENCE", "AS_OF", "EVENT_TIME", "AVAILABLE_AT", "SOURCE",
        ),
        ("PLAYER_ID", "AS_OF", "SOURCE"),
        ("PLAYER_ID",),
    ),
}

CANONICAL_TABLE_NAMES = tuple(CANONICAL_TABLE_CONTRACTS)


def empty_canonical_table(table: str) -> pd.DataFrame:
    """Return an empty frame with the table's contract columns."""

    contract = get_canonical_contract(table)
    return pd.DataFrame(columns=list(contract.required_columns))


def get_canonical_contract(table: str) -> CanonicalTableContract:
    try:
        return CANONICAL_TABLE_CONTRACTS[table]
    except KeyError as exc:
        raise ContractError(f"Unknown canonical table: {table!r}") from exc


def validate_canonical_table(
    frame: pd.DataFrame,
    table: str,
    *,
    snapshot_created_at: datetime | str | None = None,
) -> None:
    """Validate identity, uniqueness, and observation-time invariants."""

    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"Canonical {table} must be a pandas DataFrame")
    contract = get_canonical_contract(table)
    missing = sorted(set(contract.required_columns) - set(frame.columns))
    if missing:
        raise ContractError(
            f"Canonical {table} is missing columns: " + ", ".join(missing)
        )
    if frame.empty:
        return

    for column in contract.id_columns:
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ContractError(f"Canonical {table}.{column} contains empty IDs")
    if frame.duplicated(list(contract.primary_key), keep=False).any():
        raise ContractError(
            f"Canonical {table} has duplicate primary key rows: "
            + ", ".join(contract.primary_key)
        )

    if contract.require_available_at:
        available = _parse_timestamp_series(
            frame["AVAILABLE_AT"], f"Canonical {table}.AVAILABLE_AT"
        )
        if available.isna().any():
            raise ContractError(f"Canonical {table}.AVAILABLE_AT cannot be null")
        if snapshot_created_at is not None:
            created = pd.Timestamp(
                parse_aware_datetime(snapshot_created_at, field="snapshot_created_at")
            )
            if (available > created).any():
                raise ContractError(
                    f"Canonical {table} contains rows observed after the snapshot"
                )

    if table == "games":
        same_team = (
            frame["HOME_TEAM_ID"].astype(str)
            == frame["AWAY_TEAM_ID"].astype(str)
        )
        if same_team.any():
            raise ContractError("Canonical games cannot have the same home and away team")
    elif table == "team_games":
        invalid_home = ~pd.to_numeric(frame["HOME_FLAG"], errors="coerce").isin([0, 1])
        if invalid_home.any():
            raise ContractError("Canonical team_games.HOME_FLAG must be 0 or 1")
        self_opponent = frame["TEAM_ID"].astype(str) == frame["OPPONENT_ID"].astype(str)
        if self_opponent.any():
            raise ContractError("Canonical team game cannot list itself as opponent")


def normalize_id_series(values: pd.Series, *, field: str) -> pd.Series:
    """Normalize CSV-coerced identifiers without fuzzy matching."""

    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if result.isna().any() or result.eq("").any():
        raise ContractError(f"{field} contains empty identifiers")
    return result


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{source} is missing columns: " + ", ".join(missing))


def _parse_timestamp_series(values: pd.Series, field: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    except Exception as exc:
        raise ContractError(f"{field} contains malformed timestamps") from exc
    malformed = values.notna() & parsed.isna()
    if malformed.any():
        raise ContractError(f"{field} contains malformed timestamps")
    return parsed


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CANONICAL_TABLE_CONTRACTS",
    "CANONICAL_TABLE_NAMES",
    "CanonicalTableContract",
    "empty_canonical_table",
    "get_canonical_contract",
    "normalize_id_series",
    "require_columns",
    "validate_canonical_table",
]
