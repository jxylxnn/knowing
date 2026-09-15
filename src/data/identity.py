"""Exact identity resolution and quarantine reporting for canonical data.

The module never fuzzy-matches names.  A player-only game identifier may be
mapped to a team-game identifier only when game date and the complete two-team
set select exactly one team game.  Everything else is retained in a quarantine
report and excluded from official canonical joins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd

from src.contracts.canonical_data import normalize_id_series, require_columns
from src.contracts.errors import ContractError


@dataclass(frozen=True)
class IdentityResolutionReport:
    player_game_ids: int
    exact_game_ids: int
    remapped_game_ids: int
    quarantined_game_ids: int
    remapped_rows: int
    quarantined_rows: int
    reconstructed_team_games: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class IdentityResolutionResult:
    player_games: pd.DataFrame
    quarantine: pd.DataFrame
    mappings: pd.DataFrame
    report: IdentityResolutionReport


def resolve_player_game_ids(
    player_games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> IdentityResolutionResult:
    """Resolve safely identifiable player-only game IDs.

    Exact IDs pass through.  A differing ID is remapped only if:

    * every player row agrees on one game date;
    * the player rows contain exactly two distinct teams; and
    * date plus that two-team set identifies exactly one team-log game.

    The original ID is preserved in ``SOURCE_GAME_ID``.  Unresolved rows are
    returned in ``quarantine`` rather than silently dropped or guessed.
    """

    require_columns(
        player_games,
        ("GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID"),
        source="player_games",
    )
    require_columns(
        team_games,
        ("GAME_ID", "GAME_DATE", "TEAM_ID"),
        source="team_games",
    )
    players = player_games.copy()
    teams = team_games.copy()
    for frame, columns, label in (
        (players, ("GAME_ID", "TEAM_ID", "PLAYER_ID"), "player_games"),
        (teams, ("GAME_ID", "TEAM_ID"), "team_games"),
    ):
        for column in columns:
            frame[column] = normalize_id_series(
                frame[column], field=f"{label}.{column}"
            )
        frame["GAME_DATE"] = pd.to_datetime(
            frame["GAME_DATE"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        if frame["GAME_DATE"].isna().any():
            raise ContractError(f"{label}.GAME_DATE contains malformed dates")

    canonical_ids = set(teams["GAME_ID"])
    player_ids = tuple(players["GAME_ID"].drop_duplicates())
    exact_ids = {game_id for game_id in player_ids if game_id in canonical_ids}

    candidates: dict[tuple[str, frozenset[str]], list[str]] = {}
    for game_id, group in teams.groupby("GAME_ID", sort=False):
        dates = tuple(group["GAME_DATE"].drop_duplicates())
        team_ids = frozenset(group["TEAM_ID"])
        if len(dates) == 1 and len(team_ids) == 2:
            candidates.setdefault((dates[0], team_ids), []).append(str(game_id))

    mappings: list[dict[str, Any]] = []
    quarantine_ids: list[dict[str, Any]] = []
    for source_game_id, group in players.loc[
        ~players["GAME_ID"].isin(exact_ids)
    ].groupby("GAME_ID", sort=False):
        dates = tuple(group["GAME_DATE"].drop_duplicates())
        player_teams = frozenset(group["TEAM_ID"])
        reason: str | None = None
        matched: list[str] = []
        if len(dates) != 1:
            reason = "conflicting_or_missing_game_date"
        elif len(player_teams) != 2:
            reason = "incomplete_player_team_set"
        else:
            matched = candidates.get((dates[0], player_teams), [])
            if not matched:
                reason = "no_team_game_match"
            elif len(matched) > 1:
                reason = "ambiguous_team_game_match"
        if reason is None:
            mappings.append(
                {
                    "SOURCE_GAME_ID": str(source_game_id),
                    "CANONICAL_GAME_ID": matched[0],
                    "GAME_DATE": dates[0],
                    "TEAM_IDS": ",".join(sorted(player_teams)),
                    "RESOLUTION": "exact_date_and_two_team_set",
                }
            )
        else:
            quarantine_ids.append(
                {
                    "SOURCE_GAME_ID": str(source_game_id),
                    "GAME_DATE": dates[0] if len(dates) == 1 else None,
                    "TEAM_IDS": ",".join(sorted(player_teams)),
                    "REASON": reason,
                    "CANDIDATE_GAME_IDS": ",".join(sorted(matched)),
                    "ROW_COUNT": int(len(group)),
                }
            )

    mapping_frame = pd.DataFrame(
        mappings,
        columns=(
            "SOURCE_GAME_ID", "CANONICAL_GAME_ID", "GAME_DATE", "TEAM_IDS",
            "RESOLUTION",
        ),
    )
    quarantine_summary = pd.DataFrame(
        quarantine_ids,
        columns=(
            "SOURCE_GAME_ID", "GAME_DATE", "TEAM_IDS", "REASON",
            "CANDIDATE_GAME_IDS", "ROW_COUNT",
        ),
    )
    mapping = dict(
        zip(mapping_frame.get("SOURCE_GAME_ID", []),
            mapping_frame.get("CANONICAL_GAME_ID", []))
    )
    quarantined_ids = set(quarantine_summary.get("SOURCE_GAME_ID", []))
    players["SOURCE_GAME_ID"] = players["GAME_ID"]
    players["GAME_ID"] = players["GAME_ID"].replace(mapping)
    quarantine_rows = players.loc[
        players["SOURCE_GAME_ID"].isin(quarantined_ids)
    ].copy()
    resolved = players.loc[
        ~players["SOURCE_GAME_ID"].isin(quarantined_ids)
    ].copy()
    remapped_rows = int(resolved["SOURCE_GAME_ID"].isin(mapping).sum())
    report = IdentityResolutionReport(
        player_game_ids=len(player_ids),
        exact_game_ids=len(exact_ids),
        remapped_game_ids=len(mapping),
        quarantined_game_ids=len(quarantined_ids),
        remapped_rows=remapped_rows,
        quarantined_rows=len(quarantine_rows),
    )
    return IdentityResolutionResult(
        player_games=resolved.reset_index(drop=True),
        quarantine=quarantine_rows.reset_index(drop=True),
        mappings=mapping_frame,
        report=report,
    )


def apply_exact_aliases(
    values: pd.Series,
    aliases: Mapping[str, str],
    *,
    field: str,
) -> pd.Series:
    """Apply an explicit alias map; reject aliases with empty destinations."""

    normalized_aliases: dict[str, str] = {}
    for raw_source, raw_target in aliases.items():
        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if not source or not target:
            raise ContractError(f"{field} alias keys and values must be non-empty")
        previous = normalized_aliases.get(source)
        if previous is not None and previous != target:
            raise ContractError(f"Ambiguous {field} alias: {source!r}")
        normalized_aliases[source] = target
    normalized = normalize_id_series(values, field=field)
    return normalized.replace(normalized_aliases)


__all__ = [
    "IdentityResolutionReport",
    "IdentityResolutionResult",
    "apply_exact_aliases",
    "resolve_player_game_ids",
]
