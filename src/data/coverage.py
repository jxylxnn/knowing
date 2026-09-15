"""Coverage and reconciliation reports for snapshot-scoped canonical data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import pandas as pd
from pathlib import Path

from src.data.canonicalize import load_canonical_table
from src.data.snapshots import validate_source_snapshot, find_snapshot_file


@dataclass(frozen=True)
class DataCoverageReport:
    snapshot_id: str
    source_files: int
    games: int
    valid_final_games: int
    team_games: int
    player_games: int
    games_with_exactly_two_teams: int
    missing_player_team_game: int
    observed_roster_only: bool
    eligible_player_games: int = 0
    known_participation: int = 0
    unknown_participation: int = 0
    invalid_source_games: int = 0
    observed_eligibility_only: bool = True

    @property
    def core_reconciled_fraction(self) -> float:
        return self.valid_final_games / self.games if self.games else 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["core_reconciled_fraction"] = self.core_reconciled_fraction
        return payload


def build_coverage_report(data_dir: str | Path, snapshot_id: str) -> DataCoverageReport:
    manifest = validate_source_snapshot(data_dir, snapshot_id)
    games = load_canonical_table(data_dir, snapshot_id, "games")
    team_games = load_canonical_table(data_dir, snapshot_id, "team_games")
    player_games = load_canonical_table(data_dir, snapshot_id, "player_games")
    team_counts = team_games.groupby("GAME_ID")["TEAM_ID"].nunique()
    player_keys = player_games[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    team_keys = team_games[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    missing = player_keys.merge(team_keys, on=["GAME_ID", "TEAM_ID"], how="left", indicator=True)
    roster = load_canonical_table(data_dir, snapshot_id, "roster_membership")
    eligibility = load_canonical_table(data_dir, snapshot_id, "player_game_eligibility")
    source_games = pd.read_csv(find_snapshot_file(
        data_dir, snapshot_id, ("nba_games.csv",), required=True
    ), dtype=str)
    source_players = pd.read_csv(find_snapshot_file(
        data_dir, snapshot_id, ("nba_players.csv",), required=True
    ), dtype=str, usecols=["GAME_ID"])
    # A game absent from team logs but present in player outcomes still counts
    # against reconciliation. Quarantining its rows must not erase the gap.
    expected_games = len(set(source_games.GAME_ID) | set(source_players.GAME_ID))
    metadata = json.loads((Path(data_dir) / "canonical" / f"snapshot={snapshot_id}"
                           / "manifest.json").read_text())
    expected_games -= int(metadata["report"]["remapped_game_ids"])
    return DataCoverageReport(
        snapshot_id=snapshot_id,
        source_files=len(manifest.files), games=expected_games,
        valid_final_games=int(games["STATUS"].eq("final").sum()),
        team_games=len(team_games), player_games=len(player_games),
        games_with_exactly_two_teams=int(team_counts.eq(2).sum()),
        missing_player_team_game=int(missing["_merge"].eq("left_only").sum()),
        observed_roster_only=bool(roster.empty or not roster.COVERAGE_STATUS.eq("official").all()),
        eligible_player_games=len(eligibility),
        known_participation=int(eligibility.APPEARED.notna().sum()),
        unknown_participation=int(eligibility.APPEARED.isna().sum()),
        invalid_source_games=max(0, expected_games - len(games)),
        observed_eligibility_only=bool(eligibility.empty or
                                      eligibility.SOURCE.eq("derived/player_appearances").all()),
    )
