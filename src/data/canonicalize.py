"""Canonical, snapshot-scoped data products for Model v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.contracts.canonical_data import (
    CANONICAL_SCHEMA_VERSION,
    empty_canonical_table,
    get_canonical_contract,
    normalize_id_series,
    require_columns,
    validate_canonical_table,
)
from src.contracts.errors import ContractError
from src.contracts.sources import parse_aware_datetime
from src.data.identity import resolve_player_game_ids
from src.data.snapshots import find_snapshot_file, validate_source_snapshot


TRANSFORMATION_VERSION = "canonical_materializer_v1"


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class CanonicalizationReport:
    snapshot_id: str
    games: int
    team_games: int
    player_games: int
    invalid_games: int
    missing_player_team_game: int
    quarantined_player_rows: int
    remapped_game_ids: int
    observed_roster_only: bool = True
    observed_eligibility_only: bool = True
    disabled_optional_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonicalize_snapshot(
    data_dir: str | Path,
    snapshot_id: str,
) -> CanonicalizationReport:
    """Build immutable Parquet tables from a validated source snapshot."""

    root = Path(data_dir).resolve()
    manifest = validate_source_snapshot(root, snapshot_id)
    destination = root / "canonical" / f"snapshot={snapshot_id}"
    if destination.exists():
        raise FileExistsError(f"Canonical snapshot already exists: {snapshot_id}")

    players_path = find_snapshot_file(
        root, snapshot_id, ("nba_players.csv",), required=True
    )
    teams_path = find_snapshot_file(
        root, snapshot_id, ("nba_games.csv",), required=True
    )
    players_raw = pd.read_csv(players_path, dtype=str)
    teams_raw = pd.read_csv(teams_path, dtype=str)

    team_games = _team_games(teams_raw, manifest.created_at)
    historical = [record.relative_path for record in manifest.files
                  if record.relative_path.startswith("historical_schedule_")
                  and record.relative_path.endswith(".csv")]
    if historical:
        from src.features.snapshot_inputs import read_snapshot_csv

        schedules = pd.concat([
            read_snapshot_csv(root, snapshot_id, (name,)) for name in historical
        ], ignore_index=True)
        team_games = _bind_historical_schedule(team_games, schedules)
    identity = resolve_player_game_ids(players_raw, team_games)
    player_games = _player_games(identity.player_games, manifest.created_at)
    games, invalid_games = _games(team_games, manifest.created_at)

    valid_game_ids = set(games["GAME_ID"].astype(str))
    team_games = team_games.loc[team_games["GAME_ID"].isin(valid_game_ids)].copy()
    player_games = player_games.loc[player_games["GAME_ID"].isin(valid_game_ids)].copy()
    missing_count = _missing_player_team_games(player_games, team_games)

    official_roster = _captured_table(root, snapshot_id, "roster_membership",
                                     ("roster_membership.csv", "rosters.csv"))
    official_eligibility = _captured_table(root, snapshot_id, "player_game_eligibility",
                                          ("player_game_eligibility.csv",))
    roster = (official_roster if official_roster is not None
              else _observed_roster_membership(player_games, manifest.created_at))
    eligibility = (official_eligibility if official_eligibility is not None
                   else _observed_eligibility(player_games, manifest.created_at))
    tables = {
        "games": games,
        "team_games": team_games,
        "player_games": player_games,
        "roster_membership": roster,
        "player_game_eligibility": eligibility,
    }
    disabled_optional = {}
    for optional in (
        "player_status_snapshots",
        "lineup_snapshots",
        "odds_snapshots",
        "player_bios",
    ):
        try:
            captured = _captured_table(root, snapshot_id, optional, (f"{optional}.csv",))
            if captured is not None:
                validate_canonical_table(captured, optional, snapshot_created_at=manifest.created_at)
            tables[optional] = captured if captured is not None else empty_canonical_table(optional)
            if captured is None:
                disabled_optional[optional] = "No captured source table"
        except ContractError as exc:
            tables[optional] = empty_canonical_table(optional)
            disabled_optional[optional] = str(exc)
    for name, frame in tables.items():
        validate_canonical_table(
            frame, name, snapshot_created_at=manifest.created_at
        )

    report = CanonicalizationReport(
        snapshot_id=str(snapshot_id),
        games=len(games),
        team_games=len(team_games),
        player_games=len(player_games),
        invalid_games=invalid_games,
        missing_player_team_game=missing_count,
        quarantined_player_rows=len(identity.quarantine),
        remapped_game_ids=identity.report.remapped_game_ids,
        observed_roster_only=official_roster is None,
        observed_eligibility_only=official_eligibility is None,
        disabled_optional_sources=disabled_optional,
    )
    stage = Path(tempfile.mkdtemp(prefix=".canonical_", dir=root))
    try:
        for name, frame in tables.items():
            frame.to_parquet(stage / f"{name}.parquet", index=False)
        identity.quarantine.to_parquet(
            stage / "identity_quarantine.parquet", index=False
        )
        identity.mappings.to_parquet(
            stage / "identity_mappings.parquet", index=False
        )
        canonical_manifest = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "transformation_version": TRANSFORMATION_VERSION,
            "transformation_sha256": _payload_hash({
                name: _content_hash(Path(__file__).parents[1] / name)
                for name in ("data/canonicalize.py", "data/identity.py",
                             "contracts/canonical_data.py")
            }),
            "source_manifest_sha256": _payload_hash(manifest.to_dict()),
            "source_snapshot_id": str(snapshot_id),
            "source_snapshot_created_at": manifest.created_at,
            "report": report.to_dict(),
            "files": {
                path.name: {"sha256": _content_hash(path),
                            "size_bytes": path.stat().st_size}
                for path in sorted(stage.glob("*.parquet"))
            },
        }
        canonical_manifest["content_id"] = _payload_hash(canonical_manifest)
        (stage / "manifest.json").write_text(
            json.dumps(
                canonical_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def load_canonical_table(
    data_dir: str | Path,
    snapshot_id: str,
    table: str,
) -> pd.DataFrame:
    """Read and validate a canonical table tied to ``snapshot_id``."""

    root = Path(data_dir).resolve()
    manifest = validate_source_snapshot(root, snapshot_id)
    get_canonical_contract(table)
    directory = root / "canonical" / f"snapshot={snapshot_id}"
    metadata_path = directory / "manifest.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError("Canonical manifest is missing or unreadable") from exc
    if not isinstance(metadata, dict):
        raise ContractError("Canonical manifest must be an object")
    content_id = metadata.pop("content_id", None)
    if content_id != _payload_hash(metadata):
        raise ContractError("Canonical manifest content hash mismatch; rebuild required")
    if (metadata.get("source_snapshot_id") != str(snapshot_id)
            or metadata.get("source_manifest_sha256") != _payload_hash(manifest.to_dict())
            or metadata.get("schema_version") != CANONICAL_SCHEMA_VERSION
            or metadata.get("transformation_version") != TRANSFORMATION_VERSION):
        raise ContractError("Canonical manifest source or transformation mismatch")
    path = directory / f"{table}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical table does not exist: {path}")
    expected = metadata.get("files", {}).get(path.name, {})
    payload = path.read_bytes()
    if (expected.get("size_bytes") != len(payload)
            or expected.get("sha256") != hashlib.sha256(payload).hexdigest()):
        raise ContractError(f"Canonical table checksum mismatch: {table}")
    frame = pd.read_parquet(BytesIO(payload))
    validate_canonical_table(
        frame, table, snapshot_created_at=manifest.created_at
    )
    return frame


def _captured_table(root, snapshot_id, table, names):
    path = find_snapshot_file(root, snapshot_id, names)
    if path is None:
        return None
    frame = pd.read_csv(path, dtype=str)
    contract = get_canonical_contract(table)
    # Availability is never supplied from an outcome date or inferred history.
    require_columns(frame, ("AVAILABLE_AT", "SOURCE"), source=table)
    for value in frame.AVAILABLE_AT:
        parse_aware_datetime(value, field=f"{table}.AVAILABLE_AT")
    if "EVENT_TIME" not in frame:
        frame["EVENT_TIME"] = frame["AVAILABLE_AT"]
    for label in ("ACTIVE", "APPEARED", "STARTED", "DNP_REASON"):
        if table == "player_game_eligibility" and label not in frame:
            frame[label] = pd.NA
    for column in contract.id_columns:
        if column in frame:
            frame[column] = normalize_id_series(frame[column], field=column)
    if table == "roster_membership":
        require_columns(frame, ("COVERAGE_STATUS",), source=table)
        if not frame.COVERAGE_STATUS.eq("official").all():
            raise ContractError("Captured roster cannot relabel appearance-derived membership")
    return frame


def _team_games(raw: pd.DataFrame, available_at: str) -> pd.DataFrame:
    require_columns(raw, ("GAME_ID", "GAME_DATE", "TEAM_ID"), source="nba_games.csv")
    frame = raw.copy()
    for column in ("GAME_ID", "TEAM_ID"):
        frame[column] = normalize_id_series(frame[column], field=column)
    frame["GAME_DATE"] = _dates(frame["GAME_DATE"], "nba_games.csv.GAME_DATE")

    opponent_by_key: dict[tuple[str, str], str] = {}
    for game_id, group in frame.groupby("GAME_ID", sort=False):
        teams = tuple(group["TEAM_ID"].drop_duplicates())
        if len(teams) == 2:
            opponent_by_key[(game_id, teams[0])] = teams[1]
            opponent_by_key[(game_id, teams[1])] = teams[0]
    frame["OPPONENT_ID"] = [
        opponent_by_key.get((game_id, team_id), "")
        for game_id, team_id in zip(frame["GAME_ID"], frame["TEAM_ID"])
    ]
    if "HOME_FLAG" in frame:
        frame["HOME_FLAG"] = pd.to_numeric(frame["HOME_FLAG"], errors="coerce")
    elif "MATCHUP" in frame:
        frame["HOME_FLAG"] = (
            frame["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(int)
        )
    else:
        frame["HOME_FLAG"] = frame.groupby("GAME_ID", sort=False).cumcount().eq(0).astype(int)
    frame["EVENT_TIME"] = frame["GAME_DATE"]
    frame["AVAILABLE_AT"] = available_at
    frame["SOURCE"] = "core/nba_games.csv"
    return frame


def _bind_historical_schedule(team_games, schedule):
    """Repair venue identity only from an exact official NBA fixture match.

    NBA's ten-digit IDs may lose leading zeroes in CSV exports. This explicit
    crosswalk preserves the original ID and requires matching date and teams.
    Captures made now are outcome reconciliation, never historical PIT inputs.
    """
    require_columns(schedule, ("GAME_ID", "GAME_DATE", "HOME_TEAM_ID",
                              "AWAY_TEAM_ID", "SOURCE", "AVAILABLE_AT",
                              "SCHEDULED_TIP"), source="historical schedule")
    schedule = schedule.copy()
    schedule["__KEY"] = schedule.GAME_ID.map(_nba_game_key)
    if schedule.__KEY.duplicated().any():
        raise ContractError("Ambiguous historical schedule identity")
    lookup = schedule.set_index("__KEY")
    result = team_games.copy()
    for game_id, group in result.groupby("GAME_ID", sort=False):
        key = _nba_game_key(game_id)
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        if (set(group.TEAM_ID) != {str(row.HOME_TEAM_ID), str(row.AWAY_TEAM_ID)}
                or set(group.GAME_DATE) != {str(row.GAME_DATE)}):
            raise ContractError(f"Historical schedule conflicts with team game {game_id}")
        result.loc[group.index, "HOME_FLAG"] = group.TEAM_ID.eq(row.HOME_TEAM_ID).astype(int)
        result.loc[group.index, "SCHEDULED_TIP"] = row.SCHEDULED_TIP
        result.loc[group.index, "SCHEDULE_SOURCE_GAME_ID"] = row.GAME_ID
        result.loc[group.index, "SCHEDULE_SOURCE"] = row.SOURCE
        result.loc[group.index, "SCHEDULE_AVAILABLE_AT"] = row.AVAILABLE_AT
    return result


def _nba_game_key(value):
    value = str(value)
    # Restrict numeric equivalence to the documented NBA ID width.
    if value.isdigit() and 8 <= len(value) <= 10:
        return value.zfill(10)
    return value


def _player_games(raw: pd.DataFrame, available_at: str) -> pd.DataFrame:
    require_columns(
        raw,
        ("GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM_ID"),
        source="nba_players.csv",
    )
    frame = raw.copy()
    for column in ("GAME_ID", "PLAYER_ID", "TEAM_ID"):
        frame[column] = normalize_id_series(frame[column], field=column)
    frame["GAME_DATE"] = _dates(frame["GAME_DATE"], "nba_players.csv.GAME_DATE")
    frame["EVENT_TIME"] = frame["GAME_DATE"]
    frame["AVAILABLE_AT"] = available_at
    frame["SOURCE"] = "core/nba_players.csv"
    return frame


def _games(team_games: pd.DataFrame, available_at: str) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    invalid = 0
    for game_id, group in team_games.groupby("GAME_ID", sort=False):
        teams = tuple(group["TEAM_ID"].drop_duplicates())
        dates = tuple(group["GAME_DATE"].drop_duplicates())
        if len(group) != 2 or len(teams) != 2 or len(dates) != 1:
            invalid += 1
            continue
        home = group.loc[group["HOME_FLAG"].eq(1), "TEAM_ID"]
        if len(home) != 1:
            invalid += 1
            continue
        home_id = str(home.iloc[0])
        away_id = next(str(team_id) for team_id in teams if str(team_id) != home_id)
        game_date = str(dates[0])
        scheduled_tip = group.get(
            "SCHEDULED_TIP", pd.Series(index=group.index, dtype=object)
        ).dropna()
        tip = (
            str(scheduled_tip.iloc[0])
            if not scheduled_tip.empty
            else pd.Timestamp(game_date, tz="UTC").isoformat()
        )
        rows.append(
            {
                "GAME_ID": str(game_id),
                "GAME_DATE": game_date,
                "SCHEDULED_TIP": tip,
                "HOME_TEAM_ID": home_id,
                "AWAY_TEAM_ID": away_id,
                "STATUS": "final",
                "EVENT_TIME": game_date,
                "AVAILABLE_AT": available_at,
                "SOURCE": "core/nba_games.csv",
            }
        )
    columns = (
        "GAME_ID", "GAME_DATE", "SCHEDULED_TIP", "HOME_TEAM_ID",
        "AWAY_TEAM_ID", "STATUS", "EVENT_TIME", "AVAILABLE_AT", "SOURCE",
    )
    return pd.DataFrame(rows, columns=columns), invalid


def _observed_roster_membership(
    player_games: pd.DataFrame,
    available_at: str,
) -> pd.DataFrame:
    rows = []
    for (player_id, team_id), group in player_games.groupby(
        ["PLAYER_ID", "TEAM_ID"], sort=False
    ):
        dates = pd.to_datetime(group["GAME_DATE"], errors="raise")
        rows.append(
            {
                "PLAYER_ID": player_id,
                "TEAM_ID": team_id,
                "START_DATE": dates.min().date().isoformat(),
                "END_DATE": dates.max().date().isoformat(),
                "EVENT_TIME": dates.min().date().isoformat(),
                "AVAILABLE_AT": available_at,
                "SOURCE": "derived/player_appearances",
                "COVERAGE_STATUS": "observed_appearances_only",
            }
        )
    return pd.DataFrame(rows, columns=empty_canonical_table("roster_membership").columns)


def _observed_eligibility(
    player_games: pd.DataFrame,
    available_at: str,
) -> pd.DataFrame:
    result = player_games[["GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM_ID"]].copy()
    result["ROSTER_ELIGIBLE"] = 1
    result["ACTIVE"] = 1
    result["APPEARED"] = 1
    if "START_POSITION" in player_games:
        result["STARTED"] = player_games["START_POSITION"].fillna("").astype(str).ne("").astype(int)
    else:
        result["STARTED"] = pd.NA
    result["DNP_REASON"] = pd.NA
    result["EVENT_TIME"] = result["GAME_DATE"]
    result["AVAILABLE_AT"] = available_at
    result["SOURCE"] = "derived/player_appearances"
    return result


def _missing_player_team_games(
    player_games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> int:
    player_keys = player_games[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    team_keys = team_games[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    joined = player_keys.merge(
        team_keys, on=["GAME_ID", "TEAM_ID"], how="left", indicator=True
    )
    return int(joined["_merge"].eq("left_only").sum())


def _dates(values: pd.Series, field: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ContractError(f"{field} contains malformed dates")
    return parsed.dt.strftime("%Y-%m-%d")


__all__ = [
    "CanonicalizationReport",
    "canonicalize_snapshot",
    "load_canonical_table",
]
