"""Verified snapshot inputs for scheduled requests, without working-file reads."""

from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.contracts.canonical_data import normalize_id_series, require_columns
from src.contracts.errors import ContractError
from src.contracts.forecast import ForecastRequest
from src.contracts.sources import parse_aware_datetime
from src.data.snapshots import snapshot_file_path, validate_source_snapshot


def read_snapshot_csv(data_dir, snapshot_id, names, *, cutoff=None):
    """Verify and parse the same bytes, retaining string identifiers."""
    manifest = validate_source_snapshot(
        data_dir, snapshot_id, forecast_cutoff=cutoff
    )
    records = [item for item in manifest.files if item.relative_path in names]
    if len(records) != 1:
        raise ContractError(f"Snapshot must contain exactly one of {tuple(names)}")
    record = records[0]
    payload = snapshot_file_path(data_dir, snapshot_id, record).read_bytes()
    if (len(payload) != record.size_bytes
            or hashlib.sha256(payload).hexdigest() != record.sha256):
        raise ContractError("Snapshot input changed while being read")
    frame = pd.read_csv(BytesIO(payload), dtype=str)
    if "AVAILABLE_AT" in frame:
        _require_available(frame, manifest.created_datetime, record.relative_path)
    for column in ("GAME_ID", "PLAYER_ID", "TEAM_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"):
        if column in frame:
            frame[column] = normalize_id_series(frame[column], field=column)
    return frame


def load_snapshot_schedule(data_dir, snapshot_id, *, cutoff=None):
    frame = read_snapshot_csv(
        data_dir, snapshot_id, ("schedule.csv", "nba_schedule.csv"), cutoff=cutoff
    )
    require_columns(frame, (
        "GAME_ID", "GAME_DATE", "SCHEDULED_TIP", "HOME_TEAM_ID", "AWAY_TEAM_ID",
    ), source="snapshot schedule")
    if frame["GAME_ID"].duplicated().any():
        raise ContractError("Snapshot schedule contains duplicate game identities")
    for value in frame["SCHEDULED_TIP"]:
        parse_aware_datetime(value, field="SCHEDULED_TIP")
    if cutoff is not None and "AVAILABLE_AT" in frame:
        _require_available(frame, cutoff, "schedule")
    return frame


def load_request_history(data_dir, request: ForecastRequest):
    """Read only prior-game outcomes from this request's verified snapshot."""
    history = read_snapshot_csv(
        data_dir, request.source_snapshot_id, ("nba_players.csv",),
        cutoff=request.forecast_cutoff,
    )
    require_columns(history, ("GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM_ID"),
                    source="snapshot history")
    if "AVAILABLE_AT" in history:
        _require_available(history, request.forecast_cutoff, "history")
    dates = pd.to_datetime(history["GAME_DATE"], errors="raise").dt.date
    return history.loc[
        (dates < request.game_date) & history["GAME_ID"].ne(request.game_id)
    ].copy()


def assert_snapshot_game(data_dir, request: ForecastRequest):
    """Bind the supplied game identity and tip to the captured schedule."""
    schedule = load_snapshot_schedule(
        data_dir, request.source_snapshot_id, cutoff=request.forecast_cutoff
    )
    rows = schedule.loc[schedule["GAME_ID"].eq(request.game_id)]
    if len(rows) != 1:
        raise ContractError("Requested game is absent from snapshot schedule")
    row = rows.iloc[0]
    if (str(row.HOME_TEAM_ID) != str(request.home_team_id)
            or str(row.AWAY_TEAM_ID) != str(request.away_team_id)
            or pd.Timestamp(row.GAME_DATE).date() != request.game_date
            or pd.Timestamp(row.SCHEDULED_TIP) != pd.Timestamp(request.scheduled_tip)
            or str(row.get("SCHEDULE_VERSION", "schedule_v1")) != request.schedule_version):
        raise ContractError("Requested game differs from snapshot schedule")


def load_official_roster(data_dir: str | Path, request: ForecastRequest):
    """Select timestamped membership, never inferred historical appearances.

    Membership intervals are inclusive dates. Conflicting overlapping team
    memberships fail instead of silently resolving a trade by row order.
    """
    frame = read_snapshot_csv(
        data_dir, request.source_snapshot_id, ("roster_membership.csv", "rosters.csv"),
        cutoff=request.forecast_cutoff,
    )
    if request.scenario == "official":
        manifest = validate_source_snapshot(data_dir, request.source_snapshot_id,
                                            forecast_cutoff=request.forecast_cutoff)
        record = next(item for item in manifest.files
                      if item.relative_path in ("roster_membership.csv", "rosters.csv"))
        fetched = parse_aware_datetime(record.fetched_at, field="roster.fetched_at")
        local = ZoneInfo("America/New_York")
        if fetched.astimezone(local).date() != request.forecast_cutoff.astimezone(local).date():
            raise ContractError("Official roster requires a current-day capture")
    require_columns(frame, (
        "PLAYER_ID", "TEAM_ID", "START_DATE", "END_DATE", "AVAILABLE_AT",
        "SOURCE", "COVERAGE_STATUS",
    ), source="official roster")
    _require_available(frame, request.forecast_cutoff, "roster")
    if (frame["SOURCE"].isna().any()
            or frame["SOURCE"].str.contains("derived|appearance", case=False).any()
            or not frame["COVERAGE_STATUS"].eq("official").all()):
        raise ContractError("Strict roster requires official membership evidence")
    start = pd.to_datetime(frame["START_DATE"], errors="raise")
    end = pd.to_datetime(frame["END_DATE"], errors="raise")
    if start.isna().any() or ((end.notna()) & (end < start)).any():
        raise ContractError("Invalid roster membership interval")
    current = frame.loc[(start <= pd.Timestamp(request.game_date))
                        & (end.isna() | (end >= pd.Timestamp(request.game_date)))].copy()
    if current["PLAYER_ID"].duplicated().any():
        raise ContractError("Conflicting roster membership at requested game")
    teams = {str(request.home_team_id), str(request.away_team_id)}
    current = current.loc[current["TEAM_ID"].isin(teams)]
    if set(current["TEAM_ID"]) != teams:
        raise ContractError("Official roster missing a scheduled team")
    # Only identity and provenance leave this reader; status/outcome columns
    # cannot accidentally become current-game forecast features.
    return current[["PLAYER_ID", "TEAM_ID", "SOURCE", "AVAILABLE_AT"]].reset_index(drop=True)


def _require_available(frame, cutoff, source):
    cutoff = parse_aware_datetime(cutoff, field="forecast_cutoff")
    for value in frame["AVAILABLE_AT"]:
        observed = parse_aware_datetime(value, field=f"{source}.AVAILABLE_AT")
        if observed > cutoff:
            raise ContractError(f"{source} contains data unavailable at cutoff")


def load_request_status(data_dir, request: ForecastRequest, roster):
    """Read captured status for the scheduled teams; unknown stays explicit."""
    manifest = validate_source_snapshot(data_dir, request.source_snapshot_id,
                                        forecast_cutoff=request.forecast_cutoff)
    if not any(record.relative_path == "player_status_snapshots.csv" for record in manifest.files):
        return roster[["PLAYER_ID", "TEAM_ID"]].assign(STATUS="UNKNOWN")
    frame = read_snapshot_csv(data_dir, request.source_snapshot_id,
                              ("player_status_snapshots.csv",), cutoff=request.forecast_cutoff)
    require_columns(frame, ("PLAYER_ID", "TEAM_ID", "AVAILABLE_AT"), source="status snapshots")
    _require_available(frame, request.forecast_cutoff, "status")
    if "STATUS" not in frame and "STATUS_NORMALIZED" in frame:
        frame = frame.rename(columns={"STATUS_NORMALIZED": "STATUS"})
    require_columns(frame, ("STATUS",), source="status snapshots")
    if "GAME_ID" in frame:
        frame = frame.loc[frame.GAME_ID.eq(request.game_id)]
    frame = frame.assign(__TIME=pd.to_datetime(frame.AVAILABLE_AT, utc=True))
    if frame.duplicated(["PLAYER_ID", "TEAM_ID", "__TIME"]).any():
        raise ContractError("Ambiguous same-time official status records")
    latest = frame.sort_values("__TIME").drop_duplicates(["PLAYER_ID", "TEAM_ID"], keep="last")
    result = roster[["PLAYER_ID", "TEAM_ID"]].merge(
        latest[["PLAYER_ID", "TEAM_ID", "STATUS"]], on=["PLAYER_ID", "TEAM_ID"],
        how="left", validate="one_to_one",
    )
    result["STATUS"] = result.STATUS.fillna("UNKNOWN")
    return result
