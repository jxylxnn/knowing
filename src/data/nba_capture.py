"""Native NBA adapters that preserve original responses and prospective times."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.snapshots import create_source_snapshot


def capture_nba_schedule(data_dir, *, season):
    from nba_api.stats.endpoints.scheduleleaguev2 import ScheduleLeagueV2

    endpoint = ScheduleLeagueV2(season=season, timeout=15)
    observed = datetime.now(timezone.utc)
    raw = endpoint.nba_response.get_response().encode("utf-8")
    source = endpoint.nba_response.get_url()
    frame = endpoint.season_games.get_data_frame().rename(columns={
        "gameId": "GAME_ID", "gameDate": "GAME_DATE", "gameDateTimeUTC": "SCHEDULED_TIP",
        "homeTeam_teamId": "HOME_TEAM_ID", "awayTeam_teamId": "AWAY_TEAM_ID",
        "gameStatus": "STATUS",
    })
    columns = ["GAME_ID", "GAME_DATE", "SCHEDULED_TIP", "HOME_TEAM_ID", "AWAY_TEAM_ID", "STATUS"]
    if set(columns) - set(frame) or frame.empty:
        raise ValueError("Official schedule has no usable games/schema")
    frame = frame[columns].copy()
    tips = pd.to_datetime(frame.SCHEDULED_TIP, utc=True, errors="coerce")
    known = (tips.notna() & pd.to_numeric(frame.HOME_TEAM_ID, errors="coerce").gt(0)
             & pd.to_numeric(frame.AWAY_TEAM_ID, errors="coerce").gt(0))
    quarantined = frame.loc[~known].copy()
    frame = frame.loc[known].copy()
    if frame.empty:
        raise ValueError("Official schedule contains only unresolved fixtures")
    frame["SCHEDULED_TIP"] = tips.loc[known].map(lambda value: value.isoformat())
    frame["GAME_DATE"] = pd.to_datetime(frame.GAME_DATE, errors="raise").dt.strftime("%Y-%m-%d")
    frame["AVAILABLE_AT"] = observed.isoformat()
    frame["SOURCE"] = source
    frame["SCHEDULE_VERSION"] = hashlib.sha256(raw).hexdigest()
    return _publish(data_dir, "schedule.csv", frame,
                    {"schedule_response.json": raw},
                    [{"url": source, "received_at": observed.isoformat()}], quarantined)


def capture_nba_rosters(data_dir, *, season, team_ids=None):
    from nba_api.stats.endpoints.commonteamroster import CommonTeamRoster
    from nba_api.stats.static import teams

    identities = list(team_ids) if team_ids else [team["id"] for team in teams.get_teams()]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Roster capture requires distinct team identities")
    frames, raw, receipts = [], {}, []
    for index, team in enumerate(identities):
        if index:
            time.sleep(1.2)
        endpoint = CommonTeamRoster(team_id=team, season=season, timeout=15)
        observed = datetime.now(timezone.utc)
        source = endpoint.nba_response.get_url()
        payload = endpoint.nba_response.get_response().encode("utf-8")
        roster = endpoint.common_team_roster.get_data_frame().rename(columns={"TeamID": "TEAM_ID"})
        if roster.empty or not {"PLAYER_ID", "TEAM_ID"}.issubset(roster):
            raise ValueError(f"Official roster is empty or malformed for team {team}")
        if not pd.to_numeric(roster.TEAM_ID, errors="raise").eq(int(team)).all():
            raise ValueError("Official roster response belongs to another team")
        roster = roster[["PLAYER_ID", "TEAM_ID"]].copy()
        roster["START_DATE"] = observed.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        roster["END_DATE"] = None
        roster["AVAILABLE_AT"] = observed.isoformat()
        roster["EVENT_TIME"] = observed.isoformat()
        roster["SOURCE"] = source
        roster["COVERAGE_STATUS"] = "official"
        frames.append(roster)
        raw[f"roster_response_{team}.json"] = payload
        receipts.append({"url": source, "team_id": team, "received_at": observed.isoformat()})
    frame = pd.concat(frames, ignore_index=True)
    if frame.PLAYER_ID.duplicated().any():
        raise ValueError("Official responses contain conflicting team memberships")
    return _publish(data_dir, "roster_membership.csv", frame, raw, receipts, pd.DataFrame())


def _publish(data_dir, filename, frame, responses, receipts, quarantine):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native_capture_", dir=root) as temporary:
        stage = Path(temporary)
        frame.to_csv(stage / filename, index=False)
        quarantine.to_csv(stage / "quarantine.csv", index=False)
        for name, payload in responses.items():
            (stage / name).write_bytes(payload)
        receipt = {"schema_version": "native_nba_capture_v1", "receipts": receipts,
                   "rows": len(frame), "quarantined_rows": len(quarantine),
                   "raw_hashes": {name: hashlib.sha256(payload).hexdigest()
                                  for name, payload in responses.items()}}
        (stage / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        manifest = create_source_snapshot(
            stage, files=(filename, "receipt.json", "quarantine.csv", *responses), source="official"
        )
        publication = stage / "publication"
        publication.mkdir()
        (stage / "raw").rename(publication / "raw")
        (stage / "manifests").rename(publication / "manifests")
        destination = root / "official_captures" / manifest.snapshot_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        publication.rename(destination)
    return destination
