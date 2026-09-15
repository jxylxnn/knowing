"""Native NBA adapters that preserve original responses and prospective times."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from curl_cffi import requests as curl_requests
import pandas as pd
import requests

from src.data.snapshots import create_source_snapshot


NBA_SCHEDULE_URL = (
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
)
NBA_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
}


def capture_nba_schedule(
    data_dir,
    *,
    season,
    request_timeout=30,
    max_attempts=3,
    retry_delay=2.0,
):
    payload, raw, source = _download_schedule(
        timeout=request_timeout,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )
    observed = datetime.now(timezone.utc)
    frame = _schedule_frame(payload, season=season)
    columns = [
        "GAME_ID",
        "GAME_DATE",
        "SCHEDULED_TIP",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "STATUS",
    ]
    if set(columns) - set(frame) or frame.empty:
        raise ValueError("Official schedule has no usable games/schema")
    frame = frame[columns].copy()
    tips = pd.to_datetime(frame.SCHEDULED_TIP, utc=True, errors="coerce")
    known = (
        tips.notna()
        & pd.to_numeric(frame.HOME_TEAM_ID, errors="coerce").gt(0)
        & pd.to_numeric(frame.AWAY_TEAM_ID, errors="coerce").gt(0)
    )
    quarantined = frame.loc[~known].copy()
    frame = frame.loc[known].copy()
    if frame.empty:
        raise ValueError("Official schedule contains only unresolved fixtures")
    frame["SCHEDULED_TIP"] = tips.loc[known].map(lambda value: value.isoformat())
    frame["GAME_DATE"] = pd.to_datetime(
        frame.GAME_DATE, errors="raise"
    ).dt.strftime("%Y-%m-%d")
    frame["AVAILABLE_AT"] = observed.isoformat()
    frame["SOURCE"] = source
    frame["SCHEDULE_VERSION"] = hashlib.sha256(raw).hexdigest()
    return _publish(
        data_dir,
        "schedule.csv",
        frame,
        {"schedule_response.json": raw},
        [{"url": source, "received_at": observed.isoformat()}],
        quarantined,
    )


def _download_schedule(*, timeout, max_attempts, retry_delay):
    request_timeout = _request_timeout(timeout)
    attempts = _max_attempts(max_attempts)
    delay = _retry_delay(retry_delay)

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = curl_requests.get(
                NBA_SCHEDULE_URL,
                headers=NBA_REQUEST_HEADERS,
                impersonate="chrome",
                timeout=request_timeout,
            )
            response.raise_for_status()
            _require_nba_url(response.url)
            raw = response.content
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Official NBA schedule response is not an object")
            return payload, raw, response.url
        except (curl_requests.RequestsError, json.JSONDecodeError) as exc:
            last_error = exc
            if not _retryable(exc) or attempt == attempts:
                break
            _sleep_before_retry(attempt, delay)

    message = (
        f"Official NBA schedule download failed after {attempt} attempt(s) "
        f"from {NBA_SCHEDULE_URL}: {type(last_error).__name__}: {last_error}"
    )
    raise RuntimeError(message) from last_error


def _schedule_frame(payload, *, season):
    league_schedule = payload.get("leagueSchedule")
    if not isinstance(league_schedule, dict):
        raise ValueError("Official schedule is missing leagueSchedule")
    captured_season = str(league_schedule.get("seasonYear", "")).strip()
    if captured_season != str(season).strip():
        raise ValueError(
            f"Official schedule is for {captured_season or 'an unknown season'}, "
            f"not {season}"
        )

    rows = []
    game_dates = league_schedule.get("gameDates", [])
    if not isinstance(game_dates, list):
        raise ValueError("Official schedule gameDates is malformed")
    for date_group in game_dates:
        if not isinstance(date_group, dict):
            continue
        games = date_group.get("games", [])
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, dict):
                continue
            required = {
                "gameId",
                "gameDateTimeUTC",
                "gameStatus",
                "homeTeam",
                "awayTeam",
            }
            missing = sorted(required - set(game))
            if missing:
                raise ValueError(
                    "Official schedule game schema is missing: "
                    + ", ".join(missing)
                )
            home_team = game["homeTeam"]
            away_team = game["awayTeam"]
            if not isinstance(home_team, dict) or not isinstance(away_team, dict):
                raise ValueError("Official schedule team schema is malformed")
            rows.append({
                "GAME_ID": game.get("gameId"),
                "GAME_DATE": game.get("gameDate") or date_group.get("gameDate"),
                "SCHEDULED_TIP": game.get("gameDateTimeUTC"),
                "HOME_TEAM_ID": home_team.get("teamId"),
                "AWAY_TEAM_ID": away_team.get("teamId"),
                "STATUS": game.get("gameStatus"),
            })
    return pd.DataFrame(rows)


def capture_nba_rosters(
    data_dir,
    *,
    season,
    team_ids=None,
    request_timeout=30,
    max_attempts=3,
    retry_delay=2.0,
):
    from nba_api.stats.static import teams

    identities = (
        list(team_ids)
        if team_ids
        else [team["id"] for team in teams.get_teams()]
    )
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Roster capture requires distinct team identities")
    frames, raw, receipts = [], {}, []
    for index, team in enumerate(identities):
        if index:
            time.sleep(1.2)
        endpoint = _download_roster(
            team=team,
            season=season,
            timeout=request_timeout,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
        observed = datetime.now(timezone.utc)
        source = endpoint.nba_response.get_url()
        _require_nba_url(source)
        payload = endpoint.nba_response.get_response().encode("utf-8")
        roster = endpoint.common_team_roster.get_data_frame().rename(
            columns={"TeamID": "TEAM_ID"}
        )
        if roster.empty or not {"PLAYER_ID", "TEAM_ID"}.issubset(roster):
            raise ValueError(f"Official roster is empty or malformed for team {team}")
        if not pd.to_numeric(roster.TEAM_ID, errors="raise").eq(int(team)).all():
            raise ValueError("Official roster response belongs to another team")
        roster = roster[["PLAYER_ID", "TEAM_ID"]].copy()
        roster["START_DATE"] = (
            observed.astimezone(ZoneInfo("America/New_York"))
            .date()
            .isoformat()
        )
        roster["END_DATE"] = None
        roster["AVAILABLE_AT"] = observed.isoformat()
        roster["EVENT_TIME"] = observed.isoformat()
        roster["SOURCE"] = source
        roster["COVERAGE_STATUS"] = "official"
        frames.append(roster)
        raw[f"roster_response_{team}.json"] = payload
        receipts.append({
            "url": source,
            "team_id": team,
            "received_at": observed.isoformat(),
        })
    frame = pd.concat(frames, ignore_index=True)
    if frame.PLAYER_ID.duplicated().any():
        raise ValueError("Official responses contain conflicting team memberships")
    return _publish(
        data_dir,
        "roster_membership.csv",
        frame,
        raw,
        receipts,
        pd.DataFrame(),
    )


def _download_roster(*, team, season, timeout, max_attempts, retry_delay):
    from nba_api.stats.endpoints.commonteamroster import CommonTeamRoster

    request_timeout = _request_timeout(timeout)
    attempts = _max_attempts(max_attempts)
    delay = _retry_delay(retry_delay)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return CommonTeamRoster(
                team_id=team,
                season=season,
                timeout=request_timeout,
            )
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            if not _retryable(exc) or attempt == attempts:
                break
            _sleep_before_retry(attempt, delay)
    message = (
        f"Official NBA roster download failed for team {team} after "
        f"{attempt} attempt(s): {type(last_error).__name__}: {last_error}"
    )
    raise RuntimeError(message) from last_error


def _request_timeout(read_timeout):
    read_timeout = float(read_timeout)
    if read_timeout <= 0:
        raise ValueError("NBA request timeout must be positive")
    return min(10.0, read_timeout), read_timeout


def _max_attempts(value):
    attempts = int(value)
    if attempts < 1:
        raise ValueError("NBA max attempts must be at least 1")
    return attempts


def _retry_delay(value):
    delay = float(value)
    if delay < 0:
        raise ValueError("NBA retry delay cannot be negative")
    return delay


def _sleep_before_retry(attempt, delay):
    if delay:
        time.sleep(min(15.0, delay * (2 ** (attempt - 1))))


def _retryable(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return True


def _require_nba_url(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "nba.com" or host.endswith(".nba.com")
    ):
        raise ValueError("Official NBA capture redirected to a non-NBA source")


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
