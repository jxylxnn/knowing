"""Append-only evidence for consecutive strict shadow slates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from src.utils.file_lock import exclusive_file_lock


@dataclass(frozen=True)
class ShadowSlate:
    slate_date: str
    bundle_id: str
    source_snapshot_id: str
    horizons: tuple[str, ...]
    games_expected: int
    games_completed: int
    forecasts_persisted: int
    strict: bool
    contract_valid: bool
    replay_parity_valid: bool
    completed_at: str
    errors: tuple[str, ...] = ()
    game_ids: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    forecast_digests: tuple[str, ...] = ()

    @property
    def successful(self) -> bool:
        return (
            self.strict
            and self.contract_valid
            and self.replay_parity_valid
            and self.games_expected > 0
            and self.games_completed == self.games_expected
            and self.forecasts_persisted > 0
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["successful"] = self.successful
        return payload


class ShadowSlateStore:
    """Store one immutable result per date and calculate the current streak."""

    def __init__(self, path: str | Path = "models/shadow_slates.json") -> None:
        self.path = Path(path)

    def load(self) -> tuple[ShadowSlate, ...]:
        if not self.path.is_file():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load shadow-slate evidence: {self.path}") from exc
        if not isinstance(payload, list):
            raise ValueError("Shadow-slate evidence must be a JSON list")
        rows = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("Shadow-slate record must be an object")
            clean = {key: value for key, value in item.items() if key != "successful"}
            clean["horizons"] = tuple(clean.get("horizons", ()))
            clean["errors"] = tuple(clean.get("errors", ()))
            for field in ("game_ids", "request_ids", "forecast_digests"):
                clean[field] = tuple(clean.get(field, ()))
            rows.append(ShadowSlate(**clean))
        return tuple(sorted(rows, key=lambda row: row.slate_date))

    def record(self, slate: ShadowSlate) -> tuple[ShadowSlate, ...]:
        datetime.strptime(slate.slate_date, "%Y-%m-%d")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(self.path.with_suffix(".lock")):
            records = list(self.load())
            if any(row.slate_date == slate.slate_date and row.bundle_id == slate.bundle_id for row in records):
                raise FileExistsError(
                    f"Shadow slate {slate.slate_date} is immutable and already recorded"
                )
            records.append(slate)
            records.sort(key=lambda row: (row.slate_date, row.bundle_id))
            self._write([row.to_dict() for row in records])
            return tuple(records)

    def consecutive_successes(self, *, through: str | date | None = None,
                              bundle_id: str | None = None) -> int:
        records = list(self.load())
        if bundle_id is not None:
            records = [row for row in records if row.bundle_id == bundle_id]
        elif len({row.bundle_id for row in records}) > 1:
            raise ValueError("Select one bundle to count shadow successes")
        if through is not None:
            cutoff = through.isoformat() if isinstance(through, date) else str(through)
            records = [row for row in records if row.slate_date <= cutoff]
        streak = 0
        previous: date | None = None
        # "Consecutive slates" means consecutive recorded game slates, not
        # calendar days (the NBA can have dark days).
        for row in reversed(records):
            current = datetime.strptime(row.slate_date, "%Y-%m-%d").date()
            if previous is not None and current >= previous:
                raise ValueError("Shadow slates are not strictly ordered")
            if not row.successful:
                break
            streak += 1
            previous = current
        return streak

    def promotion_evidence(self, *, bundle_id=None, horizons=None, expected_calendar=None,
                           run_store=None, data_dir=None):
        """Only a named bundle over the complete expected calendar can qualify.

        Recorded counts without request/digest membership are diagnostic logs,
        not sufficient promotion evidence. Missing expected slates reset the
        streak, even when the observed logs omit those days entirely.
        """
        if not bundle_id or not horizons or not expected_calendar or run_store is None or data_dir is None:
            return {"eligible": False, "consecutive_successes": 0,
                    "reason": "Candidate, horizons, expected calendar, and persisted request evidence are required"}
        records = {row.slate_date: row for row in self.load() if row.bundle_id == bundle_id}
        streak = 0
        qualified = []
        for day in sorted(expected_calendar, reverse=True):
            row = records.get(day)
            expected_games = set(map(str, expected_calendar[day]))
            expected_requests = len(expected_games) * len(set(horizons))
            if (row is None or not row.successful or set(row.horizons) != set(horizons)
                    or not expected_games or row.games_expected != len(expected_games)
                    or set(row.game_ids) != expected_games
                    or len(row.request_ids) != expected_requests
                    or len(set(row.request_ids)) != expected_requests
                    or len(row.forecast_digests) != expected_requests
                    or not all(len(value) == 64 for value in row.forecast_digests)):
                break
            from src.features.snapshot_inputs import assert_snapshot_game, load_official_roster
            from src.operations.ledger import _frame_digest

            slots = set()
            valid = True
            for request_id, digest in zip(row.request_ids, row.forecast_digests):
                run = run_store.load_request_id(request_id)
                if (run is None or run.request.model_bundle_id != bundle_id
                        or run.request.scenario != "official"
                        or run.request.game_date.isoformat() != day
                        or _frame_digest(run.forecasts) != digest):
                    valid = False
                    break
                assert_snapshot_game(data_dir, run.request)
                load_official_roster(data_dir, run.request)
                if (not run.forecasts.SCENARIO.eq("official").all()
                        or not run.forecasts.DISTRIBUTION_KIND.eq("empirical_full_game_v2").all()
                        or run.forecasts.empty):
                    valid = False
                    break
                slots.add((run.request.game_id, run.request.horizon))
            if not valid or slots != {(game, horizon) for game in expected_games for horizon in horizons}:
                break
            streak += 1
            qualified.append(row.to_dict())
        return {"required_consecutive_successes": 7, "consecutive_successes": streak,
                "eligible": streak >= 7, "bundle_id": bundle_id,
                "horizons": sorted(horizons), "expected_calendar": expected_calendar,
                "qualified_slates": qualified}

    def _write(self, payload: Iterable[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".shadow_slates_", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(list(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def make_shadow_slate(
    *,
    slate_date: str,
    bundle_id: str,
    source_snapshot_id: str,
    horizons: Iterable[str],
    games_expected: int,
    games_completed: int,
    forecasts_persisted: int,
    strict: bool,
    contract_valid: bool,
    replay_parity_valid: bool,
    errors: Iterable[str] = (),
    game_ids: Iterable[str] = (),
    request_ids: Iterable[str] = (),
    forecast_digests: Iterable[str] = (),
) -> ShadowSlate:
    """Convenience constructor that timestamps completion in UTC."""

    return ShadowSlate(
        slate_date=slate_date,
        bundle_id=bundle_id,
        source_snapshot_id=source_snapshot_id,
        horizons=tuple(horizons),
        games_expected=int(games_expected),
        games_completed=int(games_completed),
        forecasts_persisted=int(forecasts_persisted),
        strict=bool(strict),
        contract_valid=bool(contract_valid),
        replay_parity_valid=bool(replay_parity_valid),
        completed_at=datetime.now(timezone.utc).isoformat(),
        errors=tuple(str(error) for error in errors),
        game_ids=tuple(map(str, game_ids)), request_ids=tuple(map(str, request_ids)),
        forecast_digests=tuple(map(str, forecast_digests)),
    )
