"""Focused tests for external release evidence: no network, no real data."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import shutil

import pandas as pd
import pytest

from src.contracts.forecast import ForecastRequest
from src.forecasting.distribution import GameDistribution
from src.operations.ledger import _frame_digest
from src.operations.release import (
    RELEASE_EVIDENCE_SCHEMA_VERSION,
    ReleaseEvidenceStore,
    ReleaseEvidenceVerification,
)
from src.operations.shadow import ShadowSlateStore, make_shadow_slate
from tests.test_models.test_v2_bundle import _bundle


DAYS = tuple(f"2026-10-{day:02d}" for day in range(1, 8))
HORIZONS = ("morning",)
GAMES = ("g1", "g2")
CALENDAR = {day: list(GAMES) for day in DAYS}


class _FakeRunStore:
    """Duck-typed ForecastRunStore that returns prebuilt official runs."""

    def __init__(self, runs):
        self.runs = dict(runs)

    def load_request_id(self, request_id):
        return self.runs.get(request_id)


class _DuplicateDateCalendar(dict):
    """A mapping that reports the same date twice, which JSON cannot express."""

    def items(self):
        yield ("2026-10-01", ["g1"])
        yield ("2026-10-01", ["g2"])


def _request_id(bundle_id, day, game_id, horizon):
    seed = f"{bundle_id}|{day}|{game_id}|{horizon}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _build_runs(bundle_id, calendar, horizons):
    runs = {}
    for day, games in calendar.items():
        for game_id in games:
            for horizon in horizons:
                request_id = _request_id(bundle_id, day, game_id, horizon)
                request = ForecastRequest(
                    game_id=game_id,
                    game_date=day,
                    scheduled_tip=datetime.fromisoformat(f"{day}T23:00:00+00:00"),
                    home_team_id=1,
                    away_team_id=2,
                    forecast_cutoff=datetime.fromisoformat(f"{day}T12:00:00+00:00"),
                    horizon=horizon,
                    source_snapshot_id="snapshot-a",
                    model_bundle_id=bundle_id,
                )
                forecasts = pd.DataFrame(
                    {
                        "REQUEST_ID": [request_id],
                        "SCENARIO": ["official"],
                        "DISTRIBUTION_KIND": ["empirical_full_game_v2"],
                    }
                )
                runs[request_id] = GameDistribution(
                    request=request,
                    forecasts=forecasts,
                    samples=pd.DataFrame(),
                    seed=1,
                    simulations=1,
                )
    return runs


def _record_slate(store, bundle_id, day, games, horizons, runs):
    request_ids = [
        _request_id(bundle_id, day, game_id, horizon)
        for game_id in games
        for horizon in horizons
    ]
    store.record(
        make_shadow_slate(
            slate_date=day,
            bundle_id=bundle_id,
            source_snapshot_id="snapshot-a",
            horizons=horizons,
            games_expected=len(games),
            games_completed=len(games),
            forecasts_persisted=len(request_ids),
            strict=True,
            contract_valid=True,
            replay_parity_valid=True,
            game_ids=list(games),
            request_ids=request_ids,
            forecast_digests=[
                _frame_digest(runs[request_id].forecasts) for request_id in request_ids
            ],
        )
    )


def _record_calendar(store, bundle_id, calendar, horizons, runs, days=None):
    for day in days if days is not None else calendar:
        _record_slate(store, bundle_id, day, calendar[day], horizons, runs)


def _build(store, candidate_dir, shadow_store, run_store, data_dir, *,
           horizons=HORIZONS, calendar=CALENDAR):
    return store.build(
        candidate_dir=candidate_dir,
        shadow_store=shadow_store,
        run_store=run_store,
        data_dir=data_dir,
        horizons=horizons,
        expected_calendar=calendar,
    )


def _verify(store, path, candidate_dir, shadow_store, run_store, data_dir, *,
            horizons=HORIZONS, calendar=CALENDAR):
    return store.verify(
        path,
        candidate_dir=candidate_dir,
        shadow_store=shadow_store,
        run_store=run_store,
        data_dir=data_dir,
        horizons=horizons,
        expected_calendar=calendar,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def official_snapshot_stubs(monkeypatch):
    """Shadow evidence re-reads schedule/roster bindings from the data dir."""

    from src.features import snapshot_inputs

    monkeypatch.setattr(
        snapshot_inputs, "assert_snapshot_game", lambda data_dir, request: None
    )
    monkeypatch.setattr(
        snapshot_inputs, "load_official_roster", lambda data_dir, request: None
    )


@pytest.fixture
def candidate(tmp_path):
    """A sealed v2 candidate whose scorecards recompute as eligible."""

    return _bundle(tmp_path)


def test_build_is_deterministic_idempotent_and_bound_to_the_candidate(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)

    first = _build(store, candidate_dir, shadow, run_store, tmp_path)
    assert first.bundle_id == manifest.bundle_id
    assert first.path == store.path_for(manifest.bundle_id, first.evidence_id)
    assert first.path.parent.name == manifest.bundle_id
    assert len(first.evidence_id) == 64
    assert first.payload["schema_version"] == RELEASE_EVIDENCE_SCHEMA_VERSION
    assert first.payload["source_snapshot_id"] == "snapshot-a"
    assert first.payload["bundle_manifest_sha256"] == _sha256(
        candidate_dir / "bundle_manifest.json"
    )
    assert first.payload["horizons"] == ["morning"]
    assert first.payload["expected_calendar"] == CALENDAR
    assert first.payload == json.loads(first.path.read_text(encoding="utf-8"))
    published = _sha256(first.path)

    second = _build(store, candidate_dir, shadow, run_store, tmp_path)
    assert second.evidence_id == first.evidence_id
    assert second.path == first.path
    # An identical rebuild reuses the published bytes, including created_at.
    assert _sha256(first.path) == published
    assert first.payload == second.payload
    assert second.payload == json.loads(second.path.read_text(encoding="utf-8"))
    assert second.payload["created_at"] == first.payload["created_at"]
    assert second.payload["evidence_id"] == first.evidence_id
    assert second.eligible == first.eligible
    assert not list(first.path.parent.glob("*.tmp"))

    stored = json.loads(first.path.read_text(encoding="utf-8"))
    assert stored["evidence_id"] == first.evidence_id
    assert stored["bundle_id"] == manifest.bundle_id


def test_reused_evidence_reports_the_published_payload(
    tmp_path, candidate, official_snapshot_stubs
):
    """A rebuild returns the stored document, never a freshly stamped copy."""

    from src.operations import release as release_module

    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    first = _build(store, candidate_dir, shadow, run_store, tmp_path)

    # Rewrite the stored timestamp: created_at is outside the content address,
    # so the rebuild must reuse the file and report its document verbatim.
    stored = json.loads(first.path.read_text(encoding="utf-8"))
    stored["created_at"] = "2020-01-01T00:00:00+00:00"
    first.path.write_text(
        json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8"
    )

    rebuilt = _build(store, candidate_dir, shadow, run_store, tmp_path)
    assert rebuilt.path == first.path
    assert (
        rebuilt.evidence_id
        == first.evidence_id
        == release_module._evidence_id(rebuilt.payload)
    )
    assert rebuilt.eligible == first.eligible
    assert rebuilt.payload["created_at"] == "2020-01-01T00:00:00+00:00"
    assert rebuilt.payload == json.loads(rebuilt.path.read_text(encoding="utf-8"))


def test_build_rejects_conflicting_or_corrupt_published_evidence(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    conflicting = json.loads(record.path.read_text(encoding="utf-8"))
    conflicting["eligible"] = not conflicting["eligible"]
    record.path.write_text(json.dumps(conflicting), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the computed payload"):
        _build(store, candidate_dir, shadow, run_store, tmp_path)

    record.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        _build(store, candidate_dir, shadow, run_store, tmp_path)


def test_complete_seven_slate_evidence_is_eligible(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)

    record = _build(store, candidate_dir, shadow, run_store, tmp_path)
    assert record.eligible is True
    assert record.payload["eligible"] is True
    assert record.payload["failure_reasons"] == []
    assert record.payload["dirty_worktree"] is False
    assert record.payload["model_evidence"]["recomputed"] is True
    assert record.payload["model_evidence"]["eligible"] is True
    shadow_evidence = record.payload["shadow_evidence"]
    assert shadow_evidence["eligible"] is True
    assert shadow_evidence["consecutive_successes"] == 7
    assert shadow_evidence["required_consecutive_successes"] == 7
    assert len(shadow_evidence["qualified_slates"]) == 7

    verification = _verify(
        store, record.path, candidate_dir, shadow, run_store, tmp_path
    )
    assert isinstance(verification, ReleaseEvidenceVerification)
    assert verification.evidence_id == record.evidence_id
    assert verification.bundle_id == manifest.bundle_id
    assert verification.eligible is True
    assert verification.model_evidence_eligible is True
    assert verification.shadow_evidence_eligible is True
    assert verification.dirty_worktree is False
    assert verification.failure_reasons == ()


def test_complete_evidence_with_several_horizons_is_eligible(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    horizons = ("morning", "pregame_30m")
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, horizons)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, horizons, runs)
    run_store = _FakeRunStore(runs)

    record = _build(
        store, candidate_dir, shadow, run_store, tmp_path, horizons=horizons
    )
    assert record.payload["horizons"] == ["morning", "pregame_30m"]
    assert record.eligible is True
    assert record.payload["shadow_evidence"]["consecutive_successes"] == 7


def test_missing_or_incomplete_shadow_evidence_is_ineligible(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    run_store = _FakeRunStore(runs)

    empty = ShadowSlateStore(tmp_path / "empty_shadow.json")
    empty_record = _build(store, candidate_dir, empty, run_store, tmp_path)
    assert empty_record.eligible is False
    assert empty_record.payload["eligible"] is False
    # The static model evidence is recomputed and eligible; only the shadow
    # gate blocks the release.
    assert empty_record.payload["model_evidence"]["eligible"] is True
    assert empty_record.payload["shadow_evidence"]["eligible"] is False
    assert empty_record.payload["shadow_evidence"]["consecutive_successes"] == 0
    assert any(
        "shadow slate evidence is not eligible" in reason
        for reason in empty_record.payload["failure_reasons"]
    )

    # The most recent expected slate is missing, so the streak resets to zero.
    stale = ShadowSlateStore(tmp_path / "stale_shadow.json")
    _record_calendar(
        stale, manifest.bundle_id, CALENDAR, HORIZONS, runs, days=DAYS[:-1]
    )
    stale_record = _build(store, candidate_dir, stale, run_store, tmp_path)
    assert stale_record.eligible is False
    assert stale_record.payload["shadow_evidence"]["consecutive_successes"] == 0

    # Six consecutive qualified slates are not seven.
    partial = ShadowSlateStore(tmp_path / "partial_shadow.json")
    _record_calendar(
        partial, manifest.bundle_id, CALENDAR, HORIZONS, runs, days=DAYS[1:]
    )
    partial_record = _build(store, candidate_dir, partial, run_store, tmp_path)
    assert partial_record.eligible is False
    assert partial_record.payload["shadow_evidence"]["consecutive_successes"] == 6
    assert any(
        "6 of 7" in reason for reason in partial_record.payload["failure_reasons"]
    )


def test_dirty_worktree_blocks_release(tmp_path, official_snapshot_stubs):
    candidate_dir, manifest = _bundle(tmp_path, dirty=True)
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)

    record = _build(
        store, candidate_dir, shadow, _FakeRunStore(runs), tmp_path
    )
    assert record.payload["dirty_worktree"] is True
    assert record.payload["model_evidence"]["eligible"] is True
    assert record.payload["shadow_evidence"]["eligible"] is True
    assert record.eligible is False
    assert any(
        "dirty worktree" in reason for reason in record.payload["failure_reasons"]
    )


def test_stored_promotion_decision_is_recomputed_not_trusted(
    tmp_path, official_snapshot_stubs
):
    # This candidate seals promotion_decision.json as ineligible with empty
    # checks while its immutable scorecards still recompute as eligible.
    candidate_dir, manifest = _bundle(tmp_path, eligible=False)
    decision = json.loads(
        (candidate_dir / "promotion_decision.json").read_text(encoding="utf-8")
    )
    assert decision["eligible"] is False
    assert decision["checks"] == {}

    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)

    record = _build(
        store, candidate_dir, shadow, _FakeRunStore(runs), tmp_path
    )
    assert record.payload["model_evidence"]["recomputed"] is True
    assert record.payload["model_evidence"]["eligible"] is True
    assert record.payload["model_evidence"]["checks"]


def test_unrecomputable_model_evidence_is_ineligible(
    tmp_path, candidate, monkeypatch, official_snapshot_stubs
):
    from src.operations import release as release_module

    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)

    def unrecomputable(root):
        raise ValueError("candidate_scorecard.json must declare evidence schema")

    monkeypatch.setattr(
        release_module, "_validate_and_evaluate_promotion_evidence", unrecomputable
    )
    record = _build(
        store, candidate_dir, shadow, _FakeRunStore(runs), tmp_path
    )
    assert record.payload["model_evidence"]["recomputed"] is False
    assert record.payload["model_evidence"]["eligible"] is False
    assert record.eligible is False
    assert any(
        "could not be recomputed" in reason
        and "evidence schema" in reason
        for reason in record.payload["failure_reasons"]
    )


def test_verify_rejects_tampered_evidence(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    payload = json.loads(record.path.read_text(encoding="utf-8"))
    payload["shadow_evidence"]["consecutive_successes"] = 99
    payload["eligible"] = not payload["eligible"]
    record.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its evidence ID"):
        _verify(store, record.path, candidate_dir, shadow, run_store, tmp_path)


def test_verify_rejects_internally_consistent_forgery(
    tmp_path, candidate, official_snapshot_stubs
):
    """A forged payload with a recomputed ID still fails re-derivation."""

    from src.operations import release as release_module

    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    forged = json.loads(record.path.read_text(encoding="utf-8"))
    forged["eligible"] = not forged["eligible"]
    forged["failure_reasons"] = []
    forged["evidence_id"] = release_module._evidence_id(forged)
    forged_path = store.path_for(manifest.bundle_id, forged["evidence_id"])
    forged_path.write_text(json.dumps(forged, indent=2), encoding="utf-8")

    assert forged_path != record.path
    with pytest.raises(ValueError, match="stale or tampered"):
        _verify(store, forged_path, candidate_dir, shadow, run_store, tmp_path)


def test_verify_rejects_stale_shadow_evidence(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    # A persisted forecast no longer matches its recorded digest, so the slate
    # that qualified this release no longer qualifies.
    newest = _request_id(manifest.bundle_id, DAYS[-1], GAMES[0], HORIZONS[0])
    run_store.runs[newest].forecasts.loc[:, "SCENARIO"] = "degraded"
    with pytest.raises(ValueError, match="stale or tampered"):
        _verify(store, record.path, candidate_dir, shadow, run_store, tmp_path)

    run_store.runs[newest].forecasts.loc[:, "SCENARIO"] = "official"
    later = {"2026-10-08": list(GAMES)}
    with pytest.raises(ValueError, match="expected calendar is stale or tampered"):
        _verify(
            store,
            record.path,
            candidate_dir,
            shadow,
            run_store,
            tmp_path,
            calendar={**CALENDAR, **later},
        )
    with pytest.raises(ValueError, match="horizons is stale or tampered"):
        _verify(
            store,
            record.path,
            candidate_dir,
            shadow,
            run_store,
            tmp_path,
            horizons=("morning", "pregame_30m"),
        )


def test_verify_rejects_other_candidate_or_manifest_digest(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    remanifested = tmp_path / "remanifested"
    shutil.copytree(candidate_dir, remanifested)
    manifest_path = remanifested / "bundle_manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest digest"):
        _verify(store, record.path, remanifested, shadow, run_store, tmp_path)

    broken = tmp_path / "broken"
    shutil.copytree(candidate_dir, broken)
    (broken / "feature_schema.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch|hash mismatch"):
        _verify(store, record.path, broken, shadow, run_store, tmp_path)


def test_verify_rejects_wrongly_addressed_evidence(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    elsewhere = store.root / "other-bundle" / record.path.name
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.path, elsewhere)
    with pytest.raises(ValueError, match="path does not match its content address"):
        _verify(store, elsewhere, candidate_dir, shadow, run_store, tmp_path)

    nested = store.root / manifest.bundle_id / "nested" / record.path.name
    nested.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.path, nested)
    with pytest.raises(ValueError, match="path does not match its content address"):
        _verify(store, nested, candidate_dir, shadow, run_store, tmp_path)

    renamed = tmp_path / "renamed.json"
    shutil.copyfile(record.path, renamed)
    with pytest.raises(ValueError, match="path does not match its content address"):
        _verify(store, renamed, candidate_dir, shadow, run_store, tmp_path)


@pytest.mark.parametrize(
    "horizons, calendar",
    [
        ((), CALENDAR),
        (("morning", "morning"), CALENDAR),
        (("",), CALENDAR),
        (("morning", ""), CALENDAR),
        ("morning", CALENDAR),
        (HORIZONS, []),
        (HORIZONS, {}),
        (HORIZONS, {"2026-13-40": ["g1"]}),
        (HORIZONS, {"10/01/2026": ["g1"]}),
        (HORIZONS, {"2026-10-1": ["g1"]}),
        (HORIZONS, {"2026-10-01": []}),
        (HORIZONS, {"2026-10-01": ["g1", "g1"]}),
        (HORIZONS, {"2026-10-01": "g1"}),
        (HORIZONS, {"2026-10-01": [None]}),
        (HORIZONS, _DuplicateDateCalendar()),
    ],
)
def test_invalid_horizons_and_calendars_are_rejected(
    tmp_path, candidate, horizons, calendar
):
    candidate_dir, _ = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    with pytest.raises(ValueError):
        _build(
            store,
            candidate_dir,
            shadow,
            _FakeRunStore({}),
            tmp_path,
            horizons=horizons,
            calendar=calendar,
        )


def test_invalid_inputs_are_rejected_before_the_candidate_is_read(tmp_path):
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    with pytest.raises(ValueError, match="horizons"):
        _build(
            store,
            tmp_path / "missing-candidate",
            shadow,
            _FakeRunStore({}),
            tmp_path,
            horizons=(),
        )
    with pytest.raises(ValueError, match="expected_calendar"):
        _build(
            store,
            tmp_path / "missing-candidate",
            shadow,
            _FakeRunStore({}),
            tmp_path,
            calendar={},
        )


def test_verify_rejects_invalid_inputs_and_unknown_schema(
    tmp_path, candidate, official_snapshot_stubs
):
    candidate_dir, manifest = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    shadow = ShadowSlateStore(tmp_path / "shadow.json")
    runs = _build_runs(manifest.bundle_id, CALENDAR, HORIZONS)
    _record_calendar(shadow, manifest.bundle_id, CALENDAR, HORIZONS, runs)
    run_store = _FakeRunStore(runs)
    record = _build(store, candidate_dir, shadow, run_store, tmp_path)

    with pytest.raises(ValueError, match="horizons"):
        _verify(
            store,
            record.path,
            candidate_dir,
            shadow,
            run_store,
            tmp_path,
            horizons=(),
        )

    payload = json.loads(record.path.read_text(encoding="utf-8"))
    payload["schema_version"] = "release_evidence_v0"
    record.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        _verify(store, record.path, candidate_dir, shadow, run_store, tmp_path)


def test_shadow_store_is_required(tmp_path, candidate):
    candidate_dir, _ = candidate
    store = ReleaseEvidenceStore(tmp_path / "evidence")
    with pytest.raises(ValueError, match="ShadowSlateStore is required"):
        _build(store, candidate_dir, None, _FakeRunStore({}), tmp_path)
