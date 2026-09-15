from src.operations.shadow import ShadowSlateStore, make_shadow_slate


def _record(store, day, *, success=True):
    store.record(
        make_shadow_slate(
            slate_date=f"2026-10-{day:02d}",
            bundle_id="bundle",
            source_snapshot_id=f"snapshot-{day}",
            horizons=("morning",),
            games_expected=2,
            games_completed=2 if success else 1,
            forecasts_persisted=20,
            strict=True,
            contract_valid=True,
            replay_parity_valid=True,
        )
    )


def test_shadow_store_requires_seven_successful_recorded_slates(tmp_path):
    store = ShadowSlateStore(tmp_path / "shadow.json")
    for day in range(1, 8):
        _record(store, day)
    assert store.consecutive_successes() == 7
    # A diagnostic success count cannot qualify an unspecified candidate.
    assert not store.promotion_evidence()["eligible"]
    assert not store.promotion_evidence(
        bundle_id="bundle", horizons=("morning",),
        expected_calendar={f"2026-10-{day:02d}": ["g1", "g2"] for day in range(1, 8)},
    )["eligible"]


def test_failed_shadow_slate_resets_streak_and_dates_are_immutable(tmp_path):
    store = ShadowSlateStore(tmp_path / "shadow.json")
    _record(store, 1)
    _record(store, 2, success=False)
    _record(store, 3)
    assert store.consecutive_successes() == 1

    import pytest

    with pytest.raises(FileExistsError):
        _record(store, 3)


def test_shadow_streak_cannot_mix_candidates(tmp_path):
    from dataclasses import replace
    import pytest

    store = ShadowSlateStore(tmp_path / "shadow.json")
    _record(store, 1)
    store.record(replace(store.load()[0], bundle_id="other"))
    with pytest.raises(ValueError, match="Select one bundle"):
        store.consecutive_successes()
    assert store.consecutive_successes(bundle_id="bundle") == 1
