"""Fake-object tests for truthful optimizer promotion and verification gates.

Covers dry-run safety, same-window verification, non-finite verification
rejection, weight restoration on every failure path, append-only audit
records, and promoted-JSON provenance. Only fake runner/manager/store
objects and tmp directories are used — the real ``models/`` tree is never
touched.
"""

import json
import math

import pytest

from src.evaluation.ensemble_optimizer import EnsembleOptimizer
from src.evaluation.metrics import BacktestResult, TargetMetrics
from src.evaluation.weight_store import EnsembleWeights, WeightStore

TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
HOLDOUT_START = "2026-04-15"
HOLDOUT_END = "2026-05-01"
VERIFY_START = "2026-03-01"
VERIFY_END = "2026-04-14"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeManager:
    """Minimal ModelManager stand-in: records applied weight tags."""

    def __init__(self):
        # Truthy so the optimizer skips _load_models().
        self.models = {"PTS": object()}
        self.ensemble_weights = None
        self.applied = []  # tags of weights applied, in order

    def use_ensemble_weights(self, weights):
        self.ensemble_weights = weights
        self.applied.append(getattr(weights, "tag", None))

    @property
    def current_tag(self):
        return getattr(self.ensemble_weights, "tag", None)

    def _load_models(self):
        raise AssertionError("_load_models must not be called when models are present")


class FakeRunner:
    """Scripted BacktestRunner: score = score_fn(date_start, date_end, idx).

    ``idx`` is the number of prior backtests on the same range, so the first
    call on a range scores the current weights and later calls score
    candidates — deterministic for both in-memory and on-disk stores.
    """

    def __init__(self, manager, score_fn):
        self._manager = manager
        self.score_fn = score_fn
        self.calls = []  # (tag, date_start, date_end) for every run()

    @property
    def targets(self):
        return TARGETS

    def load_feature_df(self):
        return "FEATURES"

    def run(self, date_start, date_end, *, feature_df=None,
            force_recompute=False, progress=True):
        tag = self._manager.current_tag
        idx = sum(1 for (_, s, e) in self.calls if (s, e) == (date_start, date_end))
        score = float(self.score_fn(date_start, date_end, idx))
        self.calls.append((tag, date_start, date_end))
        per_target = {
            t: TargetMetrics(target=t, mae=score, rmse=score, r2=0.0, num_samples=10)
            for t in TARGETS
        }
        return BacktestResult(
            date_start=date_start,
            date_end=date_end,
            num_games=1,
            num_players=1,
            per_target=per_target,
            weighted_score=score,
        )


class FakeStore:
    """In-memory WeightStore stand-in: counts saves, records audit runs."""

    def __init__(self, current=None):
        self.current = current if current is not None else _default_weights()
        self.save_count = 0
        self.saved = []
        self.run_records = []

    def load_current(self):
        return self.current

    def save(self, weights, set_current=True):
        self.save_count += 1
        self.saved.append(weights)
        weights.version = self.save_count
        return self.save_count

    def record_run(self, entry):
        self.run_records.append(entry)


def _default_weights():
    weights = EnsembleWeights.default_for_targets(TARGETS)
    weights.tag = "current"
    return weights


def _make_optimizer(manager, store, score_fn, **kwargs):
    runner = FakeRunner(manager, score_fn)
    optimizer = EnsembleOptimizer(runner, store, **kwargs)
    return optimizer, runner


def _run(optimizer, **kwargs):
    kwargs.setdefault("verification_start", VERIFY_START)
    kwargs.setdefault("verification_end", VERIFY_END)
    kwargs.setdefault("progress", False)
    return optimizer.optimize(HOLDOUT_START, HOLDOUT_END, **kwargs)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def _accepting_score(start, end, idx):
    """Baseline 10.0 on holdout; candidate 5.0. Verification improves 8.0→7.0."""
    if (start, end) == (HOLDOUT_START, HOLDOUT_END):
        return 10.0 if idx == 0 else 5.0
    if (start, end) == (VERIFY_START, VERIFY_END):
        return 8.0 if idx == 0 else 7.0
    raise AssertionError(f"unexpected backtest range {start} → {end}")


def _rejecting_holdout_score(start, end, idx):
    """Candidate improves too little on holdout to pass the accept gate."""
    if (start, end) == (HOLDOUT_START, HOLDOUT_END):
        return 10.0 if idx == 0 else 9.99
    if (start, end) == (VERIFY_START, VERIFY_END):
        return 8.0
    raise AssertionError(f"unexpected backtest range {start} → {end}")


def _rejecting_verify_score(start, end, idx):
    """Candidate wins holdout but degrades the verification window."""
    if (start, end) == (HOLDOUT_START, HOLDOUT_END):
        return 10.0 if idx == 0 else 5.0
    if (start, end) == (VERIFY_START, VERIFY_END):
        return 8.0 if idx == 0 else 9.0
    raise AssertionError(f"unexpected backtest range {start} → {end}")


def _invalid_verify_score(start, end, idx):
    """Candidate verification score is non-finite."""
    if (start, end) == (HOLDOUT_START, HOLDOUT_END):
        return 10.0 if idx == 0 else 5.0
    if (start, end) == (VERIFY_START, VERIFY_END):
        return 8.0 if idx == 0 else float("nan")
    raise AssertionError(f"unexpected backtest range {start} → {end}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_accepted_dry_run_performs_zero_saves():
    manager = FakeManager()
    store = FakeStore()
    optimizer, _ = _make_optimizer(manager, store, _accepting_score)

    result = _run(optimizer, dry_run=True)

    assert result.accepted
    assert result.dry_run
    assert not result.deployed
    assert store.save_count == 0
    assert store.saved == []
    assert len(store.run_records) == 1
    record = store.run_records[0]
    assert record["dry_run"] is True
    assert record["status"] == "accepted"
    assert record["promoted_version"] is None
    # Dry-run exit restores the original weights on the manager.
    assert manager.current_tag == "current"
    assert manager.applied[-1] == "current"


def test_accepted_real_run_saves_exactly_once():
    manager = FakeManager()
    store = FakeStore()
    optimizer, _ = _make_optimizer(manager, store, _accepting_score)

    result = _run(optimizer)

    assert result.accepted
    assert result.deployed
    assert not result.dry_run
    assert store.save_count == 1
    assert len(store.saved) == 1
    saved = store.saved[0]
    assert saved.optimizer_method == "Nelder-Mead"
    assert saved.backtest_score == 5.0
    assert saved.backtest_date_range == f"{HOLDOUT_START}→{HOLDOUT_END}"
    assert result.weights.version == 1
    # A real promotion keeps the candidate weights hot-loaded.
    assert manager.current_tag is None
    record = store.run_records[0]
    assert record["status"] == "accepted"
    assert record["dry_run"] is False
    assert record["promoted_version"] == 1


def test_failed_holdout_gate_restores_original_weights():
    manager = FakeManager()
    store = FakeStore()
    optimizer, _ = _make_optimizer(manager, store, _rejecting_holdout_score)

    result = _run(optimizer)

    assert not result.accepted
    assert "Improvement" in result.rejection_reason
    assert store.save_count == 0
    # Original weights restored on the manager.
    assert manager.current_tag == "current"
    assert manager.applied[-1] == "current"
    record = store.run_records[0]
    assert record["status"] == "rejected"
    assert record["promoted_version"] is None


def test_failed_verification_gate_restores_original_weights():
    manager = FakeManager()
    store = FakeStore()
    optimizer, _ = _make_optimizer(manager, store, _rejecting_verify_score)

    result = _run(optimizer)

    assert not result.accepted
    assert "Verification degradation" in result.rejection_reason
    assert store.save_count == 0
    # Original weights restored on the manager.
    assert manager.current_tag == "current"
    assert manager.applied[-1] == "current"
    record = store.run_records[0]
    assert record["current_verification_score"] == 8.0
    assert record["candidate_verification_score"] == 9.0
    assert record["promoted_version"] is None


def test_verification_uses_current_and_candidate_scores_from_same_range():
    manager = FakeManager()
    store = FakeStore()
    optimizer, runner = _make_optimizer(manager, store, _accepting_score)

    result = _run(optimizer, dry_run=True)

    assert result.accepted
    # Exactly two verification backtests ran — one per weight set — and both
    # used the exact same window.
    verify_calls = [
        (tag, start, end)
        for (tag, start, end) in runner.calls
        if (start, end) == (VERIFY_START, VERIFY_END)
    ]
    assert len(verify_calls) == 2
    assert {tag for tag, _, _ in verify_calls} == {"current", None}
    assert len({(start, end) for _, start, end in verify_calls}) == 1
    # No verification backtest ever ran on the holdout window.
    holdout_verify_calls = [
        (tag, start, end) for (tag, start, end) in runner.calls
        if start == HOLDOUT_START and tag == "current" and end == HOLDOUT_END
    ]
    assert len(holdout_verify_calls) == 1  # baseline only
    record = store.run_records[0]
    assert record["verification_range"] == f"{VERIFY_START}→{VERIFY_END}"
    assert record["current_verification_score"] == 8.0
    assert record["candidate_verification_score"] == 7.0


def test_invalid_verification_rejects():
    manager = FakeManager()
    store = FakeStore()
    optimizer, _ = _make_optimizer(manager, store, _invalid_verify_score)

    result = _run(optimizer)

    assert not result.accepted
    assert "invalid_verification" in result.rejection_reason
    assert store.save_count == 0
    # Invalid verification must never skip the gate and promote.
    assert manager.current_tag == "current"
    record = store.run_records[0]
    assert record["status"] == "rejected"
    assert "invalid_verification" in record["reason"]
    assert record["current_verification_score"] == 8.0
    assert record["candidate_verification_score"] is None
    assert record["promoted_version"] is None


def test_rejected_and_dry_run_attempts_create_audit_records_but_not_versions(tmp_path):
    store = WeightStore(str(tmp_path / "blend_weights"))
    manager = FakeManager()
    optimizer, _ = _make_optimizer(manager, store, _rejecting_verify_score)

    rejected = _run(optimizer)
    dry_run = _run(optimizer, dry_run=True)

    assert not rejected.accepted
    assert not dry_run.accepted
    # No version files, no current pointer, no promotion history — only the
    # separate append-only audit log.
    assert list(store.store_dir.glob("v*.json")) == []
    assert not (store.store_dir / "current.json").exists()
    assert not (store.store_dir / "history.json").exists()
    records = store.load_run_records()
    assert len(records) == 2
    assert records[0]["dry_run"] is False
    assert records[1]["dry_run"] is True
    for record in records:
        assert record["status"] == "rejected"
        assert record["promoted_version"] is None
        assert record["holdout_range"] == f"{HOLDOUT_START}→{HOLDOUT_END}"
        assert record["verification_range"] == f"{VERIFY_START}→{VERIFY_END}"


def test_promoted_json_has_non_null_score_date_and_method_provenance(tmp_path):
    store = WeightStore(str(tmp_path / "blend_weights"))
    manager = FakeManager()
    optimizer, _ = _make_optimizer(manager, store, _accepting_score)

    result = _run(optimizer)

    assert result.accepted
    assert result.deployed
    version_files = list(store.store_dir.glob("v*.json"))
    assert len(version_files) == 1
    data = json.loads(version_files[0].read_text(encoding="utf-8"))
    assert data["backtest_score"] is not None
    assert math.isfinite(data["backtest_score"])
    assert data["backtest_date_range"] == f"{HOLDOUT_START}→{HOLDOUT_END}"
    assert data["optimizer_method"] == "Nelder-Mead"
    # current.json mirrors the promoted version.
    current_data = json.loads(
        (store.store_dir / "current.json").read_text(encoding="utf-8")
    )
    assert current_data["version"] == data["version"]


def test_old_weight_json_without_new_fields_reads_backward_compatible(tmp_path):
    """Old weight JSON lacking optional fields still loads (req 7)."""
    store = WeightStore(str(tmp_path / "blend_weights"))
    old = {
        "version": 3,
        "created_at": "2026-01-01T00:00:00",
        "description": "old format",
        "per_target": {
            "PTS": {"catboost": 0.7, "transformer": 0.3, "intercept": 0.0},
        },
    }
    (store.store_dir / "current.json").write_text(json.dumps(old), encoding="utf-8")

    weights = store.load_current()

    assert weights is not None
    assert weights.version == 3
    assert weights.backtest_score is None
    assert weights.backtest_date_range == ""
    assert weights.catboost_mae_blend == 0.7
    assert weights.optimizer_method == ""
    assert weights.per_target["PTS"].catboost == 0.7
    assert weights.per_target["PTS"].transformer == 0.3
