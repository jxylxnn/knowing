# Self-Optimizing Ensemble Weight System — Plan & Cons Analysis

## Executive Summary

The current system has a critical architectural gap: **ensemble blend weights are computed once at training time and frozen forever.** The `blend_weights.pkl` file is generated during `train.py`, loaded once by `ModelManager`, and never updated — even as new games are played, player roles shift, and model accuracy drifts. The `nba-oracle` project solves this with a continuous self-optimization loop that retunes weights as the season progresses. We need the same capability here, adapted to this project's CatBoost+Transformer stack.

**This plan identifies 12 cons/blockers that must be resolved before or during implementation, and lays out a phased, dependency-ordered roadmap.**

---

## CONS & BLOCKERS (Must Address)

### BLOCKER 1: No Backtesting Framework Exists
**Severity:** CRITICAL — you cannot optimize what you cannot measure.

The project has no systematic way to compare predictions against actual outcomes after games complete. The pipeline is:
```
update_data → train → simulate → query
```
There is no `backtest` step that says "for historical date X, predict the games, then compare against what actually happened."

**Impact on self-optimization:** Without backtesting, the optimizer has no fitness function. It can't know if a weight change improved or degraded accuracy.

**Fix required before Phase 2:** Build a `BacktestRunner` that:
- Takes a historical date range
- Runs `predict_player_stats_batch()` for each game
- Compares predictions to actual box scores
- Computes per-stat MAE, RMSE, R², and calibration error
- Produces a `BacktestResult` dataclass the optimizer can consume

### BLOCKER 2: Fixed Weights Are Hardcoded at Multiple Levels
**Severity:** CRITICAL

There are THREE places where blend weights are frozen:

| Location | What's frozen | How |
|---|---|---|
| `model_manager.py:305` | CatBoost-to-MAE split | `0.7 * primary + 0.3 * mae_pred` — hardcoded literal |
| `model_manager.py:406-407` | CatBoost-to-Transformer blend | `blend_cfg.get("catboost", 1.0)` — loaded once from disk |
| `blend_weights.pkl` | All per-target blend weights | Computed once at training time, never refreshed |

The self-optimizer must be able to tune ALL of these — not just the CatBoost/Transformer blend.

**Fix required:** Refactor `ModelManager` so all blend coefficients come from a single `EnsembleWeights` config object that can be hot-reloaded at runtime, with proper defaults when missing.

### BLOCKER 3: Config Loader Rejects Unknown Keys
**Severity:** HIGH

From bug scan #13: `src/config/config.py:429-463` has no filtering of unknown YAML keys. Adding a `self_optimization:` section to `config/default.yaml` will crash on load.

**Fix required:** Add `ignore_unknown_keys=True` passthrough in the config loader, or add the `self_optimization` dataclass to `model_config.py` before any YAML changes.

### BLOCKER 4: SimulationConfig Missing `use_four_factors` Field
**Severity:** HIGH (prevents config from loading)

From bug scan #4: `config/default.yaml:139` has `use_four_factors: true` but `SimulationConfig` dataclass doesn't have this field → `TypeError` on config load.

**Fix required:** Add `use_four_factors: bool = True` to the `SimulationConfig` dataclass. This blocks ANY config changes including self-optimization config.

### BLOCKER 5: No Weight Versioning or Rollback
**Severity:** HIGH

`blend_weights.pkl` is a single opaque binary file. There's no:
- Version history (what changed, when, why?)
- Rollback capability (revert to last known good)
- Diff/audit (what exactly changed between versions?)
- Atomic writes (corrupted write = broken system with no recovery)

The nba-oracle solves this by snapshotting config before writes and having an atomic accept/reject gate.

**Fix required:** Store blend weights as versioned JSON (not pickle), with a `blend_weights_history/` directory containing timestamped snapshots. YAML is also acceptable since the project already uses it. Keep a `blend_weights.current.json` symlink or pointer file.

### BLOCKER 6: 25+ Known Bugs Will Corrupt Optimization Feedback
**Severity:** HIGH

From the prior bug scan, there are critical bugs in data leakage (feature engineering), wrong config keys (`lr` vs `learning_rate`), and prediction inconsistencies (different blend defaults in different entry points). If the self-optimizer tunes weights against corrupted predictions, it will converge to wrong values.

**Fix required:** Fix at minimum these bugs before Phase 2:
- Bug #1: `use_four_factors` missing from SimulationConfig
- Bug #2: `lr` vs `learning_rate` key mismatch for Transformer
- Bug #8: Different blend weight defaults between `prediction_service.py` and `model_manager.py`
- Bug #7: Data leakage in `recency_form.py:67` (re-sort misaligns transforms)

### BLOCKER 7: No Holdout/Validation Data Pipeline
**Severity:** MEDIUM

The self-optimizer needs a holdout set — games it has NOT seen during training — to validate weight changes. Currently the project has a `test_split_date` in config (2025-01-01) but this is used for model training validation, not for post-training weight optimization.

**Fix required:** The backtest runner (from Blocker 1) should support configurable date ranges. The optimizer should use recent completed games (last 2-4 weeks) as its validation holdout to ensure it's optimizing for current-season accuracy, not historical fit.

### BLOCKER 8: Training Pipeline Is Too Slow for Iterative Optimization
**Severity:** MEDIUM

Full training takes hours (CatBoost 5000 iterations × 6 targets, Transformer 200 epochs). The self-optimizer CANNOT afford to retrain models on every iteration.

**Mitigation:** The optimizer should ONLY tune blend weights, not retrain models. This is the key insight from nba-oracle — the models stay fixed; only the ensemble coefficients change. Retraining happens on a separate cadence (weekly/daily). Blend weight optimization using a lightweight optimizer (e.g., `scipy.optimize.minimize` with Nelder-Mead or Bayesian optimization) over the holdout set should take seconds to minutes.

### BLOCKER 9: No GPU Resource Scheduling
**Severity:** MEDIUM

Training uses GPU. Backtesting uses GPU (Monte Carlo). If both run simultaneously, CUDA OOM or context contention occurs. The project already has `--workers 1` for GPU mode, but the optimizer needs to coordinate.

**Mitigation:** The orchestrator should serialize GPU access: backtest first (collect actuals), then optimize weights (CPU-bound, fast), then re-backtest to verify (GPU again). No concurrent GPU operations.

### BLOCKER 10: Feature Engineering Is Not Idempotent
**Severity:** MEDIUM

Feature engineering depends on cumulative historical data. Re-running `create_features()` on the same raw data at different times can produce different results because rolling windows shift. This means backtest results aren't perfectly reproducible.

**Mitigation:** The backtest runner should snapshot the feature DataFrame used for a backtest run and tag results with a data hash. Accept minor variance as inherent to time-series ML.

### BLOCKER 11: No Integration Tests for the Full Pipeline
**Severity:** LOW (but important for confidence)

Running `update_data → train → simulate → query` end-to-end is untested. The self-optimizer will run a variant of this loop (backtest → optimize → verify). If the base pipeline is fragile, the optimizer will be too.

**Fix recommended:** Add at least one smoke test that runs the full pipeline on a tiny dataset (2 players, 10 games).

### BLOCKER 12: No Drift Detection or Performance Monitoring
**Severity:** LOW (future concern)

Without tracking model accuracy over time, the optimizer can't distinguish between "weights need tuning" vs "models need retraining" vs "data distribution shifted." The optimizer might thrash, constantly retuning weights to chase noise.

**Mitigation:** Include a drift detector in Phase 3 that tracks per-stat MAE over a rolling 30-day window and raises a flag if error exceeds 2σ of historical baseline, suggesting full retraining is needed rather than just weight tuning.

---

## UPDATED IMPLEMENTATION PLAN

### Phase 0: Prerequisite Bug Fixes (EST. 1-2 hours)
**Must complete before any self-optimization work.**

1. **Add `use_four_factors: bool = True`** to `SimulationConfig` dataclass (`src/config/model_config.py`)
2. **Fix `lr` vs `learning_rate`** key mismatch for Transformer config (`src/config/model_config.py:358` or `config/default.yaml`)
3. **Standardize blend weight defaults** between `prediction_service.py:192-193` and `model_manager.py:406-407`
4. **Add unknown key tolerance** to config loader (`src/config/config.py:429-463`)
5. **Fix data leakage** in `recency_form.py:67` (re-sort index reset)
6. **Add `self_optimization` config section** to `config/default.yaml` (with dataclass in `model_config.py`)

### Phase 1: Backtesting Framework (EST. 2-3 days)
**Dependency for everything that follows.**

New file: `src/evaluation/backtest_runner.py`
- `BacktestRunner` class that:
  - Takes a date range and runs predictions for all games in that range
  - Compares against actual box scores from `data/nba_players.csv`
  - Computes per-stat metrics: MAE, RMSE, R², calibration error (P10/P90 coverage)
  - Returns `BacktestResult` dataclass with per-stat, per-model breakdown
- CLI entry point: `python backtest.py --from 2026-04-01 --to 2026-05-01`
- Caches predictions to avoid recomputation

New file: `src/evaluation/metrics.py`
- `BacktestResult` dataclass
- Stat-by-stat accuracy metrics
- Calibration metrics (do P10/P90 intervals contain the right % of actuals?)

### Phase 2: Blend Weight Manager (EST. 1-2 days)
**Refactors the hardcoded weights into a unified, tunable system.**

Refactor: `src/models/model_manager.py`
- Extract all blend coefficients into an `EnsembleWeights` dataclass:
  ```python
  @dataclass
  class EnsembleWeights:
      catboost_mae_blend: float = 0.7    # was hardcoded at line 305
      per_target: Dict[str, TargetBlend]  # catboost_weight, transformer_weight, intercept
      version: int = 1
      created_at: str = ""
      backtest_score: Optional[float] = None
  ```
- `ModelManager` loads `EnsembleWeights` from versioned JSON/YAML, not pickle
- Hot-reload support: `manager.reload_weights()` re-reads from disk without reloading models

New file: `src/evaluation/weight_store.py`
- Versioned weight storage: `models/blend_weights/v0001.json`, `v0002.json`, etc.
- `blend_weights/current.json` → symlink or pointer to active version
- `blend_weights/history.json` → log of all versions with scores
- Atomic write: write to temp file, rename

### Phase 3: Self-Optimization Engine (EST. 2-3 days)
**The core optimization loop.**

New file: `src/evaluation/ensemble_optimizer.py`
- `EnsembleOptimizer` class:
  - **Objective function:** Run backtest with candidate weights → return weighted MAE
  - **Optimizer:** `scipy.optimize.minimize` (Nelder-Mead, bounds-constrained) OR Bayesian optimization (`scikit-optimize` gp_minimize) for the per-target blend weights
  - **Tunable parameters:**
    - Per-target CatBoost/Transformer blend ratio (6 params)
    - Per-target intercept (6 params)
    - CatBoost/MAE companion blend ratio (1 param, currently 0.7)
    - Total: 13 tunable parameters
  - **Constraints:** All weights ∈ [0, 1], intercept ∈ [-2, 2], CatBoost weight + Transformer weight should be ~1.0 (not strictly enforced — intercept handles residual)
  - **Accept gate:** New weights must improve holdout MAE by ≥ `accept_margin` (configurable, default 1.0%)
  - **Verification gate:** Re-backtest on an OLDER holdout range to verify no regression on past data (max 2.0% degradation allowed)
  - **Atomic write:** If both gates pass, write new version to weight store

New CLI entry point: `python optimize_weights.py`
```bash
python optimize_weights.py --holdout-from 2026-04-15 --holdout-to 2026-05-01
python optimize_weights.py --dry-run   # preview what would change
python optimize_weights.py --rollback  # revert to previous weights
```

Config addition (`config/default.yaml`):
```yaml
self_optimization:
  enabled: false                    # opt-in; disabled by default
  accept_margin: 0.01              # 1% minimum improvement to accept
  verification_margin: 0.02        # 2% max degradation on verification set
  holdout_window_days: 14          # use last N days of completed games
  verification_window_days: 30     # older window for verification gate
  optimizer_method: "Nelder-Mead"  # or "Bayesian"
  max_iterations: 100
  schedule: "manual"               # "manual", "daily", "weekly"
```

### Phase 4: Automation & Monitoring (EST. 1-2 days)
**Scheduling and drift detection.**

New file: `src/evaluation/drift_detector.py`
- Tracks per-stat MAE over a rolling 30-day window
- Flags when error exceeds 2σ above baseline
- Suggests: "retrain models" vs "retune weights" based on drift magnitude

Cron integration (via `cronjob` skill):
- Daily cron job: run `optimize_weights.py` after new game data is fetched
- Weekly cron job: full model retraining if drift exceeds threshold

---

## ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-OPTIMIZATION LOOP                        │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐
  │ NBA Data │────▶│ Backtest     │────▶│ Metrics: MAE,   │
  │ (actuals)│     │ Runner       │     │ RMSE, R², Calib │
  └──────────┘     │ backtest.py  │     └────────┬────────┘
                   └──────────────┘              │
                                                 ▼
  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐
  │ Current  │────▶│ Ensemble     │◀────│ Optimizer       │
  │ Weights  │     │ Optimizer    │     │ Nelder-Mead /   │
  │ (v000N)  │     │ (scipy)      │     │ Bayesian        │
  └──────────┘     └──────┬───────┘     └─────────────────┘
                          │
                          ▼ candidate weights
                   ┌──────────────┐
                   │ Accept Gate  │── NO ──▶ Discard
                   │ ΔMAE ≥ 1%?  │
                   └──────┬───────┘
                          │ YES
                          ▼
                   ┌──────────────┐
                   │ Verify Gate  │── FAIL ─▶ Discard
                   │ Regr ≤ 2%?  │
                   └──────┬───────┘
                          │ PASS
                          ▼
                   ┌──────────────┐
                   │ Atomic Write │
                   │ v000N+1.json │
                   └──────────────┘
```

---

## FILE STRUCTURE (NET NEW)

```
src/evaluation/                    # NEW directory
├── __init__.py
├── backtest_runner.py             # Phase 1: BacktestRunner, CLI
├── metrics.py                     # Phase 1: BacktestResult, stat metrics
├── weight_store.py                # Phase 2: Versioned weight storage
├── ensemble_optimizer.py          # Phase 3: Optimization engine
└── drift_detector.py              # Phase 4: Performance drift monitoring

config/default.yaml                 # MODIFIED: add self_optimization section
src/config/model_config.py          # MODIFIED: add EnsembleWeights, SimulationConfig fix
src/models/model_manager.py         # MODIFIED: use EnsembleWeights, hot-reload
src/models/base.py                  # MODIFIED: new load/save for JSON weights

backtest.py                         # NEW entry point (Phase 1)
optimize_weights.py                 # NEW entry point (Phase 3)

tests/test_evaluation/              # NEW test package
├── __init__.py
├── test_backtest_runner.py
├── test_weight_store.py
├── test_ensemble_optimizer.py
└── test_drift_detector.py
```

---

## DEPENDENCY ORDER

```
Phase 0 (bugs) ──▶ Phase 1 (backtest) ──▶ Phase 2 (weight store) ──▶ Phase 3 (optimizer) ──▶ Phase 4 (automation)
                        │                         │
                        └── No parallelism ───────┘
                        (backtest must exist before weights can be scored)
```

---

## RISK MATRIX

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Optimizer overfits to holdout period | Medium | High | Verification gate on older data; configurable windows |
| Corrupted weight file breaks predictions | Low | Critical | Atomic writes, versioned snapshots, rollback command |
| GPU OOM during backtest+training overlap | Medium | Medium | Serialize GPU access; use CPU for optimizer |
| Optimizer converges to degenerate weights | Low | High | Bounds constraints, sanity checks on weight ranges |
| Feature drift makes old weights obsolete | Medium | Medium | Drift detector triggers retrain signal; configurable thresholds |

---

## WHAT THIS PLAN DOES NOT COVER (Out of Scope)

1. **Retraining models** — This system only tunes blend weights. Full model retraining remains a separate, manual (or cron-driven) step via `train.py`.
2. **Feature weight tuning** — nba-oracle retunes per-feature weights. This project uses CatBoost feature importance and 150+ features; per-feature tuning is a future enhancement.
3. **Web API / dashboard** — nba-oracle has a FastAPI+React UI. This project stays CLI-first as per AGENTS.md.
4. **Betting/edge detection** — nba-oracle includes Kelly criterion and edge detection. That's a separate feature, not part of self-optimization.
5. **Live game prediction** — Mid-game stat prediction is out of scope for now.
