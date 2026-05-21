# Current State

## Snapshot

- Observed date: 2026-05-23
- Repository health: good
- Test status in this workspace:
  - full suite after season-context feature groups (2026-05-23): `279 passed, 1 skipped`
  - full suite after lifecycle ML integration (2026-05-22): TBD
  - full suite (2026-05-18): `178 passed, 0 failed`
  - full suite after evaluation module + ensemble optimizer: `178 passed, 0 failed`
  - full suite after simulation refactor (phase simulator, archetype, role sampler): `178 passed, 0 failed`
  - full suite after training-stopping bug fixes: `178 passed, 0 failed`
  - full suite (2026-04-25): `178 passed, 0 failed` in 74.82s
  - earlier suite after teammate-utils refactor + rest_density/lineup_stability vectorisation + performance smoke tests: `178 passed, 0 failed`
  - full suite after scraper reliability fixes (rotowire, nba_defense, schedule): `140 passed, 0 failed`
  - full suite after TRANSFORMER-SEQ-001 (M tier seq_len=20, zero-padding, new tests): `136 passed, 0 failed`
- The venv uses Python 3.12 with all dependencies installed cleanly. `torch`, `catboost`, `sklearn`, and all project modules import without error.

## What Currently Works

- Core repo structure is coherent enough to understand and extend.
- Historical data ingestion flow exists in `update_data.py` with multiple season-selection modes.
- Feature engineering is substantial, modular, and well-covered (26 feature groups, all test-covered).
- Training internals for CatBoost/Transformer components are implemented and testable in isolation.
- Six training-stopping bugs in CatBoost + Transformer pipeline are fixed (2026-05-09):
  - CatBoost import guard prevents crash when catboost isn't installed
  - feature_cols null check prevents downstream crashes on empty feature sets
  - bare raise fixed to proper exception handling
  - macOS persistent_workers guard prevents DataLoader crash
  - graceful feature missingness degradation instead of hard failure
  - parallel thread contention fix for CPU training
  - transformer save guard prevents checkpoint corruption
- Architecture cleanup completed (2026-05-09): deduplicated code paths, aligned config references, removed dead code.
- **New: Self-optimizing ensemble weight system** (2026-05-09):
  - `backtest.py` — CLI for evaluating prediction accuracy on historical games
  - `optimize_weights.py` — CLI for retuning ensemble blend weights via scipy.optimize against holdout data
  - `src/evaluation/` module with 4 subsystems:
    - `backtest_runner.py` — runs model predictions against completed games, computes per-stat MAE/RMSE/R²/calibration
    - `ensemble_optimizer.py` — 13-parameter optimizer (6 per-target CatBoost/Transformer ratios + 6 intercepts + 1 CatBoost-MAE blend)
    - `weight_store.py` — versioned JSON weight store replacing opaque binary `blend_weights.pkl`, with atomic writes and rollback
    - `drift_detector.py` — statistical process control: flags when rolling MAE exceeds 2σ above baseline
    - `metrics.py` — BacktestResult, TargetMetrics, calibration, interval coverage dataclasses
  - `ModelManager` refactored so all blend coefficients come from a single `EnsembleWeights` config object, hot-reloadable
- **New: Simulation layer refactor** (2026-05-09):
  - `src/simulation/phase_simulator.py` — extracted from GameSimulator: phase-by-phase Monte Carlo game loop
  - `src/simulation/archetype.py` — ArchetypeEngine: infers player archetypes (heliocentric star guard, 3&D wing, etc.) from projection shape
  - `src/simulation/role_sampler.py` — samples role states (limited/normal/expanded/starter/bench/closer) with archetype-aware adjustments
  - `src/simulation/sim_types.py` — typed dataclasses (RoleSample, PhaseDefinition, GameEnvironment, PlayerProjection, TeamContext) replacing raw dicts
  - `src/simulation/sim_cache.py` — JSON disk-cache mixin for simulator
  - `src/simulation/stat_utils.py` — shared statistical helpers (compute_mode, summary stats)
  - `src/simulation/game_simulator.py` — heavily refactored; dead legacy code removed, uses typed dataclasses
  - `src/query/prob_formatter.py` — ProbFormatterMixin for formatted probability output
- `config/default.yaml` now supports `self_optimization:` section for ensemble tuning config.
- **New: Lifecycle ML integration** (2026-05-22):
  - `src/lifecycle/aging_model.py` — B-Ianus Bayesian aging curves with position-specific peak-age priors. Precomputed at training startup, cached to `data/cache/aging_curves.csv`.
  - `src/lifecycle/kan_age_model.py` — KAN nonlinear age curves. CPU-only precomputation, cached to `data/cache/kan_aging_outputs.csv`.
  - `src/models/nexus_model.py` — Nexus multi-modal architecture (SSM + FT-Transformer + GAT + Copula head). Implemented and import-tested; not yet the active training path.
  - `src/training/nexus_loss.py` — GaussianNLLLoss for multivariate Gaussian with Cholesky covariance.
  - `src/preprocessing/feature_engineer_gpu.py` — GPU-accelerated feature engineering via cuDF with transparent CPU fallback.
  - 4 new feature groups (23 total): `InjuryRiskFeatureGroup`, `AgingCurveFeatureGroup`, `KANAgingFeatureGroup`, `SkillDevelopmentFeatureGroup`.
  - `src/data/player_bio_scraper.py` — PlayerBioScraper fetches AGE, POSITION, HEIGHT, WEIGHT from NBA API. Called from `update_data.py` via `enrich_with_player_bios()`.
  - `src/data/injury_history_logger.py` — InjuryHistoryLogger persists injury events across runs to `data/injury_history.csv`.
  - `update_data.py` also writes Parquet dual-write (`nba_players.parquet`, `nba_games.parquet`) for GPU-direct storage reads.
- **New: Strict simulation mode + data quality schema** (DR-025, 2026-05-22):
  - `simulate_season.py` exposes `--strict` CLI flag for fail-fast on degraded optional context.
  - `GameSimulator` and `SeasonSimulator` accept `strict_mode` parameter. Raises `RuntimeError` on degraded optional source when strict.
  - `ReportGenerator.export_player_projections()` appends `DATA_QUALITY` column (`FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`).
  - `ProjectionLoader.find_player()` surfaces visible CLI warning when loading degraded projections.
- **New: Season-context feature groups** (2026-05-23):
  - `SeasonPhaseFeatureGroup` — early-season rust detection: `DAYS_SINCE_SEASON_START` (capped at 30), `IS_SEASON_OPENER`, `GAMES_WITH_CURRENT_TEAM` (resets on trade), `IS_RECENT_TRADE` (≤5 games with new team).
  - `TeamMotivationFeatureGroup` — late-season signals: `TEAM_CUMULATIVE_WIN_PCT` (shift-1, no leakage), `IS_LATE_SEASON` (Mar+), `IS_TANKING_PROXY` (<.35 win %), `IS_PLAYOFF_LOCK_PROXY` (>0.65 win %).
  - `PostseasonContextFeatureGroup` — playoff detection: `IS_PLAYOFF_GAME` (from `SEASON_TYPE` or `GAME_TYPE`), `PLAYOFF_PACE_PRIOR` (0.95 prior for playoff pace drop).
  - `DAYS_SINCE_LAST_GAME` capped at 14 days in `RestGameDensityFeatureGroup` to prevent off-season gaps from creating infinite rest outliers.
- **Phase-aware drift detection** (2026-05-23):
  - `DriftDetector.record()` and `detect()` accept a `phase` parameter (`REGULAR`/`PLAYOFF`).
  - Auto-infers phase from date (Apr 15–Jun 20 = playoff).
  - Prevents false "major drift" alerts when the playoffs naturally lower scoring and pace.
- Transformer M tier now uses `seq_len=20` (up from 10), matching L tier's context window.
- Both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) use zero-padding for short players instead of skipping them.
- Training presets and feature engineering remain stable; 26 feature groups in the `full` preset (23 existing + 3 season-context).
- All six target stats exported from report generator and loaded correctly by projection loader.
- Training-to-simulation artifact contract is enforced at both boundaries.
- Dead code in `GameSimulator` has been removed.
- The evaluation module now provides quantitative backtest metrics for drift detection and weight optimization.

## What Partially Works

- Simulation stack uses typed dataclasses and phase simulation, but still depends on volatile third-party scrapers.
- Ensemble weight optimization framework is implemented and test-covered, but needs live backtest runs against real completed games to prove the optimization loop end-to-end.
- Drift detector is implemented but needs a rolling window of live backtest results to establish baseline statistics.
- Transformer validation inference uses eager-safe path; live CUDA smoke test still pending.
- Schedule scraping and season simulation are implemented; upstream availability remains a risk.
- Lifecycle aging curves and KAN precomputation are implemented but need live validation with real player bio data.
- Nexus multi-modal model is implemented and import-tested but not yet wired as the active training path.
- GPU feature engineering (cuDF) is implemented with CPU fallback but needs GPU-equipped runtime validation.

## What Is Broken Or Very Likely Broken

- No currently known critical bugs. The six training-stopping bugs, dead code, and legacy scraper regressions have all been fixed.
- Upstream scraper availability remains the primary operational risk — but degraded inputs are now surfaced explicitly.

## Current Limitations

- Ensemble optimizer and drift detector need live-data validation (backtest against real completed games).
- No checked-in trained models are present in this workspace beyond cache directories.
- The repo uses many local file contracts rather than strong typed interfaces between phases — though the simulation layer is moving toward typed dataclasses.
- `simulate_season.py` carries a run-level input health summary and exits non-zero for hard schedule failures, but still needs a live smoke test against current upstream sources.

## Active Risks

- The main scraper risk is upstream drift; optional failures degrade runs visibly, schedule failures are hard-required.
- Artifact naming or schema drift is now guarded by training/runtime validation and versioned weight storage.
- The blend weight system now supports hot-reloading and versioning, but operators must be aware that `optimize_weights.py` writes to the new weight store format.

## Known Workarounds

- If lineup, injury, betting, or defense context scrapers fail, `GameSimulator` now continues in explicit degraded mode (or fails fast with `--strict`).
- Query users can query all six stats (PTS, REB, AST, STL, BLK, TOV) from exported projection CSVs. A `DATA_QUALITY` warning surfaces when a projection uses fallback data.
- `clear_cache.py` can reset generated state while preserving raw input CSVs.
- `optimize_weights.py --rollback N` can revert to previous weight versions if a retune degrades accuracy.
- If `player_bios.csv` is missing, aging features default to neutral (1.0 factor). Run `update_data.py` with `--interactive` to populate it.
- Delete `data/cache/aging_curves.csv` and `data/cache/kan_aging_outputs.csv` to force recomputation of lifecycle caches after retraining.

## Immediate Priorities

1. Run `backtest.py` against a completed historical window (e.g., `--recent 14`) to establish baseline metrics and feed the drift detector.
2. Run `optimize_weights.py --recent 14` to validate the self-optimization loop end-to-end with real completed games.
3. Benchmark `python train.py --preset small` against the full preset on live CSVs to confirm iteration-speed improvement.
4. ~~Add a strict simulation mode for optional scraper degradation (CLI flag to fail-fast).~~ **DONE (DR-025, 2026-05-22)**
5. Run one live `train.py` -> `ModelManager.load_models()` -> `simulate_season.py` smoke test in a healthy local environment with real CSV inputs.
6. Collect real-data performance profile for the rolling feature path to compare against the synthetic benchmark.
7. ~~Run full test suite after lifecycle ML integration to update baseline test count.~~ **DONE (279 passed, 1 skipped)**
8. Validate lifecycle aging precomputation in `train.py` with real player bio data.
9. Run training smoke test with season-context features active to confirm `WL` and `SEASON_TYPE` columns propagate through the pipeline.

## Testing Status

- Verified on 2026-05-23: full suite `279 passed, 1 skipped` after season-context feature groups + lifecycle ML integration.
- Verified on 2026-05-18: full suite `178 passed, 0 failed` after evaluation module + simulation refactor + training bug fixes.
- Earlier test runs documented:
  - 2026-04-25: full suite `178 passed, 0 failed` in 74.82s
  - 2026-04-21 after teammate-utils and vectorization: `178 passed, 0 failed`
  - 2026-04-12 after 6-stat contract tests: `132 passed, 0 failed`
  - 2026-04-11 after torch-shim + checkpoint fix: `110 passed, 0 failed`
- Two tests are skipped by design around Transformer runtime constraints.
- pandas fragmentation warning flood is resolved; no PerformanceWarning in test runs.

## Areas That Need Confirmation In A Future Session

- Whether the self-optimization loop converges on better weights than the training-time defaults.
- Whether the drift detector correctly identifies real performance degradation vs. noise.
- Whether a full `python train.py` smoke run completes with all required runtime artifacts from real CSV data.
- Whether the new weight store format interoperates correctly with legacy `blend_weights.pkl` consumers.
