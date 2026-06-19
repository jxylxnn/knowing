# Current State

## Snapshot

- Observed date: 2026-06-19
- Repository health: good
- Test status in this workspace:
  - feature-engineer cache tests on 2026-06-19: `pytest tests/test_preprocessing/test_feature_engineer.py::TestFeatureEngineerCache -q` -> `3 passed`. Full preprocessing package: `pytest tests/test_preprocessing/ -q` -> `107 passed`. Affected modules (pipeline/models/correction/evaluation, not-slow): `182 passed, 1 skipped`.
  - diagnostic mode tests on 2026-06-14: `pytest tests/test_training/test_diagnostics.py -q` -> `28 passed`.
  - existing tests after diagnostic mode: `pytest tests/test_training/ tests/test_contracts/ tests/test_models/test_model_manager.py -q` -> `75 passed`.
  - full non-slow suite on 2026-06-12 after bug-fix batch: `pytest -m "not slow" -q` -> `368 passed, 1 skipped, 1 deselected`.
  - targeted subset on 2026-06-12 after residual interval calibration: `pytest tests/test_correction/ tests/test_query/ tests/test_contracts/ tests/test_simulation/test_game_simulator.py::TestGameSimulatorUpgrade -q` -> `90 passed`.
  - targeted subset on 2026-06-11: `pytest tests/test_evaluation/ tests/test_contracts/` -> `23 passed` (the new smart-selector suite + contracts smoke test). This is on top of the 2026-06-04 baseline.
  - full suite after smart feature selection + weight bootstrap (2026-06-04): `313 passed, 1 skipped` (slow tests deselected)
  - full suite after calibration & probability upgrade (2026-05-21): `169 passed, 0 failed`
  - full suite after season-context feature groups (2026-05-23): `279 passed, 1 skipped`
  - full suite after lifecycle ML integration (2026-05-22): `294 passed, 1 skipped`
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
- **Bug-fix batch applied 2026-06-12** (KB-022 through KB-032):
  - Fixed `TrainingPipeline._save_blend_weights()` crash (`self.targets` -> `self.TARGETS`).
  - Fixed Transformer runtime inference using categorical ID columns; now aligns to `nn_features = feature_cols - cat_features`.
  - Fixed pytest torch shim `importlib.util` import failure on PyTorch-less machines.
  - Fixed `MinutesPredictor` `MINS_LAST_3` rolling-sum misalignment that produced NaN fatigue scores.
  - Fixed defense scraper key mismatch (`pts_allowed_per_100` vs `opp_pts_per_100`).
  - Fixed injury scraper missing `TEAM_ABBR` on disk-cache loads and guarded `LineupScraper` against missing column.
  - Fixed player bio `AGE` leakage by computing age relative to each row's `GAME_DATE` in `update_data.py::enrich_with_player_bios()`.
  - Fixed Basketball Reference scraper season URL format (uses ending year instead of `202425`).
  - Fixed `InjuryRiskFeatureGroup` career count to count only pre-game injuries.
  - Fixed `SkillDevelopmentFeatureGroup` to use expanding season-to-date averages instead of full-season aggregates, eliminating future in-season leakage.
- **New: Season-context feature groups** (2026-05-23):
  - `SeasonPhaseFeatureGroup` — early-season rust detection: `DAYS_SINCE_SEASON_START` (capped at 30), `IS_SEASON_OPENER`, `GAMES_WITH_CURRENT_TEAM` (resets on trade), `IS_RECENT_TRADE` (≤5 games with new team).
  - `TeamMotivationFeatureGroup` — late-season signals: `TEAM_CUMULATIVE_WIN_PCT` (shift-1, no leakage), `IS_LATE_SEASON` (Mar+), `IS_TANKING_PROXY` (<.35 win %), `IS_PLAYOFF_LOCK_PROXY` (>0.65 win %).
  - `PostseasonContextFeatureGroup` — playoff detection: `IS_PLAYOFF_GAME` (from `SEASON_TYPE` or `GAME_TYPE`), `PLAYOFF_PACE_PRIOR` (0.95 prior for playoff pace drop).
  - `DAYS_SINCE_LAST_GAME` capped at 14 days in `RestGameDensityFeatureGroup` to prevent off-season gaps from creating infinite rest outliers.
- **Phase-aware drift detection** (2026-05-23):
  - `DriftDetector.record()` and `detect()` accept a `phase` parameter (`REGULAR`/`PLAYOFF`).
  - Auto-infers phase from date (Apr 15–Jun 20 = playoff).
  - Prevents false "major drift" alerts when the playoffs naturally lower scoring and pace.
- **New: Calibration & probability upgrade — distribution fitter, copula, CRPS** (2026-05-21):
  - `src/query/distribution_fitter.py` — `DistributionFitter`: derives Mean/Std/Skew/Zero-Prob/Lambda from P10/P50/P90 quantile outputs. Enables full distribution parameters without the Nexus copula head.
  - `src/query/empirical_covariance.py` — `CovarianceCache`: archetype-conditioned 6x6 empirical correlation matrices from residual analysis. Cached to `data/cache/archetype_covariances.npz` as .npz.
  - `ProbabilityCalculator.run_copula_simulation()` — correlated multi-stat Monte Carlo using Gaussian copula + archetype correlations + per-stat inverse CDF (skew-normal for continuous, ZIP for count stats).
  - `ReportGenerator._enrich_with_distributions()` — appends `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA` to every exported projection CSV.
  - `calculate_empirical_crps()` in `src/evaluation/metrics.py` — fast O(n log n) CRPS via Gini mean difference, for probabilistic forecast quality evaluation.
  - `optimize_variance.py` — standalone CLI to tune 7 context-specific volatility multipliers (B2B, rookie, blowout, home, away, playoff, rest) via CRPS using scipy Nelder-Mead.
  - `ProbabilityCalculator` accepts optional `CovarianceCache`; lazy-loads default on first use.
- **New: Contracts layer** (2026-06-04):
  - `src/contracts/` module with 5 files: `artifacts.py` (the `ArtifactContract` runtime validator), `features.py` (the `FeatureSchema` contract), `projections.py` (the projection-CSV schema validator), `schedule.py` (the schedule input contract), and `errors.py` (typed `ContractError` hierarchy).
  - `check_contracts.py` (root) — standalone CLI to validate the artifact contract and projection CSV between pipeline steps. Flags `--models-dir`, `--projection-csv`, `--transformer-required`.
  - Both `train.py` and `simulate_season.py` invoke `validate_runtime_artifacts()` at startup. The optimizer/selector/sim stack is now swappable behind this seam without breaking the training/runtime contract.
  - `tests/test_contracts/test_pipeline_contract_smoke.py` covers the smoke path through the contract validator.
- **New: Smart per-target feature selection** (2026-06-04):
  - `src/evaluation/feature_group_ablation.py` — `FeatureGroupAblator`: trains baseline + leave-one-out `HistGradientBoostingRegressor` per feature group, computes per-target MAE deltas (`GroupScore` / `AblationReport`). Backbone of the `backtest_gain` signal.
  - `src/evaluation/shadow_feature_filter.py` — `ShadowFeatureFilter`: injects `SHADOW_RANDOM_NORMAL`, `SHADOW_RANDOM_UNIFORM`, `SHADOW_PERMUTED_TARGET` control columns, trains a fast model, and treats the median shadow importance as a noise floor. Features scoring below the floor are flagged as `below_shadow_median`.
  - `src/evaluation/smart_feature_selector.py` — `SmartFeatureSelector` combines 5 signals per target (`0.40 * backtest_gain + 0.25 * stability + 0.20 * catboost_importance + 0.10 * permutation_importance - 0.05 * missingness_penalty`), writes a per-target `SelectionManifest` to `models/feature_selection_manifest.json`, and gates each stage by a `ProfileConfig` (fast / balanced / max_accuracy).
  - `TrainingPipeline.apply_feature_selection_manifest()` and `_feature_cols_for_target()` — per-target feature lists consumed by CatBoost training; when no manifest is loaded, the canonical `self.feature_cols` list is used (preserves the original contract).
  - `TrainingPreset` now carries optional `feature_selection` and `feature_selection_profile` fields; presets can opt into smart selection through `config/default.yaml`.
  - `Config` dataclass now exposes `feature_selection` and `feature_selection_profiles` blocks from `config/default.yaml`.
  - `train.py` flags: `--feature-selection {off,smart}` and `--selection-profile {fast,balanced,max_accuracy}`. Runs between Step 2 (feature engineering) and Step 3 (training). Failure is non-fatal — falls back to the full feature set with a warning.
  - `FeatureSelector.select_features_for_target()` — accepts an `allowed_features` allow-list to build a target-specific `FeatureSchema` without re-running the leakage-safe filter.
  - `FeatureEngineeringResult` now records `n_rows` and `n_features` for selector diagnostics.
  - `ModelManager` now bootstraps `EnsembleWeights` from `WeightStore` at load time (after legacy `blend_weights.pkl` is loaded) so the runtime uses data-driven weights even before `optimize_weights.py` is run.
  - `TrainingPipeline._save_blend_weights()` also writes the training-time blend to `WeightStore` (versioned JSON) so the new bootstrap path can pick it up.
  - `backtest_result_to_json_dict()` in `metrics.py` — machine-readable JSON serializer (used by `backtest.py --json-output`) so downstream tools don't have to scrape terminal output.
  - `backtest.py --json-output <path>` — writes a stable JSON payload including per-target metrics and `overall` aggregates.
  - `tests/test_evaluation/test_smart_feature_selector.py` (19 tests) and `tests/test_evaluation/test_backtest_json_output.py` (1 test) cover the manifest contract, end-to-end selector, JSON serialization, and feature schema round-tripping.
- Transformer M tier now uses `seq_len=20` (up from 10), matching L tier's context window.
- Both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) use zero-padding for short players instead of skipping them.
- Training presets and feature engineering remain stable; 26 feature groups in the `full` preset (19 original + 4 lifecycle: injury_risk, aging_curve, kan_aging, skill_development + 3 season-context: season_phase, team_motivation, postseason_context). The `small` preset enables 6 groups (rolling, efficiency, momentum, pace, opponent_strength, archetype) and disables the Transformer.
- All six target stats exported from report generator and loaded correctly by projection loader.
- Training-to-simulation artifact contract is enforced at both boundaries.
- Dead code in `GameSimulator` has been removed.
- The evaluation module now provides quantitative backtest metrics for drift detection and weight optimization.
- **New: Contracts layer wiring in production seams (2026-06-04)**:
  - `src/data/schedule_scraper.py` now calls `normalize_schedule_frame(...)` on every read path (cached schedule hit, fresh API, cache fallback, season cache). Empty frames are skipped from normalization.
  - `src/simulation/season_simulator.py` converts the schedule frame to `ScheduleGame` dataclasses via `schedule_rows_to_games(...)` before iterating matchups (both ThreadPoolExecutor and sequential paths).
  - `src/query/projection_loader.py::ProjectionLoader.load_projections` calls `validate_projection_frame(...)` on every load and re-raises `ProjectionSchemaContractError` typed. Legacy CSVs missing distribution or `DATA_QUALITY` columns are rejected.
  - `src/simulation/report_generator.py::ReportGenerator.export_player_projections` writes the strict 6-stat x 8-column schema (mean, P10, P50, P90, STD, SKEW, ZERO_PROB, LAMBDA per stat, plus `DATA_QUALITY`) and calls `validate_projection_frame(...)` on the resulting DataFrame before writing the CSV.
  - `src/models/model_manager.py::ModelManager.predict_player_stats` now calls `load_expected_feature_cols(models_dir)` and `align_feature_frame(df, expected_cols)` before the leakage-safe selector runs, so reordered or extra-column inference frames are coerced to the trained layout before any prediction is computed.
- **New: Residual conformal interval calibration (Ticket 4, 2026-06-12)**:
  - `calibrate_residual_intervals.py` reads `data/evaluation/residual_training.parquet` and writes per-stat conformal interval artifacts to `models/calibration/`.
  - `src/correction/calibration.py` builds quantile half-widths (`q80`, `q90`, `q95` by default) from absolute residual errors, preferring `CORRECTED_PREDICTION` / `CORRECTED_ERROR` when present and falling back to `abs(ERROR)`.
  - `src/correction/interval_store.py` loads `models/calibration/*_intervals.json` non-fatally; missing artifacts disable interval output without breaking prediction.
  - `src/correction/confidence_scorer.py` maps interval width, `DATA_QUALITY`, minutes-confidence signals, and residual-model availability into `HIGH` / `MEDIUM` / `LOW` / `NO_EDGE`.
  - `ModelManager.predict_player_stats(..., include_confidence=True)` appends `{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, and `{STAT}_CONFIDENCE_SCORE` when calibration artifacts are loaded. Default calls remain backward-compatible.
   - `GameSimulator` requests confidence-aware batch predictions and carries interval/confidence metadata into `player_averages`; `ReportGenerator.export_player_projections()` writes the new columns for all six stats and validates them before saving.
   - Projection CSV contract now requires all six stats x 14 columns: the previous 8 projection/distribution columns plus 80/90 calibrated interval bounds, confidence label, and confidence score.
- **New: Fast training diagnostic mode** (2026-06-14):
   - `python train.py --diagnose --stop-after <stage>` runs staged checks without full model training.
   - Supported stages: `preflight`, `data_load`, `feature_engineering`, `feature_selection`, `prepare_data`, `artifact_check`.
   - Each stage prints `[TRAIN-DIAG] START/OK/FAILED/SKIP` markers with exception type, message, and full traceback on failure.
   - `--diagnose` alone runs all pre-training stages and stops before model training.
   - `--diagnose --stop-after artifact_check` validates existing `models/` artifacts using the resolved preset's `transformer_enabled` setting.
   - `--feature-selection off --stop-after feature_selection` prints SKIP and still honors the stop point.
   - After data load and feature engineering, diagnostic mode prints row counts, column counts, and target-column presence.
   - `src/training/diagnostics.py` provides `diagnostic_stage`, `diagnostic_noop`, `DiagnosticStop`, `DiagnosticStageFailed`, and summary helpers.
   - Uses custom exceptions instead of `sys.exit()` in library code.
- **New: Residual correction monitoring system (Ticket 6, 2026-06-18)**:
   - `monitor_residual_corrections.py` — CLI entry point that loads prediction history and produces per-target HELPING/NEUTRAL/HURTING status.
   - `src/evaluation/residual_monitor.py` — `ResidualMonitor`: pure evaluation logic with data-quality breakdown, confidence breakdown, rolling-window status (7/14/30 day windows + season-to-date), and actionable recommendations (KEEP_ENABLED / DISABLE_CORRECTION / NEUTRAL_REVIEW / INSUFFICIENT_DATA).
   - `src/evaluation/residual_report.py` — JSON/CSV report writer with atomic temp-file writes and strict NaN-free JSON output.
   - `config/default.yaml` — `residual_monitoring:` config block with thresholds (min_rows=500, helping=1%, hurting=-1%, neutral_band=±1%), rolling windows (7/14/30 days), and fallback input paths.
   - Full test suite: `tests/test_evaluation/test_residual_monitor.py` — 47 tests covering threshold logic, metric calculation, status labels, data-quality/confidence breakdowns, rolling windows, aggregation/recommendations, report writing, console rendering, validation, NaN routing, strict JSON safety, CLI resolution, and partial corrected-prediction fallback.
- **New: Confidence adjustment helpers (2026-06-18)**:
   - `src/query/confidence_adjustment.py` — `get_projection_value`, `get_confidence_label`, `get_confidence_score`, `std_from_interval`, `damp_probability` for safer over/under recommendations.
- **New: Projection loader tests (2026-06-18)**:
   - `tests/test_query/test_projection_loader.py` — 14 tests covering core/full CSV loading, corrected projections, interval fields, confidence labels, interval validation, non-numeric column rejection, and blank confidence handling.

## What Partially Works

- Simulation stack uses typed dataclasses and phase simulation, but still depends on volatile third-party scrapers.
- Ensemble weight optimization framework is implemented and test-covered, but needs live backtest runs against real completed games to prove the optimization loop end-to-end.
- Drift detector is implemented but needs a rolling window of live backtest results to establish baseline statistics.
- Transformer validation inference uses eager-safe path; live CUDA smoke test still pending.
- Schedule scraping and season simulation are implemented; upstream availability remains a risk.
- Lifecycle aging curves and KAN precomputation are implemented but need live validation with real player bio data.
- Nexus multi-modal model is implemented and import-tested but not yet wired as the active training path.
- GPU feature engineering (cuDF) is implemented with CPU fallback but needs GPU-equipped runtime validation.
- Copula simulation and variance optimization are implemented and import-tested but need live validation against real forecast data.
- Residual interval calibration is implemented and unit-tested, but needs a real `data/evaluation/residual_training.parquet` calibration run to populate `models/calibration/` and verify empirical coverage on current residuals.
- Smart per-target feature selection is implemented and unit-tested but needs live validation on real data: confirm the per-target selected lists converge, that the manifest JSON is loaded by the pipeline, and that downstream MAE improves (or at minimum doesn't regress) versus the canonical `self.feature_cols` list.
- WeightStore bootstrap in `ModelManager` is non-fatal — if no versioned weights exist, the legacy `blend_weights.pkl` blend is used. Verify the bootstrap path actually fires on a real training run with no manual `optimize_weights.py` invocation.

## What Is Broken Or Very Likely Broken

- No currently known critical bugs. The six training-stopping bugs, dead code, legacy scraper regressions, and the 2026-06-12 bug-fix batch (KB-022 through KB-032) have all been fixed.
- Upstream scraper availability remains the primary operational risk — but degraded inputs are now surfaced explicitly.

## Current Limitations

- Ensemble optimizer and drift detector need live-data validation (backtest against real completed games).
- No checked-in trained models are present in this workspace beyond cache directories.
- The repo uses many local file contracts rather than strong typed interfaces between phases — though the simulation layer is moving toward typed dataclasses.
- `simulate_season.py` carries a run-level input health summary and exits non-zero for hard schedule failures, but still needs a live smoke test against current upstream sources.
- Feature-engineering cache (DR-034, 2026-06-19): the first call after any input-data, FE-config, or external-file change is still a cold full compute; only repeated identical calls are served from the parquet cache. Each distinct cache entry is ~36–50MB at full scale.

## Active Risks

- The main scraper risk is upstream drift; optional failures degrade runs visibly, schedule failures are hard-required.
- Artifact naming or schema drift is now guarded by training/runtime validation and versioned weight storage.
- The blend weight system now supports hot-reloading and versioning, but operators must be aware that `optimize_weights.py` writes to the new weight store format.

## Known Workarounds

- If lineup, injury, betting, or defense context scrapers fail, `GameSimulator` now continues in explicit degraded mode (or fails fast with `--strict`).
- Query users can query all six stats (PTS, REB, AST, STL, BLK, TOV) from exported projection CSVs. A `DATA_QUALITY` warning surfaces when a projection uses fallback data.
- `clear_cache.py` can reset generated state while preserving raw input CSVs.
- Feature engineering is now cached by default (DR-034, 2026-06-19): `DataPipeline` and `ModelManager` write/read `cache/training/*.parquet`. The cache key folds in the mtime/size of external files the feature groups read (`data/injury_history.csv`, `data/cache/aging_curves.csv`, `data/player_bios.csv`, `data/cache/kan_aging_outputs.csv`), so cached features auto-invalidate when those grow. To force a recompute: `python clear_cache.py --all --yes` or delete `cache/training/`.
- `optimize_weights.py --rollback N` can revert to previous weight versions if a retune degrades accuracy.
- If `player_bios.csv` is missing, aging features default to neutral (1.0 factor). Run `update_data.py` with `--interactive` to populate it.
- Delete `data/cache/aging_curves.csv` and `data/cache/kan_aging_outputs.csv` to force recomputation of lifecycle caches after retraining.
- Delete `data/cache/archetype_covariances.npz` to force recomputation of archetype correlation matrices for the copula engine.
- Delete `models/feature_selection_manifest.json` to force a fresh smart selection run on the next `train.py` invocation with `--feature-selection smart`. If the manifest is missing or invalid, the pipeline silently falls back to the canonical `feature_cols` list.
- `backtest.py --json-output <path>` writes machine-readable metrics for downstream tooling (e.g. `optimize_weights.py` data prep, dashboards). The `targets` map and `overall` aggregates are the stable contract.
- Legacy projection CSVs that omit distribution columns (P10/P50/P90/STD/SKEW/ZERO_PROB/LAMBDA per stat) or the `DATA_QUALITY` column are no longer loadable; `ProjectionLoader.load_projections` raises `ProjectionSchemaContractError` instead of falling back to defaults. Re-run `simulate_season.py` after upgrading to regenerate the CSVs in the strict schema.
- Projection CSVs generated before Ticket 4 also lack `{STAT}_INTERVAL_80_*`, `{STAT}_INTERVAL_90_*`, `{STAT}_CONFIDENCE`, and `{STAT}_CONFIDENCE_SCORE`; regenerate projections after upgrading so `ProjectionLoader` sees the current strict schema.

## Immediate Priorities

1. Run `backtest.py --recent 14 --json-output data/backtest_baseline.json` to establish baseline metrics and feed the drift detector.
2. Run `optimize_weights.py --recent 14` to validate the self-optimization loop end-to-end with real completed games.
3. Run `optimize_variance.py --recent 14` against real data to validate the CRPS-based volatility multiplier optimization.
4. Run `python calibrate_residual_intervals.py --input data/evaluation/residual_training.parquet --output-dir models/calibration` after the residual dataset exists, then run a small simulation/query smoke test to confirm interval columns and confidence labels are populated from real artifacts.
4. Benchmark `python train.py --preset small` against the full preset on live CSVs to confirm iteration-speed improvement.
4. ~~Add a strict simulation mode for optional scraper degradation (CLI flag to fail-fast).~~ **DONE (DR-025, 2026-05-22)**
5. Run one live `train.py -> ModelManager.load_models() -> simulate_season.py` smoke test in a healthy local environment with real CSV inputs.
6. Collect real-data performance profile for the rolling feature path to compare against the synthetic benchmark.
7. ~~Run full test suite after lifecycle ML integration to update baseline test count.~~ **DONE (279 passed, 1 skipped)**
8. Validate lifecycle aging precomputation in `train.py` with real player bio data.
9. Run training smoke test with season-context features active to confirm `WL` and `SEASON_TYPE` columns propagate through the pipeline.
10. Run `train.py --feature-selection smart --selection-profile balanced` end-to-end on real CSVs, inspect `models/feature_selection_manifest.json`, and confirm the per-target CatBoost models train on the selected subsets (check `model_stack_metadata.pkl` `feature_selection_enabled` flag).
11. Verify the `ModelManager` WeightStore bootstrap path fires on a fresh process load (no manual `optimize_weights.py` invocation); confirm the training-time blend appears as the current version in `models/blend_weights/`.

## Testing Status

- Verified on 2026-06-12: full non-slow suite `368 passed, 1 skipped, 1 deselected` after the KB-022–KB-032 bug-fix batch.
- Verified on 2026-06-04: full suite `313 passed, 1 skipped` after smart feature selection + contracts layer + weight bootstrap (slow tests deselected).
- Verified on 2026-05-22: full suite `294 passed, 1 skipped` after calibration & probability upgrade.
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
- Whether the copula simulation (`run_copula_simulation()`) produces realistic correlated stat draws compared to independent Monte Carlo.
- Whether `optimize_variance.py` converges on volatility multipliers that improve CRPS over the default 1.0 values.
- Whether the smart per-target feature selector produces meaningful per-target lists on real data (i.e., the lists differ across stats and removing low-scoring features improves MAE).
- Whether the `max_accuracy` profile (group ablation + per-target pruning + shadow filter + time-stability check) converges in a reasonable time on the full historical dataset.
- The WeightStore bootstrap path is wired (DR-030, 2026-06-04): `ModelManager.load_models()` calls `WeightStore.load_current()` after the legacy blend is loaded; if a versioned blend exists, it overrides the legacy blend. Hot-reload through `set_weights()` is unchanged. Live validation against a fresh training run is still pending.
- `backtest.py --json-output` is the stable machine-readable contract; the payload schema is `{"targets": {...}, "overall": {...}}` and any new downstream tool should consume it via `src/evaluation/metrics.py::backtest_result_to_json_dict`. Live round-trip with the Beast Mode training script is still pending.
