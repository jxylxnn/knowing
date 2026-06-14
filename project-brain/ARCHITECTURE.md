# Architecture

## System Shape

- Architecture style: local batch/CLI pipeline with file-based persistence.
- No server process, no request router, no auth layer, no database.
- The repo is organized around six operational phases:
  1. data ingestion
  2. feature engineering, smart feature selection, and training
  3. simulation and projection export
  4. query-time probability lookup
  5. backtesting and ensemble weight optimization
  6. variance optimization and copula simulation
  7. residual correction and residual interval calibration

## Top-Level Execution Flow

### Flow 1: Data Refresh

1. `update_data.py` fetches player and team game logs.
2. It normalizes/merges them into pandas DataFrames.
3. It writes:
   - `data/nba_players.csv`
   - `data/nba_games.csv`

Important named functions in `update_data.py`:

- `fetch_player_logs`
- `fetch_team_logs`
- `fetch_player_logs_since`
- `fetch_team_logs_since`
- `merge_data`
- `save_data`

### Flow 2: Training

1. `train.py` loads `config/default.yaml`, resolves a named preset from `training_presets`, and selects the matching feature-group/Transformer configuration.
2. `src/preprocessing/data_loader.py` reads the raw CSVs and joins player/team context.
3. `train.py` applies the preset's feature-engineering rules through `src/preprocessing/feature_engineer.py`, including the `small` preset's reduced group set and optional `SEASON_ID`-based recent-history trimming.
   - When `--feature-ablation` is enabled, the Step 2 benchmark probe also goes through `build_feature_engineer(...)` so the ablation search stays compatible with older constructors that may reject `disable_groups`.
4. **Smart feature selection (optional, Step 2.5)** — when `--feature-selection smart` (or `feature_selection.enabled: true` in YAML) is set, `train.py` runs `SmartFeatureSelector` after the full feature set is built and before `TrainingPipeline` starts. The selector writes a `SelectionManifest` to `models/feature_selection_manifest.json` containing per-target feature lists. The manifest is then loaded by `TrainingPipeline.apply_feature_selection_manifest()` and consumed by `_feature_cols_for_target()` so each per-target CatBoost model trains on its own subset. Failure is non-fatal — the pipeline falls back to the canonical `self.feature_cols` list.
5. `src/training/pipeline.py` creates chronological train/validation/test splits.
5. The training pipeline fits:
   - CatBoost regressors per target — each target's training frame is sliced to its per-target feature list when smart selection is active
   - a Transformer sequence model
   - quantile models
   - blend weights
   - the Transformer path now keeps an eager inference copy for validation/runtime use and treats `torch.compile` as opt-in only
   - deterministic player archetype similarity features, computed from past-only rolling/context signals rather than from a separate learned clustering artifact
   - both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) now use zero-padding for players with fewer than `seq_len` context games instead of skipping them entirely; players with at least 1 game always produce training samples
6. Training writes the runtime artifact set to `models/`:
    - per-target CatBoost model files
    - per-target CatBoost metadata files
    - `attention_transformer.pkl` when the Transformer is enabled
    - `feature_schema.pkl`
    - `feature_cols.pkl`
    - `blend_weights.pkl`
    - `model_stack_metadata.pkl` recording whether the Transformer was active, the expected model count, and optionally the selected preset / feature groups. When smart selection ran, also records `feature_selection_enabled=True`, `feature_selection_target_specific`, `feature_selection_profile`, and `selected_features_by_target` for runtime auditability.
   - `feature_selection_manifest.json` (NEW, when smart selection ran) — per-target feature lists consumed by `TrainingPipeline.apply_feature_selection_manifest()` on subsequent training runs and by `FeatureSelector.select_features_for_target()` for downstream inference
7. Before `TrainingPipeline.train()` returns, it validates that the required runtime files exist and raises instead of reporting a false-success training run.
8. The Colab notebook wrapper `train_colab.ipynb` resolves the repo checkout separately from Drive-backed storage, launches `train.py` as a subprocess, captures stdout/stderr, and stops immediately on nonzero exit so the operator sees the real failure stage instead of only a wrapper exception.

Important named methods in this path:

- `DataLoader.load_data`
- `DataLoader.merge_datasets`
- `FeatureEngineer.create_features`
- `FeatureEngineer.create_features_chunked`
- `TrainingPipeline.prepare_data`
- `TrainingPipeline.train`
- `create_pipeline`

Important current contract:

- `src/training/pipeline.py`, `src/training/catboost_trainer.py`, `src/models/model_manager.py`, and `simulate_season.py` now share one filesystem contract for runtime artifacts.
- `ModelManager` validates that shared artifacts plus all six per-target CatBoost backbones and metadata exist before simulation loads any models.
- `src/models/model_manager.py` now requires `model_stack_metadata.pkl` as part of the shared runtime set so the training-side contract stays aligned with what the loader considers valid.
- `ModelManager._validate_blend_contract()` raises when blend weights expect a Transformer model that is missing or failed to load, preventing silently uncalibrated partial-blend predictions.
- `TrainingPipeline._validate_blend_contract()` enforces the same contract on the pipeline's `load_models()` path.
- `TrainingPipeline._predict_transformer_batch()` delegates validation inference through `TransformerWrapper.predict_batch()`, which defaults to the eager model path and safe SDPA backend controls on CUDA.
- `ModelManager.load_models()` now bootstraps `EnsembleWeights` from `WeightStore` after the legacy `blend_weights.pkl` is loaded. If a versioned weight entry exists, it overrides the legacy blend via `use_ensemble_weights()`. The bootstrap is non-fatal — if no `WeightStore` exists, the legacy blend is used. This ensures runtime uses data-driven weights even before `optimize_weights.py` is invoked. (DR-030, 2026-06-04: `TrainingPipeline._save_blend_weights()` writes the training-time blend to the versioned store so the bootstrap path is self-sufficient.)
- **Load-time feature alignment (2026-06-04):** `ModelManager.predict_player_stats` re-aligns the inference frame against `feature_cols.pkl` before the leakage-safe selector runs. `load_expected_feature_cols(models_dir)` reads the saved list and `align_feature_frame(df, expected_cols)` reorders / drops extra columns. An inference frame with missing or non-numeric features raises `FeatureSchemaContractError`.
- `TrainingPipeline._save_blend_weights()` writes the training-time blend both to the legacy `blend_weights.pkl` (for `validate_runtime_artifacts`) and to `WeightStore` (so the bootstrap path can pick it up).
- `train.py` now performs explicit preflight checks for writable model/cache directories and required raw CSV inputs before expensive work begins, which makes notebook and CLI failures easier to diagnose.
- `train.py` now also resolves training presets, passes the preset feature-group selection through `build_feature_engineer(...)`, and records the selected preset in `model_stack_metadata.pkl`.
- `train.py` runs optional smart feature selection between Step 2 (feature engineering) and Step 3 (training) — see Flow 2.5 below.
- `train.py` exposes `--feature-selection {off,smart}` and `--selection-profile {fast,balanced,max_accuracy}` CLI flags.
- `train.py` exposes `--diagnose` and `--diagnose --stop-after <stage>` for fast crash-stage diagnostics without running full training (see Flow 8 below).
- `train_colab.ipynb` should not infer `train.py` from the Drive models directory; it now searches the actual repo checkout first and keeps Drive-backed `data/` and `models/` separate from code location.

### Flow 2.5: Smart Per-Target Feature Selection (NEW — 2026-06-04)

1. `train.py` runs `SmartFeatureSelector` between Step 2 (feature engineering) and Step 3 (training) when `--feature-selection smart` (or `feature_selection.enabled: true`) is active.
2. The selector consumes the full feature frame plus the `group_columns` map produced by `FeatureEngineer`, then runs the stages gated by `ProfileConfig`:
   - **Group ablation** (`FeatureGroupAblator`): trains a baseline `HistGradientBoostingRegressor` plus a leave-one-out model per feature group. Records per-target `GroupScore` (MAE delta when the group is removed). Positive scores mean the group helps; negative scores mean it adds noise.
   - **Per-target pruning** (`SmartFeatureSelector._per_target_signals`): for each target, fits a fast `HistGradientBoostingRegressor` on an 80/20 temporal split, then computes:
     - `catboost_importance` (gain-based, 20% weight)
     - `permutation_importance` (column-shuffle score, 10% weight)
     - `backtest_gain` (broadcast from ablation group scores, 40% weight)
     - `stability` (correlation between gain importances on the first vs second half of training data, 25% weight)
     - `missingness_penalty` (share of NaN/zero rows in training frame, 5% weight, subtracted)
   - **Shadow filtering** (`ShadowFeatureFilter`): injects `SHADOW_RANDOM_NORMAL`, `SHADOW_RANDOM_UNIFORM`, `SHADOW_PERMUTED_TARGET` control columns and uses their median importance as a noise floor. Features scoring below the floor are dropped.
   - **Time-stability check** (only in `max_accuracy` profile): re-runs the per-target score on a second temporal sub-split and drops features whose score collapses.
3. The selector writes a `SelectionManifest` to `models/feature_selection_manifest.json` with per-target feature lists, dropped features, shadow-dropped features, and signal scores. The manifest is also persisted in `TrainingPipeline.feature_selection_manifest` for the current run.
4. `TrainingPipeline.apply_feature_selection_manifest()` parses the manifest and populates `self.target_feature_cols` (per-target feature lists) and `self.feature_selection_manifest` / `self.feature_selection_profile` (audit fields).
5. `TrainingPipeline._feature_cols_for_target(target)` returns the per-target subset when smart selection is active, or falls back to the canonical `self.feature_cols` list — preserving the original contract when no manifest is loaded.
6. Per-target CatBoost trainers in `_train_catboost_parallel` and the empty-data fallback path use `_feature_cols_for_target(target)` instead of `self.feature_cols`, so each model only sees its selected subset.
7. `_save_model_stack_metadata()` records `feature_selection_enabled`, `feature_selection_target_specific`, `feature_selection_profile`, and the per-target feature lists for runtime auditability.
8. Failure mode: if any stage throws (e.g., too few samples for stable fitting), the selector logs a warning and returns an empty manifest. The pipeline silently continues with the full feature set.

Important named classes/methods for this flow:

- `FeatureGroupAblator.run` — train baseline + LOO models, return `AblationReport`
- `ShadowFeatureFilter.run` — inject shadows, return `ShadowFilterResult`
- `SmartFeatureSelector.run` — orchestrate all stages, return `SelectionManifest`
- `ProfileConfig.resolve` — resolve named profile against YAML
- `SelectorConfig.from_config` — parse YAML `feature_selection` block
- `SelectionManifest.to_dict` / `load` — manifest serialization
- `SelectionManifest.save` — atomic JSON write to `models/feature_selection_manifest.json`
- `TrainingPipeline.apply_feature_selection_manifest` — load manifest into pipeline state
- `TrainingPipeline._feature_cols_for_target` — per-target feature list lookup with fallback
- `FeatureSelector.select_features_for_target` — build target-specific `FeatureSchema` from an allow-list

### Flow 3: Simulation

1. `simulate_season.py` loads the trained artifacts through `src/models/model_manager.py`.
2. It collects schedule data through `src/data/schedule_scraper.py`.
3. For each matchup, `src/simulation/game_simulator.py`:
   - gathers contextual inputs
   - records per-source input health for those contextual inputs
   - builds projected player stat lines
   - runs repeated simulation logic
   - returns game and player summaries
4. `src/simulation/season_simulator.py` orchestrates multi-game execution.
5. `src/simulation/report_generator.py` prints and exports CSVs under `data/sim_results/`.
6. `simulate_season.py` prints a run-level input health summary and treats schedule failures as hard-required.

**Simulation Refactor (2026-05-09):** The simulation layer has been refactored into modular, typed components:

- `src/simulation/phase_simulator.py` — Phase-by-phase Monte Carlo game loop, extracted from GameSimulator
- `src/simulation/archetype.py` — ArchetypeEngine infers player archetypes (heliocentric star guard, 3&D wing, etc.) from projection shape
- `src/simulation/role_sampler.py` — Samples role states (limited/normal/expanded/starter/bench/closer) with archetype-aware adjustments
- `src/simulation/sim_types.py` — Typed dataclasses (RoleSample, PhaseDefinition, GameEnvironment, PlayerProjection, TeamContext) replacing raw dicts
- `src/simulation/sim_cache.py` — JSON disk-cache mixin for simulator caching
- `src/simulation/stat_utils.py` — Shared statistical helpers (compute_mode, summary stats)

**Distribution enrichment (NEW — 2026-05-21):** `ReportGenerator.export_player_projections()` now enriches the exported CSV with distribution parameters via `_enrich_with_distributions()`:
- Uses `DistributionFitter` to derive Std, Skew, Zero-Prob, and Lambda from P10/P50/P90 quantile columns
- Falls back to mean-based defaults when quantile columns are not present
- Appends `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA` columns for all 6 stats

**Residual interval/confidence export (NEW — 2026-06-12):** `GameSimulator` requests `ModelManager.predict_player_stats_batch(..., include_confidence=True)`. When `models/calibration/` artifacts are loaded, the returned rows include `{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, and `{STAT}_CONFIDENCE_SCORE`. `GameSimulator._build_player_projection(...)` carries those fields through roster projection objects, `_run_simulation(...)` copies them into `player_averages`, and `ReportGenerator.export_player_projections()` writes the columns for every stat. If calibration artifacts are absent, the export still includes the columns with blank numeric bounds and `NO_EDGE` labels so the strict projection schema stays stable.

Important named methods in this path:

- `ModelManager.load_models`
- `ModelManager.predict_player_stats`
- `ModelManager.predict_player_stats_batch`
- `GameSimulator._safe_get_game_lines`
- `GameSimulator._safe_get_lineup`
- `GameSimulator._safe_get_injury_probs`
- `GameSimulator._safe_get_defensive_adjustments`
- `GameSimulator._simulate_matchup_reactive`
- `PhaseSimulator.simulate`
- `ArchetypeEngine.infer_archetype`
- `RoleSampler.sample`
- `ReportGenerator.export_to_csv`
- `ReportGenerator.export_player_projections`

### Flow 4: Query-Time Probability Lookup

1. `query_prob.py` or `src/query/interactive_cli.py` loads the latest projection export.
2. `src/query/projection_loader.py` merges projection rows with player metadata and optional defense context.
3. `src/query/query_parser.py` interprets user input.
4. `src/query/probability_calculator.py` calculates over/under probability, projection summaries, or Monte Carlo output.
   - **Copula simulation** (`run_copula_simulation()`): generates correlated multi-stat draws using archetype-conditioned empirical correlation matrices from `CovarianceCache` + Gaussian copula
   - Accepts optional `CovarianceCache` for archetype-specific 6x6 correlation matrices
   - Distribution parameters come from `DistributionFitter` (PDF from P10/P50/P90 quantiles) or the `_enrich_with_distributions` columns in the export CSV

Important named methods in this path:

- `ProjectionLoader.load_projections`
- `ProjectionLoader.find_player`
- `ProjectionLoader.get_recent_games`
- `ProjectionLoader.get_opponent_defense_profile`
- `ProbabilityCalculator.calculate_from_projection`
- `ProbabilityCalculator.run_monte_carlo_simulation`
- `ProbabilityCalculator.run_copula_simulation` — correlated multi-stat draws via archetype copula
- `ProbabilityCalculator.evaluate_calibration`
- `QueryParser.parse_query`

### Flow 5: Backtest and Weight Optimization (NEW — 2026-05-09)

1. `backtest.py` evaluates model predictions against historical completed games:
   - Loads trained models via `ModelManager`
   - Runs predictions for games in the specified date range
   - Compares against actual box scores from `data/nba_players.csv`
   - Computes per-stat MAE, RMSE, R², calibration error, and prediction interval coverage
   - Outputs `BacktestResult` with per-target `TargetMetrics`

2. `optimize_weights.py` retunes ensemble blend weights:
   - Runs backtesting to establish a baseline
   - Uses `scipy.optimize` to find optimal blend coefficients (13 parameters: 6 per-target CatBoost/Transformer ratios + 6 per-target intercepts + 1 CatBoost-MAE blend)
   - Validates candidates against a holdout window
   - Applies accept/verify gates to prevent regression
   - Writes versioned weights via `WeightStore` with atomic writes and rollback support
   - Hot-reloads weights into `ModelManager` at runtime

3. Drift detection (`src/evaluation/drift_detector.py`):
   - Tracks per-stat accuracy over rolling windows
   - Uses statistical process control: flags when rolling MAE exceeds 2σ above historical baseline
   - Distinguishes minor drift (retune weights) from major drift (retrain models)

Important named classes/methods in the evaluation subsystem:

- `BacktestRunner.run` — runs predictions against historical games
- `BacktestRunner.set_weights` — injects candidate blend weights for evaluation
- `EnsembleOptimizer.optimize` — scipy-driven weight optimization
- `EnsembleOptimizer.accept` — writes accepted weights to store
- `WeightStore.save` — atomic JSON write with versioning
- `WeightStore.rollback` — revert to previous version
- `WeightStore.load_current` — load active weights
- `DriftDetector.check` — evaluate if performance has drifted
- `DriftDetector.record_result` — append a backtest result to the rolling window

### Flow 6: Variance Optimization and Copula Simulation (NEW — 2026-05-21)

1. `optimize_variance.py` tunes context-specific volatility multipliers via CRPS:
   - Loads historical player-game data from `data/nba_players.csv`
   - Tunes 7 multipliers (B2B, Rookie, Blowout, Home, Away, Playoff, RestAdvantage) via scipy Nelder-Mead
   - Uses `calculate_empirical_crps()` as the objective function (lower CRPS = sharper + better-calibrated)
   - Returns optimal multipliers that adjust per-context std deviation

2. Distribution enrichment (`_enrich_with_distributions` on `ReportGenerator`):
   - Derives Std, Skew, Zero-Prob, Lambda for each stat from P10/P50/P90 quantile columns
   - Falls back to mean × 0.7/1.3 heuristic when quantile columns absent
   - Written into the exported `player_projections_*.csv` as `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`

3. Correlated copula simulation (`ProbabilityCalculator.run_copula_simulation()`):
   - Uses `CovarianceCache` (archetype-conditioned 6x6 correlation matrices from residual analysis)
   - Gaussian copula via Cholesky decomposition + standard normal CDF → uniform marginals
   - Per-stat inverse CDF (skew-normal for continuous stats, ZIP for count stats, normal fallback)
   - Returns a DataFrame with all 6 stat columns and `num_sims` rows of correlated draws

Important named classes/methods for this flow:

- `DistributionFitter.fit_from_quantiles` — extracts distribution params from P10/P50/P90
- `StatDistribution` — dataclass: mean, std, skew, zero_prob, lambda_param
- `CovarianceCache.build_and_save` — computes + caches archetype correlation matrices
- `CovarianceCache.get_correlation` — returns 6x6 matrix for an archetype
- `calculate_empirical_crps` — O(n log n) CRPS via Gini mean difference
- `ReportGenerator._enrich_with_distributions` — appends distribution columns to export CSV
- `ProbabilityCalculator.run_copula_simulation` — correlated multi-stat Monte Carlo

### Flow 8: Fast Training Crash-Stage Diagnostics (NEW — 2026-06-14)

1. `train.py --diagnose` runs the standard training flow but wraps each major stage in a `diagnostic_stage` context manager from `src/training/diagnostics.py`.
2. Each stage prints `[TRAIN-DIAG] START|OK|FAILED` markers with elapsed time.
3. On failure, the context manager prints the exception type, message, and full traceback, then raises `DiagnosticStageFailed`. `train.py` catches `DiagnosticStop`/`DiagnosticStageFailed` in `main()` and returns 0 or 1 respectively.
4. `--stop-after <stage>` causes an early clean exit (code 0) after the named stage completes.
5. Supported stages: `preflight`, `data_load`, `feature_engineering`, `feature_selection`, `prepare_data`, `artifact_check`.
6. `--diagnose` alone (no `--stop-after`) runs all stages through `prepare_data` and stops before model training.
7. `--diagnose --stop-after artifact_check` validates current `models/` artifacts via `validate_runtime_artifacts(ArtifactContract(...))` without loading data or running any training steps.
8. After data load and feature engineering, diagnostic mode prints row counts, column counts, and target-column presence summaries.

Important named classes/functions:

- `DiagnosticConfig` — dataclass: `enabled`, `stop_after`
- `diagnostic_stage(name, config)` — context manager that prints markers, catches exceptions, and optionally exits early
- `print_data_summary(merged_df, full_df)` — prints row/column/target summaries
- `print_selection_summary(manifest)` — prints per-target selected feature counts

### Flow 7: Residual Correction and Residual Interval Calibration (NEW — 2026-06-12)

1. `build_residual_dataset.py` creates `data/evaluation/residual_training.parquet` from walk-forward historical predictions.
2. `train_residual_models.py` trains per-target residual correction models under `models/residual/`.
3. `ModelManager.load_models()` loads residual correction models best-effort through `ResidualCorrectionModel` + `CorrectionApplier`; point predictions are corrected after base CatBoost/Transformer blending.
4. `calibrate_residual_intervals.py` reads the same residual dataset and writes per-stat conformal interval artifacts under `models/calibration/`.
5. `src/correction/calibration.py::ResidualIntervalCalibrator` computes absolute calibration error using this precedence:
   - `abs(ACTUAL - CORRECTED_PREDICTION)` when `CORRECTED_PREDICTION` exists
   - `abs(CORRECTED_ERROR)` when `CORRECTED_ERROR` exists
   - `abs(ERROR)` as the fallback
6. The calibrator writes global and context-aware buckets:
   - `GLOBAL`
   - `DATA_QUALITY_FULL`
   - `DATA_QUALITY_DEGRADED`
   - `HIGH_MINUTES_CONFIDENCE` / `LOW_MINUTES_CONFIDENCE` when a minutes-confidence column exists
   - `PLAYER_HIGH_VOLATILITY` / `PLAYER_LOW_VOLATILITY` from player-level mean absolute calibration error
7. `src/correction/interval_store.py::CalibrationIntervalStore` loads those JSON artifacts non-fatally; missing stat or bucket falls back to no interval / `GLOBAL` respectively.
8. `src/correction/confidence_scorer.py::ConfidenceScorer` turns interval width, `DATA_QUALITY`, minutes confidence, and residual-model status into `HIGH`, `MEDIUM`, `LOW`, or `NO_EDGE`.
9. `ModelManager.predict_player_stats(..., include_confidence=True)` appends interval/confidence keys after residual correction. Default callers get the legacy point-prediction shape.

Important artifact paths:

- `models/calibration/{stat}_intervals.json` — per-stat bucketed width/coverage artifact.
- `models/calibration/calibration_metadata.json` — run metadata, row counts, configured confidence levels, bucket coverage summaries.

## Major Subsystems And Ownership

### `config/` and `src/config/`

- Own runtime configuration.
- `config/default.yaml` contains the operational defaults.
- `src/config/config.py` maps YAML into typed config objects.
- `src/config/model_config.py` derives hardware-aware model sizing decisions.

Do not change casually:

- field names consumed by scripts and model/training code
- default directory paths for `data`, `models`, and cache locations

### `src/data/`

- Owns external data acquisition and scraping.
- Main modules:
  - `schedule_scraper.py`
  - `injury_scraper.py`
  - `lineup_scraper.py`
  - `betting_scraper.py`
  - `basketball_ref_scraper.py`
  - `rotowire_lineup_scraper.py`
  - `nba_defense_scraper.py`
- These modules are the highest-fragility part of the repo because they depend on third-party response formats and contain several observable regressions.

### `src/preprocessing/`

- Owns data loading, joining, and engineered feature creation.
- `data_loader.py` merges raw player/team inputs and creates the base frame used by training.
- `feature_engineer.py` is the coordination layer for modular feature groups.
- `features/` contains the main feature logic.
- The active feature-group stack assembled in `FeatureEngineer._build_groups()` is (19 groups, in dependency order):
  - `RollingFeatureGroup`
  - `EfficiencyFeatureGroup`
  - `MomentumFeatureGroup`
  - `ContextualFeatureGroup`
  - `FatigueFeatureGroup`
  - `MinutesConfidenceFeatureGroup`
  - `RestGameDensityFeatureGroup`
  - `MatchupFeatureGroup`
  - `OpponentStrengthFeatureGroup`
  - `PaceFeatureGroup`
  - `TeamRoleFeatureGroup`
  - `LineupStabilityFeatureGroup`
  - `InjuryAdjustedOpportunityFeatureGroup`
  - `TeammateUsageFeatureGroup`
  - `RecencyFormFeatureGroup`
  - `PlayerArchetypeFeatureGroup`
  - `DefensePositionFeatureGroup`
  - `TargetEncodingFeatureGroup`
  - `LeagueRankingFeatureGroup`

Key invariant:

- Training and inference rely on stable feature-column semantics. Adding or renaming columns without updating saved schema expectations is risky.
- `rolling.py` now materializes its wide rolling/efficiency/momentum outputs in temporary structures and appends them with a single concat per group to avoid pandas fragmentation.
- `archetype.py` computes hard labels plus soft similarities to a fixed playstyle template set, which means cold-start players can still be mapped to a nearby bucket without a separate clustering fit artifact.

### `src/training/`

- Owns model fitting, experiment logging, and training orchestration.
- `pipeline.py` is the active end-to-end training pipeline.
- `catboost_trainer.py` owns CatBoost training and per-target artifact persistence behavior.
- `presets.py` is the named preset boundary for the CLI: it defines the small/full stack shape, rolling-window defaults, optional recent-history trimming, and the feature-group allowlist used by `train.py`.
- `experiment.py` writes experiment summaries under `experiments/`.
- `feature_cache.py` contains reusable cache infrastructure but is not clearly wired into the active top-level training flow.

### `src/models/`

- Owns runtime model definitions and model loading.
- `transformer_model.py` implements the Transformer wrapper, checkpoint compatibility handling, and the eager-safe inference path used by validation/runtime callers.
- `model_manager.py` loads saved artifacts and exposes runtime prediction methods to the simulator.

Critical coupling:

- `model_manager.py` assumes specific artifact names exist in `models/`.
- Training now validates those names before success, and runtime loading fails loudly if the on-disk set is incomplete.
- Any training-side artifact rename or format change must still be coordinated here.
- `TransformerWrapper` keeps compile behind an explicit opt-in safety flag. Validation and runtime prediction use the eager model path, and on CUDA they try to force a math SDPA backend when the backend API is available.

### `src/simulation/`

- Owns matchup simulation, season-level orchestration, report generation, and adjustment logic.
- Important modules:
  - `game_simulator.py` — orchestration hub (heavily refactored, dead code removed)
  - `phase_simulator.py` — phase-by-phase Monte Carlo game loop (extracted from GameSimulator)
  - `season_simulator.py`
  - `report_generator.py`
  - `archetype.py` — ArchetypeEngine: infers player archetypes from projection shape
  - `role_sampler.py` — role state sampling with archetype-aware adjustments
  - `sim_types.py` — typed dataclasses (RoleSample, PhaseDefinition, GameEnvironment, PlayerProjection, TeamContext)
  - `sim_cache.py` — JSON disk-cache mixin
  - `stat_utils.py` — shared statistical helpers (compute_mode, summary stats)
  - `minutes_predictor.py`
  - `context_aware_adjustments.py`
  - `player_correlation_engine.py`
  - `four_factors_engine.py`
|  - `error_calibration.py`

Fragility notes:

- `game_simulator.py` is a high-coupling orchestrator with many dependency points.
- Dead legacy simulation code has been removed; the active path is exclusively `_simulate_matchup_reactive`.
- The refactored components (phase_simulator, archetype, role_sampler) are now independently testable.
- Scraper-backed optional context now flows through a shared input-health contract in `src/simulation/input_health.py`.
- Optional context failures degrade a run visibly; required schedule failures are treated as hard failures.
- **Strict mode** (`--strict` CLI flag on `simulate_season.py`): halts execution when any optional InputHealth record reports `failed` or `fallback`. Passed through `GameSimulator(strict_mode=True)` and `SeasonSimulator(strict_mode=True)`.
- **Data quality column** (`DATA_QUALITY`): exported in `player_projections_*.csv` as `FULL`, `DEGRADED_FALLBACK`, or `DEGRADED_MISSING`. Surfaces a CLI warning in the query layer via `ProjectionLoader.find_player()`.

### `src/evaluation/` (NEW — 2026-05-09)

- Owns backtesting, ensemble weight optimization, drift detection, and weight versioning.
- Important modules:
  - `metrics.py` — `BacktestResult`, `TargetMetrics`, `compute_target_metrics` dataclasses (MAE, RMSE, R², MAPE, calibration, interval coverage), plus `calculate_empirical_crps()` for probabilistic forecast evaluation (O(n log n) Gini mean difference approximation) and `backtest_result_to_json_dict()` for machine-readable JSON serialization (used by `backtest.py --json-output`)
  - `backtest_runner.py` — `BacktestRunner`: runs model predictions against historical completed games, compares against actual box scores
  - `ensemble_optimizer.py` — `EnsembleOptimizer`: 13-parameter scipy.optimize-based weight tuner with accept/verify gates
  - `weight_store.py` — `WeightStore` + `EnsembleWeights` + `TargetBlend`: versioned JSON storage replacing opaque binary `blend_weights.pkl`, atomic writes, rollback
  - `drift_detector.py` — `DriftDetector`: statistical process control tracking per-stat accuracy over rolling windows; flags when rolling MAE exceeds 2σ above baseline
  - **`feature_group_ablation.py` (NEW — 2026-06-04)** — `FeatureGroupAblator`: trains baseline + leave-one-out `HistGradientBoostingRegressor` per feature group, computes per-target MAE deltas. Backbone of the `backtest_gain` signal in smart feature selection.
  - **`shadow_feature_filter.py` (NEW — 2026-06-04)** — `ShadowFeatureFilter`: injects random control columns (`SHADOW_RANDOM_NORMAL`, `SHADOW_RANDOM_UNIFORM`, `SHADOW_PERMUTED_TARGET`) and uses their median importance as a noise floor for pruning real features.
  - **`smart_feature_selector.py` (NEW — 2026-06-04)** — `SmartFeatureSelector` + `ProfileConfig` + `SelectorConfig` + `SelectionManifest` + `TargetSelection`: combines 5 signals (backtest_gain, stability, catboost_importance, permutation_importance, missingness_penalty) into a per-target final score and writes a manifest to `models/feature_selection_manifest.json`. Profiles (`fast` / `balanced` / `max_accuracy`) gate which stages run.
- Key contracts:
  - `BacktestRunner` depends on `ModelManager` for predictions and `DataLoader` for actual box scores
  - `EnsembleOptimizer` owns a `BacktestRunner` internally and creates/validates candidate weight configs
  - `WeightStore` writes to a versioned directory (default: `models/blend_weights/`) with a `current.json` pointer. `ModelManager` resolves the same path under `<models_dir>/blend_weights` at load time
  - `ModelManager` now accepts hot-reloadable `EnsembleWeights` objects via `set_weights()`
  - The weight store format is human-readable JSON, not opaque binary pickle

### `src/contracts/` (NEW — 2026-06-04)

- Owns inter-step artifact contract validation. Validates that artifacts produced by one pipeline step satisfy the schema expectations of the next.
- Important modules:
  - `artifacts.py` — `ArtifactContract` + `validate_runtime_artifacts()`: checks that the `models/` directory contains the required CatBoost backbones, per-target metadata, `feature_schema.pkl`, `feature_cols.pkl`, `blend_weights.pkl`, `model_stack_metadata.pkl`, and (when `transformer_required=True`) `attention_transformer.pkl`. Also validates `model_stack_metadata.pkl`'s `targets` field matches the canonical 6-stat set. Optional `max_age_hours` enforces staleness.
  - `features.py` — `FeatureSchema` contract used by `FeatureSelector` and the trainer to align training-time and inference-time feature layouts.
  - `projections.py` — `validate_projection_csv()` checks the `player_projections_*.csv` schema, including the `DATA_QUALITY` column added by DR-025.
  - `schedule.py` — schedule input contract for the simulator.
  - `errors.py` — `ContractError` base + `ArtifactContractError`, `FeatureSchemaContractError`, `ProjectionSchemaContractError`, `ScheduleContractError` typed exceptions.
- Standalone entry: `check_contracts.py` (root) — CLI for validating the artifact contract and projection CSV between pipeline steps. Accepts `--models-dir <path>`, `--projection-csv <path>`, `--transformer-required`.
- Both `train.py` and `simulate_season.py` invoke contract validation at startup; the optimizer/selector/sim stack is swappable behind this seam.
- **Wired-in production call sites (DR-031, 2026-06-04):** the contracts layer is no longer just the seam — it is invoked at every inter-step boundary:
  - `src/data/schedule_scraper.py::ScheduleScraper` calls `normalize_schedule_frame(...)` on every read path (cached schedule hit, fresh API, cache fallback, season cache). Empty frames are skipped.
  - `src/simulation/season_simulator.py::SeasonSimulator.simulate_season` converts the schedule frame to `ScheduleGame` records via `schedule_rows_to_games(...)` before iterating matchups (both ThreadPoolExecutor and sequential paths).
  - `src/query/projection_loader.py::ProjectionLoader.load_projections` calls `validate_projection_frame(...)` on every load and re-raises the typed `ProjectionSchemaContractError`.
  - `src/simulation/report_generator.py::ReportGenerator.export_player_projections` calls `validate_projection_frame(...)` on the assembled DataFrame before writing the CSV. The CSV schema is strict: 6 stats x 8 columns (`{STAT}`, `{STAT}_P10`, `{STAT}_P50`, `{STAT}_P90`, `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`) plus `DATA_QUALITY`.
  - `src/models/model_manager.py::ModelManager.predict_player_stats` calls `load_expected_feature_cols(models_dir)` and `align_feature_frame(df, expected_cols)` before the leakage-safe selector runs, so reordered or extra-column inference frames are coerced to the trained layout before any prediction is computed.
  - `train.py` calls `validate_runtime_artifacts(ArtifactContract(...))` at the bottom of the training flow so a contract violation that materializes during training is caught before the next downstream step begins.

### `src/lifecycle/` (NEW — 2026-05-22)

- Owns player aging, career trajectory, and injury risk computation — the bio-mechanical layer.
- Important modules:
  - `aging_model.py` — `BIanusAgingModel`: position-specific Bayesian aging curves (B-Ianus). Separates development (pre-peak) from decline (post-peak). MAP-estimated peak ages by position (PG: 28.5, SG: 27.8, SF: 27.5, PF: 27.0, C: 26.5). Cached to `data/cache/aging_curves.csv`.
  - `kan_age_model.py` — `KANAgeModel`: Kolmogorov-Arnold Network for nonlinear age curves. Precomputed on CPU and cached to `data/cache/kan_aging_outputs.csv`. Always runs on CPU to avoid GPU contention.
- Precomputed by `train.py` at startup (before feature engineering so caches exist for feature groups to load). Both steps are non-fatal — missing bio data defaults all aging features to neutral (1.0 factor).
- Key rule: KAN always runs on CPU (`device='cpu'`) to avoid CUDA context contention with CatBoost/Transformer.

### `src/models/nexus_model.py` (NEW — 2026-05-22)

- Nexus Multi-Modal Architecture — unified deep-learning model replacing the 6 independent CatBoost models and isolated Transformer with a single end-to-end network.
- Architecture:
  - *Temporal Backbone* — Mamba-2-style SSM (simplified pure-PyTorch implementation with fallback when CUDA kernels unavailable)
  - *Tabular Backbone* — FT-Transformer for scalar / contextual features
  - *Relational Backbone* — Lightweight Graph Attention Network (GAT) for lineup synergy
  - *Fusion & Copula Head* — concatenates backbone representations and returns a 6-dimensional mean vector + 6x6 Cholesky-decomposed covariance matrix (guarantees positive semi-definite)
- Loss: `GaussianNLLLoss` in `src/training/nexus_loss.py` — multivariate Gaussian NLL with Cholesky covariance. Natively penalizes mathematically impossible stat combinations (negative variances, non-PSD correlation matrices).
- Status: implemented and import-tested; not yet wired as the active training path. CatBoost + Transformer remains the active stack.

### `src/preprocessing/feature_engineer_gpu.py` (NEW — 2026-05-22)

- GPU-accelerated feature engineering using NVIDIA cuDF + Apache Arrow zero-copy export.
- Mirrors the public API of `FeatureEngineer` but offloads heavy groupby/rolling primitives to the GPU.
- Complex groups relying on Python loops or NumPy/scipy execute on CPU after converting the relevant partition back to pandas.
- Transparently falls back to CPU engine when cuDF not installed or CUDA unavailable.
- Activated automatically when `FeatureEngineer(use_gpu=True)` is set — no CLI flag needed.

### Feature Groups — Lifecycle & Bio-Mechanical (NEW — 2026-05-22)

Four new feature groups in `src/preprocessing/features/`, wired into the 19 existing groups:

- `injury_risk.py` — `InjuryRiskFeatureGroup`: METIC-style workload + injury history signals. Reads from the persistent `data/injury_history.csv` produced by `InjuryHistoryLogger`. Outputs: `INJURY_RISK_CAREER_COUNT`, `INJURY_RISK_LAST_90D`, `INJURY_RISK_LAST_30D`, etc.
- `aging_curve.py` — `AgingCurveFeatureGroup`: B-Ianus Bayesian model features. Outputs: `AGING_PLAYER_AGE`, `AGING_PEAK_AGE_EST`, `AGING_PRE_POST_PEAK`, `AGING_CURVE_FACTOR`, etc.
- `kan_aging.py` — `KANAgingFeatureGroup`: KAN nonlinear age features. Outputs: `KAN_AGE_NONLIN_FACTOR`, `KAN_AGE_INFLECTION_AGE`, `KAN_AGE_VOLATILITY`.
- `skill_development.py` — `SkillDevelopmentFeatureGroup`: growth velocity metrics (year-over-year stat improvements). Outputs: `SKILL_DEV_PTS_VELOCITY`, `SKILL_DEV_EFF_VELOCITY`, `SKILL_DEV_REB_VELOCITY`, `SKILL_DEV_AST_TOV_TREND`, `SKILL_DEV_YOUTH_BOOST`.

Total feature groups: 23 (19 original + 4 lifecycle).

### Feature Groups — Season Context (NEW — 2026-05-23)

Three new feature groups in `src/preprocessing/features/`, wired into the 23 existing groups:

- `season_phase.py` — `SeasonPhaseFeatureGroup`: early-season ramp-up and trade resets. Outputs: `DAYS_SINCE_SEASON_START` (capped at 30 days to isolate early-season effect), `IS_SEASON_OPENER` (first 2 days), `GAMES_WITH_CURRENT_TEAM` (consecutive games since last trade, resets on team change), `IS_RECENT_TRADE` (≤5 games with new team).
- `team_motivation.py` — `TeamMotivationFeatureGroup`: late-season tanking/load management signals. Outputs: `TEAM_CUMULATIVE_WIN_PCT` (shifted to prevent leakage), `IS_LATE_SEASON` (March+), `IS_TANKING_PROXY` (late + win pct < 0.35), `IS_PLAYOFF_LOCK_PROXY` (late + win pct > 0.65).
- `postseason_context.py` — `PostseasonContextFeatureGroup`: playoff detection. Outputs: `IS_PLAYOFF_GAME` (parsed from `SEASON_TYPE` or `GAME_TYPE`), `PLAYOFF_PACE_PRIOR` (0.95 historical prior for playoff pace drop, model learns exact coefficient).

All three groups use the batched-assembly pattern (`_concat_new_columns`) to avoid pandas fragmentation warnings. They are in the `full` training preset only (not `small`).

### Rest Density Cap

`DAYS_SINCE_LAST_GAME` in `RestGameDensityFeatureGroup` is capped at 14 days to prevent off-season gaps (180+ days) or All-Star breaks from creating infinite rest outliers. Anything beyond 14 days is functionally identical for physical recovery.

Total feature groups: 26 (19 original + 4 lifecycle + 3 season context).

### Phase-Aware Drift Detection

`DriftDetector` now supports separate baselines for regular season and playoffs:
- `record()`, `detect()`, and `record_and_detect()` accept a `phase` parameter (`'REGULAR'` or `'PLAYOFF'`).
- When phase is not provided, it's auto-inferred from the date (Apr 15–Jun 20 = playoff).
- Prevents false "major drift" alerts when the playoffs naturally lower scoring/pace.
- Falls back to full history when no phase-specific data exists yet.

### Data Enrichment — Player Bios & Injury History

- `src/data/player_bio_scraper.py` — `PlayerBioScraper`: fetches birthdate, position, height, weight from NBA API `commonplayerinfo`. Caches results to `data/player_bios.csv`. Called from `update_data.py` via `enrich_with_player_bios()`.
- `src/data/injury_history_logger.py` — `InjuryHistoryLogger`: persists injury events across runs into `data/injury_history.csv`. Deduplicates by (PLAYER_ID, DATE, INJURY_TYPE). Called from `update_data.py` via `log_injury_snapshot()`.
- `update_data.py` also performs Parquet dual-write (`nba_players.parquet`, `nba_games.parquet`) for GPU-direct storage reads.

## Input Health Contract

- Simulation-facing scraper status records now include:
  - `source_key`
  - `status` in `success`, `fallback`, `failed`, or `disabled`
  - `required`
  - `message`
  - `details`
- Required input:
  - schedule retrieval
- Optional inputs:
  - betting lines
  - lineup context
  - injury availability
  - defensive adjustments
- `GameSimulator` stores the summarized result under `result['metadata']['input_health']`.
- `SeasonSimulator` aggregates per-game source health into `last_run_summary`.
- `simulate_season.py` surfaces that summary directly to the operator and exits non-zero when schedule health fails.

### `src/query/`

- Owns CLI parsing, projection lookup, and probability math.
- Important modules:
  - `interactive_cli.py`
  - `projection_loader.py`
  - `probability_calculator.py` — also accepts optional `CovarianceCache` for copula simulation
  - `query_parser.py`
  - **`distribution_fitter.py` (NEW)** — `DistributionFitter`: derives Mean/Std/Skew/Zero-Prob/Lambda from P10/P50/P90 quantile outputs. Used by `ReportGenerator._enrich_with_distributions()` and `ProbabilityCalculator.run_copula_simulation()`.
  - **`empirical_covariance.py` (NEW)** — `CovarianceCache`: computes, caches, and retrieves archetype-conditioned 6x6 empirical correlation matrices from residual analysis. Persisted to `data/cache/archetype_covariances.npz`.

Important boundary:

- Query behavior depends on the shape of exported simulation CSVs. If report exports omit a stat, the query layer silently degrades.

### `tests/`

- Provides unit and integration-style checks for many core internals.
- Coverage is decent for preprocessing, model wrappers, parts of training, and query math.
- Coverage is weak for scraper health, schedule retrieval, and the full train-to-sim artifact contract.
- High-value starting points for future test work include:
  - `tests/test_models/test_model_manager.py`
  - `tests/test_preprocessing/test_feature_engineer.py`
  - `tests/test_pipeline/test_data_pipeline.py`
  - `tests/test_simulation/test_game_simulator.py`
  - `tests/test_query/test_probability_calculator.py`
  - `tests/test_query/test_interactive_cli.py`
  - `tests/test_training/test_training_pipeline_colab.py`
  - `tests/test_training/test_nn_trainer.py`
  - `tests/test_models/test_transformer_model.py`

## File-Based Storage Contracts

### Required Raw Inputs

- `data/nba_players.csv`
- `data/nba_games.csv`
- `data/player_bios.csv` (optional — generated by `update_data.py`; aging features fall back to neutral when missing)
- `data/injury_history.csv` (optional — generated by `update_data.py`; injury risk features fall back to near-zero when missing)

### GPU-Direct Storage (NEW)

- `data/nba_players.parquet` / `data/nba_games.parquet` — Parquet dual-write for cuDF GPU-direct reads. Written alongside CSV by `update_data.py`.

### Expected Trained Artifacts

- Per-target CatBoost model files such as:
  - `models/pts_catboost.cbm`
  - `models/reb_catboost.cbm`
  - `models/ast_catboost.cbm`
  - `models/stl_catboost.cbm`
  - `models/blk_catboost.cbm`
  - `models/tov_catboost.cbm`
- Associated metadata files such as `models/pts_metadata.joblib`
- `models/attention_transformer.pkl`
- `models/feature_schema.pkl`
- `models/feature_cols.pkl`
- `models/blend_weights.pkl`
- `models/model_stack_metadata.pkl`
- `models/blend_weights/` — versioned JSON weight store (replaces opaque binary `blend_weights.pkl`). Contains version-numbered weight files (`v0001.json`, `v0002.json`, …) + `current.json` pointer. Managed by `WeightStore`. `ModelManager` now bootstraps from this store at load time when a `current.json` exists, so the runtime picks up data-driven weights even before `optimize_weights.py` is run. (Earlier brain revisions referred to this as `data/weights/` — the actual default is `models/blend_weights/`, per `WeightStore.__init__(store_dir="models/blend_weights")` and `ModelManager._load_models`.)
- `models/feature_selection_manifest.json` (NEW — 2026-06-04) — per-target feature lists produced by `SmartFeatureSelector`. Consumed by `TrainingPipeline.apply_feature_selection_manifest()` on subsequent training runs and by `FeatureSelector.select_features_for_target()` for downstream inference.

### Generated Outputs

- `data/sim_results/sim_results_<timestamp>.csv`
- `data/sim_results/player_projections_<timestamp>.csv` — now includes `DATA_QUALITY` and distribution enrichment columns (`{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA` for all 6 stats)
- Experiment JSON files under `experiments/<experiment_name>/`

### Cache Layers

- `data/cache/` for scraped or derived data.
- `data/cache/archetype_covariances.npz` — archetype-conditioned 6x6 correlation matrices computed by `CovarianceCache` from residual analysis.
- `data/sim_cache/` for simulation-time cache data.
- `cache/` and `cache/training/` for more general caching.

## Data And Prediction Invariants

- Canonical target stats inside model/training code are uppercase:
  - `PTS`
  - `REB`
  - `AST`
  - `STL`
  - `BLK`
  - `TOV`
- Query-layer stat names are lowercase aliases that map back to those targets.
- `FeatureSchema` and `feature_cols.pkl` exist to keep training-time and inference-time feature layouts aligned.
- The simulator expects recent player/team context that is structurally compatible with the trained feature schema.
- The model manager is the intended single loader for runtime prediction artifacts.

## External Dependency Flow

### Training-Time Dependencies

- `nba_api` is the primary historical data source.
- CatBoost and PyTorch are training backends.

### Simulation-Time Dependencies

- Schedule, lineup, injury, defense, and betting information are gathered from several third-party sources.
- Most simulation scrapers are wrapped in fallback code so that a broken scraper does not necessarily crash the full run.

Tradeoff:

- This keeps the CLI usable when sources drift, but it also hides real failures and can degrade projection quality without obvious operator feedback.

## Performance-Sensitive Areas

### Feature Engineering

- `src/preprocessing/features/rolling.py` repeatedly inserts columns into pandas DataFrames and emits heavy fragmentation warnings during tests.
- This is likely a material cost center on full historical datasets.

### Training

- CatBoost training is parallelized per target.
- Transformer training introduces PyTorch overhead and environment sensitivity.
- Artifact loading and feature reconstruction can become expensive if schema handling drifts.

### Simulation

- `game_simulator.py` loops through players and repeated simulation draws.
- Every extra external scrape or repeated feature lookup in this path multiplies runtime quickly.

## Security And Safety Notes

- There is no auth or user-secrets subsystem in the repo.
- Safety concerns are mostly operational rather than application-security oriented.
- Positive example: `game_simulator.py` uses JSON cache files rather than arbitrary pickle for some simulation cache data, reducing unsafe deserialization exposure.
- Main safety risks are:
  - brittle external scraping
  - silent fallback behavior masking bad inputs
  - accidental artifact/schema drift causing incorrect predictions

## High-Risk Couplings

- `train.py` -> `src/training/pipeline.py` -> artifact files in `models/` -> `src/models/model_manager.py`
- `src/simulation/report_generator.py` export schema -> `src/query/projection_loader.py`
- `src/data/schedule_scraper.py` output shape -> `src/simulation/season_simulator.py` and CLI flags in `simulate_season.py`
- `src/preprocessing/feature_engineer.py` feature set -> `feature_schema.pkl` -> runtime model inference

## Legacy Or Drift-Prone Areas

- `src/pipeline/training_pipeline.py` is a compatibility shim, not the real implementation.
- `src/training/__init__.py` uses lazy imports to avoid heavy module import costs.
- `src/services/` is present but empty.
- `src/evaluation/` contains only `__init__.py`.
- Some scraper modules appear partially duplicated or alternate-path legacy code rather than the one true active implementation.

## Architecture Risks To Keep In Mind

- Several scraper classes use inconsistent config attribute naming and appear broken at runtime.
- Training and simulation are not currently proven end-to-end by tests.
- Some top-level CLI promises overstate what the implementation actually guarantees.
- The codebase prefers resilience and fallbacks over strict contract failure, which is useful operationally but can hide correctness issues.
