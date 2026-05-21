# Architecture

## System Shape

- Architecture style: local batch/CLI pipeline with file-based persistence.
- No server process, no request router, no auth layer, no database.
- The repo is organized around four operational phases:
  1. data ingestion
  2. feature engineering and training
  3. simulation and projection export
  4. query-time probability lookup

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
4. `src/training/pipeline.py` creates chronological train/validation/test splits.
5. The training pipeline fits:
    - CatBoost regressors per target
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
    - `model_stack_metadata.pkl` recording whether the Transformer was active, the expected model count, and optionally the selected preset / feature groups
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
- `train.py` now performs explicit preflight checks for writable model/cache directories and required raw CSV inputs before expensive work begins, which makes notebook and CLI failures easier to diagnose.
- `train.py` now also resolves training presets, passes the preset feature-group selection through `build_feature_engineer(...)`, and records the selected preset in `model_stack_metadata.pkl`.
- `train_colab.ipynb` should not infer `train.py` from the Drive models directory; it now searches the actual repo checkout first and keeps Drive-backed `data/` and `models/` separate from code location.

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

Important named methods in this path:

- `ProjectionLoader.load_projections`
- `ProjectionLoader.find_player`
- `ProjectionLoader.get_recent_games`
- `ProjectionLoader.get_opponent_defense_profile`
- `ProbabilityCalculator.calculate_from_projection`
- `ProbabilityCalculator.run_monte_carlo_simulation`
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
  - `metrics.py` — `BacktestResult`, `TargetMetrics`, `compute_target_metrics` dataclasses (MAE, RMSE, R², MAPE, calibration, interval coverage)
  - `backtest_runner.py` — `BacktestRunner`: runs model predictions against historical completed games, compares against actual box scores
  - `ensemble_optimizer.py` — `EnsembleOptimizer`: 13-parameter scipy.optimize-based weight tuner with accept/verify gates
  - `weight_store.py` — `WeightStore` + `EnsembleWeights` + `TargetBlend`: versioned JSON storage replacing opaque binary `blend_weights.pkl`, atomic writes, rollback
  - `drift_detector.py` — `DriftDetector`: statistical process control tracking per-stat accuracy over rolling windows; flags when rolling MAE exceeds 2σ above baseline
- Key contracts:
  - `BacktestRunner` depends on `ModelManager` for predictions and `DataLoader` for actual box scores
  - `EnsembleOptimizer` owns a `BacktestRunner` internally and creates/validates candidate weight configs
  - `WeightStore` writes to a versioned directory (default: `data/weights/`) with a `current.json` pointer
  - `ModelManager` now accepts hot-reloadable `EnsembleWeights` objects via `set_weights()`
  - The weight store format is human-readable JSON, not opaque binary pickle

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
  - `probability_calculator.py`
  - `query_parser.py`

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
- `data/weights/` — versioned JSON weight store (replaces opaque binary `blend_weights.pkl`). Contains version-numbered weight files + `current.json` pointer. Managed by `WeightStore`.

### Generated Outputs

- `data/sim_results/sim_results_<timestamp>.csv`
- `data/sim_results/player_projections_<timestamp>.csv`
- Experiment JSON files under `experiments/<experiment_name>/`

### Cache Layers

- `data/cache/` for scraped or derived data.
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
