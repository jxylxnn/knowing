# File Map

## Purpose

- This map identifies the files and directories that matter most for implementation, debugging, and future AI sessions.
- Paths are relative to the repository root.

## Top-Level Entry Points

### `update_data.py`

- Role: fetches and writes historical player/team data.
- High-value functions:
  - `fetch_player_logs`
  - `fetch_team_logs`
  - `fetch_player_logs_since`
  - `fetch_team_logs_since`
  - `update_season`
  - `update_since_date`
  - `save_data`
- Critical outputs:
  - `data/nba_players.csv`
  - `data/nba_games.csv`
- Risk level: medium.
- Safe edits:
  - CLI option clarity
  - logging
  - season-selection ergonomics
- Risky edits:
  - data schema changes
  - merge/save behavior affecting downstream loaders

### `train.py`

- Role: main training entry point.
- Calls into:
  - `src/preprocessing/data_loader.py`
  - `src/preprocessing/feature_engineer.py`
  - `src/training/pipeline.py`
- Current note: `train.py` now resolves `training_presets` from `config/default.yaml` through `src/training/presets.py`, and the selected preset controls feature-group selection, Transformer enablement, and optional recent-history trimming before feature engineering.
- Risk level: high because it defines the artifact contract consumed later by simulation.
- Current note: success now depends on `TrainingPipeline` validating the full runtime artifact set, so failures here can indicate missing output files rather than only model-fit errors.
- Current note: `train.py` now performs explicit preflight checks for the writable model/cache directories and required raw CSV inputs before expensive work begins, and it logs stage names so subprocess callers can surface a failure stage clearly.
- Current note: Step 2 feature-engineering setup should go through `src/preprocessing/feature_engineer.py:build_feature_engineer(...)` so mixed-version checkouts tolerate older constructor signatures; `tests/test_training/test_train_entrypoint.py` statically guards that call shape.
- Current note: the active training presets now include `archetype`, the deterministic player-style similarity group used to support cold-start prediction.

### `train_colab.ipynb`

- Role: Colab-oriented wrapper around `train.py`.
- Behavior:
  - resolves the actual repo checkout separately from Drive-backed storage paths
  - validates the repo root / `train.py`, raw CSV inputs, and writability of the Google Drive models directory before launch
  - runs `train.py` through `subprocess.run(..., capture_output=True, text=True)` and prints both stdout and stderr
  - raises a `RuntimeError` on nonzero exit so the notebook stops loudly instead of hiding the real traceback
- Risk level: medium.
- Safe edits:
  - launch UX
  - debug output and preflight checks
- Risky edits:
  - changing the training command contract without keeping it aligned with `train.py`

### `simulate_season.py`

- Role: main simulation entry point.
- Calls into schedule scraping, model loading, season simulation, and report export.
- Risk level: high.
- Current caveat: depends on runtime paths with known scraper regressions.

### `backtest.py` (NEW — 2026-05-09)

- Role: evaluate prediction accuracy on historical completed games.
- High-value functions: `run_backtest`, `parse_args`, `setup_logging`.
- Calls into: `BacktestRunner`, `ModelManager`, `DataLoader`.
- Outputs: per-stat MAE, RMSE, R², calibration error, prediction interval coverage.
- Risk level: medium. Depends on trained models and historical data being available.
- CLI modes: `--from`/`--to` for date range, `--recent N` for last N days, `--output <path>` for JSON export.

### `optimize_weights.py` (NEW — 2026-05-09)

- Role: retune ensemble blend weights via scipy.optimize against holdout data.
- High-value functions: `optimize_weights`, `parse_args`, `setup_logging`.
- Calls into: `EnsembleOptimizer`, `BacktestRunner`, `WeightStore`, `ModelManager`, `Config`.
- Risk level: medium. Modifies weight storage; rollback supported via `--rollback N`.
- CLI modes: `--from`/`--to` for date range, `--recent N`, `--dry-run`, `--rollback N`, `--list`.

### `query_prob.py`

- Role: interactive or one-shot query CLI for projection probabilities.
- Depends on the latest exported projection CSV.
- Risk level: medium to high because it surfaces user-visible output and assumes a specific export schema.

### `clear_cache.py`

- Role: safe cleanup for generated artifacts.
- Risk level: low to medium.
- Important behavior: preserves raw source data in `data/`.

## Configuration

### `config/default.yaml`

- Primary runtime config file.
- Owns paths, training defaults, training preset definitions, simulation settings, and scraper-related settings.
- Risk level: high if keys are renamed or semantics change.

### `src/config/config.py`

- Dataclass-based config loading and defaults.
- Central dependency for most scripts.
- Current note: `Config` now carries `training_presets` as a raw mapping so `train.py` can resolve preset definitions from YAML without hard-coding them in the CLI.
- Risk level: high because config drift breaks many modules at once.

### `src/config/model_config.py`

- Hardware/model-size heuristics and `SIZE_TIER_SPECS` dictionary.
- Important for training-time resource decisions.
- Current note: M tier transformer `seq_len` and `max_seq_length` are now 20 (previously 10), matching the L tier's context window.
- Risk level: medium.

## Data And Scraping Layer

### `src/data/`

- Ownership: all external data acquisition beyond the raw `update_data.py` NBA logs.
- Risk profile: highest-fragility directory in the repo.

Important files:

- `src/data/schedule_scraper.py`
  - Used by simulation CLI.
  - Currently appears broken by config attribute naming regressions.
  - High-risk edit area.
- `src/data/lineup_scraper.py`
  - Used by `GameSimulator`.
  - Currently appears broken by unreachable initialization and undefined names.
  - High-risk edit area.
- `src/data/injury_scraper.py`
  - Injury context input for simulation.
  - Medium to high risk due to external dependency volatility.
- `src/data/betting_scraper.py`
  - Betting lines input with fallback behavior.
  - Medium to high risk.
- `src/data/basketball_ref_scraper.py`
  - Appears to contain the same config drift pattern as `schedule_scraper.py`.
  - Treat as risky and likely unhealthy until proven otherwise.
- `src/data/rotowire_lineup_scraper.py`
  - Looks like an alternate/legacy lineup implementation with similar regressions.
  - Treat as confusing/possibly inactive.
- `src/data/nba_defense_scraper.py`
  - Supplies opponent-defense context used in parts of simulation/query logic.

## Preprocessing And Feature Engineering

### `src/preprocessing/data_loader.py`

- Role: reads raw CSVs and merges player/team data into the training base frame.
- High-value methods:
  - `load_data`
  - `merge_datasets`
  - `_add_team_rolling_features`
- Critical because downstream feature engineering assumes its output shape.
- Risk level: high.

### `src/preprocessing/feature_engineer.py`

- Role: central orchestrator for feature generation.
- High-value methods:
  - `_build_groups`
  - `create_features`
  - `create_features_chunked`
  - `get_group_columns`
  - `get_diagnostics`
- Compatibility helper:
  - `build_feature_engineer(...)` filters constructor kwargs and backfills `disable_groups`/`disable_columns` for older checkouts that do not accept them directly.
  - `benchmark_feature_variants()` now uses the same helper for its internal ablation variants, so this module is the compatibility boundary for both normal Step 2 construction and feature-ablation probing.
- Owns feature-group registration, diagnostics, and main feature build flow.
- Risk level: high.
- Current note: the orchestrator still drives feature-group order, but the widest groups now batch their output columns internally before concatenating them back into the frame.

### `src/training/presets.py`

- Role: named training preset registry and recent-history window helper for `train.py`.
- High-value objects/functions:
  - `TrainingPreset`
  - `BUILTIN_TRAINING_PRESETS`
  - `resolve_training_preset`
  - `apply_recent_history_window`
- Risk level: medium to high because it now controls the feature-stack shape and Transformer enablement used by the main training CLI.
- Current note: the built-in `full` preset includes all 23 feature groups (19 original + 4 lifecycle: injury_risk, aging_curve, kan_aging, skill_development). The `small` preset includes 6 groups (rolling, efficiency, momentum, pace, opponent_strength, archetype).

### `src/preprocessing/features/`

- Role: modular feature groups.
- Important files include:
  - `base.py` — `FeatureGroup` ABC, `FeatureContext`, `FeatureDiagnostics`, `fill_series_with_prior`, `add_missing_flag`, `normalize_output_columns`
  - `_teammate_utils.py` — shared roster/teammate precomputations (`TeammateContext`, `build_game_roster_map`, `build_regular_teammates_map`, `build_high_usage_teammates_map`, `build_team_totals_map`).  Used by `lineup_stability`, `injury_opportunity`, and `teammate_usage` to avoid duplicate mapping logic.
  - `rolling.py` — `RollingFeatureGroup`, `EfficiencyFeatureGroup`, `MomentumFeatureGroup`
  - `context.py` — `ContextualFeatureGroup`, `FatigueFeatureGroup`
  - `matchup.py` — `MatchupFeatureGroup`, `OpponentStrengthFeatureGroup`
  - `pace_role.py` — `PaceFeatureGroup`, `TeamRoleFeatureGroup`
  - `archetype.py` — `PlayerArchetypeFeatureGroup`
  - `target_encoding.py` — `TargetEncodingFeatureGroup`, `LeagueRankingFeatureGroup`
  - `minutes_confidence.py` — `MinutesConfidenceFeatureGroup` (rolling variance, trend, starter-rate signals for MIN)
  - `recency_form.py` — `RecencyFormFeatureGroup` (recent-vs-season deltas, form ratios, volatility)
  - `lineup_stability.py` — `LineupStabilityFeatureGroup` (starter rate, teammate Jaccard continuity, rotation variance, MIN rank)
  - `rest_density.py` — `RestGameDensityFeatureGroup` (schedule density, B2B detection, rest advantage, composite density score)
  - `injury_opportunity.py` — `InjuryAdjustedOpportunityFeatureGroup` (missing high-usage teammate detection, same-minutes missing count, MIN/usage boost, team absence rolling count)
  - `teammate_usage.py` — `TeammateUsageFeatureGroup` (top-usage teammate active flag, missing teammate FGA/AST/REB shares, missing shot volume, active scoring depth)
  - `defense_position.py` — `DefensePositionFeatureGroup` (opponent defensive stats by position group — guard/wing/big — including PTS/REB/AST/STL/BLK/TOV allowed, defensive rank, recent PTS allowed)
  - **`injury_risk.py` (NEW)** — `InjuryRiskFeatureGroup`: METIC-style workload + injury history signals from `data/injury_history.csv`. Outputs: `INJURY_RISK_CAREER_COUNT`, `INJURY_RISK_LAST_90D`, etc.
  - **`aging_curve.py` (NEW)** — `AgingCurveFeatureGroup`: B-Ianus Bayesian model features. Outputs: `AGING_PLAYER_AGE`, `AGING_PEAK_AGE_EST`, `AGING_CURVE_FACTOR`, etc.
  - **`kan_aging.py` (NEW)** — `KANAgingFeatureGroup`: KAN nonlinear age features. Outputs: `KAN_AGE_NONLIN_FACTOR`, `KAN_AGE_INFLECTION_AGE`, `KAN_AGE_VOLATILITY`.
  - **`skill_development.py` (NEW)** — `SkillDevelopmentFeatureGroup`: growth velocity metrics. Outputs: `SKILL_DEV_PTS_VELOCITY`, `SKILL_DEV_EFF_VELOCITY`, etc.
  - `__init__.py` — re-exports all feature group classes
- Safe entry point for adding new features if the feature schema contract is respected.
- All new feature groups follow the batched-column pattern: accumulate columns in a `dict[str, pd.Series]`, then `_concat_new_columns(df, new_columns)` once per group.
- Current caution: `rolling.py` was a performance hotspot due to DataFrame fragmentation warnings; the hot groups now assemble feature columns in batches and concatenate once per group.
- Current caution: `archetype.py` computes hard labels plus soft similarities from fixed playstyle templates. Keep that template set in sync with preset definitions and schema expectations if the archetypes change.
- Current note: `lineup_stability.py` Jaccard computation was refactored to a vectorised key-shift approach (no per-player Python loops) and now sources its roster maps from `_teammate_utils.py`.
- Current note: `rest_density.py` game-count windows were vectorised with pandas time-based `rolling(..., closed='left')`, and opponent-rest lookups now use `np.searchsorted` on pre-sorted `datetime64[ns]` arrays instead of nested Python loops.

## Training Layer

### `src/training/pipeline.py`

- Primary training orchestration.
- Critical responsibilities:
  - split data chronologically
  - train CatBoost models
  - train Transformer model
  - save per-target CatBoost runtime artifacts
  - save feature schema and blend weights
  - validate the runtime artifact contract before reporting success
- High-value methods:
  - `prepare_data`
  - `_train_catboost_parallel`
  - `_save_catboost_artifacts`
  - `_train_transformer_model`
  - `_build_sequence_batch` — now uses zero-padding for short players instead of skipping them
  - `_save_feature_cols`
  - `_save_blend_weights`
  - `_save_model_stack_metadata`
  - `_validate_runtime_artifact_contract`
  - `_validate_blend_contract`
  - `train`
  - `load_models`
- Risk level: very high.
- Current caution: any artifact filename or format change here must stay aligned with `CatBoostTrainer`, `ModelManager`, and `simulate_season.py`.
- Current note: Transformer validation no longer calls the compiled model directly; it delegates through `TransformerWrapper.predict_batch()` so the validation seam stays on the eager-safe path.
- Current note: `_save_model_stack_metadata()` records whether the Transformer was active and can now include the selected training preset and enabled feature groups when `train.py` provides them.
- Current note: `_build_sequence_batch()` now produces zero-padded sequences for players with fewer than `seq_len + 1` games instead of skipping them entirely.

### `src/training/catboost_trainer.py`

- Owns CatBoost fitting and artifact save/load logic.
- Important seam between training and runtime model loading.
- Important runtime helpers:
  - `metadata_path`
  - `primary_model_candidates`
  - `missing_runtime_artifacts`
  - `validate_saved_artifacts`
- Risk level: high.

### `src/training/experiment.py`

- Writes experiment metadata under `experiments/`.
- Risk level: low to medium.

### `src/training/feature_cache.py`

- Contains cache helpers not clearly wired into the main `train.py` flow.
- Treat as secondary/uncertain until a caller is confirmed.

### `src/training/__init__.py`

- Lazy-import wrapper for training package.
- Small file, but important for import behavior and backward compatibility.

### `src/training/nexus_loss.py` (NEW — 2026-05-22)

- Role: `GaussianNLLLoss` — multivariate Gaussian negative log-likelihood with Cholesky covariance.
- Used by the Nexus multi-modal model for end-to-end training.
- Natively penalizes mathematically impossible stat combinations (negative variances, non-PSD correlation matrices).
- Risk level: low (only used by Nexus model, not yet wired as active training path).

## Lifecycle Layer (NEW — 2026-05-22)

### `src/lifecycle/aging_model.py`

- Role: B-Ianus Bayesian aging curve model. Separates development (pre-peak) from decline (post-peak) with position-specific priors.
- High-value objects/functions:
  - `BIanusAgingModel` — main class with `precompute_all()` and `fit_player_curve()`
  - `normalize_position()` — maps NBA position strings to canonical 5 positions
  - Position-specific priors: `POSITION_PEAK_PRIORS`, `POSITION_DECLINE_PRIORS`
- Cache: writes `data/cache/aging_curves.csv`.
- Risk level: low to medium. Non-fatal on missing bio data — defaults all aging features to neutral (1.0 factor).

### `src/lifecycle/kan_age_model.py`

- Role: KAN (Kolmogorov-Arnold Network) for nonlinear age curves.
- High-value objects:
  - `KANAgeModel` — main class with `precompute_all()` and grid-based spline computation
- Always runs on CPU (`device='cpu'`) to avoid GPU contention with CatBoost/Transformer.
- Cache: writes `data/cache/kan_aging_outputs.csv`.
- Risk level: low. Non-fatal on failure.

### `src/lifecycle/__init__.py`

- Empty init — package marker only.

## Model Runtime Layer

### `src/models/model_manager.py`

- Runtime loader and predictor used by the simulator.
- High-value methods:
  - `prepare_data`
  - `validate_runtime_artifacts`
  - `_load_feature_cols`
  - `_load_models`
  - `load_models`
  - `predict_player_stats`
  - `predict_player_stats_batch`
  - `_fallback_prediction`
- One of the most important files in the repo.
- Breakage here can invalidate all simulation output.
- Risk level: very high.
- Current note: runtime loading is now intentionally strict about missing per-target CatBoost artifacts and shared metadata files.
- Current note: `validate_runtime_artifacts()` now treats `model_stack_metadata.pkl` as part of the shared runtime set so the loader and training pipeline stay aligned on the same contract.
- Current note: `_validate_blend_contract()` raises when blend weights expect a Transformer that is missing or failed to load, eliminating the partial-blend bug.
- Current note: Transformer predictions flow through `TransformerWrapper.predict()`, which now defaults to eager inference and can force a math SDPA backend on CUDA when backend controls are available.

### `src/models/nexus_model.py` (NEW — 2026-05-22)

- Role: Nexus multi-modal architecture — unified deep-learning model (SSM + FT-Transformer + GAT + Copula head).
- High-value classes:
  - `SimplifiedSSMBlock` — Mamba-2-style SSM using standard PyTorch ops (CPU/macOS fallback)
  - `FTTransformerEncoder` — Tabular backbone for scalar/contextual features
  - `GATRelationalLayer` — Lightweight Graph Attention Network for lineup synergy
  - `NexusModel` — top-level model with fusion & copula head returning 6-dim mean + 6x6 Cholesky covariance
- Loss: uses `GaussianNLLLoss` from `src/training/nexus_loss.py`.
- Risk level: low to medium. Import-tested but not yet the active training path.
- Status: implemented; CatBoost + Transformer remains the active stack.

### `src/preprocessing/feature_engineer_gpu.py` (NEW — 2026-05-22)

- Role: GPU-accelerated feature engineering via NVIDIA cuDF with transparent CPU fallback.
- Mirrors the public API of `FeatureEngineer` for drop-in compatibility.
- Offloads heavy groupby/rolling operations to GPU; complex groups (Python loops, scipy) fall back to CPU.
- Risk level: medium. Depends on cuDF availability; CPU fallback preserves correctness.
- Activation: automatic when `FeatureEngineer(use_gpu=True)` — no separate CLI flag.

### `src/data/player_bio_scraper.py` (NEW — 2026-05-22)

- Role: fetches player biographical data (AGE, POSITION, HEIGHT, WEIGHT, DRAFT_YEAR) from NBA API `commonplayerinfo` endpoint.
- High-value methods:
  - `fetch_all_bios(player_ids)` — batch-fetches bio data with rate limiting
  - `resolve_name_to_id(player_name)` — name-to-ID resolution for injury logging
- Cache: reads/writes `data/player_bios.csv`.
- Called from: `update_data.py` via `enrich_with_player_bios()`.
- Risk level: low. Non-fatal on failure — aging features default to neutral.

### `src/data/injury_history_logger.py` (NEW — 2026-05-22)

- Role: persists injury events across `update_data.py` runs into a longitudinal CSV log (`data/injury_history.csv`).
- High-value methods:
  - `log_injuries(events)` — appends and deduplicates by (PLAYER_ID, DATE, INJURY_TYPE)
  - `get_history()` — loads complete history as DataFrame
- Called from: `update_data.py` via `log_injury_snapshot()`.
- Risk level: low. Non-fatal on failure — injury risk features default to near-zero.

### `src/models/transformer_model.py`

- Transformer wrapper and checkpoint compatibility logic.
- Owns the eager-safe Transformer inference path used by training validation and runtime prediction.
- Owns `_create_sequences()`, which now uses zero-padding for players with fewer than `seq_len` context games instead of skipping them.
- Risk level: medium to high.
- Current caution: sequence construction behavior changed — players with 1+ games now always produce training samples, with zero-padding for early timesteps. Players with enough games still use the standard sliding window.

## Simulation Layer

### `src/simulation/game_simulator.py`

- Main matchup simulation orchestrator.
- High-value methods:
  - `_safe_get_game_lines`
  - `_safe_get_lineup`
  - `_safe_get_injury_probs`
  - `simulate_matchup`
  - `_simulate_matchup_reactive`
- Pulls together model inference, scrapers, adjustments, and repeated simulation logic.
- Highest-risk file in the repo from a coupling perspective.
- **Refactored (2026-05-09):** dead legacy code removed, delegates to PhaseSimulator, uses typed dataclasses from sim_types.
- **Strict mode (DR-025, 2026-05-22):** `__init__` now accepts `strict_mode: bool = False`. When True, `_simulate_matchup_reactive()` raises `RuntimeError` if any optional InputHealth source reports `failed` or `fallback`.

### `src/simulation/phase_simulator.py` (NEW — 2026-05-09)

- Phase-by-phase Monte Carlo game loop, extracted from GameSimulator.
- High-value methods: `simulate`, `_build_game_environment`, `_run_phase`.
- Isolates the core simulation logic for independent testing.

### `src/simulation/archetype.py` (NEW — 2026-05-09)

- ArchetypeEngine: infers player archetypes from projection shape.
- Defines ARCHETYPE_PROFILES (heliocentric star guard, 3&D wing, stretch big, etc.).
- Provides volatility/style priors for each archetype.

### `src/simulation/role_sampler.py` (NEW — 2026-05-09)

- Samples role states (limited/normal/expanded/starter/bench/closer) with archetype-aware adjustments.
- Uses pre-defined state profiles with multipliers for minutes, usage, efficiency, etc.

### `src/simulation/sim_types.py` (NEW — 2026-05-09)

- Typed dataclasses (RoleSample, PhaseDefinition, GameEnvironment, PlayerProjection, TeamContext) replacing raw dicts throughout the simulation pipeline.
- Risk level: medium — changing these types impacts all simulation consumers.

### `src/simulation/sim_cache.py` (NEW — 2026-05-09)

- JSON disk-cache mixin for GameSimulator. Uses content-hash keys for deterministic caching.

### `src/simulation/stat_utils.py` (NEW — 2026-05-09)

- Shared statistical helpers: `compute_mode` (KDE-based mode estimation), `compute_summary_stats`.

### `src/simulation/season_simulator.py`

- Iterates over schedules and runs multiple matchup simulations.
- Risk level: high because it depends on schedule output shape and simulator return contracts.

### `src/simulation/report_generator.py`

- Prints projections and exports to CSV under `data/sim_results/`.
- High-value methods:
  - `export_player_projections` — now includes `DATA_QUALITY` column (`FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`)
  - `export_to_csv`
  - `display_quick_summary`
  - `_data_quality_from_result` — static helper deriving quality from input_health metadata
- Risk level: high (export schema impacts query layer).
- Current caution: exported player projection columns appear incomplete for `STL`, `BLK`, and `TOV`.

### `src/simulation/input_health.py`

- Shared status/summary contract for scraper-backed simulation inputs.
- Used by `game_simulator.py`, `season_simulator.py`, and `simulate_season.py`.
- Risk level: medium to high because changes here alter operator-visible degraded-mode reporting.

### Supporting simulation files

- `src/simulation/minutes_predictor.py`
- `src/simulation/context_aware_adjustments.py`
- `src/simulation/player_correlation_engine.py`
- `src/simulation/four_factors_engine.py`
- `src/simulation/error_calibration.py`

These are important but secondary to the main orchestration files above.

## Evaluation Layer (NEW — 2026-05-09)

### `src/evaluation/metrics.py`

- Role: backtest metrics and result types.
- Key types: `BacktestResult`, `TargetMetrics`, `compute_target_metrics`.
- Tracks: MAE, RMSE, R², MAPE, calibration (P10/P90), bias, prediction interval coverage.
- Risk level: medium.

### `src/evaluation/backtest_runner.py`

- Role: runs model predictions against completed games and compares to actual box scores.
- High-value methods: `run`, `set_weights`, `_prepare_player_context`, `_evaluate_predictions`.
- Depends on: `ModelManager` for predictions, `DataLoader` for actual box scores, `FeatureEngineer` for feature context.
- Risk level: medium to high — this is the bridge between predictions and reality.

### `src/evaluation/ensemble_optimizer.py`

- Role: self-optimizing ensemble weight tuner using scipy.optimize.
- High-value methods: `optimize`, `accept`, `_build_candidate_weights`, `_evaluate_candidate`.
- Tunes 13 parameters: 6 per-target CatBoost/Transformer ratios + 6 per-target intercepts + 1 CatBoost-MAE blend.
- Accept/verify gates prevent regressions from being deployed.
- Risk level: medium — modifies production weight config.

### `src/evaluation/weight_store.py`

- Role: versioned JSON weight storage with atomic writes and rollback.
- Key types: `WeightStore`, `EnsembleWeights`, `TargetBlend`.
- Replaces opaque binary `blend_weights.pkl` with human-readable versioned JSON.
- Default location: `data/weights/` with `current.json` pointer.
- Risk level: medium — is the single source of truth for active ensemble weights.

### `src/evaluation/drift_detector.py`

- Role: statistical process control for model performance.
- Key types: `DriftDetector`, `DriftStatus`, `DriftReport`.
- Flags when rolling MAE exceeds 2σ above historical baseline.
- Distinguishes minor drift (retune weights) from major drift (retrain models).
- Risk level: medium — needs live backtest data to establish baseline.

### `src/evaluation/__init__.py`

- Re-exports: `BacktestResult`, `TargetMetrics`, `BacktestRunner`.

## Query Layer

### `src/query/interactive_cli.py`

- User-facing interactive shell for querying projections.
- High-value methods:
  - `run`
  - `_handle_over_under`
  - `_handle_projection`
  - `_handle_compare`
  - `_run_live_simulation`
  - `query_one_shot`
- Risk level: medium.

### `src/query/projection_loader.py`

- Loads the latest projection export and enriches it for query use.
- High-value methods:
  - `load_projections`
  - `find_player`
  - `_row_to_projection`
  - `get_recent_games`
  - `get_matchup_history`
  - `get_opponent_defense_profile`
  - `_load_cached_defense_data`
  - `get_player_context`
- Critical seam between simulation output and query-time UX.
- Current caution: hardcoded defense cache filename for season `2025-26`.

### `src/query/probability_calculator.py`

- Probability and Monte Carlo math.
- High-value methods:
  - `calculate_from_projection`
  - `run_monte_carlo_simulation`
  - `evaluate_calibration`
- Well-covered by tests relative to other areas.
- Risk level: medium.

### `src/query/query_parser.py`

- Parses user phrases into supported query operations.
- Risk level: medium.

## Pipeline Compatibility Layer

### `src/pipeline/`

- Mixed-use area containing helper abstractions and compatibility modules.
- Important files:
  - `src/pipeline/prediction_service.py`
  - `src/pipeline/training_pipeline.py`
- `training_pipeline.py` is a shim that re-exports the active training pipeline.
- Treat this directory carefully; not every file here is the canonical implementation path.

## Utilities

### `src/utils/prediction_utils.py`

- Owns `FeatureSchema`, selectors, fallback predictors, and temporal weighting helpers.
- High-risk because training and inference compatibility depend on it.
- `src/utils/__init__.py` re-exports `FeatureSchema` and the other shared helpers so compatibility-sensitive callers can import from the package namespace too.

## Tests

### `tests/`

- Best entry point for understanding expected behavior before changing core logic.
- Areas with meaningful coverage:
  - preprocessing
  - training pieces
  - model wrappers
  - query logic
  - portions of simulation
- Coverage gaps:
  - live scraper behavior
  - schedule-scraper runtime health
  - full train-to-sim artifact contract
- High-value files:
  - `tests/test_models/test_model_manager.py`
  - `tests/test_preprocessing/test_feature_engineer.py`
  - `tests/test_pipeline/test_data_pipeline.py`
  - `tests/test_simulation/test_game_simulator.py`
  - `tests/test_query/test_probability_calculator.py`
  - `tests/test_query/test_interactive_cli.py`
  - `tests/test_training/test_training_pipeline_colab.py`
  - `tests/test_training/test_nn_trainer.py`
  - `tests/test_models/test_transformer_model.py` — now includes config, zero-padding, and regression tests for the transformer sequence builder

## Generated State And Non-Source Directories

### `data/`

- Runtime-generated raw and derived datasets.
- Not reliable as checked-in source of truth because contents are ignored and environment-dependent.

### `models/`

- Runtime-generated trained artifact directory.
- Contract-critical for simulation and inference.

### `cache/` and `data/cache/`

- Cache directories.
- Safe to clear with `clear_cache.py`, but not safe to rely on as durable source of truth.

### `experiments/`

- Experiment tracking outputs.
- Present but not central to the active user journey.

## Legacy, Empty, Or Confusing Areas

- `src/services/`
  - currently empty
  - do not assume it owns anything yet
- `src/evaluation/`
  - now contains 5 active modules: `metrics.py`, `backtest_runner.py`, `ensemble_optimizer.py`, `weight_store.py`, `drift_detector.py`
  - no longer a placeholder
- `plans/`, `cline_docs/`, and local PDFs
  - useful historical context, not active runtime code

## Safe Versus Risky Edit Zones

### Safer areas

- documentation
- tests
- `clear_cache.py`
- additive feature modules that preserve schema contracts
- experiment logging improvements

### Riskier areas

- `src/training/pipeline.py`
- `src/models/model_manager.py`
- `src/simulation/game_simulator.py`
- `src/simulation/report_generator.py`
- `src/preprocessing/data_loader.py`
- `src/utils/prediction_utils.py`
- most of `src/data/`

If editing a risky area, verify both direct behavior and downstream file-contract effects.
