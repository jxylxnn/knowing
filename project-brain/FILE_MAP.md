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

- Hardware/model-size heuristics.
- Important for training-time resource decisions.
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
- Current note: the built-in `full` preset includes all 19 feature groups (rolling, efficiency, momentum, context, fatigue, minutes_confidence, rest_density, matchup, opponent_strength, pace, team_role, lineup_stability, injury_opportunity, teammate_usage, recency_form, archetype, defense_position, target_encoding, league_rank). The `small` preset includes 6 groups (rolling, efficiency, momentum, pace, opponent_strength, archetype).

### `src/preprocessing/features/`

- Role: modular feature groups.
- Important files include:
  - `base.py` — `FeatureGroup` ABC, `FeatureContext`, `FeatureDiagnostics`, `fill_series_with_prior`, `add_missing_flag`, `normalize_output_columns`
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
  - `__init__.py` — re-exports all feature group classes
- Safe entry point for adding new features if the feature schema contract is respected.
- All new feature groups follow the batched-column pattern: accumulate columns in a `dict[str, pd.Series]`, then `_concat_new_columns(df, new_columns)` once per group.
- Current caution: `rolling.py` was a performance hotspot due to DataFrame fragmentation warnings; the hot groups now assemble feature columns in batches and concatenate once per group.
- Current caution: `archetype.py` computes hard labels plus soft similarities from fixed playstyle templates. Keep that template set in sync with preset definitions and schema expectations if the archetypes change.
- Current caution: `lineup_stability.py` uses row-level iteration for Jaccard similarity and roster-size lookups; this may be slow on very large datasets. Consider vectorization if it becomes a bottleneck.
- Current caution: `rest_density.py` uses row-level iteration for game-count windows and opponent rest lookups; same performance caveat applies.

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

### `src/models/transformer_model.py`

- Transformer wrapper and checkpoint compatibility logic.
- Owns the eager-safe Transformer inference path used by training validation and runtime prediction.
- Risk level: medium to high.

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
- Current caution: contains a dead legacy simulation block after an early return.

### `src/simulation/season_simulator.py`

- Iterates over schedules and runs multiple matchup simulations.
- Risk level: high because it depends on schedule output shape and simulator return contracts.

### `src/simulation/report_generator.py`

- Console/report export layer for simulation results.
- High-value methods:
  - `format_console_report`
  - `export_to_csv`
  - `export_player_projections`
- Risk level: high because query-time behavior depends on its CSV schema.
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
  - `tests/test_models/test_transformer_model.py`

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
  - currently only `__init__.py`
  - likely placeholder for future evaluation work
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
