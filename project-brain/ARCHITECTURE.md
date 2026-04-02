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

1. `train.py` loads config from `config/default.yaml`.
2. `src/preprocessing/data_loader.py` reads the raw CSVs and joins player/team context.
3. `src/preprocessing/feature_engineer.py` applies registered `FeatureGroup` modules.
4. `src/training/pipeline.py` creates chronological train/validation/test splits.
5. The training pipeline fits:
   - CatBoost regressors per target
   - a Transformer sequence model
   - quantile models
   - blend weights
   - the Transformer path now keeps an eager inference copy for validation/runtime use and treats `torch.compile` as opt-in only
6. Training writes the runtime artifact set to `models/`:
   - per-target CatBoost model files
   - per-target CatBoost metadata files
   - `attention_transformer.pkl` when the Transformer is enabled
   - `feature_schema.pkl`
   - `feature_cols.pkl`
   - `blend_weights.pkl`
7. Before `TrainingPipeline.train()` returns, it validates that the required runtime files exist and raises instead of reporting a false-success training run.
8. The Colab notebook wrapper `train_colab.ipynb` launches `train.py` as a subprocess, captures stdout/stderr, and stops immediately on nonzero exit so the operator sees the real failure stage instead of only a wrapper exception.

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
- `TrainingPipeline._predict_transformer_batch()` delegates validation inference through `TransformerWrapper.predict_batch()`, which defaults to the eager model path and safe SDPA backend controls on CUDA.
- `train.py` now performs explicit preflight checks for writable model/cache directories and required raw CSV inputs before expensive work begins, which makes notebook and CLI failures easier to diagnose.

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

Important named methods in this path:

- `ModelManager.load_models`
- `ModelManager.predict_player_stats`
- `ModelManager.predict_player_stats_batch`
- `GameSimulator._safe_get_game_lines`
- `GameSimulator._safe_get_lineup`
- `GameSimulator._safe_get_injury_probs`
- `GameSimulator._safe_get_defensive_adjustments`
- `GameSimulator._simulate_matchup_reactive`
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
- The active feature-group stack assembled in `FeatureEngineer._build_groups()` is:
  - `RollingFeatureGroup`
  - `EfficiencyFeatureGroup`
  - `MomentumFeatureGroup`
  - `ContextualFeatureGroup`
  - `FatigueFeatureGroup`
  - `MatchupFeatureGroup`
  - `OpponentStrengthFeatureGroup`
  - `PaceFeatureGroup`
  - `TeamRoleFeatureGroup`
  - `TargetEncodingFeatureGroup`
  - `LeagueRankingFeatureGroup`

Key invariant:

- Training and inference rely on stable feature-column semantics. Adding or renaming columns without updating saved schema expectations is risky.
- `rolling.py` now materializes its wide rolling/efficiency/momentum outputs in temporary structures and appends them with a single concat per group to avoid pandas fragmentation.

### `src/training/`

- Owns model fitting, experiment logging, and training orchestration.
- `pipeline.py` is the active end-to-end training pipeline.
- `catboost_trainer.py` owns CatBoost training and per-target artifact persistence behavior.
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
  - `game_simulator.py`
  - `season_simulator.py`
  - `report_generator.py`
  - `minutes_predictor.py`
  - `context_aware_adjustments.py`
  - `player_correlation_engine.py`
  - `four_factors_engine.py`
  - `error_calibration.py`

Fragility notes:

- `game_simulator.py` is a high-coupling orchestrator with many dependency points.
- It currently contains an early return into a reactive path, with a large older simulation block left below that return. That dead code increases maintenance risk.
- Scraper-backed optional context now flows through a shared input-health contract in `src/simulation/input_health.py`.
- Optional context failures degrade a run visibly; required schedule failures are treated as hard failures.

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
