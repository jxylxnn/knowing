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
- Current note: `train.py` exposes `--feature-selection {off,smart}` and `--selection-profile {fast,balanced,max_accuracy}`. When smart selection is enabled, `train.py` runs `SmartFeatureSelector` between Step 2 (feature engineering) and Step 3 (training) and feeds the resulting `SelectionManifest` into `TrainingPipeline.apply_feature_selection_manifest()`. Failure is non-fatal — the pipeline falls back to the canonical `self.feature_cols` list.
- Current note: `train.py` now exposes `--diagnose` and `--diagnose --stop-after <stage>` for fast crash-stage diagnostics without running full training. Stages: `preflight`, `data_load`, `feature_engineering`, `feature_selection`, `prepare_data`, `artifact_check`. See `src/training/diagnostics.py` and `tests/test_training/test_diagnostics.py`.

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

### `backtest.py` (NEW — 2026-05-09, extended 2026-06-04)

- Role: evaluate prediction accuracy on historical completed games.
- High-value functions: `run_backtest`, `parse_args`, `setup_logging`.
- Calls into: `BacktestRunner`, `ModelManager`, `DataLoader`, `src.evaluation.metrics.backtest_result_to_json_dict`.
- Outputs: per-stat MAE, RMSE, R², calibration error, prediction interval coverage.
- Risk level: medium. Depends on trained models and historical data being available.
- CLI modes: `--from`/`--to` for date range, `--recent N` for last N days, `--output <path>` for the existing pretty JSON, `--json-output <path>` for the stable machine-readable JSON payload (new — used by downstream tooling that needs to consume metrics without scraping terminal output).

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

### `optimize_variance.py` (NEW — 2026-05-21)

- Role: tune context-specific volatility multipliers via CRPS.
- High-value functions: `main`, `load_backtest_data`, `crps_objective`, `_parse_args`.
- Calls into: `src.evaluation.metrics.calculate_empirical_crps`.
- Tunes 7 multipliers: B2B, Rookie, Blowout, Home, Away, Playoff, RestAdvantage.
- CLI modes: `--recent N`, `--from`/`--to`, `--target <STAT>`, `--sims N`, `--dry-run`.
- Risk level: low. Independent of training artifacts — operates on raw CSVs.

### `calibrate_residual_intervals.py` (NEW — 2026-06-12)

- Role: builds conformal confidence interval artifacts from residual prediction errors.
- Calls into: `src.correction.calibration.ResidualIntervalCalibrator`.
- Input: `data/evaluation/residual_training.parquet`.
- Outputs: `models/calibration/calibration_metadata.json` and one `{stat}_intervals.json` file for each calibrated target.
- CLI modes: `--confidence-levels`, `--min-bucket-rows`, `--targets`.
- Risk level: low to medium. It does not retrain the base model, but its artifacts influence runtime interval/confidence output.

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
  - **Schedule contract wiring (2026-06-04):** every read path (cached schedule hit, fresh API, cache fallback, season cache) calls `normalize_schedule_frame(...)` before returning. Empty frames are skipped from normalization. The contract normalizes `GAME_ID`, `GAME_DATE`, `HOME_TEAM`, and `AWAY_TEAM` and rejects frames with nulls in required columns.
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

## Residual Correction And Calibration

### `src/correction/`

- Ownership: walk-forward residual datasets, residual correction models, residual runtime application, and conformal residual interval calibration.
- Important files:
  - `walk_forward_residuals.py` — builds walk-forward base predictions for mistake-learning datasets.
  - `residual_dataset.py` — canonical residual row/dataframe schema (`STAT`, `BASE_PREDICTION`, `ACTUAL`, `ERROR`, `MODEL_FOLD`, `DATA_QUALITY`, etc.).
  - `correction_features.py` — training/runtime residual-model feature builder; maps `DATA_QUALITY` into numeric quality score.
  - `residual_trainer.py` — trains one CatBoost residual model per target and writes `models/residual/`.
  - `residual_model.py`, `correction_store.py`, `correction_applier.py` — runtime load/apply path for residual corrections.
  - `calibration.py` — `ResidualIntervalCalibrator`; computes conformal half-widths from absolute residual errors and writes bucketed JSON artifacts.
  - `interval_store.py` — `CalibrationIntervalStore`; non-fatal runtime loader with stat/bucket fallback.
  - `confidence_scorer.py` — `ConfidenceScorer`; maps interval/context signals to `HIGH`, `MEDIUM`, `LOW`, or `NO_EDGE`.
- Risk level: medium. Residual correction changes point predictions; interval calibration changes operator trust/risk output and projection CSV schema. Missing calibration artifacts must remain non-fatal.

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
- Current note: `FeatureEngineeringResult` now records `n_rows` and `n_features` for selector diagnostics. The `get_group_columns()` output is consumed by `SmartFeatureSelector.run` to map columns back to their groups for ablation scoring.

### `src/training/presets.py`

- Role: named training preset registry and recent-history window helper for `train.py`.
- High-value objects/functions:
  - `TrainingPreset`
  - `BUILTIN_TRAINING_PRESETS`
  - `resolve_training_preset`
  - `apply_recent_history_window`
- Risk level: medium to high because it now controls the feature-stack shape and Transformer enablement used by the main training CLI.
- Current note: the built-in `full` preset includes all 23 feature groups (19 original + 4 lifecycle: injury_risk, aging_curve, kan_aging, skill_development). The `small` preset includes 6 groups (rolling, efficiency, momentum, pace, opponent_strength, archetype).
- Current note: `TrainingPreset` now carries optional `feature_selection` and `feature_selection_profile` fields. Presets can opt into smart selection through `config/default.yaml` under `training_presets.<name>.feature_selection`.

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
  - **`season_phase.py` (NEW)** — `SeasonPhaseFeatureGroup`: early-season ramp-up and trade resets. Outputs: `DAYS_SINCE_SEASON_START` (capped at 30), `IS_SEASON_OPENER`, `GAMES_WITH_CURRENT_TEAM`, `IS_RECENT_TRADE` (≤5 games with new team).
  - **`team_motivation.py` (NEW)** — `TeamMotivationFeatureGroup`: late-season tanking/load management signals. Outputs: `TEAM_CUMULATIVE_WIN_PCT` (shift-1), `IS_LATE_SEASON`, `IS_TANKING_PROXY`, `IS_PLAYOFF_LOCK_PROXY`.
  - **`postseason_context.py` (NEW)** — `PostseasonContextFeatureGroup`: playoff detection. Outputs: `IS_PLAYOFF_GAME`, `PLAYOFF_PACE_PRIOR` (0.95 playoff / 1.0 regular season).
  - `__init__.py` — re-exports all feature group classes
- Safe entry point for adding new features if the feature schema contract is respected.
- All new feature groups follow the batched-column pattern: accumulate columns in a `dict[str, pd.Series]`, then `_concat_new_columns(df, new_columns)` once per group.
- Current caution: `rolling.py` was a performance hotspot due to DataFrame fragmentation warnings; the hot groups now assemble feature columns in batches and concatenate once per group.
- Current caution: `archetype.py` computes hard labels plus soft similarities from fixed playstyle templates. Keep that template set in sync with preset definitions and schema expectations if the archetypes change.
- Current note: `lineup_stability.py` Jaccard computation was refactored to a vectorised key-shift approach (no per-player Python loops) and now sources its roster maps from `_teammate_utils.py`.
- Current note: `rest_density.py` game-count windows were vectorised with pandas time-based `rolling(..., closed='left')`, and opponent-rest lookups now use `np.searchsorted` on pre-sorted `datetime64[ns]` arrays instead of nested Python loops. `DAYS_SINCE_LAST_GAME` is capped at 14 days to prevent off-season/All-Star gaps from creating infinite rest outliers.

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
- Current note: `_save_model_stack_metadata()` records whether the Transformer was active and can now include the selected training preset and enabled feature groups when `train.py` provides them. When smart selection ran, the metadata also records `feature_selection_enabled`, `feature_selection_target_specific`, `feature_selection_profile`, and `selected_features_by_target` for runtime auditability.
- Current note: `_build_sequence_batch()` now produces zero-padded sequences for players with fewer than `seq_len + 1` games instead of skipping them entirely.
- Current note: `_feature_cols_for_target(target)` returns the per-target feature list when smart selection is active, falling back to `self.feature_cols` otherwise. Used by `_train_catboost_parallel`, the empty-data fallback path, and the constant-regressor fallback.
- Current note: `apply_feature_selection_manifest(payload)` parses a `SelectionManifest` dict and populates `self.target_feature_cols` plus audit fields (`feature_selection_manifest`, `feature_selection_profile`).
- Current note: `_save_blend_weights()` now also persists the training-time blend to `WeightStore` (versioned JSON) so the `ModelManager` bootstrap path can pick it up on the next load.

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

### `src/training/diagnostics.py` (NEW — 2026-06-14)

- Role: fast crash-stage diagnostics for the training pipeline without running full model training.
- Key objects/functions:
  - `DiagnosticConfig` — dataclass controlling whether diagnostic mode is active and which stage to stop after.
  - `diagnostic_stage(name, config)` — context manager that wraps a stage block with `[TRAIN-DIAG] START/OK/FAILED` markers, exception handling, and optional early exit.
  - `print_data_summary(merged_df, full_df)` — prints row counts, column counts, and target-column presence.
  - `print_selection_summary(manifest)` — prints per-target selected feature counts.
  - `STAGES_ORDERED` — canonical stage ordering used by `train.py --stop-after`.
- Called from `train.py` when `--diagnose` is active.
- Fully tested: `tests/test_training/test_diagnostics.py` (28 tests).
- Risk level: low. New code, isolated from existing training logic.

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
- Current note: `load_models()` now bootstraps `EnsembleWeights` from `WeightStore` after the legacy `blend_weights.pkl` is loaded. If a `current.json` exists, `use_ensemble_weights()` overrides the legacy blend so the runtime uses data-driven weights. The bootstrap is non-fatal — if the WeightStore is missing or the read fails, the legacy blend is used. (DR-030, 2026-06-04.)
- **Load-time feature alignment (2026-06-04):** `predict_player_stats` calls `load_expected_feature_cols(models_dir)` to read the saved column list and `align_feature_frame(df, expected_cols)` to reorder / drop extras before the leakage-safe selector runs. Inference frames with missing or non-numeric features raise a typed `FeatureSchemaContractError`.

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
- **Schedule contract wiring (2026-06-04):** `simulate_season` converts the schedule frame to `ScheduleGame` records via `schedule_rows_to_games(...)` before iterating matchups (both ThreadPoolExecutor and sequential paths). The game iterator uses the typed `home_team`, `away_team`, and `game_date` attributes rather than raw dict lookups.

### `src/simulation/report_generator.py`

- Prints projections and exports to CSV under `data/sim_results/`.
- High-value methods:
  - `export_player_projections` — writes the strict 6-stat x 14-column schema (mean, `{STAT}_P10`, `{STAT}_P50`, `{STAT}_P90`, `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`, `{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, `{STAT}_CONFIDENCE_SCORE` for all 6 stats, plus `DATA_QUALITY` column with values `FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`). Calls `validate_projection_frame(...)` on the assembled DataFrame before writing the CSV.
  - `export_to_csv`
  - `display_quick_summary`
  - `_data_quality_from_result` — static helper deriving quality from input_health metadata
  - `_enrich_with_distributions` (NEW) — derives and appends distribution parameters from quantile columns via `DistributionFitter`
  - `_ensure_confidence_interval_columns` (NEW) — keeps interval/confidence columns present even when calibration artifacts are absent
- Risk level: high (export schema impacts query layer).
- Current caution: any new projection column must be reflected in `src/contracts/projections.py` and query fixtures before it is considered complete.

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
- **New**: `calculate_empirical_crps()` — fast O(n log n) CRPS via Gini mean difference. Evaluates probabilistic forecast quality (lower is better). Used by `optimize_variance.py`.
- **New**: `backtest_result_to_json_dict(result)` — stable JSON serializer for `BacktestResult` payloads. Produces `{"targets": {...}, "overall": {...}}`. Used by `backtest.py --json-output` and any downstream tool that consumes backtest metrics without scraping terminal output. Tolerant to dataclass and `__dict__`-shaped inputs.
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
- Default location: `models/blend_weights/` with `current.json` pointer. The legacy `blend_weights.pkl` (next to it under `models/`) is still written by `TrainingPipeline._save_blend_weights()` for backward compatibility with `validate_runtime_artifacts()`, and `WeightStore.migrate_from_legacy()` reads from it on first run. (Earlier brain revisions called this `data/weights/` — corrected 2026-06-04 to match the actual default.)
- Risk level: medium — is the single source of truth for active ensemble weights.

### `src/evaluation/drift_detector.py`

- Role: statistical process control for model performance.
- Key types: `DriftDetector`, `DriftStatus`, `DriftReport`.
- Flags when rolling MAE exceeds 2σ above historical baseline.
- Distinguishes minor drift (retune weights) from major drift (retrain models).
- Risk level: medium — needs live backtest data to establish baseline.

### `src/evaluation/__init__.py`

- Re-exports: `BacktestResult`, `TargetMetrics`, `BacktestRunner`.
- **New (2026-06-04)**: re-exports the smart feature selection types — `AblationReport`, `FeatureGroupAblator`, `GroupScore`, `SHADOW_COLUMNS`, `ShadowFeatureFilter`, `ShadowFilterResult`, `ProfileConfig`, `SelectionManifest`, `SelectorConfig`, `SmartFeatureSelector`, `TargetSelection`, `load_manifest`.

### `src/evaluation/feature_group_ablation.py` (NEW — 2026-06-04)

- Role: train-and-compare loop that scores feature groups by per-target MAE delta.
- Key types: `FeatureGroupAblator`, `AblationReport`, `GroupScore`, `filter_group_columns`.
- High-value method: `FeatureGroupAblator.run(df, feature_cols, group_columns, targets, val_ratio, min_gain)` — trains a baseline `HistGradientBoostingRegressor` plus a leave-one-out model per group, then averages MAE deltas across targets.
- `AblationReport.average_score_by_group()` provides the broadcast scores that feed `SmartFeatureSelector._backtest_gain_per_feature`.
- Risk level: low to medium. Uses sklearn models (not CatBoost) for speed — results are an approximation, not the exact CatBoost gain.

### `src/evaluation/shadow_feature_filter.py` (NEW — 2026-06-04)

- Role: cheap supervised screening that uses random control columns as a noise floor.
- Key types: `ShadowFeatureFilter`, `ShadowFilterResult`, `ShadowImportance`, `SHADOW_COLUMNS`.
- Inject columns: `SHADOW_RANDOM_NORMAL`, `SHADOW_RANDOM_UNIFORM`, `SHADOW_PERMUTED_TARGET`.
- Drops real features whose importance falls below the median shadow importance.
- Risk level: low. Pure sklearn — no GPU/CatBoost dependency.

### `src/evaluation/smart_feature_selector.py` (NEW — 2026-06-04)

- Role: combines 5 signals into a per-target final score and writes a per-target `SelectionManifest`.
- Key types: `SmartFeatureSelector`, `ProfileConfig`, `SelectorConfig`, `SelectionManifest`, `TargetSelection`, `WEIGHTS`, `load_manifest`.
- Final score: `0.40 * backtest_gain + 0.25 * stability + 0.20 * catboost_importance + 0.10 * permutation_importance - 0.05 * missingness_penalty`.
- Profiles: `fast` (group ablation only), `balanced` (group ablation + per-target pruning + shadow filter), `max_accuracy` (everything + time-stability check).
- Output: `models/feature_selection_manifest.json` with `selected_features_by_target`, `selected_features_global`, per-target scores, dropped features, shadow-dropped features, and metadata.
- Risk level: low to medium. `WEIGHTS` are documented as the ticket spec; revisit if signals prove noisy on real data.

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
  - `load_projections` — calls `validate_projection_frame(...)` on every load; raises the typed `ProjectionSchemaContractError` if the CSV is missing distribution, interval/confidence, or `DATA_QUALITY` columns. Legacy CSVs (pre-2026-06-12) are no longer loadable without regeneration.
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
  - `run_copula_simulation` — correlated multi-stat draws via archetype copula (NEW)
  - `evaluate_calibration`
- Accepts optional `CovarianceCache`; lazy-loads default on first use.
- Well-covered by tests relative to other areas.
- Risk level: medium.

### `src/query/distribution_fitter.py` (NEW — 2026-05-21)

- Derives distribution parameters from P10/P50/P90 quantile outputs.
- Key types: `DistributionFitter`, `StatDistribution` (mean, std, skew, zero_prob, lambda_param).
- Continuous stats → skew-normal approximation. Count stats → ZIP approximation.
- Used by `ReportGenerator._enrich_with_distributions()` and `ProbabilityCalculator.run_copula_simulation()`.
- Risk level: low. Pure math, no external dependencies.

### `src/query/empirical_covariance.py` (NEW — 2026-05-21)

- Archetype-conditioned 6x6 correlation matrices from residual analysis.
- Key types: `CovarianceCache` — `build_and_save()`, `load()`, `get_correlation()`.
- PSD-clipped eigenvalues ensure valid Cholesky decomposition for copula simulation.
- Falls back to identity matrix when archetype or cache unavailable.
- Persisted to `data/cache/archetype_covariances.npz`.
- Risk level: low to medium. Non-fatal fallback to identity.

### `src/query/query_parser.py`

- Parses user phrases into supported query operations.
- Risk level: medium.

## Contracts Layer (NEW — 2026-06-04)

### `src/contracts/`

- Role: inter-step artifact contract validation. The seam that lets training, simulation, optimizer, and selector swap independently while keeping a single source of truth on what each step must produce and consume.
- Important modules:
  - `artifacts.py` — `ArtifactContract` (frozen dataclass: `models_dir`, `transformer_required`, `max_age_hours`) + `validate_runtime_artifacts()`. Canonical target set is the 6 stats: `PTS, REB, AST, STL, BLK, TOV`. Validates per-target `*_catboost.cbm` and `*_metadata.joblib`, plus the shared `feature_schema.pkl`, `feature_cols.pkl`, `blend_weights.pkl`, `model_stack_metadata.pkl`, and (when transformer required) `attention_transformer.pkl`. Also unpickles `model_stack_metadata.pkl` and verifies its `targets` field matches the canonical set.
  - `features.py` — `FeatureSchema` contract used by `FeatureSelector` and the trainer to align training-time and inference-time feature layouts.
  - `projections.py` — `validate_projection_csv()` checks the `player_projections_*.csv` schema, including `DATA_QUALITY`, all six stat distribution columns, and all residual interval/confidence columns.
  - `schedule.py` — schedule input contract for the simulator.
  - `errors.py` — `ContractError` base + `ArtifactContractError`, `FeatureSchemaContractError`, `ProjectionSchemaContractError`, `ScheduleContractError` typed exceptions.
  - `__init__.py` — re-exports the five error types.
- Risk level: medium — this is the contract seam for the whole pipeline; any new mandatory artifact or schema change must be added here in lockstep with the producers and consumers.

### `check_contracts.py` (root)

- Role: standalone CLI to validate the artifact contract and projection CSV between pipeline steps.
- CLI: `python check_contracts.py [--models-dir models] [--projection-csv <path>] [--transformer-required]`. Exits 0 on success, raises the typed `ContractError` on failure and prints a human-readable summary.
- Both `train.py` and `simulate_season.py` invoke `validate_runtime_artifacts()` at startup. `train.py` re-invokes it at the bottom of its flow as a post-train check (DR-031, 2026-06-04). Use `check_contracts.py` to debug contract failures in isolation.

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
- **New (2026-06-04)**: `FeatureSelector.select_features_for_target(df, target, allowed_features=None)` — builds a target-specific `FeatureSchema` from an allow-list. When `allowed_features` is provided, the leakage-safe filter is skipped (trusting the upstream smart selector) but the schema still drops non-numeric / unsafe columns. When `allowed_features` is None, the call falls through to the existing `select_features()` path.

## Tests

### `tests/`

- Best entry point for understanding expected behavior before changing core logic.
- Areas with meaningful coverage:
  - preprocessing
  - training pieces
  - model wrappers
  - query logic
  - portions of simulation
  - **smart feature selection** (`tests/test_evaluation/`) — new in 2026-06-04
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
  - **`tests/test_evaluation/test_smart_feature_selector.py` (NEW)** — covers the `SmartFeatureSelector` end-to-end (manifest round-trip, per-target pruning, shadow filter integration, group ablation, time stability, `load_manifest`).
  - **`tests/test_evaluation/test_backtest_json_output.py` (NEW)** — covers `backtest_result_to_json_dict` for both dataclass and `__dict__`-shaped inputs and the `overall` aggregate calculation.

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
  - now contains 8 active modules: `metrics.py`, `backtest_runner.py`, `ensemble_optimizer.py`, `weight_store.py`, `drift_detector.py`, `feature_group_ablation.py`, `shadow_feature_filter.py`, `smart_feature_selector.py`
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
