# Code Rules

## Purpose

- These rules are derived from the current repository shape and are intended to keep future changes compatible with the active CLI and artifact contracts.
- When the repo is inconsistent, this file chooses a stable default to follow going forward.

## Hard Rules

- Preserve the top-level CLI workflows:
  - `update_data.py`
  - `train.py`
  - `simulate_season.py`
  - `query_prob.py`
  - `clear_cache.py`
- When a notebook or wrapper script launches one of the top-level CLIs, capture stdout and stderr before re-raising on failure so the real traceback is visible to the operator.
- When a notebook or wrapper script launches `train.py`, resolve the actual repo checkout separately from Drive-backed data/models; never derive the script path from the storage path for outputs.
- Treat filesystem artifact names and locations as API contracts unless you update every consumer in the same change.
- Update `/project-brain` after every meaningful code change.
- Do not silently change target-stat semantics, query names, or exported CSV schemas.
- Keep the named training presets in `config/default.yaml`, `src/training/presets.py`, and `train.py` synchronized; preset drift across those files is a contract bug, not a cosmetic difference. The registered presets are `small`, `laptop_quality`, and `full`.
- Do not change the canonical six-target training contract when adding presets. `small` and `laptop_quality` may change stack breadth, but every preset must still train `PTS`, `REB`, `AST`, `STL`, `BLK`, and `TOV`.
- New presets must use the canonical feature-group names from `ALL_FEATURE_GROUPS` in `src/training/presets.py` (e.g. `context`, `rest_density`, `archetype` — not `contextual`, `rest_game_density`, `player_archetype`). Non-canonical names are silently dropped by `FeatureEngineer._should_run_group`.
- A preset may turn on smart feature selection by default via its `feature_selection` / `feature_selection_profile` fields. `train.py` honors the preset default when `--feature-selection` is not explicitly passed, and resolves the profile from CLI flag → preset → global config → `balanced`. In `--diagnose` mode the expensive `SmartFeatureSelector` work is skipped so preflight stays fast.

## Naming And Data Conventions

### Canonical target names

- Use uppercase stat names in model/training code:
  - `PTS`
  - `REB`
  - `AST`
  - `STL`
  - `BLK`
  - `TOV`
- Lowercase names such as `pts` or `reb` are query/UI aliases, not the canonical modeling contract.

### File and artifact naming

- Keep model artifacts compatible with `src/models/model_manager.py`.
- If a model format or filename changes, update:
  - the training writer
  - the runtime loader
  - any existence checks in top-level scripts
  - tests covering the contract
- `train.py` must not report success if the required runtime artifact set is incomplete.
- The minimum enforced runtime set is:
  - per-target CatBoost backbone artifacts plus metadata
  - `feature_schema.pkl`
  - `feature_cols.pkl`
  - `blend_weights.pkl`
  - `model_stack_metadata.pkl`
  - `attention_transformer.pkl` when the Transformer path is enabled
- If `blend_weights.pkl` contains a non-zero Transformer weight, `attention_transformer.pkl` must be present and loadable. Do not silently fall back to partial-blend predictions.

### Config naming

- Use one config attribute naming style consistently inside a class.
- Stable default for this repo: prefer lowercase Python attribute names for instance state, even if config YAML keys are uppercase-like in concept.
- Do not mix `self.max_retries` and `self.MAX_RETRIES` in the same runtime object.

## Module Organization Rules

### Preprocessing

- New engineered features should be added as a `FeatureGroup` in `src/preprocessing/features/` unless there is a strong reason not to.
- Register new feature groups through the active feature-engineering flow instead of embedding ad hoc feature logic in `train.py`.
- When `train.py` or notebook launchers need feature-group ablation filters, construct the orchestrator through `build_feature_engineer(...)` instead of passing compatibility-sensitive kwargs like `disable_groups` directly into `FeatureEngineer(...)`.
- When `FeatureEngineer` is used for ablation benchmarking, use `build_feature_engineer(...)` there as well; the benchmark path is part of the compatibility boundary and should not assume every checkout accepts `disable_groups`.
- When a feature group reads on-disk files that are NOT part of the input DataFrame (e.g. `data/injury_history.csv`, lifecycle precompute CSVs), it MUST declare them via `FeatureGroup.external_files()` so the feature-engineering cache key invalidates when those files change. Failing to do so risks returning stale cached features (DR-034).
- When adding features:
  - document the feature purpose
  - verify column naming consistency
  - consider inference compatibility
- Keep player-style archetype outputs in `PlayerArchetypeFeatureGroup` and keep them numeric (`ARCHETYPE_*` / `SIMILARITY_TO_*`) so `FeatureSelector` can persist them cleanly; if the template set changes materially, bump the feature schema version and update the contract tests together.

### Training

- Keep `src/training/pipeline.py` as the primary training orchestration path unless a deliberate redesign is documented.
- Avoid duplicating training entry logic in new modules without a clear ownership split.
- Heavy training imports should remain lazy when practical to keep non-training commands lightweight.
- If a preset needs a recent-history window, prefer the existing `SEASON_ID` column when it is present and skip the trim rather than inventing a new date heuristic.

### Models And Inference

- `src/models/model_manager.py` is the runtime model-loading boundary for simulation.
- New runtime model artifacts should be loaded through `ModelManager` or a clearly documented successor, not by scattering ad hoc file reads across the simulator.
- The blend-weight / Transformer artifact contract is enforced at load time: if blend weights expect a Transformer and the artifact is missing or unloadable, `ModelManager.load_models()` must raise rather than produce silently uncalibrated predictions.
- Keep Transformer validation/runtime inference on the eager model path by default. If `torch.compile` is re-enabled for the Transformer, it must stay behind an explicit safety flag and should never be the only validation path.
- When touching CUDA attention code, prefer math SDPA fallback controls for validation and add regression coverage for the eager path.
- `model_stack_metadata.pkl` is part of the shared runtime contract and may now include the selected training preset and feature-group list; keep it in sync with any preset changes.
- `TrainingPipeline.cat_features` must be filtered to only columns present in `feature_cols` before being passed to CatBoost. The `FeatureSelector` lists `PLAYER_ID`/`TEAM_ID`/`OPPONENT_ID` as categorical columns, but those are in `EXCLUDE_ALWAYS` and never appear in `feature_cols`. Passing unfiltered string `cat_features` to CatBoost after converting the frame to a numpy float array causes a `CatBoostError`. The model_manager already filters at load time — the training pipeline must do the same.
- `TrainingPipeline.train()` enforces a `len(fit_df) >= 1000` minimum. Smoke-test fixtures must produce at least 1000 fit rows after the chronological split (use `tests/fixtures/laptop_quality/data/` as a reference fixture with ~2160 player-game rows).
- Both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) must use zero-padding for players with fewer than `seq_len` context games. Do not revert to skipping short players — the zero-padding behavior is now covered by regression tests.
- When changing `SIZE_TIER_SPECS` in `src/config/model_config.py`, remember that `seq_len` and `max_seq_length` affect both the Transformer model's sequence window and the training pipeline's batch construction. Changes here must be reflected in both `_create_sequences()` and `_build_sequence_batch()`.

### Simulation

- `src/simulation/game_simulator.py` is a critical orchestrator. Keep interface changes small and documented.
- New simulation logic should use typed dataclasses from `sim_types.py` rather than raw dicts.
- If a simulation path becomes obsolete, remove it or mark it clearly. Do not leave large unreachable blocks without explanation.
- New simulation components (phase simulation, archetype inference, role sampling) should be independent modules in `src/simulation/` rather than embedded in GameSimulator.
- **Strict mode**: `GameSimulator(strict_mode=True)` raises `RuntimeError` on degraded optional InputHealth sources. When adding new optional context sources, integrate them into the `input_health` contract and check them in strict mode.
- **Data quality**: `ReportGenerator.export_player_projections()` must include the `DATA_QUALITY` column. New simulation result metadata that affects projection trustworthiness should be reflected here.
- **Distribution enrichment**: `ReportGenerator.export_player_projections()` must also include distribution parameter columns (`{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`) for all 6 stats via `_enrich_with_distributions()`. If the schema of these columns changes, update `ProbabilityCalculator.run_copula_simulation()` accordingly.
- **Residual intervals/confidence**: `ReportGenerator.export_player_projections()` must include `{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, and `{STAT}_CONFIDENCE_SCORE` for all six stats. Missing `models/calibration/` artifacts should produce stable `NO_EDGE`/blank interval output, not a simulation failure.

### Lifecycle & Bio-Mechanical (NEW)

- `src/lifecycle/` owns player aging, career trajectory, and injury risk computation.
- `BIanusAgingModel` and `KANAgeModel` must be precomputed at training startup (before feature engineering) so their caches exist for `AgingCurveFeatureGroup` and `KANAgingFeatureGroup` to load.
- Lifecycle precomputation is non-fatal. Missing player bio data defaults aging features to neutral (1.0 factor) and injury risk features to near-zero. Never crash training because of missing bio data.
- KAN aging always runs on CPU (`device='cpu'`) to avoid CUDA context contention with CatBoost/Transformer training.
- New lifecycle caches (`aging_curves.csv`, `kan_aging_outputs.csv`) go in `data/cache/`. Delete them to force recomputation after retraining.
- `PlayerBioScraper` fetches from NBA API `commonplayerinfo` endpoint. Cache results to `data/player_bios.csv`. Called from `update_data.py` — non-fatal on failure.
- `InjuryHistoryLogger` appends and deduplicates by (PLAYER_ID, DATE, INJURY_TYPE). Never overwrite the full history — only append.
- `update_data.py` outputs Parquet files alongside CSVs for GPU-direct storage reads. Both formats must stay in sync.

### Nexus Multi-Modal Model (NEW)

- `NexusModel` is implemented and import-tested but is NOT the active training path. CatBoost + Transformer remains the active stack.
- If Nexus is wired into the training pipeline, update all decision records, CODE_RULES, and the simulation artifact contract simultaneously.
- The nexus loss (`GaussianNLLLoss`) may be ported as a secondary loss even if the full Nexus model stays inactive.
- The simplified SSM block in `SimplifiedSSMBlock` uses standard PyTorch ops as a fallback. If CUDA `mamba_ssm` kernels become available, the fallback must preserve exact parameter shapes for weight-porting.

### GPU Feature Engineering (NEW)

- `FeatureEngineerGPU` mirrors `FeatureEngineer`'s public API for drop-in compatibility. Keep the two engines aligned when adding new feature groups.
- The GPU engine transparently falls back to CPU when cuDF is unavailable or CUDA is missing. Never crash feature engineering because of GPU unavailability.
- Complex feature groups (Python loops, scipy) execute on CPU after pandas conversion — do not attempt full GPU port of groups that are fundamentally iterative.

### Season-Context Feature Groups (NEW)

- `src/preprocessing/features/season_phase.py` — early-season ramp-up (cap `DAYS_SINCE_SEASON_START` at 30 days so the feature isolates the opening-month effect). `GAMES_WITH_CURRENT_TEAM` resets on trade — ensure `PLAYER_ID` and `TEAM_ID` columns are present and sorted by `GAME_DATE` before the groupby.
- `src/preprocessing/features/team_motivation.py` — always shift `TEAM_CUMULATIVE_WIN_PCT` by 1 to prevent label leakage (the model should see the team's record *before* the current game). `WL` column must be present with values `'W'`/`'L'`.
- `src/preprocessing/features/postseason_context.py` — checks both `SEASON_TYPE` and `GAME_TYPE` columns for playoff strings (`'Playoffs'`, `'Postseason'`, `'4'`). If neither column exists, all games default to regular season.
- Season-context groups are in the `full` preset but intentionally excluded from `small` and `laptop_quality` to preserve their iteration speed.
- The rest cap in `RestGameDensityFeatureGroup` (`DAYS_SINCE_LAST_GAME.clip(upper=14.0)`) applies to ALL games, not just off-season gaps. Do not remove or raise the cap without validating the effect on B2B detection (a 1-day gap still correctly produces B2B=1).

### Phase-Aware Drift Detection (NEW)

- `DriftDetector` now maintains a single `_history` list with a `phase` tag on each entry (vs separate lists per phase). The `detect()` method filters the list by phase at query time.
- When passing `phase` to `detect()`, the function filters history to that phase only. If no phase-specific data exists, it falls back to the full history.
- The `_infer_phase_from_date()` heuristic (Apr 15–Jun 20 = PLAYOFF) is a rough approximation. Explicitly pass `phase='PLAYOFF'` during postseason backtesting for accuracy.
- The phase tag is persisted in `drift_state.json` so it survives restarts — old (pre-phase) entries will not have a `phase` key and are treated as `REGULAR`.

### Residual Correction Monitoring (NEW — 2026-06-18)

- `monitor_residual_corrections.py` is the top-level CLI entry point. It delegates to `ResidualMonitor` (pure evaluation) and `residual_report.py` (file I/O). Keep the separation: the monitor does not write files; the report writer does not evaluate.
- `MonitoringThresholds` is the single source of truth for HELPING/NEUTRAL/HURTING classification rules. If the rules change (e.g., widening the neutral band), update `MonitoringThresholds.status_for_pct()` and the ticket spec in lockstep.
- The overall_status aggregation rule is conservative: HURTING > INSUFFICIENT_DATA > NEUTRAL > HELPING. A single hurting target dominates the report. Do not soften this — the report is a safety check.
- JSON reports must never emit NaN/Infinity tokens (invalid strict JSON). `_json_safe()` in `residual_report.py` converts non-finite floats to `None` before serialization with `allow_nan=False`. Any new report format added to the module must do the same.
- The monitor supports two input column paths: `CORRECTED_PREDICTION` (direct) or `RESIDUAL_CORRECTION` + `BASE_PREDICTION` (derived). When both are present, NaN rows in `CORRECTED_PREDICTION` fall back row-by-row to `BASE + RESIDUAL_CORRECTION`. Do not drop rows with partial coverage.
- Rolling windows use `min_window_rows` (not `min_rows`) for their status threshold so that short windows are not silenced by a high global row requirement. This is a deliberate design choice — short windows should surface recent degradation even with limited data.
- `_resolve_input_path()` in the CLI entry point is strict about explicit `--input`: if the user passes an explicit path, it must exist. No silent fallback to config defaults when the user meant a specific file.

### Evaluation And Optimization (NEW)

- `src/evaluation/` is the evaluation and optimization subsystem. All backtesting, weight optimization, drift detection, and weight versioning lives here.
- Backtest results flow through `BacktestResult` and `TargetMetrics` dataclasses — do not return raw dicts or ad hoc metrics.
- `BacktestRunner` depends on `ModelManager` for predictions and `DataLoader` for actual box scores. Both must be available for backtesting to work.
- `EnsembleOptimizer` uses scipy.optimize with 13 parameters. If the parameter space changes, update `_TUNABLE_DIMS` and all code that builds/flattens weight vectors.
- `WeightStore` is the single source of truth for active ensemble weights. Never bypass it with direct `blend_weights.pkl` writes.
- Weight versions are numbered and stored under `models/blend_weights/` (NOT `data/weights/` — earlier brain revisions had the wrong path; corrected 2026-06-04). The `current.json` pointer indicates the active version.
- `optimize_weights.py` always writes through `WeightStore`. Use `--rollback N` to revert, not manual file edits.
- `DriftDetector` uses SPC (2σ threshold). The baseline window and detection threshold are configurable through `config/default.yaml` under `self_optimization:`.
- Keep the blend-weight / Transformer artifact contract enforced at load time. `ModelManager.set_weights()` must validate the new weights before applying them.
- `backtest_result_to_json_dict()` in `metrics.py` is the stable JSON serializer for `BacktestResult` payloads. Downstream tooling must consume the `{"targets": {...}, "overall": {...}}` schema from `--json-output` rather than scraping terminal output.
- `ModelManager.load_models()` bootstraps `EnsembleWeights` from `WeightStore` after the legacy blend is loaded. If the store has no `current.json`, the legacy blend stands. The bootstrap is non-fatal.

### Contracts Layer (NEW)

- `src/contracts/` is the seam for inter-step artifact contract validation. Every producer/consumer pair across the pipeline must be expressible in this layer, not scattered as ad hoc `os.path.exists()` checks.
- `contracts/artifacts.py::ArtifactContract` is the canonical list of required runtime artifacts. The required set is: per-target `*_catboost.cbm` + `*_metadata.joblib` for all 6 stats, plus `feature_schema.pkl`, `feature_cols.pkl`, `blend_weights.pkl`, `model_stack_metadata.pkl`, and (when transformer is required) `attention_transformer.pkl`. If a new mandatory artifact is introduced, add it to `_required_files()` and update the producer/consumer in the same change.
- `contracts/artifacts.py::validate_runtime_artifacts()` is invoked at the top of both `train.py` and `simulate_season.py`. Do not bypass it with a direct file existence check — fail-loud is the contract.
- `contracts/projections.py::validate_projection_csv()` is invoked when the query layer loads a `player_projections_*.csv` export. The CSV must include the `DATA_QUALITY` column (DR-025).
- `contracts/features.py::FeatureSchema` is the single source of truth for training-time and inference-time feature layout. The trainer writes the schema, the loader reads it; do not re-derive column expectations implicitly.
- `contracts/errors.py` defines the typed exception hierarchy. New contract types should subclass `ContractError` and live here.
- Standalone entry: `check_contracts.py` (root) — CLI for debugging contract failures in isolation. Exits 0 on success, raises the typed exception on failure.
- When you add a new mandatory runtime artifact, schema column, or contract type, update `src/contracts/` first, then update the producer and consumer. Never introduce a contract in only one of the three places.
- **Contract wiring at seams (DR-031, 2026-06-04):** introducing a new contract is not complete until the production call sites are wired:
  - Scraper read paths must call their frame normalizer on every cache and API path. For the schedule contract, every `ScheduleScraper` method that returns a DataFrame must call `normalize_schedule_frame(...)` (empty frames are skipped).
  - The simulator must convert raw schedule rows into the contract's typed records before iterating. `SeasonSimulator.simulate_season` calls `schedule_rows_to_games(...)` and reads the typed `home_team`, `away_team`, `game_date`, `game_id` attributes.
  - The projection loader must call `validate_projection_frame(...)` on read; the report generator must call it on write. Both raise / re-raise the typed `ProjectionSchemaContractError`.
  - `ModelManager.predict_player_stats` must call `load_expected_feature_cols(models_dir)` and `align_feature_frame(df, expected_cols)` before prediction so reordered or extra-column frames are coerced to the trained layout.
  - `train.py` must call `validate_runtime_artifacts(ArtifactContract(...))` at the bottom of its training flow as a post-train check.
  - Failure to update any of these call sites in the same change as a new contract is itself a contract violation.
- **Strict projection schema (DR-031/DR-032, updated 2026-06-12):** projection CSVs written by `ReportGenerator` and loaded by `ProjectionLoader` must include all 6 stats x 14 columns (`{STAT}`, `{STAT}_P10`, `{STAT}_P50`, `{STAT}_P90`, `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`, `{STAT}_INTERVAL_80_LOW`, `{STAT}_INTERVAL_80_HIGH`, `{STAT}_INTERVAL_90_LOW`, `{STAT}_INTERVAL_90_HIGH`, `{STAT}_CONFIDENCE`, `{STAT}_CONFIDENCE_SCORE`) and a `DATA_QUALITY` column with values in {`FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`}. Confidence labels must be one of {`HIGH`, `MEDIUM`, `LOW`, `NO_EDGE`}. `tests/test_query/test_six_stat_contract.py::TestMissingStatColumnsFailsLoudly::test_missing_tov_columns_fails_loudly` and `tests/test_correction/test_calibration.py` guard the load/export path. Do not reintroduce defaults for missing columns — the schema is strict and legacy CSVs must be regenerated.

### Residual Correction And Calibration (NEW)

- Residual correction artifacts live under `models/residual/`; residual interval artifacts live under `models/calibration/`. Keep those directories separate.
- `calibrate_residual_intervals.py` is the top-level calibration entry point. It must consume `data/evaluation/residual_training.parquet` and write JSON artifacts through `ResidualIntervalCalibrator`, not ad hoc files.
- Runtime interval loading must remain best-effort. `CalibrationIntervalStore` should disable itself safely when files are missing or malformed; `ModelManager.predict_player_stats(..., include_confidence=True)` should not crash just because calibration has not been run.
- Prefer corrected residual errors for calibration when available: `CORRECTED_PREDICTION`, then `CORRECTED_ERROR`, then `ERROR`.
- Confidence labels are coarse trust/risk signals, not probability recommendations. Keep scoring logic in `ConfidenceScorer` so query, simulation, and future reporting do not invent conflicting label rules.

### Smart Per-Target Feature Selection (NEW)

- `SmartFeatureSelector`, `FeatureGroupAblator`, and `ShadowFeatureFilter` live in `src/evaluation/`. They are training-time helpers, not runtime predictors.
- The selector writes a `SelectionManifest` to `models/feature_selection_manifest.json`. The manifest is the contract between selector, training, and downstream inference. Do not write a parallel manifest format.
- `TrainingPipeline._feature_cols_for_target(target)` is the single source of truth for "what features does this target see during training". It must fall back to the canonical `self.feature_cols` list when no manifest is loaded, preserving the original contract.
- `TrainingPreset` may carry `feature_selection` and `feature_selection_profile`. If the YAML config sets `feature_selection.enabled: true` on a preset (as `laptop_quality` does), the manifest is applied to the pipeline before training. `train.py` also honors the preset default at runtime when `--feature-selection` is not explicitly passed, and skips the expensive selector work in `--diagnose` mode.
- The selector's signal weights (`WEIGHTS` in `smart_feature_selector.py`) are the ticket spec. If you change them, update DR-028 and the docstrings in lockstep.
- Selector failure (insufficient samples, fitting error, etc.) must be non-fatal. `train.py` and the pipeline fall back to the full feature set with a warning.
- `FeatureGroupAblator` uses `HistGradientBoostingRegressor` (sklearn) for speed — its scores are an approximation, not the exact CatBoost gain. Do not assume equivalence with the production trainer.
- `ShadowFeatureFilter` injects `SHADOW_*` columns. Real features must never share these prefixes (they are reserved in `FeatureSelector.EXCLUDE_ALWAYS`).
- `SelectionManifest.target_specific` controls per-target vs. global feature lists. When `False`, every target uses the same `selected_features_global` list. When `True`, the per-target list in `selected_features_by_target` is used. The pipeline respects this via `_feature_cols_for_target`.
- `FeatureSelector.select_features_for_target(df, target, allowed_features=None)` is the inference-time hook. When the caller passes an allow-list (e.g. from the manifest), the leakage-safe filter is bypassed but the schema still drops non-numeric / unsafe columns.
- `model_stack_metadata.pkl` is the audit record for smart selection. When `feature_selection_enabled=True`, the manifest's per-target lists and the active profile are persisted alongside the existing preset / feature-group metadata.

### Query

- The query layer should consume exported projection artifacts rather than reimplement full model inference.
- If the exported schema changes, update `src/query/projection_loader.py` and query tests in the same change.
- `ProbabilityCalculator.run_copula_simulation()` uses archetype-conditioned correlation matrices from `CovarianceCache`. Always provide an archetype label — "GLOBAL" is the universal fallback.
- Distribution parameters for copula simulation should come from `DistributionFitter.fit_from_quantiles()` (which works with P10/P50/P90 quantile outputs) rather than from ad hoc heuristic calculations.
- `CovarianceCache` must always have a sensible fallback: identity matrix when no cache exists, archetype-specific matrix when available, GLOBAL matrix as first fallback, identity matrix as last resort.
- The CRPS function in `metrics.py` (`calculate_empirical_crps`) is a fast O(n log n) Gini approximation. For small-n regimes (< 100 samples) where exact CRPS would differ materially, use the exact formula instead.
- `optimize_variance.py` is the variance counterpart of `optimize_weights.py`. It tunes volatility (std multipliers) via CRPS rather than mean accuracy via MAE. Both should be run after significant training changes.

### Refactor Rules

- Avoid cross-cutting rewrites unless the user explicitly asks for architectural change.
- Before moving or renaming important files, update:
  - `project-brain/FILE_MAP.md`
  - `project-brain/ARCHITECTURE.md`
  - any import shims or callers
- If a refactor changes subsystem ownership or active flow, add a decision record in `project-brain/DECISIONS.md`.
- Remove compatibility shims only after verifying all active callers.

## Testing Expectations

- Run targeted pytest coverage for the modules you change.
- Prefer adding regression tests when touching:
  - artifact persistence
  - scraper config handling
  - CSV export schemas
  - feature schema reconstruction
- Do not rely only on unit tests when changing a contract seam between:
  - training and model loading
  - simulation and query
  - scraper outputs and simulator inputs

## Performance Rules

- Avoid repeated column-by-column pandas insertion for large feature sets when a batched concat or vectorized operation is feasible.
- In `src/preprocessing/features/*`, prefer collecting wide feature outputs in temporary dicts/DataFrames and attaching them with one `pd.concat(axis=1)` per feature group.
- Be cautious with extra work inside simulation loops; seemingly small per-player or per-draw overhead multiplies quickly.
- Keep lazy imports for PyTorch-heavy code paths when they reduce startup overhead for non-training commands.
- Feature engineering is the dominant CPU bottleneck; the in-place parquet cache (`cache/training/*.parquet`, DR-034) is active by default in `DataPipeline` and `ModelManager`. Never construct a production `FeatureEngineer` without `cache_dir` unless you intentionally want to recompute. The cache key already invalidates on input data, FE config, and external-file changes, so prefer letting it reuse results over hand-recomputing.

## Reliability Rules For Scrapers And External Data

- Expect third-party sources to drift.
- Prefer explicit, well-logged fallbacks over accidental exceptions or silent bad state.
- Cache behavior should be deterministic and easy to inspect on disk.
- If a scraper is effectively inactive, document that reality instead of implying full support.

## Documentation Rules

- Every meaningful behavior change must update the relevant `/project-brain` docs.
- `CURRENT_STATE.md` should always reflect the latest known repo health.
- `TASKS.md` should be updated when work is completed or new debt is found.
- `KNOWN_BUGS.md` should be updated when a defect is confirmed, mitigated, or fixed.

## Stable Defaults For Inconsistent Areas

- Prefer one clear active path over multiple semi-maintained alternatives.
- Prefer explicit schema persistence over implicit reconstruction.
- Prefer small pure functions or focused classes over giant orchestration objects, except where existing architecture already centralizes critical flow.
- Prefer deleting dead code to leaving unreachable legacy blocks in place.
