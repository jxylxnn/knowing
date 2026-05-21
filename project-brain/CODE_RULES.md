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
- Keep the named training presets in `config/default.yaml`, `src/training/presets.py`, and `train.py` synchronized; preset drift across those files is a contract bug, not a cosmetic difference.
- Do not change the canonical six-target training contract when adding presets. `small` may change stack breadth, but it must still train `PTS`, `REB`, `AST`, `STL`, `BLK`, and `TOV`.

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
- Both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) must use zero-padding for players with fewer than `seq_len` context games. Do not revert to skipping short players — the zero-padding behavior is now covered by regression tests.
- When changing `SIZE_TIER_SPECS` in `src/config/model_config.py`, remember that `seq_len` and `max_seq_length` affect both the Transformer model's sequence window and the training pipeline's batch construction. Changes here must be reflected in both `_create_sequences()` and `_build_sequence_batch()`.

### Simulation

- `src/simulation/game_simulator.py` is a critical orchestrator. Keep interface changes small and documented.
- New simulation logic should use typed dataclasses from `sim_types.py` rather than raw dicts.
- If a simulation path becomes obsolete, remove it or mark it clearly. Do not leave large unreachable blocks without explanation.
- New simulation components (phase simulation, archetype inference, role sampling) should be independent modules in `src/simulation/` rather than embedded in GameSimulator.
- **Strict mode**: `GameSimulator(strict_mode=True)` raises `RuntimeError` on degraded optional InputHealth sources. When adding new optional context sources, integrate them into the `input_health` contract and check them in strict mode.
- **Data quality**: `ReportGenerator.export_player_projections()` must include the `DATA_QUALITY` column. New simulation result metadata that affects projection trustworthiness should be reflected here.

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
- Season-context groups are in the `full` preset but intentionally excluded from `small` to preserve its iteration speed.
- The rest cap in `RestGameDensityFeatureGroup` (`DAYS_SINCE_LAST_GAME.clip(upper=14.0)`) applies to ALL games, not just off-season gaps. Do not remove or raise the cap without validating the effect on B2B detection (a 1-day gap still correctly produces B2B=1).

### Phase-Aware Drift Detection (NEW)

- `DriftDetector` now maintains a single `_history` list with a `phase` tag on each entry (vs separate lists per phase). The `detect()` method filters the list by phase at query time.
- When passing `phase` to `detect()`, the function filters history to that phase only. If no phase-specific data exists, it falls back to the full history.
- The `_infer_phase_from_date()` heuristic (Apr 15–Jun 20 = PLAYOFF) is a rough approximation. Explicitly pass `phase='PLAYOFF'` during postseason backtesting for accuracy.
- The phase tag is persisted in `drift_state.json` so it survives restarts — old (pre-phase) entries will not have a `phase` key and are treated as `REGULAR`.

### Evaluation And Optimization (NEW)

- `src/evaluation/` is the evaluation and optimization subsystem. All backtesting, weight optimization, drift detection, and weight versioning lives here.
- Backtest results flow through `BacktestResult` and `TargetMetrics` dataclasses — do not return raw dicts or ad hoc metrics.
- `BacktestRunner` depends on `ModelManager` for predictions and `DataLoader` for actual box scores. Both must be available for backtesting to work.
- `EnsembleOptimizer` uses scipy.optimize with 13 parameters. If the parameter space changes, update `_TUNABLE_DIMS` and all code that builds/flattens weight vectors.
- `WeightStore` is the single source of truth for active ensemble weights. Never bypass it with direct `blend_weights.pkl` writes.
- Weight versions are numbered and stored under `data/weights/`. The `current.json` pointer indicates the active version.
- `optimize_weights.py` always writes through `WeightStore`. Use `--rollback N` to revert, not manual file edits.
- `DriftDetector` uses SPC (2σ threshold). The baseline window and detection threshold are configurable through `config/default.yaml` under `self_optimization:`.
- Keep the blend-weight / Transformer artifact contract enforced at load time. `ModelManager.set_weights()` must validate the new weights before applying them.

### Query

- The query layer should consume exported projection artifacts rather than reimplement full model inference.
- If the exported schema changes, update `src/query/projection_loader.py` and query tests in the same change.

## Refactor Rules

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
