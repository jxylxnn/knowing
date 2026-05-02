# Tasks

## NOW

### Apply planned bug-fix batch (groupby KeyError, CatBoost GPU callback, device AttributeError)

- Status: DONE on 2026-04-25.
- `pace_role.py` already uses string column names in groupby; no code change required.
- `catboost_trainer.py` already conditionally disables callbacks for GPU and rebuilds fit_kwargs on fallback; no code change required.
- `nn_trainer.py` received defensive `isinstance(self.device, str)` coercion in `_create_loader()`.
- All targeted tests pass (preprocessing 19/19, nn_trainer 2/2, runtime artifact 5/5, transformer 8/8).

### Validate zero-padding behavior on real training data

- Why it matters: the zero-padding change increases the number of training samples for players with short careers. A live training run should confirm that the Transformer still converges and that the padded samples do not degrade validation MAE.
- Likely files:
  - `src/models/transformer_model.py`
  - `src/training/pipeline.py`
  - `train.py`
- Done when:
  - `python train.py` completes with the M tier (seq_len=20) and the Transformer validation MAE is comparable to or better than before
  - the training log shows more sequences generated than before (due to short-player inclusion)

### Wire new feature groups into FeatureEngineer and training presets — DONE

- Completed 2026-04-12. All seven new feature groups are now wired into `FeatureEngineer._build_groups()`, the `full` training preset in both `config/default.yaml` and `src/training/presets.py`, and `FeatureSelector.SAFE_PREFIXES` in `src/utils/prediction_utils.py`. The `small` preset is unchanged. Preprocessing tests pass (19/19). `FeatureEngineer()` instantiates with 19 groups total.

### Add unit tests for the seven new feature groups — DONE

- Completed on 2026-04-21.
- Delivered:
  - `tests/test_preprocessing/test_new_feature_groups.py` with 38 tests covering all 7 new feature groups plus shared utilities and performance smoke tests:
    - `RestGameDensityFeatureGroup` — 5 tests (output columns, first-game priors, B2B flag, no leakage, missing-opponent fallback)
    - `LineupStabilityFeatureGroup` — 4 tests (output columns, first-game priors, team-change stability, no leakage)
    - `InjuryAdjustedOpportunityFeatureGroup` — 4 tests (output columns, first-game priors, missing high-usage teammate, no leakage)
    - `TeammateUsageFeatureGroup` — 4 tests (output columns, first-game priors, empty roster fallback, no leakage)
    - `RecencyFormFeatureGroup` — 4 tests (output columns, first-game priors, cold-start player, no leakage)
    - `MinutesConfidenceFeatureGroup` — 4 tests (output columns, first-game priors, cold-start flag, no leakage)
    - `DefensePositionFeatureGroup` — 5 tests (output columns, first-game priors, missing-opponent fallback, no-archetype inference, no leakage)
    - 2 integration tests for `FeatureDiagnostics` tracking and `FeatureContext` prior usage
    - `TestTeammateUtils` — 4 tests for shared `_teammate_utils.py` (game roster map, regular teammates map, high-usage teammates map, empty-roster handling)
    - `TestPerformanceSmoke` — 2 performance smoke tests (`rest_density` < 2s, `lineup_stability` < 3s on 500-row synthetic DataFrames)
  - All tests use small synthetic DataFrames and run fast/offline.
  - Full suite result: `178 passed, 0 failed`.

### Export complete projection stats for query-time use — DONE

- Completed 2026-04-12. All six target stats (PTS, REB, AST, STL, BLK, TOV) are now exported from `report_generator.py::export_player_projections()` with proper column names (`PROJ_STL_MEAN/MODE`, `STL_CI_LOW/HIGH`, `STL_99_CI_LOW/HIGH`, etc.). `projection_loader.py::STAT_COLUMNS` now maps all six stats consistently, and `_row_to_projection()` uses this mapping uniformly instead of fallback `get_first_float()` logic. `interactive_cli.py::HELP_TEXT` now includes `tov/turnovers` in the stats list.

### Remove or clearly quarantine dead code in `GameSimulator`

- Why it matters: `src/simulation/game_simulator.py` contains a large unreachable block after an early return, creating uncertainty about the true active algorithm.
- Likely files:
  - `src/simulation/game_simulator.py`
  - `project-brain/ARCHITECTURE.md`
  - `project-brain/FILE_MAP.md`
- Done when:
  - there is one clearly documented active simulation path
  - dead legacy code is removed or intentionally isolated with comments and tests

### Verify Colab launcher repo-root detection against a live mounted Drive run

- Why it matters: the notebook now resolves `train.py` from the cloned repo checkout instead of inferring code location from Drive-backed model storage, but the full Colab path still needs a real smoke test.
- Likely files:
  - `train_colab.ipynb`
  - `project-brain/CURRENT_STATE.md`
  - `project-brain/KNOWN_BUGS.md`
- Done when:
  - a Colab checkout under `/content/knowing` launches `train.py` successfully while `data/` and `models/` stay on Drive
  - the notebook prints the resolved repo root, script path, and storage directories before launch
  - a missing `train.py` produces the new path-diagnostic error

## NEXT

### Benchmark the archetype feature group on live CSVs

- Why it matters: the deterministic player-style features are now wired into the default stack, but the actual cold-start lift still needs a live measurement on current data.
- Likely files:
  - `src/preprocessing/features/archetype.py`
  - `src/preprocessing/feature_engineer.py`
  - `src/training/presets.py`
  - `train.py`
  - `data/nba_players.csv`
  - `data/nba_games.csv`
- Done when:
  - `python train.py` or `python train.py --preset small` completes with the archetype group active
  - baseline and archetype-enabled runs are compared on the current dataset
  - any unexpected schema or loading mismatch is documented

### Benchmark the new `small` training preset on live CSVs

- Why it matters: the preset is implemented and test-covered, but the real speedup versus the full stack still needs a live measurement on the current data files.
- Likely files:
  - `train.py`
  - `src/training/presets.py`
  - `src/training/pipeline.py`
  - `data/nba_players.csv`
  - `data/nba_games.csv`
- Done when:
  - `python train.py --preset small` completes on the current dataset
  - runtime and artifact output are compared against the full preset
  - any unexpected artifact or load mismatch is documented

### Add a strict simulation mode for optional scraper degradation

- Why it matters: the repo now defaults to visible degraded mode for optional context failures, but advanced operators may want fail-fast behavior.
- Likely files:
  - `simulate_season.py`
  - `src/simulation/season_simulator.py`
  - `src/simulation/game_simulator.py`
- Done when:
  - the CLI exposes a strict option
  - optional degraded inputs can trigger non-zero exit status when requested
  - tests cover default vs strict behavior

### Make `--season` simulate the actual remaining season

- Why it matters: the current implementation appears to fetch only roughly the next 30 days, which mismatches the CLI description.
- Likely files:
  - `src/data/schedule_scraper.py`
  - `simulate_season.py`
  - `src/simulation/season_simulator.py`
- Done when:
  - the CLI description matches behavior
  - season mode spans the real remaining schedule or is explicitly renamed/scoped

### Add end-to-end regression coverage for training and simulation contracts

- Why it matters: the highest-impact remaining defects are in seams between modules rather than within isolated units.
- Likely files:
  - `tests/test_training/`
  - `tests/test_models/`
  - `tests/test_simulation/`
- Coverage targets:
  - schedule scraping config resolution
  - projection export schema
  - simulation startup from a freshly trained artifact set

### Run a live CLI smoke test for the repaired training/runtime artifact contract

- Why it matters: focused regression tests now cover the CatBoost save/load seam, but one real `train.py` -> `ModelManager.load_models()` -> `simulate_season.py` verification is still needed in a healthy environment with actual CSV inputs.
- Likely files:
  - `train.py`
  - `simulate_season.py`
  - `src/models/model_manager.py`
  - local `data/` and `models/` directories
- Done when:
  - `python train.py` produces the expected runtime artifacts from real data
  - a fresh process can run `ModelManager.load_models()`
  - `simulate_season.py` gets through startup model loading with that artifact set
- Progress on 2026-04-11:
  - venv rebuilt with Python 3.12; all 110 tests pass
  - data fetched via `update_data.py`
  - preflight failure modes (missing CSV, unwritable models dir) live-verified
  - full training run deferred due to runtime length; unit tests confirm artifact contract

### Audit remaining legacy scraper modules for inactive drift — DONE

- Completed on 2026-04-21.
- Delivered:
  - `src/data/rotowire_lineup_scraper.py`: removed unreachable code after `return` in `_get_config_value`, moved `self._cache_timestamp` initialization into `__init__`, fixed uppercase attribute references (`MAX_RETRIES`, `RETRY_DELAY`, `CACHE_TTL_MINUTES`) to lowercase instance attributes, removed undefined `ROTONAME_TO_TEAM` usage and replaced with `normalize_team()`.
  - `src/data/nba_defense_scraper.py`: added missing import of `ABBR_TO_ID` and `ID_TO_ABBR` from `src.utils.team_mappings`, replaced all undefined `TEAM_ID_MAP` and `ID_TO_TEAM` references, fixed `DefensiveMatchupAnalyzer` to initialize `max_retries`, `retry_delay`, and `headers` and to use `defense_scraper._session` for HTTP calls instead of undefined `self._session`/`self.HEADERS`.
  - `src/data/schedule_scraper.py`: replaced the 30-day stub in `get_remaining_season()` with a computed season-end horizon (capped at 180 days), added TODO/fallback logging, and updated docstring to document the upstream limitation.
  - `tests/test_data/test_scraper_health.py`: added 4 new regression tests covering rotowire constructor/attrs, unreachable-code safety, defense-scraper import/analyzer attrs, and schedule-scraper horizon cap.
  - Full test suite: `140 passed, 0 failed`.

## LATER

### Decide whether `src/training/feature_cache.py` should become part of the active training pipeline

- Why it matters: cache infrastructure exists, but the top-level training path does not clearly rely on it.
- Outcome options:
  - wire it into `train.py`
  - remove it
  - document it as experimental/inactive

### Improve experiment tracking usefulness

- Why it matters: `ExperimentTracker` writes JSON, but the repo does not surface a strong workflow for comparing or promoting runs.
- Likely files:
  - `src/training/experiment.py`
  - `train.py`
  - `README.md`

### Add clearer calibration/backtesting reporting

- Why it matters: probability outputs are more trustworthy if calibration quality is surfaced alongside raw projections.
- Likely files:
  - `src/query/probability_calculator.py`
  - `src/simulation/report_generator.py`
  - evaluation-oriented modules that may need to be created

### Clean up empty or placeholder module areas

- Why it matters: `src/services/` and `src/evaluation/` create ambiguity about planned architecture versus active code.
- Done when:
  - these areas are either populated with real ownership or removed/documented as intentionally reserved

## BLOCKED

### Live Colab training smoke test with mounted Drive data

- Why it is blocked:
  - this workspace does not have the `/content/drive/MyDrive/nba_model/data/*.csv` inputs mounted locally
  - the notebook fix can be validated only in a Colab-like environment with the real Drive paths and an actual repo checkout
- What is needed:
  - run the hardened `train_colab.ipynb` cell against the real Drive data and models directories
  - confirm that stdout/stderr from `train.py` are printed when the subprocess fails
  - confirm the notebook exits immediately on nonzero status

### Full end-to-end validation with live data

- Why it is blocked:
  - a full `python train.py` run takes too long for a quick smoke test; only preflight checks and unit tests were run
- What is needed:
  - a full `train.py` run with real CSV data (data is now available via `update_data.py`)
  - a `simulate_season.py --today` or `--date` run with observed outputs
- Progress on 2026-04-11:
  - venv rebuilt with Python 3.12; all 110 tests pass
  - data fetched successfully; preflight checks verified
  - remaining blocker is just runtime for the full training loop

### Confidence in third-party scraper reliability over time

- Why it is blocked:
  - scraper health depends on external sites and can change without repo changes
- What is needed:
  - live integration checks or scheduled smoke tests

## DONE

### Stabilize the Step 2 feature-ablation compatibility path

- Completed on 2026-04-11.
- Delivered:
  - `train.py` now creates the ablation probe through `build_feature_engineer(...)` instead of a raw `FeatureEngineer()` call.
  - `src/preprocessing/feature_engineer.py` now uses `build_feature_engineer(...)` inside `benchmark_feature_variants()` so the ablation benchmark itself can tolerate older constructors that lack `disable_groups`.
  - `tests/test_preprocessing/test_feature_engineer.py` now simulates a legacy constructor without `disable_groups` and verifies the ablation benchmark still completes.
  - `tests/test_training/test_train_entrypoint.py` now guards both the helper call and the absence of a bare Step 2 `FeatureEngineer()` instantiation.
- Note:
  - The compatibility-safe Step 2 path now covers both the final feature-engineering setup and the ablation benchmark probe.

### Add deterministic player archetype similarity features

- Completed on 2026-04-11.
- Delivered:
  - `src/preprocessing/features/archetype.py` adds a dedicated `PlayerArchetypeFeatureGroup` that builds past-only style profiles and compares them against fixed playstyle templates.
  - The group emits `ARCHETYPE_ID`, `ARCHETYPE_CONFIDENCE`, `ARCHETYPE_SIMILARITY_PRIMARY`, `ARCHETYPE_SIMILARITY_SECONDARY`, and `SIMILARITY_TO_*` columns for playmaker/shot-creator/big/wing/rim-runner/bench-scorer style buckets.
  - `FeatureEngineer` now runs the archetype group as part of the default preprocessing flow.
  - `src/training/presets.py` and `config/default.yaml` now include `archetype` in both the `full` and `small` preset group lists.
  - `FeatureSelector` now treats `ARCHETYPE_` and `SIMILARITY_TO_` columns as safe engineered features.
  - Regression tests now cover archetype column creation, schema compatibility, and cold-start style assignment.
- Note:
  - The live performance gain against the current CSVs is still pending and has been promoted to `NEXT`.

### Add a small CatBoost-first training preset

- Completed on 2026-04-11.
- Delivered:
  - `src/training/presets.py` defines `small` and `full` presets plus the recent-history trim helper.
  - `config/default.yaml` now carries YAML preset definitions for CLI/config resolution.
  - `src/config/config.py` now loads `training_presets` as raw config data.
  - `train.py` now resolves presets, passes the preset feature-group allowlist into `build_feature_engineer(...)`, disables the Transformer for the small preset, and trims to the most recent two seasons when `SEASON_ID` is present.
  - `src/training/pipeline.py` now records preset/feature-group metadata in `model_stack_metadata.pkl` when available.
  - `src/models/model_manager.py` now treats `model_stack_metadata.pkl` as part of the shared runtime artifact set.
  - Regression tests now cover preset resolution, recent-history trimming, feature-group allowlists, and CatBoost-only artifact loading.
- Note:
  - The live real-data timing comparison is still pending and has been promoted to `NEXT`.

### Stabilize Transformer validation inference on CUDA

- Completed on 2026-04-02 in code and unit tests.
- Delivered:
  - `src/models/transformer_model.py` now keeps an eager base model for validation/runtime inference, only compiles when explicitly allowed, and prefers a safe math SDPA backend during eager inference on CUDA.
  - `src/training/pipeline.py` now delegates Transformer validation batch prediction through the wrapper instead of calling the compiled model directly.
  - `src/config/model_config.py` now generates Transformer configs with `use_compile` disabled by default and marks compile as opt-in.
  - `train_colab.ipynb` now launches training/simulation through `subprocess.run(..., check=True)` so failed runs stop the notebook instead of printing fake success text.
  - Regression tests now cover the eager validation path and the training pipeline delegation seam.

### Harden scraper reliability and surface degraded simulation inputs

- Completed on 2026-04-02.
- Delivered:
  - `src/data/schedule_scraper.py` now uses initialized lowercase config state consistently and records required schedule health.
  - `src/data/lineup_scraper.py` now initializes cache/state in `__init__`, replaces undefined team/header references, and returns explicit failed/fallback lineup states.
  - `src/data/basketball_ref_scraper.py` now uses consistent lowercase config-backed access plus real Basketball Reference abbreviation mappings.
  - `src/data/injury_scraper.py` and `src/data/betting_scraper.py` now expose last-fetch health for simulation wrappers.
  - `src/simulation/input_health.py`, `src/simulation/game_simulator.py`, `src/simulation/season_simulator.py`, `src/simulation/report_generator.py`, and `simulate_season.py` now propagate per-source input health into results and CLI summaries.
  - Focused regression tests now cover scraper regressions, degraded-mode reporting, and hard-required schedule failures.

### Repair the CatBoost runtime artifact contract between training and simulation

- Completed on 2026-04-01.
- Delivered:
  - `src/training/pipeline.py` now persists per-target CatBoost artifacts and validates the runtime artifact set before success.
  - `src/training/catboost_trainer.py` now exposes shared artifact validation helpers and raises clearly on missing runtime files.
  - `src/models/model_manager.py` now validates the full runtime artifact set before loading.
  - `simulate_season.py` now relies on `ModelManager.load_models()` for startup validation.
  - `tests/test_training/test_runtime_artifact_contract.py` now covers artifact creation, fresh-process loading, and fail-loud behavior.

### Establish a project memory system in `/project-brain`

- Completed on 2026-04-01.
- Delivered:
  - `PROJECT_CONTEXT.md`
  - `ARCHITECTURE.md`
  - `CURRENT_STATE.md`
  - `TASKS.md`
  - `DECISIONS.md`
  - `CODE_RULES.md`
  - `FILE_MAP.md`
  - `KNOWN_BUGS.md`

### Verify current automated test baseline

- Completed on 2026-04-01.
- Result:
  - `pytest tests/ -q`
  - `83 passed, 2 skipped`
- Important follow-up:
  - AGENTS guidance about a known failing test is stale and should not be trusted as current repo state.

### Reduce pandas fragmentation in rolling feature generation

- Completed on 2026-04-02.
- Delivered:
  - `src/preprocessing/features/rolling.py` now batches new feature columns into temporary Series/DataFrames and concatenates them once per feature group.
  - `tests/test_preprocessing/test_feature_engineer.py` now asserts the feature-engineering path does not emit pandas `PerformanceWarning` fragmentation warnings.
  - Full test suite runtime after the refactor: `94 passed, 2 skipped`.
- Follow-up:
  - a real-data large-scale profile can still be collected later if the team wants a stricter production baseline.

### Harden Colab training launch and train.py preflight logging

- Completed on 2026-04-02.
- Delivered:
  - `train.py` now preflights writable model/cache directories and required raw CSV inputs before expensive work starts.
  - `train.py` now logs explicit stage names so failures point to the loading, feature engineering, pipeline init, split, or training stage.

### Decouple Colab repo root from Drive-backed training storage

- Completed on 2026-04-02.
- Delivered:
  - `train_colab.ipynb` now resolves `train.py` from the cloned repo checkout instead of assuming the Drive models directory is the code root.
  - The notebook now supports an explicit `project_root_override` and falls back to common repo checkout candidates like `/content/knowing`.
  - The launcher now prints the resolved `project_root`, `train_script`, `data_dir`, and `models_dir` before starting training.
  - `train_colab.ipynb` now captures stdout/stderr from `train.py`, prints the return code, and raises immediately on nonzero exit.

### Restore and guard the `FeatureSchema` import contract

- Completed on 2026-04-02.
- Delivered:
  - `src/utils/prediction_utils.py` now explicitly declares `FeatureSchema` in its public exports.
  - `src/utils/__init__.py` re-exports `FeatureSchema` so package-level imports stay compatible.
  - `tests/test_training/test_runtime_artifact_contract.py` now verifies the import contract in a clean subprocess with a stubbed `torch` module.
  - Targeted regression run: `4 passed`.

### Make `train.py` tolerate older `FeatureEngineer` constructors

- Completed on 2026-04-03.
- Delivered:
  - `src/preprocessing/feature_engineer.py` now exposes `build_feature_engineer(...)`, a compatibility-safe constructor that backfills `disable_groups` and `disable_columns` when an older checkout does not accept them.
  - `train.py` now routes Step 2 feature-engineering setup through that helper so the CLI keeps working across mixed-version checkouts.
  - `tests/test_preprocessing/test_feature_engineer.py` now covers both the current constructor contract and the legacy-ctor fallback path.
  - `tests/test_training/test_train_entrypoint.py` now guards the `train.py` entrypoint so Step 2 keeps using `build_feature_engineer(...)` instead of drifting back to a direct constructor call.
  - Latest targeted regression run: `19 passed`.

### Fix silently uncalibrated predictions when Transformer artifact is missing

- Completed on 2026-04-11.
- Delivered:
  - `src/models/model_manager.py` now includes `_validate_blend_contract()`, which raises `FileNotFoundError` when blend weights expect a Transformer but `attention_transformer.pkl` is missing, and `RuntimeError` when the file exists but failed to load.
  - `src/training/pipeline.py` now includes `_validate_blend_contract()` on the `load_models()` path and `_save_model_stack_metadata()` to persist `model_stack_metadata.pkl` during training.
  - `model_stack_metadata.pkl` is now part of the training output artifact set, recording whether the Transformer was active and the expected model count.
  - `tests/test_models/test_model_manager.py` now has 6 `TestBlendContractEnforcement` tests covering missing-Transformer, corrupt-Transformer, zero-weight, loaded-Transformer, empty-blend-weights, and no-partial-blend scenarios.
  - `tests/test_training/test_runtime_artifact_contract.py` now verifies that `ModelManager.load_models()` raises on tampered blend weights and that `model_stack_metadata.pkl` is persisted.
  - Targeted regression run: `21 passed`.

### Fix torch shim clobbering real PyTorch and legacy checkpoint load failure

- Completed on 2026-04-11.
- Delivered:
  - `src/__init__.py:_install_test_torch_shim()` now checks `importlib.util.find_spec('torch')` before installing the fake torch module, preventing clobbering of real PyTorch in healthy environments.
  - `src/models/transformer_model.py:TransformerWrapper.load()` now infers model architecture from state_dict tensor shapes when the checkpoint config is empty, via the new `_infer_config_from_state()` static method.
  - All 110 tests now pass (previously 4 `test_transformer_model.py` tests failed due to the torch shim).
  - Preflight failure modes for `train.py` live-verified: missing CSV and unwritable models dir both produce clear stage-specific errors.
  - venv rebuilt from Python 3.13 to Python 3.12 with all dependencies installing cleanly.
  - Data fetched via `update_data.py --season 2024-25 --season 2025-26`.
  - Full regression run: `110 passed, 0 failed`.

### Add regression tests for the 6-stat export/load contract

- Completed on 2026-04-12.
- Delivered:
  - `tests/test_query/test_six_stat_contract.py` with 15 regression tests covering:
    - Export contains all 6 stat columns (PTS, REB, AST, STL, BLK, TOV)
    - Export values match input values
    - Loader reads all 6 stats correctly
    - `get_stat_mean()` works for all 6 stats
    - `get_stat_ci()` works for all 6 stats
    - `STAT_COLUMNS` mapping contains all 6 stats with correct keys
    - Missing TOV columns loads with defaults (graceful degradation)
    - HELP_TEXT includes `tov/turnovers`
    - All 6 stats return real values, not 0.0 defaults
    - `STAT_DISPLAY_NAMES` contains all 6 stats
  - Full regression run: `132 passed, 0 failed`.
