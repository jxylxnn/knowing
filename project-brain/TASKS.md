# Tasks

## NOW

### Export complete projection stats for query-time use

- Why it matters: the query CLI claims support for `stl`, `blk`, and `tov`, but exported projection CSVs currently appear incomplete for those stats.
- Likely files:
  - `src/simulation/report_generator.py`
  - `src/query/projection_loader.py`
  - `src/query/interactive_cli.py`
  - `tests/test_query/`
- Done when:
  - projection exports include all six target stats
  - loader code reads them consistently
  - cached query results for those stats are non-default and tested

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

### Audit remaining legacy scraper modules for inactive drift

- Why it matters: `rotowire_lineup_scraper.py` and similar alternate paths still look legacy relative to the active simulator stack.
- Likely files:
  - `src/data/rotowire_lineup_scraper.py`
  - other non-primary `src/data/` scrapers
- Done when:
  - active vs inactive scraper modules are documented clearly
  - duplicate broken implementations are either fixed or marked inactive

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
