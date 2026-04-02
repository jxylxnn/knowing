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

### Full end-to-end validation with live data

- Why it is blocked:
  - this workspace does not include checked-in raw data or trained models
  - the checked-in `venv/` is inconsistent: `venv/bin/python` points to Python 3.13 without the installed packages, while `venv/bin/python3.12` has the packages but crashes on `torch` import in this sandbox
- What is needed:
  - a controlled run of `update_data.py`
  - a fresh `train.py` run
  - a `simulate_season.py --today` or `--date` run with observed outputs

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
