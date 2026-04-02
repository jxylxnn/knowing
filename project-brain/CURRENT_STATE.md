# Current State

## Snapshot

- Observed date: 2026-04-02
- Repository health: mixed
- Test status in this workspace:
  - baseline audit run: `83 passed, 2 skipped`
  - post-fix targeted regression run: `12 passed`
  - scraper/input-health regression run: `29 passed`
- Important correction to repo instructions: the AGENTS note claiming `test_registry_initialization` is a known failure is stale in the current repo state; the full test suite passed during this audit.

## What Currently Works

- Core repo structure is coherent enough to understand and extend.
- Historical data ingestion flow exists in `update_data.py` with multiple season-selection modes.
- Feature engineering is substantial, modular, and well-covered compared with other areas.
- Training internals for CatBoost/Transformer components are implemented and testable in isolation.
- The active training path now persists per-target CatBoost runtime artifacts and validates the required `models/` contract before returning success.
- `ModelManager` now rejects incomplete runtime artifact sets instead of silently loading a partial model directory.
- `simulate_season.py` now uses `ModelManager.load_models()` for startup validation instead of hard-coding a single `pts_catboost.cbm` existence check.
- The query subsystem in `src/query/` is structurally complete and supported by tests.
- Many unit tests around preprocessing, model wrappers, and probability math are green.
- Cleanup tooling in `clear_cache.py` is clear and conservative about preserving raw data.

## What Partially Works

- Simulation stack is feature-rich and still depends on volatile third-party scrapers, but degraded optional inputs are now surfaced explicitly in per-game metadata and CLI output.
- Query flow supports six stats at the parser/calculator level, but projection exports only appear complete for `PTS`, `REB`, and `AST`.
- The train-to-simulation artifact contract is covered by focused regression tests, but a true CLI smoke run of `python train.py` could not be completed in this workspace because no raw training CSVs are present and the usable local interpreter crashes when importing `torch`.
- Schedule scraping and season simulation features are implemented, and the previously confirmed config/state regressions in schedule/lineup/betting-support scrapers are now fixed in code.

## What Is Broken Or Very Likely Broken

### Scraper reliability still depends on upstream availability

- The concrete config/state regressions previously identified in `ScheduleScraper`, `LineupScraper`, and `BasketballRefScraper` are fixed.
- Live upstream changes can still degrade optional context inputs, but the simulator now reports those conditions explicitly instead of silently masking them.

### Projection export/query mismatch

- `src/simulation/report_generator.py` exports `PTS`, `REB`, and `AST` projection columns.
- `src/query/projection_loader.py` and the CLI support `STL`, `BLK`, and `TOV`, but those values are not fully exported from simulation output.
- Cached queries for those stats therefore degrade to incomplete or default behavior.

## Current Limitations

- No checked-in raw data or trained models are present in this workspace beyond cache directories, so true end-to-end execution was not validated here.
- Network-reliant flows such as `update_data.py` and live scraping were not exercised in this audit environment.
- The checked-in `venv/` is internally inconsistent:
  - `venv/bin/python` points to Python 3.13 without the installed project packages.
  - `venv/bin/python3.12` has the packages, but importing `torch` fails in this sandbox with `OMP: Error #179: Function Can't open SHM2 failed`.
- The repo uses many local file contracts rather than strong typed interfaces between phases.
- There is visible architecture drift between some legacy/alternate modules and the active top-level scripts.
- `simulate_season.py` now carries a run-level input health summary and exits non-zero for hard schedule failures, but this still needs a live smoke test against current upstream sources.

## Active Risks

- The main remaining scraper risk is upstream drift, not silent masking: optional failures now degrade runs visibly, and schedule failures are treated as hard-required.
- Artifact naming or schema drift is now guarded by training/runtime validation, but future changes can still break the contract if tests are not kept in sync.
- Scraper defects are under-tested and concentrated in modules that directly affect user-visible simulation output.
- Feature engineering performance is degraded by pandas DataFrame fragmentation warnings.

## Known Workarounds

- If lineup, injury, betting, or defense context scrapers fail, `GameSimulator` now continues in explicit degraded mode and records which sources fell back or failed.
- Query users can still ask for points, rebounds, and assists from exported projection CSVs with higher confidence than for steals, blocks, or turnovers.
- `clear_cache.py` can reset generated state while preserving raw input CSVs.

## Immediate Priorities

1. Export `STL`, `BLK`, and `TOV` projection columns and align `ProjectionLoader` expectations with the report schema.
2. Remove or reconcile dead simulation code in `src/simulation/game_simulator.py` to reduce maintenance ambiguity.
3. Run one live `train.py` -> `ModelManager.load_models()` -> `simulate_season.py` smoke test in a healthy local environment with real CSV inputs.
4. Decide whether to add a strict fail-fast mode for optional scraper degradation on top of the new default warn-and-continue behavior.

## Testing Status

- Verified on 2026-04-01: `pytest tests/ -q` passed with `83 passed, 2 skipped`.
- Verified on 2026-04-01 after the artifact-contract fix:
  - `python3 -m pytest tests/test_training/ tests/test_models/test_model_manager.py -q`
  - Result: `12 passed`
- Verified on 2026-04-02 after the scraper/input-health hardening:
  - `venv/bin/python3.12 -m pytest tests/test_data/test_scraper_health.py tests/test_simulation/test_game_simulator.py tests/test_simulation/test_simulation_health_reporting.py -q`
  - Result: `29 passed`
- Two tests are skipped by design around Transformer runtime constraints.
- Test output produced 1612 warnings, dominated by pandas `PerformanceWarning` messages from `src/preprocessing/features/rolling.py`.

## Areas That Need Confirmation In A Future Session

- Whether additional scraper modules beyond the identified ones have similar uppercase/lowercase config drift.
- Whether any local uncommitted changes in this worktree are part of an in-progress fix for the observed regressions.
- Whether a healthy local runtime with real `data/nba_players.csv` and `data/nba_games.csv` can complete a full `python train.py` smoke run and then start `simulate_season.py` successfully.
