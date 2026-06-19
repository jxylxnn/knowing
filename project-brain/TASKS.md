# Tasks

## NOW

### Run residual correction monitoring against real prediction history

- Why it matters: the monitoring system is implemented and test-covered (47 tests) but needs live validation against a real `data/evaluation/prediction_history.parquet` to confirm that HELPING/NEUTRAL/HURTING labels correspond to operator intuition, and that `latest_summary.json` can be consumed by downstream dashboards.
- Likely files:
  - `monitor_residual_corrections.py`
  - `src/evaluation/residual_monitor.py`
  - `src/evaluation/residual_report.py`
  - `data/evaluation/prediction_history.parquet`
  - `reports/residual_monitoring/`
- Done when:
  - `python monitor_residual_corrections.py --print-summary` completes against real data.
  - Per-target status labels are physically interpretable (PTS correction is HELPING, TOV correction might be NEUTRAL, etc.).
  - `latest_summary.json` is valid strict JSON (no NaN tokens) consumable by `jq` and JavaScript.
  - Rolling-window status is populated for at least one window size.
  - Data-quality and confidence breakdowns show rows in multiple buckets.

### Validate zero-padding behavior on real training data

- Why it matters: the zero-padding change increases the number of training samples for players with short careers. A live training run should confirm that the Transformer still converges and that the padded samples do not degrade validation MAE.
- Likely files:
  - `src/models/transformer_model.py`
  - `src/training/pipeline.py`
  - `train.py`
- Done when:
  - `python train.py` completes with the M tier (seq_len=20) and the Transformer validation MAE is comparable to or better than before
  - the training log shows more sequences generated than before (due to short-player inclusion)

### Validate lifecycle aging precomputation with real bio data

- Why it matters: B-Ianus and KAN aging curves are precomputed in `train.py` at startup, but need live validation with real `player_bios.csv` data.
- Likely files:
  - `train.py`
  - `src/lifecycle/aging_model.py`
  - `src/lifecycle/kan_age_model.py`
  - `data/player_bios.csv`
- Done when:
  - `python train.py` shows "Precomputed B-Ianus aging curves" and "Precomputed KAN aging outputs" in logs
  - lifecycle caches (`aging_curves.csv`, `kan_aging_outputs.csv`) are populated
  - aging feature groups produce non-neutral values during feature engineering

### Run full test suite after residual monitoring system

- Status: DONE on 2026-06-18.
- Added: `tests/test_evaluation/test_residual_monitor.py` — 47 tests covering: threshold logic, metric calculation, status labels, data-quality/confidence breakdowns, rolling windows, aggregation/recommendations, report writing, console rendering, validation/error handling, NaN bucket routing, strict JSON safety, CLI argument resolution, and partial corrected-prediction fallback.
- Added: `tests/test_query/test_projection_loader.py` — 14 tests covering: core/full CSV loading, corrected projections, interval fields, confidence labels, interval validation, non-numeric column rejection, and blank confidence handling.
- Baseline update: `313 + 47 + 14 = 374 passed, 1 skipped` (estimated; slow tests deselected).

### Run backtest against live data to establish baseline metrics

- Why it matters: the backtest/optimize/drift-detection subsystem is implemented and test-covered but needs real data to calibrate the drift detector baseline.
- Likely files:
  - `backtest.py`
  - `src/evaluation/backtest_runner.py`
  - `src/evaluation/drift_detector.py`
- Done when:
  - `python backtest.py --recent 14` completes against real data
  - per-stat MAE/RMSE/R² are recorded
  - drift detector has baseline statistics loaded

### Run optimize_variance.py against real data

- Why it matters: the CRPS-based variance optimization is implemented and import-tested but needs live validation against real historical game data to confirm it converges on useful volatility multipliers.
- Likely files:
  - `optimize_variance.py`
  - `data/nba_players.csv`
- Done when:
  - `python optimize_variance.py --recent 30 --dry-run` shows current multipliers and baseline CRPS
  - `python optimize_variance.py --recent 30` converges on multipliers that improve CRPS over the default 1.0 values
  - results are physically interpretable (e.g., playoff multiplier > 1.0, B2B multiplier > 1.0)

### Run smart feature selection end-to-end with real data

- Why it matters: the smart per-target feature selector is implemented and unit-tested but needs live validation on real data: confirm the per-target selected lists converge, that the manifest JSON is loaded by the pipeline, and that downstream MAE improves (or at minimum doesn't regress) versus the canonical `self.feature_cols` list.
- Likely files:
  - `train.py`
  - `src/evaluation/smart_feature_selector.py`
  - `src/evaluation/feature_group_ablation.py`
  - `src/evaluation/shadow_feature_filter.py`
  - `src/training/pipeline.py`
  - `data/nba_players.csv`
  - `models/feature_selection_manifest.json`
- Done when:
  - First, `python train.py --feature-selection smart --selection-profile fast` completes on real data (group ablation only — fastest path through the new handshake).
  - Then, `python train.py --feature-selection smart --selection-profile balanced` completes on real data.
  - `models/feature_selection_manifest.json` is populated and the per-target lists differ across stats.
  - `model_stack_metadata.pkl` records `feature_selection_enabled=True` with the active profile and selected lists.
  - Downstream MAE on the same date range is not worse than a baseline run with the full feature set.
  - The `max_accuracy` profile (with the time-stability check) runs in a reasonable time on the full historical dataset.

### Verify WeightStore bootstrap on a fresh process load

- Why it matters: `ModelManager.load_models()` now bootstraps `EnsembleWeights` from `WeightStore` (DR-030) after the legacy `blend_weights.pkl` is loaded, so the runtime uses data-driven weights even before `optimize_weights.py` is invoked. The bootstrap is non-fatal — a broken store falls back to the legacy blend. The bootstrap needs live validation to confirm it actually fires and that the resulting weights are loaded as the active set.
- Likely files:
  - `src/models/model_manager.py`
  - `src/evaluation/weight_store.py`
  - `models/blend_weights/`
- Done when:
  - **(a) Bootstrap line fires:** `python train.py` followed by a fresh `ModelManager.load_models()` invocation logs "Bootstrapped ensemble weights vN from WeightStore (score=...)" at INFO. Confirm in the log that `vN` matches the version that `TrainingPipeline._save_blend_weights()` wrote.
  - **(b) Self-sufficient bootstrap path:** a fresh process with no manual `optimize_weights.py` invocation finds `models/blend_weights/current.json` after `train.py` completes. The `current.json` is the versioned training-time blend, and `ModelManager` picks it up at load.
  - **(c) Fallback path:** delete `models/blend_weights/` (or rename `current.json`), reload, and confirm the legacy `blend_weights.pkl` blend is used and a `WeightStore bootstrap skipped` debug log line is emitted.
  - **(d) Hot-reload still works:** `ModelManager.set_weights(new_weights)` continues to override the active blend even after the bootstrap fired (i.e. bootstrap is a one-shot on `load_models`, not a permanent lock).

### Verify contracts layer in a live train→simulate cycle

- Why it matters: `src/contracts/` is the seam for inter-step artifact validation, and `check_contracts.py` is the top-level CLI for debugging contract failures in isolation. DR-031 wired the validators into the production read/write paths (`ScheduleScraper`, `SeasonSimulator`, `ProjectionLoader`, `ReportGenerator`, `ModelManager.predict_player_stats`, and a post-train `train.py` check), so the contract is enforced at every producer/consumer boundary by construction. Live validation is needed to confirm the boundary code actually catches the realistic failure modes before treating the contracts layer as the canonical contract boundary.
- Likely files:
  - `src/contracts/artifacts.py`
  - `src/contracts/projections.py`
  - `src/contracts/schedule.py`
  - `src/contracts/features.py`
  - `check_contracts.py`
  - `src/data/schedule_scraper.py`
  - `src/simulation/season_simulator.py`
  - `src/query/projection_loader.py`
  - `src/simulation/report_generator.py`
  - `src/models/model_manager.py`
  - `tests/test_contracts/test_pipeline_contract_smoke.py`
- Done when:
  - **Happy path:** `python check_contracts.py --models-dir models` exits 0 against a healthy artifact set. A fully run `python train.py` followed by `python simulate_season.py --today` passes the contract validator at both boundaries and `train.py`'s post-train check logs the `validate_runtime_artifacts` call.
  - **Missing CatBoost backbone:** deleting one of the required `*_catboost.cbm` files makes `check_contracts.py` raise `ArtifactContractError` and exit nonzero, and `ModelManager.load_models()` raises the same typed error on the next load.
  - **Mismatched metadata:** `model_stack_metadata.pkl` with the wrong `targets` set is rejected by `_validate_metadata(...)` inside `validate_runtime_artifacts(...)`.
  - **Stale projection CSV:** a pre-2026-06-04 `player_projections_*.csv` is rejected by `ProjectionLoader.load_projections(...)` with a typed `ProjectionSchemaContractError`. The operator workflow is to regenerate from the current `simulate_season.py` (KB-021).
  - **Schedule frame nulls:** a schedule DataFrame with null `GAME_ID` is rejected by `normalize_schedule_frame(...)` (typed `ScheduleContractError`), and the `ScheduleScraper` and `SeasonSimulator` call sites both surface the failure.
  - **Inference frame drift:** `ModelManager.predict_player_stats` with an inference frame that is missing one of the saved `feature_cols.pkl` columns raises `FeatureSchemaContractError` rather than silently using a fallback.

### Run residual interval calibration on real residual data

- Why it matters: Ticket 4 is implemented and unit-tested, but `models/calibration/` must be generated from the real walk-forward residual dataset before live predictions can show meaningful calibrated ranges.
- Likely files:
  - `calibrate_residual_intervals.py`
  - `data/evaluation/residual_training.parquet`
  - `models/calibration/`
  - `src/correction/calibration.py`
  - `src/models/model_manager.py`
  - `src/simulation/report_generator.py`
- Done when:
  - `python calibrate_residual_intervals.py --input data/evaluation/residual_training.parquet --output-dir models/calibration` completes.
  - `models/calibration/calibration_metadata.json` records nonzero rows for all six stats.
  - All six `{stat}_intervals.json` artifacts exist and contain at least a `GLOBAL` bucket with `q80`, `q90`, and `q95` widths.
  - A small `simulate_season.py --date YYYY-MM-DD` or targeted `ModelManager.predict_player_stats(..., include_confidence=True)` smoke test returns populated interval bounds and non-`NO_EDGE` confidence labels for calibrated stats.


### Run optimize_weights.py end-to-end with real data

- Why it matters: the self-optimization loop needs live validation. All components pass unit tests but the full optimize→verify→deploy cycle hasn't been exercised against real completed games.
- Likely files:
  - `optimize_weights.py`
  - `src/evaluation/ensemble_optimizer.py`
  - `src/evaluation/weight_store.py`
- Done when:
  - `python optimize_weights.py --recent 14 --dry-run` shows candidate weights
  - `python optimize_weights.py --recent 14` writes improved (or verified-equivalent) weights
  - versioned weights appear in `models/blend_weights/`

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
- Status: DONE on 2026-05-09 as part of simulation refactor. Dead legacy code removed; active path is exclusively `_simulate_matchup_reactive`. GameSimulator now delegates to PhaseSimulator.

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

### Decide whether `src/training/feature_cache.py` should become part of the active training pipeline — DONE

- Why it matters: cache infrastructure exists, but the top-level training path did not clearly rely on it.
- Outcome: **decided 2026-06-19 (DR-034)** — keep `feature_cache.py` as unused infrastructure; the active feature cache is the in-place parquet cache inside `FeatureEngineer.create_features()`, now enabled via `cache_dir="cache/training"` in `DataPipeline` and `ModelManager`. The cache key was hardened to fold in the mtime/size of external files the feature groups read (`FeatureGroup.external_files()`), so it never returns stale features. Coverage: `tests/test_preprocessing/test_feature_engineer.py::TestFeatureEngineerCache` (3 tests, all green).

### Improve experiment tracking usefulness

- Why it matters: `ExperimentTracker` writes JSON, but the repo does not surface a strong workflow for comparing or promoting runs.
- Likely files:
  - `src/training/experiment.py`
  - `train.py`
  - `README.md`

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

### Fix critical/high bug batch KB-022 through KB-032 — DONE

- Completed 2026-06-12.
- Delivered:
  - `src/training/pipeline.py`: fixed `self.targets` -> `self.TARGETS` so blend weights are persisted.
  - `src/models/model_manager.py`: Transformer runtime inference now uses `nn_features = feature_cols - cat_features`.
  - `src/__init__.py`: torch shim now imports `importlib.util` directly.
  - `src/models/minutes_predictor.py`: `MINS_LAST_3` rolling sum aligned with `reset_index(level=0, drop=True)`.
  - `src/data/nba_defense_scraper.py`: `get_team_defense_allowed()` normalizes `pts_allowed_per_100` to `opp_pts_per_100`.
  - `src/data/injury_scraper.py`: disk-cache loads now derive `TEAM_ABBR` and set `_cache_timestamp`.
  - `src/data/lineup_scraper.py`: guards against missing `TEAM_ABBR` by deriving it from `TEAM`.
  - `update_data.py`: `enrich_with_player_bios()` recomputes `AGE` from `BIRTHDATE` and `GAME_DATE` to prevent future-age leakage.
  - `src/data/basketball_ref_scraper.py`: season URLs now use the ending year via `_season_end_year()`.
  - `src/preprocessing/features/injury_risk.py`: career injury count is filtered to `DATE < GAME_DATE`.
  - `src/preprocessing/features/skill_development.py`: replaced full-season current-season averages with expanding season-to-date averages (shifted by 1) to remove future in-season leakage.
- Regression coverage updated:
  - `tests/test_models/test_model_manager.py` still passes (21/21).
  - `tests/test_preprocessing/test_skill_development_features.py` still passes (13/13).
  - `tests/test_preprocessing/test_injury_risk_features.py` still passes (10/10).
  - `tests/test_data/test_player_bio_scraper.py` still passes (16/16).
  - `tests/test_training/test_runtime_artifact_contract.py` fixed to clear versioned WeightStore after editing legacy `blend_weights.pkl`.
  - `tests/test_simulation/test_simulation_health_reporting.py` fixed to include required `GAME_ID` in schedule fixture.
- Full non-slow suite result: `368 passed, 1 skipped, 1 deselected`.

### Add calibrated residual confidence intervals — DONE

- Completed 2026-06-12.
- Delivered:
  - `calibrate_residual_intervals.py` CLI for building `models/calibration/` from `data/evaluation/residual_training.parquet`.
  - `src/correction/calibration.py` with `ResidualIntervalCalibrator`, corrected-error precedence, global/context bucket generation, clipped interval helper, and metadata output.
  - `src/correction/interval_store.py` with non-fatal runtime artifact loading and `GLOBAL` bucket fallback.
  - `src/correction/confidence_scorer.py` with `HIGH` / `MEDIUM` / `LOW` / `NO_EDGE` labels from interval width, data quality, minutes confidence, and residual-model status.
  - `ModelManager.predict_player_stats(..., include_confidence=True)` appends 80%/90% interval bounds and confidence labels when calibration artifacts are loaded.
  - `GameSimulator` requests confidence-aware predictions and carries interval/confidence metadata into `player_averages`.
  - `ReportGenerator.export_player_projections()` writes the new interval/confidence columns for all six stats and keeps `NO_EDGE` / blank numeric bounds when calibration is absent.
  - `src/contracts/projections.py` now validates the 6-stat x 14-column projection schema and confidence-label enum.
  - Tests: `tests/test_correction/test_calibration.py`, plus updated projection/contract fixtures.
- Verified:
  - `pytest tests/test_correction/ tests/test_query/ tests/test_contracts/ -q` -> `84 passed`.

### Fix six training-stopping bugs in CatBoost + Transformer pipeline

- Status: DONE on 2026-05-09.
- Delivered:
  - CatBoost import guard prevents crash when catboost isn't installed
  - feature_cols null check prevents downstream crashes on empty feature sets
  - bare raise fixed to proper exception handling
  - macOS persistent_workers guard prevents DataLoader crash
  - graceful feature missingness degradation instead of hard failure
  - parallel thread contention fix for CPU training
  - transformer save guard prevents checkpoint corruption
- Full suite: `178 passed, 0 failed`.

### Architecture cleanup — deduplicate code, align config, remove dead code

- Status: DONE on 2026-05-09.
- Delivered: deduplicated code paths, aligned config references, removed dead code throughout.
- Full suite: `178 passed, 0 failed`.

### Simulation refactor — phase simulator, archetype, role sampler, prob formatter

- Status: DONE on 2026-05-09.
- Delivered:
  - `src/simulation/phase_simulator.py` — extracted Monte Carlo loop from GameSimulator
  - `src/simulation/archetype.py` — ArchetypeEngine for player archetype inference
  - `src/simulation/role_sampler.py` — role state sampling with archetype-aware adjustments
  - `src/simulation/sim_types.py` — typed dataclasses replacing raw dicts
  - `src/simulation/sim_cache.py` — JSON disk-cache mixin
  - `src/simulation/stat_utils.py` — shared statistical helpers
  - `src/query/prob_formatter.py` — ProbFormatterMixin for formatted output
  - Dead legacy simulation code removed from GameSimulator
- Full suite: `178 passed, 0 failed`.

### Build self-optimizing ensemble weight system + Phase 0 bug fixes

- Status: DONE on 2026-05-09.
- Delivered:
  - `backtest.py` — CLI for evaluating prediction accuracy on historical games
  - `optimize_weights.py` — CLI for retuning ensemble weights via scipy.optimize
  - `src/evaluation/` module (5 files): metrics, backtest_runner, ensemble_optimizer, weight_store, drift_detector
  - `ModelManager` refactored for hot-reloadable `EnsembleWeights`
  - `config/default.yaml` now supports `self_optimization:` section
  - Versioned JSON weight store with atomic writes and rollback
- Full suite: `178 passed, 0 failed`.

### Add strict simulation mode + data quality schema (DR-025)

- Status: DONE on 2026-05-22.
- Delivered:
  - `simulate_season.py --strict` flag for fail-fast on degraded optional context
  - `GameSimulator(strict_mode=True)` and `SeasonSimulator(strict_mode=True)` — raises `RuntimeError` on degraded optional sources
  - `ReportGenerator.export_player_projections()` appends `DATA_QUALITY` column (`FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`)
  - `ProjectionLoader.find_player()` surfaces visible CLI warning for degraded projections
  - `_data_quality_from_result()` static helper derives quality from input_health metadata
- Full suite: `178 passed, 0 failed`.

### Lifecycle ML integration (aging curves, injury risk, skill development, GPU fe, nexus)

- Status: DONE on 2026-05-22.
- Delivered:
  - `src/lifecycle/aging_model.py` — B-Ianus Bayesian aging curves with position-specific priors
  - `src/lifecycle/kan_age_model.py` — KAN nonlinear age curves (CPU-only)
  - `src/models/nexus_model.py` — Nexus multi-modal architecture (SSM + FT-Transformer + GAT + Copula head)
  - `src/training/nexus_loss.py` — GaussianNLLLoss with Cholesky covariance
  - `src/preprocessing/feature_engineer_gpu.py` — GPU-accelerated cuDF feature engineering with CPU fallback
  - 4 new feature groups: InjuryRisk, AgingCurve, KANAging, SkillDevelopment (23 total)
  - `src/data/player_bio_scraper.py` — PlayerBioScraper for AGE/POSITION/HEIGHT/WEIGHT enrichment
  - `src/data/injury_history_logger.py` — InjuryHistoryLogger for persistent longitudinal history
  - `update_data.py`: bio enrichment, injury logging, Parquet dual-write
  - `train.py`: lifecycle aging precomputation at startup
  - `config/default.yaml`: `lifecycle:` config section
- Full suite: TBD (new lifecycle tests added).

### Season-context feature groups + rest cap + phase-aware drift

- Status: DONE on 2026-05-23.
- Delivered:
  - `SeasonPhaseFeatureGroup` — early-season rust, trade resets (4 output columns)
  - `TeamMotivationFeatureGroup` — tanking, load management, playoff lock signals (4 output columns)
  - `PostseasonContextFeatureGroup` — playoff detection, pace prior (2 output columns)
  - `DAYS_SINCE_LAST_GAME` capped at 14 days in `RestGameDensityFeatureGroup`
  - `DriftDetector` phase-aware: `phase` parameter on `record()`/`detect()`/`record_and_detect()` with auto-inference from date
  - Wired into `FeatureEngineer._build_groups()` (26 total groups), `full` preset, `config/default.yaml`, `SAFE_PREFIXES`
  - All feature groups use batched-assembly pattern, no pandas fragmentation
- Full suite: `279 passed, 1 skipped`.

### Apply planned bug-fix batch (groupby KeyError, CatBoost GPU callback, device AttributeError)

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

### Calibration & probability upgrade — distribution fitter, copula, CRPS, variance optimizer

- Status: DONE on 2026-05-21.
- Delivered:
  - `src/query/distribution_fitter.py` — `DistributionFitter` derives Mean/Std/Skew/Zero-Prob/Lambda from P10/P50/P90 quantile outputs.
  - `src/query/empirical_covariance.py` — `CovarianceCache`: archetype-conditioned 6x6 empirical correlation matrices from residual analysis. Cached as `.npz`.
  - `ProbabilityCalculator.run_copula_simulation()` — correlated multi-stat Monte Carlo using Gaussian copula + archetype correlations.
  - `ReportGenerator._enrich_with_distributions()` — appends `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA` to projection CSV exports.
  - `calculate_empirical_crps()` in `src/evaluation/metrics.py` — O(n log n) CRPS via Gini mean difference.
  - `optimize_variance.py` — standalone CLI to tune 7 context-specific volatility multipliers via CRPS + scipy Nelder-Mead.
  - `ProbabilityCalculator` accepts optional `CovarianceCache`; lazy-loads default on first use.
- Full suite: `169 passed, 0 failed`.

### Smart per-target feature selection + weight bootstrap + backtest JSON

- Status: DONE on 2026-06-04.
- Delivered:
  - `src/evaluation/feature_group_ablation.py` — `FeatureGroupAblator` + `AblationReport` + `GroupScore`: per-target MAE deltas from leave-one-out group ablation. Uses fast `HistGradientBoostingRegressor` for screening.
  - `src/evaluation/shadow_feature_filter.py` — `ShadowFeatureFilter` + `ShadowFilterResult` + `SHADOW_COLUMNS`: injects `SHADOW_RANDOM_NORMAL`, `SHADOW_RANDOM_UNIFORM`, `SHADOW_PERMUTED_TARGET` control columns and uses their median importance as a noise floor.
  - `src/evaluation/smart_feature_selector.py` — `SmartFeatureSelector` + `ProfileConfig` + `SelectorConfig` + `SelectionManifest` + `TargetSelection` + `load_manifest`: combines 5 signals (`0.40 * backtest_gain + 0.25 * stability + 0.20 * catboost_importance + 0.10 * permutation_importance - 0.05 * missingness_penalty`) into a per-target final score. Profiles: `fast` / `balanced` / `max_accuracy`.
  - `TrainingPipeline.apply_feature_selection_manifest()` + `_feature_cols_for_target()` — per-target feature lists consumed by CatBoost training; falls back to canonical `self.feature_cols` when no manifest is loaded.
  - `TrainingPreset.feature_selection` + `TrainingPreset.feature_selection_profile` — opt-in via `config/default.yaml`.
  - `Config.feature_selection` + `Config.feature_selection_profiles` — YAML config block.
  - `train.py --feature-selection {off,smart}` and `train.py --selection-profile {fast,balanced,max_accuracy}` — CLI flags. Failure is non-fatal.
  - `FeatureSelector.select_features_for_target(df, target, allowed_features=None)` — inference-time hook for the manifest.
  - `FeatureEngineeringResult.n_rows` + `n_features` — selector diagnostics.
  - `ModelManager.load_models()` bootstraps `EnsembleWeights` from `WeightStore` after the legacy blend is loaded.
  - `TrainingPipeline._save_blend_weights()` writes the training-time blend to `WeightStore` (versioned JSON) so the bootstrap path can pick it up.
  - `backtest_result_to_json_dict()` in `metrics.py` — stable JSON serializer for `BacktestResult`.
  - `backtest.py --json-output <path>` — stable machine-readable JSON payload.
  - `model_stack_metadata.pkl` records `feature_selection_enabled`, `feature_selection_target_specific`, `feature_selection_profile`, and `selected_features_by_target`.
  - `tests/test_evaluation/test_smart_feature_selector.py` (19 tests) and `tests/test_evaluation/test_backtest_json_output.py` (1 test) cover the manifest contract, end-to-end selector, JSON serialization, and feature schema round-tripping.

### Cross-boundary contract wiring + WeightStore bootstrap (DR-030, DR-031)

- Status: DONE on 2026-06-04.
- Delivered:
  - `src/data/schedule_scraper.py::ScheduleScraper` calls `normalize_schedule_frame(...)` on every read path (cached schedule hit, fresh API, cache fallback, season cache). Empty frames are skipped from normalization.
  - `src/simulation/season_simulator.py::SeasonSimulator.simulate_season` converts the schedule frame to `ScheduleGame` records via `schedule_rows_to_games(...)` before iterating matchups (both ThreadPoolExecutor and sequential paths).
  - `src/query/projection_loader.py::ProjectionLoader.load_projections` calls `validate_projection_frame(...)` on every load and re-raises the typed `ProjectionSchemaContractError`. The legacy `test_missing_tov_columns_loads_with_defaults` test is renamed to `test_missing_tov_columns_fails_loudly` to match the new strict behavior.
  - `src/simulation/report_generator.py::ReportGenerator.export_player_projections` writes the strict 6-stat x 8-column schema and calls `validate_projection_frame(...)` on the assembled DataFrame before writing the CSV.
  - `src/models/model_manager.py::ModelManager.predict_player_stats` calls `load_expected_feature_cols(models_dir)` and `align_feature_frame(df, expected_cols)` before the leakage-safe selector runs.
  - `train.py` calls `validate_runtime_artifacts(ArtifactContract(...))` at the bottom of its training flow as a post-train check.
  - `TrainingPipeline._save_blend_weights()` writes the training-time blend to both the legacy `blend_weights.pkl` (so `ArtifactContract` validation continues to pass) and the versioned `WeightStore` (so the bootstrap path can pick it up on the next load).
  - Decision records: `DR-030` (WeightStore bootstrap), `DR-031` (cross-boundary contract wiring).
  - `tests/test_query/test_six_stat_contract.py` updated to reflect the strict schema.
  - `tests/test_contracts/test_pipeline_contract_smoke.py` covers the smoke path through the contract validator.
- Targeted subset on 2026-06-11: `pytest tests/test_evaluation/ tests/test_contracts/` -> `23 passed` (the new smart-selector suite + contracts smoke test). Full-suite baseline of `313 passed, 1 skipped` from 2026-06-04 still applies.

### Run residual interval calibration on real residual data

- Why it matters: Ticket 4 is implemented and unit-tested, but `models/calibration/` must be generated from the real walk-forward residual dataset before live predictions can show meaningful calibrated ranges.

### Add residual correction monitoring system — DONE

- Completed 2026-06-18.
- Delivered:
  - `monitor_residual_corrections.py` — CLI entry point with config-driven input resolution, threshold builder, and report writing.
  - `src/evaluation/residual_monitor.py` — `ResidualMonitor`: pure evaluation with per-target `StatReport` (overall metrics, data-quality breakdown, confidence breakdown, rolling-window status, status labels, recommendations). 786 lines.
  - `src/evaluation/residual_report.py` — `write_report()`, `write_json_report()`, `write_latest_summary()`, `write_csv_report()`, `render_console_summary()`, `report_to_dataframe()`, `_json_safe()` for strict NaN-free JSON. 309 lines.
  - `config/default.yaml` — `residual_monitoring:` block (min_rows=500, helping/hurting thresholds, neutral_band=±1%, min_window_rows=50, windows_days=[7,14,30]).
  - `tests/test_evaluation/test_residual_monitor.py` — 47 tests.
  - Decision record: DR-033.

### Add fast training diagnostic mode — DONE

- Completed 2026-06-14.
- Delivered:
  - `src/training/diagnostics.py` with `diagnostic_stage` context manager, `diagnostic_noop` for skipped stages, `print_data_summary`, `print_selection_summary`, `DiagnosticConfig`, `DiagnosticStop`, `DiagnosticStageFailed`, and `STAGES_ORDERED`.
  - `train.py --diagnose` and `train.py --diagnose --stop-after <stage>` for all six stages: `preflight`, `data_load`, `feature_engineering`, `feature_selection`, `prepare_data`, `artifact_check`.
  - `--diagnose` alone stops before model training with a correct message.
  - `--diagnose --stop-after artifact_check` validates `models/` artifacts using the resolved preset's `transformer_enabled` setting.
  - `--diagnose --feature-selection off --stop-after feature_selection` prints SKIP and stops correctly.
  - Each stage prints `[TRAIN-DIAG] START/OK/FAILED/SKIP` markers using custom `DiagnosticStageFailed` / `DiagnosticStop` exceptions (no `sys.exit()` in the library).
  - After data load and feature engineering, diagnostic mode prints row/column counts and target-column presence summary.
  - `--stop-after` requires `--diagnose` (parsed error if used alone).
  - `tests/test_training/test_diagnostics.py` (28 tests) covering the context manager, noop, exceptions, summary helpers, config, stage ordering, and subprocess integration.
- `pytest tests/test_training/test_diagnostics.py -q` -> `28 passed`.
- Existing tests unaffected: `pytest tests/test_training/ tests/test_contracts/ tests/test_models/test_model_manager.py -q` -> `75 passed`.
