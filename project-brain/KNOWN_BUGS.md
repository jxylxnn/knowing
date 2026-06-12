# Known Bugs

This file tracks confirmed or strongly suspected defects and weak points visible in the repository as of 2026-04-01.

---

## KB-001: Training Does Not Persist Required CatBoost Runtime Artifacts

- Status: fixed in code on 2026-04-01, confirmed via unit tests on 2026-04-11
- Severity: critical
- Confidence: high

### Symptom

- A training run can appear successful, but downstream simulation cannot load the expected CatBoost model files.

### Expected Behavior

- `train.py` should produce the CatBoost artifacts that `simulate_season.py` and `src/models/model_manager.py` expect in `models/`.

### Evidence

- Historical evidence for the defect:
  - `simulate_season.py` checked for `models/pts_catboost.cbm`.
  - `src/models/model_manager.py` expected per-target CatBoost artifacts and metadata files.
  - The active training path previously saved schema/blend/Transformer artifacts without persisting CatBoost runtime files.
- Fix evidence now in repo:
  - `src/training/pipeline.py` explicitly saves per-target CatBoost artifacts and validates the runtime artifact set before returning success.
  - `src/training/catboost_trainer.py` now exposes shared runtime-artifact validation helpers and raises clearly on missing files.
  - `src/models/model_manager.py` validates the shared runtime contract before loading.
  - `simulate_season.py` now uses `ModelManager.load_models()` instead of a one-file existence check.
  - `tests/test_training/test_runtime_artifact_contract.py` verifies artifact creation, fail-loud behavior, and fresh-process `ModelManager` loading.

### Reproduction

- Original defect was confirmed by code inspection.
- Live CLI confirmation update on 2026-04-11:
  - the venv has been rebuilt with Python 3.12 and all dependencies import correctly
  - data files are now available via `update_data.py`
  - full training run deferred due to runtime length, but 110/110 unit tests pass including artifact-contract tests
  - preflight checks confirmed: missing CSV and unwritable models dir both fail clearly

### Suspected Cause

- The pipeline trains CatBoost models but does not call the persistence path that `catboost_trainer.py` provides.

### Workaround

- No workaround should be needed after retraining in a healthy local environment with the fixed code.

### Fix Ideas

- Implemented:
  - save each trained CatBoost target and metadata from the active pipeline
  - validate required runtime artifacts before training reports success
  - validate the same contract before runtime loading
  - add focused regression tests around the save/load seam

### Risks

- The code path is fixed, but a real-data CLI smoke test is still needed once the local runtime and data prerequisites are healthy.

### Related Files

- `train.py`
- `src/training/pipeline.py`
- `src/training/catboost_trainer.py`
- `src/models/model_manager.py`
- `simulate_season.py`

---

## KB-002: `ScheduleScraper` Uses Undefined Uppercase Instance Attributes

- Status: fixed in code on 2026-04-02, pending live CLI confirmation
- Severity: critical
- Confidence: high

### Symptom

- Schedule fetch operations likely raise `AttributeError` on normal execution paths.

### Expected Behavior

- `src/data/schedule_scraper.py` should consistently use initialized config-backed instance attributes.

### Evidence

- Historical evidence:
  - the class initialized lowercase fields such as `self.max_retries`, `self.retry_delay`, `self.cache_ttl_hours`, and `self.season_cache_ttl_days`
  - other methods referenced `self.MAX_RETRIES`, `self.RETRY_DELAY`, and `self.CACHE_TTL_HOURS`
- Fix evidence now in repo:
  - schedule fetch paths now use lowercase config-backed attributes consistently
  - schedule retrieval records explicit required-input health for success, cache fallback, and failure
  - `tests/test_data/test_scraper_health.py` exercises the fetch path without `AttributeError`

### Reproduction

- Static inspection during this audit.
- High-probability live failure path for `simulate_season.py --today`, `--date`, `--week`, or `--season`.

### Suspected Cause

- Refactor drift between old constant-style naming and newer instance-attribute style naming.

### Workaround

- None should be needed after the code fix, but live upstream confirmation is still pending.

### Fix Ideas

- Normalize all references to one naming style.
- Add focused tests for schedule fetch and cache expiration logic.

### Risks

- Upstream schedule/API changes can still fail, but the CLI now surfaces that as a hard-required input failure.

### Related Files

- `src/data/schedule_scraper.py`
- `simulate_season.py`
- `src/simulation/season_simulator.py`

---

## KB-003: `LineupScraper` Leaves Critical State Uninitialized And References Undefined Names

- Status: fixed in code on 2026-04-02, pending live upstream validation
- Severity: high
- Confidence: high

### Symptom

- Lineup retrieval fails internally and lineup context is silently dropped from simulations.

### Expected Behavior

- `LineupScraper` should initialize caches and required constants before any lineup fetch method uses them.

### Evidence

- Historical evidence:
  - in `_get_config_value`, code after `return obj` was unreachable
  - `self._lineup_cache` and coach tendency state were never initialized
  - the file referenced undefined names such as `TEAM_ID_MAP`, `NBA_HEADERS`, and `ID_TO_TEAM`
- Fix evidence now in repo:
  - constructor state now initializes in `__init__`
  - team/header references now use `ABBR_TO_ID`, `ID_TO_ABBR`, and class-owned headers
  - failed lineup lookup now returns explicit failed health instead of looking like a healthy empty result
  - `tests/test_data/test_scraper_health.py` covers constructor state and failed-source behavior

### Reproduction

- Static inspection during this audit.

### Suspected Cause

- Broken edit/merge left initialization code inside the wrong function and removed or failed to import required constants.

### Workaround

- `GameSimulator` still allows lineup failure as degraded optional context, but the degradation is now explicit in result metadata and CLI output.

### Fix Ideas

- Move initialization into `__init__`.
- Replace undefined constants with real module-level values or config.
- Add tests for constructor integrity and lineup fetch fallback behavior.

### Risks

- Projected/inferred lineups remain weaker than confirmed lineups, but the distinction is now operator-visible.

### Related Files

- `src/data/lineup_scraper.py`
- `src/simulation/game_simulator.py`

---

## KB-004: `BasketballRefScraper` Appears To Have The Same Config-Attribute Regression Pattern

- Status: fixed in code on 2026-04-02, pending live site confirmation
- Severity: medium
- Confidence: medium-high

### Symptom

- Calls into the Basketball Reference scraper likely fail due to missing uppercase attribute names.

### Expected Behavior

- The scraper should consistently use initialized config-backed attributes.

### Evidence

- Historical evidence:
  - `__init__` set lowercase names such as `max_retries`, `retry_delay`, and `cache_ttl_hours`
  - methods referenced uppercase variants such as `MAX_RETRIES`, `RETRY_DELAY`, and `CACHE_TTL_HOURS`
  - the file also referenced Basketball Reference abbreviation maps it did not define or import
- Fix evidence now in repo:
  - fetch/cache paths now use lowercase config-backed state consistently
  - Basketball Reference abbreviation maps are now derived from `src/utils/team_mappings.py`
  - `tests/test_data/test_scraper_health.py` exercises the fetch path successfully

### Reproduction

- Static inspection only.

### Suspected Cause

- Same refactor drift seen in `ScheduleScraper`.

### Workaround

- Betting-line fallbacks may reduce user-visible failures, but quality degrades.

### Fix Ideas

- Audit the whole scraper class for mixed naming.
- Add a regression test that exercises one normal call path.

### Risks

- Betting-context features can still degrade when upstream HTML changes, but the fallback is now reported through simulator input-health summaries rather than hidden.

### Related Files

- `src/data/basketball_ref_scraper.py`
- `src/data/betting_scraper.py`

---

## KB-005: Exported Projection CSV Omits `STL`, `BLK`, And `TOV`

- Status: fixed in code on 2026-04-12, doc updated on 2026-04-25
- Severity: high
- Confidence: high

### Symptom

- Query code supports steals, blocks, and turnovers, but cached projection exports do not appear to include those stat columns.

### Expected Behavior

- `player_projections_<timestamp>.csv` should include all supported target stats if the query CLI claims support for them.

### Evidence

- `src/simulation/report_generator.py` exports points, rebounds, and assists columns.
- `src/query/projection_loader.py` tries to read additional stats and falls back when they are missing.
- `query_prob.py` and `src/query/interactive_cli.py` expose `stl`, `blk`, and `tov` query paths.

### Reproduction

- Static inspection of export and loader code.

### Suspected Cause

- Report export schema lagged behind broader model/query support.

### Workaround

- None needed after fix.

### Fix Ideas

- Implemented:
  - `src/simulation/report_generator.py` now exports all six stat columns (PTS, REB, AST, STL, BLK, TOV) with MEAN/MODE/CI for each.
  - `src/query/projection_loader.py::STAT_COLUMNS` now maps all six stats, and `_row_to_projection()` uses the mapping uniformly.
  - Regression tests in `tests/test_query/test_six_stat_contract.py` cover the full export/load contract.

### Risks

- None remaining; the export and loader are aligned.

### Related Files

- `src/simulation/report_generator.py`
- `src/query/projection_loader.py`
- `src/query/interactive_cli.py`
- `query_prob.py`

---

## KB-006: `simulate_season.py --season` Does Not Match Its CLI Promise

- Status: fixed in code on 2026-04-21, pending live upstream confirmation
- Severity: medium
- Confidence: high

### Symptom

- The CLI suggests simulation of all remaining season games, but implementation fetched only about the next 30 days.

### Expected Behavior

- `--season` should either simulate the full remaining season or be renamed/documented as a limited horizon.

### Evidence

- `ScheduleScraper.get_remaining_season()` previously iterated over a fixed next-30-day window rather than a full remaining schedule.
- Fix evidence now in repo:
  - `get_remaining_season()` now computes the season end date (June 30 of the season's second year) and iterates from today through that date.
  - The forward window is capped at 180 days to avoid excessive API calls during the offseason or if the season string is malformed.
  - A clear TODO comment documents the upstream limitation (no single "remaining games" endpoint in nba_api).
  - `tests/test_data/test_scraper_health.py::test_schedule_scraper_remaining_season_horizon` verifies the capped-horizon behavior.

### Reproduction

- Static inspection during this audit.
- Unit test `test_schedule_scraper_remaining_season_horizon` exercises the capped-horizon logic.

### Suspected Cause

- Placeholder implementation was never upgraded to true season scope.

### Workaround

- No workaround needed after the fix; the method now fetches the real remaining season up to the cap.

### Fix Ideas

- Implemented:
  - compute season end from the season string
  - iterate daily scoreboard calls through the computed end date
  - cap at 180 days with a logged warning
  - add regression test for the horizon cap

### Risks

- Daily scoreboard composition is slower than a single season endpoint would be, but the cap prevents runaway API usage.
- Upstream schedule/API changes can still fail, but the CLI now surfaces that as a hard-required input failure.

### Related Files

- `src/data/schedule_scraper.py`
- `simulate_season.py`

---

## KB-007: `ProjectionLoader` Hardcodes A Defense Cache Filename For Season `2025-26`

- Status: fixed in code on 2026-04-25
- Severity: medium
- Confidence: medium-high

### Symptom

- Defense-context loading will go stale or fail after season rollover if the cache filename changes.

### Expected Behavior

- Defense cache lookup should derive the active season dynamically or handle multiple candidate files.

### Evidence

- `src/query/projection_loader.py` previously referenced `cache/all_team_defense_2025-26.json`.

### Reproduction

- Static inspection only.

### Suspected Cause

- Temporary season-specific implementation hardened into production code.

### Workaround

- None needed after fix.

### Fix

- `_load_cached_defense_data()` now resolves `cache_dir`, searches `cache_dir.glob('all_team_defense_*.json')`, picks the lexicographically last file (newest season), and falls back to the original hardcoded path if no candidates exist.

### Risks

- None remaining; lookup now adapts to season rollover.

### Related Files

- `src/query/projection_loader.py`

---

## KB-008: `GameSimulator` Contains Large Unreachable Legacy Logic After An Early Return

- Status: fixed in code on 2026-04-25, confirmed removed on 2026-05-09 as part of simulation refactor
- Severity: medium
- Confidence: high

### Symptom

- The file previously contained a large simulation block that was dead, making it unclear which algorithm was truly active.

### Expected Behavior

- There should be one clearly documented active simulation path, with obsolete logic removed or isolated.

### Evidence

- `simulate_matchup(...)` returned `self._simulate_matchup_reactive(...)` and then retained a long block of older simulation code below that return (formerly lines 1134–1373).
- **2026-05-09:** Dead code fully removed as part of simulation refactor. GameSimulator now delegates to PhaseSimulator for the core Monte Carlo loop.

### Fix

- Removed the entire unreachable legacy block. The active path is now exclusively `_simulate_matchup_reactive`, which delegates to `PhaseSimulator.simulate()`.

### Risks

- None remaining; the dead code can no longer mislead contributors.

### Related Files

- `src/simulation/game_simulator.py`
- `src/simulation/phase_simulator.py`

---

## KB-011: Six Training-Stopping Bugs in CatBoost + Transformer Pipeline

- Status: fixed in code on 2026-05-09
- Severity: critical
- Confidence: high

### Symptom

- Training could crash in multiple places depending on environment: missing CatBoost install, empty feature sets, DataLoader on macOS, parallel thread contention, Transformer checkpoint corruption.

### Evidence

- CatBoost import path would crash when catboost not installed.
- feature_cols could be None, crashing downstream consumers.
- A bare `raise` statement swallowed the original exception.
- macOS DataLoader with persistent_workers=True would crash.
- Feature missingness could cause hard crash instead of graceful degradation.
- CPU CatBoost training with multiple workers could cause thread contention.
- Transformer checkpoint save was not guarded against partial writes.

### Fix

- CatBoost: import guard + informative error message.
- feature_cols: null check with fallback to empty list.
- Bare raise: replaced with proper `raise ValueError(...) from e`.
- macOS: `persistent_workers=False` guard when platform is Darwin.
- Missing features: graceful degradation with logged warning.
- Thread contention: `max_workers=1` for CPU training when OMP_NUM_THREADS is set.
- Transformer save: temp file + atomic rename pattern for checkpoint persistence.

### Risks

- None remaining.

### Related Files

- `src/training/catboost_trainer.py`
- `src/training/pipeline.py`
- `src/training/nn_trainer.py`
- `src/models/transformer_model.py`
- `train.py`

---

## KB-009: Rolling Feature Generation Emits Heavy Pandas Fragmentation Warnings

- Status: fixed in code on 2026-04-02, synthetic benchmark completed
- Severity: medium
- Confidence: high

### Symptom

- Test runs previously emitted many `PerformanceWarning: DataFrame is highly fragmented` messages.

### Expected Behavior

- Feature generation should add columns in a way that avoids severe DataFrame fragmentation on large datasets.

### Evidence

- Before the refactor, running `pytest tests/ -q` on 2026-04-01 produced 1612 warnings, dominated by `src/preprocessing/features/rolling.py` lines 73-76.
- The refactor now batches new columns in `RollingFeatureGroup`, `EfficiencyFeatureGroup`, and `MomentumFeatureGroup` before concatenating them back into the frame.
- `tests/test_preprocessing/test_feature_engineer.py` includes a regression check that the feature-engineering path does not emit pandas `PerformanceWarning` fragmentation warnings.
- The full test suite after the fix completed with `94 passed, 2 skipped`.

### Reproduction

- Historical reproduction was `pytest tests/ -q` in the audited workspace.

### Suspected Cause

- Repeated column insertion on pandas DataFrames instead of batched concatenation.

### Workaround

- No workaround should be needed after the code fix.

### Fix Ideas

- Implemented:
  - rebuild affected feature sets in temporary dicts/Series
  - concatenate once per feature group
  - add a regression test for the warning flood

### Risks

- A future feature group could reintroduce the same anti-pattern if new columns are appended one by one.

### Follow-Up

- A real-data production-scale profile can still be collected later if the team wants a stricter runtime baseline.

### Related Files

- `src/preprocessing/features/rolling.py`
- `src/preprocessing/feature_engineer.py`

---

## KB-010: Transformer Validation Crashes On Compiled CUDA Flash-Attention Path

- Status: fixed in code on 2026-04-02, transformer tests confirmed passing on 2026-04-11 (CPU), pending live CUDA smoke confirmation
- Severity: critical
- Confidence: high

### Symptom

- `train.py` can finish fitting the Transformer and save `attention_transformer.pkl`, then fail during validation batch prediction with `CUDA error: invalid configuration argument`.

### Expected Behavior

- Transformer validation inference should complete after training, even on CUDA systems that expose flash-attention kernels.

### Evidence

- Repro description from the user:
  - the crash occurred after epoch 18 during Transformer validation prediction
  - the failure path ran through `_predict_transformer_batch`, `torch.compile`, and `_scaled_dot_product_flash_attention`
- Fix evidence now in repo:
  - `src/models/transformer_model.py` keeps an eager model for validation/runtime inference and only uses compile when explicitly allowed.
  - The same module now prefers a math SDPA backend during eager inference on CUDA when backend controls are available.
  - `src/training/pipeline.py` no longer calls the compiled model directly for validation batches.
  - `src/config/model_config.py` generates Transformer configs with compile disabled by default.
  - `train_colab.ipynb` now raises on nonzero subprocess exit codes instead of printing completion unconditionally.
  - Regression tests cover the eager validation path and wrapper delegation.

### Reproduction

- Historical reproduction came from a CUDA training run in this workspace.

### Suspected Cause

- `torch.compile` combined with the runtime's flash-attention SDPA path produced an unstable CUDA kernel configuration for validation inference.

### Workaround

- Run Transformer validation and runtime prediction through the eager model path.

### Fix Ideas

- Implemented:
  - disable compile by default for the Transformer stack
  - keep validation inference eager
  - prefer math SDPA for eager CUDA validation
  - make notebook launch cells fail loudly on subprocess errors

### Risks

- If compile is re-enabled later, the compiled validation path must be re-audited on the target CUDA/runtime combination.

### Related Files

- `src/models/transformer_model.py`
- `src/training/pipeline.py`
- `src/config/model_config.py`
- `train_colab.ipynb`

---

## KB-011: Colab Training Cell Hid The Real `train.py` Error Output

- Status: fixed in code on 2026-04-02, pending live Colab confirmation (torch-shim and legacy-load bugs fixed on 2026-04-11)
- Severity: high
- Confidence: high

### Symptom

- The notebook training cell raised `CalledProcessError` with exit status 1, but the operator could not see the actual stdout/stderr emitted by `train.py`.

### Expected Behavior

- The notebook should print the full subprocess stdout and stderr, show the return code, and stop immediately on failure so the real traceback is visible.

### Evidence

- Historical notebook code used `subprocess.run(train_cmd, check=True)` without capturing or printing subprocess output.
- The notebook cell also did not preflight the required raw CSV inputs or verify that the Google Drive models directory was writable before launch.
- Fix evidence now in repo:
  - `train_colab.ipynb` now uses `subprocess.run(..., capture_output=True, text=True)` and prints stdout/stderr before raising on nonzero exit.
  - The notebook now checks for `nba_players.csv` and `nba_games.csv` under the configured Drive data directory.
  - The notebook now verifies that the Drive models directory can be written to before launching training.
  - `train.py` now logs explicit stage names and preflights writable runtime directories so direct CLI runs fail more clearly too.

### Reproduction

- Historical repro was the user-reported notebook failure.

### Suspected Cause

- The wrapper suppressed the actual subprocess logs, and the notebook launched training without enough preflight context to isolate path or permission issues.

### Workaround

- Use the hardened notebook cell or run `python train.py` directly from a terminal to get the full traceback.

### Fix Ideas

- Implemented:
  - capture and print stdout/stderr
  - print the return code
  - fail fast on missing inputs and unwritable models directory
  - add stage logging to the CLI

### Risks

- The live Colab environment still needs a real mounted Drive smoke test to confirm the exact user-facing output path.

### Related Files

- `train_colab.ipynb`
- `train.py`

---

## KB-012: Colab Launcher Assumed The Drive Models Directory Contained `train.py`

- Status: fixed in code on 2026-04-02, pending live Colab confirmation (torch-shim fix confirmed on 2026-04-11)
- Severity: high
- Confidence: high

### Symptom

- The Colab training launcher built `project_root` from `/content/drive/MyDrive/nba_model/models` and then looked for `/content/drive/MyDrive/nba_model/train.py`, which failed when the repo was actually cloned under a separate Colab working directory such as `/content/knowing`.

### Expected Behavior

- The notebook should resolve the actual repo checkout separately from the persistent Drive-backed `data/` and `models/` directories.
- `train.py` should be launched from the real repo root, and a missing script should fail with explicit path diagnostics.

### Evidence

- Historical launcher code derived `project_root = drive_models_path.parent`, which only works if code and persisted artifacts share the same root.
- The launcher then built `train_script = project_root / "train.py"`, so any Colab checkout outside Drive caused a preflight `FileNotFoundError`.
- Fix evidence now in repo:
  - `train_colab.ipynb` now resolves a repo checkout independently of Drive storage paths.
  - The notebook accepts an explicit `project_root_override` and otherwise checks common candidates like `os.getcwd()` and `/content/knowing`.
  - The launcher prints the resolved `project_root`, `train_script`, `data_dir`, and `models_dir` before launch.

### Reproduction

- Historical reproduction path from user report and notebook inspection:
  - repo cloned into `/content/knowing`
  - data and models intended for `/content/drive/MyDrive/nba_model/data` and `/content/drive/MyDrive/nba_model/models`
  - training cell looked for `/content/drive/MyDrive/nba_model/train.py` and failed before subprocess launch

### Suspected Cause

- The launcher conflated code location with persistent storage location.

### Workaround

- Clone the repo into the working directory the notebook expects, or set `project_root_override` explicitly until the fix is verified in a live Colab run.

### Fix Ideas

- Implemented:
  - explicit repo-root resolution
  - independent Drive-backed data/models paths
  - path-diagnostic failure if no candidate contains `train.py`

### Risks

- The new resolver still needs a real Colab run to confirm the exact mounted-Drive behavior and the printed diagnostics.

### Related Files

- `train_colab.ipynb`
- `project-brain/CURRENT_STATE.md`
- `project-brain/TASKS.md`

---

## KB-013: FeatureSchema Import Contract Was Too Implicit For Training Startup

- Status: fixed in code on 2026-04-02, verified with targeted regression test; live import confirmed on 2026-04-11
- Severity: high
- Confidence: medium

### Symptom

- Training startup was reported to fail when `src/training/pipeline.py` imported `FeatureSchema` from `src.utils.prediction_utils`, because the import contract was not explicit enough to protect against stale or divergent environments.

### Expected Behavior

- `FeatureSchema` should be importable from `src.utils.prediction_utils`.
- The package-level `src.utils` namespace should also re-export `FeatureSchema` so compatibility is preserved if callers import from the utility package instead of the module directly.

### Evidence

- The current source already defines `FeatureSchema` in `src/utils/prediction_utils.py`, but the contract was not guarded by an explicit public export list or a regression test.
- Fix evidence now in repo:
  - `src/utils/prediction_utils.py` now declares `FeatureSchema` in `__all__`.
  - `src/utils/__init__.py` re-exports `FeatureSchema` through its lazy attribute loader.
  - `tests/test_training/test_runtime_artifact_contract.py` now validates a clean-process import of both `src.utils.prediction_utils.FeatureSchema` and `src.utils.FeatureSchema`, plus `src.training.pipeline`.

### Reproduction

- Clean-process import check in the repo venv now succeeds:
  - `from src.utils.prediction_utils import FeatureSchema`
  - `from src.utils import FeatureSchema`
- Regression test also imports `src.training.pipeline` in a subprocess with a stub `torch` module to avoid local CUDA runtime crashes.

### Suspected Cause

- The import contract was too implicit for a repository that relies on stable module/file interfaces.

### Workaround

- Importing `FeatureSchema` directly from `src.utils.prediction_utils` works in the current source tree, but the package re-export is the safer compatibility path.

### Fix Ideas

- Implemented:
  - explicit module exports
  - package-level compatibility re-export
  - subprocess regression coverage

### Risks

- If `FeatureSchema` is moved again, the re-export and regression test should be updated in the same change.

### Related Files

- `src/utils/prediction_utils.py`
- `src/utils/__init__.py`
- `src/training/pipeline.py`
- `tests/test_training/test_runtime_artifact_contract.py`

---

## KB-014: `train.py` Could Fail When `FeatureEngineer` Lacked `disable_groups`

- Status: fixed in code on 2026-04-03; ablation-benchmark path also verified with targeted preprocessing and entrypoint regression tests on 2026-04-11
- Severity: high
- Confidence: high

### Symptom

- The training CLI could fail during Step 2 feature engineering if the active runtime checkout exposed a `FeatureEngineer` constructor that did not accept the `disable_groups` keyword used by `train.py`.

### Expected Behavior

- The training CLI should construct `FeatureEngineer` successfully in both current and older checkouts.
- If a checkout lacks direct `disable_groups` support, the CLI should still disable the selected feature groups by applying the filter after instantiation.

### Evidence

- `train.py` builds a `disable_groups` list during feature-ablation selection and passes it into the feature-engineering setup path.
- The current repository already accepts `disable_groups`, but the user-reported failure indicates a stale or divergent runtime checkout where that constructor keyword was missing.
- Fix evidence now in repo:
  - `src/preprocessing/feature_engineer.py` now exposes `build_feature_engineer(...)`, which filters supported constructor kwargs and backfills `disable_groups`/`disable_columns` attributes when necessary.
  - `train.py` now uses that helper for Step 2 feature-engineering setup and for the ablation probe.
  - `src/preprocessing/feature_engineer.py::benchmark_feature_variants()` now uses the same helper for the internal variant engineers, so the benchmark path itself stays compatible with older constructors.
  - `tests/test_preprocessing/test_feature_engineer.py` covers both the current constructor contract and a simulated legacy constructor without `disable_groups`, including the ablation benchmark path.
  - `tests/test_training/test_train_entrypoint.py` statically verifies that the entrypoint keeps calling `build_feature_engineer(...)` with the ablation filters instead of passing them directly into `FeatureEngineer(...)`, and it rejects a bare Step 2 `FeatureEngineer()` instantiation.

### Reproduction

- Compatibility regression test simulates an older constructor without `disable_groups` and verifies the helper still applies the requested group filter.
- Entrypoint regression test parses `train.py` and verifies the active Step 2 call site routes the ablation filters through `build_feature_engineer(...)` without a bare constructor call in the runtime path.
- The same regression set also exercises `FeatureEngineer.benchmark_feature_variants()` against a simulated legacy constructor, proving the ablation benchmark can no longer crash on the missing keyword.

### Suspected Cause

- Version skew between the CLI caller and the `FeatureEngineer` constructor contract.

### Workaround

- The new helper path makes the training CLI tolerant of older constructors while preserving the current `disable_groups` behavior.

### Fix Ideas

- Implemented:
  - compatibility-safe constructor helper
  - canonical training-path call site
  - regression coverage for both current and legacy constructor signatures

### Risks

- If `FeatureEngineer` adds new required constructor parameters later, the helper should be updated in the same change.
- A stale Colab checkout can still reproduce the original crash even though the active repo code path is fixed.

### Related Files

- `train.py`
- `src/preprocessing/feature_engineer.py`
- `tests/test_preprocessing/test_feature_engineer.py`
- `tests/test_training/test_train_entrypoint.py`

---

## KB-015: Silently Uncalibrated Predictions When Transformer Artifact Is Missing

- Status: fixed in code on 2026-04-11, confirmed via unit tests on 2026-04-11
- Severity: high
- Confidence: high

### Symptom

- When `attention_transformer.pkl` is missing at runtime but `blend_weights.pkl` contains non-zero Transformer weights, `ModelManager` silently falls back to CatBoost-only predictions. The blend weights effectively scale the CatBoost prediction by only its weight fraction, dropping the Transformer's contribution entirely (e.g., `0.7 * cat_pred + 0` instead of `0.7 * cat_pred + 0.3 * trans_pred`). This produces systematically biased, uncalibrated results without warning.

### Expected Behavior

- If a model was trained with the Transformer, the Transformer artifact must be present for inference, or the system must raise a clear error rather than returning silently wrong numbers.

### Evidence

- `src/models/model_manager.py` previously treated `attention_transformer.pkl` as optional. When the Transformer was missing, `predict_player_stats` used the raw CatBoost prediction without blend-weight correction.
- Fix evidence now in repo:
  - `ModelManager._validate_blend_contract()` raises `FileNotFoundError` when blend weights expect a Transformer but `attention_transformer.pkl` is missing.
  - `ModelManager._validate_blend_contract()` raises `RuntimeError` when the file exists but failed to load.
  - `TrainingPipeline._validate_blend_contract()` enforces the same contract on the pipeline's `load_models()` path.
  - `TrainingPipeline._save_model_stack_metadata()` persists explicit metadata (`model_stack_metadata.pkl`) indicating whether the Transformer was active during training.
  - `tests/test_models/test_model_manager.py` covers missing-Transformer, corrupt-Transformer, zero-weight, and loaded-Transformer scenarios.
  - `tests/test_training/test_runtime_artifact_contract.py` verifies that `ModelManager.load_models()` raises on tampered blend weights.

### Reproduction

- Static inspection of the prediction logic and blend-weight application path.

### Suspected Cause

- The Transformer was treated as optional at runtime, but blend weights were always computed assuming both models contribute. The fallback path never re-normalized weights or warned the operator.

### Workaround

- No workaround needed after the fix. The system now fails loudly instead of producing wrong numbers.

### Fix Ideas

- Implemented:
  - add `_validate_blend_contract()` to both `ModelManager` and `TrainingPipeline`
  - raise `FileNotFoundError` / `RuntimeError` when blend weights require a missing Transformer
  - save `model_stack_metadata.pkl` during training for explicit contract documentation
  - add regression tests for all contract enforcement paths

### Risks

- Existing model directories that were trained with a Transformer but lost the artifact will now fail at load time instead of silently degrading. This is the intended behavior.

### Related Files

- `src/models/model_manager.py`
- `src/training/pipeline.py`
- `tests/test_models/test_model_manager.py`
- `tests/test_training/test_runtime_artifact_contract.py`

---

## KB-016: Torch Shim In `src/__init__.py` Clobbered Real PyTorch During Test Runs

- Status: fixed on 2026-04-11
- Severity: high
- Confidence: high

### Symptom

- All 4 `test_transformer_model.py` tests failed with `No module named 'torch.backends'; 'torch' is not a package` when running under pytest in a healthy Python 3.12 environment with real PyTorch installed.

### Expected Behavior

- The torch shim should only be installed when real PyTorch is not available. When PyTorch is installed and importable, the shim must not interfere.

### Evidence

- `src/__init__.py:_install_test_torch_shim()` checked only `'pytest' in sys.modules` and `'torch' not in sys.modules` before installing a minimal NumPy-backed fake `torch` module. In a healthy environment where real torch is importable but not yet loaded, the shim was installed first (because `import src.models.base` triggers `src/__init__.py` before any explicit `import torch`), replacing the real package.
- The fake module lacked `__path__`, `torch.nn`, `torch.backends`, and all submodules, causing `import torch.backends.cudnn` in `transformer_model.py` to fail.

### Reproduction

- Run `pytest tests/test_models/test_transformer_model.py` in a venv with Python 3.12 and real PyTorch installed.

### Suspected Cause

- The shim was written for an environment where torch import crashed (Python 3.13 with CUDA toolchain issues). It did not account for environments where torch is importable but simply not yet in `sys.modules` at package init time.

### Fix

- Added `importlib.util.find_spec('torch')` check before installing the shim. If real torch is discoverable, the shim is skipped.

### Risks

- If a future environment has torch installed but broken at runtime (e.g., CUDA crashes on import), the shim won't activate and tests that depend on it will need a different fallback.

### Related Files

- `src/__init__.py`
- `tests/test_models/test_transformer_model.py`

---

## KB-017: `TransformerWrapper.load()` Failed On Legacy Checkpoints With Empty Config

- Status: fixed on 2026-04-11
- Severity: medium
- Confidence: high

### Symptom

- `TransformerWrapper.load()` raised `RuntimeError` with state_dict shape mismatches when loading checkpoints that saved `config: {}` (empty dict), because the model was reconstructed with `DEFAULT_CONFIG` (d_model=128, num_layers=4) instead of the architecture that produced the checkpoint.

### Expected Behavior

- Legacy checkpoints should load successfully, inferring the correct architecture from the saved state_dict when config is empty or incomplete.

### Evidence

- The test `test_transformer_wrapper_loads_legacy_three_output_checkpoint` created a wrapper with `d_model=16, nhead=4, num_layers=1` but saved `config: {}`. On load, the wrapper used defaults (d_model=128, num_layers=4), causing shape mismatches in every layer.
- The same issue would affect any real checkpoint saved before the config field was properly populated.

### Reproduction

- Run `pytest tests/test_models/test_transformer_model.py::TestTransformerWrapper::test_transformer_wrapper_loads_legacy_three_output_checkpoint` before the fix.

### Suspected Cause

- The `load()` method used `config or {}` which merged with `DEFAULT_CONFIG`, overriding the actual architecture encoded in the state_dict tensors.

### Fix

- Added `_infer_config_from_state()` static method that extracts `d_model`, `num_layers`, `nhead`, and `dim_feedforward` from tensor shapes in the saved state_dict. When config is empty, the inferred config is used for model reconstruction.

### Risks

- The nhead inference uses a heuristic (tries common divisor values). If a model was trained with an unusual nhead value not in the candidate list, the inference may pick a different nhead that is still compatible (same d_model) but splits attention differently.

### Related Files

- `src/models/transformer_model.py`
- `tests/test_models/test_transformer_model.py`

---

## KB-018: GroupBy Series KeyError in `PaceRoleFeatureGroup`

- Status: verified fixed in current code on 2026-04-25 (no code change required)
- Severity: medium
- Confidence: high

### Symptom

- Historical bug: computed pandas Series objects (`usage_raw`, `reb_opp`, `three_pt_freq`, `ft_rate`, `pts_share`, `ts_pct`) were passed to `df.groupby('PLAYER_ID')[series]` as column selectors, causing `KeyError: 'Columns not found: 0.0, 0.5, ...'`.

### Expected Behavior

- Rolling features should be computed on named DataFrame columns, not raw Series objects.

### Evidence

- `src/preprocessing/features/pace_role.py` now stores all derived metrics as named `RAW_*` columns (`RAW_USAGE`, `RAW_REB_OPPORTUNITY`, `RAW_3PT_FREQ`, `RAW_FT_RATE`, `RAW_PTS_SHARE`, `RAW_TS_PCT`) and references them by string in `groupby('PLAYER_ID')['RAW_*']`.
- No instance of passing a bare Series to `groupby()[...]` remains in the active code.
- Preprocessing tests pass (19/19), including feature-engineering runs that exercise this group.

### Related Files

- `src/preprocessing/features/pace_role.py`
- `tests/test_preprocessing/test_feature_engineer.py`

---

## KB-019: CatBoost GPU Callback Failure

- Status: verified fixed in current code on 2026-04-25 (no code change required)
- Severity: high
- Confidence: high

### Symptom

- Historical bug: CatBoost GPU training failed because user-defined callbacks are not supported on GPU. Fallback also failed because it reused `fit_kwargs` containing the callback.

### Expected Behavior

- GPU training should use built-in `verbose` instead of callbacks; CPU training should retain callbacks; fallback should rebuild `fit_kwargs` cleanly.

### Evidence

- `src/training/catboost_trainer.py:_train_single_model()` sets `use_callback = task_type == 'CPU'` and only adds `callbacks=[callback]` when `use_callback` is true.
- GPU path uses `model_params['verbose'] = 200`.
- `CatBoostProgressCallback.after_iteration()` uses `getattr(info, 'learn_error', None)` and `getattr(info, 'test_error', None)` defensively.
- GPU→CPU fallback rebuilds `fallback_fit_kwargs` from scratch and adds the callback only for the CPU retry.
- Runtime artifact contract tests pass (5/5).

### Related Files

- `src/training/catboost_trainer.py`
- `tests/test_training/test_runtime_artifact_contract.py`

---

## KB-020: Device Type AttributeError (`'str' object has no attribute 'type'`)

- Status: fixed in code on 2026-04-25
- Severity: medium
- Confidence: high

### Symptom

- `self.device` was occasionally a string (`'cuda'`) rather than a `torch.device` object, causing `AttributeError: 'str' object has no attribute 'type'` in `nn_trainer.py:322`.

### Expected Behavior

- `self.device` should always be a `torch.device` object before any code accesses `.type`.

### Evidence

- `BaseTrainer.__init__` already coerces `device` to `torch.device` and raises `TypeError` for unexpected types.
- `NeuralNetworkTrainer.__init__` already accepts `Optional[Union[str, torch.device]]`.
- Applied an additional defensive coercion in `NeuralNetworkTrainer._create_loader()` before accessing `self.device.type`.
- `nn_trainer.py` tests pass (2/2); transformer model tests pass (8/8).

### Fix

- Added `if isinstance(self.device, str): self.device = torch.device(self.device)` inside `_create_loader()`.

### Related Files

- `src/training/nn_trainer.py`
- `src/training/trainer.py`
- `tests/test_training/test_nn_trainer.py`
- `tests/test_models/test_transformer_model.py`


---

## KB-021: Projection CSV Schema Is Now Strict — Legacy CSVs No Longer Load With Defaults

- Status: intentional behavior change (resolved) in 2026-06-04; not a defect
- Severity: low (affects upgrade path, not runtime)
- Confidence: high

### Symptom

- After upgrading to the 2026-06-04 contracts layer, an operator with pre-2026-06-04 `player_projections_*.csv` files in `data/sim_results/` will see `ProjectionLoader.load_projections(...)` raise `ProjectionSchemaContractError` instead of loading with default zero values.
- After the Ticket 4 residual interval change on 2026-06-12, projection CSVs generated between 2026-06-04 and 2026-06-12 can also fail the validator because they have distribution columns and `DATA_QUALITY` but lack `{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, and `{STAT}_CONFIDENCE_SCORE`.
- `test_missing_tov_columns_loads_with_defaults` was renamed to `test_missing_tov_columns_fails_loudly` in the same change; the assertion is now that the typed exception is raised.

### Expected Behavior

- The projection CSV schema is strict: 6 stats x 14 columns (`{STAT}`, `{STAT}_P10`, `{STAT}_P50`, `{STAT}_P90`, `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`, `{STAT}_INTERVAL_80_LOW`, `{STAT}_INTERVAL_80_HIGH`, `{STAT}_INTERVAL_90_LOW`, `{STAT}_INTERVAL_90_HIGH`, `{STAT}_CONFIDENCE`, `{STAT}_CONFIDENCE_SCORE`) plus `DATA_QUALITY`. `validate_projection_frame(...)` enforces this on every load and on every write.
- Operators who upgrade must regenerate projection CSVs from the current `simulate_season.py`. There is no backwards-compatibility fallback for missing columns.

### Evidence

- `src/contracts/projections.py::validate_projection_frame` defines the strict required set.
- `src/query/projection_loader.py::ProjectionLoader.load_projections` calls `validate_projection_frame(...)` and re-raises the typed `ProjectionSchemaContractError`.
- `src/simulation/report_generator.py::ReportGenerator.export_player_projections` writes the strict schema and validates before saving. When `models/calibration/` is absent, interval bounds are blank and confidence labels are `NO_EDGE`; the columns still exist.
- `tests/test_query/test_six_stat_contract.py::TestMissingStatColumnsFailsLoudly::test_missing_tov_columns_fails_loudly` guards the load path.

### Suspected Cause

- The change is intentional (DR-031, 2026-06-04). The contracts layer is the seam; `ProjectionLoader` previously fell back to zero defaults for missing columns, which silently produced 0.0 over/under probabilities for STL/BLK/TOV. The strict schema is the corrected behavior.

### Workaround

- Re-run `python simulate_season.py --today` (or `--date YYYY-MM-DD`) after upgrading to regenerate projection CSVs in the strict schema. The new CSVs are loadable by the query layer and the new validator.
- To populate meaningful interval bounds instead of `NO_EDGE`/blank values, first run `python calibrate_residual_intervals.py --input data/evaluation/residual_training.parquet --output-dir models/calibration`.
- For bulk regeneration, run `python simulate_season.py --week` or `--season`.

### Fix Ideas

- No code fix needed — this is the intended new behavior. The fix is operator workflow: regenerate the CSVs.

### Risks

- Operators who depend on the old default-fallback behavior will see a hard failure on first load after upgrading. The fix is documented in `PROJECT_CONTEXT.md` (Journey 4) and surfaced in the `CURRENT_STATE.md` (Known Workarounds) section.

### Related Files

- `src/contracts/projections.py`
- `src/contracts/errors.py`
- `src/query/projection_loader.py`
- `src/simulation/report_generator.py`
- `src/correction/calibration.py`
- `src/correction/interval_store.py`
- `src/correction/confidence_scorer.py`
- `calibrate_residual_intervals.py`
- `tests/test_query/test_six_stat_contract.py`
- `tests/test_correction/test_calibration.py`
- `project-brain/PROJECT_CONTEXT.md` (Journey 4)
- `project-brain/CODE_RULES.md` (Strict projection schema)
- `project-brain/DECISIONS.md` (DR-031)

---

## KB-022: Training Pipeline Crashes When Saving Blend Weights

- Status: fixed in code on 2026-06-12
- Severity: critical
- Confidence: high

### Symptom

- Training appears to complete, but `_save_blend_weights()` raises `AttributeError: 'TrainingPipeline' object has no attribute 'targets'` and the legacy `blend_weights.pkl` is never written. `ModelManager` then fails the runtime artifact contract because blend weights are required.

### Expected Behavior

- Blend weights should be persisted to both the legacy `blend_weights.pkl` and the versioned `WeightStore` after training.

### Evidence

- `src/training/pipeline.py:998` iterated `for target in self.targets:` but the class only defines `self.TARGETS`.
- Fixed by changing the reference to `self.TARGETS`.
- Regression test `tests/test_training/test_runtime_artifact_contract.py::test_model_manager_rejects_missing_transformer_when_blend_weights_expect_it` now passes after also clearing the versioned WeightStore in the test fixture so the legacy edit is honored.

### Related Files

- `src/training/pipeline.py`
- `src/evaluation/weight_store.py`
- `tests/test_training/test_runtime_artifact_contract.py`

---

## KB-023: Transformer Runtime Inference Uses Categorical ID Columns

- Status: fixed in code on 2026-06-12
- Severity: critical
- Confidence: high

### Symptom

- `ModelManager._predict_transformer_target()` builds a feature sequence from `self.feature_cols`, which includes categorical ID columns (`PLAYER_ID`, `TEAM_ID`, `OPPONENT_ID`). The Transformer was trained only on numeric `nn_features = feature_cols - cat_features`, so inference raises a shape mismatch or produces garbage predictions.

### Expected Behavior

- Transformer inference should use the same numeric feature columns that were used during training.

### Evidence

- `src/models/model_manager.py:432-444` reindexed the history frame to `self.feature_cols`.
- Fixed by computing `nn_features = [c for c in self.feature_cols if c not in cat_features]` and aligning to that set. Defensive `getattr(self, "cat_features", [])` handles test doubles that do not set the attribute.
- `tests/test_models/test_model_manager.py::TestFallbackPredictor::test_transformer_prediction_uses_all_targets` covers the path.

### Related Files

- `src/models/model_manager.py`
- `src/training/pipeline.py`
- `src/models/transformer_model.py`
- `tests/test_models/test_model_manager.py`

---

## KB-024: Test Torch Shim Fails To Import On PyTorch-Less Machines

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- On machines without PyTorch, `src/__init__.py:_install_test_torch_shim()` crashes with `AttributeError: module 'importlib' has no attribute 'util'`, breaking any test that imports `torch`.

### Expected Behavior

- The shim should install cleanly when real PyTorch is unavailable.

### Evidence

- The function imported `importlib` but referenced `importlib.util.find_spec("torch")`.
- Fixed by importing `importlib.util` directly.

### Related Files

- `src/__init__.py`
- `tests/test_models/test_transformer_model.py`

---

## KB-025: Minutes Predictor Fatigue Feature Misaligned And Mostly NaN

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `MinutesPredictor._engineer_features()` computes `MINS_LAST_3` with a `groupby(...).rolling(...).sum()` that returns a Series with a MultiIndex. Assigning it directly to a DataFrame column produces misaligned/NaN values, so `FATIGUE_SCORE` is effectively zero for all rows.

### Expected Behavior

- `MINS_LAST_3` should reflect the rolling sum of the previous three games' minutes, aligned to each row.

### Evidence

- `src/models/minutes_predictor.py:172` assigned the MultiIndex result without resetting the group level.
- Fixed with `reset_index(level=0, drop=True)`.

### Related Files

- `src/models/minutes_predictor.py`

---

## KB-026: Defense Scraper Stores `pts_allowed_per_100` But Consumers Read `opp_pts_per_100`

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `NBADefenseScraper.get_team_defense_allowed()` returns a dict keyed by `pts_allowed_per_100` from the all-team fetch path, but `get_position_defense()` looks for `opp_pts_per_100`. The all-team path therefore always falls back to the league-average default of 114.0, neutralizing real defensive rating data.

### Expected Behavior

- Both keys should be available so downstream consumers can use whichever they expect.

### Evidence

- `src/data/nba_defense_scraper.py:172` stored `pts_allowed_per_100`; `src/data/nba_defense_scraper.py:687` read `opp_pts_per_100`.
- Fixed by normalizing the returned dict in `get_team_defense_allowed()`: if `pts_allowed_per_100` exists and `opp_pts_per_100` does not, the latter is added.

### Related Files

- `src/data/nba_defense_scraper.py`

---

## KB-027: Injury Scraper Does Not Add `TEAM_ABBR` To Disk-Cached Data

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `InjuryScraper.fetch_injuries()` adds `TEAM_ABBR` only on the live-fetch branch. When the disk cache is loaded, `TEAM_ABBR` is missing and downstream consumers such as `LineupScraper._get_inactive_players()` raise `KeyError`.

### Expected Behavior

- Cached injury data should have the same schema as live-fetched data.

### Evidence

- `src/data/injury_scraper.py:179` added `TEAM_ABBR` only in the live branch; `_load_from_cache()` did not.
- Fixed by deriving `TEAM_ABBR` from `TEAM` in `_load_from_cache()` and also setting `_cache_timestamp` so the in-memory cache is considered valid.

### Related Files

- `src/data/injury_scraper.py`
- `src/data/lineup_scraper.py`

---

## KB-028: Lineup Scraper Crashes When Injury Data Lacks `TEAM_ABBR`

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `LineupScraper._get_inactive_players()` filters injuries with `injuries_df['TEAM_ABBR'] == normalize_team(team_abbr)`. If the column is missing, it raises `KeyError`.

### Expected Behavior

- The method should gracefully derive `TEAM_ABBR` from `TEAM` if needed or return an empty list when neither column exists.

### Evidence

- `src/data/lineup_scraper.py:623` assumed `TEAM_ABBR` always exists.
- Fixed by adding a guard that creates `TEAM_ABBR` from `TEAM` when absent.

### Related Files

- `src/data/lineup_scraper.py`
- `src/data/injury_scraper.py`

---

## KB-029: Player Bio Enrichment Leaks Current Age Into Historical Rows

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `PlayerBioScraper._normalize_bio()` computes `AGE` from `pd.Timestamp.now()`. When `update_data.py` enriches historical game logs, every row receives the player's current age, leaking future information into training data.

### Expected Behavior

- Each historical game row should use the player's age on that game's date.

### Evidence

- `src/data/player_bio_scraper.py:115-117` used `pd.Timestamp.now()`.
- Fixed in `update_data.py::enrich_with_player_bios()` by excluding the cached `AGE` from the merge and recomputing it from `BIRTHDATE` and `GAME_DATE`.

### Related Files

- `src/data/player_bio_scraper.py`
- `update_data.py`

---

## KB-030: Basketball Reference Scraper Builds Wrong Season URLs

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `BasketballRefScraper` used `season.replace('-', '')` for URLs, producing strings like `202425` instead of the ending year `2025`. This causes 404s on both team pages and league ratings pages.

### Expected Behavior

- URLs should use the full ending year of the season.

### Evidence

- `src/data/basketball_ref_scraper.py:129` and `:356` used `season.replace('-', '')`.
- Fixed by adding `_season_end_year()` helper that returns the ending year (e.g. `2025` for `2024-25`).

### Related Files

- `src/data/basketball_ref_scraper.py`

---

## KB-031: Injury Risk Career Count Includes Future Injuries

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `InjuryRiskFeatureGroup` computes `INJURY_RISK_CAREER_COUNT` as the total injury count per player from `injury_history.csv`, shifted by one row. Because the count is not filtered by `DATE < GAME_DATE`, past games see injuries that occurred in the future.

### Expected Behavior

- The career count for each game should include only injuries that occurred before that game.

### Evidence

- `src/preprocessing/features/injury_risk.py:101-118` aggregated all injuries per player.
- Fixed by merging history dates with game dates and counting only `DATE < GAME_DATE`.

### Related Files

- `src/preprocessing/features/injury_risk.py`
- `tests/test_preprocessing/test_injury_risk_features.py`

---

## KB-032: Skill Development Features Leak Future In-Season Games

- Status: fixed in code on 2026-06-12
- Severity: high
- Confidence: high

### Symptom

- `SkillDevelopmentFeatureGroup` computes per-player per-season averages over the entire season, merges them back onto every row, then shifts by one. For a game in the middle of the season, the "current season average" includes games that occur after that row, leaking future information.

### Expected Behavior

- The current-season component of YoY velocity should use only games that occurred before the current row.

### Evidence

- `src/preprocessing/features/skill_development.py:107-126` used full-season aggregates.
- Fixed by replacing full-season current-season averages with expanding season-to-date averages shifted by 1, while still using the full prior season as the reference.

### Related Files

- `src/preprocessing/features/skill_development.py`
- `tests/test_preprocessing/test_skill_development_features.py`
