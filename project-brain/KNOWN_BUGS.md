# Known Bugs

This file tracks confirmed or strongly suspected defects and weak points visible in the repository as of 2026-04-01.

---

## KB-001: Training Does Not Persist Required CatBoost Runtime Artifacts

- Status: fixed in code on 2026-04-01, pending live CLI confirmation
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
- Live CLI confirmation remains pending in this workspace because:
  - there are no checked-in raw training CSVs
  - the usable local interpreter crashes on `torch` import in this sandbox

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

- Status: open
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

- Prefer querying `pts`, `reb`, and `ast` from cached projection CSVs.

### Fix Ideas

- Extend projection export to all six targets.
- Align loader defaults and add tests for each supported stat.

### Risks

- Users may trust incomplete cached projections for unsupported exported stats.

### Related Files

- `src/simulation/report_generator.py`
- `src/query/projection_loader.py`
- `src/query/interactive_cli.py`
- `query_prob.py`

---

## KB-006: `simulate_season.py --season` Does Not Match Its CLI Promise

- Status: open
- Severity: medium
- Confidence: high

### Symptom

- The CLI suggests simulation of all remaining season games, but implementation appears to fetch only about the next 30 days.

### Expected Behavior

- `--season` should either simulate the full remaining season or be renamed/documented as a limited horizon.

### Evidence

- `ScheduleScraper.get_remaining_season()` appears to iterate over a fixed next-30-day window rather than a full remaining schedule.

### Reproduction

- Static inspection during this audit.

### Suspected Cause

- Placeholder implementation was never upgraded to true season scope.

### Workaround

- Use `--week` or date-scoped runs when the intent is near-term forecasting.

### Fix Ideas

- Pull the full remaining schedule from a season endpoint or iterate to the actual season end date.
- Update CLI help text if full-season support is intentionally deferred.

### Risks

- Misleading user expectations and incomplete simulation coverage.

### Related Files

- `src/data/schedule_scraper.py`
- `simulate_season.py`

---

## KB-007: `ProjectionLoader` Hardcodes A Defense Cache Filename For Season `2025-26`

- Status: open
- Severity: medium
- Confidence: medium-high

### Symptom

- Defense-context loading will go stale or fail after season rollover if the cache filename changes.

### Expected Behavior

- Defense cache lookup should derive the active season dynamically or handle multiple candidate files.

### Evidence

- `src/query/projection_loader.py` references `cache/all_team_defense_2025-26.json`.

### Reproduction

- Static inspection only.

### Suspected Cause

- Temporary season-specific implementation hardened into production code.

### Workaround

- Manually rename or copy cache files to the expected name if necessary.

### Fix Ideas

- Compute season names dynamically from date/config.
- Search for the newest matching defense cache file instead of hardcoding one season.

### Risks

- Query-time contextual adjustments become stale or unavailable across seasons.

### Related Files

- `src/query/projection_loader.py`

---

## KB-008: `GameSimulator` Contains Large Unreachable Legacy Logic After An Early Return

- Status: open
- Severity: medium
- Confidence: high

### Symptom

- The file contains a large simulation block that appears dead, making it unclear which algorithm is truly active.

### Expected Behavior

- There should be one clearly documented active simulation path, with obsolete logic removed or isolated.

### Evidence

- `simulate_matchup(...)` returns `self._simulate_matchup_reactive(...)` and then retains a long block of older simulation code below that return.

### Reproduction

- Static inspection during this audit.

### Suspected Cause

- Partial migration from an older vectorized/legacy implementation to a newer reactive path.

### Workaround

- Treat `_simulate_matchup_reactive` as the current source of truth until the dead block is removed.

### Fix Ideas

- Delete or quarantine the unreachable block.
- Add tests that make the active path explicit.

### Risks

- Future contributors may patch dead code and believe they changed live behavior.

### Related Files

- `src/simulation/game_simulator.py`

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

- Status: fixed in code on 2026-04-02, pending live CUDA smoke confirmation
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

- Status: fixed in code on 2026-04-02, pending live Colab confirmation
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
