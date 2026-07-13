# Repository Improvement Execution Plan

**Project:** NBA Player Stats Prediction System  
**Audit date:** 2026-07-13  
**Audience:** An implementation engineer who should be able to execute the work in order without inventing architecture or acceptance criteria  
**Status:** Ready for implementation planning; no production code was changed during this audit  
**Companion design:** [`plans/2026_27_architecture_rebuild.md`](2026_27_architecture_rebuild.md)

---

## 1. Mission

Convert the current project into a point-in-time-correct, reproducible, auditable NBA forecasting pipeline while preserving its local, pure-Python CLI design.

The work is complete only when:

1. No current-game or post-cutoff value can enter a training or inference feature schema.
2. Feature selection, early stopping, blending, calibration, and residual correction cannot see the outer test window.
3. The final test score and historical replay use the same serialized artifacts and runtime forecast path used by simulation.
4. Every forecast identifies its data snapshot, forecast cutoff, horizon, model bundle, schema, and scenario.
5. Model bundles are immutable, checksum-verified, and promoted atomically.
6. Simulation preserves declared basketball invariants and does not silently apply the same context twice.
7. Data collection is atomic, versioned, and honest about missing optional sources.
8. The test suite, CI, dependencies, configuration, documentation, and operational commands describe the same system.

This plan intentionally does **not** add a web server, database, Docker deployment, distributed training system, or public API.

---

## 2. Audited baseline and evidence

Treat these facts as the starting state. Re-run the baseline commands in Section 5 before editing code because the working tree is active and may change.

### 2.1 Repository state

- The worktree is heavily modified. The audit saw 94 tracked files changed or deleted plus many untracked files.
- The active changes include forecast contracts, a forecast service, bundle versioning, prediction ledgering, continual-learning gates, residual monitoring, feature extensions, and reasoning support.
- Existing work belongs to the repository owner. Do not reset, discard, overwrite, or silently re-create it.
- Python source and tests total approximately 53,602 lines.
- The largest production modules are:
  - `src/data/nba_defense_scraper.py`: 1,790 lines.
  - `src/training/pipeline.py`: 1,375 lines.
  - `src/simulation/game_simulator.py`: 1,239 lines.
  - `src/query/interactive_cli.py`: 1,130 lines.
  - `src/query/probability_calculator.py`: 967 lines.
  - `src/models/model_manager.py`: 934 lines.
  - `src/models/gpu_utils.py`: 928 lines.

### 2.2 Verification baseline

The following checks passed during the audit:

```text
python -m compileall -q src tests *.py     PASS
git diff --check                           PASS
python check_contracts.py --models-dir models
                                             PASS
pytest -m "not slow" -q                    476 passed, 7 deselected
```

The non-slow suite took 152.04 seconds and emitted 305 warnings:

- 304 warnings were scikit-learn feature-name warnings from smart feature selection.
- 1 warning was joblib CPU-core detection.

There are 483 collected tests. The green suite is useful, but it does not prove point-in-time correctness or production/replay parity.

### 2.3 Active data and artifact state

- `data/nba_players.csv`: 52,583 rows, 71 columns, 686 players, 2,442 game IDs, dates 2024-10-22 through 2026-04-10.
- `data/nba_games.csv`: 4,882 rows, 57 columns, 2,441 game IDs, dates 2024-10-22 through 2026-04-10.
- The one-game mismatch between player and team game IDs is unresolved.
- The following optional data files are absent:
  - `data/player_bios.csv`
  - `data/injury_history.csv`
  - `data/advanced_tracking.csv`
- The active model bundle is a legacy flat `small` preset with CatBoost only.
- `models/champion.json` is absent.
- `models/bundle_manifest.json` is absent.
- The active feature schema has 309 features and still contains these current-game team outcomes:
  - `PTS_TEAM`
  - `REB_TEAM`
  - `AST_TEAM`
  - `FGA_TEAM`
  - `FTA_TEAM`
  - `OREB_TEAM`
  - `DREB_TEAM`
  - `TOV_TEAM`
- `check_contracts.py` passes despite those unsafe columns. The current contract is therefore a file-presence check, not a sufficient semantic safety gate.

### 2.4 Confirmed integration and correctness gaps

The executor must treat these as defects, not optional refactors:

1. **Unsafe active artifacts:** the checked-in runtime schema contains same-game team outcomes.
2. **Smart feature selection is not reliably applied:** `train.py` calls `TrainingPipeline.apply_feature_selection_manifest()` before `prepare_data()` establishes `pipeline.feature_cols`. The method filters manifest columns against an empty master schema, so per-target selections can be discarded.
3. **Smart-selection candidates bypass the canonical safety filter:** `train.py` builds `master_feature_cols` by excluding targets and a few identifiers rather than taking `FeatureSelector.fit(...).feature_cols`. Current-game `MIN`, shooting, and other raw outcomes can reach selection logic.
4. **The target-specific whitelist is not fully safe:** `FeatureSelector.select_features_for_target()` checks a few exclusion sets but does not call `_is_safe_feature()` for every whitelisted feature.
5. **Outer-test scoring is not production scoring:** `TrainingPipeline.evaluate_test_window()` calls each primary CatBoost model directly. It does not score the serialized production blend, Transformer path, quantiles, residual layer, or calibration path.
6. **The recorded cutoff is incomplete:** `_training_data_cutoff` is the maximum date in `fit_df`; validation data still influences early stopping and blending. The backtest can therefore accept dates already used by model selection.
7. **The existing backtest is row scoring, not historical replay:** it recomputes a current feature table, passes `history_df=None`, and has no source snapshot or forecast horizon.
8. **The scheduled-game row is stale:** `GameSimulator._build_scheduled_game_row()` copies the latest already-engineered row and overwrites identifiers. Shifted rolling features are not recomputed after appending the latest completed game, so next-game context can omit that game.
9. **Canonical forecast types are not on the main path:** `ForecastRequest`, `FeatureMaterializer`, and `ForecastService.predict_forecast_frame()` exist, but `simulate_season.py`, training, and replay do not use them end to end.
10. **The module-level materializer disables strict point-in-time checks:** `_DEFAULT_MATERIALIZER` is created with `strict_point_in_time=False`.
11. **Several `FeatureSpec` fields are declarative only:** entity keys, required sources, maximum lookback, and declared dtype are not all enforced.
12. **Ledger identity is incompatible with multiple horizons:** `PredictionLedger.KEY_COLUMNS` omits snapshot, horizon, scenario, request ID, and forecast cutoff. Morning and pregame forecasts can collapse into one row.
13. **Ledger output is not wired into the normal simulation CLI:** `ReportGenerator.record_prediction_ledger()` exists, but `simulate_season.py` does not call it.
14. **Canonical forecast output and ledger naming disagree:** forecasts emit `MODEL_BUNDLE_ID`; the ledger requires `MODEL_VERSION`.
15. **Promotion does not use faithful paired replay:** `improve_models.py` reads/reconciles the ledger for counts but evaluates candidates using the current `BacktestRunner`.
16. **Bootstrap units are mislabeled:** promotion fields named `*_pct` are produced by multiplying raw absolute-error deltas by 100. Rows are also bootstrapped independently even though players in the same game are correlated.
17. **Bundle writing is unsafe in the flat directory:** once `bundle_manifest.json` exists, retraining different files in the same directory makes `ModelBundleManifest.write()` fail. The exception is logged and a stale manifest can remain.
18. **Champion resolution can fail open:** `ModelVersionRegistry.active_dir()` silently falls back to flat root artifacts when a champion path is missing.
19. **Configuration is fragmented:** `Config` silently ignores top-level YAML sections such as `residual_correction` and `residual_monitoring`; some scripts parse YAML directly instead.
20. **Extension configuration is not loaded by the normal updater:** `_run_configured_extensions()` calls `get_config()` without loading `config/default.yaml`, so configured `data_sources` can remain empty.
21. **Core data writes are non-atomic:** CSV is written before Parquet, without a shared manifest or rollback. A crash can leave formats inconsistent.
22. **Parquet is a runtime feature but `pyarrow` is not a declared direct dependency.** It happens to be installed in the current venv.
23. **Query-time double adjustment:** `InteractiveCLI._handle_over_under()` modifies an exported model/simulation mean again using recent form, minutes, defense, trend, matchup, and venue heuristics.
24. **Simulation has an asymmetric-team context defect:** `_run_simulation()` derives `off_env` for both teams from the first key in `team_targets`.
25. **Overtime is simulated incorrectly:** overtime is included in `PhaseSimulator.phase_definitions()` and may be simulated again in `GameSimulator._run_simulation()`. It is triggered for margins within a configurable threshold rather than only for ties.
26. **Simulation normalization is inconsistent:** the post-sampling normalization explicitly covers PTS/REB/AST but not STL/BLK/TOV.
27. **Code-quality automation is absent:** no CI workflow, formatter, linter, or centralized project configuration is present.

---

## 3. Priority order

Do not reorder these priorities.

| Priority | Outcome | Why it comes now |
|---|---|---|
| P0 | Artifact and feature leakage quarantine | Existing metrics and models are unsafe until this passes. |
| P0 | Correct split, selection, cutoff, and final evaluation wiring | Every later experiment depends on a trustworthy score. |
| P0 | Scheduled-row, overtime, and asymmetric-team correctness | These are direct runtime correctness defects. |
| P0 | Immutable candidate bundles and strict loading | Prevents stale or mixed artifacts from being treated as a model. |
| P1 | Canonical point-in-time forecast path and replay | Establishes training/live/evaluation parity. |
| P1 | Horizon-aware ledger and promotion gates | Makes continual learning auditable and statistically valid. |
| P1 | Atomic data snapshots and unified configuration | Supplies reproducible inputs and makes configuration effective. |
| P1 | Availability and minutes as first-class models | Opportunity must be resolved before player totals. |
| P2 | Conditional stat rates and calibration | Improves model structure after the baseline is trustworthy. |
| P2 | Simulation decomposition and heuristic ablation | Simplifies a monolith after its behavior is protected by tests. |
| P2 | Query simplification, tooling, CI, dependency hygiene | Makes the corrected system maintainable. |

---

## 4. Execution rules

These rules apply to every pull request in this plan.

1. Activate the repository venv before every Python or test command:

   ```bash
   source venv/bin/activate
   ```

2. Do not run `git reset --hard`, `git checkout --`, or any command that discards existing changes.
3. Do not implement on top of an unidentified dirty baseline. The repository owner must first identify a commit containing all intended in-progress changes. If no such commit exists, stop and ask the owner to create one.
4. Create one branch per pull request using the `codex/` prefix or the owner's requested equivalent.
5. Keep compatibility wrappers until parity tests prove the replacement.
6. Do not promote a model because a command completed. Promotion requires the gates defined in PR 08.
7. Do not use the current legacy model metrics as a comparison baseline after PR 01. Label them `pre_clean_baseline`.
8. Do not add paid data, a database, a web service, Docker, or distributed infrastructure.
9. Do not make live NBA, ESPN, Rotowire, Action Network, or Basketball Reference requests in unit tests or CI. Use recorded fixtures.
10. A pull request may proceed only when its listed tests and acceptance gate pass.
11. If a new test exposes an unrelated pre-existing bug, record a separate issue and either fix it in a prerequisite PR or keep the current PR blocked. Do not silently weaken, skip, or `xfail` the assertion.
12. Every new artifact format must include a `schema_version` and have a migration or explicit incompatibility error.
13. Every fallback must be visible in structured metadata. Evaluation and promotion run in strict mode and may not use fallbacks.
14. Update `README.md`, `AGENTS.md`, `TASKS.md`, and `DECISIONS.md` in the same PR that changes an operator-facing command or architectural decision.

---

## 5. PR 00 — Establish a preserved, reproducible baseline

### Objective

Create a safe starting point without modifying model behavior.

### Required files

- Add the tracked baseline report at `plans/audit_baseline_2026_07_13.md`.
- Update no production module in this PR.

### Steps

1. Ask the repository owner for the baseline commit containing the intended current worktree changes.
2. Confirm the worktree for implementation is clean:

   ```bash
   source venv/bin/activate
   git status --short
   ```

   Expected: no output. If there is output, stop. Do not clean it automatically.

3. Create a branch:

   ```bash
   git switch -c codex/baseline-audit
   ```

4. Record:
   - baseline commit SHA;
   - Python version;
   - installed direct dependency versions;
   - test counts and duration;
   - data file row counts, date ranges, hashes, and game counts;
   - model file list and hashes;
   - feature schema count and forbidden feature matches;
   - whether champion and bundle manifests exist.
5. Add a small read-only root entry point named `audit_project.py`. The script must:
   - never load network resources;
   - never mutate data or models;
   - emit JSON to stdout;
   - accept `--data-dir` and `--models-dir`;
   - return exit code 0 even when it reports unsafe legacy artifacts;
   - return nonzero only for unreadable inputs or internal errors.
6. Add `tests/test_audit_project.py` using temporary CSV and artifact fixtures.
7. Run the baseline commands below and attach output to the PR.

### Verification

```bash
source venv/bin/activate
python -m compileall -q src tests *.py
git diff --check
pytest -m "not slow" -q
pytest -m "slow or gpu or integration" -q
python check_contracts.py --models-dir models
python audit_project.py --data-dir data --models-dir models
```

### Acceptance gate

- The report reproduces the unsafe eight `*_TEAM` features.
- The report identifies the missing optional data files and missing manifests.
- The report includes the baseline commit and dependency versions.
- No runtime behavior or artifact is changed.

### Rollback

Delete only the newly added audit script, its test, and its report. Do not touch data or models.

---

## 6. PR 01 — Make semantic feature safety a hard contract

### Objective

Make it impossible to train, load, validate, or promote a schema containing current-game outcomes or undeclared features.

### Files to change

- `src/contracts/features.py`
- `src/contracts/artifacts.py`
- `src/utils/prediction_utils.py`
- `src/preprocessing/data_loader.py`
- `src/evaluation/smart_feature_selector.py`
- `check_contracts.py`
- `tests/test_contracts/test_feature_safety.py` (new)
- `tests/test_preprocessing/test_leak_quarantine.py`
- `tests/test_preprocessing/test_feature_engineer.py`
- `tests/test_evaluation/test_smart_feature_selector.py`

### Steps

1. In `src/contracts/features.py`, define one canonical policy:
   - `TARGET_COLUMNS` for PTS/REB/AST/STL/BLK/TOV;
   - `RAW_CURRENT_GAME_COLUMNS` for MIN, shooting, offensive rebound, and other current-game outcomes;
   - `CURRENT_GAME_TEAM_OUTCOMES` for every same-game team outcome suffix;
   - `FORBIDDEN_EXACT_COLUMNS` as the union;
   - `FORBIDDEN_PATTERNS` for future same-game team outcome variants;
   - `is_forbidden_feature(name: str) -> bool`;
   - `validate_feature_names(names: Sequence[str]) -> None`.
2. Keep identifiers such as `PLAYER_ID`, `TEAM_ID`, and `OPPONENT_ID` separate from forbidden outcomes. They may be metadata/categorical inputs only when the schema explicitly allows them.
3. In `DataLoader.merge_datasets()`, stop merging raw same-game team totals into the player training table. Merge only:
   - opponent identity;
   - team/opponent rolling values that were shifted before the current game;
   - pregame metadata.
4. If a legacy caller requires team outcomes for reporting, return them from a separate method or a separately named result object. Do not retain them in the feature-engineering input frame.
5. Refactor `FeatureSelector` to call `is_forbidden_feature()` before any exact, prefix, keyword, registry, or target-encoding allow rule.
6. Change `FeatureSelector.select_features_for_target()` so every allow-listed column must satisfy `_is_safe_feature()`. Do not “trust” a manifest merely because it is a manifest.
7. Change smart-selection input construction to receive only a canonical safe candidate list. Add an assertion inside `SmartFeatureSelector.run()` that validates every input feature name.
8. Update `validate_runtime_artifacts()` to load both `feature_schema.pkl` and `feature_cols.pkl`, require them to describe the same ordered feature list, and reject forbidden features.
9. Update `check_contracts.py` so the current `models/` directory fails with an actionable message listing all eight leaking columns.
10. Add a schema-version bump from `feature_schema_v3` to `feature_schema_v4`. Reject older schemas in strict runtime mode. Add `--allow-legacy-artifacts` immediately to model-consuming CLIs so operators can deliberately run the quarantined bundle during migration; print an `UNSAFE LEGACY ARTIFACT MODE` warning and disable replay/promotion in that mode. PR 04 will centralize this behavior in the bundle loader.
11. Add future-mutation tests:
    - mutating another player's same-game outcome does not change the row;
    - mutating either team's same-game outcome does not change the row;
    - mutating a later game does not change an earlier feature row;
    - a malicious extension prefix cannot re-allow a forbidden exact column.
12. Add a regression test that loads the currently known eight names and asserts rejection.

### Verification

```bash
source venv/bin/activate
pytest tests/test_contracts/test_feature_safety.py -v
pytest tests/test_preprocessing/test_leak_quarantine.py -v
pytest tests/test_preprocessing/test_feature_engineer.py -v
pytest tests/test_evaluation/test_smart_feature_selector.py -v
python check_contracts.py --models-dir models
```

Expected `check_contracts.py` result before retraining: nonzero with the eight unsafe columns listed. That failure is the desired quarantine behavior.

### Acceptance gate

- No same-game player or team outcome can pass `FeatureSelector`, a registry allow rule, or a feature-selection manifest.
- `DataLoader.merge_datasets()` no longer creates the eight unsafe `*_TEAM` columns.
- Strict artifact validation rejects the current legacy bundle.
- Legacy execution remains available only through an explicit unsafe-mode flag and can never feed promotion.
- All feature mutation tests pass.

### Rollback

Revert this PR as one unit. Do not selectively restore the permissive contract while leaving the data merge changed.

---

## 7. PR 02 — Correct split and smart-selection execution order

### Objective

Guarantee that smart selection sees only allowed training history and that the selected columns actually reach each target model.

### Files to change

- `train.py`
- `src/training/pipeline.py`
- `src/evaluation/smart_feature_selector.py`
- `src/utils/prediction_utils.py`
- `tests/test_training/test_smart_selection_integration.py` (new)
- `tests/test_evaluation/test_smart_feature_selector.py`
- `tests/test_training/test_training_presets.py`
- `tests/test_training/test_train_entrypoint.py`

### Steps

1. Move pipeline creation and `prepare_data(full_df)` before smart feature selection.
2. Make `prepare_data()` return a named dataclass such as `TemporalSplit` containing:
   - `fit_df`;
   - `validation_df`;
   - `outer_test_df`;
   - `fit_end`;
   - `validation_start` and `validation_end`;
   - `outer_test_start` and `outer_test_end`;
   - `split_policy` and any fallback reason.
3. Keep a tuple compatibility method for older tests/callers during this PR.
4. Build the canonical safe master schema from `fit_df` only:

   ```python
   master_schema = pipeline.feature_selector.fit(split.fit_df)
   ```

5. Pass only `master_schema.feature_cols` into `SmartFeatureSelector`.
6. Run selection on training history only. The selector may create an internal chronological training/validation split inside `fit_df`, or use `fit_df` plus the declared inner validation window. It may not use `outer_test_df`.
7. Apply the selection manifest **after** `pipeline.feature_cols` exists.
8. Change `apply_feature_selection_manifest()` to:
   - validate the manifest schema version;
   - validate the manifest input schema hash against the master schema;
   - validate every selected feature through the semantic safety contract;
   - reject unknown target names;
   - reject an empty list for any required target;
   - retain a deterministic master-column order.
9. Add manifest metadata:
   - `schema_version`;
   - `input_schema_hash`;
   - `selection_data_start` and `selection_data_end`;
   - `outer_test_start` for audit only;
   - profile/config hash;
   - random seed;
   - code version.
10. Change failure behavior. If smart selection was explicitly requested, a selection error must fail training. If it was enabled only by a preset default, allow fallback only when `--allow-selection-fallback` is explicitly passed.
11. After training each target, compare the model's serialized feature names to the manifest list. Fail before artifact finalization on any mismatch.
12. Add an end-to-end test that executes this exact order:
   - construct fit/validation/test frames with a unique “test-only” signal;
   - run smart selection;
   - prove the test-only column/value cannot influence the manifest;
   - prove `pipeline.target_feature_cols` is populated;
   - prove each trainer receives the expected per-target columns.
13. Add a regression test that calls `apply_feature_selection_manifest()` before schema creation and requires an explicit error instead of silent filtering.

### Verification

```bash
source venv/bin/activate
pytest tests/test_training/test_smart_selection_integration.py -v
pytest tests/test_evaluation/test_smart_feature_selector.py -v
pytest tests/test_training/test_training_presets.py -v
pytest tests/test_training/test_train_entrypoint.py -v
```

### Acceptance gate

- The outer test partition is created before selection and is never passed to selection.
- Explicit smart selection cannot silently fall back.
- Every required target has a non-empty selected list.
- Serialized model feature names exactly equal the manifest lists.
- The regression reproducing the previous empty-schema ordering now fails loudly.

---

## 8. PR 03 — Score the exact runtime bundle and record complete cutoffs

### Objective

Replace primary-model-only test scoring with strict scoring through the exact serialized production path.

### Files to change

- `src/training/pipeline.py`
- `src/pipeline/forecast_service.py`
- `src/models/model_manager.py`
- `src/evaluation/metrics.py`
- `src/contracts/artifacts.py`
- `train.py`
- `tests/test_training/test_outer_test_runtime_parity.py` (new)
- `tests/test_training/test_runtime_artifact_contract.py`
- `tests/test_models/test_model_manager.py`

### Steps

1. Replace the single `_training_data_cutoff` field with explicit cutoff metadata:
   - `fit_data_cutoff`;
   - `selection_data_cutoff`;
   - `validation_data_cutoff`;
   - `blend_data_cutoff`;
   - `calibration_data_cutoff`;
   - `residual_data_cutoff` when applicable;
   - `knowledge_cutoff = max(all influencing cutoffs)`.
2. Make `knowledge_cutoff` the cutoff used by replay-overlap validation and champion manifests.
3. Define an artifact-finalization sequence:
   1. train models on fit history;
   2. choose stopping/blending/calibration on inner history;
   3. save candidate component files to a temporary candidate directory;
   4. load those files through `ModelManager` and `ForecastService` in strict mode;
   5. score the outer test;
   6. write scorecard and metadata;
   7. write the immutable bundle manifest last.
4. Add `strict=True` to evaluation APIs. In strict mode:
   - no historical-average fallback;
   - no skipped prediction rows;
   - no missing target model;
   - no missing quantile/calibration artifact when declared;
   - no partial target scorecard.
5. Implement a vectorized or batched `ForecastService` path that uses the same blending, quantile, correction, and interval logic as single-row inference.
6. Make outer-test scoring call that service after reloading the candidate from disk. Do not call `model.predict()` directly from the training pipeline.
7. Emit `outer_test_scorecard.json` with:
   - bundle ID placeholder or candidate run ID;
   - schema hash and config hash;
   - all cutoffs;
   - exact window and row/game/player counts;
   - prediction/fallback/error counts;
   - MAE, RMSE, bias, R² by target;
   - P10/P50/P90 pinball loss;
   - 50/80/90 interval coverage and sharpness;
   - simple rolling-average and per-minute baseline metrics;
   - slice metrics for starter/bench, minutes bands, injury status when data exists;
   - model-only metrics separated from market-assisted metrics.
8. Add `evaluation_status` values `strict_complete` and `invalid`. Never write `strict_complete` if any target or row failed.
9. Make `validate_runtime_artifacts()` require a complete outer scorecard for promotable bundles.
10. Add parity tests using fake CatBoost/Transformer/residual components with deliberately different outputs. Assert the scorecard equals `ForecastService` output, not the primary model output.
11. Add a cutoff test where validation ends after fit. Assert a replay starting between those dates is rejected.
12. Add a serialized reload test so in-memory and reloaded predictions match within declared tolerance.

### Verification

```bash
source venv/bin/activate
pytest tests/test_training/test_outer_test_runtime_parity.py -v
pytest tests/test_training/test_runtime_artifact_contract.py -v
pytest tests/test_models/test_model_manager.py -v
pytest tests/test_pipeline/test_forecast_service.py -v
```

### Acceptance gate

- The outer test is scored only after loading serialized artifacts.
- The scorecard represents the same forecast path as runtime.
- `knowledge_cutoff` includes validation, selection, blend, calibration, and residual influence.
- A partial or fallback-filled scorecard cannot be marked complete or promoted.

---

## 9. PR 04 — Make model bundles immutable and runtime loading fail closed

### Objective

Eliminate mixed flat artifacts, stale manifests, and silent champion fallback.

### Files to change

- `src/models/versioning.py`
- `src/models/model_manager.py`
- `src/contracts/artifacts.py`
- `train.py`
- `check_contracts.py`
- `improve_models.py`
- `clear_cache.py`
- `tests/test_models/test_bundle_lifecycle.py` (new)
- `tests/test_models/test_bundle_manifest.py`
- `tests/test_models/test_versioning.py`

### Target layout

```text
models/
  champion.json
  versions/
    <run_id>/
      bundle_manifest.json
      model_stack_metadata.json
      feature_schema.json
      feature_schema.pkl
      outer_test_scorecard.json
      dependencies.json
      config.resolved.yaml
      <model components>
```

Legacy flat artifacts may remain temporarily but are not valid in strict mode.

### Steps

1. Make `train.py` write to `models/versions/<run_id>.tmp/`, never directly to the active root.
2. Choose `run_id` deterministically as UTC timestamp plus short commit SHA. If the tree is dirty, append `-dirty` and record a diff hash.
3. Write every component into the temporary directory.
4. Generate JSON metadata in addition to any required pickle/joblib runtime objects. Operators must not need unsafe deserialization to inspect a bundle.
5. Record:
   - Python version;
   - platform;
   - direct dependency versions;
   - Git commit and dirty status;
   - resolved config;
   - schema version/hash;
   - source snapshot IDs/hashes;
   - all cutoffs;
   - target feature lists;
   - blend/calibration/residual component versions;
   - test scorecard hash.
6. Write `bundle_manifest.json` last, fsync it, validate every hash, then atomically rename the temporary directory to `models/versions/<run_id>/`.
7. Make `ModelVersionRegistry.promote()` require a valid bundle manifest, complete strict scorecard, and all required component hashes.
8. Make `active_dir(strict=True)` raise an actionable error if:
   - `champion.json` is missing;
   - the target directory is missing;
   - a component hash is wrong;
   - the manifest/schema/config versions are incompatible.
9. Keep legacy fallback only behind `--allow-legacy-artifacts`. Print a prominent `UNSAFE LEGACY ARTIFACT MODE` warning and disable promotion in this mode.
10. Validate checksums before loading pickle/joblib/CatBoost artifacts.
11. Make promotion an atomic pointer replacement. Fsync the temporary pointer before `os.replace()`.
12. Make rollback validate the destination bundle before changing the pointer.
13. Update `clear_cache.py --keep-models` to preserve champion and all versions. Add an explicit `--prune-model-versions --keep-last N` command with dry-run and confirmation; do not overload `--all`.
14. Add lifecycle tests for:
    - interrupted temporary bundle;
    - stale manifest;
    - changed component hash;
    - missing champion path;
    - incompatible schema version;
    - successful promotion;
    - rollback to a valid prior version;
    - refusal to rollback to a corrupt version.

### Verification

```bash
source venv/bin/activate
pytest tests/test_models/test_bundle_lifecycle.py -v
pytest tests/test_models/test_bundle_manifest.py -v
pytest tests/test_models/test_versioning.py -v
pytest tests/test_training/test_runtime_artifact_contract.py -v
python clear_cache.py --all --keep-models --dry-run
```

### Acceptance gate

- Training cannot overwrite an existing immutable bundle.
- Runtime never silently substitutes flat root artifacts for a broken champion.
- Checksums are verified before deserialization.
- Promotion and rollback are atomic and validation-gated.

---

## 10. PR 05 — Fix scheduled-game materialization and wire the canonical forecast path

### Objective

Make training, live prediction, simulation, and later replay build the same next-game feature row under a strict point-in-time request.

### Files to change

- `src/contracts/forecast.py`
- `src/features/materializer.py`
- `src/preprocessing/feature_engineer.py`
- `src/preprocessing/feature_engineer_gpu.py`
- `src/pipeline/forecast_service.py`
- `src/simulation/game_simulator.py`
- `src/simulation/season_simulator.py`
- `simulate_season.py`
- `tests/test_contracts/test_forecast_contract.py`
- `tests/test_pipeline/test_forecast_materialization_parity.py` (new)
- `tests/test_simulation/test_game_simulator.py`

### Steps

1. Extend `ForecastRequest` with explicit schema-versioned identity:
   - request ID;
   - game ID/date;
   - home/away team IDs;
   - forecast cutoff;
   - horizon;
   - source snapshot ID;
   - model bundle ID;
   - scenario;
   - optional operator/run ID.
2. Require timezone-aware cutoffs. Normalize storage to UTC while retaining the game date in the NBA schedule timezone.
3. Remove `strict_point_in_time=False` from the module-level default materializer. Strict is the default everywhere; tests must opt out explicitly if needed.
4. Enforce every `FeatureSpec` field:
   - required source must be present;
   - entity keys must exist and be non-null;
   - event and availability timestamps must parse;
   - availability must be at or before forecast cutoff;
   - event time must be before the forecasted game event;
   - maximum lookback must be applied per feature, not cumulatively across unrelated specs;
   - horizon must be allowed;
   - output must match declared dtype and row cardinality;
   - missing policy must be one of a closed enum.
5. Do not filter one shared frame cumulatively for different feature specs. Build each feature from its declared source/time window and join by entity keys.
6. Replace `GameSimulator._build_scheduled_game_row()` behavior:
   - start from raw completed player history, not the latest engineered row;
   - append a synthetic row with no outcomes and the scheduled identifiers/context;
   - run the same feature groups used by training;
   - select only the synthetic row;
   - validate it against the active bundle schema.
7. Ensure the latest completed game is included in all shifted rolling features for the scheduled row.
8. Make `ForecastService` accept a `ForecastRequest` plus a materialized `ForecastFrame`. Keep the legacy DataFrame-only method as a compatibility wrapper that emits a deprecation warning.
9. Update `simulate_season.py` to create one request per game and pass it through season/game simulation.
10. Include request ID, bundle ID, snapshot ID, cutoff, horizon, and scenario in simulation metadata and report exports.
11. Add parity fixtures:
    - materialize a historical game as if it were upcoming;
    - compare training materialization and forecast materialization for the same cutoff;
    - assert exact column order and values within numeric tolerance;
    - mutate all later rows and assert the forecast frame is unchanged;
    - append the immediately prior game and assert the next rolling feature changes.
12. Add CPU/GPU feature parity for the canonical fixture. GPU-only execution may be marker-gated, but CPU fallback semantics must match.

### Verification

```bash
source venv/bin/activate
pytest tests/test_contracts/test_forecast_contract.py -v
pytest tests/test_pipeline/test_forecast_materialization_parity.py -v
pytest tests/test_simulation/test_game_simulator.py -v
pytest tests/test_preprocessing/test_feature_engineer_gpu.py -v
```

### Acceptance gate

- The scheduled row includes the most recently completed game in shifted history.
- Future data mutations cannot affect a prior request.
- Training and forecast materialization use the same feature definitions and column order.
- Every simulation result contains full request identity.

---

## 11. PR 06 — Correct immediate simulation defects before deeper refactoring

### Objective

Fix proven game-simulation logic errors behind regression tests without redesigning the full simulator yet.

### Files to change

- `src/simulation/phase_simulator.py`
- `src/simulation/game_simulator.py`
- `src/simulation/sim_types.py`
- `config/default.yaml`
- `tests/test_simulation/test_overtime_rules.py` (new)
- `tests/test_simulation/test_asymmetric_team_context.py` (new)
- `tests/test_simulation/test_game_simulator.py`

### Steps

1. Remove overtime from the regulation phase list. `phase_definitions()` must return exactly 48 regulation minutes.
2. Trigger overtime only when the simulated regulation score is tied after integer scoring.
3. Simulate one five-minute overtime period at a time until the score is no longer tied. Add a configurable safety cap such as six periods; if the cap is reached, resolve with a documented one-possession tiebreak and emit metadata.
4. Remove `overtime_margin_threshold` as an overtime trigger. If retained for compatibility, rename it to a clearly non-triggering diagnostic field and deprecate it.
5. In `_run_simulation()`, map each team context to its own `team_targets[team]`. Never choose a target using `list(team_targets.keys())[0]`.
6. Add an asymmetric fixture with one team target at 130 points and the other at 95. Assert each context receives its own target and swapping team order swaps, rather than duplicates, the contexts.
7. Define one normalization policy for all six stats:
   - normalize every player marginal to its declared team target when a calibrated team target exists;
   - for a stat without an independently calibrated team target, define the team target as the sum of calibrated player means and leave the samples at that aggregate rather than applying a new heuristic scale;
   - record the target source per team/stat.
   Do not leave PTS/REB/AST on a different undocumented path from STL/BLK/TOV.
8. Add output metadata:
   - regulation score;
   - overtime periods;
   - tiebreak used;
   - per-stat normalization applied;
   - target source.
9. Add deterministic tests proving the same request and seed produce identical samples.
10. Add invariants:
    - team stat equals sum of its player stats for every simulation;
    - no nonnegative stat is negative;
    - a player marked not played has zero minutes and zero stats;
    - no overtime occurs for a non-tied regulation result;
    - exactly one overtime period is recorded when the first overtime breaks a tie.

### Verification

```bash
source venv/bin/activate
pytest tests/test_simulation/test_overtime_rules.py -v
pytest tests/test_simulation/test_asymmetric_team_context.py -v
pytest tests/test_simulation/test_game_simulator.py -v
pytest tests/test_simulation/test_simulation_health_reporting.py -v
```

### Acceptance gate

- No regulation game receives overtime unless tied.
- Overtime cannot be simulated twice by two separate code paths.
- Each team uses its own target context.
- All six stat normalization rules are explicit and tested.

---

## 12. PR 07 — Replace row scoring with point-in-time replay

### Objective

Build the evaluation authority used by model comparison and promotion.

### Files to add

- `src/evaluation/replay_evaluator.py`
- `replay.py`
- `tests/test_evaluation/test_replay_evaluator.py`
- `tests/fixtures/replay_week/` with small frozen source snapshots

### Files to change

- `src/evaluation/backtest_runner.py`
- `backtest.py`
- `src/evaluation/metrics.py`
- `src/pipeline/forecast_service.py`
- `src/contracts/forecast.py`
- `README.md`

### Steps

1. Define `ReplayConfig` with:
   - bundle ID/path;
   - date range;
   - horizon;
   - snapshot root;
   - strict mode;
   - targets;
   - output path;
   - seed;
   - slice definitions.
2. For each historical game and horizon:
   - load the exact source snapshot available at that cutoff;
   - build a `ForecastRequest`;
   - materialize features through `FeatureMaterializer`;
   - predict through `ForecastService`;
   - store canonical forecast rows;
   - join actuals only after the forecast rows are frozen.
3. Reject the entire replay if `request.forecast_cutoff <= bundle.knowledge_cutoff` or if any source row violates availability time.
4. Include the Transformer history path when the bundle declares a Transformer. Do not pass `history_df=None` as a shortcut.
5. Add baselines:
   - last game;
   - rolling 5 and rolling 10;
   - season average;
   - per-minute rate times rolling minutes;
   - optional market-assisted baseline in a separate scorecard.
6. Add rolling-origin outer folds. For each fold, store train/selection/validation/test cutoffs and bundle ID.
7. Add metrics:
   - target MAE/RMSE/bias;
   - normalized MAE relative to the declared baseline;
   - pinball loss by quantile;
   - interval coverage and sharpness;
   - Brier/log loss for availability when present;
   - threshold Brier/ECE for prop probabilities;
   - required slices.
8. Write immutable replay outputs under `reports/replay/<bundle_id>/<replay_id>/`:
   - `requests.parquet`;
   - `forecasts.parquet`;
   - `actuals.parquet`;
   - `scorecard.json`;
   - `slice_metrics.csv`;
   - `manifest.json` with hashes.
9. Turn `BacktestRunner` into a compatibility wrapper. Mark its current-row mode `legacy_row_score` and prohibit its output from promotion.
10. Update `backtest.py` to call replay by default. Require `--legacy-row-score` to run the old behavior.
11. Add a frozen-week golden replay fixture that runs with no network and validates:
    - request count;
    - source cutoff selection;
    - exact schema hash;
    - no future mutation effect;
    - metric ranges rather than brittle exact model values.

### Verification

```bash
source venv/bin/activate
pytest tests/test_evaluation/test_replay_evaluator.py -v
pytest tests/test_pipeline/test_forecast_materialization_parity.py -v
python replay.py --help
```

### Acceptance gate

- Replay uses the same request, materializer, bundle loader, and forecast service as live simulation.
- A bundle cannot score dates it already used for any training decision.
- Replay outputs are immutable and hash-addressed.
- Legacy row scoring is visibly labeled and cannot feed promotion.

---

## 13. PR 08 — Make the prediction ledger horizon-aware and promotion statistically valid

### Objective

Preserve every pregame forecast and compare champion/challenger bundles only on paired, point-in-time-correct requests.

### Files to change

- `src/evaluation/prediction_ledger.py`
- `src/evaluation/continual_learning.py`
- `src/simulation/report_generator.py`
- `simulate_season.py`
- `improve_models.py`
- `src/evaluation/residual_monitor.py`
- `requirements.txt`
- `tests/test_evaluation/test_prediction_ledger.py`
- `tests/test_evaluation/test_continual_learning.py`
- `tests/test_evaluation/test_promotion_pairing.py` (new)

### Ledger schema

Use canonical names. The immutable identity must include at least:

```text
REQUEST_ID
MODEL_BUNDLE_ID
SNAPSHOT_ID
FORECAST_CUTOFF
HORIZON
SCENARIO
GAME_ID
PLAYER_ID
STAT
```

`GENERATED_AT` is metadata, not a substitute for a forecast cutoff.

### Steps

1. Introduce `ledger_schema_version = 2`.
2. Use `REQUEST_ID + PLAYER_ID + STAT` as the uniqueness key, with a validation assertion that request metadata is internally consistent.
3. Keep all morning, pregame, scenario, and bundle variants. Never deduplicate across horizons or bundle IDs.
4. Accept canonical long-form forecast frames directly. Remove hand-built field translation from `ReportGenerator` after a compatibility migration.
5. Write ledger updates atomically via a temporary Parquet file plus `os.replace()`. Add an advisory lock or a single-writer guard so parallel workers cannot lose rows.
6. Add the tested `pyarrow` version as a direct runtime dependency in this PR. Fail with an actionable dependency error rather than falling back to pickle.
7. Wire `simulate_season.py` to append successful canonical forecasts automatically. Add:
   - `--ledger-path`;
   - `--no-ledger` for explicit opt-out;
   - structured logging of rows appended.
8. Ledger writing must be independent of `--no-csv`; the ledger is an audit artifact, not a presentation export.
9. Reconciliation must only add outcome fields such as `ACTUAL`, `ACTUAL_AVAILABLE_AT`, and `RECONCILED_AT`. It may never rewrite a forecast field.
10. Migrate legacy `MODEL_VERSION` rows with a one-time `migrate_prediction_ledger.py` script. Rows without horizon/snapshot must be labeled `legacy_unknown` and excluded from promotion.
11. Change promotion to use paired canonical replay or ledger rows with identical request/player/stat keys for champion and candidate.
12. Bootstrap at the game cluster level, not the player-row level. Sample `GAME_ID`s with replacement and carry all player/stat rows from each sampled game.
13. Report both:
    - raw MAE delta in stat units;
    - relative MAE improvement percentage.
    Never label raw units as percent.
14. Compute interval coverage from actual quantile bounds, not from a normal approximation unless the bundle explicitly declares that distribution.
15. Require gates:
    - minimum paired games and rows per target/horizon;
    - positive lower confidence bound on normalized improvement;
    - no target regression beyond policy;
    - no critical-slice regression beyond policy;
    - no interval-coverage regression beyond policy;
    - identical source snapshot policy;
    - strict replay status for both bundles.
16. Make `improve_models.py --dry-run` the default. Add `--promote` as the explicit state-changing flag.
17. Write the complete decision input and output to `reports/continual_learning/<decision_id>/` before changing the champion pointer.

### Verification

```bash
source venv/bin/activate
pytest tests/test_evaluation/test_prediction_ledger.py -v
pytest tests/test_evaluation/test_continual_learning.py -v
pytest tests/test_evaluation/test_promotion_pairing.py -v
pytest tests/test_evaluation/test_residual_monitor.py -v
python improve_models.py --help
```

### Acceptance gate

- Morning and pregame forecasts for the same player/game/stat remain distinct.
- Promotion compares only paired strict-replay rows.
- Bootstrap confidence is clustered by game and correctly labeled.
- `improve_models.py` cannot promote unless `--promote` is explicit.

---

## 14. PR 09 — Unify configuration and make every declared setting effective

### Objective

Use one strict, typed configuration path for every CLI and library component.

### Files to change

- `src/config/config.py`
- `src/config/model_config.py`
- `src/training/presets.py`
- `config/default.yaml`
- all root CLI entry points that load configuration
- `tests/test_config/test_config.py`
- `tests/test_config/test_strict_config.py` (new)
- `tests/test_data/test_extension_config.py` (new)
- `AGENTS.md`
- `README.md`

### Steps

1. Add `architecture_version: 2` and `config_schema_version: 2` at the root of `config/default.yaml`.
2. Add typed config dataclasses for currently untyped or ignored sections:
   - residual correction;
   - residual monitoring;
   - reasoning;
   - continual learning;
   - weighting;
   - data sources;
   - forecast horizons/snapshots;
   - replay;
   - artifact policy.
3. Define a strict loader mode that rejects unknown top-level and nested keys with the full dotted path.
4. Make strict mode the default for training, replay, simulation, and promotion. Allow `--allow-unknown-config` only as a temporary migration flag.
5. Remove direct `yaml.safe_load()` calls from scripts. They must call `load_config(path)` and use typed fields.
6. Remove the global default-config trap. Delete mutable global config state from runtime paths and pass the loaded `Config` object explicitly through constructors and orchestration functions. Keep `get_config()` only as a temporary deprecated compatibility wrapper that raises unless `set_config()` was explicitly called by a legacy caller; remove both wrappers by PR 18.
7. Add `--config` consistently to `update_data.py`, `train.py`, `simulate_season.py`, `backtest.py`/`replay.py`, query, residual tools, and promotion tools.
8. Fix `_run_configured_extensions()` to use the CLI-loaded config instead of a fresh default object.
9. Make `residual_correction.enabled: false` actually prevent model loading/application.
10. Resolve duplicate configuration sources between `config.py`, `model_config.py`, presets, and hardcoded simulator defaults. Define precedence exactly:
    1. built-in dataclass defaults;
    2. YAML;
    3. preset;
    4. explicit CLI flags.
11. Emit `config.resolved.yaml` in every training/simulation/replay run.
12. Add tests for:
    - every key in `config/default.yaml` is consumed;
    - unknown key rejection;
    - CLI override precedence;
    - extension enablement through `update_data.py`;
    - residual disablement;
    - config round-trip;
    - path resolution relative to project root/config location.

### Verification

```bash
source venv/bin/activate
pytest tests/test_config/ -v
pytest tests/test_training/test_training_presets.py -v
pytest tests/test_data/test_extension_config.py -v
python update_data.py --help
python train.py --help
python simulate_season.py --help
```

### Acceptance gate

- No YAML key is silently ignored.
- All CLIs resolve configuration through the same loader and precedence rules.
- Configured extension sources and residual disablement work in integration tests.

---

## 15. PR 10 — Add atomic, immutable data snapshots and data-health gates

### Objective

Make every training or forecast input reproducible and prevent partial data writes.

### Files to add

- `src/data/snapshot_store.py`
- `src/contracts/data.py`
- `tests/test_data/test_snapshot_store.py`
- `tests/test_contracts/test_data_contracts.py`

### Files to change

- `update_data.py`
- `src/data/base_scraper.py`
- `src/data/injury_history_logger.py`
- `src/data/player_bio_scraper.py`
- `src/data/extensions/advanced_tracking_scraper.py`
- `check_contracts.py`
- `clear_cache.py`
- `config/default.yaml`

### Target layout

```text
data/
  current.json
  snapshots/
    <snapshot_id>/
      manifest.json
      core/
        player_games.parquet
        team_games.parquet
      optional/
        player_bios.parquet
        injury_history.parquet
        lineups.parquet
        betting.parquet
        advanced_tracking.parquet
      raw/
        <source payloads or source response hashes>
```

CSV compatibility exports may remain at `data/nba_players.csv` and `data/nba_games.csv`, but they are derived outputs, not the authoritative snapshot.

### Steps

1. Define schema-versioned contracts for:
   - player games;
   - team games;
   - schedule;
   - roster membership;
   - injuries/status;
   - lineups;
   - betting lines;
   - player bios;
   - source manifest.
2. Every source row must have:
   - source name;
   - source entity ID;
   - event time;
   - available-at time;
   - fetched-at time;
   - source status/confidence where applicable.
3. `update_data.py` must write into `<snapshot_id>.tmp/`, validate every table, write hashes/row counts/min-max dates, then atomically rename and update `data/current.json`.
4. Write CSV compatibility files only after the snapshot is valid. Write each to a temporary file and atomically replace it.
5. Treat Parquet failure as a failed snapshot when Parquet is configured as authoritative. Do not log and continue with mismatched formats.
6. Make extension outputs participate in the same transaction. A non-required extension may be absent, but its failure/status must be in the manifest.
7. Add data-health gates:
   - unique key checks;
   - exactly two team rows per completed game;
   - player-game IDs must map to a known team-game;
   - expected columns/dtypes;
   - monotonic/valid dates;
   - plausible ranges;
   - source freshness;
   - optional-source coverage by date/team/player.
8. Resolve the current 2,442-versus-2,441 game mismatch. Produce a report identifying the exact player-only game and choose one of:
   - repair from source;
   - exclude with a documented reason;
   - retain as quarantined data not eligible for training.
9. Make missing `player_bios.csv`, `injury_history.csv`, or tracking data a structured feature-group health state. Disable or neutralize affected groups explicitly; do not advertise them as fully active.
10. Start immutable collection of injuries/lineups/odds immediately even if historical backfill is incomplete.
11. Add a `python check_data.py --snapshot <id>` CLI that emits human-readable and JSON reports.
12. Update feature-cache keys to include snapshot ID and external source hashes, not only mutable file mtimes/sizes.
13. Add crash tests that simulate failure between writes and prove `data/current.json` still points to the last complete snapshot.

### Verification

```bash
source venv/bin/activate
pytest tests/test_data/test_snapshot_store.py -v
pytest tests/test_contracts/test_data_contracts.py -v
pytest tests/test_data/ -v
python check_data.py --snapshot current
python update_data.py --help
```

### Acceptance gate

- A consumer can load a named snapshot without reading newer mutable files.
- Partial updates never replace `data/current.json`.
- Every forecastable value has event and availability provenance.
- Optional-source absence is explicit and feature groups react deterministically.

---

## 16. PR 11 — Build a historical availability and minutes foundation

### Objective

Model whether a player participates and how many minutes they play before forecasting box-score totals.

### Files to add

- `src/preprocessing/participation_panel.py`
- `src/models/availability_model.py`
- `src/simulation/rotation_allocator.py`
- `tests/test_preprocessing/test_participation_panel.py`
- `tests/test_models/test_availability_model.py`
- `tests/test_models/test_minutes_model.py`
- `tests/test_simulation/test_rotation_allocator.py`

### Files to change

- `src/models/minutes_predictor.py`
- `src/training/pipeline.py`
- `src/models/versioning.py`
- `src/pipeline/forecast_service.py`
- `src/simulation/game_simulator.py`
- `src/contracts/forecast.py`
- `config/default.yaml`

### Steps

1. Build one player-team-game participation panel by crossing effective-dated roster membership with team schedule rows.
2. Define mutually exclusive labels:
   - not on roster;
   - active and played;
   - active but DNP-CD;
   - inactive/injured;
   - suspended/personal/other unavailable;
   - two-way/G League assignment when observable;
   - unknown.
3. Preserve label provenance and availability time. Do not infer a historical pregame status from a source first observed after tipoff.
4. Add player/team identity resolution and transaction effective dates. A traded player must not appear on both teams for the same forecast cutoff unless the source explicitly indicates ambiguity.
5. Train a CatBoost binary classifier for `P(active and plays)` using point-in-time features only. Produce OOF probabilities over chronological inner folds. On the final inner calibration window, compare sigmoid/Platt and isotonic calibration; choose lower Brier score, break ties with log loss, and use sigmoid automatically when either class has fewer than 500 calibration examples. Refit the chosen calibrator on the complete inner calibration window and record the method/sample counts.
6. Evaluate against simple baselines:
   - last-game appearance;
   - current injury status mapping;
   - rolling appearance rate.
7. Report Brier score, log loss, calibration curve, and slices for injury return, late scratch, bench, two-way, trade, and cold start.
8. Rebuild `MinutesPredictor` as a required LightGBM bundle component conditional on playing. Train one central model plus quantile objectives at alpha 0.10, 0.50, and 0.90, then enforce monotonic output ordering. It must return:
   - point estimate;
   - P10/P50/P90;
   - role/starter probability;
   - uncertainty for missing lineup/status.
9. Train minutes with only features available at the forecast horizon. Any coach or role feature currently represented by a constant must be either sourced, declared missing, or removed.
10. Evaluate minutes against rolling-5, rolling-10, season-average, and role-average baselines using repeated rolling folds.
11. Implement `RotationAllocator`:
    - sample availability first;
    - require at least five active players or emit a structured invalid-roster result;
    - allocate exactly 240 regulation player-minutes per team;
    - allocate five additional team-player minutes per overtime minute;
    - respect per-player bounds and role ordering;
    - expose any constraint relaxation in metadata.
12. Add availability and minutes artifacts to `ModelBundleManifest` and make them required for architecture version 2.
13. Extend canonical forecast rows with availability probability and minutes quantiles produced by the bundle, not copied from context defaults.
14. Do not remove the existing heuristic minutes fallback until parity tests and promotion gates accept the learned component. In strict replay, fallback is prohibited.

### Verification

```bash
source venv/bin/activate
pytest tests/test_preprocessing/test_participation_panel.py -v
pytest tests/test_models/test_availability_model.py -v
pytest tests/test_models/test_minutes_model.py -v
pytest tests/test_simulation/test_rotation_allocator.py -v
```

### Acceptance gate

- The panel contains non-appearance rows, not only box-score appearances.
- Availability beats at least one declared simple baseline and is calibrated.
- Minutes beat rolling-10 on the primary replay score without unacceptable slice regressions.
- Regulation allocations always sum to exactly 240 team-player minutes.

### Stop condition

If historical roster/status coverage is insufficient for a horizon, do not manufacture labels. Mark that horizon unsupported, begin immutable collection, and keep the heuristic component as a visible degraded fallback outside strict evaluation.

---

## 17. PR 12 — Add conditional stat-rate challengers and honest calibration

### Objective

Evaluate whether opportunity-conditioned stat models outperform direct full-game totals without contaminating the clean baseline.

### Files to add

- `src/models/stat_rate_model.py`
- `src/models/distribution_calibrator.py`
- `tests/test_models/test_stat_rate_model.py`
- `tests/test_models/test_distribution_calibrator.py`
- `tests/test_evaluation/test_probabilistic_metrics.py`

### Files to change

- `src/training/pipeline.py`
- `src/training/catboost_trainer.py`
- `src/models/model_manager.py`
- `src/query/distribution_fitter.py`
- `src/query/probability_calculator.py`
- `src/evaluation/metrics.py`
- `src/evaluation/replay_evaluator.py`
- `config/default.yaml`

### Steps

1. Keep the direct-total CatBoost model as the champion baseline.
2. Add CatBoost regression challengers using a declared exposure:
   - per-minute rates for player opportunity;
   - per-possession rates only where possession exposure is reliable;
   - zero-inflated/count-aware formulations for sparse stats.
3. During training and replay, use **out-of-fold predicted minutes**, never actual future minutes, as the exposure supplied to downstream stat models.
4. Add hierarchical shrinkage for:
   - rookies/cold starts;
   - low-minute players;
   - recent trades;
   - abrupt role changes;
   - rare STL/BLK/TOV outcomes.
5. Train target-appropriate CatBoost quantile outputs:
   - ordered quantiles for all targets;
   - nonnegative support;
   - optional zero-inflation for sparse stats;
   - discrete count treatment where it improves replay.
6. Add a monotonic quantile correction at the model-output boundary and count how often correction was needed. Excessive crossing is a model-quality failure, not something to hide.
7. Fit calibration only on inner holdout or OOF predictions. Store calibrators by stat, horizon, and supported minutes band.
8. Make residual correction eligible only when trained on OOF residuals from the exact base bundle. Record base bundle ID and schema hash in residual metadata.
9. Add ablations:
   - direct total only;
   - conditional rate only;
   - blend;
   - with/without residual correction;
   - with/without lifecycle features;
   - with/without Transformer.
10. Compare using replay, not validation-only metrics.
11. Require probabilistic metrics:
    - pinball loss;
    - CRPS where supported;
    - empirical 50/80/90 coverage;
    - interval width/sharpness;
    - Brier/ECE for requested thresholds;
    - zero-probability calibration.
12. Make `ProbabilityCalculator` consume the calibrated canonical forecast distribution. It must not independently infer a new distribution when the bundle provides one.
13. If a distribution fallback is needed, label it in output metadata and exclude it from strict promotion comparisons.

### Verification

```bash
source venv/bin/activate
pytest tests/test_models/test_stat_rate_model.py -v
pytest tests/test_models/test_distribution_calibrator.py -v
pytest tests/test_evaluation/test_probabilistic_metrics.py -v
pytest tests/test_query/test_probability_calculator.py -v
```

### Acceptance gate

- All downstream exposure comes from OOF/live-predicted minutes, never actual minutes.
- Quantiles are ordered and nonnegative.
- Calibration is fit outside the outer test window.
- The conditional challenger is promoted only through paired replay gates.

---

## 18. PR 13 — Decompose simulation and remove unproven double adjustments

### Objective

Turn the simulator into a small orchestration layer over independently testable, calibrated components.

### Files to add

- `src/simulation/roster_builder.py`
- `src/simulation/joint_stat_sampler.py`
- `src/simulation/coherence.py`
- `src/simulation/orchestrator.py`
- `tests/test_simulation/test_roster_builder.py`
- `tests/test_simulation/test_joint_stat_sampler.py`
- `tests/test_simulation/test_coherence.py`
- `tests/test_simulation/test_simulator_ablation.py`

### Files to change

- `src/simulation/game_simulator.py`
- `src/simulation/phase_simulator.py`
- `src/simulation/player_correlation_engine.py`
- `src/simulation/four_factors_engine.py`
- `src/simulation/game_context_engine.py`
- `src/simulation/role_sampler.py`
- `src/simulation/report_generator.py`
- `config/default.yaml`

### Target flow

```text
ForecastRequest
  -> point-in-time roster/status
  -> availability samples
  -> constrained minutes/role samples
  -> possession/game-environment samples
  -> conditional player stat samples
  -> residual dependence/correlation
  -> coherence validation
  -> canonical simulation result
```

### Steps

1. Protect current corrected behavior with golden deterministic tests before moving code.
2. Extract roster selection and status handling from `GameSimulator` into `RosterBuilder`.
3. Use the availability/minutes bundle from PR 11. Remove starter probability floors and recent-player cutoffs from the strict path unless they are explicit evaluated policies.
4. Extract joint sampling into `JointStatSampler`. It must consume calibrated player marginals rather than re-deriving means.
5. Fit player/team dependence on OOF residual ranks using a Gaussian copula and Ledoit-Wolf shrinkage covariance. Fit a global matrix first; fit role/lineup partitions only when a partition has at least 1,000 complete OOF examples. Fall back to the global learned matrix when a partition is undersized. Do not use same-game outcomes or hand-maintained matrices as if they were learned production parameters.
6. Keep a fixed correlation matrix only as a named fallback scenario with `DATA_QUALITY=DEGRADED_FALLBACK`.
7. Implement `CoherenceValidator` assertions for every simulation draw:
   - team stat equals player-stat sum;
   - player who does not play has zero minutes/stats;
   - regulation minutes sum to 240;
   - overtime minutes are added consistently;
   - counts are nonnegative and integral where declared discrete;
   - quantile/marginal summaries match forecast distributions within tolerance;
   - team points and game total are internally consistent.
8. Inventory every multiplier and hardcoded constant in the current simulator. For each, create a row in `docs/simulation_adjustment_inventory.md` with:
   - symbol/file;
   - value;
   - intended effect;
   - data source;
   - whether already represented in model features;
   - replay ablation result;
   - decision: learned, scenario-only, retained, or removed.
9. Remove any adjustment that duplicates model inputs unless replay proves incremental value.
10. Separate model-only and market-assisted scenarios. Vegas calibration may change the market-assisted result only; it may not overwrite the model-only forecast.
11. Replace global random seeding with request-scoped `numpy.random.Generator` instances derived from request ID plus user seed.
12. Batch simulation arrays where possible. Preserve deterministic equivalence for a fixed seed.
13. Reduce `GameSimulator` to a compatibility facade of roughly 250–400 lines. Do not use line count alone as an acceptance criterion; responsibilities and tests are the real gate.
14. Benchmark 100, 1,000, and 10,000 simulations on CPU and available GPU. Record wall time and peak memory in `reports/benchmarks/`.

### Verification

```bash
source venv/bin/activate
pytest tests/test_simulation/ -v
pytest tests/test_simulation/test_simulator_ablation.py -v
pytest -m gpu -v
python simulate_season.py --date <frozen-fixture-date> --sims 100 --workers 1 --strict
```

Use a checked-in frozen schedule/snapshot fixture for the CLI verification command; do not use the network in CI.

### Acceptance gate

- All coherence invariants pass across large randomized property-style runs.
- Fixed seed plus request ID is deterministic.
- Model-only output is never silently changed by market data.
- Every retained heuristic has a documented replay ablation.

---

## 19. PR 14 — Make query a presentation layer over canonical distributions

### Objective

Stop query-time code from changing model predictions and simplify the 1,100-line interactive CLI.

### Files to add

- `src/query/commands.py`
- `src/query/presenters.py`
- `tests/test_query/test_query_prediction_invariance.py`

### Files to change

- `src/query/interactive_cli.py`
- `src/query/probability_calculator.py`
- `src/query/projection_loader.py`
- `src/query/distribution_fitter.py`
- `src/query/prob_formatter.py`
- `query_prob.py`
- `tests/test_query/`

### Steps

1. Define one `PlayerForecastView` adapter over canonical forecast rows.
2. Make `ProjectionLoader` select by:
   - player;
   - game;
   - horizon;
   - scenario;
   - bundle;
   - latest valid cutoff.
3. Require the loader to display bundle ID, horizon, cutoff, snapshot ID, and data quality in `info` output.
4. Remove numerical mutation from `InteractiveCLI._handle_over_under()`:
   - no recent-form reblend;
   - no second minutes scaling;
   - no second defense multiplier;
   - no trend or matchup-history adjustment;
   - no manual home/away addition.
5. Preserve those values only as explanatory evidence when they are already represented in the prediction trace.
6. Make `ProbabilityCalculator` calculate probabilities from canonical quantiles/distribution parameters. It may use Monte Carlo to integrate the supplied distribution, but it may not invent a different mean.
7. Add an invariant test: asking `projection`, `over/under`, `why`, and `compare` for the same player must report the identical canonical mean and bundle identity.
8. Split command parsing/dispatch from formatting and I/O. Keep `InteractiveCLI` as the loop and compatibility facade.
9. Replace direct `print()` in library code with returned presentation strings. Root CLI code may print.
10. Add one-shot, noninteractive JSON output for automation:

    ```bash
    python query_prob.py --player "Player Name" --stat PTS --line 24.5 --json
    ```

11. JSON output must include request/bundle identity, probability method, fallback status, and distribution metadata.

### Verification

```bash
source venv/bin/activate
pytest tests/test_query/ -v
pytest tests/test_query/test_query_prediction_invariance.py -v
python query_prob.py --help
```

### Acceptance gate

- Query commands never change the canonical forecast mean or intervals.
- All query modes select the same request identity.
- Library modules return data/strings; only CLI boundaries print.

---

## 20. PR 15 — Consolidate scrapers, logging, exceptions, and large-module boundaries

### Objective

Reduce copy-paste behavior and make failures observable without changing numerical forecasts.

### Files to change or add

- `src/data/nba_defense_scraper.py`
- `src/data/lineup_scraper.py`
- `src/data/betting_scraper.py`
- `src/data/basketball_ref_scraper.py`
- `src/data/schedule_scraper.py`
- `src/data/http_client.py` (new shared client)
- `src/utils/logging_config.py`
- `src/contracts/errors.py`
- root CLI entry points
- focused tests under `tests/test_data/` and `tests/test_config/`

### Steps

1. Add a shared HTTP client with:
   - configured timeout;
   - retry/backoff with jitter;
   - rate limiting;
   - user-agent/headers;
   - response status classification;
   - circuit-breaker state;
   - injectable session/clock for tests.
2. Keep source-specific parsing in source adapters. Do not create one giant generic scraper.
3. Split `NBADefenseScraper` source acquisition from `DefensiveMatchupAnalyzer` analysis. Remove copied player-vs-team, switchability, rim-protection, and comprehensive-factor fetch implementations from the analyzer; delegate to the scraper.
4. Add fixture tests for every parsing strategy and fallback. Include malformed HTML/JSON, missing keys, rate limits, and stale cache.
5. Define typed exceptions:
   - configuration error;
   - data contract error;
   - source unavailable;
   - source malformed;
   - artifact incompatible;
   - prediction unavailable;
   - strict degradation error.
6. Replace broad `except Exception` only where the caller can take a specific action. At source isolation boundaries, broad catch is allowed but must preserve exception type/message in structured health metadata and log a traceback at debug level.
7. Centralize logging configuration. Only root entry points call `configure_logging()`; imported modules must not call `logging.basicConfig()`.
8. Add run/request/bundle/snapshot IDs to structured log context.
9. Remove executable demo `print()` blocks from production modules or move them to explicit CLI examples under `scripts/`.
10. Continue decomposing large modules only along tested responsibility boundaries:
    - training orchestration vs evaluation vs artifact writing;
    - query dispatch vs presentation;
    - GPU discovery vs memory management vs feature execution.
11. Do not combine this PR with numerical model changes. Golden parity tests must show unchanged output for fixed fixtures except for intentionally corrected error handling.

### Verification

```bash
source venv/bin/activate
pytest tests/test_data/ -v
pytest tests/test_config/ -v
pytest tests/test_simulation/test_simulation_health_reporting.py -v
rg -n "logging\.basicConfig" src
```

Expected final `rg`: no `logging.basicConfig` calls under `src/`.

### Acceptance gate

- Analyzer code delegates data acquisition instead of duplicating it.
- Every network call uses the shared configured client or has a documented exception.
- Imported modules do not configure global logging.
- Refactoring preserves golden outputs.

---

## 21. PR 16 — Add dependency, lint, type, warning, and CI discipline

### Objective

Make a clean CPU installation reproducible and run fast safety checks automatically.

### Files to add

- `pyproject.toml`
- `.github/workflows/ci.yml`
- `.pre-commit-config.yaml`
- `requirements-lock.txt`

### Files to change

- `requirements.txt`
- `requirements-dev.txt`
- `AGENTS.md`
- `README.md`
- warning-producing smart-selection code/tests

### Steps

1. Confirm `pyarrow` remains a direct runtime dependency because Parquet ledger/snapshot writes require it.
2. Apply this dependency policy:
   - use compatible bounded ranges in `requirements.txt` and `requirements-dev.txt`;
   - generate exact transitive pins in `requirements-lock.txt` on Python 3.12;
   - keep PyTorch installation in separate documented CPU/CUDA/macOS commands and record the tested Torch version in bundle metadata.
3. Add bounded `ruff`, `mypy`, `pre-commit`, and `pip-tools` dependencies to `requirements-dev.txt`. Generate the lock on Python 3.12 with:

   ```bash
   source venv/bin/activate
   pip-compile --resolver=backtracking --output-file=requirements-lock.txt requirements-dev.txt
   ```

   Record platform-specific exceptions instead of pretending one CUDA wheel works everywhere.
4. Configure Ruff in `pyproject.toml` with a conservative initial ruleset:
   - syntax/undefined names;
   - import ordering;
   - high-value bugbear rules;
   - PEP 8 basics.
5. Run Ruff once and fix violations in bounded batches. Do not apply a repository-wide behavioral rewrite in the same PR as model logic.
6. Configure mypy as the staged type checker. Start with contracts, config, versioning, forecast service, snapshot store, and replay. Do not require full strict typing over all 53k lines immediately.
7. Configure pre-commit to run trailing-whitespace/end-of-file checks, `ruff check`, `ruff format --check`, and the staged mypy targets. Document that hooks supplement, rather than replace, CI.
8. Register pytest markers in one configuration source and add warning policy:
   - project deprecations as errors;
   - selected third-party warnings filtered with a reason;
   - no blanket warning suppression.
9. Fix the 304 scikit-learn feature-name warnings by passing consistent DataFrame feature names through fit and prediction/permutation paths.
10. Resolve joblib core detection by setting an explicit test worker/core policy, not by suppressing all joblib warnings.
11. Add CPU CI on Python 3.12 with jobs:
    - dependency install and `pip check`;
    - compileall;
    - Ruff;
    - staged type check;
    - fast tests (`pytest -m "not slow and not gpu"`);
    - artifact/contract fixture tests;
    - frozen replay smoke test.
12. Do not make CI call external sports sites.
13. Add an optional/manual GPU workflow only if a compatible runner exists. Its absence must not block CPU correctness.
14. Upload test reports and coverage XML on failure. Set an initial coverage floor based on measured baseline, then raise it only with new tests. Do not invent a floor before measuring.
15. Add commands to `AGENTS.md` and README:

    ```bash
    source venv/bin/activate
    ruff check .
    pytest -m "not slow and not gpu" -q
    python -m compileall -q src tests *.py
    pip check
    ```

### Verification

```bash
source venv/bin/activate
pip check
ruff check .
python -m compileall -q src tests *.py
pytest -m "not slow and not gpu" -q
pytest --disable-warnings -q
```

Do not use the last command as the official test result; it is only a diagnostic comparison. The official suite must run with warnings visible and governed.

### Acceptance gate

- A fresh CPU Python 3.12 environment installs from documented files.
- `pyarrow` is declared directly.
- Fast CI is network-independent and green.
- The 304 repeated feature-name warnings are eliminated.
- Static checks cover new architecture modules.

---

## 22. PR 17 — Add performance budgets and batch critical paths

### Objective

Improve throughput only after semantic parity is locked down.

### Files to add

- `benchmarks/benchmark_materialization.py`
- `benchmarks/benchmark_forecast_batch.py`
- `benchmarks/benchmark_simulation.py`
- `tests/test_performance/test_batch_parity.py`

### Files to change

- `src/pipeline/forecast_service.py`
- `src/models/model_manager.py`
- `src/evaluation/replay_evaluator.py`
- `src/preprocessing/feature_engineer.py`
- `src/preprocessing/feature_engineer_gpu.py`
- `src/simulation/orchestrator.py`
- cache modules

### Steps

1. Measure baseline wall time and peak memory for:
   - feature materialization for one game and full slate;
   - 100 and full-slate forecasts;
   - frozen-week replay;
   - 100/1,000/10,000 simulations.
2. Record hardware, dependency versions, bundle, snapshot, and seed with every benchmark.
3. Replace `ForecastService.predict_player_stats_batch()` row loops with actual batch model inference while preserving per-player Transformer history and per-target schema alignment.
4. Batch replay by request/slate where semantics allow. Never batch across different cutoffs or snapshots in a way that changes input visibility.
5. Prefer Parquet over pickle for tabular caches. Keep model-specific joblib/CatBoost formats only where required.
6. Include schema/config/snapshot/external-source hashes in cache keys.
7. Make cache writes atomic and checksum-validated.
8. For GPU execution:
   - retain `max_workers=1` for a shared CUDA context;
   - verify CPU/GPU feature parity;
   - measure transfer overhead before moving small operations to GPU;
   - fail back to CPU visibly, not silently, in non-strict mode.
9. Add batch-vs-single prediction parity tests with tight tolerances.
10. Establish budgets from measured results, for example “no more than 10% regression from the accepted baseline” rather than arbitrary absolute numbers.

### Verification

```bash
source venv/bin/activate
pytest tests/test_performance/test_batch_parity.py -v
python benchmarks/benchmark_materialization.py
python benchmarks/benchmark_forecast_batch.py
python benchmarks/benchmark_simulation.py --sims 100 1000 10000
```

### Acceptance gate

- Batch and single predictions are numerically equivalent within declared tolerance.
- Performance improves without changing cutoff semantics or cache identity.
- Regressions beyond the recorded budget fail CI or a required benchmark gate.

---

## 23. PR 18 — Final documentation, operating playbook, and clean-baseline release

### Objective

Publish a coherent local CLI workflow and promote the first trustworthy bundle.

### Files to change

- `README.md`
- `AGENTS.md`
- `TASKS.md`
- `DECISIONS.md`
- `plans/2026_27_architecture_rebuild.md`
- this plan
- CLI `--help` text

### Steps

1. Update architecture diagrams and project tree to match the implemented packages.
2. Document the authoritative pipeline:

   ```text
   update/snapshot -> validate data -> train candidate -> strict outer score
   -> strict replay -> promotion decision -> champion -> forecast/simulate
   -> ledger -> reconcile -> monitor
   ```

3. Document exact first-time commands, daily commands, weekly evaluation, retraining, promotion, rollback, and cleanup.
4. Document artifact/data layouts and retention policy.
5. Document supported forecast horizons and source-coverage limitations.
6. Clearly label model-only versus market-assisted outputs.
7. Add troubleshooting for:
   - unsafe legacy schema;
   - missing champion;
   - corrupt bundle hash;
   - stale/missing snapshot;
   - optional source degradation;
   - CUDA unavailable;
   - Parquet dependency missing;
   - replay cutoff overlap;
   - insufficient promotion pairs.
8. Record decisions in `DECISIONS.md` for:
   - strict point-in-time provenance;
   - immutable snapshot and bundle policy;
   - `knowledge_cutoff` definition;
   - replay as promotion authority;
   - availability/minutes first;
   - model-only/market-assisted separation;
   - query as presentation only.
9. Mark completed tasks in `TASKS.md` and remove stale claims that conflict with actual behavior.
10. Run a clean small/laptop-quality candidate training against a validated snapshot.
11. Verify its schema contains no forbidden columns.
12. Run strict outer scoring and frozen replay.
13. Compare against simple baselines. Do **not** compare against the unsafe legacy bundle as evidence of improvement.
14. Promote only if all final gates below pass.
15. Retain legacy artifacts as an explicitly labeled archive until the new champion is proven operational; do not delete them in the release PR.

### Final verification commands

```bash
source venv/bin/activate
pip check
ruff check .
python -m compileall -q src tests *.py
git diff --check
pytest tests/ -v
python check_data.py --snapshot current
python check_contracts.py --models-dir models/versions/<candidate_id>
python replay.py --bundle models/versions/<candidate_id> --fixture tests/fixtures/replay_week --strict
python simulate_season.py --date <fixture-date> --sims 100 --workers 1 --strict
python query_prob.py --player "<fixture-player>" --stat PTS --line 20.5 --json
```

### Final release gate

All conditions must be true:

- Full test suite passes, including slow/integration tests.
- Static checks, compileall, diff check, and dependency check pass.
- No forbidden feature exists in any master or per-target schema.
- Candidate outer scorecard is strict and complete for all six targets.
- Candidate replay is point-in-time and immutable.
- Bundle manifest verifies every component.
- Champion pointer update and rollback are tested.
- Forecast, simulation, ledger, query, and replay show the same bundle/request identity.
- Simulation invariants pass.
- No query-time mean mutation occurs.
- Optional data limitations are visible.
- Documentation commands match actual CLI help.

---

## 24. Required pull-request order and dependency map

Use this exact merge order:

```text
PR 00 Baseline preservation
  -> PR 01 Feature safety contract
  -> PR 02 Split/selection wiring
  -> PR 03 Exact runtime outer scoring
  -> PR 04 Immutable bundles
  -> PR 05 Scheduled-game/canonical forecast path
  -> PR 06 Immediate simulation correctness
  -> PR 07 Point-in-time replay
  -> PR 08 Ledger and promotion
  -> PR 09 Strict unified config
  -> PR 10 Atomic data snapshots
  -> PR 11 Availability/minutes
  -> PR 12 Conditional rates/calibration
  -> PR 13 Simulator decomposition
  -> PR 14 Query simplification
  -> PR 15 Scraper/logging/module cleanup
  -> PR 16 Dependencies/static checks/CI
  -> PR 17 Performance
  -> PR 18 Documentation and release
```

Parallel work is allowed only after contracts are merged:

- PR 06 may be developed alongside PR 04–05 after PR 01 is merged, but it must rebase on the canonical request work before merge.
- PR 09 configuration work may be developed alongside PR 07–08, but no CLI migration merges before PR 08’s schema is stable.
- PR 15 and PR 16 may be prepared after PR 10, but repository-wide formatting must wait until behavior PRs are merged to reduce conflicts.
- PR 17 never starts before parity tests in PR 05, PR 07, and PR 13 exist.

---

## 25. Cross-cutting test matrix

Every affected pull request must select rows from this matrix and show results in its description.

| Risk | Required test |
|---|---|
| Current-game leakage | Same-game player/team target mutation invariance |
| Future leakage | Mutate all rows after cutoff; prior forecast remains byte/numerically identical |
| Selection leakage | Outer-test-only signal never appears in manifest |
| Manifest ignored | Serialized target feature names equal manifest |
| Split overlap | Replay rejects any date at/before knowledge cutoff |
| Serialization drift | In-memory and reloaded forecast parity |
| Snapshot drift | Named snapshot read never touches newer current files |
| Horizon collapse | Morning and pregame ledger rows both persist |
| Promotion mismatch | Only identical request keys are paired |
| Correlated errors | Bootstrap resamples games, not independent player rows |
| Scheduled-row staleness | Latest completed game changes next shifted rolling feature |
| Overtime | Non-tie has zero overtime; tie receives periods until broken |
| Team-order bug | Swapping home/away swaps contexts without duplicating first-team target |
| Simulation coherence | Player sums equal team sums; minutes constraints hold |
| Query double adjustment | Mean/interval invariant across query commands |
| Cache staleness | Source/config/schema change produces a new cache key |
| Artifact corruption | Hash mutation prevents load/promotion |
| Config drift | Every YAML key is consumed or rejected |
| Optional source missing | Explicit degraded state; strict path fails if required |
| CPU/GPU divergence | Frozen feature/prediction parity within tolerance |

---

## 26. Artifact and schema versioning rules

Apply these rules consistently:

1. Increment a schema version for any incompatible column, key, dtype, or semantic change.
2. Store schema version in both the artifact and its manifest.
3. Never infer compatibility from a filename alone.
4. A migration must be a separate explicit command; runtime loading does not silently migrate.
5. Pickle/joblib objects are runtime details. Pair them with inspectable JSON metadata and verify hashes before load.
6. Use these initial version labels unless implementation already establishes compatible higher ones:
   - `feature_schema_v4` for semantic feature safety;
   - `forecast_schema_v2` for full request identity;
   - `ledger_schema_v2` for horizon-aware keys;
   - `snapshot_schema_v1` for immutable data snapshots;
   - `bundle_schema_v2` for complete immutable bundles;
   - `replay_schema_v1` for canonical replay outputs.
7. Store old-to-new mappings in `docs/migrations/` and add migration fixture tests.

---

## 27. Deterministic decision rules for the executor

Use these rules instead of making ad hoc choices:

1. **A new feature is proposed:** reject it unless its source, entity keys, event time, availability time, lookback, horizon, missing policy, and dtype are declared.
2. **A source lacks historical availability timestamps:** it may be used for live scenario context but not historical strict replay until a defensible snapshot policy exists.
3. **A complex component does not beat its simple baseline:** keep it as a disabled challenger or remove it; do not blend it merely because it exists.
4. **A fallback is required during strict evaluation:** fail the evaluation. Do not fill it and continue.
5. **A candidate improves average MAE but hurts a critical target/slice beyond policy:** reject promotion.
6. **A config key is unknown:** fail with its dotted path. Do not ignore it.
7. **A bundle component hash differs:** refuse load, promotion, and rollback.
8. **A champion pointer is broken:** fail closed and instruct the operator to run validated rollback. Do not use flat root models.
9. **CPU and GPU disagree beyond tolerance:** treat CPU as the correctness reference, disable the divergent GPU path, and investigate.
10. **Historical availability data is sparse:** report supported coverage and defer that horizon. Do not fabricate status labels.
11. **Market data improves score:** report a separate market-assisted result. Do not overwrite the model-only score.
12. **A refactor changes fixed-seed output unexpectedly:** stop and identify whether it is an intended correction. Update golden fixtures only with a documented decision.
13. **A test produces repeated warnings:** fix the cause or add a narrowly scoped filter with a linked reason. Never disable all warnings.
14. **A performance optimization needs a different cutoff/snapshot grouping:** reject it. Correctness and replay identity take precedence over throughput.

---

## 28. Definition of done for the entire plan

The project is improved—not merely rearranged—when an operator can execute this lifecycle without hidden state:

1. Fetch data into an immutable snapshot.
2. Validate snapshot contracts and source coverage.
3. Train a candidate from an explicit historical window.
4. Select features and tune only inside training history.
5. Score a completely untouched outer window through reloaded runtime artifacts.
6. Replay supported historical horizons through the same forecast path.
7. Compare the candidate to simple baselines and the champion on paired requests.
8. Atomically promote a checksum-verified bundle only when all gates pass.
9. Generate a live forecast with full request identity.
10. Simulate coherent outcomes without double adjustments.
11. Append every forecast horizon to an immutable ledger.
12. Reconcile actuals without changing forecast fields.
13. Query the stored calibrated distribution without mutating its mean.
14. Trace any forecast back to exact code, config, data snapshot, schema, and component artifacts.
15. Roll back safely if the champion is damaged or regresses.

Until all fifteen are true, retain the architecture version 2 work behind explicit compatibility boundaries and do not describe it as a fully self-optimizing production system.
