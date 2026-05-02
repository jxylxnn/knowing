# Current State

## Snapshot

- Observed date: 2026-04-25
- Repository health: good
- Test status in this workspace:
  - full suite (2026-04-25): `178 passed, 0 failed` in 74.82s
  - full suite after teammate-utils refactor + rest_density/lineup_stability vectorisation + performance smoke tests: `178 passed, 0 failed`
  - full suite after adding dedicated unit tests for the 7 new feature groups: `172 passed, 0 failed`
  - full suite after scraper reliability fixes (rotowire, nba_defense, schedule): `140 passed, 0 failed`
  - full suite after TRANSFORMER-SEQ-001 (M tier seq_len=20, zero-padding, new tests): `136 passed, 0 failed`
  - full suite after adding 6-stat export/load regression tests: `132 passed, 0 failed`
  - preprocessing tests after wiring 7 new feature groups into pipeline: `19 passed, 0 failed`
  - full suite after the player archetype feature-group implementation: `116 passed, 0 failed`
  - full suite after the small-preset implementation: `115 passed, 0 failed`
  - full suite after environment fix and torch-shim fix: `110 passed, 0 failed`
  - prior full suite after blend-contract fix: `106 passed, 4 failed` (4 torch import failures now resolved)
  - baseline audit run: `83 passed, 2 skipped`
  - post-fix targeted regression run: `12 passed`
  - scraper/input-health regression run: `29 passed`
  - preprocessing regression run after the rolling refactor: `14 passed`
  - train entrypoint / FeatureEngineer compatibility regression run: `20 passed, 1 warning`
  - full suite after the rolling refactor: `94 passed, 2 skipped`
  - blend-contract enforcement regression run: `21 passed` (model-manager + runtime-artifact tests)
- The venv has been rebuilt with Python 3.12 and all dependencies install cleanly. `torch`, `catboost`, `sklearn`, and all project modules import without error.
- Preflight failure modes for `train.py` are now live-verified: missing CSV produces `FileNotFoundError`, unwritable `models/` produces `RuntimeError`.

## What Currently Works

- Core repo structure is coherent enough to understand and extend.
- Historical data ingestion flow exists in `update_data.py` with multiple season-selection modes.
- Feature engineering is substantial, modular, and well-covered compared with other areas.
- Training internals for CatBoost/Transformer components are implemented and testable in isolation.
- Transformer validation and runtime inference now default to an eager path with a safe SDPA backend fallback; `torch.compile` is disabled by default for this model path.
- The active training path now persists per-target CatBoost runtime artifacts and validates the required `models/` contract before returning success.
- `train.py` now supports named presets from `config/default.yaml` and `src/training/presets.py`; the new `small` preset disables the Transformer, keeps all six targets, trims the feature stack to six high-value groups, and can optionally limit training to the two most recent seasons when `SEASON_ID` exists.
- Feature engineering now includes deterministic player archetype/similarity features that map sparse players to a stable style bucket using past-only rolling/context signals.
- `train.py` now preflights writable model/cache directories and required raw CSV inputs before doing expensive work, and it emits stage names so subprocess callers can tell where a failure occurred.
- `FeatureSchema` is explicitly exported from `src.utils.prediction_utils` and re-exported from `src.utils`, with a clean-process regression test guarding the import contract.
- `train.py` now routes feature-engineer construction through a compatibility-safe helper that backfills `disable_groups` and related filters for older checkouts that do not accept that keyword directly.
- The Step 2 ablation benchmark in `FeatureEngineer.benchmark_feature_variants()` now uses the same compatibility-safe helper, so older constructors that reject `disable_groups` no longer crash during the benchmark probe itself.
- The training pipeline now records `model_stack_metadata.pkl` with the selected preset and feature-group list when the CLI provides them; `ModelManager.validate_runtime_artifacts()` now treats that metadata file as part of the shared runtime contract.
- The `train.py` Step 2 path now has an entrypoint-level regression test that asserts the CLI uses `build_feature_engineer(...)` for ablation filters instead of passing compatibility-sensitive kwargs directly into `FeatureEngineer(...)`.
- `train_colab.ipynb` now resolves the repo checkout separately from Drive-backed data/models, captures full stdout/stderr from `train.py`, prints the return code, and fails immediately on nonzero exit instead of hiding the real traceback behind `CalledProcessError`.
- `ModelManager` now rejects incomplete runtime artifact sets instead of silently loading a partial model directory.
- `ModelManager._validate_blend_contract()` now raises `FileNotFoundError` or `RuntimeError` when blend weights require a Transformer model that is missing or failed to load, eliminating the `0.7 * cat_pred + 0` partial-blend bug.
- `TrainingPipeline._validate_blend_contract()` enforces the same contract on the pipeline's `load_models()` path.
- `TrainingPipeline._save_model_stack_metadata()` persists explicit metadata (`model_stack_metadata.pkl`) indicating whether the Transformer was active during training and the expected model count.
- `simulate_season.py` now uses `ModelManager.load_models()` for startup validation instead of hard-coding a single `pts_catboost.cbm` existence check.
- Rolling feature generation now batches wide feature columns before concatenating them back into the frame, which removed the prior pandas fragmentation warning flood.
- Seven new feature groups are implemented, import-tested, and **wired into the active pipeline**: `MinutesConfidenceFeatureGroup` (minutes variance, trend, starter rate, cold start), `RecencyFormFeatureGroup` (recent-vs-season deltas, form ratios, volatility, usage/efficiency/min deltas), `LineupStabilityFeatureGroup` (starter rate, teammate Jaccard continuity, rotation size variance, minutes rank), `RestGameDensityFeatureGroup` (schedule density, B2B detection, rest advantage, composite density score), `InjuryAdjustedOpportunityFeatureGroup` (missing high-usage teammate detection, same-minutes-position missing count, MIN/usage boost, team absence rolling count), `TeammateUsageFeatureGroup` (top-usage teammate active flag, missing teammate FGA/AST/REB shares, missing shot volume, active scoring depth), and `DefensePositionFeatureGroup` (opponent defensive stats allowed by position group — guard/wing/big — including PTS/REB/AST/STL/BLK/TOV allowed, defensive rank, recent PTS allowed). All follow the batched-column pattern, use `shift(1)` for leakage prevention, and are registered in `__init__.py`.
- `FeatureEngineer._build_groups()` now returns 19 groups (12 original + 7 new). The `full` training preset in both `config/default.yaml` and `src/training/presets.py` includes all 19 groups. The `small` preset is unchanged and does not include the new groups by default.
- `FeatureSelector.SAFE_PREFIXES` in `src/utils/prediction_utils.py` now includes 7 new prefixes (`MIN_CONF_`, `RECENCY_`, `LINEUP_`, `INJURY_OPP_`, `TEAMMATE_`, `DEF_POS_`, `SCHED_`) so the new feature columns pass the safe-column filter during training and inference.
- The query subsystem in `src/query/` is structurally complete and supported by tests.
- Many unit tests around preprocessing, model wrappers, and probability math are green.
- Cleanup tooling in `clear_cache.py` is clear and conservative about preserving raw data.
- The M tier transformer config now uses `seq_len=20` and `max_seq_length=20` (up from 10), matching the L tier's sequence length for better temporal context.
- Both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) now use zero-padding instead of skipping players with fewer than `seq_len + 1` games. Players with at least 1 game produce training samples with zero-padded context for early timesteps.

## What Partially Works

- Simulation stack is feature-rich and still depends on volatile third-party scrapers, but degraded optional inputs are now surfaced explicitly in per-game metadata and CLI output.
- `RotoWireLineupScraper` and `NBADefenseScraper` previously had undefined references and unreachable code that would have caused runtime `AttributeError`/`NameError`; these are now fixed and covered by tests.
- `ScheduleScraper.get_remaining_season()` previously hard-coded a 30-day window; it now iterates to the computed season end (capped at 180 days) with explicit TODOs about upstream limitations.
- The blend-weight/Transformer artifact contract is now enforced at load time, but existing model directories trained with a Transformer that later lose the `attention_transformer.pkl` artifact will fail loudly instead of silently degrading — which is the intended behavior.
- Query flow supports all six stats (PTS, REB, AST, STL, BLK, TOV) at the parser/calculator level, and projection exports are now complete for all six stats.
- The train-to-simulation artifact contract is covered by focused regression tests. A live `python train.py` smoke test was not completed due to training runtime length, but preflight checks and all 116 unit tests pass.
- The archetype feature group is implemented and test-covered, but a live benchmark against the current CSVs is still pending, so the real cold-start lift is not yet measured in this workspace.
- The new small preset is test-covered and loads successfully with CatBoost-only artifacts, but a live runtime measurement on the current CSVs is still pending.
- The Colab notebook launch path has been hardened in code, but a live run against mounted Drive data is still needed to confirm the exact traceback the user was seeing and to verify the full notebook-to-training flow with real CSV inputs and the repo-root detection flow.
 - The `disable_groups` constructor mismatch appears fixed in the active repo checkout; if Step 2 still fails in Colab, the most likely cause is a stale or mixed checkout rather than the current `train.py` call site.
 - Transformer validation inference has been repaired in code, but a live CUDA smoke test is still needed to confirm the exact GPU/runtime combination that previously crashed.
- Schedule scraping and season simulation features are implemented, and the previously confirmed config/state regressions in schedule/lineup/betting-support scrapers are now fixed in code.
- Rolling feature generation no longer emits the large pandas fragmentation warning flood; the hot feature groups now batch new columns before concatenating them back into the frame.

## New Bugs Found And Fixed During This Session

### Planned bug-fix batch (2026-04-25) — all verified / applied

- **GroupBy Series KeyError in `pace_role.py`** (plan: `bugfix_groupby_series_keyerror.md`):
  - The bug described computed pandas Series being passed directly to `groupby()[...]` as column selectors. Current code already stores these metrics as named `RAW_*` DataFrame columns and references them by string in `groupby('PLAYER_ID')['RAW_*']`, so the KeyError path is no longer present. No code change was required; the fix is live.
- **CatBoost GPU callback failure** (plan: `catboost_gpu_callback_fix.md`):
  - Current code already conditionally attaches callbacks only for CPU training (`use_callback = task_type == 'CPU'`), sets `verbose=200` for GPU, uses defensive `getattr(info, 'learn_error', None)` in `after_iteration()`, and rebuilds `fit_kwargs` from scratch on GPU→CPU fallback. No code change was required; the fix is live.
- **Device type AttributeError in `nn_trainer.py`** (plan: `fix_device_type_attribute_error.md`):
  - `BaseTrainer.__init__` already coerces `device` to `torch.device` and raises `TypeError` for unexpected types. `NeuralNetworkTrainer.__init__` already accepts `Optional[Union[str, torch.device]]`.
  - Applied an additional defensive check in `NeuralNetworkTrainer._create_loader()` before accessing `self.device.type`, ensuring a string value is coerced to `torch.device` on the spot.
  - All targeted tests pass: preprocessing (19 passed), nn_trainer (2 passed), runtime artifact contract (5 passed), transformer model (8 passed).

### Torch shim clobbered real PyTorch when pytest was running

- `src/__init__.py` installed a fake `torch` module whenever `pytest` was in `sys.modules` and `torch` was not yet loaded. In a healthy Python 3.12 environment with real PyTorch installed, this shim replaced the real package with a minimal NumPy-backed stand-in that lacked `torch.backends`, `torch.nn`, etc. This caused all 4 `test_transformer_model.py` tests to fail with `No module named 'torch.backends'; 'torch' is not a package`.
- Fixed by adding `importlib.util.find_spec('torch')` check before installing the shim. If real torch is importable, the shim is skipped.

### Legacy TransformerWrapper.load() failed on checkpoints with empty config

- `TransformerWrapper.load()` reconstructed the model using `DEFAULT_CONFIG` (d_model=128, num_layers=4) when the checkpoint stored `config: {}`, causing state_dict shape mismatches. This affected legacy three-output checkpoints and any checkpoint saved before the config field was populated.
- Fixed by adding `_infer_config_from_state()`, which extracts `d_model`, `num_layers`, `nhead`, and `dim_feedforward` from the checkpoint's state_dict tensor shapes before reconstructing the model.

## What Is Broken Or Very Likely Broken

### Scraper reliability still depends on upstream availability

- The concrete config/state regressions previously identified in `ScheduleScraper`, `LineupScraper`, and `BasketballRefScraper` are fixed.
- Live upstream changes can still degrade optional context inputs, but the simulator now reports those conditions explicitly instead of silently masking them.

### Projection export/query mismatch — FIXED

- `src/simulation/report_generator.py` now exports all 6 stat columns (PTS, REB, AST, STL, BLK, TOV) with MEAN/MODE/CI for each.
- `src/query/projection_loader.py::STAT_COLUMNS` now maps all 6 stats and `_row_to_projection()` uses this mapping uniformly.
- `src/query/interactive_cli.py::HELP_TEXT` now includes `tov/turnovers`.
- Regression tests in `tests/test_query/test_six_stat_contract.py` cover the full export/load contract.

## Current Limitations

- No checked-in raw data or trained models are present in this workspace beyond cache directories, so true end-to-end execution was not validated here.
- Network-reliant flows such as `update_data.py` and live scraping were not exercised in this audit environment.
- The `venv/` has been rebuilt with Python 3.12 and all dependencies install and import cleanly. The prior Python 3.13 mismatch has been resolved.
 - The repo uses many local file contracts rather than strong typed interfaces between phases.
- There is visible architecture drift between some legacy/alternate modules and the active top-level scripts.
- `simulate_season.py` now carries a run-level input health summary and exits non-zero for hard schedule failures, but this still needs a live smoke test against current upstream sources.

## Active Risks

- The main remaining scraper risk is upstream drift, not silent masking: optional failures now degrade runs visibly, and schedule failures are treated as hard-required.
- Artifact naming or schema drift is now guarded by training/runtime validation, but future changes can still break the contract if tests are not kept in sync.
- Blend-weight contract enforcement means a missing `attention_transformer.pkl` is now a hard failure when blend weights expect it. Operators who previously relied on silent CatBoost-only fallback will see load errors instead of degraded predictions.
- Scraper defects are under-tested and concentrated in modules that directly affect user-visible simulation output.
- The previously dominant fragmentation warning source in `src/preprocessing/features/rolling.py` has been addressed with batched column assembly, but no real-data production-scale benchmark was run in this workspace.

## Known Workarounds

- If lineup, injury, betting, or defense context scrapers fail, `GameSimulator` now continues in explicit degraded mode and records which sources fell back or failed.
- Query users can still ask for points, rebounds, and assists from exported projection CSVs with higher confidence than for steals, blocks, or turnovers.
- `clear_cache.py` can reset generated state while preserving raw input CSVs.

## Immediate Priorities

1. Benchmark `python train.py --preset small` against the full preset on live CSVs, now that archetype features are part of the default stack, to confirm the intended iteration-speed improvement and quantify any cold-start gain.
2. ~~Export `STL`, `BLK`, and `TOV` projection columns and align `ProjectionLoader` expectations with the report schema.~~ DONE — completed 2026-04-12 with regression tests.
3. Remove or reconcile dead simulation code in `src/simulation/game_simulator.py` to reduce maintenance ambiguity.
4. Run one live `train.py` -> `ModelManager.load_models()` -> `simulate_season.py` smoke test in a healthy local environment with real CSV inputs (preflight checks and unit tests confirmed; full training run deferred due to runtime length).
5. Decide whether to add a strict fail-fast mode for optional scraper degradation on top of the new default warn-and-continue behavior.
6. If a Transformer-trained model directory is deployed where the Transformer artifact may be unavailable, decide whether to retrain with the Transformer disabled or provide the artifact.
7. If needed, collect a real-data performance profile for the rolling feature path to compare against the synthetic benchmark.

## Testing Status

- Verified on 2026-04-01: `pytest tests/ -q` passed with `83 passed, 2 skipped`.
- Verified on 2026-04-01 after the artifact-contract fix:
  - `python3 -m pytest tests/test_training/ tests/test_models/test_model_manager.py -q`
  - Result: `12 passed`
- Verified on 2026-04-02 after the scraper/input-health hardening:
  - `venv/bin/python3.12 -m pytest tests/test_data/test_scraper_health.py tests/test_simulation/test_game_simulator.py tests/test_simulation/test_simulation_health_reporting.py -q`
  - Result: `29 passed`
- Verified on 2026-04-02 after the rolling feature refactor:
  - `venv/bin/python3.12 -m pytest tests/test_preprocessing/test_feature_engineer.py -q`
  - Result: `14 passed`
- Verified on 2026-04-02 after the rolling feature refactor:
  - `venv/bin/python3.12 -m pytest tests -q`
  - Result: `94 passed, 2 skipped`
- Verified on 2026-04-02 after the Transformer validation fix:
  - `venv/bin/python3.12 -m pytest tests/test_models/test_transformer_model.py tests/test_training/test_runtime_artifact_contract.py -q`
  - Result: `3 passed, 4 skipped`
- Verified on 2026-04-02 after the notebook/training launch hardening:
  - `python3 -m py_compile train.py`
  - `python3 -m json.tool train_colab.ipynb > /tmp/train_colab_validated.json`
  - `python3 -m pytest tests/test_training/test_training_pipeline_colab.py tests/test_training/test_runtime_artifact_contract.py tests/test_models/test_transformer_model.py -q`
  - Result: `5 passed, 4 skipped`
- Verified on 2026-04-02 after the FeatureSchema import-contract hardening:
  - `./venv/bin/python3.12 -m pytest tests/test_training/test_runtime_artifact_contract.py -q`
  - Result: `4 passed`
- Verified on 2026-04-11 after the blend-contract enforcement fix:
  - `./venv/bin/python3.12 -m pytest tests/test_models/test_model_manager.py tests/test_training/test_runtime_artifact_contract.py -q`
  - Result: `21 passed`
- Verified on 2026-04-11 full suite after the blend-contract fix:
  - `./venv/bin/python3.12 -m pytest tests/ -q`
  - Result: `106 passed, 4 failed` (4 failures are pre-existing torch import issues, unrelated to this change)
- Verified on 2026-04-11 full suite after venv rebuild (Python 3.12) and torch-shim fix:
  - `pytest tests/ -v`
  - Result: `110 passed, 0 failed`
  - Environment: Python 3.12.0, torch 2.6.0, catboost 1.2.8, sklearn 1.8.0
  - Data: `data/nba_players.csv` (52,583 records), `data/nba_games.csv` (4,882 records) fetched via `update_data.py`
- Verified on 2026-04-11 train.py preflight failure modes:
  - Missing `data/nba_players.csv`: exits with code 1, `FileNotFoundError: Missing required data files` at "preflight checks" stage
  - Unwritable `models/` directory (chmod 000): exits with code 1, `RuntimeError: models directory is not writable` at "preflight checks" stage
- Verified on 2026-04-03 after the FeatureEngineer compatibility hardening:
  - `python3 -m py_compile train.py src/preprocessing/feature_engineer.py`
  - `./venv/bin/python3.12 -m pytest tests/test_preprocessing/test_feature_engineer.py -q`
  - Result: `16 passed`
- Verified on 2026-04-03 after guarding the `train.py` entrypoint path:
  - `./venv/bin/python3.12 -m py_compile train.py src/preprocessing/feature_engineer.py tests/test_training/test_train_entrypoint.py`
  - `./venv/bin/python3.12 -m pytest tests/test_preprocessing/test_feature_engineer.py tests/test_training/test_training_pipeline_colab.py tests/test_training/test_train_entrypoint.py -q`
  - Result: `19 passed`
- Verified on 2026-04-11 after closing the ablation-benchmark compatibility gap:
  - `python -m py_compile train.py src/preprocessing/feature_engineer.py tests/test_training/test_train_entrypoint.py`
  - `pytest tests/test_preprocessing/test_feature_engineer.py tests/test_training/test_train_entrypoint.py -q`
  - Result: `20 passed, 1 warning`
- Verified on 2026-04-03 during the Colab-launch regression check in this workspace:
  - `./venv/bin/python3.12 -m pytest tests/test_training/test_training_pipeline_colab.py tests/test_training/test_runtime_artifact_contract.py tests/test_models/test_transformer_model.py -q`
  - Result: `6 passed, 4 skipped`
- Synthetic timing check on the rolling feature path with representative 2000-row input:
  - batched implementation: `1.554s`
  - reconstructed old insertion path: `1.543s`
  - result: roughly parity runtime while eliminating the fragmentation warnings from the new path
- Verified on 2026-04-12 after adding 6-stat export/load regression tests:
  - `pytest tests/ -v`
  - Result: `132 passed, 0 failed`
  - New tests in `tests/test_query/test_six_stat_contract.py` cover the full export/load contract for all 6 stats (PTS, REB, AST, STL, BLK, TOV)
- Verified on 2026-04-21 after teammate-utils extraction and vectorization refactor:
  - `pytest tests/ -v`
  - Result: `178 passed, 0 failed`
  - New tests: `TestTeammateUtils` (4 tests) and `TestPerformanceSmoke` (2 tests) in `tests/test_preprocessing/test_new_feature_groups.py`
  - Refactored: `src/preprocessing/features/_teammate_utils.py` (new), `lineup_stability.py`, `injury_opportunity.py`, `teammate_usage.py`, `rest_density.py`
- Two tests are skipped by design around Transformer runtime constraints.
- The earlier 1612-warning flood from pandas `PerformanceWarning` messages in `src/preprocessing/features/rolling.py` is no longer present in the post-refactor test runs.

## Areas That Need Confirmation In A Future Session

- Whether additional scraper modules beyond the identified ones have similar uppercase/lowercase config drift.
- Whether any local uncommitted changes in this worktree are part of an in-progress fix for the observed regressions.
- Whether a full `python train.py` smoke run completes and produces all required runtime artifacts when run with real CSV data (deferred due to training runtime length; unit tests and preflight checks confirmed).
- Whether the user-reported Colab crash reproduces after pulling a checkout that includes the current `train.py` helper path and the notebook repo-root fixes.
