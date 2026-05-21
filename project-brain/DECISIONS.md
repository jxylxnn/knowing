# Decisions

This file records confirmed or strongly inferred architectural decisions visible in the repository as of 2026-04-12.

When a decision is labeled "inferred", it means the repo shows a clear implementation choice but does not contain explicit historical rationale.

---

## DR-020: Batch-1 Feature Groups — Minutes Confidence, Recency Form, Lineup Stability, Rest Density

- Status: active
- Date: 2026-04-12
- Confidence: high

### Context

- The existing feature groups (rolling, efficiency, momentum, context, fatigue, matchup, opponent_strength, pace, team_role, archetype, target_encoding) cover core box-score signals but lack explicit schedule-density, minutes-confidence, recency-form, and lineup-stability features.
- These four feature categories were identified as high-value additions for predicting NBA player props because they capture role stability, schedule load, and recent-form signals that the existing groups do not.

### Options Considered

1. Add all four groups as separate `FeatureGroup` subclasses following the existing batched-column pattern.
2. Merge some of these signals into existing groups (e.g., rest density into fatigue).
3. Add them as a single monolithic group.

### Decision

- Option 1: four separate `FeatureGroup` subclasses, each in its own module under `src/preprocessing/features/`.
- This keeps each group independently testable, independently disableable via `FeatureContext.enabled_groups`/`disabled_groups`, and consistent with the existing architecture.

### Tradeoffs

- More files to maintain, but each file is small and single-purpose.
- `LineupStabilityFeatureGroup` and `RestGameDensityFeatureGroup` use row-level iteration for Jaccard similarity and game-count windows, which may be slower on very large datasets than vectorized alternatives. This is acceptable for the current data scale but should be revisited if performance becomes an issue.

### Consequences

- Four new modules: `minutes_confidence.py`, `recency_form.py`, `lineup_stability.py`, `rest_density.py`.
- All seven are registered in `src/preprocessing/features/__init__.py`.
- All seven are now wired into `FeatureEngineer._build_groups()`, the `full` training preset, and `FeatureSelector.SAFE_PREFIXES`; see CURRENT_STATE.md for details.
- No existing tests were broken by this addition.

---

## DR-001: Keep The Product CLI-First And Local

- Status: active
- Date: unknown, inferred from current codebase on 2026-04-01
- Confidence: high

### Context

- The repo exposes top-level scripts for all major workflows:
  - `update_data.py`
  - `train.py`
  - `simulate_season.py`
  - `query_prob.py`
  - `clear_cache.py`
- There is no server, API router, database layer, or frontend application.

### Options Considered

- Local CLI-first workflow.
- Hosted service or web app.
- Background-job or daemon-based pipeline.

### Decision

- The active product shape is a local CLI application, not a service.

### Why

- The codebase is optimized around batch scripts, file artifacts, and local iteration.
- This keeps the architecture simple and reduces operational surface area.

### Tradeoffs

- Easier local development and debugging.
- Harder to share results across users or expose functionality as an API.
- More dependence on file contracts and operator discipline.

### Consequences

- Script UX and file-path stability matter more than HTTP interface design.
- Documentation should explain script flows clearly.

### Revisit Triggers

- If the project adds multi-user workflows, remote execution, or an API consumer.

---

## DR-002: Use File-Based Persistence Instead Of A Database

- Status: active
- Date: unknown, inferred from current codebase on 2026-04-01
- Confidence: high

### Context

- The repo reads and writes CSV, JSON, joblib/pickle, and CatBoost artifacts.
- No database adapter, schema migration system, or ORM is present.

### Options Considered

- File-based persistence on local disk.
- Database-backed persistence.

### Decision

- Persist raw data, caches, projections, and model artifacts on the filesystem.

### Why

- The workflows are batch-oriented and local.
- CSV and model files are straightforward to inspect and move between runs.

### Tradeoffs

- Simple operational model.
- Weak schema enforcement across pipeline stages.
- Higher risk of artifact drift and partial-state bugs.

### Consequences

- Artifact names and directory layouts are architectural contracts.
- Testing should focus on verifying those contracts explicitly.

### Revisit Triggers

- If the project needs concurrent users, transactional updates, or remote-serving guarantees.

---

## DR-003: Train One CatBoost Model Per Target And Blend With A Transformer

- Status: active
- Date: unknown, partially confirmed by comments and implementation on 2026-04-01
- Confidence: high

### Context

- `src/training/pipeline.py` describes the active stack as:
  - CatBoost regressors per target
  - one Transformer sequence model
  - inverse-error blending
  - quantile models for uncertainty
- `src/models/model_manager.py` is built to load this hybrid artifact set.

### Options Considered

- Tree-only models.
- Sequence-only models.
- Hybrid stack with blending.

### Decision

- Use per-target CatBoost models as the core deterministic predictors and optionally blend them with a Transformer model.

### Why

- The implementation treats CatBoost as the primary per-target backbone while using the Transformer as an additional signal source.

### Tradeoffs

- Potentially better predictive flexibility.
- More artifact complexity.
- More end-to-end contract surface between training and inference.

### Consequences

- Training changes must preserve artifact compatibility with `ModelManager`.
- Inference behavior should stay tolerant of missing optional components but not silently hide required backbone artifacts.

### Revisit Triggers

- If the hybrid stack becomes too costly to maintain or backtesting shows no meaningful benefit from blending.

---

## DR-004: Persist Feature Schema Metadata To Align Training And Inference

- Status: active
- Date: unknown, inferred from current codebase on 2026-04-01
- Confidence: high

### Context

- `src/training/pipeline.py` writes `feature_schema.pkl` and `feature_cols.pkl`.
- `src/models/model_manager.py` and utility code depend on reconstructing the right feature layout at inference time.

### Options Considered

- Recompute feature expectations implicitly at runtime.
- Persist explicit schema metadata.

### Decision

- Persist training-time feature layout metadata and reload it during inference.

### Why

- The feature space is large and modular.
- Explicit schema persistence reduces accidental mismatch between training and simulation/query-time prediction.

### Tradeoffs

- Better stability across phases.
- Added artifact management complexity.

### Consequences

- Feature-column renames or removals are risky and must be coordinated.
- Tests should validate schema compatibility when feature groups change.

### Revisit Triggers

- If the project adopts a stronger typed feature-store approach or a simpler feature set.

---

## DR-005: Prefer Resilient Simulation Fallbacks Over Hard Failure On External Scraper Errors

- Status: active
- Date: unknown, inferred from current codebase on 2026-04-01
- Confidence: high

### Context

- `src/simulation/game_simulator.py` wraps several external inputs through safe helper methods and fallback defaults.
- This is especially visible around lineup and betting/injury context retrieval.

### Options Considered

- Fail fast when external context is unavailable.
- Continue simulation with default assumptions.

### Decision

- Continue simulation when external context scrapers fail, using fallback values when possible.

### Why

- External sports sites are volatile.
- A degraded result is often operationally preferable to no result in a CLI workflow.

### Tradeoffs

- Better runtime robustness.
- Higher risk of silently degraded prediction quality.
- Harder debugging because failures may not surface prominently.

### Consequences

- Logging and status reporting should make degraded-input runs visible.
- Tests should distinguish fallback behavior from successful live integrations.

### Revisit Triggers

- If silent degradation causes repeated trust issues or if higher-confidence data sources are added.

---

## DR-006: Use JSON For Simulation Cache Artifacts Where Practical

- Status: active
- Date: unknown, inferred from current codebase on 2026-04-01
- Confidence: medium

### Context

- `src/simulation/game_simulator.py` uses JSON-backed cache files in `data/sim_cache/`.

### Options Considered

- Pickle/joblib for arbitrary Python objects.
- JSON for simpler, safer portable cache structures.

### Decision

- Prefer JSON for the visible simulation cache layer.

### Why

- JSON is easier to inspect and safer than arbitrary object deserialization.

### Tradeoffs

- Simpler and safer.
- Less flexible for complex Python-native object graphs.

### Consequences

- Cache payloads should remain schema-simple and serializable.

### Revisit Triggers

- If simulation cache needs richer object persistence or stronger versioning guarantees.

---

## DR-007: Enforce The Training-To-Runtime Artifact Contract At Both Boundaries

- Status: active
- Date: 2026-04-01
- Confidence: high

### Context

- The repo’s main CLI journey depends on `train.py` writing artifacts that `ModelManager` and `simulate_season.py` can load later.
- A critical bug showed that training could appear successful while the required CatBoost runtime files were still missing from `models/`.
- This repo uses file-based persistence, so artifact names and presence are effectively API boundaries.

### Options Considered

- Allow training to save best-effort artifacts and let runtime discover missing pieces later.
- Validate only in `simulate_season.py` with a narrow single-file existence check.
- Validate the full runtime artifact set during training and again at runtime loading.

### Decision

- Validate the runtime artifact contract in `src/training/pipeline.py` before training reports success.
- Validate the same contract in `src/models/model_manager.py` before loading runtime models.
- Keep `simulate_season.py` dependent on `ModelManager.load_models()` instead of maintaining a separate hard-coded startup check.

### Why

- The broken path was a contract failure, not a modeling failure.
- Catching missing artifacts immediately after training prevents false-success runs.
- Reusing the same runtime validation boundary reduces drift between training, loading, and simulation startup.

### Tradeoffs

- Stricter failures can surface incomplete experimental runs that previously limped forward.
- The code is more opinionated about which artifacts are required.
- The validation logic must be updated whenever the runtime artifact contract changes.

### Consequences

- Training-side and runtime-side artifact names now need coordinated updates.
- Regression tests around artifact persistence and fresh-process loading are now required maintenance, not optional coverage.
- Partial model directories are treated as errors instead of tolerated ambiguous state.

### Revisit Triggers

- If the project intentionally changes the runtime artifact set or introduces a new inference loader boundary.

- If simulation caching needs richer structures or becomes a performance bottleneck.

---

## DR-008: Preserve Compatibility Shims And Lazy Imports Around Heavy Training Modules

- Status: active
- Date: unknown, inferred from current codebase on 2026-04-01
- Confidence: medium

### Context

- `src/pipeline/training_pipeline.py` re-exports `src.training.pipeline.TrainingPipeline`.
- `src/training/__init__.py` uses lazy imports to avoid torch-heavy import work at package import time.

### Options Considered

- Hard cutover to one import path.
- Compatibility shims and lazy-loading.

### Decision

- Keep lightweight compatibility layers and lazy import behavior where they reduce breakage or import cost.

### Why

- The repo shows evidence of evolution and import-path drift.
- These shims lower migration friction.

### Tradeoffs

- Easier incremental refactors.
- More architecture ambiguity if shims accumulate without documentation.

### Consequences

- Future cleanup should remove shims only when callers are updated and the active path is clearly documented.

### Revisit Triggers

- If compatibility layers start obscuring the true architecture or create duplicate logic.

---

## DR-009: Surface Degraded Optional Simulation Inputs Explicitly

- Status: active
- Date: 2026-04-02
- Confidence: high

### Context

- The simulator historically preferred fallback/default behavior when scraper-backed context failed.
- Several scraper modules had real runtime regressions, and the fallback layer made some broken runs look healthy.
- Schedule retrieval is required to select games, but betting, lineup, injury, and defense context are optional enrichments.

### Options Considered

- Fail the whole CLI run whenever any external context source degrades.
- Keep silent fallback behavior.
- Keep optional fallback behavior but surface degraded state clearly.

### Decision

- Keep warn-and-continue behavior for optional live context failures.
- Treat schedule retrieval as hard-required.
- Surface structured per-source health status in simulator metadata and CLI summaries so degraded runs cannot appear fully healthy.

### Why

- Optional context is valuable but upstream scraper volatility should not block all simulation output.
- Silent degradation created a reliability and trust problem that was harder to debug than explicit degraded mode.

### Tradeoffs

- Better transparency and faster debugging.
- More console/report noise during scraper instability.
- Still relies on fallback/default projections when optional sources drift.

### Consequences

- New simulator/reporting code must preserve the input-health contract.
- Tests should cover both degraded optional-input runs and hard schedule failures.
- A future strict mode can be added without changing the default product choice.

### Revisit Triggers

- If operators want fail-fast behavior for optional input degradation.
- If a more reliable upstream data provider replaces the current scraper set.

---

## DR-010: Batch Wide Feature-Group Column Assembly Before Concatenation

- Status: active
- Date: 2026-04-02
- Confidence: high

### Context

- `src/preprocessing/features/rolling.py` generates a wide set of rolling, efficiency, and momentum features.
- The previous implementation added many columns one at a time inside loops, which fragmented the DataFrame and produced thousands of pandas `PerformanceWarning` messages during test runs.

### Options Considered

- Keep repeated `df[col] = ...` assignments.
- Rebuild each feature group in a temporary structure and concatenate once.

### Decision

- Accumulate new feature columns in temporary dicts/Series and attach them with a single `pd.concat(axis=1)` per feature group.

### Why

- This preserves the same feature values and schema while avoiding the fragmentation penalty and warning flood.

### Tradeoffs

- Slightly more temporary memory during feature generation.
- Much lower fragmentation risk and cleaner test output.

### Consequences

- New wide feature groups should follow the same batched-assembly pattern.
- Performance changes in preprocessing should be measured against this style, not against repeated column insertion.

### Revisit Triggers

- If pandas internals change enough that the concat path is no longer the better default.
- If a future feature group has a narrow enough output that repeated insertion is clearly cheaper and still safe.

---

## DR-011: Keep Transformer Validation And Runtime Prediction On The Eager Path By Default

- Status: active
- Date: 2026-04-02
- Confidence: high

### Context

- A CUDA training run in this workspace completed Transformer fitting and saved `attention_transformer.pkl`, but failed during validation prediction in a compiled flash-attention path with `CUDA error: invalid configuration argument`.
- `src/models/transformer_model.py` now retains an eager base model for validation/runtime inference and only compiles when the caller explicitly allows it.

### Options Considered

- Leave `torch.compile` enabled by default and treat the crash as an environment-specific edge case.
- Disable compile for the Transformer and validate with the eager model path.
- Build a separate inference-only Transformer wrapper with a more complex dispatch layer.

### Decision

- Default Transformer validation/runtime prediction to the eager path, keep `torch.compile` behind an explicit safety flag, and prefer a math SDPA backend when backend controls are available.

### Why

- The compiled flash-attention path is not stable across the observed CUDA/runtime combination.
- The eager path still allows safe GPU acceleration through TF32, cuDNN benchmark, and BF16 autocast where supported.

### Tradeoffs

- Slightly lower peak throughput for the Transformer validation/runtime path.
- Much lower risk of a post-fit validation crash.

### Consequences

- Future compile re-enablement needs regression coverage on the exact target GPU/runtime combination.
- Training configs must continue to carry an eager-safe inference path.

### Revisit Triggers

- If a future runtime proves the compiled path stable across the supported CUDA matrix and there is a measurable performance win.

---

## DR-012: Notebook Training Launches Should Capture Subprocess Output And Preflight Inputs

- Status: active
- Date: 2026-04-02
- Confidence: high

### Context

- The Colab training notebook historically launched `train.py` with `subprocess.run(..., check=True)` and did not print captured stdout/stderr.
- That made a real training failure indistinguishable from a wrapper-level `CalledProcessError`.

### Options Considered

- Keep the simple `check=True` launch and let notebook exceptions surface without captured logs.
- Add a separate debug notebook cell that prints logs only when manually enabled.
- Capture stdout/stderr and print them every time, alongside basic input/writability preflight checks.

### Decision

- The notebook launch should always capture subprocess output, print the return code, and fail fast after printing stdout/stderr.
- The notebook should also preflight the required raw CSV inputs and the output directory before launching training.

### Why

- Training failures are only actionable when the underlying traceback is visible.
- The notebook is part of the operator workflow, so it should optimize for immediate diagnosis rather than terse subprocess behavior.

### Tradeoffs

- Slightly more verbose notebook output.
- A little more preflight logic in the wrapper.
- Much better debugging signal when training fails.

### Consequences

- Future notebook wrappers that invoke the CLI should follow the same capture-and-print pattern.
- `train.py` stage logging becomes more valuable because the notebook now preserves it.

### Revisit Triggers

- If notebook output becomes too noisy for normal use or if the workflow is replaced by a different launcher.

---

## DR-013: Colab Training Launchers Must Resolve Code Checkout Independently From Drive Storage

- Status: active
- Date: 2026-04-02
- Confidence: high

### Context

- The Colab training notebook originally derived `project_root` from the Drive models directory, which only works when code and persisted artifacts share the same filesystem root.
- In the real Colab flow, code may live under a working checkout such as `/content/knowing` while data and models remain on Drive.

### Options Considered

- Keep the old assumption that Drive-backed models storage and code live under the same root.
- Require a single explicit project root and force both code and artifacts to live there.
- Resolve the repo checkout independently from Drive-backed storage, with an explicit override and common fallback candidates.

### Decision

- The notebook should resolve `train.py` from the actual repo checkout and treat Drive paths as data/model storage only.
- An explicit `project_root_override` should win when provided; otherwise the launcher may search common Colab checkout locations.

### Why

- Training needs the real code checkout, not just the artifact directory.
- The old root inference failed before `train.py` could run whenever the notebook used a different checkout directory.

### Tradeoffs

- Slightly more launcher logic.
- A small amount of extra diagnostic output.
- Much less ambiguity between code location and storage location.

### Consequences

- Notebook launchers should not derive script paths from model/data storage roots.
- Colab workflows can keep Drive-backed inputs and outputs while still using a normal repo checkout.

### Revisit Triggers

- If the notebook is replaced by a different launcher pattern or if the repo adopts a single canonical Colab mount location.

---

## DR-014: Keep `FeatureSchema` Explicitly Exported From The Utility Layer

- Status: active
- Date: 2026-04-02
- Confidence: medium

### Context

- `src/training/pipeline.py` and related loaders depend on `FeatureSchema` as part of the training/inference feature-layout contract.
- The utility module already defined `FeatureSchema`, but the export contract was not explicit enough for a repo that relies on stable import paths.

### Options Considered

- Leave the class defined only in `src.utils.prediction_utils` and rely on implicit module attributes.
- Move the class elsewhere and update all import sites.
- Keep the canonical definition in `src.utils.prediction_utils` and re-export it from `src.utils` for compatibility.

### Decision

- Keep `FeatureSchema` defined in `src.utils.prediction_utils` and make the export contract explicit with `__all__` plus a package-level re-export from `src.utils`.

### Why

- The repository depends heavily on stable module/file contracts.
- An explicit export is easier to verify and less likely to regress than an implicit attribute.

### Tradeoffs

- Slightly more explicit module surface area.
- Better compatibility for both direct module imports and package-level imports.

### Consequences

- Future refactors that move `FeatureSchema` must preserve the re-export or update all callers in the same change.
- Regression tests should validate the import contract in a clean process, not just by in-process attribute access.

### Revisit Triggers

- If the feature schema ownership changes to a dedicated shared-contract module or package.

---

## DR-015: Wrap FeatureEngineer Construction In A Compatibility-Safe Helper

- Status: active
- Date: 2026-04-03
- Confidence: high

### Context

- `train.py` needs to disable feature groups dynamically during feature ablation.
- Runtime checkouts may differ: current code accepts `disable_groups`, but older or divergent checkouts may not.

### Options Considered

- Keep constructing `FeatureEngineer` directly from `train.py` and accept runtime failures in older checkouts.
- Remove the ablation-time `disable_groups` flow.
- Add a compatibility-safe builder that uses supported kwargs and backfills the filter as attributes when needed.

### Decision

- Use a compatibility-safe `build_feature_engineer(...)` helper as the canonical Step 2 construction path in `train.py`.

### Why

- This preserves the existing feature-group disabling behavior while making the training CLI tolerant of older constructor signatures.

### Tradeoffs

- Slightly more indirection in the training script.
- Better protection against version skew between top-level scripts and feature-engineering internals.

### Consequences

- Future constructor changes in `FeatureEngineer` should be reflected in the helper and its regression test.
- `train.py` remains the single canonical orchestration path for feature-engineer startup.

### Revisit Triggers

- If feature ablation is removed or the feature-engineering constructor is replaced with a different factory pattern.

---

## DR-016: Enforce The Blend-Weight / Transformer Artifact Contract At Load Time

- Status: active
- Date: 2026-04-11
- Confidence: high

### Context

- The hybrid prediction model blends CatBoost and Transformer predictions using `blend_weights.pkl`, which is computed during training under the assumption that both models contribute.
- When `attention_transformer.pkl` was missing at runtime, `ModelManager` silently fell back to CatBoost-only predictions, effectively computing `0.7 * cat_pred + 0` instead of `0.7 * cat_pred + 0.3 * trans_pred`. This produced systematically biased, uncalibrated results without any operator warning.
- DR-005 (resilient fallbacks) and DR-007 (artifact contract enforcement) were in tension here: DR-005 prefers resilience, but DR-007 prefers contract correctness. The partial-blend bug demonstrated that silent fallback in the model stack is not mathematically neutral.

### Options Considered

- Option A: Treat `attention_transformer.pkl` as required when blend weights expect it, and fail loudly on missing or corrupt artifacts.
- Option B: Revert to pure CatBoost predictions (bypassing blend weights entirely), mark the run as degraded, and log a visible warning.

### Decision

- Adopt Option A: enforce the contract. If blend weights contain a non-zero Transformer weight, the Transformer artifact must be present and loadable. Otherwise, `ModelManager.load_models()` and `TrainingPipeline.load_models()` raise a descriptive `FileNotFoundError` or `RuntimeError`.

### Why

- Predictions are only valid if the math matches the training setup. Returning confidently wrong numbers is worse than failing.
- Option B would preserve resilient fallback behavior but adds complexity in weight re-normalization and still produces a fundamentally different prediction than what was trained.
- The contract is now explicit: `model_stack_metadata.pkl` records whether the Transformer was active during training.

### Tradeoffs

- Harder failure surface: deployed model directories that lose a Transformer artifact will now error instead of producing degraded output.
- Better mathematical correctness: the partial-blend bug is eliminated.
- Simpler code: no dynamic weight re-normalization or degraded-mode tracking is needed.

### Consequences

- Operators must ensure `attention_transformer.pkl` is present when blend weights expect it, or retrain with the Transformer disabled.
- The blend-weight contract is validated at both training and runtime loading boundaries.
- `model_stack_metadata.pkl` is now part of the training output artifact set.

### Revisit Triggers

- If the project needs a CatBoost-only degraded mode for deployed environments where Transformer artifacts are unreliable.
- If the hybrid model stack is simplified to CatBoost-only and the Transformer path is removed entirely.

---

## DR-017: Only Install The Torch Shim When Real PyTorch Is Not Importable

- Status: active
- Date: 2026-04-11
- Confidence: high

### Context

- `src/__init__.py` contained a `_install_test_torch_shim()` that installed a minimal NumPy-backed fake `torch` module when pytest was running and `torch` was not yet in `sys.modules`.
- In a healthy Python 3.12 environment with real PyTorch installed, this shim was installed before any explicit `import torch`, replacing the real package and breaking all transformer tests.

### Options Considered

- Keep the unconditional shim and require explicit `import torch` before importing `src`.
- Remove the shim entirely.
- Check whether real torch is importable before installing the shim.

### Decision

- Use `importlib.util.find_spec('torch')` to detect whether real PyTorch is available. Only install the shim when real torch cannot be found.

### Why

- The shim was designed for broken environments where torch crashes on import. It should not interfere with working environments.
- `find_spec` is lightweight and does not actually import the module, so it does not trigger the crashes the shim was designed to avoid.

### Tradeoffs

- Environments where torch is installed but crashes at import time (the original use case for the shim) will no longer get the fake module. Tests in those environments will need a different fallback or should skip.

### Consequences

- The shim is now opt-in for broken environments rather than opt-out for healthy ones.
- Environments with working torch will always use real torch, even during test runs.

### Revisit Triggers

- If a new environment pattern emerges where torch is installed but crashes at import, and the shim is still needed for that case.

---

## DR-018: Infer Transformer Architecture From State Dict When Config Is Missing

- Status: active
- Date: 2026-04-11
- Confidence: high

### Context

- `TransformerWrapper.load()` reconstructed models using `DEFAULT_CONFIG` when the checkpoint stored an empty config dict. This caused state_dict shape mismatches for checkpoints saved with non-default architectures (e.g., d_model=16 instead of 128).
- Legacy checkpoints and the test suite both triggered this failure.

### Options Considered

- Load with `strict=False` and ignore shape mismatches.
- Require all future checkpoints to store complete config.
- Infer architecture from the state_dict tensor shapes.

### Decision

- Add `_infer_config_from_state()` that extracts `d_model`, `num_layers`, `nhead`, and `dim_feedforward` from the saved state_dict. Use inferred config when the checkpoint config is empty.

### Why

- The state_dict contains sufficient information to reconstruct the correct architecture. Loading with wrong shapes is always a failure; requiring complete config breaks backward compatibility.

### Tradeoffs

- More code in the load path.
- nhead inference is a heuristic (tries common divisors) and may not match the exact value used during training, though any valid divisor produces a working model.

### Consequences

- Legacy checkpoints with empty config now load correctly.
- New checkpoints should store complete config for exact reconstruction, but the inference fallback provides safety.

### Revisit Triggers

- If a model is trained with an unusual nhead value that the heuristic cannot infer correctly.

---

## DR-019: Keep The Small Training Preset CatBoost-First Without Changing Runtime Artifacts

- Status: active
- Date: 2026-04-11
- Confidence: high

### Context

- The repo already treats CatBoost as the core per-target backbone and the Transformer as an optional blended component.
- The new `small` preset in `src/training/presets.py` was added to make iteration faster without changing the downstream artifact names, canonical target list, or simulator/query load path.
- A recent-history trim was requested as an optional speedup, but the codebase did not have a general season-window abstraction. The current training data does include `SEASON_ID`, so the preset can trim safely when that field exists.

### Options Considered

- Redesign the training architecture around a smaller model family.
- Change artifact filenames for the smaller preset.
- Add a named preset layer that only changes stack breadth and training window behavior.

### Decision

- Add named presets in `config/default.yaml` and `src/training/presets.py`, wire `train.py` to resolve them, keep the six canonical targets, and keep the runtime artifact contract unchanged.
- Use `SEASON_ID` for the recent-history trim when available; if it is missing, skip the trim instead of inventing a heuristic fallback.

### Why

- This preserves the existing simulator/query contract and keeps the training path aligned with the current file-based architecture.
- The smaller preset can now be used regularly without introducing a separate model family or a second runtime loader path.

### Tradeoffs

- Faster iteration with less code churn.
- The preset layer adds another moving part that must stay synchronized across config, CLI, and tests.
- The recent-history trim is intentionally conservative and may do nothing in older datasets without `SEASON_ID`.

### Consequences

- `model_stack_metadata.pkl` now records the selected preset and feature groups when available.
- Preset changes must be coordinated across `config/default.yaml`, `src/training/presets.py`, `train.py`, and the preset-focused tests.

### Revisit Triggers

- If live training benchmarks show the small preset is still too slow for routine iteration.
- If future data sources lose `SEASON_ID` and the project needs a more general recent-history abstraction.

---

## DR-020: Use Deterministic Playstyle Templates For Player Archetype Features

- Status: active
- Date: 2026-04-11
- Confidence: high

### Context

- The new player-archetype feature family needs to help with cold-start and low-sample players without introducing another learned-model artifact that would have to be trained, saved, and reloaded separately.
- The active preprocessing stack already has stable rolling and role features that can describe a player with enough history to compare them against a few fixed style buckets.
- The training/runtime contract in this repo is file- and schema-driven, so adding another learned artifact would increase the risk of mismatched persistence or runtime load failures.

### Options Considered

- Fit a separate clustering model and persist the centroids as another training artifact.
- Use ad hoc rule-based labels in `train.py` or `ModelManager`.
- Build a dedicated feature group that computes deterministic similarities against a fixed template set.

### Decision

- Implement player archetype features in `src/preprocessing/features/archetype.py` as a deterministic `FeatureGroup` that emits hard labels and soft similarities against fixed playstyle templates.

### Why

- This keeps the feature logic in the existing preprocessing boundary.
- Training and inference will always use the same templates as long as they run the same code revision, without requiring a separate learned clustering artifact to be persisted or loaded.
- The soft similarity scores are more flexible than a single hard cluster assignment for sparse players.

### Tradeoffs

- Simpler runtime contract and no extra model artifact.
- Less expressive than a fully learned clustering pipeline.
- Template tuning now lives in code, so changes to archetype definitions require a conscious schema/preset update instead of being learned automatically from data.

### Consequences

- `FeatureSelector` must keep the archetype outputs in the safe engineered-feature set.
- Preset definitions now need to include the archetype group explicitly so the default stack remains aligned with the feature contract.
- If the archetype template set changes materially, the feature schema version and tests should be updated together.

### Revisit Triggers

- If live benchmarking shows the deterministic templates are too coarse for the desired cold-start lift.
- If the repo later gains a proper persisted feature-store or clustering artifact boundary that can absorb a learned archetype model cleanly.

---

## DR-021: Increase M Tier Transformer seq_len to 20 and Zero-Pad Short Player Sequences

- Status: active
- Date: 2026-04-14
- Confidence: high

### Context

- The M tier transformer config used `seq_len=10` and `max_seq_length=10`, while the L tier already used 20/20 and the XL tier used 50/50. The M tier's shorter window limited the temporal context available to the model for medium-sized hardware.
- Both sequence builders (`TransformerWrapper._create_sequences()` and `TrainingPipeline._build_sequence_batch()`) skipped players with fewer than `seq_len + 1` games entirely, discarding potentially useful training signal from short-career players.

### Options Considered

1. Keep seq_len=10 for M tier and continue skipping short players.
2. Increase seq_len to 20 for M tier and continue skipping short players.
3. Increase seq_len to 20 for M tier and add zero-padding for short players so they still contribute training samples.

### Decision

- Option 3: increase M tier `seq_len` and `max_seq_length` from 10 to 20, and add zero-padding for players with fewer than `seq_len` context games so they still produce training samples.

### Why

- A longer sequence window gives the M tier more temporal context, matching the L tier's window on hardware that can support it.
- Zero-padding preserves training signal from short-career players instead of discarding them. The model can learn to attend to the padding mask implicitly.
- The zero-padding approach is standard in transformer sequence modeling and preserves the sliding-window behavior for players with enough games.

### Tradeoffs

- More training samples per epoch (short players now contribute), which increases training time slightly.
- Zero-padded sequences may introduce noise early in training until the model learns to down-weight padded positions.
- The M tier model size (d_model=128, 3 layers) is unchanged, so the longer sequence increases memory per batch slightly.

### Consequences

- `SIZE_TIER_SPECS['M']['transformer']['seq_len']` and `max_seq_length` are now 20.
- `_create_sequences()` and `_build_sequence_batch()` now produce sequences for all players with at least 1 game, zero-padding the beginning of short sequences.
- Players with more than `seq_len + 1` games still produce the same sliding-window samples as before (no regression).
- New tests in `tests/test_models/test_transformer_model.py` verify the config change, zero-padding behavior, and backward compatibility.

### Revisit Triggers

- If zero-padded sequences degrade validation MAE on real training data.
- If the M tier training time becomes prohibitive due to the increased sample count.

---

## DR-023: Scraper Reliability Fixes — rotowire_lineup, nba_defense, schedule

- Status: active
- Date: 2026-04-21
- Confidence: high

### Context

- A project audit identified multiple confirmed or suspected scraper reliability bugs:
  1. `rotowire_lineup_scraper.py` had unreachable code after a `return`, uppercase attribute references that didn't match lowercase instance attributes, and an undefined `ROTONAME_TO_TEAM` constant.
  2. `nba_defense_scraper.py` referenced `TEAM_ID_MAP` and `ID_TO_TEAM` without importing them, and `DefensiveMatchupAnalyzer` referenced `self._session`/`self.HEADERS` which it did not define.
  3. `schedule_scraper.py::get_remaining_season()` was a stub that only fetched the next 30 days instead of the actual remaining season.

### Options Considered

1. Delete the legacy scraper modules and rely solely on the actively maintained `LineupScraper`/`ScheduleScraper` paths.
2. Fix the bugs in place, add regression tests, and document upstream limitations.
3. Leave the bugs open and add runtime guards that skip the broken modules.

### Decision

- Option 2: fix in place. These modules are still imported by downstream code and by the `__main__` smoke-test blocks, so deleting them would create import breakage. The fixes are small and surgical.

### Tradeoffs

- `rotowire_lineup_scraper.py` remains a secondary/legacy path, but it now compiles and its constructor is safe.
- `nba_defense_scraper.py` still depends on upstream NBA.com Stats API rate limits, but undefined-name crashes are eliminated.
- `schedule_scraper.py::get_remaining_season()` still composes daily scoreboard calls because nba_api does not expose a single "remaining games" endpoint; the 180-day cap prevents runaway API usage.

### Consequences

- `src/data/rotowire_lineup_scraper.py`: `self._cache_timestamp` moved to `__init__`, dead code removed, uppercase references normalized to lowercase instance attributes, `ROTONAME_TO_TEAM` replaced with `normalize_team()`.
- `src/data/nba_defense_scraper.py`: imports `ABBR_TO_ID` and `ID_TO_ABBR` from `src.utils.team_mappings`, replaces all `TEAM_ID_MAP`/`ID_TO_TEAM` references, and initializes `DefensiveMatchupAnalyzer` with `max_retries`, `retry_delay`, `headers`, plus uses `defense_scraper._session` for HTTP.
- `src/data/schedule_scraper.py`: `get_remaining_season()` now iterates to the computed season end (June 30) with a 180-day cap, explicit TODO, and logged warnings.
- `tests/test_data/test_scraper_health.py`: 4 new regression tests added; full suite now `140 passed, 0 failed`.

### Revisit Triggers

- If nba_api adds a single "remaining games" endpoint, replace the composed daily fetches.
- If `rotowire_lineup_scraper.py` or `nba_defense_scraper.py` are fully superseded by active paths, consider deprecating or removing them.

---

## DR-024: Extract Shared Teammate/Roster Precomputations Into `_teammate_utils.py`

- Status: active
- Date: 2026-04-21
- Confidence: high

### Context

- `lineup_stability`, `injury_opportunity`, and `teammate_usage` each independently built `(TEAM_ID, GAME_DATE) → roster` mappings, `regular_teammates` maps, and `high_usage_teammates` maps.  This was wasteful (O(n) work repeated 3×) and risked subtle semantic drift if thresholds or window sizes diverged.
- `RestGameDensityFeatureGroup` used nested Python loops over player games to count games in X-day windows, which was O(n²) in the worst case.
- `LineupStabilityFeatureGroup` computed Jaccard similarity with per-player loops, also O(n²) in pathological cases.

### Options Considered

1. Keep the duplicated logic and accept the redundancy.
2. Extract a shared utility module (`_teammate_utils.py`) with pure functions and a `TeammateContext` container.
3. Move the maps into `FeatureContext` in `base.py`.

### Decision

- Option 2: create `src/preprocessing/features/_teammate_utils.py` with:
  - `build_game_roster_map`
  - `build_team_games_map`
  - `build_regular_teammates_map`
  - `build_high_usage_teammates_map`
  - `build_team_totals_map`
  - `TeammateContext` class that bundles the above for easy reuse.
- Refactor `lineup_stability.py`, `injury_opportunity.py`, and `teammate_usage.py` to import and use `TeammateContext` instead of recomputing internally.
- Vectorise `RestGameDensityFeatureGroup` game-count windows with pandas `rolling(..., closed='left')` on a temporary unit series.
- Replace `LineupStabilityFeatureGroup` per-player Jaccard loops with a vectorised key-shift approach: shift the `(TEAM_ID, GAME_DATE)` tuple within each player group, then compute Jaccard in a single list comprehension over all rows.
- Replace `rest_density.py` opponent-rest nested loops with `np.searchsorted` on pre-sorted `datetime64[ns]` arrays per team.

### Why

- Centralising roster logic guarantees consistency across feature groups and makes the maps cheap to reuse.
- Vectorised pandas/numpy operations remove the dominant O(n²) Python loops without changing feature semantics.
- The `_teammate_utils.py` boundary is private (underscore prefix) so it can evolve without affecting public feature-group contracts.

### Tradeoffs

- Slightly more indirection for readers of the individual feature groups.
- The vectorised `rest_density` path uses a temporary `tmp` DataFrame; memory cost is negligible for current data scale.
- `np.searchsorted` requires homogeneous `datetime64[ns]` arrays; a dtype mismatch (e.g. object array of `Timestamp`s) will raise, so the code explicitly casts.

### Consequences

- `lineup_stability.py`, `injury_opportunity.py`, and `teammate_usage.py` are now ~30–50 % shorter and no longer contain duplicated roster-building logic.
- `rest_density.py` game-count loops are gone; performance on synthetic 500-row DataFrames dropped from >2 s to <0.2 s.
- `lineup_stability.py` Jaccard computation no longer uses per-player Python loops.
- Full test suite after refactor: `178 passed, 0 failed`.
- New tests added: `TestTeammateUtils` (4 tests) and `TestPerformanceSmoke` (2 tests) in `tests/test_preprocessing/test_new_feature_groups.py`.

### Revisit Triggers

- If additional feature groups need roster context, evaluate whether `_teammate_utils.py` should be promoted to a public module.
- If data scale grows to the point where even the vectorised paths become bottlenecks, consider numba or polars.

---

## DR-020: Refactor GameSimulator Into Typed, Modular Components

- Status: active
- Date: 2026-05-09
- Confidence: high

### Context

- `game_simulator.py` was a monolithic 1900+ line file with dead legacy code, raw dict data flow, and tightly coupled simulation logic.
- The simulation layer needed to be split into independently testable, typed components.

### Options Considered

1. Leave as monolithic file with minor cleanup.
2. Extract into typed dataclasses + separate modules for phase simulation, archetype inference, and role sampling.
3. Full rewrite with a new simulation engine.

### Decision

- Option 2: extract simulation logic into modular, typed components while preserving the existing reactive simulation algorithm.
- `game_simulator.py` retains orchestration but delegates to PhaseSimulator, ArchetypeEngine, and RoleSampler.
- All inter-component data flows through typed dataclasses defined in `sim_types.py`.

### Tradeoffs

- More files to track, but each is focused and testable in isolation.
- Typed dataclasses add boilerplate but eliminate key-typo bugs and make contracts explicit.
- Phase simulator extraction means changing simulation logic now only requires touching `phase_simulator.py`.

### Consequences

- Six new simulation modules: `phase_simulator.py`, `archetype.py`, `role_sampler.py`, `sim_types.py`, `sim_cache.py`, `stat_utils.py`.
- Dead legacy simulation code removed from `game_simulator.py`.
- Full test suite: `178 passed, 0 failed`.

### Revisit Triggers

- If the phase-simulation algorithm needs fundamental changes, the extraction should make that simpler.

---

## DR-021: Build Self-Optimizing Ensemble Weight System

- Status: active
- Date: 2026-05-09
- Confidence: high

### Context

- The current system computed ensemble blend weights once at training time and froze them forever in `blend_weights.pkl`.
- There were three frozen weight points: CatBoost-MAE split (hardcoded 0.7/0.3), CatBoost-Transformer blend (loaded once), and per-target blend weights (training-time only).
- New games, shifting player roles, and model accuracy drift could not be adapted to without full retraining.

### Options Considered

1. Keep frozen training-time weights; manually retrain when accuracy degrades.
2. Build a cron-based weight retuning system triggered by drift detection.
3. Build an on-demand self-optimization system: backtest → optimize → verify → deploy.

### Decision

- Option 3: on-demand self-optimization with human-in-the-loop accept/verify gates.
- `backtest.py` evaluates model accuracy on recent completed games.
- `optimize_weights.py` uses scipy.optimize to find better blend coefficients.
- `WeightStore` provides versioned, atomic JSON storage with rollback.
- `DriftDetector` monitors performance and alerts when retuning may help.
- All blend coefficients now come from a single `EnsembleWeights` config object, hot-reloadable at runtime.

### Tradeoffs

- Adds complexity (new entry points, evaluation subsystem, versioned storage).
- Requires operator to run backtest/optimize periodically — not fully automated.
- Human-in-the-loop accept gate prevents bad automated weight changes.
- Versioned JSON storage is simpler and safer than opaque binary pickle.

### Consequences

- New entry points: `backtest.py`, `optimize_weights.py`.
- New subsystem: `src/evaluation/` (5 modules: metrics, backtest_runner, ensemble_optimizer, weight_store, drift_detector).
- `ModelManager` refactored to accept hot-reloadable `EnsembleWeights`.
- `config/default.yaml` now supports `self_optimization:` section.
- Full test suite: `178 passed, 0 failed`.
- The old `blend_weights.pkl` binary format is superseded by the versioned JSON store.

### Revisit Triggers

- If the self-optimization loop proves the system can converge reliably, consider adding a cron-based auto-retune mode.
- If 13 parameters are too many to optimize reliably, reduce the parameter space (e.g., per-target intercepts only).
- If the weight store needs to be shared across users (service deployment), consider a database-backed store.

---

## DR-025: Enforce Strict Mode and Data Quality Schema for Simulator Degradation

- Status: active
- Date: 2026-05-22
- Confidence: high

### Context

- DR-005 and DR-009 established resilient fallbacks and visible degradation summaries for optional scrapers, but the query layer remained blind to which specific player projections relied on fallback math.
- Advanced operators lacked a fail-fast mechanism for production-grade runs.

### Options Considered

1. Add a `--strict` flag to `simulate_season.py` that halts on any degraded optional source.
2. Append a `DATA_QUALITY` column to the export schema so the query layer can surface degradation.
3. Both (strict mode for CLI + data quality for exports).

### Decision

- Option 3: both fail-fast (strict mode) and persistent quality tracking (data quality column).
- `GameSimulator.__init__` accepts `strict_mode: bool = False`. When True, raises `RuntimeError` if any optional `InputHealth` source reports `failed` or `fallback`.
- `SeasonSimulator` threads `strict_mode` through to the game simulator.
- `ReportGenerator.export_player_projections()` appends `DATA_QUALITY` (`FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`).
- `ProjectionLoader.find_player()` reads `DATA_QUALITY` and emits a visible warning during interactive queries.

### Tradeoffs

- Strict mode trades resilience for safety — operators must decide which matters more for their run.
- Data quality column adds export schema surface area but requires no new pipeline state.

### Consequences

- `simulate_season.py --strict` exits with `RuntimeError` on first degraded optional source.
- `player_projections_<timestamp>.csv` includes a `DATA_QUALITY` column.
- Projection queries show warnings when using degraded data.
- Full test suite: 178 passed, 0 failed (plus new strict mode tests).

### Revisit Triggers

- If the data quality schema needs to carry more granular information (e.g., per-source quality breakdown per player), extend `DATA_QUALITY` to a composite enum or structured field.
- If strict mode proves too brittle for production use, consider adding a `--warn-only` variant.

---

## DR-026: Season-Context Feature Groups for Temporal Awareness

- Status: active
- Date: 2026-05-23
- Confidence: high

### Context

- The model treated all regular-season and playoff games identically, failing to account for off-season rest gaps, trade deadline resets, late-season tanking/load management, and playoff pace/minute shifts.
- This caused predictable distribution errors in October (players returning from off-season), February (traded players), March (tanking), and May (playoff intensity shift).

### Options Considered

1. Single feature group with all season-context signals.
2. Separate feature groups for early-season, late-season motivation, and postseason context — keeping each focused and independently testable.
3. Hard-coded adjustments in the simulator only (no feature-level signal).

### Decision

- Option 2: three separate feature groups:
  - `SeasonPhaseFeatureGroup` — early-season ramp-up (days since season start capped at 30) and trade-reset tracking (games with current team, recent trade flag).
  - `TeamMotivationFeatureGroup` — late-season tanking/load management signals using team cumulative win percentage as a proxy.
  - `PostseasonContextFeatureGroup` — playoff detection with a 0.95 pace prior (the model learns the exact coefficient).

### Tradeoffs

- Three small files vs one larger file — each is independently testable and follows the batched-assembly pattern.
- Team motivation uses win percentage as a proxy, not actual front-office intent (which is unobservable).
- Playoff pace prior is a fixed starting point (0.95) — the model can scale or ignore it.

### Consequences

- Three new feature files, 10 new output columns total.
- `DAYS_SINCE_LAST_GAME` capped at 14 days in `RestGameDensityFeatureGroup` (the rest cap applies to all games, not just off-season).
- `DriftDetector` updated with phase-aware baselines to prevent false playoff drift alerts.
- `full` preset includes all three groups; `small` does not.
- Full test suite: `279 passed, 1 skipped`.

### Revisit Triggers

- If actual team injury/rest data becomes available, the tanking/load-management proxy can be replaced with real load-management tracking.
- If the 30-day cap on `DAYS_SINCE_SEASON_START` proves too tight or too loose for the early-season effect, adjust the cap based on empirical validation.
