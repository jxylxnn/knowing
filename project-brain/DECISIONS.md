# Decisions

This file records confirmed or strongly inferred architectural decisions visible in the repository as of 2026-04-12.

When a decision is labeled "inferred", it means the repo shows a clear implementation choice but does not contain explicit historical rationale.

> **Numbering note (2026-06-04)**: DR-020 and DR-021 are each used twice in this file (DR-020 appears once for the Batch-1 feature groups and again for the deterministic playstyle templates; DR-021 appears once for the Transformer seq_len/zero-padding work and again for the self-optimizing ensemble weight system). DR-022 is missing. The duplicates and the gap were preserved rather than renumbered to avoid churning cross-references in `CODE_RULES.md`, `ARCHITECTURE.md`, and the existing TASK log. New decisions should pick the next unused number (DR-029 for the contracts layer).

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

---

## DR-027: Distribution Fitter, Empirical Copula, and CRPS-Based Variance Optimization

- Status: active
- Date: 2026-05-21
- Confidence: high

### Context

- The existing probability system used per-stat projections (mean, std) with independent Monte Carlo draws, ignoring correlation between stats (e.g., high AST correlates with high TOV).
- The existing ensemble optimizer (`optimize_weights.py`) tuned the *mean* (MAE/RMSE) but not the *variance* — volatility context multipliers (B2B, playoff, etc.) were fixed at 1.0.
- Quantile model outputs (P10/P50/P90) were generated during training but not used to derive full distribution parameters for downstream consumers.
- The Nexus model's copula head could produce correlated draws, but Nexus was not (and may never be) the active training path.

### Options Considered

1. Block entirely on Nexus copula head being wired into the active pipeline.
2. Build a lightweight empirical copula using archetype-conditioned residual correlations + Gaussian copula, with distribution parameters derived from quantile outputs.
3. Keep independent Monte Carlo (no correlation) — simplest but least accurate.

### Decision

- Option 2: lightweight empirical copula pipeline.
- `DistributionFitter` derives Std/Skew/Zero-Prob/Lambda from P10/P50/P90 quantile outputs (no Nexus needed).
- `CovarianceCache` computes 6x6 archetype-conditioned correlation matrices from historical residuals (actual - projected).
- `ProbabilityCalculator.run_copula_simulation()` generates correlated multi-stat draws via Gaussian copula (Cholesky + inverse CDF per stat).
- `calculate_empirical_crps()` in metrics.py provides the objective function for variance optimization.
- `optimize_variance.py` tunes 7 context-specific volatility multipliers via scipy Nelder-Mead using CRPS.
- `ReportGenerator._enrich_with_distributions()` writes distribution params into every projection CSV export.

### Tradeoffs

- Empirical copula needs historical data to build residual matrices — cold-start archetypes fall back to identity matrix.
- CRPS optimization is per-target and independent of the mean optimizer — two-step tuning (mean then variance).
- Distribution parameters derived from P10/P50/P90 are an approximation, not the exact posterior from a full copula head.
- Lightweight and testable — no neural network training required.

### Consequences

- New entry point: `optimize_variance.py`.
- New modules: `distribution_fitter.py` (86 lines), `empirical_covariance.py` (135 lines).
- Modified: `probability_calculator.py` (run_copula_simulation + CovarianceCache support), `metrics.py` (CRPS), `report_generator.py` (distribution enrichment).
- Projection CSVs now carry `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA` columns for all 6 stats.
- Archetype correlation matrices cached to `data/cache/archetype_covariances.npz`.
- Full test suite: `169 passed, 0 failed`.

### Revisit Triggers

- If the empirical copula proves insufficiently accurate compared to the Nexus copula head, wire Nexus into the active pipeline and deprecate the empirical engine.
- If 7 CRPS context multipliers per target are too many to optimize reliably, reduce the parameter space or share multipliers across related stats.
- If the `calculate_empirical_crps()` approximation (Gini mean difference) diverges from exact CRPS, switch to the exact O(n²) computation for small-n regimes.

---

## DR-028: Smart Per-Target Feature Selection

- Status: active
- Date: 2026-06-04
- Confidence: high

### Context

- The 26 feature groups produce a feature matrix that grew to 150+ columns. Empirically, only a subset is useful per stat — `STL` and `BLK` benefit from lineup/role signals but not from pace or efficiency signals, while `PTS` and `AST` benefit from usage signals. Training all 6 targets on the full feature set wastes capacity on noise and risks overfitting on weak signals.
- There is no per-target feature list — every CatBoost model sees the same `self.feature_cols` list.
- Existing tools (`optimize_weights.py`, `optimize_variance.py`) tune the mean and variance after training but cannot undo a feature set that includes strong noise.
- Manual feature pruning doesn't scale — different game contexts (regular vs playoff, early vs late season) need different subsets.

### Options Considered

1. Block on a separate feature-pruning research project that uses CatBoost importance + a held-out backtest loop.
2. Build a self-contained `SmartFeatureSelector` that combines 5 signals (group ablation, per-target pruning, shadow filtering, stability, missingness) into a per-target feature score, runnable from `train.py` with one CLI flag.
3. Keep the canonical `self.feature_cols` list and rely on CatBoost's built-in L2/regularization.

### Decision

- Option 2: a self-contained selector with a documented signal weight recipe and a profile-driven stage gate.
- `FeatureGroupAblator` produces per-group MAE deltas (backtest_gain signal, 40% weight).
- `SmartFeatureSelector._per_target_signals` runs an 80/20 temporal split, fits a fast `HistGradientBoostingRegressor`, and records catboost-style gain importance (20%), permutation importance (10%), and stability (correlation between gain importances on the first vs second half of training data, 25%).
- A missingness penalty (5%, subtracted) drops features with too many NaN/zero rows.
- `ShadowFeatureFilter` injects `SHADOW_RANDOM_NORMAL`, `SHADOW_RANDOM_UNIFORM`, `SHADOW_PERMUTED_TARGET` control columns and uses their median importance as a noise floor.
- Profile gating: `fast` (group ablation only), `balanced` (+ per-target pruning + shadow filter), `max_accuracy` (+ time-stability check).
- The selector writes a `SelectionManifest` to `models/feature_selection_manifest.json` and `TrainingPipeline.apply_feature_selection_manifest()` parses it for the current run.
- `ModelManager` now bootstraps `EnsembleWeights` from `WeightStore` at load time (so the runtime uses data-driven weights even before `optimize_weights.py` runs). `TrainingPipeline._save_blend_weights()` also writes the training-time blend to `WeightStore`.
- `backtest_result_to_json_dict()` exposes a stable JSON contract for downstream tooling.

### Tradeoffs

- `HistGradientBoostingRegressor` is faster than CatBoost but the gain importances don't exactly match the production trainer. The selector's score is a useful approximation, not a final answer.
- Per-target lists double the per-target feature bookkeeping. The manifest JSON is the source of truth, but if the manifest goes missing, the pipeline falls back to the canonical `self.feature_cols` list — there is no error.
- Profile gating trades coverage for runtime: `max_accuracy` adds a second temporal sub-split for stability, which roughly doubles selector runtime.
- The signal-weight recipe (`WEIGHTS` in `smart_feature_selector.py`) is fixed in code. If the signals prove miscalibrated, both the weights and the test assertions need to move in lockstep.
- WeightStore bootstrap in `ModelManager` is non-fatal — if no `current.json` exists, the legacy `blend_weights.pkl` blend stands. The bootstrap doesn't validate that the versioned weights are *better* than the legacy blend.

### Consequences

- New CLI flags: `train.py --feature-selection {off,smart}`, `train.py --selection-profile {fast,balanced,max_accuracy}`.
- New YAML config: `feature_selection:` and `feature_selection_profiles:` blocks in `config/default.yaml`.
- `Config` dataclass carries `feature_selection` and `feature_selection_profiles`.
- `TrainingPreset` carries optional `feature_selection` and `feature_selection_profile` fields.
- `TrainingPipeline` carries `target_feature_cols`, `feature_selection_manifest`, `feature_selection_profile`, `feature_selection_config`. `_feature_cols_for_target(target)` returns the per-target list with fallback to `self.feature_cols`.
- `model_stack_metadata.pkl` records `feature_selection_enabled`, `feature_selection_target_specific`, `feature_selection_profile`, and `selected_features_by_target` when smart selection ran.
- `FeatureSelector.select_features_for_target(df, target, allowed_features=None)` accepts an allow-list and skips the leakage-safe filter when the upstream selector already validated the columns.
- `ModelManager.load_models()` bootstraps `EnsembleWeights` from `WeightStore` after the legacy blend is loaded.
- `backtest.py --json-output <path>` writes a stable machine-readable JSON payload.
- New modules: `feature_group_ablation.py` (289 lines), `shadow_feature_filter.py` (279 lines), `smart_feature_selector.py` (680 lines).
- New test files: `tests/test_evaluation/test_smart_feature_selector.py` (19 tests), `tests/test_evaluation/test_backtest_json_output.py` (1 test).

### Revisit Triggers
- If the 5-signal score proves noisy on real data, retune `WEIGHTS` and the test assertions together. Do not change one without the other.
- If `HistGradientBoostingRegressor` importances diverge too much from CatBoost's, switch the per-target signal to a small CatBoost trainer (slower but closer to production).

---

## DR-029: Add a Contracts Layer for Inter-Step Artifact Validation

- Status: active
- Date: 2026-06-04
- Confidence: high

### Context

- The pipeline is a sequence of CLI steps (`update_data.py` → `train.py` → `simulate_season.py` → `query_prob.py`) connected entirely by file-based artifacts in `data/` and `models/`.
- DR-007 (2026-04-01) established that the training-to-runtime artifact contract must be enforced at both boundaries, but the contract was implemented as ad hoc file existence checks scattered across `train.py`, `simulate_season.py`, `ModelManager`, `TrainingPipeline`, and the loader.
- The self-optimizing loop (DR-021, 2026-05-09) and the smart per-target feature selector (DR-028, 2026-06-04) introduce more contract-shaped artifacts (`models/blend_weights/current.json`, `models/feature_selection_manifest.json`). The training/runtime code path now depends on the union of these contracts but has no single source of truth.
- Scrapers drift, retraining can land on a different feature set, and the optimizer writes a new weight version — all of these are seam failures that are hard to attribute when the contract checks are scattered.

### Options Considered

1. Keep adding per-call-site `os.path.exists()` checks and accept the drift between training, simulation, and query loaders.
2. Build a dedicated `src/contracts/` module that owns the canonical contract for each artifact class (runtime artifacts, projection CSV, feature schema, schedule input) and is invoked at every step boundary.
3. Move contract checks to a third-party schema validator (e.g., pydantic, jsonschema) and treat the contracts as data, not code.

### Decision

- Option 2: a dedicated `src/contracts/` module with one file per artifact class, plus a typed exception hierarchy in `errors.py`.
- `contracts/artifacts.py::ArtifactContract` is the canonical list of required runtime artifacts. `validate_runtime_artifacts(contract)` checks the union of: per-target `*_catboost.cbm` + `*_metadata.joblib` for all 6 stats, plus `feature_schema.pkl`, `feature_cols.pkl`, `blend_weights.pkl`, `model_stack_metadata.pkl`, and (when `transformer_required=True`) `attention_transformer.pkl`. It also unpickles `model_stack_metadata.pkl` and verifies its `targets` field matches the canonical 6-stat set. Optional `max_age_hours` enforces staleness.
- `contracts/projections.py::validate_projection_csv()` checks the `player_projections_*.csv` schema, including the `DATA_QUALITY` column added by DR-025.
- `contracts/features.py::FeatureSchema` is the single source of truth for training-time and inference-time feature layout.
- `contracts/schedule.py` validates the schedule input.
- `contracts/errors.py` defines `ContractError` and the four typed subclasses (`ArtifactContractError`, `FeatureSchemaContractError`, `ProjectionSchemaContractError`, `ScheduleContractError`).
- Standalone entry: `check_contracts.py` (root) — CLI for debugging contract failures in isolation. Flags: `--models-dir`, `--projection-csv`, `--transformer-required`. Exits 0 on success, raises the typed exception on failure.
- `train.py` and `simulate_season.py` both invoke `validate_runtime_artifacts()` at startup. `ProjectionLoader` calls `validate_projection_csv()` when loading `player_projections_*.csv`.

### Tradeoffs

- A new module is more code than a single helper, but the typed exceptions and the per-artifact file layout make the seam obvious to future contributors.
- Adding the contract layer means there is now a third place (alongside training and runtime) where a new mandatory artifact must be registered. CODE_RULES.md mandates updating all three in lockstep.
- The contract validator currently does not check the `WeightStore` (`models/blend_weights/current.json`) or the smart-selection manifest (`models/feature_selection_manifest.json`) — those are opt-in/optional and have their own validators downstream. Future work may fold them in.
- DR-022 was supposed to capture this decision but is missing. Recorded here as DR-029 to avoid renumbering existing cross-references.

### Consequences

- New CLI: `python check_contracts.py [--models-dir models] [--projection-csv <path>] [--transformer-required]`.
- `src/contracts/` is the canonical seam. New mandatory runtime artifacts go in `_required_files()`; new CSV schemas go in `projections.py`; new feature layout rules go in `features.py`.
- Both `train.py` and `simulate_season.py` now fail fast at startup if the contract is violated, instead of producing partial-success results.
- `ProjectionLoader` raises `ProjectionSchemaContractError` (typed) on schema mismatch, replacing what was previously a silent fallback.
- New test: `tests/test_contracts/test_pipeline_contract_smoke.py` covers the smoke path through the contract validator.
- When a new artifact is added, CODE_RULES.md mandates updating `src/contracts/`, the producer, and the consumer in the same change.

### Revisit Triggers
- If the per-artifact file layout becomes unwieldy, consider a single `contracts.py` that owns all schemas declaratively (Option 3 above).
- If the typed exceptions need richer cross-artifact correlation (e.g., a single report covering all failed contracts), add an aggregator.
- If the WeightStore and the smart-selection manifest become required runtime artifacts, fold their validators into `validate_runtime_artifacts()`.


---

## DR-030: WeightStore Bootstrap in ModelManager.load_models()

- Status: active
- Date: 2026-06-04
- Confidence: high

### Context

- The training pipeline produces an `EnsembleWeights` object and persists it via the legacy binary `blend_weights.pkl` (used by the `ModelManager` loader and the `ArtifactContract` validator). DR-021 introduced `WeightStore` for versioned JSON weights, but `ModelManager` still defaulted to the legacy blend at load time.
- `optimize_weights.py` writes to `WeightStore` rather than to the legacy `blend_weights.pkl`. If `optimize_weights.py` has never been run, the runtime uses the training-time blend, which is the same data the legacy `blend_weights.pkl` carries — but operators who only ran training did not have a way to inspect or hot-reload the blend without restarting the simulator.
- A new agent on a freshly trained project would discover that the runtime silently uses the training-time blend even when a `current.json` exists in `models/blend_weights/`. There was no explicit path that prefers the versioned blend.

### Options Considered

1. Force operators to run `optimize_weights.py` after every `train.py` run so the versioned blend is the canonical source of truth.
2. Make `ModelManager.load_models()` non-fatally bootstrap from `WeightStore.current.json` at load time; if a versioned blend exists, override the legacy blend; otherwise keep the legacy blend.
3. Drop the legacy `blend_weights.pkl` path entirely and require `WeightStore` to be the only source of truth.

### Decision

- Option 2: non-fatal bootstrap from `WeightStore` at `ModelManager.load_models()`.
- After the legacy `blend_weights.pkl` is loaded, `ModelManager.load_models()` calls `WeightStore.load_current()` and, when the result is non-None, overrides the active blend via `use_ensemble_weights(...)`. The bootstrap is wrapped in a try/except so a broken store is logged and the legacy blend stands.
- `TrainingPipeline._save_blend_weights()` writes the training-time blend to both the legacy `blend_weights.pkl` (so `ArtifactContract` validation continues to pass) and the versioned `WeightStore` (so the bootstrap path can pick it up on the next load).
- `WeightStore` remains the single source of truth for hot-reload, accept/verify gates, and rollback. The legacy `blend_weights.pkl` stays as a fallback for fresh process loads that have not yet had the bootstrap fire.

### Tradeoffs

- Two storage locations for the same data is a maintenance hazard. We accept it for now because the legacy path is what `ArtifactContract` validates against, and dropping it would break `validate_runtime_artifacts(...)` until a new contract check is wired in.
- The bootstrap is non-fatal, so a broken `WeightStore` does not crash simulation. Operators see the legacy blend in use and a `WeightStore bootstrap skipped` debug log line. This trades strictness for the ability to run a fresh simulator with no `WeightStore` on disk.
- `_blend_requires_transformer()` inspects the active blend weights to decide whether the runtime requires a Transformer artifact; the bootstrap path therefore influences the contract check indirectly.

### Consequences

- New behavior: `ModelManager.load_models()` logs "Bootstrapped ensemble weights vN from WeightStore (score=...)" at INFO when a versioned blend is picked up.
- `TrainingPipeline._save_blend_weights()` now writes twice (legacy + WeightStore) per training run.
- Hot-reload via `ModelManager.set_weights()` continues to work — the bootstrap only fires on a fresh load.
- The decision is cross-referenced from `project-brain/ARCHITECTURE.md` (Flow 2 contract notes), `project-brain/FILE_MAP.md` (ModelManager notes), and `project-brain/CURRENT_STATE.md` (Areas That Need Confirmation, rewritten to status).

### Revisit Triggers

- If `ArtifactContract` is updated to check `WeightStore.current.json` instead of `blend_weights.pkl`, the legacy dual-write can be removed.
- If `set_weights()` hot-reload proves brittle under concurrent loads, replace the bootstrap with a load-time cache invalidation hook.
- If the bootstrap debug line becomes noise, gate it behind a verbose flag rather than always-on at INFO.

---

## DR-031: Cross-Boundary Contract Wiring in Production Code Paths

- Status: active
- Date: 2026-06-04
- Confidence: high

### Context

- DR-029 introduced the `src/contracts/` layer as the canonical seam, but at the time the contracts lived in the seam file only — the production code paths that should have invoked them did not. The schedule, projection, and feature-schema validators were defined and unit-tested but not called from the read/write paths in the simulator, the projection loader, the report generator, or the model manager.
- `tests/test_query/test_six_stat_contract.py::test_missing_tov_columns_loads_with_defaults` was a deliberate design choice (legacy CSVs loaded with zero defaults) that no longer matched the new schema reality. The contracts layer cannot succeed if the boundary code does not invoke it.
- Scrapers and report generators drift, retraining can land on a different feature set, and downstream tools evolve. The contracts layer is the safety net, but it only catches drift if the boundary code consults it.

### Options Considered

1. Trust the contracts layer to catch problems in isolation (e.g., via `check_contracts.py` run ad hoc).
2. Wire the contract validators into the actual read/write paths in the production code, so the contract is enforced at the seam by construction.
3. Move all boundary code to a third-party schema library (pydantic, jsonschema) and centralize the validation declaratively.

### Decision

- Option 2: wire the contract validators into the boundary code paths.
- `src/data/schedule_scraper.py::ScheduleScraper` calls `normalize_schedule_frame(...)` on every read path (cached schedule hit, fresh API, cache fallback, season cache). Empty frames are skipped from normalization.
- `src/simulation/season_simulator.py::SeasonSimulator.simulate_season` converts the schedule frame to `ScheduleGame` records via `schedule_rows_to_games(...)` before iterating matchups (both ThreadPoolExecutor and sequential paths). The iterator uses the typed `home_team`, `away_team`, `game_date`, `game_id` attributes rather than raw dict lookups.
- `src/query/projection_loader.py::ProjectionLoader.load_projections` calls `validate_projection_frame(...)` on every load and re-raises the typed `ProjectionSchemaContractError`. Legacy CSVs missing distribution or `DATA_QUALITY` columns are now rejected; the previous default-fallback behavior is gone.
- `src/simulation/report_generator.py::ReportGenerator.export_player_projections` calls `validate_projection_frame(...)` on the assembled DataFrame before writing the CSV. The CSV schema is strict: 6 stats x 8 columns plus `DATA_QUALITY`.
- `src/models/model_manager.py::ModelManager.predict_player_stats` calls `load_expected_feature_cols(models_dir)` and `align_feature_frame(df, expected_cols)` before the leakage-safe selector runs, so reordered or extra-column inference frames are coerced to the trained layout.
- `train.py` calls `validate_runtime_artifacts(ArtifactContract(...))` at the bottom of the training flow as a post-train check, in addition to the `ModelManager` and `simulate_season.py` startup checks.
- `tests/test_query/test_six_stat_contract.py` is updated: `test_missing_tov_columns_loads_with_defaults` becomes `test_missing_tov_columns_fails_loudly` and asserts the typed exception.

### Tradeoffs

- Strictness is the right default for projection CSVs, but it is a breaking change for any operator with legacy CSVs on disk. The workaround is documented in `KNOWN_BUGS.md` (KB-021) and `PROJECT_CONTEXT.md` (Journey 4) — regenerate from the current `simulate_season.py`.
- Adding a `load_expected_feature_cols(...)` call in `predict_player_stats` is a one-time inference-frame alignment; in exchange, reordered or extra-column inference frames are now caught at the seam rather than producing subtly wrong predictions.
- The contract seam is now load-bearing — bypassing it (e.g., a direct `pd.read_csv(...)` in a new consumer) silently produces a result that the contract should have rejected. `CODE_RULES.md` (Contracts Layer) calls this out.

### Consequences

- `ProjectionLoader` raises `ProjectionSchemaContractError` on missing columns; `tests/test_query/test_six_stat_contract.py` guards the load path.
- `ScheduleScraper` and `SeasonSimulator` can no longer accept malformed schedule frames; the contract catches null or missing `GAME_ID`, `GAME_DATE`, `HOME_TEAM`, `AWAY_TEAM` at the seam.
- `ModelManager.predict_player_stats` is now defensive about inference frame layout; future drift in the trained feature set is caught at prediction time rather than after the first incorrect batch.
- `train.py` fails fast if a training run produces an incomplete artifact set, even if the pipeline itself reported success.
- The decision is cross-referenced from `project-brain/ARCHITECTURE.md` (contracts layer paragraph), `project-brain/CODE_RULES.md` (Contract wiring at seams + Strict projection schema), `project-brain/FILE_MAP.md` (production call-site bullets), and `project-brain/KNOWN_BUGS.md` (KB-021).

### Revisit Triggers

- If the per-call-site validators become unwieldy, consider moving to a central boundary middleware (e.g., a single `validate_artifact(...)` decorator on the seam functions) — Option 3 above.
- If the strict projection schema blocks legitimate operator workflows, add an explicit `--allow-legacy-csv` flag to `ProjectionLoader.load_projections` that logs a clear warning rather than disabling the validator globally.
- If the schedule scraper needs to ingest an API that does not match the canonical column shape, add a per-source normalizer under `src/data/normalizers/` rather than relaxing `normalize_schedule_frame(...)`.

---

## DR-032: Residual-Calibrated Confidence Intervals Are JSON Artifacts Loaded Best-Effort

- Status: active
- Date: 2026-06-12
- Confidence: high

### Context

- Tickets 1-3 added a residual correction path: build walk-forward residual data, train per-target residual models, and apply corrections during runtime prediction.
- The next missing capability was uncertainty around corrected predictions. The system already had distribution-enriched projection columns and probability tooling, but it did not have a residual-error calibration layer that could say how wide the prediction range should be after correction.
- The project is CLI-first and file-based. Any new calibration layer needs to fit the existing local artifact style and must not make simulation unusable when artifacts have not been generated yet.

### Options Considered

1. Fold interval widths into residual model metadata under `models/residual/`.
2. Create separate JSON interval artifacts under `models/calibration/` and load them best-effort at runtime.
3. Recompute interval widths directly inside `ModelManager` from the residual parquet on every load.

### Decision

- Option 2: separate JSON calibration artifacts under `models/calibration/`.
- `calibrate_residual_intervals.py` is the explicit CLI for building these artifacts from `data/evaluation/residual_training.parquet`.
- `ResidualIntervalCalibrator` writes one `{stat}_intervals.json` file per target plus `calibration_metadata.json`.
- `CalibrationIntervalStore` loads artifacts non-fatally. Missing files disable interval output; missing bucket falls back to `GLOBAL`; missing stat produces no interval.
- `ModelManager.predict_player_stats(..., include_confidence=True)` is opt-in. Default prediction output remains backward-compatible.
- Projection CSVs always include the new interval/confidence columns so the strict query contract remains stable even when calibration is absent.

### Why

- Keeping calibration separate from residual models lets operators rebuild intervals without retraining correction models.
- JSON artifacts are inspectable and match the repository's file-based persistence style.
- Best-effort runtime loading prevents a missing optional calibration step from breaking normal simulation.
- The projection CSV schema must still be strict; optional calibration is represented as blank interval bounds plus `NO_EDGE`, not missing columns.

### Tradeoffs

- There are now two residual-related artifact directories (`models/residual/` and `models/calibration/`), so docs and operator workflows must distinguish point correction from uncertainty calibration.
- The confidence scorer is intentionally coarse. It is useful for surfacing risk, but it is not a betting recommendation and should not be treated as a calibrated probability by itself.
- Real empirical coverage is not proven until `calibrate_residual_intervals.py` is run on current residual data and checked against holdout outcomes.

### Consequences

- New files:
  - `calibrate_residual_intervals.py`
  - `src/correction/calibration.py`
  - `src/correction/interval_store.py`
  - `src/correction/confidence_scorer.py`
  - `tests/test_correction/test_calibration.py`
- Updated runtime/export seam:
  - `ModelManager` loads `models/calibration/` and appends interval keys only when `include_confidence=True`.
  - `GameSimulator` requests confidence-aware batch predictions.
  - `ReportGenerator` writes interval/confidence columns for all six stats.
  - `src/contracts/projections.py` now validates the 6-stat x 14-column projection schema plus confidence labels.

### Revisit Triggers

- If live calibration coverage is materially off target, revise bucket definitions, confidence thresholds, or quantile selection.
- If probability calculation needs calibrated intervals directly, Ticket 5 should consume these artifacts through `CalibrationIntervalStore` rather than re-reading JSON ad hoc.
- If operators need backwards-compatible legacy projection loading, add an explicit migration/regeneration command rather than weakening `validate_projection_frame(...)`.
