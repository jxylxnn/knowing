# Decisions

This file records confirmed or strongly inferred architectural decisions visible in the repository as of 2026-04-01.

When a decision is labeled "inferred", it means the repo shows a clear implementation choice but does not contain explicit historical rationale.

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
