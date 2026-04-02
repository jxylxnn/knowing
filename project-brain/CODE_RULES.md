# Code Rules

## Purpose

- These rules are derived from the current repository shape and are intended to keep future changes compatible with the active CLI and artifact contracts.
- When the repo is inconsistent, this file chooses a stable default to follow going forward.

## Hard Rules

- Preserve the top-level CLI workflows:
  - `update_data.py`
  - `train.py`
  - `simulate_season.py`
  - `query_prob.py`
  - `clear_cache.py`
- Treat filesystem artifact names and locations as API contracts unless you update every consumer in the same change.
- Update `/project-brain` after every meaningful code change.
- Do not silently change target-stat semantics, query names, or exported CSV schemas.

## Naming And Data Conventions

### Canonical target names

- Use uppercase stat names in model/training code:
  - `PTS`
  - `REB`
  - `AST`
  - `STL`
  - `BLK`
  - `TOV`
- Lowercase names such as `pts` or `reb` are query/UI aliases, not the canonical modeling contract.

### File and artifact naming

- Keep model artifacts compatible with `src/models/model_manager.py`.
- If a model format or filename changes, update:
  - the training writer
  - the runtime loader
  - any existence checks in top-level scripts
  - tests covering the contract
- `train.py` must not report success if the required runtime artifact set is incomplete.
- The minimum enforced runtime set is:
  - per-target CatBoost backbone artifacts plus metadata
  - `feature_schema.pkl`
  - `feature_cols.pkl`
  - `blend_weights.pkl`
  - `attention_transformer.pkl` when the Transformer path is enabled

### Config naming

- Use one config attribute naming style consistently inside a class.
- Stable default for this repo: prefer lowercase Python attribute names for instance state, even if config YAML keys are uppercase-like in concept.
- Do not mix `self.max_retries` and `self.MAX_RETRIES` in the same runtime object.

## Module Organization Rules

### Preprocessing

- New engineered features should be added as a `FeatureGroup` in `src/preprocessing/features/` unless there is a strong reason not to.
- Register new feature groups through the active feature-engineering flow instead of embedding ad hoc feature logic in `train.py`.
- When adding features:
  - document the feature purpose
  - verify column naming consistency
  - consider inference compatibility

### Training

- Keep `src/training/pipeline.py` as the primary training orchestration path unless a deliberate redesign is documented.
- Avoid duplicating training entry logic in new modules without a clear ownership split.
- Heavy training imports should remain lazy when practical to keep non-training commands lightweight.

### Models And Inference

- `src/models/model_manager.py` is the runtime model-loading boundary for simulation.
- New runtime model artifacts should be loaded through `ModelManager` or a clearly documented successor, not by scattering ad hoc file reads across the simulator.
- Keep Transformer validation/runtime inference on the eager model path by default. If `torch.compile` is re-enabled for the Transformer, it must stay behind an explicit safety flag and should never be the only validation path.
- When touching CUDA attention code, prefer math SDPA fallback controls for validation and add regression coverage for the eager path.

### Simulation

- `src/simulation/game_simulator.py` is a critical orchestrator. Keep interface changes small and documented.
- If a simulation path becomes obsolete, remove it or mark it clearly. Do not leave large unreachable blocks without explanation.

### Query

- The query layer should consume exported projection artifacts rather than reimplement full model inference.
- If the exported schema changes, update `src/query/projection_loader.py` and query tests in the same change.

## Refactor Rules

- Avoid cross-cutting rewrites unless the user explicitly asks for architectural change.
- Before moving or renaming important files, update:
  - `project-brain/FILE_MAP.md`
  - `project-brain/ARCHITECTURE.md`
  - any import shims or callers
- If a refactor changes subsystem ownership or active flow, add a decision record in `project-brain/DECISIONS.md`.
- Remove compatibility shims only after verifying all active callers.

## Testing Expectations

- Run targeted pytest coverage for the modules you change.
- Prefer adding regression tests when touching:
  - artifact persistence
  - scraper config handling
  - CSV export schemas
  - feature schema reconstruction
- Do not rely only on unit tests when changing a contract seam between:
  - training and model loading
  - simulation and query
  - scraper outputs and simulator inputs

## Performance Rules

- Avoid repeated column-by-column pandas insertion for large feature sets when a batched concat or vectorized operation is feasible.
- In `src/preprocessing/features/*`, prefer collecting wide feature outputs in temporary dicts/DataFrames and attaching them with one `pd.concat(axis=1)` per feature group.
- Be cautious with extra work inside simulation loops; seemingly small per-player or per-draw overhead multiplies quickly.
- Keep lazy imports for PyTorch-heavy code paths when they reduce startup overhead for non-training commands.

## Reliability Rules For Scrapers And External Data

- Expect third-party sources to drift.
- Prefer explicit, well-logged fallbacks over accidental exceptions or silent bad state.
- Cache behavior should be deterministic and easy to inspect on disk.
- If a scraper is effectively inactive, document that reality instead of implying full support.

## Documentation Rules

- Every meaningful behavior change must update the relevant `/project-brain` docs.
- `CURRENT_STATE.md` should always reflect the latest known repo health.
- `TASKS.md` should be updated when work is completed or new debt is found.
- `KNOWN_BUGS.md` should be updated when a defect is confirmed, mitigated, or fixed.

## Stable Defaults For Inconsistent Areas

- Prefer one clear active path over multiple semi-maintained alternatives.
- Prefer explicit schema persistence over implicit reconstruction.
- Prefer small pure functions or focused classes over giant orchestration objects, except where existing architecture already centralizes critical flow.
- Prefer deleting dead code to leaving unreachable legacy blocks in place.
