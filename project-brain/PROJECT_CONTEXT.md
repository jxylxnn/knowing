# Project Context

## Project Snapshot

- Repository type: local, CLI-driven Python machine learning project.
- Domain: NBA player stat prediction, game simulation, and prop-style probability queries.
- Runtime shape: no web server, no database, no Docker, no background worker system.
- Primary storage: local CSV, JSON, pickle/joblib, and CatBoost model artifacts under `data/`, `cache/`, `models/`, and `experiments/`.
- Primary entry points:
  - `update_data.py`
  - `train.py`
  - `simulate_season.py`
  - `query_prob.py`
  - `backtest.py` (NEW)
  - `optimize_weights.py` (NEW)
  - `optimize_variance.py` (NEW — tune context-specific volatility multipliers via CRPS)
  - `calibrate_residual_intervals.py` (NEW — build conformal interval artifacts from residual errors)
  - `clear_cache.py`
  - `check_contracts.py` (NEW — 2026-06-04) — standalone artifact-contract validator; also wired into `train.py` and `simulate_season.py` startup
  - `train.py --feature-selection smart --selection-profile {fast,balanced,max_accuracy}` (NEW — per-target feature selection)

## What The Project Is

- A local analytics workbench for building NBA player projections from historical game logs and using them in simulations and probability queries.
- The codebase combines:
  - historical data ingestion from public NBA-related sources
  - feature engineering over player and team time series
  - model training for multiple stat targets
  - schedule-aware simulation of matchups
  - interactive querying of projected outcomes and over/under probabilities

## What Problem It Solves

- Reduces manual work required to go from raw NBA game logs to reusable player-stat projections.
- Gives a single local workflow for:
  - collecting player and team history
  - training predictive models
  - generating matchup projections
  - querying stat probabilities for props, fantasy, or research use
- Preserves a file-based workflow instead of requiring an application server or managed infrastructure.

## Who It Appears To Be For

Confirmed from repo evidence:

- A technically comfortable single-user operator running Python scripts locally.
- Contributors iterating on data features, training logic, and simulation heuristics.

Strongly suggested by `README.md` and CLI flows:

- NBA prop/betting research users.
- Fantasy sports or sports analytics users who want player-level projections.

Not supported by repo evidence:

- Multi-user product workflows.
- Hosted SaaS usage.
- Automated production deployment.

## Core User Journeys

### 1. Refresh Historical Data

- Run `python update_data.py`.
- The script pulls player game logs and team game logs through `nba_api`.
- Also enriches player data with biographical info (AGE, POSITION) via `PlayerBioScraper` → `data/player_bios.csv`.
- Logs current injury report to longitudinal history (`data/injury_history.csv`) via `InjuryHistoryLogger`.
- Writes Parquet dual-write (`nba_players.parquet`, `nba_games.parquet`) for GPU-direct storage reads.
- Output artifacts:
  - `data/nba_players.csv`
  - `data/nba_games.csv`
  - `data/player_bios.csv` (optional)
  - `data/injury_history.csv` (optional)
- This is the raw-data prerequisite for model training.

### 2. Train Models

- Run `python train.py`.
- The script loads the CSV data through `src/preprocessing/data_loader.py`.
- It resolves a named training preset from `config/default.yaml` plus `src/training/presets.py` before feature engineering starts.
- **Precomputes lifecycle aging curves** (B-Ianus Bayesian + KAN nonlinear) at startup, before feature engineering, so caches exist for feature groups to load. Non-fatal on missing bio data.
- It engineers features through `src/preprocessing/feature_engineer.py` (or the GPU-accelerated `feature_engineer_gpu.py` when use_gpu=True and cuDF is available).
- It trains the model stack through `src/training/pipeline.py`.
- The `small` preset keeps the six canonical targets, disables the Transformer, uses the `rolling`, `efficiency`, `momentum`, `pace`, `opponent_strength`, and `archetype` feature groups, and trims to the most recent two seasons when `SEASON_ID` is available.
- In Colab, `train_colab.ipynb` resolves the repo checkout separately from Drive-backed storage so code can run from `/content/knowing` while `data/` and `models/` stay in Google Drive.
- Expected artifact contract for downstream consumers:
  - CatBoost model files per target in `models/`
  - per-target CatBoost metadata files in `models/`
  - `attention_transformer.pkl`
  - `feature_schema.pkl`
  - `feature_cols.pkl`
  - `blend_weights.pkl`
  - `model_stack_metadata.pkl` recording Transformer active status, expected model count, and optionally the selected preset / feature groups
- Important behavior: the active pipeline now validates this artifact contract before reporting success, so incomplete CatBoost runtime output is treated as a hard failure instead of a silent partial success.
- Important behavior: `ModelManager._validate_blend_contract()` now raises when blend weights expect a Transformer that is missing or failed to load, eliminating the partial-blend bug where CatBoost-only fallback silently produced uncalibrated predictions.
- Transformer validation/runtime inference now stays on an eager-safe path by default; `torch.compile` is opt-in rather than the default runtime behavior for this model.

### 3. Simulate Upcoming Games

- Run `python simulate_season.py --today`, `--date`, `--week`, or `--season`.
- Pass `--strict` to fail fast if optional context (injuries, lineups, betting) is degraded.
- The script uses:
  - `src/models/model_manager.py`
  - `src/simulation/game_simulator.py`
  - `src/simulation/season_simulator.py`
  - `src/simulation/report_generator.py`
- Outputs:
  - `data/sim_results/sim_results_<timestamp>.csv`
  - `data/sim_results/player_projections_<timestamp>.csv` — includes `DATA_QUALITY` column (`FULL`, `DEGRADED_FALLBACK`, `DEGRADED_MISSING`), distribution enrichment columns (`{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA` for all 6 stats), and residual-calibrated interval/confidence columns (`{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, `{STAT}_CONFIDENCE_SCORE`).
- This is the bridge between trained models and query-time projection usage.

### 3.5. Calibrate Residual Prediction Intervals

- Run `python calibrate_residual_intervals.py --input data/evaluation/residual_training.parquet --output-dir models/calibration`.
- The CLI consumes the residual dataset produced by the walk-forward residual builder and writes:
  - `models/calibration/pts_intervals.json`
  - `models/calibration/reb_intervals.json`
  - `models/calibration/ast_intervals.json`
  - `models/calibration/stl_intervals.json`
  - `models/calibration/blk_intervals.json`
  - `models/calibration/tov_intervals.json`
  - `models/calibration/calibration_metadata.json`
- Runtime loading is best-effort. If `models/calibration/` is missing or malformed, `ModelManager.predict_player_stats(..., include_confidence=True)` simply omits interval keys and the simulator/export path fills projection columns with `NO_EDGE` / blank numeric bounds.

### 4. Query Projection Probabilities

- Run `python query_prob.py`.
- The interactive CLI in `src/query/interactive_cli.py` loads the most recent projection CSV and answers questions such as:
  - over/under probability
  - direct projection lookup
  - player comparisons
- Projections relying on fallback data show a `DATA_QUALITY` warning.
- Supported stat aliases in query code:
  - `pts`
  - `reb`
  - `ast`
  - `stl`
  - `blk`
  - `tov`
- All six stats (PTS, REB, AST, STL, BLK, TOV) are first-class in the projection CSV schema. Each stat row carries the mean plus distribution columns `{STAT}_P10`, `{STAT}_P50`, `{STAT}_P90`, `{STAT}_STD`, `{STAT}_SKEW`, `{STAT}_ZERO_PROB`, `{STAT}_LAMBDA`, and residual interval/confidence columns `{STAT}_INTERVAL_80_LOW/HIGH`, `{STAT}_INTERVAL_90_LOW/HIGH`, `{STAT}_CONFIDENCE`, `{STAT}_CONFIDENCE_SCORE`. `ProjectionLoader.load_projections` enforces the schema via `validate_projection_frame(...)` and raises the typed `ProjectionSchemaContractError` when required columns are missing. Legacy CSVs (pre-2026-06-12) must be regenerated from the current `simulate_season.py` before the query layer can load them. See `KNOWN_BUGS.md` (KB-021).

### 5. Clear Generated State

- Run `python clear_cache.py`.
- This removes generated caches, models, experiments, and simulation outputs while intentionally preserving raw source data files in `data/`.

### 6. Backtest Prediction Accuracy (NEW)

- Run `python backtest.py --recent 14` (or `--from`/`--to`).
- Loads trained models and evaluates predictions against historical completed games.
- Computes per-stat MAE, RMSE, R², calibration error, and prediction interval coverage.
- Output: JSON results and console summary of `BacktestResult` with per-target `TargetMetrics`.

### 7. Optimize Ensemble Weights (NEW)

- Run `python optimize_weights.py --recent 14` (or `--from`/`--to`).
- Runs backtesting to establish baseline, then uses scipy.optimize to find better blend coefficients.
- 13 tunable parameters: 6 per-target CatBoost/Transformer ratios + 6 per-target intercepts + 1 CatBoost-MAE blend.
- Accept/verify gates prevent regression. Output: versioned JSON weights in `models/blend_weights/` (not `data/weights/` — earlier docs had the wrong path).
- Supports `--dry-run`, `--rollback N`, and `--list` modes.

### 8. Optimize Variance Multipliers (NEW)

- Run `python optimize_variance.py --recent 30` (or `--from`/`--to`, `--target <STAT>`).
- Tunes 7 context-specific volatility multipliers (B2B, Rookie, Blowout, Home, Away, Playoff, RestAdvantage) via scipy Nelder-Mead.
- Uses CRPS (Continuous Ranked Probability Score) as the objective — lower CRPS = sharper + better-calibrated forecasts.
- Loads historical data from `data/nba_players.csv` — does not require trained model artifacts.
- Supports `--dry-run` to preview current multipliers without optimizing.
- Risk level: low. Independent of training artifacts, operates on raw CSVs.

### 9. Run Smart Per-Target Feature Selection (NEW)

- Run `python train.py --feature-selection smart --selection-profile balanced` (or `fast` / `max_accuracy`).
- Runs `SmartFeatureSelector` between Step 2 (feature engineering) and Step 3 (training).
- Combines 5 signals (group ablation, per-target pruning, shadow filter, stability, missingness) into a per-target feature score.
- Writes a `SelectionManifest` to `models/feature_selection_manifest.json` with per-target feature lists, dropped features, shadow-dropped features, and per-signal scores.
- `TrainingPipeline` consumes the manifest and trains each per-target CatBoost model on its own subset of features.
- `model_stack_metadata.pkl` records the active profile and per-target feature lists for runtime auditability.
- Failure is non-fatal — `train.py` logs a warning and falls back to the full feature set.
- Same data requirements as a normal `train.py` run — operates on `data/nba_players.csv` after feature engineering.
- Use `--json-output` on `backtest.py` to feed downstream metrics into the same tooling that consumes the selector's outputs.

## Core Features That Actually Exist

- Config-driven runtime via `config/default.yaml` and dataclass loaders in `src/config/config.py`.
- Historical ingestion from `nba_api` with multiple season-range modes in `update_data.py`.
- Modular feature engineering based on `FeatureGroup` implementations in `src/preprocessing/features/`.
- Player-style archetype and similarity features built from rolling/season-level player context, used to generalize better on new or low-sample players.
- Multi-target training centered on six box-score stats:
  - `PTS`
  - `REB`
  - `AST`
  - `STL`
  - `BLK`
  - `TOV`
- A hybrid model stack described in code as:
  - per-target CatBoost regressors
  - one Transformer sequence model
  - inverse-error blending
  - quantile models for uncertainty
- Season-context features for temporal awareness:
  - `SeasonPhaseFeatureGroup` — early-season ramp-up and trade resets
  - `TeamMotivationFeatureGroup` — late-season tanking/load management signals
  - `PostseasonContextFeatureGroup` — playoff detection with pace prior
- Simulation pipeline that tries to enrich predictions with:
  - schedule context
  - injuries
  - lineups
  - betting lines
  - matchup and pace context
- Interactive probability querying and Monte Carlo probability estimation in `src/query/`.
- **Distribution-aware over/under querying (DR-027, 2026-05-21):** `DistributionFitter` derives Mean/Std/Skew/Zero-Prob/Lambda from P10/P50/P90 quantile outputs; `CovarianceCache` stores archetype-conditioned 6x6 empirical correlation matrices from residual analysis; `ProbabilityCalculator.run_copula_simulation()` produces correlated multi-stat Monte Carlo draws via Gaussian copula; `calculate_empirical_crps()` in `src/evaluation/metrics.py` is the probabilistic forecast objective used by `optimize_variance.py`. Copula simulation is the active path for over/under probability when correlation is requested; independent Monte Carlo is still supported for legacy callers.
- **Inter-step artifact contracts (DR-029, 2026-06-04):** `src/contracts/` is the canonical seam between training, simulation, optimizer, and query layers. `ArtifactContract`, `FeatureSchema`, the projection-CSV validator, and the schedule validator are invoked at every producer/consumer boundary (see `project-brain/ARCHITECTURE.md` for the call-site list). `check_contracts.py` is the standalone CLI for debugging contract failures in isolation.
- **Residual-calibrated confidence intervals (DR-032, 2026-06-12):** `ResidualIntervalCalibrator` turns walk-forward residual errors into per-stat conformal half-widths under `models/calibration/`; `CalibrationIntervalStore` loads them non-fatally; `ConfidenceScorer` converts width/data-quality/minutes/residual signals into user-facing confidence labels. This complements residual correction: the residual model adjusts the point estimate, while calibration describes uncertainty around the corrected prediction.

## Features That Appear Intended But Are Not Reliably Delivered

- Fully reliable real-time simulation inputs from scraper integrations.
- Full remaining-season simulation for `simulate_season.py --season`.
- Complete cached query support for `stl`, `blk`, and `tov`.
- Clean end-to-end training-to-simulation artifact persistence.

These are visible as implementation goals, but the current code contains blocking gaps and regressions.

## Key Integrations

### Confirmed External Dependencies

- `nba_api` for player and team game logs.
- CatBoost for gradient-boosted models.
- PyTorch for the Transformer model.
- Pandas / NumPy / scikit-learn for feature engineering and evaluation plumbing.

### Confirmed External Data Scrapers

- NBA.com data and scoreboard endpoints.
- ESPN injury data.
- Basketball Reference.
- Action Network.
- RotoWire.

Important boundary:

- These scrapers are not equally healthy. Several contain observable runtime regressions or are only used behind safe fallbacks. Presence in the repo does not mean the integration is currently dependable.

## Project-Specific Terms

- `FeatureGroup`: plugin-like feature module under `src/preprocessing/features/` that adds a coherent set of engineered columns.
- `PlayerArchetypeFeatureGroup`: deterministic style-profile feature group that emits `ARCHETYPE_*` and `SIMILARITY_TO_*` columns for playstyle matching.
- `TrainingPreset`: named preset in `src/training/presets.py` that controls feature-group selection, rolling windows, Transformer enablement, and optional recent-history trimming.
- `FeatureSchema`: saved metadata describing how training features are organized and later reconstructed for inference.
- `ModelManager`: runtime loader that reconstructs the trained-model stack for simulator and query-time prediction.
- `GameSimulator`: main simulation engine for a single matchup, including context gathering and repeated simulation draws.
- `ProjectionLoader`: query-time loader that reads the most recent exported player projections.
- `sim_results`: CSV export directory for simulation summaries and player projections.

## Scope Boundaries And Non-Goals

This repo does not currently implement:

- an HTTP API
- user accounts or authentication
- a persistent relational or document database
- a web frontend
- cloud deployment or orchestration
- background job queues
- guaranteed real-time or production-grade external data reliability

## Important Assumptions

- The operator can run Python locally and manage data/model artifacts on disk.
- Colab notebook launchers may run code from a repo checkout while persisting data and models in Drive; code paths should not be inferred from Drive artifact paths.
- Raw data is expected in CSV form under `data/`.
- Downstream inference assumes training artifacts exist in `models/` with stable file names.
- The model stack assumes six canonical target stats in uppercase naming inside training/inference code.
- The system prefers fallback behavior over hard failures when external data scrapers break during simulation.
- Transformer training still uses mixed-precision and GPU-friendly settings where supported, but validation intentionally avoids the compiled flash-attention path unless an operator explicitly opts in.
- Training presets are intentionally conservative about artifact naming: the small preset changes feature breadth and Transformer enablement, not downstream filenames or the canonical six-target contract.

## Important Constraints

- The project is highly file-contract driven. Changing artifact names, CSV schemas, or feature column expectations can break the simulator and query layers.
- External scraping code is a fragile dependency surface and is not comprehensively covered by tests.
- Some README-style claims are aspirational relative to the current implementation. Future work should document reality before expanding claims.

## Confidence Notes

- Everything above is grounded in repository code, tests, config, or checked-in scripts unless explicitly labeled as inferred or suggested.
- User intent around betting/fantasy use is inferred from `README.md`, query semantics, and simulation/report naming rather than from product code with explicit personas.
