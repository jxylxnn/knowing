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
  - `clear_cache.py`

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
- Output artifacts:
  - `data/nba_players.csv`
  - `data/nba_games.csv`
- This is the raw-data prerequisite for model training.

### 2. Train Models

- Run `python train.py`.
- The script loads the CSV data through `src/preprocessing/data_loader.py`.
- It engineers features through `src/preprocessing/feature_engineer.py`.
- It trains the model stack through `src/training/pipeline.py`.
- Expected artifact contract for downstream consumers:
  - CatBoost model files per target in `models/`
  - per-target CatBoost metadata files in `models/`
  - `attention_transformer.pkl`
  - `feature_schema.pkl`
  - `feature_cols.pkl`
  - `blend_weights.pkl`
- Important behavior: the active pipeline now validates this artifact contract before reporting success, so incomplete CatBoost runtime output is treated as a hard failure instead of a silent partial success.
- Transformer validation/runtime inference now stays on an eager-safe path by default; `torch.compile` is opt-in rather than the default runtime behavior for this model.

### 3. Simulate Upcoming Games

- Run `python simulate_season.py --today`, `--date`, `--week`, or `--season`.
- The script uses:
  - `src/models/model_manager.py`
  - `src/simulation/game_simulator.py`
  - `src/simulation/season_simulator.py`
  - `src/simulation/report_generator.py`
- Outputs:
  - `data/sim_results/sim_results_<timestamp>.csv`
  - `data/sim_results/player_projections_<timestamp>.csv`
- This is the bridge between trained models and query-time projection usage.

### 4. Query Projection Probabilities

- Run `python query_prob.py`.
- The interactive CLI in `src/query/interactive_cli.py` loads the most recent projection CSV and answers questions such as:
  - over/under probability
  - direct projection lookup
  - player comparisons
- Supported stat aliases in query code:
  - `pts`
  - `reb`
  - `ast`
  - `stl`
  - `blk`
  - `tov`
- Important caveat: the exported projection CSV currently includes only points, rebounds, and assists. The query layer handles steals, blocks, and turnovers only partially through defaults/fallbacks. See `KNOWN_BUGS.md`.

### 5. Clear Generated State

- Run `python clear_cache.py`.
- This removes generated caches, models, experiments, and simulation outputs while intentionally preserving raw source data files in `data/`.

## Core Features That Actually Exist

- Config-driven runtime via `config/default.yaml` and dataclass loaders in `src/config/config.py`.
- Historical ingestion from `nba_api` with multiple season-range modes in `update_data.py`.
- Modular feature engineering based on `FeatureGroup` implementations in `src/preprocessing/features/`.
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
- Simulation pipeline that tries to enrich predictions with:
  - schedule context
  - injuries
  - lineups
  - betting lines
  - matchup and pace context
- Interactive probability querying and Monte Carlo probability estimation in `src/query/`.

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
- Raw data is expected in CSV form under `data/`.
- Downstream inference assumes training artifacts exist in `models/` with stable file names.
- The model stack assumes six canonical target stats in uppercase naming inside training/inference code.
- The system prefers fallback behavior over hard failures when external data scrapers break during simulation.
- Transformer training still uses mixed-precision and GPU-friendly settings where supported, but validation intentionally avoids the compiled flash-attention path unless an operator explicitly opts in.

## Important Constraints

- The project is highly file-contract driven. Changing artifact names, CSV schemas, or feature column expectations can break the simulator and query layers.
- External scraping code is a fragile dependency surface and is not comprehensively covered by tests.
- Some README-style claims are aspirational relative to the current implementation. Future work should document reality before expanding claims.

## Confidence Notes

- Everything above is grounded in repository code, tests, config, or checked-in scripts unless explicitly labeled as inferred or suggested.
- User intent around betting/fantasy use is inferred from `README.md`, query semantics, and simulation/report naming rather than from product code with explicit personas.
