# NBA Player Stats Prediction System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/catboost-1.2+-yellow.svg)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Optional-76b900.svg)](https://developer.nvidia.com/cuda-zone)

A production-grade CLI ML system for predicting NBA player statistics (PTS, REB, AST, STL, BLK, TOV). Supports a CatBoost + Transformer full training preset, modular feature engineering (26 built-in registered feature groups plus extensions), GPU-accelerated Monte Carlo simulation with archetype-conditioned correlations, a manual blend-weight tuner (disabled by default), and an interactive probability query CLI. No web server, no Docker, no database — pure-Python with CSV/JSON artifacts. The loaded model stack is artifact-dependent, not set by the CLI default — run `python inspect_artifacts.py --models-dir models` to see what the deployed artifacts actually contain.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Key Subsystems](#key-subsystems)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Pipeline — End to End](#pipeline--end-to-end)
  - [1. Data Collection](#1-data-collection)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training](#3-model-training)
  - [4. Simulation Engine](#4-simulation-engine)
  - [5. Query System](#5-query-system)
  - [6. Ensemble Weight Optimization (Manual)](#6-ensemble-weight-optimization-manual)
- [Configuration Reference](#configuration-reference)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Cache & Artifact Cleanup](#cache--artifact-cleanup)
- [Development Notes](#development-notes)
- [License](#license)

---

## What It Does

**Inputs**: NBA game logs, injury reports, betting lines, lineups, defense ratings, player bios.

**Outputs**:

1. Per-player point projections with a distribution (mean, std, skew, zero-prob), not just a point estimate.
2. Calibrated over/under probabilities using the best-fit distribution per stat (empirical bootstrap, gamma, Poisson, NB, ZIP, Normal).
3. Per-game Monte Carlo simulations with archetype-conditioned stat correlations.
4. Manual ensemble-weight tuning (disabled by default) that evaluates candidates against backtest evidence; an optimization attempt may validly reject a candidate, leaving the current weights untouched.

Intended use cases: sports analytics, fantasy projections, betting-market evaluation.

---

## Architecture

```
+---------------------------------------------------------------------------------+
|                            NBA PREDICTION SYSTEM                                |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  +------------------+    +--------------------+    +--------------------+        |
|  |   DATA LAYER     |    | FEATURE ENGINEERING |    |    MODEL LAYER     |        |
|  |   ----------     |    | -----------------   |    |    ---------       |        |
|  | NBA.com API      |--->| 25+ FeatureGroup    |--->| CatBoost (per-     |        |
|  | ESPN Injuries    |    | classes (rolling,  |    |  target, primary)  |        |
|  | Rotowire Lineups |    | efficiency, matchup,|    | Transformer (seq,  |        |
|  | Action Net Bet   |    | aging, KAN aging,   |    |  secondary)        |        |
|  | Basketball Ref   |    | injury risk, season |    | MAE-companion CB   |        |
|  | Player Bio       |    | phase, motivation,  |    | per-target         |        |
|  |                  |    | postseason context) |    | Versioned weight   |        |
|  |                  |    | 150+ features       |    | store + auto-tuner |        |
|  +------------------+    +--------------------+    +--------------------+        |
|           |                       |                        |                    |
|           v                       v                        v                    |
|  +------------------+    +--------------------+    +--------------------+        |
|  | CONTRACTS        |    | EVALUATION /       |    | SIMULATION ENGINE  |        |
|  | (artifact/schema |    | SELF-OPTIMIZATION  |    | (GPU Monte Carlo)  |        |
|  |  validation      |<-->|  backtest_runner   |<-->| phase_simulator    |        |
|  |  between steps)  |    |  weight_store      |    | archetype engine   |        |
|  +------------------+    |  ensemble_optimizer|    | role sampler       |        |
|                          |  drift_detector    |    | empirical copula   |        |
|                          |  smart selector    |    | (archetype-cond.)  |        |
|                          |  + shadow filter   |    | input_health       |        |
|                          |  + group ablation  |    | strict mode (DR-25)|        |
|                          +--------------------+    +--------------------+        |
|                                                              |                   |
|                                                              v                   |
|                                                  +--------------------+          |
|                                                  |     QUERY CLI      |          |
|                                                  | distribution fitter|          |
|                                                  | prob_formatter     |          |
|                                                  | proj loader w/     |          |
|                                                  | DATA_QUALITY warn  |          |
|                                                  +--------------------+          |
+---------------------------------------------------------------------------------+
```

---

## Key Subsystems

### 1. Modular Feature Engineering (registered groups, not a fixed pipeline)

Each feature group is an independently toggleable `FeatureGroup` class. The `full` preset enables every registered group; `small` enables six core groups and skips Transformer; `laptop_quality` enables 13 groups plus smart feature selection.

- **Performance core** — rolling windows (3/5/10/20/50), efficiency (TS%, eFG%), EWMA momentum, hot/cold streaks
- **Context** — home/away, rest, back-to-back, fatigue, recency form
- **Matchup** — career vs opponent, opponent defensive rating, defense-position matchup
- **Pace & role** — pace factor, usage share, teammate usage, team role
- **Minutes** — minutes confidence, lineup stability, rest density
- **Opportunity** — injury-adjusted opportunity (usage bump when teammates are out)
- **Lifecycle** — B-Ianus Bayesian aging curve, KAN aging factor, skill-development trajectory, injury risk
- **Season context** — season phase (early/mid/late/playoff), team motivation (tank/playoff), postseason context
- **Encoding** — target encoding, league percentile ranking, archetype prior

### 2. Model Stack

| Model | Role | Preset availability |
|-------|------|---------------------|
| CatBoost (per-target, RMSE + MAE multi-loss + quantile regression) | Primary | all presets |
| Transformer (attention over recent games) | Secondary, sequence context | `full` only |
| CatBoost MAE-companion (per-target) | Blended with primary CatBoost | optional — only when multi-loss companions are trained |

| Preset | Stack | Feature groups |
|--------|-------|----------------|
| `full` | CatBoost + Transformer | all registered groups |
| `small` | CatBoost only | six groups |
| `laptop_quality` | CatBoost only | 13 groups plus smart feature selection |

The default active CatBoost path has no MAE companion model: the active training config forces `use_multi_loss=false`, so the deployed bundle contains no `*_catboost_mae.cbm` files and the global MAE-blend field in `current.json` is currently inert.

### 3. Probability Distribution Engine

Not just `Normal(μ, σ)`. The query layer fits the best of:

- **Empirical bootstrap** (preferred when historical density is high)
- **Gamma** (positive continuous: PTS, REB, AST, MIN)
- **Poisson** / **Negative Binomial** / **Zero-Inflated Poisson** (counts: STL, BLK, TOV)
- **Degenerate-zero** / **Normal** (last-resort fallbacks)

Distribution is derived from P10/P50/P90 quantile model output via `DistributionFitter`, which yields (mean, std, skew, zero_prob, λ).

### 4. Ensemble Weight Optimization

The blend weights are **not** hardcoded. They live in a versioned JSON store at `models/blend_weights/` (atomic writes, rollback, parent-version tracking). `optimize_weights.py` is a manual, disabled-by-default capability — run it explicitly with a date range (see Section 6):

```
BacktestRunner → baseline MAE on holdout
EnsembleOptimizer → scipy.optimize over a component-derived parameter layout
  ├─ evaluate candidate on holdout → candidate MAE
  ├─ compare candidate vs current weights on the same verification window
  ├─ if candidate improves by > threshold AND verification passes → promote
  └─ else reject (current weights untouched; rejection is a valid outcome)
DriftDetector → flags when rolling MAE exceeds 2σ above baseline
```

The search layout is derived from the loaded model components, not a fixed
dimension: per-target CatBoost/Transformer ratios and intercepts (12
parameters for six targets when a Transformer is loaded), and 13 parameters
only when a global CatBoost/MAE companion blend is also active. A
CatBoost-only bundle never receives non-zero Transformer weights.

### 5. Contracts Layer

A dedicated `src/contracts/` module validates the inter-step artifact contract. Run `python check_contracts.py` between steps; both `train.py` and `simulate_season.py` validate at startup.

Validates: required model files, target set match, feature schema consistency, projection CSV schema (incl. `DATA_QUALITY` column), schedule schema.

Feature schemas are strict `feature_schema_v4` contracts. Same-game player and team outcomes are rejected before model loading; the current legacy flat bundle is expected to fail `check_contracts.py` until retraining produces a clean schema. For migration-only diagnostics, pass `--allow-legacy-artifacts` to a model-consuming CLI. This prints `UNSAFE LEGACY ARTIFACT MODE`; replay and promotion are disabled in that mode.

### 6. Input Health & Strict Mode (DR-025)

Every optional context source (injuries, lineups, betting) reports a health status. `simulate_season.py --strict` fails fast if any optional source is `fallback` or `failed`. Without `--strict`, the simulation runs with visible warnings and the `DATA_QUALITY` column on `player_projections_*.csv` flags degraded inputs (FULL / DEGRADED_FALLBACK / DEGRADED_MISSING). The query CLI surfaces these warnings at lookup time.

### 7. Smart Feature Selection (opt-in)

`train.py --feature-selection smart --selection-profile {fast,balanced,max_accuracy}` runs:

1. **Shadow filter** — inject random control features, drop real features below their median importance
2. **Group ablation** — leave-one-group-out MAE deltas
3. **CatBoost-style gain importances** (via fast `HistGradientBoostingRegressor`)
4. **Permutation importance** on the validation split
5. **Stability** — split-half importance correlation
6. **Final score** = `0.40·backtest_gain + 0.25·stability + 0.20·importance + 0.10·permutation − 0.05·missingness`

Result: a per-target feature list written to `models/feature_selection_manifest.json`, consumed by training and inference.

---

## Tech Stack

| Category | Package | Version |
|----------|---------|---------|
| **Core ML** | PyTorch | 2.0+ (CUDA optional) |
| | CatBoost | 1.2.8 |
| | LightGBM | 4.6.0 (minutes predictor) |
| | scikit-learn | 1.8.0 |
| **Data** | pandas | 2.3.3 |
| | numpy | 2.3.5 |
| | scipy | 1.16.3 |
| **NBA Data** | nba_api | 1.11.3 |
| | beautifulsoup4 | 4.14.3 |
| | requests | 2.32.5 |
| **Testing** | pytest | 9.0.2 |
| | coverage | 7.13.4 |

Runtime dependencies are in `requirements.txt`; test tooling is in `requirements-dev.txt`. Install PyTorch separately per your platform (CUDA 12.1 / CPU / macOS) — see the runtime file's header.

---

## Installation

### Prerequisites

- Python 3.12 (project ships a venv at root with this version)
- (Optional) NVIDIA GPU with CUDA 11.8+ / 12.1+

### Setup

```bash
# Activate the bundled venv
source venv/bin/activate

# Install/update runtime dependencies
pip install -r requirements.txt

# Include test tooling when developing the project
pip install -r requirements-dev.txt
```

> **Note**: The README's old "Python 3.10+" badge is stale — the project is on 3.12 (see `AGENTS.md`).

---

## Pipeline — End to End

Each step depends on the prior step's output. Run in order.

### 1. Data Collection

```bash
# First-time setup (interactive season picker)
python update_data.py --interactive

# Fetch last 10 seasons
python update_data.py --all-seasons

# Incremental (only new games since last run)
python update_data.py --update

# Re-fetch even if data exists
python update_data.py --all-seasons --force
```

| Source | What it provides | Update cadence |
|--------|------------------|----------------|
| NBA.com Stats API | Player/team game logs | Daily |
| Basketball Reference | Advanced historical stats | On-demand |
| ESPN Health API | Injury status | Game day |
| Rotowire | Confirmed starters + projections | Pre-game |
| Action Network | Spreads, totals, moneylines | Pre-game |
| NBA.com defense data | Team defensive ratings | Weekly |

**Required artifacts for downstream**: `data/nba_players.csv` (must include `AGE` and `POSITION` columns for lifecycle features), `data/nba_games.csv`.

### 2. Feature Engineering

The pipeline is invoked automatically by `train.py` (no standalone CLI), but you can see the registered groups in `src/preprocessing/features/__init__.py`.

| Group family | Examples |
|--------------|----------|
| Rolling | `ROLL_PTS_AVG_10`, `ROLL_TS_PCT_10`, `ROLL_USG_PCT_10` |
| Context | `IS_HOME`, `REST_DAYS`, `IS_B2B`, `FATIGUE_SCORE` |
| Matchup | `VS_OPP_PTS_AVG`, `RELATIVE_OPP_DEF_PTS`, `DEF_MATCHUP_*_IMPACT` |
| Opportunity | `TEAMMATE_USAGE_*`, `INJURY_OPPORTUNITY_FACTOR` |
| Lifecycle | `B_IANUS_AGE_FACTOR`, `KAN_AGE_FACTOR`, `INJURY_RISK_*`, `SKILL_DEV_TREND` |
| Season context | `SEASON_PHASE_*`, `TEAM_MOTIVATION_*`, `POSTSEASON_CONTEXT_*` |
| Encoding | `PTS_PLAYER_TE`, `PTS_LEAGUE_PCTILE` |

### 3. Model Training

```bash
# Full stack (default preset)
python train.py

# Smaller stack for iteration
python train.py --preset small

# Mode override
python train.py --preset full --mode quick

# Smart feature selection
python train.py --feature-selection smart --selection-profile balanced

# Parallel CatBoost targets on CPU
python train.py --parallel --max-workers 4

# Pre-train group ablation
python train.py --feature-ablation
```

Training modes (per preset):

| Mode | CatBoost iters | NN epochs | Features |
|------|---------------|-----------|----------|
| quick | 500 | 20 | CatBoost only |
| standard | 3000 | 100 | All models |
| full | 5000 | 200 | All models |

**Ensemble blend weights** (the old hardcoded `0.50 / 0.15 / 0.15 / 0.15 / 0.05`) are **no longer in source**. They live in `models/blend_weights/current.json` and **can be tuned** via the manual optimizer (Section 6). The currently deployed version 1 is a training-time initialization, not an optimizer promotion — its `backtest_score` is `null` until an optimization run is accepted.

**CatBoost parallelism**:
- CPU: `--parallel --max-workers N` (size to cores)
- GPU: `--max-workers 1` (avoid CUDA context contention)

**Required artifacts** (validated by `src/contracts/artifacts.py`):

```
models/
├── pts_catboost.cbm   reb_catboost.cbm   ast_catboost.cbm
├── stl_catboost.cbm   blk_catboost.cbm   tov_catboost.cbm
├── {stat}_metadata.joblib  (× 6)
├── feature_schema.pkl
├── feature_cols.pkl
├── blend_weights.pkl
├── model_stack_metadata.pkl
├── attention_transformer.pkl   (only if transformer_required)
├── blend_weights/              (versioned JSON store — see Section 6)
│   ├── v0001.json
│   ├── v0002.json
│   └── current.json            # copy of the active version (not a pointer/symlink)
└── feature_selection_manifest.json   (if --feature-selection smart)
```

### 4. Simulation Engine

```bash
# Today's games
python simulate_season.py --today

# Specific date
python simulate_season.py --date 2026-06-04

# Upcoming 7 days
python simulate_season.py --week

# Remaining season
python simulate_season.py --season

# High-accuracy
python simulate_season.py --today --sims 1000 --stat both

# Fail fast on degraded inputs
python simulate_season.py --today --strict
```

The simulator no longer uses flat correlated-normal samples. It now:

1. Infers a per-player **archetype** (heliocentric_star_guard, low_usage_3_and_d_wing, rebound_first_center, microwave_bench_scorer, secondary_creator_forward, balanced) via `ArchetypeEngine`
2. Loads the **archetype-conditioned 6×6 correlation matrix** for multi-stat draws (`empirical_covariance.CovarianceCache`)
3. Samples a **role state** (limited/normal/expanded/starter/bench/closer) per player per sim with coach-tightness and close-game-prob context adjustments
4. Progresses the game in **phases** (`phase_simulator`)
5. Adjusts via **four factors** and **game context** engines
6. Reports **input health** (full / degraded-fallback / degraded-missing) per projection

**Output schema** (per run):

```
data/sim_results/
├── sim_results_<timestamp>.csv            # team-level summaries
└── player_projections_<timestamp>.csv     # per-player (with DATA_QUALITY column)
data/sim_cache/                            # cached rosters (TTL-based)
```

### 5. Query System

**One-shot**:

```bash
python query_prob.py -p "LeBron James" -s pts -l 25.5
python query_prob.py -p "Jokic" -s reb -l 12.5 -o BOS
python query_prob.py -p "Curry" -s pts -l 30.5 --json
python query_prob.py --compare "LeBron James" "Jayson Tatum" --stat pts
python query_prob.py --list-players
python query_prob.py --list-teams
```

**Interactive REPL**:

```bash
python query_prob.py
```

```
NBA Probability Query Tool
Type 'help' for commands

> LeBron James over 25.5 pts vs BOS

══════════════════════════════════════════════════════════════════════
LeBron James (LAL) vs BOS
══════════════════════════════════════════════════════════════════════
┌─ RECENT PERFORMANCE (Last 5 Games) ──────────────────────────────────┐
│  DATE        MIN    PTS    REB    AST   STL   BLK   TOV   RESULT      │
│  ...
├──────────────────────────────────────────────────────────────────────┤
│  5-GAME AVG: 36.4 MIN, 25.4 PTS, 7.0 REB, 8.0 AST, 1.2 STL, 0.8 BLK  │
│  TREND: ↑ Hot (+12% over season avg)                                  │
└──────────────────────────────────────────────────────────────────────┘

┌─ PROJECTION CALCULATION ─────────────────────────────────────────────┐
│  Base Projection:      25.2 PTS                                       │
│  Home/Away Adj:        -0.3 PTS                                      │
│  Opponent Def Rank:    -0.4 PTS (Elite D)                            │
│  ─────────────────────────────────────────────────────────────────── │
│  FINAL PROJECTION:     25.8 ± 6.2 PTS                                │
│  95% Confidence: 13.6 - 38.0 PTS                                    │
│  Distribution: Empirical Bootstrap (selected, samples=72)            │
└──────────────────────────────────────────────────────────────────────┘

┌─ OVER/UNDER: 25.5 PTS ───────────────────────────────────────────────┐
│  Distribution: Empirical Bootstrap                                    │
│  OVER  25.5:  51.8%  ██████████░░░░░░░░░░                           │
│  UNDER 25.5:  48.2%  █████████░░░░░░░░░░░                            │
│  Z-SCORE: (25.5 - 25.8) / 6.2 = -0.05                                │
│  ▸ RECOMMENDATION: PASS (too close)                                  │
│    - Edge: 1.8% above 50/50                                          │
└──────────────────────────────────────────────────────────────────────┘
```

A query will print a visible warning if the underlying projection was flagged `DEGRADED_FALLBACK` or `DEGRADED_MISSING`.

### 6. Ensemble Weight Optimization (Manual)

```bash
# Backtest a date range
python backtest.py --from 2026-04-15 --to 2026-05-01

# Tune ensemble weights on a holdout (manual, disabled by default)
python optimize_weights.py --from 2026-04-15 --to 2026-05-01 --verify-from 2026-05-02 --verify-to 2026-05-15

# Dry-run: evaluate a candidate without saving anything
python optimize_weights.py --from 2026-04-15 --to 2026-05-01 --dry-run

# Variance reduction (CRPS-driven)
python optimize_variance.py

# Drift detection across all targets
python -c "from src.evaluation.drift_detector import DriftDetector; print(DriftDetector('data/').detect())"
```

The optimizer:

1. Evaluates baseline weights on a holdout
2. Searches a component-derived parameter layout via `scipy.optimize` — per-target CatBoost/Transformer ratios and intercepts (12 parameters for six targets) plus a global CatBoost/MAE blend parameter only when an MAE companion model is loaded (13 parameters)
3. Re-evaluates the candidate and the current weights on the same verification window
4. If improvement exceeds threshold AND a verification run passes → atomic promotion of a new `blend_weights/v####.json`, and `current.json` is updated to a copy of the active version
5. Otherwise rejects — the store is untouched and the current weights stay; rejection is a valid outcome, not a failure

---

## Configuration Reference

`config/default.yaml` is the single source of truth.

```yaml
data:
  data_dir: "data"
  models_dir: "models"
  cache_dir: "cache"

training:
  targets: ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
  test_split_date: "2025-01-01"
  temporal_decay_lambda: 0.023
  outlier_percentile: 0.99
  use_sample_weights: true
  use_adversarial_validation: true
  use_feature_selection: true
  min_samples_per_player: 10

training_presets:
  full:
    default_mode: "standard"
    default_model_size: "M"
    transformer_enabled: true
    feature_engineer:
      enable_groups: ["all"]   # every registered group (built-ins + extensions)
  small:
    default_mode: "quick"
    default_model_size: "S"
    transformer_enabled: false
    feature_engineer:
      enable_groups: ["rolling", "efficiency", "momentum", "pace",
                      "opponent_strength", "archetype"]   # six core groups
  laptop_quality:
    default_mode: "standard"
    default_model_size: "S"
    transformer_enabled: false
    feature_selection:
      enabled: true            # smart selection on by default for this preset
      profile: balanced
    feature_engineer:
      enable_groups: ["rolling", "efficiency", "momentum", "context", "fatigue",
                      "minutes_confidence", "rest_density", "matchup",
                      "opponent_strength", "pace", "team_role", "recency_form",
                      "archetype"]   # 13 groups

catboost:
  iterations: 2000
  learning_rate: 0.03
  depth: 8
  l2_leaf_reg: 3.0

transformer:
  d_model: 128
  nhead: 8
  num_encoder_layers: 4

simulation:
  default_num_sims: 100
  use_gpu: true
  injury_probability_threshold: 0.1
  correlation_injection: true
  strict_mode: false   # overridable via --strict
```

---

## CLI Reference

| Command | Purpose |
|---------|---------|
| `python update_data.py` | Fetch NBA data |
| `python train.py` | Train models |
| `python simulate_season.py` | Run simulations |
| `python query_prob.py` | Interactive / one-shot probability queries |
| `python backtest.py` | Standalone backtest |
| `python optimize_weights.py` | Manually tune ensemble weights (opt-in, disabled by default) |
| `python optimize_variance.py` | Variance reduction via CRPS |
| `python check_contracts.py` | Validate artifact contract between steps |
| `python clear_cache.py --all --yes` | Wipe generated artifacts (keeps raw CSVs) |
| `python train_colab.ipynb` | Colab training notebook |

`train.py` and `simulate_season.py` have full `--help` output. The most important flags:

```bash
# train.py
--preset {small,full}                 # feature + model preset
--mode {quick,standard,full}          # training depth
--model-size {auto,S,M,L,XL}          # capacity
--parallel --max-workers N            # CatBoost target parallelism
--feature-ablation                    # pre-train group benchmark
--feature-selection {None,off,smart}  # opt into smart selector
--selection-profile {fast,balanced,max_accuracy}
--no-gpu                              # force CPU

# simulate_season.py
(--today | --date YYYY-MM-DD | --week | --season)
--sims N                              # simulations per matchup
--workers N                           # 1 recommended on GPU
--stat {mode,mean,both}
--strict                              # fail fast on degraded inputs
--no-csv
```

---

## Project Structure

```
knowing/
├── config/
│   └── default.yaml              # Master config (presets, paths, feature groups)
│
├── data/                          # Gitignored at root
│   ├── nba_players.csv            # Player game logs (AGE + POSITION required for lifecycle)
│   ├── nba_games.csv              # Team game logs
│   ├── injury_history.csv         # Incremental injury log
│   ├── cache/                     # aging_curves.csv, kan_aging_outputs.csv, archetype_covariances.npz
│   ├── sim_cache/                 # Roster cache (TTL)
│   └── sim_results/               # sim_results_*.csv, player_projections_*.csv (with DATA_QUALITY)
│
├── models/                        # Gitignored at root
│   ├── {stat}_catboost.cbm        # Per-target CatBoost
│   ├── {stat}_metadata.joblib     # Per-target metadata
│   ├── attention_transformer.pkl  # Transformer (optional)
│   ├── feature_schema.pkl
│   ├── feature_cols.pkl
│   ├── blend_weights.pkl          # Backward-compat shim → blend_weights/current.json
│   ├── blend_weights/             # Versioned JSON store
│   │   ├── v0001.json … vNNNN.json
│   │   └── current.json           # copy of the active version
│   ├── feature_selection_manifest.json  # From --feature-selection smart
│   └── model_stack_metadata.pkl
│
├── src/
│   ├── config/                    # config.py, model_config.py
│   ├── data/                      # Scrapers (10 modules)
│   ├── preprocessing/
│   │   ├── feature_engineer.py
│   │   ├── feature_engineer_gpu.py
│   │   ├── data_loader.py
│   │   └── features/              # 25+ FeatureGroup classes
│   ├── models/                    # CatBoost + Transformer + runtime utilities
│   ├── pipeline/                  # data/training/prediction pipelines
│   ├── training/                  # Modular training (pipeline, catboost_trainer, presets, ...)
│   ├── simulation/                # game_simulator, phase_simulator, archetype, role_sampler, input_health
│   ├── query/                     # probability_calculator, distribution_fitter, empirical_covariance, prob_formatter
│   ├── evaluation/                # backtest_runner, weight_store, ensemble_optimizer, drift_detector, smart_feature_selector, shadow_feature_filter, feature_group_ablation, metrics
│   ├── lifecycle/                 # aging_model (B-Ianus), kan_age_model
│   ├── contracts/                 # artifact/feature/projection/schedule schema validators
│   └── utils/                     # logging, reproducibility, team_mappings, prediction_utils
│
├── tests/                         # Mirrors src/ structure; new test_contracts/, test_evaluation/
│
├── cache/                         # Legacy scraper cache
├── experiments/                   # JSON-based experiment tracking
│
├── update_data.py                 # Data fetch orchestrator
├── train.py                       # Training orchestrator
├── simulate_season.py             # Simulation orchestrator
├── query_prob.py                  # Query CLI
├── backtest.py                    # Backtest CLI
├── optimize_weights.py            # Manual weight tuner CLI (opt-in)
├── optimize_variance.py           # CRPS variance reduction
├── check_contracts.py             # Artifact contract validator
├── clear_cache.py                 # Cleanup utility
│
├── AGENTS.md                      # Runtime + gotchas (NEW ENTRY POINT for AI agents)
├── DECISIONS.md                   # Decision records (DR-025, ...)
├── TASKS.md                       # Task log
├── IMPROVEMENTS_SUMMARY.md        # Historical improvements
├── cline_docs/                    # Cline IDE doc folder
│   ├── codebaseSummary.md         # Structural map of the code
│   ├── currentTask.md             # Current focus
│   ├── projectRoadmap.md          # Roadmap + future enhancements
│   ├── techStack.md               # Pinned tech stack
│   └── bugfixes_summary.md        # Historical bugfix log
├── project-brain/                 # Curated architectural brain
│
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Test dependencies
├── train_colab.ipynb              # Colab training notebook
└── query_prob.ipynb               # Jupyter query interface
```

---

## Testing

```bash
source venv/bin/activate
pytest tests/ -v                   # full suite (allow several minutes)
pytest tests/test_evaluation/ -v   # backtest + smart selector (fast)
pytest tests/test_contracts/ -v    # artifact contract smoke (fast)
pytest tests/test_query/ -v        # prob calculator (fast)
pytest -m "not slow"               # skip slow-marked
```

Custom markers: `slow`, `gpu`, `integration` (registered in `tests/conftest.py`).
`pytest-timeout` is not installed — bump the tool timeout for the full suite.

---

## Cache & Artifact Cleanup

```bash
# Preview what will be deleted
python clear_cache.py --all --dry-run

# Remove generated caches, reports, logs, and experiments while preserving models
python clear_cache.py --all --keep-models --yes

# Wipe generated artifacts, including trained models; raw CSVs in data/ are preserved
python clear_cache.py --all --yes
```

Removes: `cache/`, `data/cache/`, `data/sim_cache/`, `data/sim_results/`, `experiments/`, `reports/`, logs, `.DS_Store`, and all `__pycache__/`. `models/` is removed unless `--keep-models` is supplied.
Preserves: `data/nba_players.csv`, `data/nba_games.csv`, `data/injury_history.csv`, source code, venv.

---

## Development Notes

See `AGENTS.md` for the agent-facing gotchas (the source of truth for AI work in this repo). Highlights:

- **Supported model stack**: the `full` preset supports CatBoost + Transformer; `small` and `laptop_quality` are CatBoost-only. The deployed stack is artifact-dependent — run `python inspect_artifacts.py --models-dir models` instead of inferring it from the CLI default or this file.
- **CatBoost GPU**: `--max-workers 1` to avoid CUDA context contention.
- **Gitignore is root-only** (`/data/`, `/models/`), so `src/data/` and `src/models/` are tracked.
- **PyTorch test shim** in `src/__init__.py` activates under pytest on machines without a working PyTorch.
- **Lifecycle features need `AGE` + `POSITION`** in `nba_players.csv` — run `update_data.py` after the PlayerBioScraper is added.
- **Injury history is incremental** — first runs have sparse injury-risk features.
- **B-Ianus / KAN aging caches** live in `data/cache/`. Delete to force recompute. KAN always runs on CPU.
- **Ensemble weights are versioned** — never edit `blend_weights.pkl` directly; use `optimize_weights.py` or the `WeightStore` API.
- **Contracts validation** runs at startup in `train.py` and `simulate_season.py`. Run `python check_contracts.py` to debug.

---

## License

MIT — see `LICENSE`.
