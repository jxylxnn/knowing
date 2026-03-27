# Codebase Summary

## Overview

This is a production-grade machine learning system for predicting NBA player statistics. It combines ensemble deep learning, advanced feature engineering, and GPU-accelerated Monte Carlo simulations to predict player performance (PTS, REB, AST, STL, BLK, TOV) for upcoming games.

---

## Key Components and Their Interactions

### 1. Data Layer (`src/data/`)

Data scrapers that fetch information from various external sources:

| File | Purpose | Data Source |
|------|---------|-------------|
| `basketball_ref_scraper.py` | Historical stats, advanced metrics | Basketball Reference |
| `nba_defense_scraper.py` | Team defensive ratings | NBA.com Stats API |
| `injury_scraper.py` | Player injury status | ESPN Health API |
| `lineup_scraper.py` | Starting lineups | Rotowire |
| `rotowire_lineup_scraper.py` | Daily lineup projections | Rotowire |
| `betting_scraper.py` | Vegas lines, spreads, totals | Action Network |
| `schedule_scraper.py` | Game schedules | NBA.com Stats API |

**Output**: Raw data stored in `data/` directory and cached in `cache/`

### 2. Feature Engineering (`src/preprocessing/`)

Transforms raw game data into 150+ predictive features:

| File | Purpose |
|------|---------|
| `feature_engineer.py` | Main 15-phase feature pipeline |
| `data_loader.py` | Data loading and merging utilities |
| `features/base.py` | Base feature engineering classes |
| `features/rolling.py` | Rolling average/statistical features |

**15-Phase Pipeline**:
1. Rolling Features (AVG, STD, MIN, MAX, RANGE)
2. Efficiency Features (TS%, eFG%, Per-Minute)
3. Momentum Features (EWMA, Trends, Streaks)
4. Contextual Features (Home/Away, Rest, Fatigue)
5. Matchup History (Career vs Opponent)
6. Advanced Scoring (Usage Rate, Shot Profiles)
7. Pace-Adjusted Stats
8. Teammate Features (Role, Share)
9. Opponent Strength (Defensive Ratings)
10. Interaction Features (Rest x Performance)
11. Bayesian Estimates (Shrinkage to League Avg)
12. Seasonality (Fourier Features)
13. Advanced Custom (SOS, Z-Score)
14. League Rankings (Percentile by Stat)
15. Target Encoding (Player/Team Historical)

### 3. Model Layer (`src/models/`)

Multi-model ensemble architecture:

| File | Purpose | Architecture |
|------|---------|--------------|
| `model_manager.py` | Orchestrates training/prediction | Meta-coordinator |
| `gpu_utils.py` | GPU handling | CUDA detection, tensor operations |
| `transformer_model.py` | Attention mechanism | Transformer encoder |
| `minutes_predictor.py` | Minutes allocation | Regression model |
| `base.py` | Base model classes | Abstract interfaces |
| `error_calibration.py` | Uncertainty calibration | Error distribution modeling |

**Ensemble Weights**:
```
Final_Prediction = 0.50 x CatBoost
                 + 0.15 x Joint_NN
                 + 0.15 x LSTM
                 + 0.15 x Transformer
                 + 0.05 x GNN
```

### 4. Simulation Engine (`src/simulation/`)

GPU-accelerated Monte Carlo game simulation:

| File | Purpose |
|------|---------|
| `game_simulator.py` | Main game simulation |
| `season_simulator.py` | Full season simulation |
| `four_factors_engine.py` | Four factors model |
| `game_context_engine.py` | Game context adjustments |
| `player_correlation_engine.py` | Stat correlation handling |
| `possession_simulator.py` | Possession-level simulation |
| `report_generator.py` | Output formatting |

### 5. Query System (`src/query/`)

Interactive CLI for probability queries:

| File | Purpose |
|------|---------|
| `interactive_cli.py` | REPL interface |
| `probability_calculator.py` | Over/under probability calculation |
| `projection_loader.py` | Load player projections |
| `query_parser.py` | Parse natural language queries |

### 6. Pipeline Layer (`src/pipeline/`)

Orchestrates end-to-end workflows:

| File | Purpose |
|------|---------|
| `data_pipeline.py` | Data fetching and preprocessing |
| `training_pipeline.py` | Model training workflow |
| `prediction_service.py` | Prediction serving |

### 7. Configuration (`src/config/`, `config/`)

Centralized configuration management:

| File | Purpose |
|------|---------|
| `config/default.yaml` | Main configuration file |
| `src/config/config.py` | Configuration loading |
| `src/config/model_config.py` | Model-specific configs |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

External APIs          Data Layer           Preprocessing        Models
─────────────          ──────────           ────────────        ──────
    │                      │                     │                  │
    ▼                      ▼                     ▼                  ▼
┌──────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────┐
│NBA Stats │───▶│ Data Scrapers   │───▶│ Feature         │───▶│ CatBoost │
│API       │    │ (src/data/)     │    │ Engineer        │    │ (x6)     │
└──────────┘    └─────────────────┘    │ (15 phases)     │    ├──────────┤
                                        └─────────────────┘    │ LSTM     │
┌──────────┐    ┌─────────────────┐           │              ├──────────┤
│ESPN      │───▶│ data/           │───────────┘              │Transform │
│Injuries  │    │ nba_players.csv │                          ├──────────┤
└──────────┘    │ nba_games.csv   │                          │ GNN      │
                                        ┌─────────────────┐    ├──────────┤
┌──────────┐    ┌─────────────────┐    │ Training        │    │ Joint NN │
│Rotowire  │───▶│ cache/          │───▶│ Pipeline        │    ├──────────┤
│Lineups   │    │ injuries.csv    │    │ (train.py)      │    │ Ensemble │
└──────────┘    │ schedules.csv   │    └─────────────────┘    └──────────┘
                                        │
┌──────────┐    ┌─────────────────┐     │              ┌─────────────────┐
│Action    │───▶│ models/         │─────┘              │ Simulation      │
│Network   │    │ *.cbm, *.pt     │                    │ Engine          │
└──────────┘    └─────────────────┘                    │ (game_simulator)│
                                                         └────────┬────────┘
                                                                  │
                                                         ┌────────▼────────┐
                                                         │ Query CLI       │
                                                         │ (query_prob.py) │
                                                         └─────────────────┘
```

---

## External Dependencies

### APIs (Rate-Limited)
- **NBA.com Stats API**: Player/Team game logs, box scores
  - Rate limit: ~0.6s between requests
- **ESPN Health API**: Injury reports
- **Basketball Reference**: Advanced stats, historical data
- **Rotowire**: Starting lineups, projections
- **Action Network**: Betting lines, spreads, totals

### Python Packages
See `requirements.txt` and `techStack.md` for complete list.

### Cache Strategy
- Schedule: 1 hour TTL
- Betting lines: 4 hours TTL
- Defense stats: 12 hours TTL
- Lineups: 6 hours TTL
- Injuries: 30 minutes TTL

---

## Recent Significant Changes (March 2026)

### Model Improvements
- **XGBoost/LightGBM Configuration Added**: Added XGBoost and LightGBM configurations in `src/config/model_config.py`
  - Auto-sized models based on hardware compute score
  - GPU-accelerated training with hist method (XGBoost) and leaf-wise growth (LightGBM)
  - Configurable regularization parameters per model type

- **Calibrated Quantile Models**: Enhanced quantile regression in `src/models/model_manager.py`
  - Added `_calibrate_quantile()` method with residual-based adjustments
  - Better uncertainty estimates for prediction intervals

### Feature Engineering Enhancements
- **Defensive Matchup Features** (6 new feature categories):
  - `DEF_MATCHUP_*_IMPACT` - Defense vs player performance interaction
  - `OPP_DEF_RATING` - Combined defensive difficulty metric
  - `DEF_OVERAGE_*` - How players perform against elite defenses
  - `DEF_MATCHUP_TREND_*` - Rolling trend vs tough defenses
  - `DEF_MATCHUP_HOME_ADJ` - Home court advantage adjustments based on opponent defense
  - `OPP_DEF_RANK` - Defensive ranking and avoidance scores

- **Enhanced Opponent Strength**: Improved opponent defensive strength calculations

### Performance Targets (Achieved)
- PTS prediction: MAE ~4.82, RMSE ~6.31
- REB prediction: MAE ~2.15, RMSE ~2.89
- AST prediction: MAE ~1.89, RMSE ~2.54
- Simulation speed: 1000+ sims/game in <1 second (GPU)

---

## User Feedback Integration

Not applicable - this is a personal project without formal user feedback tracking.

---

## Entry Points

### Main Scripts (Root Directory)

| Script | Purpose | Usage |
|--------|---------|-------|
| `update_data.py` | Fetch NBA data | `python update_data.py --all-seasons` |
| `train.py` | Train all models | `python train.py` |
| `simulate_season.py` | Run simulations | `python simulate_season.py --today` |
| `query_prob.py` | Interactive queries | `python query_prob.py` |

### Notebooks
- `train_colab.ipynb`: Google Colab training notebook
- `query_prob.ipynb`: Jupyter query interface

---

## File Storage

### Gitignored Directories (Created at Runtime)
- `data/`: Player game logs, team data
- `models/`: Trained model files
- `cache/`: API response cache
- `logs/`: Application logs (if enabled)

### Generated Files
```
data/
├── nba_players.csv          # 500K+ player game records
├── nba_games.csv            # Team game logs
├── cache/
│   ├── schedule_*.csv       # Daily schedules
│   ├── latest_injuries.csv
│   └── defense_*.json       # Team defense ratings
└── sim_results/
    ├── sim_results_*.csv    # Game predictions
    └── player_projections_*.csv

models/
├── pts_catboost.cbm         # Points model
├── reb_catboost.cbm         # Rebounds model
├── ast_catboost.cbm         # Assists model
├── stl_catboost.cbm         # Steals model
├── blk_catboost.cbm         # Blocks model
├── tov_catboost.cbm         # Turnovers model
├── attention_transformer.pkl # Transformer
├── blenders.pkl             # Ensemble weights
└── feature_cols.pkl         # Feature names
```

---

## Testing

### Test Structure
```
tests/
├── conftest.py              # Pytest fixtures
├── test_config/
│   └── test_config.py       # Configuration tests
├── test_models/
│   └── test_model_manager.py # Model tests
├── test_pipeline/
│   └── test_data_pipeline.py # Pipeline tests
├── test_preprocessing/
│   └── test_feature_engineer.py # Feature tests
├── test_query/
│   └── test_interactive_cli.py # Query tests
└── test_simulation/
    └── test_game_simulator.py # Simulation tests
```

### Running Tests
```bash
source venv/bin/activate
pytest tests/ -v
```

**Note**: One pre-existing test (`test_registry_initialization`) has a known PosixPath vs string assertion mismatch.

---

## Configuration

Main configuration file: `config/default.yaml`

Key configuration sections:
- `data`: Paths and storage settings
- `training`: Training parameters, targets, decay
- `features`: Feature engineering settings
- `simulation`: Monte Carlo parameters
- `catboost`, `lstm`, `transformer`, `gnn`: Model configs
- `ensemble`: Blending configuration
- `api`: API endpoints and rate limits
- `cache`: TTL settings
