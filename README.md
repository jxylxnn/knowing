# NBA Player Stats Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/catboost-1.2+-yellow.svg)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Optional-76b900.svg)](https://developer.nvidia.com/cuda-zone)

A production-grade machine learning system for predicting NBA player statistics using ensemble deep learning, advanced feature engineering, and GPU-accelerated Monte Carlo simulations.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Data Collection](#1-data-collection)
  - [Feature Engineering](#2-feature-engineering)
  - [Model Training](#3-model-training)
  - [Simulation Engine](#4-simulation-engine)
  - [Query System](#5-query-system)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This system predicts NBA player performance (PTS, REB, AST, STL, BLK, TOV) for upcoming games by combining:

- **5 Deep Learning Models** trained on historical player data
- **150+ Engineered Features** capturing performance trends, matchups, and context
- **GPU-Accelerated Monte Carlo Simulations** running 1000+ iterations per game in seconds
- **Real-Time Data Integration** from injury reports, betting lines, and lineups

The system is designed for sports analytics, fantasy sports optimization, and betting market analysis.

---

## Architecture

```
+---------------------------------------------------------------------------------+
|                           NBA PREDICTION SYSTEM                                  |
+---------------------------------------------------------------------------------+
|                                                                                  |
|  +------------------+    +--------------------+    +--------------------+        |
|  |   DATA LAYER     |    | FEATURE ENGINEERING |    |    MODEL LAYER     |        |
|  |   ----------     |    | -----------------   |    |    ---------       |        |
|  |                  |    |                     |    |                    |        |
|  |  * NBA.com API   |--->| * Rolling Averages  |--->| * CatBoost (x6)    |        |
|  |  * Injury Report |    | * Efficiency Stats  |    | * LSTM Temporal    |        |
|  |  * Betting Lines |    | * Bayesian Est.     |    | * Transformer      |        |
|  |  * Starting 5    |    | * Matchup History   |    | * GNN Chemistry    |        |
|  |  * Defense Data  |    | * Pace Adjustment   |    | * Joint Multi-NN   |        |
|  |  * Schedule      |    | * 15-Phase Pipeline |    | * Stacked Ensemble |        |
|  |                  |    |                     |    |                    |        |
|  +------------------+    +--------------------+    +--------------------+        |
|           |                                                 |                     |
|           |                                                 v                     |
|           |                                      +--------------------+             |
|           |                                      |  SIMULATION ENGINE |             |
|           |                                      |  -----------------  |             |
|           +------------------------------------->|                    |             |
|                                                  | * GPU Vectorization|             |
|                                                  | * Correlated Stats |             |
|                                                  | * Injury Modeling  |             |
|                                                  | * 1000 sims/sec    |             |
|                                                  +--------------------+             |
|                                                           |                      |
|                                                           v                      |
|                                                  +--------------------+             |
|                                                  |     QUERY CLI      |             |
|                                                  |   -----------      |             |
|                                                  |                    |             |
|                                                  | * Interactive REPL |             |
|                                                  | * Probability Calc |             |
|                                                  | * Player Compare   |             |
|                                                  | * JSON API Output  |             |
|                                                  +--------------------+             |
|                                                                                  |
+---------------------------------------------------------------------------------+
```

---

## Key Features

### Multi-Model Ensemble Architecture

| Model | Purpose | Key Strength |
|-------|---------|--------------|
| **CatBoost** | Primary predictor | Handles categorical features, GPU-accelerated |
| **LSTM** | Temporal patterns | Captures sequence dependencies in recent games |
| **Transformer** | Attention mechanism | Long-range dependencies, game context |
| **Graph Neural Network** | Team chemistry | Player synergy and lineup interactions |
| **Joint Neural Network** | Multi-output | Correlation between PTS/REB/AST |
| **Stacked Ensemble** | Meta-learner | Optimal blending of all models |

### Advanced Feature Engineering (150+ Features)

**Performance Features:**
- Rolling averages (3, 5, 10, 20, 50 game windows)
- Exponentially weighted moving averages (EWMA)
- Performance trends and streaks (hot/cold detection)

**Contextual Features:**
- Home/away splits and adjustments
- Rest days and back-to-back fatigue
- Minutes distribution and usage rate

**Matchup Features:**
- Historical performance vs specific opponents
- Opponent defensive rankings (PTS/REB/AST allowed)
- Pace-adjusted statistics

**Advanced Metrics:**
- True Shooting % (TS%), Effective FG% (eFG%)
- Usage rate and shot profiles
- Bayesian shrinkage estimates
- League percentile rankings

### GPU-Accelerated Simulation

- **Vectorized PyTorch**: 1000+ simulations per game in <1 second
- **Correlated Sampling**: Maintains realistic PTS/REB/AST correlations
- **Injury-Aware**: Real-time injury probability integration
- **Dirichlet Allocation**: Realistic stat distribution across teammates

### Real-Time Data Integration

- **Injury Reports**: Automatic injury status from NBA injury reports
- **Starting Lineups**: Confirmed starters before tip-off
- **Betting Lines**: Vegas totals and spreads for calibration
- **Defense Ratings**: Opponent defensive efficiency by stat

---

## Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Core ML** | PyTorch | 2.0+ |
| | CatBoost | 1.2+ |
| | scikit-learn | 1.8+ |
| | LightGBM | 4.6+ |
| **Data Processing** | pandas | 2.3+ |
| | numpy | 2.3+ |
| | scipy | 1.16+ |
| **NBA Data** | nba_api | 1.11+ |
| | beautifulsoup4 | 4.14+ |
| **Visualization** | plotly | 6.5+ |
| | matplotlib | 3.10+ |
| **NLP/Transformers** | transformers | 4.57+ |
| **Testing** | pytest | 9.0+ |
| | coverage | 7.13+ |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- (Optional) NVIDIA GPU with CUDA 11.8+ for acceleration

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jxylxnn/knowing.git
cd knowing

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fetch latest NBA data
python update_data.py

# Train models (first run takes ~10-15 minutes)
python train.py

# Run simulation for today's games
python simulate_season.py --today

# Query player probabilities (interactive)
python query_prob.py
```

### GPU Setup (Optional)

For GPU acceleration, ensure you have:
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 11.8+ installed
3. PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Pipeline Deep Dive

### 1. Data Collection

The system aggregates data from multiple sources:

| Source | Data | Update Frequency |
|--------|------|------------------|
| **NBA.com Stats API** | Player/Team game logs, box scores | Daily |
| **Basketball Reference** | Advanced stats, historical data | On-demand |
| **NBA Injury Report** | Player availability probabilities | Game day |
| **Rotowire Lineups** | Starting lineups | Pre-game |
| **Sportsbook APIs** | Betting lines, totals, spreads | Pre-game |
| **NBA Defense Data** | Opponent defensive ratings | Weekly |

**Data Files Generated:**
```
data/
+-- nba_players.csv      # 500K+ player game records
+-- nba_games.csv        # Team game logs
+-- cache/
|   +-- schedule_*.csv   # Daily schedules
|   +-- latest_injuries.csv
|   +-- defense_*.json   # Team defense ratings
+-- sim_results/
    +-- sim_results_*.csv         # Game predictions
    +-- player_projections_*.csv  # Player stats
```

### 2. Feature Engineering

The 15-phase feature engineering pipeline transforms raw game data into 150+ predictive features:

```
+-------------------------------------------------------------------------+
|              FEATURE ENGINEERING PIPELINE                                |
+-------------------------------------------------------------------------+
| Phase 1:  Rolling Features (AVG, STD, MIN, MAX, RANGE)                  |
| Phase 2:  Efficiency Features (TS%, eFG%, AST/TOV, Per-Minute)           |
| Phase 3:  Momentum Features (EWMA, Trends, Streaks)                      |
| Phase 4:  Contextual Features (Home/Away, Rest, Fatigue)                 |
| Phase 5:  Matchup History (Career vs Opponent)                           |
| Phase 6:  Advanced Scoring (Usage Rate, Shot Profiles)                   |
| Phase 7:  Pace-Adjusted Stats (Possessions, Pace Factor)                 |
| Phase 8:  Teammate Features (Role, Share)                                |
| Phase 9:  Opponent Strength (Defensive Ratings)                          |
| Phase 10: Interaction Features (Rest x Performance)                      |
| Phase 11: Bayesian Estimates (Shrinkage to League Avg)                   |
| Phase 12: Seasonality (Fourier Features, Day of Year)                    |
| Phase 13: Advanced Custom (SOS, Z-Score, Fantasy Pts)                    |
| Phase 14: League Rankings (Percentile by Stat)                           |
| Phase 15: Target Encoding (Player/Team Historical)                       |
+-------------------------------------------------------------------------+
```

**Key Feature Examples:**

```python
# Rolling Performance
ROLL_PTS_AVG_10           # 10-game rolling average points
ROLL_PTS_STD_10           # Volatility measure

# Efficiency
ROLL_TS_PCT_10            # True shooting percentage
ROLL_USG_PCT_10           # Usage rate

# Matchup Context
VS_OPP_PTS_AVG            # Career average vs this opponent
RELATIVE_OPP_DEF_PTS      # Opponent defense vs league avg

# Situational
FATIGUE_SCORE             # Combined rest/mins factor
IS_B2B                    # Back-to-back game flag
IS_HOME                   # Home court advantage

# Advanced
PTS_BAYESIAN              # Bayesian-adjusted projection
EFF_Z_SCORE               # Hot hand / slump indicator
```

### 3. Model Training

#### Training Pipeline

```python
# Temporal Decay Weighting
# Recent games weighted higher: weight = exp(-lambda x days_ago)
# lambda = 0.023 -> 30-day-old games get ~50% weight

# Adversarial Validation
# Detects train/test distribution drift
# Automatically adjusts for concept drift

# Target Capping
# Outliers capped at 99th percentile per player
# Prevents 60-point games from skewing model
```

#### Model Architecture

**CatBoost Configuration:**
```yaml
iterations: 2000
learning_rate: 0.03
depth: 8
l2_leaf_reg: 3.0
early_stopping_rounds: 100
task_type: GPU  # Auto-fallback to CPU
```

**LSTM Architecture:**
```
Input: 10-game sequence x features
+-- Embedding Layer (input_dim -> 128)
+-- LSTM Layer 1 (128 hidden, bidirectional)
+-- LSTM Layer 2 (128 hidden, bidirectional)
+-- Dropout (0.2)
+-- Dense Layer (256 -> 64)
+-- Output Layer (64 -> 3) [PTS, REB, AST]
```

**Transformer Architecture:**
```
Input: 50-game sequence x features
+-- Linear Embedding (input_dim -> 128)
+-- Positional Encoding
+-- Transformer Encoder (4 layers, 8 heads)
+-- Layer Norm + Dropout
+-- Global Average Pooling
+-- Output Layer (128 -> 3)
```

**GNN Architecture:**
```
Input: Player graph (nodes = players, edges = teammates)
+-- Node Features: Player stats embedding
+-- Edge Features: Games played together
+-- GraphConv Layer 1 (64 -> 128)
+-- GraphConv Layer 2 (128 -> 128)
+-- GraphConv Layer 3 (128 -> 64)
+-- Team Synergy Score: Mean node embedding
```

#### Ensemble Blending

Final predictions are weighted combinations:
```
Final_Prediction = 0.50 x CatBoost
                 + 0.15 x Joint_NN
                 + 0.15 x LSTM
                 + 0.15 x Transformer
                 + 0.05 x GNN
```

Weights are learned via Ridge regression on validation data.

### 4. Simulation Engine

#### GPU-Accelerated Monte Carlo

```python
# Vectorized simulation for N games x K simulations
# All operations on GPU tensors

# 1. Roster Building
active_players = sample_with_injury_probability(injury_probs)

# 2. Minutes Allocation (Dirichlet distribution)
minutes = allocate_minutes(expected_mins, usage_rates, total=240)

# 3. Stat Generation (Correlated Normal)
# Covariance Matrix:
#        PTS   REB   AST
# PTS   1.00  0.15 -0.05
# REB   0.15  1.00 -0.10
# AST  -0.05 -0.10  1.00

# 4. Clutch Adjustment
if game_total > 115:
    top_scorers += bonus_minutes
    bench -= minutes

# 5. Team Total Calibration
team_pts = sum(player_pts) + home_advantage
```

#### Simulation Output

```json
{
  "team_a": "LAL",
  "team_b": "BOS",
  "win_prob_a": 58.2,
  "team_summaries": {
    "LAL": {"pts": {"mean": 112.4, "mode": 111.0}, "reb": {...}},
    "BOS": {"pts": {"mean": 108.7, "mode": 109.0}, "reb": {...}}
  },
  "player_averages": [
    {
      "name": "LeBron James",
      "team": "LAL",
      "pts": 25.8,
      "reb": 7.2,
      "ast": 8.1,
      "pts_95_ci": [18.5, 33.2],
      "play_probability": 0.95
    }
  ]
}
```

### 5. Query System

#### Probability Calculation

The system calculates over/under probabilities using:

**Normal Distribution Method:**
```
P(Over) = 1 - Phi((line - mu) / sigma)

Where:
  mu = predicted mean
  sigma = predicted standard deviation
  Phi = standard normal CDF
```

**Monte Carlo Method (for complex scenarios):**
```python
simulations = np.random.normal(mean, std, n_sims)
prob_over = np.mean(simulations > line) * play_probability
prob_under = 1 - prob_over
```

**Injury Adjustment:**
```
P(Over_adjusted) = P(Over | Plays) x P(Plays)
P(Under_adjusted) = P(Under | Plays) x P(Plays) + P(DNP)
```

#### Recommendation Engine

```
Edge = |P(Over) - 0.5|

if Edge > 0.15:
    Recommend OVER/UNDER (strong)
elif Edge > 0.05:
    Recommend OVER/UNDER (moderate)
else:
    PASS (too close to 50/50)
```

---

## Configuration Reference

Edit `config/default.yaml` to customize behavior:

```yaml
# Data Configuration
data:
  data_dir: "data"
  models_dir: "models"
  cache_dir: "cache"

# Training Configuration
training:
  targets: ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
  test_split_date: "2024-03-01"
  temporal_decay_lambda: 0.023      # Weight decay for old games
  outlier_percentile: 0.99          # Cap outliers at this percentile
  use_sample_weights: true          # Enable temporal weighting
  use_adversarial_validation: true  # Detect train/test drift
  min_samples_per_player: 10        # Minimum games per player

# Feature Engineering
features:
  rolling_windows: [3, 5, 10, 20, 50]
  use_matchup_features: true
  use_fatigue_features: true
  use_momentum_features: true
  use_contextual_features: true

# Simulation Configuration
simulation:
  default_num_sims: 1000
  max_workers: 4
  use_gpu: true
  cache_rosters: true
  injury_probability_threshold: 0.1
  correlation_injection: true       # Maintain stat correlations
  clutch_adjustment: true           # Adjust for close games

# CatBoost Model
catboost:
  iterations: 2000
  learning_rate: 0.03
  depth: 8
  l2_leaf_reg: 3.0
  early_stopping_rounds: 100

# LSTM Model
lstm:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 512
  epochs: 100
  sequence_length: 10
  bidirectional: true

# Transformer Model
transformer:
  d_model: 128
  nhead: 8
  num_encoder_layers: 4
  dim_feedforward: 512
  max_seq_length: 50
  dropout: 0.2
  epochs: 100

# GNN Model
gnn:
  num_node_features: 64
  num_edge_features: 16
  num_graph_layers: 3
  hidden_size: 128
  dropout: 0.2

# Ensemble Configuration
ensemble:
  method: "ridge"       # Meta-learner type
  cv_folds: 5
  use_temporal_refinement: true
```

---

## API Reference

### Core Classes

#### `ModelManager`

Orchestrates training and prediction across all models.

```python
from src.models.model_manager import ModelManager

# Initialize
manager = ModelManager(data_dir='data', models_dir='models')

# Train all models
train_df, test_df = manager.prepare_data()
manager.train_all(train_df)
results = manager.evaluate_all(test_df)

# Single player prediction
predictions = manager.predict_player_stats(player_context_df, history_df)
# Returns: {'PTS': 25.3, 'REB': 7.1, 'AST': 8.2, ...}

# Batch prediction (multiple players)
predictions_df = manager.predict_player_stats_batch(context_df, histories_map)
```

#### `GameSimulator`

GPU-accelerated Monte Carlo game simulation.

```python
from src.simulation.game_simulator import GameSimulator
from src.models.model_manager import ModelManager

manager = ModelManager()
manager._load_models()

simulator = GameSimulator(manager)

# Simulate single game
result = simulator.simulate_matchup(
    team_a='LAL',
    team_b='BOS',
    num_sims=1000
)

# Access results
print(f"LAL Win Probability: {result['win_prob_a']:.1f}%")
print(f"Predicted Score: LAL {result['team_summaries']['LAL']['pts']['mean']:.1f} - "
      f"BOS {result['team_summaries']['BOS']['pts']['mean']:.1f}")
```

#### `ProbabilityCalculator`

Calculate over/under probabilities.

```python
from src.query.probability_calculator import ProbabilityCalculator

calc = ProbabilityCalculator()

# From mean/std
result = calc.calculate_from_projection(
    player_name="LeBron James",
    stat="pts",
    line=25.5,
    mean=27.2,
    std=6.8,
    opponent="BOS",
    play_probability=0.95
)

print(f"OVER {result.line}: {result.prob_over*100:.1f}%")
print(f"Recommendation: {result.recommendation}")

# From simulations
result = calc.run_monte_carlo_simulation(
    player_name="LeBron James",
    stat="pts",
    line=25.5,
    mean=27.2,
    std=6.8,
    num_sims=5000
)
```

#### `FeatureEngineer`

Create 150+ features from raw game data.

```python
from src.preprocessing.feature_engineer import FeatureEngineer

fe = FeatureEngineer(rolling_windows=[3, 5, 10, 20, 50])

# Transform raw data
features_df = fe.create_features(merged_df, is_training=True)

# Get feature names
feature_cols = fe._select_features(features_df)
print(f"Generated {len(feature_cols)} features")
```

---

## Usage Examples

### 1. Update Data

```bash
# Update current season data (default)
python update_data.py

# Fetch specific season
python update_data.py --season 2024-25

# Fetch multiple specific seasons
python update_data.py --season 2023-24 --season 2024-25

# Interactive season selection (RECOMMENDED for first-time setup)
python update_data.py --interactive
python update_data.py -i

# Fetch last 10 seasons (2015-16 to present)
python update_data.py --all-seasons

# Full historical scrape (ALL NBA seasons since 1946-47)
python update_data.py --full-scrape

# Force refresh (ignore existing data)
python update_data.py --all-seasons --force
```

**Interactive Mode:**
```
======================================================================
                         NBA SEASONS
======================================================================
    1. 1946-47    2. 1947-48    3. 1948-49    4. 1949-50
    5. 1950-51    6. 1951-52  ...            80. 2025-26
======================================================================

Input options:
  - Single:     5
  - Multiple:   1,5,10,15
  - Range:      50-80
  - Combined:   1,5,10-15,20
  - 'all'       All NBA seasons (1946-47 to present)
  - 'recent'    Last 10 seasons
======================================================================

Select seasons: 50-80
```

> **Note:** nba_api has reliable data from 1996-97 onward (~season 50). Earlier seasons may have limited or missing data.

### 2. Train Models

```bash
# Train all models
python train.py
```

**Output:**
```
--- NBA Game Simulator: Training Phase ---
Step 1: Preparing Data and Engineering Features...
Phase 1/15: Creating Rolling Features...
...
Feature engineering complete. Final shape: (458231, 187)

Step 2: Training Stacked Ensemble Models (PTS, REB, AST)...
Training Advanced CatBoost for: PTS
...
Training Joint Stats NN...
Training sequence models...

Step 3: Evaluating Performance on Hold-out Test Set...
--- Model Performance Results ---
PTS Prediction Stats:
  MAE: 4.82
  RMSE: 6.31
REB Prediction Stats:
  MAE: 2.15
  RMSE: 2.89
...
Training and Evaluation Complete. Models saved in './models/'
```

### 3. Run Simulations

```bash
# Today's games
python simulate_season.py --today

# Specific date
python simulate_season.py --date 2026-02-20

# Next 7 days
python simulate_season.py --week

# Remaining season
python simulate_season.py --season

# High-accuracy mode
python simulate_season.py --today --sims 1000 --stat both
```

**Output:**
```
===============================================================================
GAME: LAL @ BOS - Feb 19, 2026
===============================================================================
LAL Win Probability: 52.3%

              PTS        REB        AST
LAL         112.4       44.2       26.8
BOS         110.1       42.8       24.5

Top Performers:
  LeBron James (LAL): 25.8 pts, 7.2 reb, 8.1 ast
  Jayson Tatum (BOS): 24.3 pts, 8.1 reb, 5.2 ast

Game predictions exported to: data/sim_results/sim_results_20260216_124413.csv
Player projections exported to: data/sim_results/player_projections_20260216_124413.csv
```

### 4. Query Probabilities

**Interactive Mode:**
```bash
python query_prob.py
```

```
NBA Probability Query Tool
Type 'help' for commands

> LeBron James over 25.5 pts vs BOS

===============================================================================
LeBron James (LAL) vs BOS - 2026-02-19
===============================================================================

+-- RECENT PERFORMANCE (Last 5 Games) --------------------------------------+
|  DATE        MIN    PTS    REB    AST   FG%     RESULT                    |
|  Feb 16    36.1     28       8       9   52%    W LAL vs. MIA            |
|  Feb 14    35.2     22       6       7   48%    W LAL @ CHA              |
|  Feb 12    37.8     31       7      10   55%    L LAL vs. DEN            |
|  Feb 10    34.5     19       5       6   44%    W LAL vs. DET            |
|  Feb 08    38.2     27       9       8   51%    W LAL @ ATL              |
+---------------------------------------------------------------------------+
|  5-GAME AVG: 36.4 MIN, 25.4 PTS, 7.0 REB, 8.0 AST                        |
|  TREND: ^ Hot (+12% over season avg)                                      |
+---------------------------------------------------------------------------+

+-- MATCHUP ANALYSIS -------------------------------------------------------+
|  OPPONENT: BOS                                                            |
|  Defense vs Points:                                                       |
|    * Points Allowed: 108.2/100 poss (#4 in NBA - Elite)                   |
|  vs BOS (Last 5 meetings):                                                |
|    * Average vs BOS: 26.4 Points                                          |
|  Venue: AWAY (expected adjustment: -0.3 Points)                           |
+---------------------------------------------------------------------------+

+-- OVER/UNDER: 25.5 POINTS ------------------------------------------------+
|  Distribution: Normal(mu=25.8, sigma=6.2)                                 |
|  OVER  25.5:  51.8%  ##########-----------                                |
|  UNDER 25.5:  48.2%  #########------------                                |
|                                                                           |
|  > RECOMMENDATION: PASS (too close)                                       |
|    - Edge: 1.8% above 50/50                                              |
+---------------------------------------------------------------------------+
```

**One-Shot Mode:**
```bash
# Basic query
python query_prob.py -p "LeBron James" -s pts -l 25.5

# With opponent
python query_prob.py -p "Jokic" -s reb -l 12.5 -o BOS

# JSON output (for scripting)
python query_prob.py -p "Curry" -s pts -l 30.5 --json

# Compare players
python query_prob.py --compare "LeBron James" "Jayson Tatum" --stat pts

# List available data
python query_prob.py --list-players
python query_prob.py --list-teams
```

---

## Performance Metrics

### Model Accuracy (Test Set)

| Stat | MAE | RMSE | R^2 |
|------|-----|------|-----|
| **PTS** | 4.82 | 6.31 | 0.72 |
| **REB** | 2.15 | 2.89 | 0.68 |
| **AST** | 1.89 | 2.54 | 0.71 |
| **STL** | 0.82 | 1.12 | 0.45 |
| **BLK** | 0.64 | 0.89 | 0.52 |
| **TOV** | 1.12 | 1.48 | 0.58 |

### Simulation Performance

| Hardware | Sims/Game | Time/Game |
|----------|-----------|-----------|
| RTX 4090 | 10,000 | 0.3s |
| RTX 3080 | 10,000 | 0.5s |
| CPU (16-core) | 1,000 | 2.1s |

### Feature Importance (Top 10 for PTS)

1. `ROLL_PTS_AVG_10` - 10-game rolling average
2. `ROLL_MIN_AVG_10` - Expected minutes
3. `ROLL_USG_PCT_10` - Usage rate
4. `PTS_PLAYER_TE` - Target encoding
5. `VS_OPP_PTS_AVG` - Matchup history
6. `ROLL_TS_PCT_10` - Efficiency
7. `IS_HOME` - Home court
8. `REST_DAYS` - Rest advantage
9. `PTS_BAYESIAN` - Bayesian estimate
10. `RELATIVE_OPP_DEF_PTS` - Opponent defense

---

## Project Structure

```
knowing/
+-- config/
|   +-- default.yaml              # Configuration file
|
+-- data/                         # Data storage (gitignored)
|   +-- nba_players.csv           # Player game logs
|   +-- nba_games.csv             # Team game logs
|   +-- cache/                    # Scraped data cache
|   |   +-- schedule_*.csv
|   |   +-- latest_injuries.csv
|   |   +-- defense_*.json
|   +-- sim_results/              # Simulation outputs
|       +-- sim_results_*.csv
|       +-- player_projections_*.csv
|
+-- models/                       # Trained models (gitignored)
|   +-- pts_catboost.cbm          # Points model
|   +-- reb_catboost.cbm          # Rebounds model
|   +-- ast_catboost.cbm          # Assists model
|   +-- stl_catboost.cbm          # Steals model
|   +-- blk_catboost.cbm          # Blocks model
|   +-- tov_catboost.cbm          # Turnovers model
|   +-- joint_stats_nn.pt         # Multi-output NN
|   +-- temporal_lstm.pkl         # LSTM model
|   +-- attention_transformer.pkl # Transformer model
|   +-- team_chemistry_gnn.pkl    # GNN model
|   +-- blenders.pkl              # Ensemble weights
|   +-- feature_cols.pkl          # Feature column names
|
+-- src/
|   +-- config/                   # Configuration handling
|   +-- data/                     # Data scrapers
|   |   +-- basketball_ref_scraper.py
|   |   +-- betting_scraper.py
|   |   +-- injury_scraper.py
|   |   +-- lineup_scraper.py
|   |   +-- nba_defense_scraper.py
|   |   +-- schedule_scraper.py
|   |
|   +-- models/                   # ML models
|   |   +-- model_manager.py      # Training orchestrator
|   |   +-- stacked_ensemble.py   # Ensemble model
|   |   +-- lstm_model.py         # LSTM temporal
|   |   +-- transformer_model.py  # Attention model
|   |   +-- gnn_model.py          # Graph neural network
|   |   +-- multi_output_nn.py    # Joint NN
|   |   +-- temporal_attention.py # Advanced temporal
|   |   +-- advanced_trainer.py   # Training utilities
|   |   +-- gpu_utils.py          # GPU handling
|   |
|   +-- pipeline/                 # Training & prediction
|   |   +-- training_pipeline.py
|   |   +-- prediction_service.py
|   |   +-- data_pipeline.py
|   |
|   +-- preprocessing/            # Feature engineering
|   |   +-- data_loader.py
|   |   +-- feature_engineer.py
|   |   +-- features/
|   |       +-- base.py
|   |       +-- rolling.py
|   |
|   +-- query/                    # Probability queries
|   |   +-- probability_calculator.py
|   |   +-- projection_loader.py
|   |   +-- query_parser.py
|   |   +-- interactive_cli.py
|   |
|   +-- simulation/               # Monte Carlo simulation
|   |   +-- game_simulator.py
|   |   +-- season_simulator.py
|   |   +-- report_generator.py
|   |   +-- enhanced_game_simulator.py
|   |   +-- four_factors_engine.py
|   |   +-- game_context_engine.py
|   |   +-- lineup_predictor.py
|   |   +-- player_correlation_engine.py
|   |   +-- possession_simulator.py
|   |
|   +-- utils/                    # Utilities
|       +-- logging_config.py
|       +-- prediction_utils.py
|       +-- reproducibility.py
|       +-- team_mappings.py
|
+-- tests/                        # Unit tests
|   +-- test_config/
|   +-- test_models/
|   +-- test_pipeline/
|   +-- test_preprocessing/
|
+-- train.py                      # Model training script
+-- update_data.py                # Data fetch script
+-- simulate_season.py            # Season simulation script
+-- query_prob.py                 # Probability query tool
+-- requirements.txt              # Dependencies
+-- README.md                     # This file
+-- .gitignore
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install pytest pytest-cov coverage

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings for public methods
- Keep functions under 50 lines

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NBA Stats API for providing comprehensive basketball data
- CatBoost team for the excellent gradient boosting library
- PyTorch team for GPU-accelerated deep learning
- Basketball Reference for historical data validation

---

*Built for sports analytics enthusiasts, fantasy sports players, and anyone interested in applying machine learning to basketball prediction.*
