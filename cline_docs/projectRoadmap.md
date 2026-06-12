# NBA Player Stats Prediction System - Project Roadmap

## Project Overview

A production-grade machine learning system for predicting NBA player statistics using ensemble deep learning, advanced feature engineering, and GPU-accelerated Monte Carlo simulations.

---

## Main Goals

### Goal 1: Accurate Player Stat Prediction
Predict NBA player performance (PTS, REB, AST, STL, BLK, TOV) for upcoming games with high accuracy using multi-model ensemble architecture.

### Goal 2: Real-Time Data Integration
Aggregate data from multiple sources (NBA.com API, injury reports, betting lines, lineups) for up-to-date predictions.

### Goal 3: Intuitive Query Interface
Provide an interactive CLI for probability queries and player comparisons.

---

## Key Features

- [x] Multi-model ensemble (CatBoost primary + Transformer secondary)
- [x] 150+ engineered features with 25+ toggleable FeatureGroup modules
- [x] GPU-accelerated Monte Carlo simulations with archetype-conditioned correlations
- [x] Real-time injury/lineup integration
- [x] Interactive probability query CLI with distribution zoo
- [x] Season simulation capabilities
- [x] Comprehensive configuration system
- [x] Self-optimizing ensemble weight tuner
- [x] Smart feature selection (shadow filter + group ablation + permutation importance)
- [x] Data quality / strict mode for degraded inputs
- [x] Inter-step artifact contracts validation

---

## Completion Criteria

### Core System (Completed)
- [x] Data collection from multiple sources
- [x] Modular FeatureGroup architecture (25+ toggleable groups)
- [x] Model training with temporal decay weighting
- [x] CatBoost + Transformer ensemble with versioned blend weights
- [x] GPU-accelerated game simulation with archetype-conditioned copulas
- [x] Probability calculation with distribution zoo (empirical/gamma/Poisson/NB/ZIP/Normal)
- [x] Inter-step artifact contracts (`src/contracts/`)
- [x] Self-optimizing weight tuner with rollback (`src/evaluation/ensemble_optimizer.py`)

---

## Completed Tasks

- [x] Project structure and configuration system
- [x] NBA data scrapers (NBA.com API, Basketball Reference)
- [x] Injury report integration (ESPN)
- [x] Betting lines scraper (Action Network)
- [x] Starting lineup scraper (Rotowire)
- [x] Player bio scraper (NBA.com — populates AGE + POSITION)
- [x] Modular FeatureGroup architecture (25+ groups, each independently toggleable)
- [x] CatBoost model training with multi-loss and quantile regression
- [x] Transformer attention model implementation
- [x] Modular training pipeline v2.0 (parallel targets, feature caching, experiment tracking)
- [x] GPU-accelerated game simulator with phase-by-phase progression
- [x] Archetype-conditioned empirical copula correlation engine
- [x] Role state sampling (limited/normal/expanded/starter/bench/closer)
- [x] Four factors engine + game context engine
- [x] Interactive query CLI with probability calculations
- [x] Distribution zoo (empirical bootstrap, gamma, Poisson, NB, ZIP, Normal)
- [x] Season simulation capabilities
- [x] Self-optimizing ensemble weight system (backtest + optimize + drift detection)
- [x] Smart feature selection (shadow filter + group ablation + permutation + stability)
- [x] Input health reporting + strict mode (`--strict` flag)
- [x] DATA_QUALITY column on player projections
- [x] Inter-step artifact contract validation
- [x] B-Ianus Bayesian aging curves + KAN aging factors (lifecycle ML)
- [x] Season phase, team motivation, postseason context feature groups
- [x] Unit test suite (260+ test functions across 32 test files)

---

## Future Enhancements (Potential)

### Model Improvements
- [ ] Wire LightGBM / XGBoost into the active training pipeline (installed but not wired)
- [ ] Add cross-position models (guard/forward/center specific)
- [ ] Incorporate player tracking data (shot distance, speed, distance to basket)
- [ ] Add uncertainty calibration with isotonic regression for quantiles

### Data Enhancements
- [ ] Include coaching impact factors (rotation tendencies, play style)
- [ ] Add travel impact analysis (time zones crossed, miles traveled)
- [ ] Historical playoff performance data
- [ ] Depth chart position information

### System Improvements
- [ ] REST API for external access
- [ ] Web dashboard interface
- [ ] Automated daily predictions
- [ ] Model performance monitoring dashboard

---

## Notes

- Current system is feature-complete for core use cases
- Active stack: CatBoost (primary) + Transformer (secondary). LSTM and GNN are `enabled: false`.
- Ensemble weights are versioned in `models/blend_weights/` — not hardcoded in source.
- GPU acceleration optional but recommended for high-volume simulations
- Data directories (`data/`, `models/`, `cache/`) are gitignored and created on first use
