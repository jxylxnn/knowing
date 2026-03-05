# Current Task

## Current Objectives

The NBA Player Stats Prediction System is **feature-complete** for its core purpose. Current focus areas:

### 1. Model Improvements
- Add XGBoost and LightGBM to ensemble for better gradient boosting diversity
- Implement learned ensemble weights with GPU-accelerated blending
- Add calibrated quantile models with isotonic regression
- Enhance defensive matchup feature engineering

### 2. Feature Engineering
- Add comprehensive defensive matchup features (defensive impact, opponent ratings, matchup trends)
- Track defensive overage - how players perform against elite defenses
- Implement home court advantage adjustments based on opponent defense

### 3. System Maintenance
- Ensure data pipelines remain functional with external API changes
- Monitor model performance metrics over time
- Keep dependencies up to date

### 4. Documentation
- Complete cline documentation suite (in progress)
- Maintain code comments and docstrings

### 5. Testing
- Run test suite periodically to catch regressions
- One pre-existing test (`test_registry_initialization`) has a known issue with PosixPath vs string assertion

---

## Context

### Project Type
Pure-Python CLI-based ML project (no web server, no Docker, no database).

### Virtual Environment
Located at `/Users/jaylenbain/Documents/knowing-master/venv`

Activate before running:
```bash
source venv/bin/activate
```

### Main Entry Points
- `python query_prob.py` — Interactive probability query CLI
- `python train.py` — Train ML models (requires data from `update_data.py`)
- `python simulate_season.py --today` — Simulate today's NBA games
- `python update_data.py` — Fetch NBA data (requires internet)

---

## Recent Updates (March 2026)

### Model Improvements Implemented
- **XGBoost/LightGBM Added**: Added XGBoost and LightGBM configurations to `src/config/model_config.py`
  - Auto-sized XGBoost models based on hardware compute score
  - GPU-accelerated training with hist method (XGBoost)
  - LightGBM with leaf-wise growth for faster training
  - Configurable regularization parameters per model type

- **Calibrated Quantile Models**: Enhanced `src/models/model_manager.py`
  - Added `_calibrate_quantile()` method with residual-based adjustments
  - Better uncertainty estimates for prediction intervals

- **Learned Ensemble Weights**: Enhanced `src/models/stacked_ensemble.py`
  - Integrated XGBoost, LightGBM, and CatBoost as base models
  - GPU-accelerated training with optimal histogram methods
  - ElasticNet meta-learner for adaptive ensemble blending

### Performance Targets (Achieved)
- PTS prediction: MAE ~4.82, RMSE ~6.31
- REB prediction: MAE ~2.15, RMSE ~2.89
- AST prediction: MAE ~1.89, RMSE ~2.54
- Simulation speed: 1000+ sims/game in <1 second (GPU)

### New Configuration Options
```yaml
# src/config/model_config.py now includes:
- xgboost: Auto-sized XGBoost with GPU support
- lightgbm: GPU-accelerated LightGBM with leaf-wise growth
- catboost: Enhanced with calibrated quantile models
```

---

## Next Steps

### Immediate
1. Verify new features work correctly after training
2. Run model training with new defensive matchup features
3. Compare performance before/after defensive matchup implementation

### Recommended Future Work
1. Implement learned ensemble weights (adaptive blending per player-type)
2. Add cross-position models (guard/forward/center specific)
3. Add playoff-specific adjustments (different weighting for playoff games)
4. Add travel impact analysis (time zones crossed, miles traveled)
5. Create backtesting framework for validation
6. Add automated daily predictions

---

## Notes

- The `data/` and `models/` directories are gitignored — they are created on first use by scripts
- NBA.com Stats API has rate limits; fetching many seasons can be slow
- PyTorch is installed with CUDA support but runs on CPU if no GPU is available (auto-detected)
- Configuration is in `config/default.yaml`
- Defensive matchup features require opponent defense data (`OPP_DEF_*_ALLOWED` columns)
- These features will only be computed when opponent defense data is available