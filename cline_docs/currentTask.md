# Current Task

## System Status: Feature-Complete Core

The system is in a mature, feature-complete state. All core pipeline components are implemented and tested.

---

## Recently Completed (2026)

### Self-Optimizing Ensemble Weight System
- `backtest.py` + `optimize_weights.py` + `optimize_variance.py` — all implemented
- Versioned weight store at `models/blend_weights/` with atomic writes and rollback
- Drift detection via statistical process control (2σ above baseline)
- EnsembleOptimizer uses scipy.optimize over 13-dim weight space with accept/verify gates

### Smart Feature Selection
- `train.py --feature-selection smart --selection-profile {fast,balanced,max_accuracy}`
- Shadow filter + group ablation + permutation importance + stability scoring
- Disabled by default (`config/default.yaml` → `feature_selection.enabled: false`)

### Inter-Step Artifact Contracts
- `src/contracts/` — validates model files, feature schema, projection CSV schema, schedule schema
- `check_contracts.py` — standalone validator
- Both `train.py` and `simulate_season.py` validate at startup

### Lifecycle ML Integration
- B-Ianus Bayesian aging curves (`src/lifecycle/aging_model.py`)
- KAN aging factors (`src/lifecycle/kan_age_model.py`, CPU-only)
- Cached to `data/cache/aging_curves.csv` and `data/cache/kan_aging_outputs.csv`

### Season Context Features
- Season phase (early/mid/late/playoff)
- Team motivation (tank/playoff/injury-prior)
- Postseason context

### Calibration & Probability Upgrade
- Distribution zoo: empirical bootstrap, gamma, Poisson, NB, ZIP, Normal
- DistributionFitter derives (mean, std, skew, zero_prob, λ) from P10/P50/P90 quantiles
- Archetype-conditioned empirical copula for correlated multi-stat draws

---

## Potential Next Steps

1. **Wire LightGBM/XGBoost** — both are installed in requirements.txt but not part of the active training pipeline
2. **Cross-position models** — guard/forward/center specific model variants
3. **Player tracking data** — shot distance, speed, distance to basket
4. **Coaching impact factors** — rotation tendencies, play style
5. **Travel impact analysis** — time zones crossed, miles traveled
6. **REST API** — external access layer
7. **Web dashboard** — visualization interface
8. **Automated daily predictions** — scheduled pipeline runs

---

## Notes

- Active stack: CatBoost (primary) + Transformer (secondary). LSTM/GNN disabled.
- Ensemble weights are versioned in `models/blend_weights/` — not hardcoded.
- `src/services/` directory exists but is empty (unused).
- `plans/` and `.hermes/plans/` contain implementation plan docs.
- `project-brain/` contains curated architectural brain docs (including 2MB full project dump).
