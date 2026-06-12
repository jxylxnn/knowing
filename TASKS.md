# Tasks

## DONE

### Add a strict simulation mode for optional scraper degradation
Completed 2026-05-22.

Delivered:
- `simulate_season.py` exposes `--strict` CLI flag.
- `SeasonSimulator.__init__` accepts `strict_mode` parameter (default `False`).
- `GameSimulator.__init__` accepts `strict_mode` parameter (default `False`).
- `GameSimulator.simulate_matchup()` enforces `InputHealth` contract, raising `RuntimeError` on optional fallback/failed statuses when strict mode is active.
- `ReportGenerator.export_player_projections()` exports `DATA_QUALITY` column (FULL, DEGRADED_FALLBACK, DEGRADED_MISSING).
- `ReportGenerator._data_quality_from_result()` static helper derives quality from input_health metadata.
- `ProjectionLoader.find_player()` surfaces visible CLI warning when querying degraded projections.

### Self-Optimizing Ensemble Weight System
Completed 2026-05.

Delivered:
- `backtest.py` — standalone backtest CLI on date ranges.
- `optimize_weights.py` — self-optimize ensemble blend weights via scipy.optimize over 13-dim weight space.
- `optimize_variance.py` — CRPS-driven variance reduction.
- `src/evaluation/backtest_runner.py` — runs predictions on holdout → per-stat MAE/RMSE.
- `src/evaluation/ensemble_optimizer.py` — scipy.optimize with accept/verify gates.
- `src/evaluation/weight_store.py` — versioned JSON store (`models/blend_weights/`) with atomic writes and rollback.
- `src/evaluation/drift_detector.py` — statistical process control (2σ above baseline).
- `src/evaluation/metrics.py` — `BacktestResult`, `TargetMetrics`, `compute_target_metrics`.

### Inter-Step Artifact Contracts
Completed 2026-06.

Delivered:
- `src/contracts/artifacts.py` — validates required model files, target set match, metadata age.
- `src/contracts/features.py` — FeatureSchema consistency between trainer and inference.
- `src/contracts/projections.py` — player_projections CSV schema (incl. DATA_QUALITY column).
- `src/contracts/schedule.py` — schedule input contract for sim.
- `src/contracts/errors.py` — ContractError hierarchy.
- `check_contracts.py` — standalone validator CLI.
- Both `train.py` and `simulate_season.py` validate at startup.

### Smart Feature Selection
Completed 2026.

Delivered:
- `src/evaluation/smart_feature_selector.py` — combines 4 signals per target.
- `src/evaluation/shadow_feature_filter.py` — inject random control features, drop below noise floor.
- `src/evaluation/feature_group_ablation.py` — leave-one-group-out MAE deltas.
- `train.py --feature-selection smart --selection-profile {fast,balanced,max_accuracy}` — CLI integration.
- Output: `models/feature_selection_manifest.json` consumed by training and inference.

### Lifecycle ML Integration
Completed 2026.

Delivered:
- `src/lifecycle/aging_model.py` — B-Ianus Bayesian age-performance curve.
- `src/lifecycle/kan_age_model.py` — KAN-network (CPU) age factor.
- Feature groups: `aging_curve.py`, `kan_aging.py`, `skill_development.py` in `src/preprocessing/features/`.
- Caches: `data/cache/aging_curves.csv`, `data/cache/kan_aging_outputs.csv`.

### Season Context Features
Completed 2026.

Delivered:
- `src/preprocessing/features/season_phase.py` — early/mid/late/playoff phase.
- `src/preprocessing/features/team_motivation.py` — tank/playoff/injury-prior motivations.
- `src/preprocessing/features/postseason_context.py` — playoff vs regular-season weight.

### Modular Training Pipeline v2.0
Completed 2026.

Delivered:
- Replaced 1800-line god class with modular components.
- `src/training/pipeline.py` — main orchestrator (50KB).
- `src/training/catboost_trainer.py` — per-target CatBoost with multi-loss + quantile regression.
- `src/training/nn_trainer.py` — unified PyTorch trainer (Transformer, optional Nexus).
- `src/training/presets.py` — preset resolution from config.
- `src/training/feature_cache.py` — hash-keyed feature/split cache.
- `src/training/experiment.py` — JSON-based experiment tracking.
- Parallel training across targets (joblib).

### Calibration & Probability Upgrade
Completed 2026-05.

Delivered:
- `src/query/distribution_fitter.py` — derives distribution params from P10/P50/P90 quantiles.
- Distribution zoo: empirical bootstrap, gamma, Poisson, NB, ZIP, Normal.
- `src/query/empirical_covariance.py` — archetype-conditioned 6×6 correlation matrices.
- `src/training/nexus_loss.py` — CRPS loss for distribution scoring.
- `optimize_variance.py` — CRPS-driven variance reduction.

### Zero-Padding Fix (DR-021)
Not yet applicable — depends on the training pipeline. Deferred.

## NEXT

- Wire LightGBM / XGBoost into active training pipeline (installed but not wired)
- Cross-position models (guard/forward/center specific)
- Player tracking data integration
- REST API / web dashboard
