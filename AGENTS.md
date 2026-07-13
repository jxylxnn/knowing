# AGENTS.md

Pure-Python CLI ML project (no web server, Docker, or database). Predicts NBA player stats via ensemble deep learning + GPU-accelerated Monte Carlo simulation. venv at project root (Python 3.12).

## Environment

Always activate venv before running anything:

```bash
source venv/bin/activate
```

No linter or formatter is installed (no ruff/flake8/pylint). Follow PEP 8.

## Tests (260+ total)

```bash
pytest tests/ -v                  # full suite (slow — several minutes)
pytest tests/test_config/ -v      # single package (fast)
pytest tests/test_query/ -v       # single package (fast)
pytest -m "not slow"              # skip slow-marked tests
```

Custom markers: `slow`, `gpu`, `integration` (registered in `tests/conftest.py`).
No `--timeout` flag (pytest-timeout not installed); some suites are slow and may exceed default 2-min tool timeout — increase timeout if needed.

## Pipeline (order matters: each step depends on prior output)

1. `python update_data.py` — fetch NBA data (requires internet; NBA.com API rate-limited at ~0.6s/request)
   - `--interactive` / `-i` for first-time setup (recommended)
   - `--all-seasons` for last 10 seasons
   - `--update` for incremental (only new games since last run)
   - `--force` to re-fetch even if data exists
2. `python train.py` — train models (requires `data/` from step 1)
   - Training preset: `--preset {small,full}` (default: `full`; `small` skips Transformer, fewer features)
   - Training mode: `--mode {quick,standard,full}`; defaults come from preset
   - Config: `config/default.yaml` → `training_presets`
   - Will fail if `data/nba_players.csv` is missing. Run `update_data.py` first.
3. `python simulate_season.py --today` — simulate games (requires `models/` from step 2)
   - Other modes: `--date YYYY-MM-DD`, `--week`, `--season`
   - `--sims N` for simulation count (default: 100)
   - `--workers 1` recommended when using GPU (avoid CUDA context contention)
4. `python query_prob.py` — interactive probability query CLI (requires models + data)

### Optional / Supporting Entry Points

- `python backtest.py` — standalone backtest on a date range (uses `evaluation/backtest_runner.py`)
- `python optimize_weights.py` — self-optimize ensemble blend weights (uses `evaluation/ensemble_optimizer.py`)
- `python optimize_variance.py` — CRPS-driven variance reduction
- `python check_contracts.py` — validate artifact contract between pipeline steps

### Cleanup

```bash
python clear_cache.py --all --dry-run               # preview
python clear_cache.py --all --keep-models --yes     # preserve trained models
python clear_cache.py --all --yes                   # remove all generated artifacts
```

Raw CSV files in `data/` are preserved by `clear_cache`.

## Architecture

```
Root entry points:  update_data.py, train.py, simulate_season.py, query_prob.py
                    backtest.py, optimize_weights.py, optimize_variance.py, check_contracts.py
src/
  config/          — config loading (main config: config/default.yaml)
  data/            — scrapers (NBA API, ESPN injuries, Rotowire lineups, Action Network betting, Basketball Reference, player bios)
                     base_scraper.py + extensions/ — pluggable data-source registry (BaseScraper ABC + ScraperRegistry)
  preprocessing/   — modular FeatureGroup architecture → 150+ features (25+ toggleable groups)
    features/      — 21 feature modules: rolling, efficiency, momentum, context, fatigue, matchup,
                     opponent_strength, pace, team_role, target_encoding, league_rank, minutes_confidence,
                     recency_form, lineup_stability, rest_density, injury_opportunity, teammate_usage,
                     defense_position, injury_risk, aging_curve, kan_aging, skill_development,
                     archetype, season_phase, team_motivation, postseason_context
                     registry.py — FeatureGroupRegistry (single source of truth for group names + leak-safe prefixes)
                     extensions/ — pluggable feature groups (auto-discovered, no edits to __init__/_build_groups/presets)
  models/          — model_manager.py (live bridge), transformer_model.py,
                     error_calibration.py, minutes_predictor.py, gpu_utils.py
  pipeline/        — training_pipeline.py, data_pipeline.py, prediction_service.py
  training/        — pipeline.py (orchestrator, 50KB), catboost_trainer.py,
                     presets.py, experiment.py, training_logger.py
  simulation/      — game_simulator.py, season_simulator.py, phase_simulator.py,
                     four_factors_engine.py, game_context_engine.py,
                     player_correlation_engine.py, archetype.py, role_sampler.py,
                     input_health.py, report_generator.py, sim_types.py, sim_cache.py, stat_utils.py
  query/           — interactive_cli.py, probability_calculator.py, distribution_fitter.py,
                     empirical_covariance.py, prob_formatter.py, projection_loader.py, query_parser.py
  evaluation/      — backtest_runner.py, weight_store.py, ensemble_optimizer.py,
                     drift_detector.py, smart_feature_selector.py, shadow_feature_filter.py,
                     feature_group_ablation.py, metrics.py
  contracts/       — artifacts.py, features.py, projections.py, schedule.py, errors.py
  lifecycle/       — aging_model.py (B-Ianus Bayesian), kan_age_model.py (KAN network, CPU-only)
  utils/           — logging, reproducibility, team_mappings, prediction_utils
tests/             — mirrors src/ structure; conftest.py injects project root into sys.path
```

## Adding new data sources (extension mechanism)

The system supports adding a new data source + feature group **without editing the four historical sync points** (the group module, `features/__init__.py`, `FeatureEngineer._build_groups`, and config/preset enable-lists). Drop a self-contained module into an extensions directory instead.

**Data side** — add a scraper to `src/data/extensions/` subclassing `BaseScraper` (`src/data/base_scraper.py`): implement `name`, `output_files()`, and `fetch(config) -> {relative_csv_name: DataFrame}`. Then enable it in `config/default.yaml`:
```yaml
data_sources:
  my_source:
    enabled: true
```
`update_data.py` runs enabled scrapers after the core fetch (failures isolated per-scraper; core outputs `nba_players.csv`/`nba_games.csv` are protected from overwrite). The built-in example `advanced_tracking_scraper.py` derives tracking data from the game logs and writes `data/advanced_tracking.csv`.

**Feature side** — add a feature group to `src/preprocessing/features/extensions/` subclassing `FeatureGroup`. Declare `feature_prefixes`/`feature_keywords` (so `FeatureSelector` keeps the columns — otherwise they're dropped as not leakage-safe) and `external_files()` (so the feature cache invalidates when the source CSV changes). `FeatureGroupRegistry` auto-discovers it; `FeatureEngineer._build_groups` appends extension groups after the built-ins (preserving built-in column order for contract stability). Enable via a preset's `enable_groups` or the `"all"` sentinel. The example `advanced_tracking_features.py` reads the tracking CSV and emits `ADVTRACK_*` rolling features.

**Key safety properties**:
- Additive feature changes are contract-safe (`contracts/features.py` uses `allow_extra=True`; extra columns are ignored at inference, picked up on retrain).
- `FeatureSelector` auto-folds registry-declared safe prefixes/keywords, fixing the historical silent-drop trap; it now also *warns* when numeric columns are dropped for not matching any safe rule.
- Extensions default to `gpu_compatible = False` and run on the CPU pandas path in `FeatureEngineerGPU` (safe default for arbitrary new logic).
- `presets.ALL_FEATURE_GROUPS` and the `"all"` sentinel resolve to the registry truth, so the enable-list can never drift out of sync with registered groups (this fixed a live bug where the presets tuple was missing `injury_risk`, `aging_curve`, `kan_aging`, `skill_development` that config enabled).
- A broken extension module is isolated — import/instantiation failures are logged and skipped, never aborting the core pipeline.

See `tests/test_preprocessing/test_extension_mechanism.py` for the full worked example and tests.

## Gotchas

- **Active model stack**: CatBoost + Transformer.
- **Feature engineering is cached**: `FeatureEngineer.create_features()` caches results to `cache/training/*.parquet`, keyed on the input DataFrame hash + FE config (rolling windows, enabled/disabled groups) + mtime/size of external files some groups read (`data/injury_history.csv`, `data/cache/aging_curves.csv`, `data/player_bios.csv`, `data/cache/kan_aging_outputs.csv`). The cache is active by default in `DataPipeline` and `ModelManager` (training + live/simulation paths). Feature groups declare their external deps via `FeatureGroup.external_files()` so cached features are never stale when those files grow/update. To force recompute: `python clear_cache.py --all --yes` (or delete `cache/training/`).
- **FeatureSelector leak filter (no more silent drops)**: `FeatureSelector` only keeps columns matching a safe prefix/keyword/exact rule. New feature groups MUST declare `feature_prefixes`/`feature_keywords` (the registry folds these into the selector automatically); otherwise the selector now *warns* about dropped numeric columns instead of silently discarding them. Never add features whose names collide with `RAW_BOX_SCORE`/`EXCLUDE_ALWAYS`/target names.
- **Feature-group registry is the source of truth**: `FeatureGroupRegistry` (`src/preprocessing/features/registry.py`) drives `FeatureEngineer._build_groups`, `FeatureSelector` safe prefixes, and `presets.ALL_FEATURE_GROUPS`. Use the `"all"` sentinel in a preset's `enable_groups` to enable every registered group — do not hand-maintain the full list (the old hardcoded tuple had drifted and was missing 4 lifecycle groups).
- **Extension scrapers are opt-in**: `update_data.py` only runs scrapers listed under `config/default.yaml` `data_sources:` with `enabled: true`. With no `data_sources` block, zero extensions run and behaviour is unchanged. Extension scrapers can never overwrite the core `nba_players.csv`/`nba_games.csv`.
- **CatBoost parallelism**: On GPU use `max_workers=1` (CUDA context contention); on CPU size workers to machine cores.
- **Gitignored at root only**: `.gitignore` uses `/data/` and `/models/` (with leading slash), so only root-level `data/` and `models/` are ignored — `src/data/` and `src/models/` are tracked.
- **PyTorch test shim**: `src/__init__.py` installs a NumPy-backed torch shim when running under pytest on machines without a working PyTorch. Tests import `src` and get the shim automatically; real runtime still requires real torch.
- **PyTorch CUDA**: installed with CUDA support but auto-detects CPU on machines without GPU. No setup needed.
- **NBA API rate limits**: `update_data.py` is slow for many seasons; use `--interactive` to select specific seasons.
- **Data must exist before train**: `train.py` will fail if `data/nba_players.csv` is missing. Run `update_data.py` first.
- **Player bio data required**: Lifecycle features need AGE and POSITION columns in `nba_players.csv`. Run `update_data.py` at least once after adding the PlayerBioScraper to populate them. Missing AGE defaults all aging features to neutral (1.0 factor).
- **Injury history is incremental**: The `injury_history.csv` file grows over time. First runs will have sparse data — injury risk features will be near-zero for most players until several update cycles accumulate.
- **KAN model precomputation**: KAN aging factors are pre-computed on CPU and cached to `data/cache/kan_aging_outputs.csv`. If you retrain with `--force`, delete this file to force recomputation. KAN always runs on CPU to avoid GPU contention with CatBoost/Transformer.
- **B-Ianus aging precomputation**: Aging curves are cached to `data/cache/aging_curves.csv`. Same rule — delete to force recomputation.
- **Ensemble weights are versioned**: Blend weights live in `models/blend_weights/` as versioned JSON (not hardcoded in source). Use `optimize_weights.py` or the `WeightStore` API — never edit `blend_weights.pkl` directly.
- **Contracts validation**: Both `train.py` and `simulate_season.py` validate artifact contracts at startup. Run `python check_contracts.py` to debug inter-step contract issues.
- **Smart feature selection**: `train.py --feature-selection smart --selection-profile {fast,balanced,max_accuracy}` runs shadow filtering + group ablation + permutation importance. Disabled by default (`config/default.yaml` → `feature_selection.enabled: false`).
- **Minutes predictor**: LightGBM is used by `src/models/minutes_predictor.py` when an optional minutes model is trained; its deterministic fallback remains available when no such artifact exists.
