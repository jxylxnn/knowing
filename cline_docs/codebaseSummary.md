# Codebase Summary

Last updated: 2026-06-06 (sync with current `src/` tree; supersedes March 2026 version).

## What This Project Is

A pure-Python CLI system that predicts per-game NBA player stats (PTS, REB, AST, STL, BLK, TOV) and converts those projections into calibrated probability distributions for over/under queries. Stack: CatBoost (primary) + Transformer (secondary) ensemble, modular 25+ group feature engineering pipeline, GPU-accelerated Monte Carlo game simulation, and a self-optimizing weight tuner that can rewrite its own blend weights from backtest evidence.

No web server, no Docker, no database. Everything is CLI + CSV + JSON artifacts. See `AGENTS.md` for runtime/env instructions and `project-brain/` for the curated architectural brain — this file is a structural map, not a design rationale.

---

## Pipeline at a Glance

```
update_data.py  →  train.py  →  simulate_season.py  →  query_prob.py
   (scrapers)     (models)      (Monte Carlo)            (prob queries)
       │              │                 │                       │
       ▼              ▼                 ▼                       ▼
  data/*.csv    models/*.cbm     data/sim_results/       terminal + JSON
                models/*.pkl     data/sim_cache/         + degraded-input
                                models/blend_weights/   warnings
```

Strict ordering — each step depends on the prior step's artifacts. `check_contracts.py` validates the artifact contract between steps.

---

## 1. Data Layer — `src/data/`

Scrapers that feed the raw `data/` CSVs.

| File | Source | Output |
|------|--------|--------|
| `update_data.py` (root) | Orchestrator | Wires scrapers together, `--interactive` / `--all-seasons` / `--update` / `--force` |
| `basketball_ref_scraper.py` | Basketball Reference | Advanced historical stats |
| `nba_defense_scraper.py` | NBA.com Stats API | Team defensive ratings |
| `injury_scraper.py` | ESPN Health API | Current injury status |
| `injury_history_logger.py` | Local | Incremental injury log → drives injury-risk features |
| `lineup_scraper.py` | Rotowire | Confirmed starters |
| `rotowire_lineup_scraper.py` | Rotowire | Daily lineup projections |
| `betting_scraper.py` | Action Network | Spreads, totals, moneylines |
| `schedule_scraper.py` | NBA.com Stats API | Game schedule |
| `player_bio_scraper.py` | NBA.com | Populates `AGE` + `POSITION` (required by lifecycle features) |

Rate-limited at ~0.6s/request against NBA.com — full multi-season pulls are slow; use `--interactive` for first-time setup.

**Required outputs for downstream features**: `nba_players.csv` (player_id × game with box-score cols) and `nba_games.csv` (team × game). `AGE` and `POSITION` columns are required for any lifecycle/aging feature group to compute non-default values.

---

## 2. Feature Engineering — `src/preprocessing/`

Modular `FeatureGroup` architecture, **not** a fixed 15-phase pipeline. Each group is independently toggleable via `config/default.yaml` → `training_presets.*.feature_engineer.enable_groups`.

### Coordinator
| File | Purpose |
|------|---------|
| `feature_engineer.py` | Runs the registered groups, owns the schema |
| `feature_engineer_gpu.py` | Optional GPU-accelerated variant |
| `data_loader.py` | Reads/merges raw CSVs into the player-game frame |
| `features/base.py` | `FeatureGroup` abstract base class |
| `features/_teammate_utils.py` | Shared helpers for the teammate-usage groups |

### Feature groups (`src/preprocessing/features/`)

| Group | Module | Output flavor |
|-------|--------|---------------|
| Rolling | `rolling.py` (`RollingFeatureGroup`) | Window-3/5/10/20/50 means, std, min, max, range |
| Efficiency | `rolling.py` (`EfficiencyFeatureGroup`) | TS%, eFG%, per-minute |
| Momentum | `rolling.py` (`MomentumFeatureGroup`) | EWMA, trends, hot/cold streaks |
| Context | `context.py` | Home/away, rest days, back-to-back |
| Fatigue | `context.py` | Combined rest × minutes factor |
| Matchup | `matchup.py` | Career vs opponent |
| Opponent Strength | `matchup.py` | Defensive rank, points/rebs/asts allowed |
| Pace | `pace_role.py` | Possessions, pace factor |
| Team Role | `pace_role.py` | Role share, usage share |
| Target Encoding | `target_encoding.py` | Player/team historical means |
| League Ranking | `target_encoding.py` | Percentile by stat |
| Minutes Confidence | `minutes_confidence.py` | Reliability of minutes projection |
| Recency Form | `recency_form.py` | Last-N-game decay-weighted form |
| Lineup Stability | `lineup_stability.py` | Teammate-locked uncertainty |
| Rest Density | `rest_density.py` | Games-per-week fatigue |
| Injury-Adjusted Opportunity | `injury_opportunity.py` | Usage bump when teammates are out |
| Teammate Usage | `teammate_usage.py` | Aggregated usage of teammates |
| Defense Position | `defense_position.py` | Positional matchup quality |
| Injury Risk | `injury_risk.py` | Per-player DNP probability proxy |
| Aging Curve | `aging_curve.py` | B-Ianus Bayesian age-performance curve |
| KAN Aging | `kan_aging.py` | KAN-network (CPU) age factor |
| Skill Development | `skill_development.py` | Year-over-year trajectory |
| Archetype | `archetype.py` | Player archetype prior (also feeds sim) |
| Season Phase | `season_phase.py` | Early/mid/late/playoff phase |
| Team Motivation | `team_motivation.py` | Tank/playoff/injury-prior motivations |
| Postseason Context | `postseason_context.py` | Playoff vs regular-season weight |

The first 10 (rolling through teammate_usage) form the **core feature set**; the lifecycle/season-context groups (aging curves, KAN aging, injury risk, skill dev, season phase, team motivation, postseason context) are added by the `full` preset and can be disabled in `small`. Pre-computed cache files live in `data/cache/` (`aging_curves.csv`, `kan_aging_outputs.csv`) — delete to force recompute.

Output is 150+ features in the smart-preset, ~120 in the small preset.

---

## 3. Model Layer — `src/models/`

**Active stack: CatBoost (per-target) + Transformer (optional, secondary) + per-target MAE-companion CatBoost.**

| File | Purpose |
|------|---------|
| `model_manager.py` | Live bridge: loads artifacts, runs predictions, applies blend weights |
| `transformer_model.py` | Sequence model (attention over recent games) |
| `nexus_model.py` | Joint multi-output neural model (extras; not on by default) |
| `minutes_predictor.py` | Regression for expected minutes |
| `error_calibration.py` | Residual-based quantile calibration |
| `gpu_utils.py` | CUDA detection, tensor utilities, TF32 helper |
| `base.py` | `BaseModel` abstract interface |

**LSTM and GNN are explicitly `enabled: false` in `config/default.yaml`** (see AGENTS.md "Disabled models" gotcha). Joint_NN and GNN are not part of the active stack; the README diagram's "5 models" is stale — it's 2 with 1 optional.

### Ensemble weights

**No hardcoded `.50 / .15 / .15 / .15 / .05` blend.** Blend weights are stored in a versioned JSON store at `models/blend_weights/` (see `weight_store.py` + `check_contracts.py` → `ArtifactContract`) and **tuned by the self-optimizing loop** (see Section 6). Defaults live in the versioned store, not the source.

---

## 4. Training Orchestration — `src/training/` + `src/pipeline/`

Modular training pipeline (v2.0, replaces the legacy god class). See `currentTask.md` in this folder for the migration history.

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator: quick/standard/full modes, parallel target training |
| `catboost_trainer.py` | Per-target CatBoost with multi-loss + quantile regression |
| `nn_trainer.py` | Unified PyTorch trainer (Transformer, optional Nexus) |
| `presets.py` | Resolves `--preset {small,full}` and `--mode {quick,standard,full}` from `config/default.yaml` |
| `trainer.py` | `BaseTrainer` abstract interface |
| `feature_cache.py` | Hash-keyed feature/split cache |
| `experiment.py` | JSON-based experiment tracking |
| `training_logger.py` | Per-run structured logs |
| `nexus_loss.py` | CRPS loss for the optional Nexus model |
| `pipeline/data_pipeline.py` | Data prep step |
| `pipeline/training_pipeline.py` | Backwards-compat shim for legacy callers |
| `pipeline/prediction_service.py` | Inference entry for downstream scripts |

CLI (verified from `train.py --help`):

```bash
python train.py --preset {small,full} --mode {quick,standard,full} \
  [--model-size {auto,S,M,L,XL}] [--parallel] [--max-workers N] \
  [--feature-ablation] [--feature-selection {None,off,smart}] \
  [--selection-profile {fast,balanced,max_accuracy}] [--no-gpu] \
  [--experiment-name NAME] [--cache-dir DIR]
```

`--feature-selection smart` runs group ablation + per-target pruning + shadow filtering (see Section 6). `--feature-ablation` benchmarks each group before training.

---

## 5. Simulation Engine — `src/simulation/`

GPU-vectorized Monte Carlo that turns projections into correlated, archetype-aware stat samples.

| File | Purpose |
|------|---------|
| `game_simulator.py` | Top-level per-matchup simulator; enforces `InputHealth` contract (DR-025) |
| `season_simulator.py` | Multi-game batch driver; `--strict` mode (DR-025) |
| `phase_simulator.py` | Phase-by-phase game progression (replaces flat Monte Carlo) |
| `possession_simulator.py` | Possession-level internals (rebound/turnover/shot chains) |
| `four_factors_engine.py` | Dean Oliver four-factors adjustments |
| `game_context_engine.py` | Home/away, rest, schedule context |
| `player_correlation_engine.py` | Stat correlation injection (delegates to empirical copula when available) |
| `archetype.py` | `ArchetypeEngine` — coarse player archetype + style priors |
| `role_sampler.py` | Per-sim role state sampling (limited/normal/expanded/starter/bench/closer) |
| `input_health.py` | `InputHealthStatus` + `summarize_input_health` — source-health reporting that drives `--strict` and the `DATA_QUALITY` column |
| `sim_types.py` | Shared dataclasses (`RoleSample`, etc.) |
| `sim_cache.py` | Roster/cache for repeated sims |
| `stat_utils.py` | Stat helpers |
| `report_generator.py` | CSV/stdout formatting, exports `player_projections_*.csv` with `DATA_QUALITY` column |

The simulator is no longer a "sample correlated normals" black box — it uses **archetype-conditioned empirical copulas** (`src/query/empirical_covariance.py` + `data/cache/archetype_covariances.npz`) to generate correlated multi-stat draws, plus archetype-aware role multipliers from `role_sampler.py`.

CLI (verified from `simulate_season.py --help`):

```bash
python simulate_season.py (--today | --date YYYY-MM-DD | --week | --season) \
  [--sims N] [--workers N] [--stat {mode,mean,both}] \
  [--no-csv] [--strict] [--data-dir DIR] [--models-dir DIR] [--output-dir DIR]
```

`--strict` fails fast if any optional context source is `fallback` or `failed` (DR-025). `--workers 1` is recommended on GPU (CUDA context contention).

---

## 6. Evaluation & Self-Optimization — `src/evaluation/`

The closed-loop subsystem that can rewrite its own ensemble weights. **New in 2026.**

| File | Purpose |
|------|---------|
| `backtest_runner.py` | Runs predictions on a date range of completed games → per-stat MAE/RMSE |
| `metrics.py` | `BacktestResult`, `TargetMetrics`, `compute_target_metrics` |
| `weight_store.py` | Versioned JSON store of ensemble blend weights (replaces `blend_weights.pkl`) |
| `ensemble_optimizer.py` | scipy.optimize over the 13-dim weight space with accept/verify gates |
| `drift_detector.py` | Statistical process control (2σ above baseline) over rolling MAE |
| `feature_group_ablation.py` | Leave-one-group-out MAE deltas → "group helps/hurts" signal |
| `shadow_feature_filter.py` | Inject random control features, drop any feature below their median importance |
| `smart_feature_selector.py` | Combines 4 signals (backtest gain, stability, importance, permutation) per target → writes `models/feature_selection_manifest.json` |

Self-optimizing loop (used by `optimize_weights.py` at the repo root):

```
backtest_runner.run(holdout_start, holdout_end)   →   baseline_score
ensemble_optimizer.optimize(holdout)              →   candidate_weights
  ↳ BacktestRunner.run(holdout) on candidate      →   candidate_score
  ↳ If improvement > threshold AND verify run     →   accept → weight_store.promote
  ↳ Else reject (atomic write never happens)
drift_detector.detect()                           →   "ok" / "warning" / "critical" per target
```

Smart feature selection (used by `train.py --feature-selection smart --selection-profile {fast,balanced,max_accuracy}`):

```
shadow_feature_filter  →  drop features below noise floor
feature_group_ablation →  per-group MAE delta
HistGradientBoosting   →  CatBoost-style gain importances
permutation_importance →  permutation signal
stability              →  split-half importance correlation
final_score = 0.40·backtest_gain + 0.25·stability + 0.20·importance
             + 0.10·permutation - 0.05·missingness
output  →  models/feature_selection_manifest.json
```

---

## 7. Query System — `src/query/`

| File | Purpose |
|------|---------|
| `interactive_cli.py` | REPL for one-shot and comparison queries |
| `probability_calculator.py` | Over/under probability math (CRPS-aware, multiple distribution models) |
| `projection_loader.py` | Loads `player_projections_*.csv`; surfaces `DATA_QUALITY` warnings (DR-025) |
| `query_parser.py` | Natural-language query parsing |
| `distribution_fitter.py` | Derives (mean, std, skew, zero_prob, λ) from P10/P50/P90 quantile predictions |
| `empirical_covariance.py` | `CovarianceCache` — archetype-conditioned 6×6 correlation matrices |
| `prob_formatter.py` | `ProbFormatterMixin` — pretty-printed result rendering |

Probability math has moved from "Normal(mu, sigma) fallback" to a **distribution zoo** with the following selectable models (chosen per-stat based on data characteristics):

- `empirical_bootstrap` — preferred when historical sample is dense
- `gamma` — positive-continuous stats (PTS, REB, AST, MIN)
- `poisson` / `negative_binomial` / `zero_inflated_poisson` — count stats (STL, BLK, TOV)
- `degenerate_zero` / `normal` — last-resort fallbacks

CRPS is used to score distribution fit (see `src/training/nexus_loss.py`).

CLI: `python query_prob.py -p "Player Name" -s pts -l 25.5 [-o OPP] [--json]` or interactive REPL. Use `--list-players` / `--list-teams` to enumerate.

---

## 8. Contracts Layer — `src/contracts/` (new in 2026)

Validates the inter-step artifact contract. Run `python check_contracts.py` between pipeline steps.

| File | Validates |
|------|-----------|
| `artifacts.py` | Required `models/*.cbm`, `models/*_metadata.joblib`, `feature_schema.pkl`, `feature_cols.pkl`, `blend_weights.pkl`, `model_stack_metadata.pkl` (plus `attention_transformer.pkl` if `transformer_required=True`); checks `max_age_hours` and that metadata `targets` matches the canonical 6-stat set |
| `features.py` | `FeatureSchema` consistency between trainer and inference |
| `projections.py` | `player_projections_*.csv` schema (including `DATA_QUALITY` column) |
| `schedule.py` | Schedule input contract for sim |
| `errors.py` | `ContractError`, `ArtifactContractError`, `FeatureSchemaContractError`, `ProjectionSchemaContractError`, `ScheduleContractError` |

This is the seam that makes the optimizer / selector / sim-stack swappable. `train.py` and `simulate_season.py` both validate at startup.

---

## 9. Lifecycle ML — `src/lifecycle/`

Standalone lifecycle models consumed by feature groups.

| File | Purpose |
|------|---------|
| `aging_model.py` | B-Ianus Bayesian age-performance curve (caches to `data/cache/aging_curves.csv`) |
| `kan_age_model.py` | KAN-network (Kolmogorov-Arnold) age factor (caches to `data/cache/kan_aging_outputs.csv`, always CPU) |

Both are pre-computed and cached; delete the cache file to force a recompute. `KAN` always runs on CPU to avoid GPU contention with CatBoost/Transformer (per AGENTS.md gotcha).

---

## 10. Pipeline / Config / Utils

| Path | Purpose |
|------|---------|
| `config/default.yaml` | Single source of truth for paths, training presets, feature-group toggles, sim settings |
| `src/config/config.py` | YAML loader with preset resolution |
| `src/config/model_config.py` | Per-model hyperparameter profiles |
| `src/utils/logging_config.py` | Rich + file logger setup |
| `src/utils/reproducibility.py` | Seed control |
| `src/utils/prediction_utils.py` | Shared prediction helpers |
| `src/utils/team_mappings.py` | NBA team code lookups |

`training_presets` in `default.yaml` has `full` and `small` keys. `full` is the default; `small` skips Transformer and trims feature groups. `--mode` defaults come from the preset.

---

## 11. Tests — `tests/`

~260+ tests, mirrors `src/` structure. Registered markers: `slow`, `gpu`, `integration` (see `tests/conftest.py`). No `pytest-timeout` is installed — some suites exceed the default 2-min tool timeout, raise the timeout when running the full suite.

New test packages: `test_contracts/`, `test_evaluation/`.

```bash
source venv/bin/activate
pytest tests/ -v                   # full suite (slow — allow several minutes)
pytest tests/test_evaluation/ -v   # fast — backtest + smart selector
pytest tests/test_contracts/ -v    # fast — artifact contract smoke
pytest -m "not slow"               # skip slow-marked tests
```

---

## 12. Entry Points (Root)

| Script | Purpose |
|--------|---------|
| `update_data.py` | Run scrapers; `--interactive`, `--all-seasons`, `--update`, `--force` |
| `train.py` | Train; see Section 4 for flags |
| `simulate_season.py` | Simulate; see Section 5 for flags (incl. `--strict`) |
| `query_prob.py` | Interactive prob queries; see Section 7 |
| `backtest.py` | Standalone backtest CLI (uses `evaluation/backtest_runner.py`) |
| `optimize_weights.py` | Standalone self-optimization CLI (uses `evaluation/ensemble_optimizer.py`) |
| `optimize_variance.py` | Standalone variance-reduction CLI (CRPS-driven) |
| `clear_cache.py` | Wipes `cache/`, `data/cache/`, `data/sim_*`, `models/`, `experiments/`, `__pycache__/`; preserves raw `data/*.csv` |
| `check_contracts.py` | Standalone artifact-contract validator |
| `train_colab.ipynb` | Colab training notebook (pins a specific branch) |
| `query_prob.ipynb` | Jupyter query interface |

---

## What Stale Parts of Older Docs to Ignore

- **"5 Deep Learning Models"** — only CatBoost + Transformer are active. LSTM, GNN, Joint_NN are disabled.
- **"15-Phase Feature Engineering"** — was true; now 25+ independently-toggleable `FeatureGroup` classes.
- **Hardcoded ensemble weights `0.50·CB + 0.15·JointNN + 0.15·LSTM + 0.15·Trans + 0.05·GNN`** — superseded by `models/blend_weights/current.json` (versioned, self-optimized).
- **`Phi(z)` Normal distribution only** for probabilities — replaced by the distribution zoo (empirical/gamma/Poisson/NB/ZIP/Normal).
- **"PTS MAE 4.82, RMSE 6.31"** — these are legacy test-set numbers, not a current benchmark. Re-run after the smart selector + optimizer are applied.
- **Joint_NN, LSTM, GNN classes in `src/models/`** — `nexus_model.py` exists but is not wired into the default training path.
