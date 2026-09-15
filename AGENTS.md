# AGENTS.md

Pure-Python CLI project for NBA player-stat forecasts. The supported runtime
architecture is Model v2 only. Do not add compatibility fallbacks to legacy
flat model artifacts, the old `ModelManager`, or the retired simulator.

## Environment

Always activate the Python 3.12 virtual environment before running commands:

```bash
source venv/bin/activate
```

No linter or formatter is installed. Follow PEP 8.

## Model v2 workflow

1. `python update_data.py` fetches source data and creates immutable snapshots.
2. `python canonicalize_data.py --snapshot-id ID` writes canonical tables.
3. `python train.py --snapshot-id ID` creates an immutable candidate bundle.
4. `python replay.py --predictions FILE --actuals FILE` produces replay evidence.
5. `python promote_model.py --candidate ID` promotes only a sealed, eligible v2 bundle.
6. `python simulate_season.py --today --snapshot-id ID` forecasts scheduled games.
7. `python reconcile_predictions.py --actuals FILE` reconciles the official ledger.
8. `python query_prob.py --forecast-file FILE` answers probability queries from v2 forecasts.

The candidate produced by the lightweight rolling baseline is intentionally
not promotion-eligible until official point-in-time roster/status data and
multi-fold replay evidence exist. Never weaken that gate to make a run pass.

## Architecture rules

- `plans/model_v2_full_architecture_plan.md` is the design source of truth.
- Source snapshots, canonical tables, bundles, forecasts, and reconciliation
  records are immutable and content-addressed.
- Every forecast is for an explicit scheduled game and cutoff horizon.
- Runtime prediction must load the configured v2 champion; no heuristic or
  legacy artifact fallback is allowed.
- Strict official forecasts require point-in-time roster membership. Historical
  appearances are allowed only in explicitly degraded, non-official runs.
- Participation is sampled before minutes, and each team receives exactly 240
  regulation minutes before stat rates are sampled.
- Training, replay, simulation, and live forecasting must share the same
  contracts and component implementations.

## Lightweight verification

Do not run the full test suite on this machine unless the owner explicitly asks.
Use focused V2 tests and CLI import checks:

```bash
pytest tests/test_data/test_snapshots.py \
  tests/test_data/test_canonicalize.py \
  tests/test_models/test_v2_bundle.py \
  tests/test_evaluation/test_v2_evaluation.py \
  tests/test_evaluation/test_v2_replay.py \
  tests/test_operations/test_shadow.py \
  tests/test_operations/test_v2_ledger.py \
  tests/test_query/test_v2_probability.py \
  tests/test_simulation/test_joint_sampler.py \
  tests/test_simulation/test_v2_runner.py \
  tests/test_training/test_v2_baseline.py -q
```

Custom markers are registered in `tests/conftest.py`: `slow`, `gpu`, and
`integration`. No `pytest-timeout` plugin is installed.

## Cleanup

```bash
python clear_cache.py --all --dry-run
python clear_cache.py --all --yes
```

Raw CSV inputs are preserved. Never delete a snapshot or sealed bundle as part
of routine cache cleanup. The approved tracked-file retirement list lives in
`plans/model_v2_cleanup_manifest.md`.

## Extension points retained by v2

- Data scrapers in `src/data/extensions/` subclass `BaseScraper` and are enabled
  under `data_sources` in configuration.
- Feature groups in `src/preprocessing/features/extensions/` subclass
  `FeatureGroup` and declare leak-safe prefixes/keywords plus external files.
- `FeatureGroupRegistry` remains the source of truth for registered groups.
- CatBoost and Transformer components remain available as future v2 challengers,
  but they may enter production only through the v2 bundle and promotion gates.

## Safety notes

- NBA API requests are rate-limited; avoid unnecessary refreshes.
- Current-game outcomes are forbidden as forecast features.
- GPU CatBoost runs should use one worker to avoid CUDA contention.
- KAN and B-Ianus aging caches are derived inputs; invalidate them when their
  underlying data changes.
- Preserve unrelated changes in a dirty worktree.
