# NBA Player Forecasting — Model v2

This repository contains one supported forecasting architecture: **Model v2**.
It is a local, pure-Python CLI pipeline for point-in-time NBA player forecasts,
immutable model bundles, historical replay, and coherent simulation.

The design source of truth is
[`plans/model_v2_full_architecture_plan.md`](plans/model_v2_full_architecture_plan.md).
Code or artifacts that cannot satisfy its cutoff, replay, calibration, and
bundle contracts are not production forecasts.

## Current status

- The unsafe flat model directory has been removed.
- The initial implemented candidate is an honest rolling Model v2 baseline.
- Baseline training writes separate availability, minutes, conditional-rate,
  and calibration components into a checksummed immutable bundle.
- It deliberately records `promotion_eligible=false` until inactive-player
  labels, repeated replay evidence, calibration evidence, and seven shadow
  slates exist.
- There is no automatic fallback to a legacy model.

A missing champion produces a clear error instead of an apparently valid
forecast from stale or leaking artifacts.

## Environment

```bash
source venv/bin/activate
```

Python 3.12 is used by the project environment. Raw CSVs live under `data/` and
are preserved by the cleanup command.

## Model v2 workflow

### 1. Update and snapshot data

```bash
python update_data.py --update --snapshot
```

### 2. Canonicalize and inspect a snapshot

```bash
python canonicalize_data.py --snapshot-id SNAPSHOT_ID
python check_data.py --snapshot-id SNAPSHOT_ID
```

Canonical tables are immutable Parquet products under
`data/canonical/snapshot=<id>/`. Ambiguous game identities are quarantined; no
fuzzy identity match is performed.

### 3. Train an immutable baseline candidate

```bash
python train.py --architecture v2 --preset baseline
```

To reuse a snapshot:

```bash
python train.py --architecture v2 --preset baseline \
  --snapshot-id SNAPSHOT_ID --run-id candidate-name
```

Training writes a new directory below `models/versions/`. It never mutates the
champion pointer and never promotes itself.

### 4. Validate promotion evidence

```bash
python promote_model.py --candidate CANDIDATE_ID --dry-run
```

Promotion is expected to fail until the candidate contains complete replay,
calibration, contract, and shadow-slate evidence. Once all gates pass, omit
`--dry-run`; `models/champion.json` is updated atomically and rolled back if its
post-check fails.

### 5. Score and reconcile forecasts

```bash
python replay.py --predictions path/to/forecast.parquet \
  --actuals data/nba_players.csv --json

python reconcile_predictions.py --date YYYY-MM-DD --json
```

Official forecast payloads use the append-only ledger in
`src/operations/ledger.py`. Reconciliation writes a new content-addressed
record and never mutates the original prediction.

## Supported forecast contract

Each request identifies the game, schedule version, timezone-aware cutoff,
horizon, immutable source snapshot, model bundle, and scenario. Supported
horizons are `previous_night`, `morning`, `pregame_90m`, and `pregame_30m`.

Forecasts expose separate active and conditional-play probabilities,
conditional regulation minutes, and unconditional full-game distributions for
PTS, REB, AST, STL, BLK, and TOV.

## Architecture

```text
raw data -> immutable snapshot -> canonical tables -> scheduled player rows
         -> participation -> constrained minutes -> conditional stat rates
         -> calibrated distributions -> joint simulation -> official ledger
         -> actual reconciliation -> replay scorecards -> promotion decision
```

Important packages:

```text
src/contracts/     immutable request, forecast, source, and artifact contracts
src/data/          source snapshots, canonicalization, identity, and coverage
src/features/      scheduled rows and point-in-time feature materialization
src/forecasting/   participation, minutes, rates, and baseline runtime backend
src/evaluation/    rolling folds, baselines, replay, significance, promotion
src/models/        immutable bundles and atomic champion versioning
src/operations/    official ledger and shadow-slate evidence
src/simulation/    opportunity-first joint sampling
```

## Lightweight verification

The full suite is intentionally not required for routine local changes.

```bash
python -m compileall -q src *.py
pytest tests/test_data/test_snapshots.py \
       tests/test_data/test_canonicalize.py \
       tests/test_contracts/test_forecast_contract.py \
       tests/test_forecasting/test_opportunity_kernel.py \
       tests/test_training/test_v2_baseline.py \
       tests/test_evaluation/test_v2_replay.py \
       tests/test_models/test_v2_bundle.py \
       tests/test_operations/test_v2_ledger.py -q
```

## Cleanup

```bash
python clear_cache.py --all --dry-run
python clear_cache.py --all --yes
```

The second command removes generated caches, reports, experiments, and model
artifacts while preserving raw CSV data. Train and promote a new Model v2
bundle before attempting an official forecast.
