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

### Self-Optimizing Loop (backtest + weight optimization)
Code for `backtest.py` and `optimize_weights.py` does not exist yet. Operational execution (running the pipeline to establish baselines) deferred until those scripts are implemented.

### Zero-Padding Fix (DR-021)
Not yet applicable — depends on the training pipeline. Deferred.

## NEXT

(No pending items beyond the lifecycle ML integration completed in earlier sessions.)