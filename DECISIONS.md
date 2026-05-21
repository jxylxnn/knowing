# Decisions

## DR-025: Enforce Strict Mode and Data Quality Schema for Simulator Degradation
Status: active
Date: 2026-05-22
Confidence: high

### Context
While DR-005 and DR-009 established resilient fallbacks and visible degradation summaries for optional scrapers, the query layer remained blind to which specific player projections relied on fallback math. Furthermore, advanced operators lacked a fail-fast mechanism for production-grade runs.

### Decision
1. Introduce a `--strict` CLI flag to `simulate_season.py` that halts execution if any optional `InputHealth` record reports `failed` or `fallback`.
2. Append a `DATA_QUALITY` column to the `player_projections_<timestamp>.csv` export schema.
3. Wire `ProjectionLoader` to read `DATA_QUALITY` and emit a visible warning during interactive queries.

### Consequences
The query layer is no longer blind to upstream scraper volatility. Operators can choose between resilient (default) and strict (fail-fast) execution modes based on their use case.