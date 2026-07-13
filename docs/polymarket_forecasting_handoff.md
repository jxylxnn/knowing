# Polymarket Forecasting Platform — Architecture Handoff

Prepared 2026-07-11 from a full architecture review of the local NBA prediction repository. This is a build brief for a next model or engineering agent.

## Executive conclusion

Do **not** copy the NBA model weights, player features, fixed correlation matrix, team constants, or simulation heuristics into a Polymarket project. They encode a different prediction problem and would create false confidence.

Do transfer the system's strongest ideas:

- an immutable, point-in-time forecast contract;
- modular data and feature extensions;
- distribution-first forecasts rather than only point estimates;
- schema and artifact contracts at every boundary;
- an append-only prediction ledger with post-resolution reconciliation;
- rolling, time-respecting evaluation and promotion gates;
- clear data-quality and confidence labels; and
- a single runtime forecasting service used by every CLI, backtest, and report.

The target is a **read-only-first, domain-pluggable market-forecasting workbench**. It should estimate a fair probability for a Polymarket outcome, compare it with the executable order-book price after fees and slippage, explain the evidence, record the decision-time snapshot, and measure calibration after resolution. Automated trading is explicitly out of scope for the first release.

## What this NBA repository actually contributes

| Area | Transfer | Why it matters for Polymarket |
|---|---|---|
| Forecast identity | `ForecastRequest`, request hash, cutoff time, source snapshot ID, model bundle ID, scenario | A historical result must be reproducible from only information known when the forecast was made. |
| Feature provenance | registered feature outputs with source, event time, allowed horizon, missing policy, and version | Prevents outcome leakage and makes generic/domain features safely extensible. |
| Model artifacts | immutable bundle manifests, hashes, data cutoff, configuration, metrics, atomic champion switch | Stops a changed file or mismatched schema from silently changing live outputs. |
| Probabilistic output | quantiles, zero mass, interval calibration, Monte Carlo/query layer | A market price is a probability, so the model must produce calibrated probabilities/uncertainty. |
| Evaluation | temporal splits, walk-forward residuals, drift, paired champion/challenger gates | A model is promoted only with out-of-sample evidence, not a single retrospective score. |
| Operational safety | degraded input health, strict mode, fallbacks surfaced to users | A stale source, thin book, missing feed, or unknown resolution rule must not look like high confidence. |
| Extension model | isolated data scraper and feature-group discovery, opt-in configuration | Lets sports, politics, macro, crypto, weather, and other market families evolve separately. |

## The central architecture

Build around one generic instrument: an outcome token that settles to 1 or 0. An event can contain two outcomes, many mutually exclusive outcomes, or a set of independent markets. The market catalog must classify the structure instead of assuming every question is a simple Yes/No proposition.

```text
Immutable source snapshots
        ↓
Canonical market, outcome, order-book, and evidence tables
        ↓
Point-in-time FeatureFrame for one market/outcome/cutoff
        ↓
Domain forecast plugin → calibrated fair probability distribution
        ↓
Market microstructure + fee/slippage model → executable edge decision
        ↓
Read-only recommendation/report + append-only prediction ledger
        ↓
Resolution reconciliation → calibration, realized P&L simulation, drift
        ↓
Champion/challenger bundle promotion
```

There must be exactly one production path: `ForecastService`. CLIs, scheduled jobs, backtests, dashboards, reports, and optional later execution adapters call it. Do not replicate loading, feature alignment, calibration, and fallback logic in multiple components.

## Canonical data contracts

### 1. Market catalog

One record per Polymarket market/outcome with stable platform identifiers. Preserve the question verbatim and store its resolution source and rules.

```text
platform, event_id, market_id, condition_id, token_id, outcome_id,
event_title, question, outcome_label, market_type, category, tags,
open_time, close_time, resolution_time, resolution_source, rules_url,
active, closed, resolved, negative_risk, observed_at, snapshot_id
```

`market_type` must be explicit: `binary`, `categorical_exclusive`, `scalar_bucket`, `sports_moneyline`, `sports_spread`, `sports_total`, or `custom`. Reject unsupported/ambiguous structures until a plugin understands their settlement semantics.

### 2. Market microstructure snapshots

Capture raw order-book information repeatedly; do not keep only a latest value. For a buying decision, use a conservative executable price, not a displayed midpoint.

```text
snapshot_id, observed_at, token_id, market_id, best_bid, best_ask,
midpoint, last_trade_price, spread, bid_depth_usd, ask_depth_usd,
depth_at_1c, depth_at_2c, depth_at_5c, imbalance, volume_1h,
volume_24h, open_interest, minimum_tick, minimum_order_size,
maker_fee_bps, taker_fee_bps, raw_payload_hash
```

### 3. External evidence snapshots

Use a source-neutral, append-only structure. Every value needs both the time it happened and the time it became observable.

```text
snapshot_id, source, source_record_id, entity_type, entity_id,
event_time, available_at, fetched_at, field_name, value_json,
reliability_tier, raw_payload_path, payload_hash
```

Examples: official sports injury/lineup data, bookmaker spreads, economic releases, election polls, weather forecasts, on-chain market activity, or a trusted news/document feed. The system must never infer availability time from the final article date.

### 4. Forecast request and long-form output

Use an immutable request for every decision:

```python
@dataclass(frozen=True)
class ForecastRequest:
    platform: str
    event_id: str
    market_id: str
    token_id: str
    outcome_id: str
    forecast_cutoff: datetime
    horizon: str              # e.g. 24h, 4h, 1h, 15m before close
    source_snapshot_id: str
    model_bundle_id: str
    scenario: str = "official"
```

Its `request_id` is a hash of all fields. A canonical output must be long-form and include:

```text
REQUEST_ID, MODEL_BUNDLE_ID, SOURCE_SNAPSHOT_ID, GENERATED_AT,
FORECAST_CUTOFF, MARKET_ID, TOKEN_ID, OUTCOME_ID, HORIZON, SCENARIO,
FAIR_PROBABILITY, P10, P50, P90, CALIBRATED_PROBABILITY,
MARKET_BID, MARKET_ASK, EXECUTABLE_BUY_PRICE, EXECUTABLE_SELL_PRICE,
FEE_ESTIMATE, SLIPPAGE_ESTIMATE, NET_EDGE, EXPECTED_VALUE_PER_SHARE,
LIQUIDITY_TIER, DATA_QUALITY, CONFIDENCE, CALIBRATION_VERSION,
MODEL_FAMILY, EXPLANATION_ID
```

For a mutually exclusive set of outcomes, enforce non-negative probabilities that sum to one after calibration. For independent markets, never force a sum-to-one constraint.

## Point-in-time safety: the most important rule

The NBA repository has begun implementing this correctly through `ForecastRequest`, `FeatureSpec`, and `FeatureMaterializer`. Keep this design, and make it stricter for Polymarket.

Every joined field must prove `available_at <= forecast_cutoff`. Every historical replay must use:

1. the market/order-book state captured at that cutoff;
2. only external evidence visible by that cutoff;
3. a model bundle whose training cutoff predates the forecasted event; and
4. the exact same `ForecastService` used live.

Never create a historical feature table with today's completed data and score old rows from it. That is retrospective row scoring, not a backtest. Never use final outcomes, resolution comments, post-close liquidity, revised injuries, revised polls, or later news in a forecast feature frame.

## Generic feature framework

Adopt a registry with exact `FeatureSpec` metadata. A feature is admitted because it is registered with provenance, not because its name matches a friendly prefix.

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    dtype: str
    entity_keys: tuple[str, ...]
    event_time_column: str
    max_lookback: timedelta | None
    required_sources: tuple[str, ...]
    allowed_horizons: tuple[str, ...]
    missing_policy: str
    version: str
```

Start with generic features that work across most markets:

- time to close/resolution, market age, event category, outcome count, and rules complexity;
- bid/ask, spread, depth, imbalance, volume, open interest, price momentum, realized volatility, and order-book staleness;
- market cross-sectional information from related outcomes/events, clearly tagged as contemporaneous;
- source freshness, number of independent source confirmations, disagreement, coverage, and missingness;
- model disagreement and calibration-bucket performance; and
- horizon-specific features, because a 24-hour and a 15-minute forecast do not have the same information set.

Add domain plugins only when their raw source and resolution semantics are understood:

- **sports**: normalized teams/players, schedule/rest, lineups/availability, external consensus spread/total, rates/ratings, injuries, and weather where relevant;
- **politics**: polls with field/end/release dates, forecast aggregates, candidate/party/entity resolution, official filing/calendar events;
- **macro**: release calendar, consensus vs prior, nowcasts, policy schedules, revisions;
- **crypto**: price/volatility, derivatives funding/open interest, liquidity and event calendar; and
- **weather/science/custom**: the authoritative measured variable, forecast provider version, geographic/entity mapping, and resolution threshold.

For generic language from questions or news, use it only as a traceable input. Store retrieval time, source, document hash, chunk ID, model/prompt version, and a citation. Do not use an unlogged LLM summary as a model feature.

## Modeling strategy

### Baseline before sophistication

First build baselines that are hard to beat:

- displayed midpoint and last trade price;
- bid/ask-based conservative executable probability;
- simple market-price calibration by category and time-to-close;
- regularized logistic/CatBoost model on generic tabular features; and
- a domain-specific baseline where a trustworthy external forecast exists.

If a more complex model cannot beat these on rolling-origin Brier score, log loss, calibration, and simulated executable value, do not promote it.

### Model families

Use a router to select a domain model family rather than a universal black box:

1. **Market calibration model** — maps market price, spread, depth, time, and category to calibrated probability. This is the cross-domain fallback.
2. **Structured domain model** — sports/macro/politics/etc. predictors that generate an independent prior using features valid at the cutoff.
3. **Evidence model** — combines trusted, timestamped external signals; it should output both probability and uncertainty.
4. **Ensemble/calibration layer** — combines independently validated components only; use out-of-fold predictions for blend fitting.
5. **Conformal or reliability calibration** — calibrated separately by market family and horizon. Store a versioned calibration artifact.

Use CatBoost or a regularized generalized linear model as the initial structured tabular champion. The NBA code's CatBoost-per-target and quantile path is useful. Treat its Transformer and Nexus-style architectures as challengers, not default complexity.

### Probability and uncertainty

For a binary outcome, the core output is `P(outcome settles YES)`, calibrated at the chosen cutoff/horizon. Report probability intervals or posterior/ensemble uncertainty, but evaluate the calibrated point probability with proper scoring rules.

For multi-outcome markets, model a coherent probability vector. For scalar buckets, map to a CDF/distribution and derive the probability of each bucket from its resolution boundaries. For sports spreads/totals, model the underlying margin/total distribution, then convert the specified threshold into the contract's Yes/No probability.

Do not carry over NBA's fixed six-stat correlation matrix. If a portfolio or multi-leg analysis needs dependence, estimate it empirically from out-of-fold residuals for the exact market family and horizon, validate it, and fall back to conservative independence when evidence is thin.

## From probability to a read-only decision

The NBA query system's “probability of a line” is useful, but Polymarket has a different final step: compare fair probability with **executable** price.

For a proposed YES buy of `q` shares:

```text
fair_p = calibrated_probability
fill_price = simulated_vwap_to_buy(q, ask_book)
net_edge = fair_p - fill_price - estimated_fees_per_share - reserve_for_model_error
expected_value_per_share = fair_p * payout_if_yes - fill_price - fees
```

Use the corresponding NO token/price for the opposite outcome. Get tick size, minimum size, and fee parameters from current market metadata rather than hard-coding them. Quote an action only when all gates pass:

- market is active and its settlement rule was parsed successfully;
- order book is fresh, sufficiently deep, and not abnormally wide;
- model/data quality is `FULL` or a deliberately permitted degraded class;
- calibrated edge clears fees, slippage, and a model-uncertainty reserve;
- category/horizon calibration sample is large enough; and
- risk limits are satisfied.

The first release must emit `WATCH`, `PASS`, or a **paper-trade** recommendation. It must not place orders, manage a wallet, or imply an assured return.

## Evaluation that decides whether anything is real

Use rolling-origin, cutoff-specific replay. Split by event resolution time, never random row splits.

Score at least:

- Brier score and log loss for binary probability quality;
- calibration curve, expected calibration error, and reliability by category/horizon/liquidity bucket;
- sharpness and interval coverage where intervals are reported;
- multiclass log loss/Brier for mutually exclusive outcomes;
- missingness, data freshness, and fallbacks by source;
- decision coverage: fraction of eligible markets that pass all gates; and
- paper portfolio P&L using bid/ask depth, fees, slippage, close/settlement handling, and no post-hoc fills.

Keep the NBA project's promotion philosophy: compare a candidate to a champion on the same held-out events; require a positive paired-bootstrap lower bound; reject a material regression in any critical market family or calibration bucket; and promote with an atomic manifest only after all gates pass.

Do not optimize only simulated P&L. It is easily overfit. A candidate must first improve proper probabilistic scores and calibration, then demonstrate robust executable-value improvement.

## Artifact, ledger, and audit design

Every model bundle should be immutable and contain:

```text
bundle_manifest.json
  bundle_id, created_at, code_version, config_hash, feature_schema_hash,
  training_cutoff, source_snapshot_manifest_ids, model files, calibration files,
  weighting policy, metrics, market-family coverage, known limitations
```

Switch the active champion pointer atomically. Never overwrite an in-use bundle.

Maintain an append-only prediction ledger with one immutable pre-resolution row per `(model_bundle, request_id, token_id)`. Later reconciliation may add `resolved_outcome`, `actual_settlement`, `realized_paper_pnl`, and scoring fields, but it must never replace the contemporaneous probability, quote, source snapshot, or explanation.

Store an explanation sidecar with the exact data-quality status, top evidence factors, price/depth data, feature version, calibration version, and citations. The NBA repository's numeric reasoning sidecar is a good pattern; opaque prose alone is not an audit trail.

## Reusable repository components and their Polymarket equivalents

| NBA component | Keep/change | Polymarket equivalent |
|---|---|---|
| `src/data/base_scraper.py` | Keep the opt-in discovery and output protection model | connector registry for public market APIs, external evidence providers, and domain sources; every connector writes timestamped snapshots. |
| `FeatureGroup` + registry | Keep modularity; replace name-based safety with exact specs | generic and domain feature plugins, each with point-in-time metadata and leakage tests. |
| `src/features/materializer.py` | Keep and finish as the one production materializer | builds market/outcome `FeatureFrame` as of a cutoff from immutable snapshots. |
| `src/contracts/forecast.py` | Keep the immutable request/output design | extend IDs for platform/event/market/token/outcome and add executable-price fields. |
| CatBoost pipeline | Keep as initial tabular benchmark | probability classifier/regressor with time-respecting OOF calibration; category/horizon-aware. |
| quantile/distribution code | Adapt | binary/multiclass probability calibration; CDF approach for spreads/buckets; no assumed NBA distributions. |
| `ForecastService` | Keep as the sole runtime seam | receives request, materializes features, routes model, calibrates, calculates edge, writes trace. |
| contracts + versioning | Keep and strengthen | required snapshot IDs, training cutoff, feature/config/model/calibration hashes, evaluation scorecard. |
| prediction ledger + continual learning | Keep | immutable probability/quote ledger, resolution reconciliation, promotion gates. |
| residual correction + monitor | Adapt carefully | only train correction from honest OOF predictions; continuously prove it helps Brier/log loss, not only point error. |
| input health + strict mode | Keep | stale order book, failed connector, low depth, unknown rules, missing resolution source, or degraded identity map become visible gates. |

## Do not port these NBA-specific patterns unchanged

- Player-stat values, league-average fallbacks, age curves, archetypes, position defense, minutes rules, roster selection, and team mappings.
- The fixed basketball simulation parameters: home edge, pace bounds, shot frequencies, 240-minute constraints, starter bonuses, blowout logic, and stat correlation matrix.
- Any model file, old `feature_schema.pkl`, blend weights, residual corrections, or reported backtest result. These have no Polymarket domain validity.
- A broad substring/prefix feature allow-list as the safety mechanism. The repository itself identified this as a source of leakage risk.
- Retrospective historical scoring that loads a current bundle and current feature table without reconstructing what was known at each past cutoff.
- Silent fallback predictions. Missing model/source/metadata must produce a labeled degraded result or a hard failure in strict mode.
- Automatic blending or residual correction without out-of-fold validation and a rollback gate.
- Any auto-execution interface in the first release.

## Important code-review findings to preserve as lessons

1. **Leakage control is a product requirement, not an afterthought.** The NBA loader historically merged current-game team totals into player rows. The selector now excludes known current-game outputs, but previously trained artifacts may still be contaminated and need retraining. Polymarket must block any feature whose availability cannot be proven.
2. **A scheduled prediction needs a synthetic request/frame.** Reusing a player’s last completed NBA row left its dates and rolling context stale. The repository has introduced a scheduled-row helper and a dedicated materializer. For Polymarket, materialize an actual `(market, outcome, cutoff)` frame every time.
3. **The untouched test window must stay untouched.** Feature selection, tuning, blending, calibration, and early stopping belong inside the inner training loop. The final window produces a mandatory scorecard.
4. **Simulation constants are assumptions until learned/calibrated.** Generic Monte Carlo can be useful for threshold/bucket probabilities, but every distributional or microstructure adjustment must be learned, calibrated, or explicitly disclosed as a scenario assumption.
5. **Evaluation/live parity is non-negotiable.** Backtests must invoke the same forecast service and include all enabled components. Otherwise a live model can never be trusted from its test result.
6. **Version every dependency, not only the estimator.** A probability changes when the data snapshot, rules parser, feature schema, calibration artifact, or fee model changes.

## Suggested repository layout

```text
polymarket_forecaster/
  config/
    default.yaml
    market_families.yaml
  src/
    contracts/          # market, snapshot, forecast, feature, artifact contracts
    connectors/         # Polymarket public APIs + external/domain connectors
    canonical/          # IDs, market normalization, rules parser, resolution map
    snapshots/          # append-only raw/canonical snapshot store and manifests
    features/           # FeatureSpec registry, materializer, generic/domain plugins
    models/             # baseline, structured domain models, calibration, bundles
    pricing/            # order book, VWAP, fees, slippage, position/paper risk
    pipeline/           # one ForecastService; training and replay services
    evaluation/         # rolling replay, proper scores, calibration, promotion, drift
    ledger/             # append-only forecasts and resolution reconciliation
    reasoning/          # numerical trace + cited evidence sidecars
    cli/                # fetch, snapshot, forecast, replay, report, paper-trade
  data/
    raw/<source>/snapshot_date=.../
    canonical/
    features/feature_set=<hash>/
    ledger/
    evaluation/
  models/versions/<bundle_id>/
  tests/
```

Use Parquet for append-only raw/canonical/evaluation data and small JSON manifests. A database is not required for a capable local first version.

## Delivery plan

### Phase 0 — foundation and read-only catalog

1. Connect to public Polymarket market catalog and CLOB read endpoints.
2. Create snapshot manifests, immutable raw payload storage, canonical market/outcome IDs, and order-book snapshots.
3. Implement question/rules ingestion plus manual review status for ambiguous market types.
4. Implement `ForecastRequest`, artifact contracts, data health, and the append-only ledger.
5. Deliver a CLI that lists/filters eligible markets and prints only snapshot/quality information.

### Phase 1 — generic calibrated baseline

1. Build cutoff-specific historical datasets from snapshots and resolved outcomes.
2. Train category/horizon price-calibration baselines and score them with rolling replay.
3. Add fee/depth-aware paper execution, no wallet or order placement.
4. Publish calibration and paper-decision reports; promote only versioned bundles.

### Phase 2 — sports markets first, if desired

1. Add a sports plugin with a reliable schedule/results provider, roster/availability source, and market-type parser.
2. Model game margin/total or win probability, then derive price for each Polymarket contract.
3. Compare independent sports forecasts, Polymarket price calibration, and external-book consensus without leakage.
4. Start with moneylines/spreads/totals; add player props only after their data/resolution requirements are covered.

### Phase 3 — additional market families

Add one family at a time with its own source reliability rubric, identity normalization, resolution mapping, feature specs, training/replay cohort, calibration, and promotion thresholds. “Any market” is a catalog and plugin problem, not one universal model.

## Definition of done for the first credible version

- Every forecast has a stable request ID, cutoff, source snapshot ID, model bundle ID, market/token IDs, and explanation trace.
- A replay can reproduce a historical forecast from stored data without reading post-cutoff information.
- The generic baseline beats or matches uncalibrated market prices on predeclared rolling evaluation metrics, or the project reports no edge honestly.
- Each report shows fair probability, executable price, fees/slippage, net edge, liquidity/data quality, confidence, and a `WATCH/PASS/PAPER` decision.
- A prediction ledger reconciles resolved markets and produces calibration/drift reports by category and horizon.
- Tests mutate all data available after a forecast cutoff and confirm feature values do not change.
- No live order placement exists.

## Copy/paste brief for the next model

```text
You are building a local, read-only-first Polymarket forecasting workbench in Python. Do not reuse NBA-specific weights, features, constants, or simulations. Reuse only the attached architecture patterns: immutable point-in-time forecast requests, snapshot provenance, registered features, artifact contracts, calibrated probabilistic outputs, an append-only forecast ledger, rolling-origin replay, data-quality gates, and atomic champion/challenger bundles.

Goal: for an arbitrary Polymarket market/outcome, estimate a calibrated fair settlement probability as of a named cutoff; compare it with a conservative executable order-book price after fees/slippage; explain the evidence; paper-trade only; and learn from resolved outcomes without leakage.

Required first deliverable:
1. Inspect the existing repository and preserve unrelated user changes.
2. Propose a small implementation plan before editing.
3. Build public read-only Polymarket connectors for market catalog, market metadata, order books/prices, and historical price snapshots. Do not add wallet keys or trading.
4. Store raw and canonical snapshots append-only with `observed_at`, `available_at`, payload hashes, and snapshot manifests.
5. Implement canonical `Market`, `Outcome`, `ForecastRequest`, `FeatureSpec`, `ForecastFrame`, `ForecastResult`, `ModelBundleManifest`, and `PredictionLedger` contracts.
6. Implement one `ForecastService` that materializes a cutoff-safe feature frame, routes a market-family model, calibrates probability, computes executable price/fees/slippage/net edge, writes a numerical trace, and returns only WATCH/PASS/PAPER outcomes.
7. Start with a generic binary-market price-calibration model and a time-respecting rolling replay. Evaluate Brier, log loss, calibration, coverage, and conservative paper P&L. Do not claim edge until this baseline is measured.
8. Classify binary, mutually exclusive categorical, scalar-bucket, sports moneyline/spread/total, and unsupported markets. Reject ambiguous settlement semantics.
9. Implement strict leakage tests: change every source row with `available_at > cutoff`; materialized features and forecast must be byte-equivalent. Ensure the model training cutoff predates each replayed event.
10. Version model, feature, configuration, snapshot, calibration, fee-model, and code hashes together. Promote a candidate only after paired out-of-sample evidence improves proper scores without material regression in any critical cohort.

Important constraints:
- Market midpoint is not an executable price; model bids/asks, depth, tick/minimum order size, fees, and slippage using current documented metadata.
- The public market APIs can change; consult current official Polymarket documentation before coding endpoints or fee logic.
- Treat question text/news/LLM extraction as audited evidence with source/cutoff/citation metadata, never as an unlogged black box.
- Design market families as plugins. Do not pretend one model will forecast sports, politics, crypto, macro, weather, and custom markets equally well.
- Default to strict failure for unavailable/stale/ambiguous critical data; otherwise return an explicit degraded quality label and `PASS`.
- Do not execute trades or store credentials in the first release.

After implementation, run focused tests and provide: architecture summary, files changed, test results, known limitations, and the next measurable milestone.
```

## Reviewed repository map

The review covered the root CLIs and configuration; all `src/` packages; current architecture/rebuild notes; and the mirrored test packages. The most important reusable source areas are:

- `src/contracts/`: artifact, feature-frame, projection, schedule, and new forecast request/output contracts.
- `src/data/`: public-data scrapers, cache/health handling, and the opt-in extension registry.
- `src/preprocessing/` and `src/features/`: feature groups, extension discovery, diagnostics/cache, registry, and new point-in-time materializer.
- `src/training/` and `src/models/`: CatBoost/Transformer training, quantiles, weighting, schema alignment, model manager, and bundle versioning.
- `src/pipeline/`: legacy bridge plus the new single `ForecastService` runtime seam.
- `src/query/`: distribution fitting, empirical covariance, probability calculation, parsing, reporting, and projection loading.
- `src/simulation/`: roster/context/role/phase/possession/correlation/report layers; keep only the general simulation decomposition, not basketball logic.
- `src/correction/` and `src/evaluation/`: residual calibration/correction, backtesting, metrics, weights, drift, feature selection, ledger, monitoring, and promotion gates.
- `src/reasoning/`: auditable numerical evidence reports and JSONL sidecars.
- `tests/`: contracts, feature leakage, extension mechanisms, model bundles, forecast service, ledger, continual learning, correction, query, simulation, and training smoke coverage.

## Official Polymarket references to re-check before implementation

- API introduction: https://docs.polymarket.com/api-reference/introduction
- Public market-data overview: https://docs.polymarket.com/market-data/overview
- Prices and order books: https://docs.polymarket.com/concepts/prices-orderbook
- Historical price endpoint: https://docs.polymarket.com/api-reference/markets/get-prices-history
- CLOB market metadata: https://docs.polymarket.com/api-reference/markets/get-clob-market-info

These official materials describe separate Gamma, Data, and CLOB APIs; public read endpoints; market/event organization; and CLOB pricing/order-book concepts. Recheck the current API schema, fees, trading rules, eligibility, and applicable terms before adding a live connector or any execution capability.
