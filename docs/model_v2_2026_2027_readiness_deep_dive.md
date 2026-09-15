# Model v2: Detailed 2026–2027 NBA Season Readiness Audit

**Audit date:** September 5, 2026  
**Repository:** `/Users/jaylenbain/Documents/knowing-master`  
**Scope:** Current local working tree, Model v2 runtime, source data, canonicalization, training, replay, simulation, probability queries, and release operations.  
**Deliverable:** Findings and implementation plan. This audit does not modify model code, refresh NBA data, train a production candidate, or change the champion.  
**Readiness decision:** **Not ready for official 2026–27 forecasts. Continue development and diagnostic work; retain all promotion gates.**

## 1. Executive assessment

The project has a useful v2 foundation: immutable raw snapshots with checksums, explicit forecast requests, canonical table contracts, game-isolated fold helpers, lagged baselines, a v2 bundle validator, promotion policies, an official ledger, and probability-query support. The correct decision is to complete and connect these parts.

The biggest problem is that these pieces do not yet form a complete, validated forecasting system. The supported trainer produces an explicitly ineligible rolling baseline. The scheduled-game runner always rejects strict mode because it has no official point-in-time roster path. The replay command scores supplied predictions but does not generate historical forecasts through the live service. The local workspace has no `models/` directory or discovered `champion.json` pointer. Existing on-disk source snapshots use a schema the current loader rejects.

There are also numerical and operational bugs that remain hidden by passing tests. Most seriously, the minutes allocator and forecast service use incompatible definitions of expected minutes. A controlled reproduction produced 120 participation-weighted expected minutes per team while the simulation allocator assigns 240. This means forecast means, quantiles, and simulated distributions cannot currently be treated as equivalent views of one model.

**Recommended order:** repair input provenance and numerical contracts; implement official rosters and historical replay; fit and evaluate the simplest complete v2 model; then prove shadow operations and promotion. More features, a larger Transformer, or additional simulator adjustments will not solve the current release blockers.

### 1.1 What “ready” should mean

Opening-night readiness requires evidence for all of the following:

1. Every official forecast refers to an actual scheduled game, a pre-tip cutoff, a verified source snapshot, and a supported v2 bundle.
2. Eligible players come from membership records known at that cutoff, including inactive players and DNPs where appropriate.
3. Participation, conditional minutes, and full-game stat distributions have consistent definitions across training, replay, forecasting, simulation, and queries.
4. The model beats the declared baselines on repeated unseen, game-isolated folds under the saved promotion policy.
5. Calibration is measured using held-out evidence, including participation, minutes, stat tails, zero mass, and important player slices.
6. Forecasts and reconciliation records survive retries without contradictory output or silent replacement.
7. Seven consecutive qualifying shadow slates, a clean-process inference test, and a rollback/restore drill have stored evidence.

The architecture plan remains the design source of truth. Its older references to migration-time legacy support should not override the current [AGENTS.md](/Users/jaylenbain/Documents/knowing-master/AGENTS.md) requirement that the supported runtime is v2 only.

## 2. Audit method and evidence limits

### 2.1 Work performed

- Read the full-architecture plan and inspected the relevant v2 CLI and implementation paths.
- Inspected snapshot creation and validation, canonicalization, identity resolution, coverage reporting, scheduled materialization, availability, minutes, rate conversion, the baseline backend, forecasting, joint sampling, training, folds, replay scoring, promotion validation, ledgering, shadow evidence, and probability queries.
- Read the current local CSVs to establish actual date coverage, row counts, missingness, and identity mismatches.
- Executed the repository-prescribed focused v2 tests: **39 passed in 2.45 seconds**.
- Executed additional targeted opportunity, scheduled-feature, forecast-service, and forecast/feature-contract tests: **26 passed in 0.98 seconds**.
- Checked `--help` for nine supported CLIs; all exited successfully.
- Executed small, offline reproductions for numerical, query, retry, status, season-selection, and calibration defects.
- Checked official NBA schedule information only for season-planning context. The NBA has published the 2026–27 regular-season schedule; use that schedule as the authoritative fixture source. [NBA schedule announcement](https://www.nba.com/news/2026-27-nba-regular-season-schedule)

### 2.2 What this audit does not prove

The full suite was not run, as instructed by [AGENTS.md](/Users/jaylenbain/Documents/knowing-master/AGENTS.md). No current model accuracy, profitability, calibrated coverage, or production latency is claimed. No production bundle was available in the default local model directory to evaluate. Passing 65 focused tests establishes that those tests pass; it does not establish season readiness.

The working tree already contained many modified and untracked files. Findings refer to the inspected local state, including those changes, rather than an assumed clean release commit. Existing work was preserved. File locations below should be rechecked after subsequent edits.

### 2.3 Evidence labels

| Label | Meaning |
|---|---|
| **Reproduced** | A small offline execution demonstrated the behavior during this audit. |
| **Code-confirmed** | The relevant implementation was read and the behavior follows directly from that path; a complete production run was not attempted. |
| **Observed artifact state** | The finding comes from local files or a loader result. |
| **Proposed improvement** | A design or modeling recommendation that must be validated experimentally. |

Priorities: **P0** blocks official release or undermines validity; **P1** is required for a reliable seasonal workflow; **P2** is a useful improvement after the core path is trustworthy. A P0 in a currently disabled path means “fix before enabling,” not that official production predictions are presently being published.

## 3. Current data and artifact inventory

| Asset | Observed state | Implication |
|---|---|---|
| `data/nba_players.csv` | 104,878 rows; October 18, 2022 through April 10, 2026 | More history than the older plan described, but not a current offseason/season preparation dataset. |
| Player rows by season | 25,894 in 2022–23; 26,401 in 2023–24; 26,306 in 2024–25; 26,277 in 2025–26 | Enough history to build useful rolling experiments, subject to actual coverage and provenance checks. |
| Player identity keys | 4,902 distinct game IDs; zero duplicate `(GAME_ID, PLAYER_ID)` rows | Row-level duplicate checking passed; cross-table completeness still fails. |
| Player minutes | No missing numeric `MIN`; nine rows with `MIN == 0` | Appearance labels must not simply be set to one for every row. |
| `data/nba_games.csv` | 9,802 rows; 4,901 distinct games; same date endpoints as player data | One player-side game lacks a team-side match. |
| Identity resolution | 4,901 exact IDs; one quarantined game; 28 quarantined player rows; zero remaps | The current resolver safely quarantines the mismatch, but the upstream missing game still needs repair. |
| Quarantined game | Source ID `22501183`, April 10, 2026; team IDs `1610612746` and `1610612757` | Investigate official source records for both teams and this date; do not guess a replacement ID. |
| `data/player_bios.csv` | 876 rows; birthdate coverage 100% in this file | Bios now exist. Presence does not mean the v2 canonicalizer uses them. |
| `data/injury_history.csv` | 36 rows; `DATE` entirely missing | Cannot serve as historical point-in-time injury evidence. |
| Local source snapshots | Two August 27 snapshot manifests | Both rejected by current loading with `Unsupported source snapshot schema: 'source_snapshot_v1'`. |
| Canonical artifacts | An older canonical directory exists for one snapshot | Current source-schema rejection blocks normal validated loading; the directory layout also predates the currently produced set of tables. |
| Default model artifacts | `models/` absent; no `champion.json` found by the repository scan | The default runtime has no local configured champion. This does not establish whether artifacts exist outside this workspace. |

### 3.1 Corrections to older documentation

Do not copy the older plan’s inventory or accuracy tables into a current readiness claim. In particular, the present core history begins in 2022, and bios/injury files are now present. The injury file is unusable for dated replay in its present form, and bios are not wired into the v2 canonical path. Historical candidate MAE tables in prior plans were not rerun here and should remain clearly labeled historical evidence.

## 4. Prioritized repair register

| ID | Priority | Finding | Evidence |
|---|---|---|---|
| B01 | P0 | Participation and minutes semantics disagree across service and sampler | Reproduced |
| B02 | P0 | Strict scheduled forecasting has no official roster implementation | Code-confirmed |
| B03 | P0 | Forecast input history and schedule are not bound to the declared snapshot | Code-confirmed |
| B04 | P0 | Replay is a scorer, not a historical execution pipeline | Code-confirmed |
| B05 | P0 | Training scorecards do not measure the fitted runtime candidate | Code-confirmed |
| B06 | P0 | Existing snapshots cannot load under the current schema | Observed artifact state |
| B07 | P0 | Runtime service retains silent heuristic fallback behavior | Code-confirmed |
| B08 | P0 | Forecast contract does not verify request/backend identity or completeness | Code-confirmed |
| B09 | P0 | Stat distributions, zero mass, and simulation marginals disagree | Code-confirmed |
| B10 | P1 | 2026–27 is omitted by season-range selection | Reproduced |
| B11 | P1 | Forecast retry changes `GENERATED_AT` and conflicts with the ledger | Reproduced |
| B12 | P1 | Date-filtered query fails on service-generated forecasts | Reproduced |
| B13 | P0 | Minutes allocator permits impossible rotations | Reproduced |
| B14 | P0 | Coverage reporting can conceal excluded games and empty datasets | Code-confirmed |
| B15 | P1 | Undated status snapshots are accepted despite a requested cutoff | Reproduced |
| B16 | P1 | Canonicalization ignores supplied optional and official-roster sources | Code-confirmed |
| B17 | P1 | Canonical outputs lack content verification and transformation identity | Code-confirmed |
| B18 | P1 | Simulation output files overwrite earlier runs | Code-confirmed |
| B19 | P1 | Shadow streak evidence is not scoped tightly enough to a candidate | Code-confirmed |
| B20 | P1 | Reconciliation omits entire games absent from the actuals input | Code-confirmed |
| B21 | P1 | Probability calibrator fails for a supported method configuration | Reproduced |
| B22 | P1 | Probability parsing accepts invalid NBA count distributions | Reproduced |
| B23 | P1 | Promotion/load checks need stronger runtime and evidence binding | Code-confirmed |
| B24 | P1 | No complete seasonal operations and failure-recovery workflow | Code-confirmed gap |

## 5. Detailed bugs and required fixes

### B01 — Repair the participation/minutes contract before changing model complexity

**Locations:** [src/forecasting/minutes.py:9](/Users/jaylenbain/Documents/knowing-master/src/forecasting/minutes.py:9), [src/forecasting/baseline_backend.py:40](/Users/jaylenbain/Documents/knowing-master/src/forecasting/baseline_backend.py:40), [src/pipeline/forecast_service.py:178](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:178), [src/forecasting/rates.py:9](/Users/jaylenbain/Documents/knowing-master/src/forecasting/rates.py:9).

The allocator calculates weights as raw conditional minutes multiplied by `PLAY_PROB`, normalizes those weights to 240 per team, and writes the result as `EXPECTED_MINUTES`. The backend multiplies those allocated minutes by a stat rate. The forecast service subsequently treats that prediction as conditional and multiplies by `PLAY_PROB` again.

The architecture instead defines public `EXPECTED_MINUTES` as conditional on appearing. It requires the sum of `PLAY_PROB × EXPECTED_MINUTES` over a team's players to equal 240. The current allocator guarantees the unweighted sum equals 240. These are different contracts.

**Reproduction:** Ten players on each team, raw minutes 24 each, `P_ACTIVE=1`, `P_PLAY_GIVEN_ACTIVE=0.5`, and deterministic PTS rate 1 per minute. The actual backend's preparation/prediction methods and forecast service produced team `MEAN` totals of **120**, and participation-weighted minutes also totaled **120**. The sampler's successful rotations allocate **240**. The fixture was in-memory; it did not load or promote an artifact.

**Required repair:**

1. Introduce explicitly named internal conditional and unconditional minute fields.
2. Define a feasible joint participation-and-rotation distribution.
3. Derive public conditional minutes from that distribution, rather than relabeling normalized expected shares.
4. Compute stat means, quantiles, and zero mass from the same full-game distribution used by simulation.
5. Remove or align the separate `totals_from_minutes_and_rates` helper so it cannot become a third interpretation of the contract.

**Acceptance tests:** A deterministic one-rate fixture conserves team expectation; mixtures with unequal participation probabilities behave correctly; each simulated regulation team totals 240; public conditional means agree with empirical conditional sample means; direct and sampled unconditional totals agree within a prespecified Monte Carlo tolerance.

Do not fix this merely by deleting one multiplication. That may repair one fixture while leaving conditional minutes, unequal probabilities, and the allocator's feasibility assumptions inconsistent.

### B02 — Implement the official roster path without weakening strict mode

**Location:** [src/simulation/v2_runner.py:25](/Users/jaylenbain/Documents/knowing-master/src/simulation/v2_runner.py:25).

The runner always calls `latest_observed_roster`, sets degraded quality, and raises when `strict=True`. Supplying official roster data does not change that control flow. The normal CLI therefore cannot complete an official forecast. This refusal is an appropriate safeguard, but the implementation behind it is missing.

**Required repair:** Resolve membership from canonical effective intervals with `AVAILABLE_AT <= cutoff`; apply pre-cutoff transactions and team changes; construct the complete eligible player universe; attach timestamped status and lineup records; explicitly reject unknown mandatory membership evidence. Keep appearance-derived membership only in the declared non-official scenario.

Add an explicit v2 candidate execution mode for replay and shadow evaluation. It must not require promoting an unqualified candidate first, and must not silently replace champion selection. This resolves the present practical dead end: candidates need shadow evidence, while the runner only loads the champion.

**Acceptance tests:** Valid official roster data allows strict execution; missing official membership fails; trades and waive/re-sign intervals choose exactly one eligible team; rookies with no NBA appearances remain forecastable; inactive players are retained with honest labels; degraded runs never enter the official ledger.

### B03 — Bind actual feature inputs to the request's snapshot

**Locations:** `simulate_season.py:47`, `simulate_season.py:104`, [src/simulation/v2_runner.py:25](/Users/jaylenbain/Documents/knowing-master/src/simulation/v2_runner.py:25), [src/features/scheduled.py:10](/Users/jaylenbain/Documents/knowing-master/src/features/scheduled.py:10).

The CLI reads mutable `data/nba_players.csv` and either an external schedule CSV or a live schedule fetch. The runner validates the named snapshot, but that validation does not establish that these supplied frames came from it. Scheduled materialization filters history by game date, not row-level availability time. As a result, a valid snapshot ID can accompany unrelated or later input facts.

**Required repair:** Build requests from a versioned canonical schedule and load history, membership, status, and context through snapshot-bound interfaces. Gate observable facts by `AVAILABLE_AT`. Preserve source references or feature-lineage hashes. Verify that a bundle's fitted data and any updated history state are eligible for the request cutoff.

**Acceptance tests:** Editing mutable root CSVs after snapshot creation cannot change a forecast for the original request. Facts arriving after cutoff are excluded even when their event date is earlier. A post-cutoff schedule correction does not rewrite a historical tip. Same-game outcome mutations never change that game's feature frame.

The generic feature materializer has a different interface from the scheduled-row helper. Consolidate their ownership. Explicitly test the canonical uppercase timestamp fields against any lower-case adapter expectations; do not assume the existing generic filter is automatically applied to scheduled rows.

### B04 — Build real replay, then treat scoring as a separate stage

**Locations:** `replay.py:16`, [src/evaluation/replay.py:25](/Users/jaylenbain/Documents/knowing-master/src/evaluation/replay.py:25).

The replay command reads Parquet forecasts and CSV actuals and computes MAE, RMSE, 80% coverage, and reconciliation fraction. It does not rebuild requests, load a fold-specific candidate, materialize historical inputs, generate forecasts, or prove live/replay parity. Configured rolling-window settings are not an execution plan in this command.

**Required repair:** For each frozen fold and horizon, generate historical requests from eligible sources, load the correct candidate, invoke the same v2 forecasting service used live, persist immutable forecast distributions, and reconcile against separate labels. Keep `score_replay` as a reusable scoring component.

Persist fit/tune/calibration/test partitions, source evidence tiers, request sets, excluded games with reasons, baseline predictions, candidate predictions, and paired comparisons. Tier C outcome history may support diagnostic fitting; it must not masquerade as official injury- or lineup-aware replay.

**Acceptance tests:** One golden request produces the same semantic forecast through live, replay, and simulation consumers. Changing an outer-test label cannot change trained components or calibration. All requested games and targets are accounted for, including failed and missing forecasts. At least two real evaluated folds satisfy the current bundle contract; prefer more independent seasonal windows when coverage permits.

### B05 — Measure the candidate actually trained, and make cutoffs truthful

**Location:** [src/training/v2_baseline.py:33](/Users/jaylenbain/Documents/knowing-master/src/training/v2_baseline.py:33) and its `_component_payloads`, `_baseline_scorecard`, `_cutoffs` helpers.

The trainer generates fold definitions, but fits component summaries over all `player_games`. It computes a rolling-10 total-stat scorecard and writes that same scorecard as both baseline and candidate evidence. Runtime uses rolling-20 per-minute components plus team allocation, so that copied scorecard is not a measurement of the runtime predictor.

The manifest's `training` cutoff comes from the final fold's `fit_end`, while fitted component payloads use the full input frame. A final refit can legitimately use later history, but its provenance must say so and its scorecards must refer to the separately evaluated fold models. Validation and calibration also receive the same recorded cutoff without a distinct calibration partition.

**Required repair:** Implement fold-specific fitting and scoring, distinct tune/calibration partitions, then a separately identified final refit. Save the candidate's actual predictions rather than copying a baseline metric object. Record the maximum label availability used for each component. Honor or explicitly reject model/config options rather than saving configurable-looking settings while using hard-coded rolling-20 calculations.

**Acceptance tests:** Modifying runtime rates changes candidate metrics; modifying outer labels cannot change a fold model; every fitted record belongs to its declared partition; final-refit metadata covers its true data; unsupported settings fail with an actionable error.

The current explicit `eligible=False` decision is correct and must remain until these missing evidence paths are implemented.

### B06 — Resolve the existing source-snapshot schema break explicitly

**Location:** [src/data/snapshots.py](/Users/jaylenbain/Documents/knowing-master/src/data/snapshots.py) and the current source contract; local `data/manifests/`.

Both stored source snapshots fail current loading with `ContractError: Unsupported source snapshot schema: 'source_snapshot_v1'`. Thus the normal coverage/canonical-loading path cannot use them as-is.

**Required repair:** Preserve the old snapshots. Either create a documented one-way migration into new immutable records, retaining old manifest identities and provenance, or create a fresh supported snapshot from current sources. A newly captured file has new availability evidence; do not backdate it to make historical replay pass. Keep the old schema unsupported in production if that is the intended contract.

**Acceptance tests:** A supported snapshot can be validated, canonicalized, and loaded end to end. Any migration produces a new identity with explicit lineage. Old raw payloads and manifests remain unchanged. Unsupported versions give a clear recovery message. No legacy model fallback is introduced.

### B07 — Remove silent forecast fallback paths from the supported service

**Locations:** [src/pipeline/forecast_service.py:92](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:92), `:117`, and `_predict_legacy_pipeline`.

The service still returns fixed/rolling heuristics on empty inputs, catches per-player prediction errors and substitutes fallback values, and supports a legacy prediction branch. Missing target values can also become fixed defaults while constructing the canonical forecast. These behaviors conflict with a strict v2-only runtime.

The default constructor correctly refuses a missing champion and rejects a non-v2 champion. Preserve that behavior. The issue is the remaining prediction and injected-backend paths, not an assertion that default construction currently loads legacy artifacts.

**Required repair:** Require an explicit v2 backend protocol, validate required outputs, propagate strict failures, and record failed request/player coverage. A declared cold-start prior inside a trained and evaluated v2 component is acceptable; silently rescuing an execution error with fixed points/rebounds is not.

**Acceptance tests:** A component exception, missing stat, malformed prediction, or unsupported backend prevents official publication. No fallback result inherits `FULL` quality. Diagnostic scenarios are explicitly labeled and excluded from official evidence.

### B08 — Validate provenance and completeness at the service/ledger boundary

**Locations:** [src/contracts/forecast.py:159](/Users/jaylenbain/Documents/knowing-master/src/contracts/forecast.py:159), [src/pipeline/forecast_service.py:178](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:178), [src/operations/ledger.py:24](/Users/jaylenbain/Documents/knowing-master/src/operations/ledger.py:24).

The output validator checks numeric domains, probability multiplication, and quantile ordering, but does not fully validate required identity values, timestamps, request consistency, eligible team membership, required target coverage, or the expected player universe. The service writes `request.model_bundle_id` without comparing it with the loaded backend. The existing service fixture even uses different request and backend bundle IDs while expecting successful output.

The ledger checks the `official` scenario string, but that string alone does not establish source or bundle eligibility. An empty context can also return an untyped empty frame before output validation.

**Required repair:** Bind request, actual loaded bundle, source manifest, schedule, eligible players, and complete target set at one boundary. Validate request-level uniformity, uniqueness, finite timezone-aware timestamps, team/opponent relationships, data quality, and distribution identity. Validate the complete request against the saved manifest, not just rows independently.

**Acceptance tests:** Wrong bundle ID, unknown snapshot, missing player, missing required stat, duplicate player/stat, blank request ID, mismatched opponent, and a falsely labeled official/degraded row all fail before official persistence. Legitimate no-game days use an operations status rather than fabricated forecasts.

### B09 — Make one distribution drive forecasts, samples, and probability queries

**Locations:** [src/pipeline/forecast_service.py:214](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:214), `_mixture_quantiles`, [src/simulation/joint_sampler.py:11](/Users/jaylenbain/Documents/knowing-master/src/simulation/joint_sampler.py:11), [src/query/v2_probability.py:13](/Users/jaylenbain/Documents/knowing-master/src/query/v2_probability.py:13).

The service builds zero-inflated clipped-Normal quantiles from mean/std outputs. The sampler independently draws a Normal rate, clips negative rates to zero, multiplies by allocated minutes, and rounds to an integer. Its output is not the distribution serialized by the service. The baseline backend supplies no conditional zero probability, so forecast `ZERO_PROB` generally accounts for nonparticipation only, despite the sampler generating zeros among participants.

Clipping a Normal variable also changes its mean. Using the unclipped location parameter as the expected value is not generally consistent with the clipped distribution. This matters particularly for low-count blocks and steals. Minute uncertainty is not sampled from `MINUTES_STD`; the sampler reallocates deterministic raw minute weights after participation. Overtime is absent from the v2 sampler.

**Required repair:** Choose and persist one coherent full-game distribution representation: calibrated count PMFs, supported distribution parameters, or reproducible joint samples. Compute all reported summaries from it. Model regulation and overtime separately. Evaluate Poisson, negative-binomial, hurdle, or empirical alternatives as experiments, not assumptions of improvement.

**Acceptance tests:** Reported mean, quantiles, and zero mass match the persisted distribution; simulated conditional and unconditional marginals agree; integer-line push probabilities are reproducible; all stats have nonnegative integer support; full-game labels and overtime modeling share definitions.

### B10 — Fix season rollover and incremental date-range enumeration

**Locations:** `update_data.py:55`, `:104`.

The static `SEASONS` list ends at `2025-26`. `get_seasons_between_dates` also advances by year in a way that skips an October boundary within an April-to-October interval. The reproduced call `get_seasons_between_dates('2026-04-10', '2026-10-25')` returns only `['2025-26']`.

This is directly relevant to the local April cutoff and the upcoming season: an incremental refresh can fail to request 2026–27. Explicit single-season fetching is a separate path; this finding does not claim every way of requesting 2026–27 fails.

**Required repair:** Generate season identifiers dynamically from the requested interval and enumerate every crossed season boundary. Use a clear season policy, explicit timezone, and explicit preseason/regular-season/postseason handling. Avoid annual maintenance of a static terminal season.

**Acceptance tests:** April-to-October 2026 includes both seasons; October-only 2026 includes 2026–27; multiple-year ranges include every expected season; leap-day starts do not fail; default and explicit season choices behave consistently.

### B11 — Make forecast retries return the original result

**Locations:** [src/pipeline/forecast_service.py:195](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:195), [src/operations/ledger.py:24](/Users/jaylenbain/Documents/knowing-master/src/operations/ledger.py:24).

The request ID is stable, but each forecast generation gets a fresh `GENERATED_AT`. The ledger compares the full frame. Two identical calls followed by writes produced `ValueError: Conflicting immutable forecast payload for REQUEST_ID ...` on the second write.

**Required repair:** Look up a saved request before recomputation and return its original payload. If computation retries are necessary, use a persisted creation time and distinguish semantic prediction content from execution-attempt metadata. Keep contradictory-value detection. Do not broadly ignore metadata differences without defining what establishes forecast identity.

**Acceptance tests:** Sequential and concurrent identical retries return the same saved forecast and original generation time; differing prediction content for the same request is rejected; retries do not append new official rows or overwrite existing payloads.

### B12 — Persist enough game identity for date and horizon queries

**Locations:** [src/contracts/forecast.py:25](/Users/jaylenbain/Documents/knowing-master/src/contracts/forecast.py:25), [src/pipeline/forecast_service.py:231](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:231), [src/query/v2_probability.py:270](/Users/jaylenbain/Documents/knowing-master/src/query/v2_probability.py:270).

The service does not emit `GAME_DATE`, while `select_forecast(..., game_date=...)` requires it. Querying a service-produced forecast by date reproduced `ValueError: Forecast rows do not contain GAME_DATE`.

Selection also chooses the latest `GENERATED_AT` among matching player/stat rows. That is not necessarily the intended game, cutoff horizon, or official scenario, particularly when a file contains reruns or multiple slates.

**Required repair:** Save `GAME_DATE`, scheduled tip, and request identity in a versioned output contract or resolve them through the immutable request manifest. Support explicit game, horizon, cutoff, bundle, and scenario filters. Reject ambiguity rather than picking a different game because it was generated later.

**Acceptance tests:** Date queries work on actual service output; multiple horizons are selected explicitly; two games in a combined file cannot be confused; non-official rows do not silently win over official rows.

### B13 — Enforce feasible rotations and handle infeasible participation draws

**Locations:** [src/forecasting/minutes.py:9](/Users/jaylenbain/Documents/knowing-master/src/forecasting/minutes.py:9), [src/simulation/joint_sampler.py:11](/Users/jaylenbain/Documents/knowing-master/src/simulation/joint_sampler.py:11).

Two participating players receive 120 regulation minutes each in a reproduced allocator call. Independent participation sampling can also select no players; the sampler then raises `Cannot allocate minutes ... no participants`. Fewer than five participants cannot supply 240 regulation player-minutes with a 48-minute per-player cap.

**Required repair:** Define a feasible joint participation model or an explicit conditioning procedure; cap regulation minutes at 48; allocate within available capacity; reject infeasible roster inputs with a useful reason. Preserve calibrated marginal participation where possible and document any conditioning that changes it. Do not silently force an `OUT` player to participate to fill minutes.

**Acceptance tests:** Every valid regulation rotation totals exactly 240 with each player's minutes between 0 and 48; nonparticipants have zero; impossible rosters fail deterministically; realistic uncertain rosters complete without random whole-slate failures; observed participation frequencies match the model's stated probabilities after conditioning.

### B14 — Measure coverage against expected inputs, including quarantined records

**Locations:** [src/data/canonicalize.py:42](/Users/jaylenbain/Documents/knowing-master/src/data/canonicalize.py:42), [src/data/coverage.py:25](/Users/jaylenbain/Documents/knowing-master/src/data/coverage.py:25), `check_data.py:12`.

Canonicalization removes invalid games before the coverage report counts games. Coverage then computes `valid_final_games / games`, where the canonical game builder labels retained games `final`. This can yield 100% among survivors while hiding missing or quarantined source games. The CLI's success decision checks only whether player/team joins are missing, not the configured fraction, official rosters, freshness, or nonempty data.

**Required repair:** Retain denominators for expected scheduled games, raw source games, accepted canonical games, quarantined games, and expected player rows. Report each separately. Apply the configured core threshold and strict required-source/freshness gates. Empty inputs must not pass because they contain zero missing joins.

**Acceptance tests:** Drop a complete game, quarantine a malformed game, omit all player rows for a team, and submit an empty core dataset; each lowers coverage or fails a clearly named gate. The current 28-row quarantined game must remain visible in readiness reporting.

### B15 — Reject undated status evidence when a cutoff is requested

**Location:** [src/forecasting/availability.py:163](/Users/jaylenbain/Documents/knowing-master/src/forecasting/availability.py:163).

`_latest_as_of` filters timestamps only if the availability column exists. Passing an undated `OUT` status with a historical cutoff resulted in `PLAY_PROB=0.0`, despite the public helper's documentation saying undated snapshots are excluded.

**Required repair:** With a cutoff, exclude or reject undated external status rows; expose a missing-evidence reason. Only permit a pre-materialized candidate status when its lineage has already been validated. Key status observations by the relevant game/team context where the provider's record is game-specific.

**Acceptance tests:** Missing timestamps cannot alter a strict historical prediction; a post-cutoff status is ignored; an eligible pre-cutoff status is used; timestamp precision, null statuses, and traded-player records are handled deliberately.

### B16 — Ingest real canonical membership/status/bios instead of empty placeholders

**Location:** [src/data/canonicalize.py:42](/Users/jaylenbain/Documents/knowing-master/src/data/canonicalize.py:42), `:246`, `:270`.

The raw snapshot file list can include rosters, eligibility, injuries, lineups, bios, and odds. Canonicalization nevertheless derives membership from player appearances and creates empty status, lineup, odds, and bios tables. Supplying these files is therefore insufficient to make them operational.

The observed eligibility builder sets `ACTIVE=1` and `APPEARED=1` for every player-game row. This does not establish the inactive/DNP denominator, and the local player file contains nine zero-minute rows.

**Required repair:** Add versioned source-specific adapters, validate identity and publication time, preserve unknown labels, and separate official eligibility from appearance-derived diagnostics. Use positive official playing time to define appearances; investigate zero-minute representations rather than inferring all labels from row presence. Join bios using stable facts such as birthdate and calculate age at the historical request date.

**Acceptance tests:** Supplied official records populate the corresponding tables; malformed optional inputs produce visible coverage errors; unknown active labels remain unknown; zero-minute rows receive a defensible appearance label; rookie and traded-player membership is covered.

### B17 — Make canonical data content-verifiable and versioned by transformation

**Locations:** [src/data/canonicalize.py:42](/Users/jaylenbain/Documents/knowing-master/src/data/canonicalize.py:42), `:136`.

Canonical outputs are scoped only by snapshot ID, and their manifest does not checksum each Parquet table or identify the full canonicalization code/config transformation. `load_canonical_table` validates the raw snapshot and table schema, but a schema-valid edit to a canonical statistic is not detected by a canonical content hash.

**Required repair:** Address canonical products by source identity plus schema, code, and configuration hashes; checksum tables and reconciliation reports; verify those checksums on load. Recanonicalizing after an identity fix must create a new immutable version, with the old one retained for existing forecasts.

**Acceptance tests:** A schema-valid change to one stat fails checksum validation. A changed canonicalization policy creates a different product identity. Concurrent writers cannot replace an already published canonical product. Loading an old forecast resolves its original canonical version.

### B18 — Stop overwriting forecast and sample exports

**Location:** `simulate_season.py:82`.

Each run writes to fixed `forecasts.parquet` and `samples.parquet` paths under the chosen output directory. Running another date or horizon replaces the prior files. Official ledger persistence is separate, but these exports and degraded run evidence still lose reproducibility.

**Required repair:** Use immutable request/run-addressed output directories with a manifest. Include request ID, bundle, snapshot, cutoff, scenario, seed, sampler version, simulation count, and content checksums. A convenience `latest` pointer can refer to an immutable run.

**Acceptance tests:** Different dates and horizons retain separate artifacts; same-request retries are idempotent; samples can be tied unambiguously to a forecast distribution; changing simulation count or seed changes sample-artifact identity.

### B19 — Tie shadow evidence to the candidate, horizons, and expected slate calendar

**Locations:** [src/operations/shadow.py:30](/Users/jaylenbain/Documents/knowing-master/src/operations/shadow.py:30), `:72`, `:84`; [src/models/bundle.py:314](/Users/jaylenbain/Documents/knowing-master/src/models/bundle.py:314).

The shadow store counts consecutive successful recorded rows without filtering for a single candidate bundle or required horizon set. It only requires a positive persisted count, not expected player/stat/request completeness. Missing scheduled slates are invisible if they were never recorded. The load-modify-write path also has no cross-process lock. Promotion reads a stored success count rather than resolving the underlying slate artifacts in this section.

**Required repair:** Scope evidence to the frozen candidate and required horizons; compare against expected game slates; count complete persisted requests; record missed runs; bind every slate to forecast/parity/coverage evidence. Use locked append-only records or equivalent concurrency-safe publication. Preserve earlier attempts when a date must be retried for another candidate.

**Acceptance tests:** Six slates from candidate A plus one from B do not qualify either candidate; a missing horizon or omitted scheduled slate breaks eligibility; partial persistence fails; dark days do not count as failures; simultaneous writers cannot lose a record.

### B20 — Reconcile the expected ledger population, not only games present in actuals

**Location:** [src/operations/ledger.py:60](/Users/jaylenbain/Documents/knowing-master/src/operations/ledger.py:60).

Reconciliation first selects forecast game IDs from the actuals input. If a whole expected game's actuals are absent, its forecasts are omitted from the selected population. Reconciliation can therefore look complete for the submitted subset while an entire game remains unaccounted for.

**Required repair:** Select expected forecast requests by the immutable slate/game-date manifest first, then left-join actuals. Persist missing games, missing players, unresolved DNPs, and corrected actual revisions. Do not invent zero outcomes merely because a player is missing from an incomplete box score.

**Acceptance tests:** Removing one game's actuals lowers reconciliation completeness; missing player rows remain explicit; final DNPs reconcile as zero only with supporting eligibility/final-status evidence; stat corrections create new immutable revisions.

### B21 — Repair probability-calibrator configuration and refit behavior

**Location:** [src/forecasting/availability.py:454](/Users/jaylenbain/Documents/knowing-master/src/forecasting/availability.py:454).

`ProbabilityCalibrator(methods=('platt',))` with four held-out observations reproduced `ValueError: tuple.index(x): x not in tuple`. Small datasets force an identity candidate even when identity is absent from `self.methods`; tie-breaking calls `self.methods.index('identity')`.

The fit method also retains `method_scores_` across calls instead of clearing per-fit state. That is a code-confirmed risk of stale candidates influencing a later fit, especially when the set of usable methods changes.

**Required repair:** Define a consistent minimum-data policy and ordering for any implicit identity method, or require identity explicitly with clear validation. Reset all fitted state at the start of a new fit. Persist the evidence partition and row count, not only a caller-supplied `heldout`/`oof` label.

**Acceptance tests:** Every supported method configuration behaves predictably on small or single-class evidence; repeated fits cannot reuse old scores; insufficient evidence is explicit; prediction remains bounded and finite.

### B22 — Validate PMF/sample support and probability provenance

**Location:** [src/query/v2_probability.py:193](/Users/jaylenbain/Documents/knowing-master/src/query/v2_probability.py:193), `:233`.

The parser checks finite masses and total mass, but accepts negative and fractional support values. A PMF with values `-1` and `1.5`, each with mass 0.5, returned an exact-method result for an NBA count-stat query.

**Required repair:** Require nonnegative integer support for the six count targets; validate duplicate support handling and consistency with the row's summaries. Distinguish “exact calculation from the supplied PMF” from “empirically calibrated predictive probability.” A PMF field name does not establish calibration. Persist calibration identity and evidence separately.

Quantile interpolation is already labeled approximate, which is appropriate. Keep that distinction. Do not present interpolation as an exact push probability or interpret a zero-mass field as validated when the producing model does not model participant zeros correctly.

**Acceptance tests:** Negative/fractional support fails; PMF total and summary consistency are checked; valid integer PMFs satisfy under + push + over = 1; empirical samples retain their approximate/Monte Carlo interpretation.

### B23 — Complete runtime backend selection and strengthen promotion evidence binding

**Locations:** [src/pipeline/forecast_service.py:49](/Users/jaylenbain/Documents/knowing-master/src/pipeline/forecast_service.py:49), `promote_model.py:37`, `promote_model.py:63`, [src/models/versioning.py:241](/Users/jaylenbain/Documents/knowing-master/src/models/versioning.py:241), [src/models/bundle.py:364](/Users/jaylenbain/Documents/knowing-master/src/models/bundle.py:364), `:606`.

The default service always constructs `V2BaselineBackend`. A future v2 CatBoost candidate needs a validated backend resolver based on supported component metadata. The clean-process promotion check validates artifact contracts but does not execute a representative forecast, so a structurally present but unusable component may evade that particular check.

Legacy branches remain in candidate validation/promotion. Whether an individual legacy artifact passes depends on lower-level contracts; remove the branches explicitly so the supported boundary is unambiguously v2-only. The promotion wrapper reads the previous pointer, promotes, and restores on failure without a lock covering the whole transaction, creating a concurrency risk if two promotion attempts overlap.

The current v2 validator usefully recomputes decisions from scorecards and rejects incomplete/invalid numeric evidence. However, some operational/contract facts remain supplied booleans or counts. Fold evidence validation checks required fields and positive counts but does not itself reconstruct partitions or predictions from immutable row evidence.

**Required repair:** Add supported v2 backend dispatch; perform clean-process inference and distribution validation; bind scorecards to actual fold/request/source hashes; independently validate fold chronology and overlap; resolve shadow evidence artifacts; serialize promotion plus rollback with a cross-process lock. Define the final immutable evidence-sealing lifecycle so new shadow evidence does not require editing a sealed bundle.

**Acceptance tests:** Unsupported component kinds fail before pointer mutation; a checksum-valid but unloadable model is rejected; wrong fold evidence cannot qualify; overlapping or duplicated test folds fail; concurrent promotion failure cannot restore a stale pointer over a different successful promotion; rollback restores the exact previous validated champion.

### B24 — Add an operational workflow that can run through a season

**Locations:** Supported top-level CLIs and `src/operations/`.

The architecture calls for a daily orchestration and durable run manifest. The inspected supported path does not provide the complete refresh → snapshot → canonicalize → validate → forecast → persist → reconcile → monitor workflow. CLI success/error conventions also differ, and [update_data.py](/Users/jaylenbain/Documents/knowing-master/update_data.py) creates a snapshot only when `--snapshot` is selected, despite the simplified workflow description implying snapshot creation is automatic.

**Required repair:** Implement a resumable daily command with explicit date/horizons, stage outputs, deadlines, source health, attempt IDs, immutable result pointers, and actionable failure codes. Treat a verified no-game day as a successful no-op. Distinguish it from a failed schedule fetch. Use bounded provider retries and avoid refreshing the same NBA source for every player or game.

**Acceptance tests:** A source outage, partial refresh, missing champion, failed player forecast, delayed actuals, no-game day, interrupted run, and retry each leave a clear durable status. The next run resumes safely. Benchmark typical and high-volume slates on the supported CPU environment. Record a restore and rollback drill.

## 6. Modeling improvements after the correctness fixes

These recommendations are experiments, not claims that a particular algorithm will improve accuracy. Make the evaluation path trustworthy before choosing winners.

### 6.1 Build an honest eligible player-game panel

Appearance logs alone exclude many examples the system most needs to learn: inactive stars, available bench players who do not enter, returning players, rookies, and traded players. Build one row per eligible player-team-game at each supported cutoff.

Keep separate targets for official active status, appearance given active, regulation minutes given appearance, overtime minutes, and full-game stats. Unknown labels should stay unknown and have explicit scoring eligibility. Include source coverage and identity resolution in every evaluation slice.

For 2026–27, prioritize opening-week rookie coverage, summer transactions, two-way/assignment status where supported by sources, injury-return minutes, and changed rotations. A last-appearance roster from April cannot establish October membership.

### 6.2 Train participation before total-stat prediction

The repository has useful status-rule, appearance-frequency, calibrated binary, and two-stage participation utilities. Wire them into actual training and inference with official labels. The current baseline backend mainly consumes supplied probabilities or constant defaults; it does not demonstrate that the richer utilities improve live forecasts.

Evaluate both `P_ACTIVE` and `P_PLAY_GIVEN_ACTIVE`, plus their final product, using Brier score, log loss, reliability curves, and sample counts. Compare against both status rules and honest recent-appearance baselines on the same eligible universe. Inspect unknown-status and questionable-status slices separately.

Do not learn injury probabilities from the current undated 36-row file. Begin collecting trustworthy snapshots now, and declare which historical periods cannot support horizon-specific claims.

### 6.3 Learn minutes and rotation uncertainty

Use rolling minutes and role medians as mandatory comparators. Candidate features may include prior rotation role, pre-cutoff expected starters, teammate availability, rest, recent workload, and return-from-absence indicators, but only when their source history supports replay.

Predict conditional opportunity and uncertainty, then reconcile through the feasible joint rotation model. Assess whether teammate absences redistribute minutes in a way that improves held-out predictions. Learn blowout and foul-risk effects only if they add value under realistic uncertainty; do not multiply several hand-tuned penalties onto a calibrated model.

Score conditional minutes MAE, pinball loss, interval coverage, interval width, nonparticipation mass, and rotation constraint violations. Separate regulation from overtime rather than training the 240-minute allocator against unadjusted full-game minutes.

### 6.4 Improve rate estimation with shrinkage and role information

The current component builder averages recent per-game per-minute rates. Tiny-minute appearances can produce unstable rates; cold-start players receive broad global values. Compare this method with ratio-of-sums estimates and exposure-aware shrinkage toward role/position priors.

Test rolling and exponentially weighted history windows; reset or adapt role information after trades and material role changes. Keep a player's long-term skill history separate from current team opportunity. Use cold-start uncertainty that reflects limited evidence, rather than a single generic expected rate.

A CatBoost challenger is a reasonable next complete v2 model because relevant training components already exist in the repository. Its entry into production still requires shared v2 contracts and promotion evidence. Transformer and other challengers should follow only after a simpler complete model provides a credible benchmark.

### 6.5 Calibrate count distributions on independent evidence

Select count families separately where appropriate for PTS, REB, AST, STL, BLK, and TOV. Compare proper scores, quantiles, zeros, and tail behavior, not only MAE. A distribution can have reasonable mean error and still give unusable probabilities around common thresholds.

Fit calibrators on a held-out chronological window or correctly generated out-of-fold predictions. Record which rows fit the predictor, select its settings, fit calibration, and evaluate the outer test. The existing fold helper's single validation partition must be extended or split explicitly for tune versus calibration.

Store actual calibrated distribution evidence with the forecast. The query module already supports PMFs and samples; the producing model must supply valid, traceable values rather than relying on a handful of quantiles.

### 6.6 Estimate dependence only after marginals are right

The current sampler induces some dependence through shared minute allocation but draws target rates independently. That is insufficient evidence for combined-stat or multi-player probability quality.

After marginal calibration, test dependence models using out-of-fold residuals or learned shared game factors: pace, team scoring environment, rotation competition, and within-player stat relationships. Compare joint outputs against observed covariance and combined-stat distributions. A more detailed possession simulator is not a prerequisite for launch, but advertised joint markets need explicit validation.

Keep scenario adjustments separate from official learned forecasts. Every production adjustment should have a version, rationale, and ablation result.

### 6.7 Disable unused or starved feature families

The repository contains a large feature-group ecosystem, while the supported v2 baseline uses a small amount of information. Do not equate an enabled YAML family with a feature actually materialized, fitted, and used in inference.

Create a feature audit table with registered name, source, timestamp rule, coverage by season/horizon, missingness policy, and measured ablation effect. Preserve `FeatureGroupRegistry` and the extension contracts. Disable unsupported families prospectively until their coverage and replay evidence are sufficient. Calculate time-varying bios such as age and experience at the historical cutoff instead of using present-day values.

## 7. Required evaluation design and scorecards

### 7.1 Chronology

For each outer fold, assign whole games to fit, tune, calibrate, and outer-test roles. Save concrete request IDs and source evidence, not only date boundaries. Calendar-based fold generation can create low- or zero-game windows around the offseason, so validate actual eligible counts before running a fold.

Compare expanding and bounded training history only inside the experiment protocol. Record all tried candidates and keep the final outer test out of feature selection, early stopping, blend selection, and calibration selection.

The final chosen architecture can be refit through the latest safe data only after those decisions are frozen. Give this refit its own truthful provenance and a fresh reserved calibration window or proper cross-fitted calibration evidence.

### 7.2 Mandatory scorecard contents

| Area | Required evidence |
|---|---|
| Point accuracy | MAE and RMSE by target, horizon, fold, and declared major slice. |
| Headline comparison | Per-target candidate MAE divided by the eligible baseline MAE; pre-registered macro aggregation. |
| Participation | Active and play-given-active Brier/log loss, product calibration, reliability curves, denominators. |
| Minutes | Conditional MAE, pinball loss, coverage, width, and all rotation-constraint violations. |
| Stat distribution | Proper score such as CRPS or discrete equivalent, quantile loss, 80%/90% coverage and width, zero/tail calibration. |
| Baselines | Rolling-5/10/20, EMA-10, minutes × rate, and applicable role/position priors; identical eligible request universe. |
| Uncertainty in improvement | Paired game-level bootstrap with seed, resampling scheme, interval, and target aggregation saved. |
| Data coverage | Expected games/players, predicted rows, missing rows, quarantine, evidence tier, and source age. |
| Operations | On-time persistence, strict/degraded counts, failures, deterministic replay, latency and memory. |

### 7.3 Existing thresholds to preserve

The current [config/model_v2.yaml](/Users/jaylenbain/Documents/knowing-master/config/model_v2.yaml) declares these thresholds; implement evidence generation against them rather than lowering them to qualify the baseline:

| Gate | Current configured value |
|---|---|
| Normalized MAE improvement | At least 1% |
| Maximum core-target regression | 1% |
| Maximum secondary-target regression | 2% |
| Overall interval coverage tolerance | 3 percentage points |
| Major-slice interval coverage tolerance | 5 percentage points |
| Maximum fallback rate | 5% |
| Maximum degraded rate | 10% |
| Minimum major-slice reconciliation | 95% |
| Core reconciled fraction | 99.9% |

The existence of aggregate fallback/degraded ceilings is not permission to publish heuristic rescues as official forecasts. Strict requests still follow the v2-only, no-silent-fallback rule. Confirm the precise eligible denominator for each gate before implementation and record it prospectively.

Require enough games for each slice to support a claim. Small slices should be labeled insufficient evidence and handled under the saved policy, not silently omitted to improve an aggregate.

### 7.4 Minimum season-readiness slices

Report starters/bench, high/low minutes, rookies or no-history players, traded/team-changed players, injury return, questionable/unknown status, back-to-backs, recent workload changes, and each supported horizon. Include early-season windows separately: October roles often differ from the previous season's closing rotations.

Do not assert quality for a slice without the relevant labels and pre-cutoff source history. Unsupported injury/lineup horizons remain diagnostic or shadow-only until evidence exists.

## 8. Implementation sequence

Estimates below are relative work sizes, not delivery promises: **S** is a narrow repair, **M** spans several components, and **L** is a substantial data/evaluation subsystem. Official point-in-time history is an external dependency that cannot be created by changing code alone.

| Work package | Scope | Depends on | Size | Completion evidence |
|---|---|---|---|---|
| W1: Reproducible release baseline | Preserve local changes; define a clean integration state; record focused checks; resolve schema migration plan | None | M | Known commit and artifact inventory |
| W2: Immediate deterministic bugs | B10–B12, B15, B21–B22; regression fixtures | W1 | M | Reproductions now pass without relaxed contracts |
| W3: Input integrity | B03, B06, B14, B16–B17; repair quarantined game through source evidence | W1 | L | Supported immutable canonical dataset and honest coverage |
| W4: Opportunity and distribution contracts | B01, B08–B09, B13; remove B07 fallbacks | W1; W3 for real-data validation | L | Forecast/sample/query parity and feasible rotations |
| W5: Official roster and candidate execution | B02; complete eligible panel and candidate selection | W3, stable W4 interfaces | L | Strict candidate forecast from real timestamped inputs |
| W6: Real fold training and replay | B04–B05; four-way chronology; baselines and proper scoring | W3–W5 | L | Immutable out-of-fold forecasts and scorecards |
| W7: First learned challenger | Participation, minutes, rate and count calibration experiments | W6 | L | Honest improvement over complete baseline |
| W8: Release operations | B18–B20, B23–B24; retries, shadow, rollback, backup | W4–W6; final candidate for release evidence | L | Qualifying shadow evidence and recovery drills |
| W9: Promotion decision | Review exact sealed candidate against unchanged gates | W7–W8 | S/M | Passing decision, clean-process inference, validated champion |

### 8.1 First implementation session

1. Preserve this audit and establish a reviewable integration state without deleting unrelated local work.
2. Add regression fixtures for season range, idempotent retry, missing date, undated status, calibrator method selection, and invalid count PMFs.
3. Write the shared conditional/unconditional minutes and distribution contract before editing the allocator/service.
4. Decide the supported snapshot migration/rebuild route and document provenance consequences.
5. Begin trustworthy roster/status capture. Time lost collecting native evidence cannot be recovered with an end-of-month retrain.

### 8.2 Before attempting a learned-model comparison

Finish a runnable strict candidate path, real historical request execution, baseline parity, complete eligible labels, and independent calibration partitions. A successful call to [train.py](/Users/jaylenbain/Documents/knowing-master/train.py) is not sufficient evidence that these dependencies exist.

### 8.3 If evidence is insufficient before the season

Continue explicitly non-official diagnostic/shadow runs and collect the missing evidence. Support fewer horizons or a narrower replayable input set only through an explicit prospective scope decision. Keep an unqualified baseline unpromoted. Do not backfill availability timestamps from current files or remove inactive-player validation to meet a calendar deadline.

## 9. Regression-test backlog

The following are proposed tests, not claims that those tests already exist. Add them to the closest focused modules rather than expanding every low-impact helper into redundant tests.

| Proposed test | Regression prevented |
|---|---|
| `test_service_and_sampler_share_unconditional_minutes` | B01: opportunity counted inconsistently |
| `test_conditional_minutes_match_sampled_participants` | Incorrect public minute semantics |
| `test_strict_runner_uses_official_roster_snapshot` | B02: strict mode permanently unusable |
| `test_root_csv_mutation_cannot_change_snapshot_forecast` | B03: provenance detached from inputs |
| `test_post_cutoff_status_and_schedule_revision_excluded` | Historical lookahead |
| `test_replay_executes_fold_candidate_through_service` | B04: scoring-only replay mistaken for execution |
| `test_candidate_scorecard_uses_candidate_predictions` | B05: copied baseline scores |
| `test_fold_fit_and_refit_cutoffs_match_used_labels` | Misleading training provenance |
| `test_schema_migration_preserves_old_snapshot` | B06: destructive migration/backdating |
| `test_strict_backend_failure_prevents_publication` | B07: silent fixed-value rescue |
| `test_backend_bundle_must_match_request` | B08: false bundle attribution |
| `test_missing_player_or_required_target_fails_completeness` | Partial output treated as successful |
| `test_zero_mass_and_quantiles_match_persisted_pmf` | B09: multiple incompatible distributions |
| `test_incremental_range_crosses_2026_october_boundary` | B10: missing new season |
| `test_identical_forecast_retry_returns_original_payload` | B11: timestamp-only ledger conflicts |
| `test_date_query_on_service_output` | B12: missing `GAME_DATE` integration |
| `test_rotation_caps_and_infeasible_roster_handling` | B13: impossible minutes or random slate failure |
| `test_coverage_includes_quarantined_and_missing_games` | B14: survivor-only completeness |
| `test_undated_status_rejected_with_cutoff` | B15: undocumented availability leak |
| `test_canonicalizer_consumes_official_optional_sources` | B16: silently empty tables |
| `test_canonical_content_mutation_rejected` | B17: undetected canonical tampering/corruption |
| `test_distinct_horizons_keep_distinct_exports` | B18: overwritten run evidence |
| `test_shadow_streak_scoped_to_bundle_and_horizons` | B19: mixed-candidate evidence |
| `test_reconciliation_counts_entire_missing_game` | B20: omitted denominator |
| `test_calibrator_small_sample_custom_methods_and_refit` | B21: method selection/stale state |
| `test_pmf_rejects_negative_and_fractional_support` | B22: invalid count probabilities |
| `test_clean_process_performs_actual_v2_forecast` | B23: artifact-only load evidence |
| `test_concurrent_promotion_rollback_preserves_winner` | B23: stale-pointer rollback |
| `test_daily_resume_and_verified_no_game_day` | B24: operational ambiguity |

Keep same-game mutation tests, checksum corruption tests, game-level partition tests, and promotion denial tests already present. Do not weaken assertions to align with an incorrect implementation.

## 10. Local verification results and how to repeat them

Always activate the project's Python 3.12 environment first. These commands are focused checks and do not refresh NBA data.

### 10.1 Prescribed v2 verification — passed

```bash
source venv/bin/activate
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

Observed result: **39 passed in 2.45 seconds**.

### 10.2 Additional targeted verification — passed

```bash
source venv/bin/activate
pytest tests/test_forecasting/test_opportunity_kernel.py \
  tests/test_features/test_scheduled.py \
  tests/test_pipeline/test_forecast_service.py \
  tests/test_contracts/test_forecast_contract.py \
  tests/test_contracts/test_feature_safety.py -q
```

Observed result: **26 passed in 0.98 seconds**.

### 10.3 CLI import/help checks — passed

`--help` exited 0 for [update_data.py](/Users/jaylenbain/Documents/knowing-master/update_data.py), [canonicalize_data.py](/Users/jaylenbain/Documents/knowing-master/canonicalize_data.py), [check_data.py](/Users/jaylenbain/Documents/knowing-master/check_data.py), [train.py](/Users/jaylenbain/Documents/knowing-master/train.py), [replay.py](/Users/jaylenbain/Documents/knowing-master/replay.py), [promote_model.py](/Users/jaylenbain/Documents/knowing-master/promote_model.py), [simulate_season.py](/Users/jaylenbain/Documents/knowing-master/simulate_season.py), [reconcile_predictions.py](/Users/jaylenbain/Documents/knowing-master/reconcile_predictions.py), and [query_prob.py](/Users/jaylenbain/Documents/knowing-master/query_prob.py). This verifies import/parser startup, not data access, training, promotion, or forecasting success.

### 10.4 Offline reproductions — defects confirmed

| Probe | Observed result |
|---|---|
| Equal uncertain-player fixture through real baseline methods + service | Team stat means 120; participation-weighted minutes 120 |
| Two-player allocation | `[120.0, 120.0]` regulation minutes |
| All nonparticipants | `ValueError: Cannot allocate minutes ... no participants` |
| Repeat same forecast request and ledger write | Conflicting immutable forecast payload |
| Date query on service forecast | `Forecast rows do not contain GAME_DATE` |
| Undated `OUT` snapshot with cutoff | Accepted; play probability `[0.0]` |
| Small-sample calibrator with only `platt` configured | `tuple.index(x): x not in tuple` |
| April 10–October 25, 2026 season interval | Only `['2025-26']` |
| PMF supported at -1 and 1.5 | Accepted as declared discrete PMF |
| Coverage report on either local stored source snapshot | Unsupported `source_snapshot_v1` |
| Raw player/team identity resolution | One quarantined game; 28 player rows |

### 10.5 Compact deterministic probes

Run from the project root after activating the environment:

```python
import pandas as pd
from update_data import get_seasons_between_dates
from src.forecasting.availability import (
    ProbabilityCalibrator,
    status_rule_baseline,
)
from src.forecasting.minutes import allocate_team_minutes
from src.query.v2_probability import probability_at_line

print(get_seasons_between_dates("2026-04-10", "2026-10-25"))

players = pd.DataFrame({
    "PLAYER_ID": [1, 2],
    "TEAM_ID": [1, 1],
    "EXPECTED_MINUTES_RAW": [24.0, 24.0],
    "PLAY_PROB": [1.0, 1.0],
})
print(allocate_team_minutes(players)["EXPECTED_MINUTES"].tolist())

print(status_rule_baseline(
    pd.DataFrame({"PLAYER_ID": [1]}),
    pd.DataFrame({"PLAYER_ID": [1], "STATUS": ["OUT"]}),
    cutoff="2026-10-20T13:00:00Z",
)["PLAY_PROB"].tolist())

print(probability_at_line(
    pd.Series({"PMF": {"-1": 0.5, "1.5": 0.5}}),
    0,
))

try:
    ProbabilityCalibrator(methods=("platt",)).fit(
        [0.2, 0.3, 0.4, 0.5],
        [0, 1, 0, 1],
        evidence_kind="heldout",
    )
except ValueError as error:
    print(type(error).__name__, str(error))
```

These are bug demonstrations, not desirable expected outputs for future regression tests. After repair, update the tests to assert the intended semantics described in each finding.

## 11. Opening-night release checklist

### Data and identity

- [ ] 2026–27 schedule is ingested and versioned with trustworthy tips and availability timestamps.
- [ ] The incremental season selection includes 2026–27.
- [ ] Core history is refreshed through the latest safely available completed games and checked for missing finals.
- [ ] The quarantined April 10 game is repaired from source evidence or remains explicitly excluded with denominator accounting.
- [ ] Supported snapshots and canonical products load with verified hashes.
- [ ] Official membership, status, and eligibility exist for the advertised horizons.
- [ ] Rookies, offseason trades, waived/re-signed players, and unknown labels have explicit handling.
- [ ] Each feature family has adequate point-in-time coverage or is disabled.

### Forecasting and modeling

- [ ] All supported runtime paths use v2 components; unsupported backends and prediction failures cannot produce official heuristics.
- [ ] Participation/minutes/stat definitions match across every stage.
- [ ] Regulation rotations are feasible and exactly conserve 240 team minutes.
- [ ] Full-game distributions account for participant zero outcomes and overtime according to the declared scope.
- [ ] Forecast summaries, simulation samples, and query probabilities come from the same distribution.
- [ ] Candidate predictions beat the required baselines under the frozen policy.
- [ ] Independent calibration and slice evidence pass.

### Evaluation and release evidence

- [ ] Repeated real replay folds use the live forecast service.
- [ ] Fit/tune/calibration/test partitions and final-refit provenance are correct.
- [ ] Scorecards resolve to immutable row evidence and source tiers.
- [ ] No outer-test outcome influenced model selection or calibration.
- [ ] Seven qualifying shadow slates belong to the exact candidate and horizon scope.
- [ ] A clean-process test executes actual inference, not only artifact validation.
- [ ] Promotion dry-run passes without editing the sealed candidate or thresholds.

### Operations and recovery

- [ ] Identical retries return the original forecast.
- [ ] Date/game/horizon queries select the intended saved request.
- [ ] Exports are immutable and reconciliation includes entirely missing games.
- [ ] No-game days and source failures are distinguishable.
- [ ] Typical and high-volume slate runs meet recorded deadline and resource budgets.
- [ ] Backup restore and champion rollback drills pass.
- [ ] The release working tree and runtime environment are recorded and reproducible.

Every checked item should link to a test result, immutable artifact, or decision record. Until that evidence exists, the item remains incomplete even if a helper function or placeholder file exists.

## 12. Source map and maintenance notes

Primary project references:

- [AGENTS.md](/Users/jaylenbain/Documents/knowing-master/AGENTS.md): v2-only runtime and lightweight verification requirements.
- [plans/model_v2_full_architecture_plan.md](/Users/jaylenbain/Documents/knowing-master/plans/model_v2_full_architecture_plan.md): target architecture, forecast semantics, evaluation protocol, and release gates.
- [plans/model_v2_bug_fix_plan.md](/Users/jaylenbain/Documents/knowing-master/plans/model_v2_bug_fix_plan.md): earlier bug backlog; revalidate each entry against current code before treating it as open.
- [config/model_v2.yaml](/Users/jaylenbain/Documents/knowing-master/config/model_v2.yaml) and [config/replay_v2.yaml](/Users/jaylenbain/Documents/knowing-master/config/replay_v2.yaml): declared model, fold, source, and promotion settings.
- The exact source files and functions named under B01–B24.

Useful improvements already present should be retained: the official ledger's publication lock, snapshot checksums, exact identity quarantine, numeric/complete-evidence promotion validation, explicit probability interpolation limitations, game-isolated fold utilities, and the baseline's refusal to claim promotion eligibility. Earlier reports of defects in these areas may describe code that has since changed; this audit does not automatically restate them as current failures.

After each repair package, update this report's finding status with the fixing commit and focused regression result. Keep experimental model gains separate from correctness fixes. A reliable 2026–27 release requires both, but correctness and honest evidence come first.

## Implementation follow-up — September 5, 2026

The first deterministic repair increment and remaining acceptance gaps are recorded in
[implementation status](../plans/model_v2_readiness_implementation_status.md).
B10 and B12 have regression coverage; B15, B21, and B22 have partial repairs.
The audit findings above preserve the original observations. The complete
readiness plan and release checklist remain unfinished; no promotion is claimed.
