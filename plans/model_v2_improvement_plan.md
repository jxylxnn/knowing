# Model v2 improvement plan

Prepared September 5, 2026 from the current working tree on `V2-WIP`.
Status: proposed implementation backlog; no model changes or accuracy gains
are claimed by this review.

## Objective and scope

Build a reproducible v2 model that beats point-in-time baselines on unseen
games and produces calibrated player-stat probabilities. Prioritize correct
opportunity modeling and trustworthy evaluation before increasing complexity.

The design source of truth remains
[the full architecture plan](model_v2_full_architecture_plan.md). This document
turns the current implementation into a shorter execution plan, incorporating
[the readiness audit](../docs/model_v2_2026_2027_readiness_deep_dive.md) and
[the implementation status](model_v2_readiness_implementation_status.md).
Existing partial fixes must be retained; the audit's older findings are not
all still open. Its B/W identifiers below provide backlog traceability.

## What the current code establishes

| Observation | Evidence | Why it matters |
|---|---|---|
| The supported trainer builds only a rolling baseline. | `train.py`; `src/training/v2_baseline.py::_component_payloads` | Learned CatBoost/Transformer code elsewhere is not an integrated v2 champion. |
| Candidate parameters use the latest 20 appearances, but both scorecards use lagged rolling-10 predictions over the available frame. | `train_v2_baseline`; `_baseline_scorecard` | Reported errors do not measure the candidate being packaged. |
| Fold boundaries are recorded but component construction receives the entire player-game table. | `train_v2_baseline`; `_cutoffs` | Evaluation artifacts and a final refit need separate, truthful training provenance. |
| Fold partitions expose fit/validation/test; configuration names four chronological roles. | `src/evaluation/folds.py`; `config/model_v2.yaml` | Tuning and calibration need independently identified partitions. |
| Replay currently joins supplied forecasts to actuals and scores them. | `replay.py`; `src/evaluation/replay.py` | This does not establish historical execution through the live path. |
| Minutes are probability-weighted and normalized into `EXPECTED_MINUTES`, then the service multiplies predicted totals by play probability. | `src/forecasting/minutes.py`; `src/forecasting/baseline_backend.py`; `src/pipeline/forecast_service.py` | Conditional and unconditional opportunity semantics are inconsistent. |
| The allocator conserves team minutes but has no individual cap; independent participation draws can be infeasible. | `src/forecasting/minutes.py`; `src/simulation/joint_sampler.py` | Conservation alone does not guarantee a feasible rotation. |
| Simulation rounds clipped normal rate draws; the service builds summaries separately. | `src/simulation/joint_sampler.py`; `src/pipeline/forecast_service.py` | Means, zeros, quantiles, and queried probabilities need one shared distribution. |
| Baseline calibration is explicitly undeclared as fitted, and promotion eligibility is false. | `src/training/v2_baseline.py` | Preserve this honest gate until the missing evidence exists. |

The readiness record also identifies unfinished snapshot, roster/status,
provenance, and operational work. Those are dependencies below, not claims
that this review independently reproduced every prior audit defect.

## Ordered implementation packages

### 1. Establish immutable inputs and a complete eligible-player panel

Scope: W1/W3/W5; B02–B03, B06, B14, B16–B17.

- Inventory the dirty integration state and record a reproducible code version
  before creating release artifacts. Preserve unrelated edits and all existing
  snapshots and sealed bundles.
- Finish the supported snapshot rebuild and canonical transformation identity.
  Verify content hashes on read and bind actual materializer inputs to the
  requested snapshot, source availability times, and scheduled game.
- Capture timestamped official roster/status records prospectively. Historical
  appearance data must not be relabeled as historical roster evidence; current
  captures cannot be backdated to qualify past forecasts.
- Build eligible player-team-game rows including inactive players and active
  DNPs. Retain unknown labels as unknown. Distinguish active status, appearance,
  regulation minutes, overtime, and full-game stat outcomes.
- Report coverage against the expected slate and eligible population, including
  missing and quarantined rows. Audit enabled feature families for actual
  materialization, source coverage, and cutoff safety.

Acceptance: strict candidate requests execute from verified timestamped inputs;
future, mismatched, corrupt, and undated time-sensitive inputs fail explicitly.
Trades and no-appearance players have correct membership and scoring eligibility.
Where official history is unavailable, keep experiments explicitly diagnostic.

### 2. Repair the shared participation, minutes, and distribution contract

Scope: W4; B01, B07–B09, B13. Start offline fixtures while package 1 proceeds.

- Use the existing W4 contract: internal unconditional minutes are E[M]; public
  `EXPECTED_MINUTES` is E[M | appears], with zero as the p=0 sentinel.
  Enforce E[M] = p * E[M | appears] and team unconditional expectation of 240.
- Sample a feasible participant set first, then positive regulation minutes
  capped at 48 per participant with exactly 240 per team draw. Reject impossible
  rosters. Do not silently rescue Bernoulli draws and claim unchanged marginals.
- Model minutes uncertainty in the allocation, then sample count outcomes.
  Separate overtime so regulation constraints remain intact.
- Generate MEAN, ZERO_PROB, quantiles, stored PMFs/samples, and query answers
  from the same full-game law, including participant zero outcomes. Preserve
  minutes/rate dependence when computing expected totals.
- Enforce bundle/request identity and remove silent heuristic substitutions in
  strict execution. Integrate these contracts across service, runner, and query.

Acceptance: unequal participation probabilities, five-player and impossible
rosters, individual caps, participant zeroes, and conditional moments have
meaningful fixtures. Forecast summaries agree with seeded simulation within
predeclared Monte Carlo tolerances. No single-factor allocator patch suffices.

### 3. Make fold training and replay measure the actual candidate

Scope: W6; B04–B05. Requires packages 1–2 for official evidence.

- Assign whole games to fit, tune, calibrate, and outer-test partitions, storing
  input hashes and concrete request membership. Check nonempty eligible windows
  around offseason gaps. Freeze minimum sample-size policy before comparison.
- Fit one candidate per fold using only its fit inputs; select settings within
  tune and fit calibrators only on independent calibration predictions.
- Regenerate every historical request through the shared forecast service using
  its historical cutoff, candidate bundle, and eligible source snapshot.
- Score the actual candidate against rolling-5/10/20, EMA-10, minutes-times-rate,
  and participation/minutes rule baselines on identical eligible requests.
  Select the comparator using development evidence and freeze that policy before
  outer evaluation; report all declared baselines as well.
- Separate diagnostic baseline scores, candidate out-of-fold scores, and final
  refit provenance. Never copy baseline scores into candidate evidence.

Acceptance: changing outer-test labels cannot change fitting, selection, or
calibration. Live/replay parity is demonstrated from persisted requests.
Multiple unseen folds produce immutable candidate-specific scorecards, including
all six targets, horizons, major slices, proper scores, and missingness counts.

### 4. Run small, attributable modeling experiments

Scope: W7. Begin only after package 3 yields an honest complete baseline.
Each experiment changes one component, records all trials, and retains the
simpler model if held-out evidence does not justify the change.

| Order | Experiment | Comparison and decision evidence |
|---|---|---|
| A | Shrink player rates toward supported role/team priors; compare exposure-weighted rates, rolling windows, and recency weighting. | Test whether short appearances and sparse history improve without harming established players; measure per-target error and proper scores. |
| B | Integrate learned active and play-given-active models, beginning with logistic regression and a compact CatBoost challenger. | Beat status and appearance baselines on Brier/log loss; inspect calibration by horizon and status, including unknowns. |
| C | Learn conditional minutes and quantiles from prior workload, role, rest, and timestamped teammate availability. | Beat rolling-10 and role-median minutes; score MAE, pinball loss, coverage, width, and feasible team draws. |
| D | Train compact CatBoost conditional rate components using verified form, schedule, opponent, and role features. | Ablate each family; use exposure-aware objectives/weights selected inside folds and check low-minute/cold-start slices. |
| E | Compare count distributions and independent calibration per target, including overdispersion and zero behavior. | Use CRPS/discrete proper scores, quantile loss, interval width, zero calibration, and threshold reliability alongside MAE. |
| F | Test shared game factors or out-of-fold residual dependence after marginals pass. | Validate within-player combinations and cross-player covariance; do not infer joint quality from marginal accuracy. |

These are hypotheses, not promised gains. Learnable components must be loadable
from sealed v2 bundles through explicit backend selection. Keep Transformer,
complex ensembles, and possession reconstruction behind these experiments.
Use one worker for any GPU CatBoost run.

### 5. Qualify the exact candidate for release

Scope: W8/W9; B11, B18–B20, B23–B24, plus remaining B15/B21/B22 evidence binding.

- Make retries return the original persisted forecast; prevent export overwrite.
  Reconcile the expected ledger population and bind shadow evidence to candidate,
  horizon, source lineage, and expected slate calendar.
- Freeze the chosen design, refit with truthful cutoffs and independent
  calibration evidence, seal it, and run shadow evaluation on that exact bundle.
- Require clean-process loading, scheduled cutoff execution, immutable outputs,
  operational failure visibility, and a tested rollback to a valid v2 champion.
- Preserve all configured promotion gates: normalized MAE improvement at least
  1%; core-target regression at most 1%, secondary at most 2%; overall 80%/90%
  coverage within 3 percentage points and major slices within 5; fallback rate
  at most 5%, degraded rate at most 10%; major-slice reconciliation at least 95%
  and core reconciliation at least 99.9%.
- Also require participation/minutes baseline wins, live/replay parity, complete
  provenance, and a paired game-bootstrap candidate-minus-baseline MAE interval
  whose upper bound is below zero. Predeclare macro target normalization,
  denominators, slice eligibility, and resampling seed before scoring.

Acceptance: evidence is recomputed from immutable records for the exact sealed
candidate. Insufficient historical or shadow evidence keeps it ineligible;
aggregate degraded/fallback allowances never authorize strict heuristic rescues.

## Immediate next implementation increment

Start with package 2's regression fixtures and shared distribution interface,
then implement the feasible opportunity model across all consumers. Follow with
snapshot binding and the official panel, then actual per-fold candidate training
and replay. This order makes early work testable locally while treating official
point-in-time history as an external dependency, not a coding deadline.

## Verification performed for this review

- Renamed the current local branch from `codex/baseline-audit` to `V2-WIP` and
  verified the resulting branch name.
- Ran the eleven prescribed focused v2 test files from `AGENTS.md` in the
  activated virtual environment: **43 passed in 2.39 seconds**.
- Reviewed training, runtime backend, opportunity components, sampling, service
  summaries, fold construction, replay scoring, and promotion policy.
- Existing passing tests do not establish forecast accuracy or cover all gaps
  above. No full suite, data refresh, training, promotion, or production forecast
  was performed. No remote branch was renamed or pushed.
