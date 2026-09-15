# Improvement plan execution status

## September 7, 2026 status — current

The plan is **not fully complete or release-qualified**. The shared runtime,
immutable input workflow, and diagnostic fold campaign are implemented and
exerciseable. The required historical eligible-player evidence is unavailable
in the supplied archives. Packages 4 and the final release run must follow an
honest complete baseline; they have not been run or claimed successful.

| Package | Implemented and exercised | Acceptance still outstanding |
|---|---|---|
| 1. Inputs and eligible panel | Verified snapshot reads; explicit v1 archive import into a new v2 snapshot; transformation and file hashes; native official schedule/roster captures; cutoff-bound membership with current-day official freshness; eligible-panel left joins retaining unknowns; source coverage and quarantine reporting. | Historical timestamped roster/status and complete inactive/DNP outcomes; enabled feature-family materialization/coverage audit. Current roster captures cannot qualify past requests. |
| 2. Shared opportunity law | Feasible dependent participation; positive capped regulation allocation; optional shared overtime periods; gamma-Poisson counts; optional minutes/rate dependence; one sampled law for service, runner, summaries, and queries. | Fitted and held-out validated overtime, minutes/rate, and cross-player dependence. Diagnostic sampler assumptions are not basketball validation. |
| 3. Actual candidate evaluation | Concrete whole-game fit/tune/calibrate/outer memberships; fit-only sealed candidates; rolling-window tune selection; independent mean calibration; same-path request replay; all five stat baselines plus opportunity baselines; target/slice proper scores; frozen normalization and paired game bootstrap; immutable campaign records. Two tiny synthetic folds exercise the wiring. | Real multi-fold evidence on complete official eligible populations. Mean calibration alone does not establish interval or zero calibration. |
| 4. Controlled modeling experiments | Registered window, exposure-weighted rate, and shrinkage settings; non-window campaign settings require completed checksummed baseline evidence. | Experiments A–F, including learned logistic/CatBoost opportunity/minutes/rates, recency/role-prior comparisons, distribution calibration and residual dependence. These are deliberately gated on package 3 evidence. |
| 5. Exact release | Original forecast retry recovery, immutable paired exports, ledger population correction, bundle-specific shadow request/digest/calendar checks, v2-only promotion and rollback loading, source-tree fingerprints, 99.9% core reconciliation gate. | Exact frozen refit, seven real shadow slates, complete immutable release evidence binding and end-to-end qualification. The separate stronger shadow verifier is not yet the final promotion evidence attachment workflow; the old embedded shadow count is not sufficient evidence. |

### Real data execution

All captures retain original response bytes, hashes, and actual receipt times.
No capture was backdated and no existing snapshot or sealed bundle was deleted.

- Current 2026–27 schedule: `20260907T140259Z_685a65c34250`, 1,266 resolved
  scheduled games, October 3, 2026 through April 11, 2027.
- Current rosters: `20260907T140446Z_624430450bbc`, 587 players across 30 teams.
- Historical outcome-reconciliation schedules: 2024–25
  `20260907T141702Z_107268812df0`; 2025–26
  `20260907T141728Z_d3f8282b2dd2`. These are current captures of historical
  schedules, not historical forecast availability evidence.
- Latest rebuilt source/canonical snapshot: **`20260907T141914Z_223db8d81aea`**.
  Its base is `20260907T140919Z_73e1ed38b17a`, an explicit verified forward
  import of `20260827T130226Z_c8e31fbaf7c3` plus prospective captures.
- Canonical outputs: **4,901 valid games, 9,802 team-game rows, 104,850 player
  rows, zero invalid game records, 28 quarantined player rows**. The ten
  neutral-site home assignments were resolved only through matching official
  game ID, date, and both teams. Original IDs and schedule provenance remain
  in the team rows.
- The 28 quarantined rows concern game `22501183`, Clippers–Trail Blazers on
  April 10, 2026, absent from the team logs. It remains in the denominator:
  **4,901 / 4,902 = 99.9796% core reconciliation**. Passing this threshold does
  not establish historical eligible-population coverage.
- Historical eligibility remains appearance-derived. Status, lineup and odds
  tables are absent. Bios lack required availability/source provenance and
  are disabled. No official historical replay or real modeling campaign can
  be qualified from these inputs.

### Verification scope

Only small focused fixtures and CLI import checks were used. No full suite,
GPU run, large training benchmark, promotion, or production forecast ran.
The two-fold integration fixture uses two draws per request and one declared
window; it is a wiring test, not accuracy evidence. Current tests and command
results are also recorded in the execution guide. Prior dated entries below
are historical and do not describe the current implementation.

Final small verification batches: canonicalization plus the two-fold campaign,
**4 passed in 10.81s**; snapshot/eligible-panel/shadow/fit-replay fixtures,
**10 passed in 4.31s**; final canonical/shadow changes, **6 passed in 1.03s**.
These batches overlap and are not a unique-test total. Seven CLI help/import
checks passed. The real coverage command exited successfully.

Runtime/configuration source hashes and remaining qualification flags are
recorded in `data/execution_checkpoints/e644887feacbf31355e07f006fd97922ba40dc89ee7d372dc8bacac9b44a0af5.json`.
This fingerprint includes the dirty integration state; it does not turn it
into clean release provenance.

### Unblocking the remaining plan

1. Supply authentic timestamped historical roster/status archives and explicit
   inactive/DNP/active outcome labels, with regulation/overtime labels where
   known. Rebuild new snapshots and concrete cutoff requests; do not relabel
   present-day captures as historical evidence.
2. Run the predeclared complete baseline campaign, then attributable A–F
   experiments and feature-family ablations. Retain the simpler candidate if
   gains and calibration do not meet the frozen policy.
3. Finish release evidence attachment/recomputation around an exact sealed
   refit, then collect seven complete actual shadow slates and exercise clean
   process execution and valid-champion rollback. Today’s future schedule is
   not shadow evidence. No promotion is authorized by the diagnostic results.

See [the execution guide](../docs/model_v2_execution_guide.md) for the implemented
commands and the distinction between diagnostic execution and qualification.

## September 5, 2026 increment

The full improvement plan remains incomplete. This increment implements
offline package 2 foundations; it is not release or accuracy evidence.

- Conditional/public and unconditional/internal minutes are separate.
- Capped allocation rejects insufficient expected roster capacity.
- Dependent rounding preserves input participation marginals and samples
  floor/ceil(sum(p)) participants; it deliberately introduces dependence.
- Positive uncertain allocation weights produce capped 240-minute draws.
- Gamma-Poisson counts include participant zero outcomes.
- V2 service summaries and query samples share a seeded empirical law.
  Published opportunity probabilities are empirical; input probabilities
  are retained separately. Active status is sampled conditional on the
  feasible appearance draw, preserving its input marginal.
- Runner and service use identical inputs, seed, and draw count.
- Bundle mismatch fails. Regulation-only samples cannot be official.

## Remaining acceptance work

Package 2 still needs an explicit overtime component, fitted minutes/rate
dependence, and independent Monte Carlo validation of population moments.
The deterministic capped allocation is a point projection, not a claim
that its individual moments equal the nonlinear sampled allocation.
The service publishes empirical sampled moments for consistency.
The fixed-size participation dependence is a diagnostic assumption,
not validated basketball covariance. The runner currently regenerates
identical samples for summaries; a bound sample artifact could avoid
that duplicate work. No immutable release artifact was created.

Packages 1 and 3–5 remain outstanding: verified snapshot/materializer
binding and eligible official panels; actual candidate fold replay;
controlled learned-component experiments; exact-bundle shadow/release
qualification. Official historical data availability is an external
dependency. Existing promotion gates remain unchanged.

## Verification and integration provenance

The eleven prescribed V2 files plus opportunity and service checks:
**51 passed**. After final validation hardening, the directly affected
tests passed again (**7 passed**) and modified runtime modules compiled.
No full suite, API refresh, training run, promotion, or data deletion.

Base commit: `97ac1f9c5e276b8d424fffd4dab25d682e226b4f`. The pre-existing worktree is dirty.
These SHA-256 hashes identify only this increment’s resulting files;
they do not identify the entire integration tree or a reproducible release.

- `src/forecasting/minutes.py`: `142ea6c24058380493c602505309652e860aa34f74264b6496672ad3037a2086`
- `src/simulation/joint_sampler.py`: `1540745e47b6f60e43d5c3a5340e34bd8c59b673a6a80f91ec56a178090896c0`
- `src/pipeline/forecast_service.py`: `7ab55cf0efdb16e8bb4ed1714d7fcaec72d1a0118f7329d9cd490ae5e55a75d6`
- `src/simulation/v2_runner.py`: `d7bd7f650e7074e0a7a290355c5e6ea0443aa7ec01bc8936c009c194677c28c1`
- `tests/test_forecasting/test_feasible_opportunity.py`: `ae3070fd5622611388a9f505d6496840b23356c73efedd3206cf2e082b9e3302`
- `tests/test_training/test_v2_baseline.py`: `34557f4167b56ce5e1dc857853fe61ce5b22780cdfa92c0a156a1ce0bb0b9ce8`

## September 6, 2026 increment

Continued package 1 input binding, training-evidence corrections from package 3,
and immutable exports from package 5. The complete plan is still unfinished.

- The simulation CLI reads the named snapshot's captured schedule. It does not
  fetch a fresh schedule or read mutable working history. An optional schedule
  file must match the captured schedule. The runner verifies the requested
  teams, date, tip, schedule version, and snapshot availability, then loads
  previous-game history from verified snapshot bytes.
- Scheduled rows copy roster identities only, preventing carried-over outcome
  columns from becoming forecast context. Official roster ingestion requires
  explicit timestamped membership and official coverage, retains players with
  no appearances, checks date intervals, and rejects overlapping membership.
  This is an ingestion contract, not a new official feed or a completed
  inactive/DNP labeling panel. The regulation-only baseline still cannot
  produce official full-game forecasts.
- New canonical manifests identify the source manifest, transformation version
  and code digest, and every Parquet file's checksum. Reads verify manifest
  identity and parse the same bytes whose checksum was checked. Old canonical
  outputs without these hashes fail explicitly; none were deleted or rebuilt.
  A fresh snapshot/canonical build is required for an old unverifiable artifact.
- A named missing training snapshot now fails instead of silently creating one
  from current files. Candidate scorecards explicitly say not evaluated;
  diagnostic rolling baseline scores are separate. Final-refit provenance
  records the actual full training range, row count, canonical content identity,
  and absence of selection/calibration/outer-test evidence. No fit-boundary
  claim is borrowed from a diagnostic fold definition. Promotion stays disabled.
- Forecast/sample exports are staged as a pair in content-addressed directories.
  Identical payloads reuse an intact export; changed payloads get a different
  directory, and corrupt existing payloads fail. This does not yet implement
  request-level ledger retry recovery, because regenerated timestamps can change
  an export's content identity.

Verification was deliberately small: eight input/materialization/canonical
checks, one synthetic baseline-bundle test, and one immutable-export test all
passed. After final read/CLI hardening and a missing-snapshot regression, four
canonical/training tests passed in 0.60 seconds. Modified runtime modules passed
Python compilation, and the simulation CLI help loaded. No full suite, source
refresh, real-data training, GPU work, model promotion, or snapshot deletion.

Still required: complete official eligible-player/outcome panels and capture
coverage; overtime and validated distribution dependence; four-role fitting and
candidate-specific historical replay; controlled experiments; exact-candidate
shadow qualification and request-level retry/rollback evidence. No measured
accuracy gain or release eligibility is claimed. Earlier file hashes above
refer only to the September 5 state and are superseded for files changed here.
