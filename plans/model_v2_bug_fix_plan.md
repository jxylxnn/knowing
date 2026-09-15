# Model v2 bug review and fix plan

Review date: 2026-09-04 (America/New_York).

Scope: focused review of the current working-tree Model v2 promotion,
forecast contracts, probability queries, replay, ledger, and simulation code.
This is not an exhaustive repository audit. Existing tracked and untracked
changes were treated as the current implementation and left untouched.
Only this plan was added; the fixes below are not implemented.

The design authority is `plans/model_v2_full_architecture_plan.md`, especially
sections 9.7 and the release requirements for immutable evidence and explicit
push probabilities. Preserve strict point-in-time roster requirements, clean
bundle promotion, and the lightweight baseline's promotion-ineligible status.

## Findings and priority

P1 means release integrity or stored evidence is at risk. P2 means incorrect
results or insufficient validation should be corrected before relying on the
affected path.

| ID | Priority | Confirmed bug | Main location |
| --- | --- | --- | --- |
| B1 | P1 | Promotion accepts bundles without mandatory performance evidence | `src/models/bundle.py:280` |
| B2 | P1 | Complete-evidence evaluation accepts NaN and empty evidence | `src/evaluation/promotion.py:70` |
| B3 | P1 | Concurrent conflicting ledger writes both succeed | `src/operations/ledger.py:22` |
| B4 | P2 | Different actual revisions can collapse into one reconciliation | `src/operations/ledger.py:66` |
| B5 | P2 | Integer-line queries count pushes as unders | `src/query/v2_probability.py:9` |
| B6 | P2 | Forecast validation accepts infinite means and quantiles | `src/contracts/forecast.py:165` |

## B1 — Enforce performance evidence at the actual promotion boundary

**Evidence:** `validate_v2_bundle(..., promotion=True)` requires four named
checks, seven shadow successes, at least two fold entries, and nonempty
scorecards. It never invokes `evaluate_v2_promotion` or enforces the aggregate,
per-target, bootstrap, participation, minutes, and operational gates. The
existing `tests/test_models/test_v2_bundle.py::_bundle` fixture is accepted as
promotable with only those four checks, fold IDs without evaluation details,
and a slice scorecard containing an empty `slices` mapping. This was reproduced
in a temporary directory. `ModelVersionRegistry.promote` uses this validator,
so the omission reaches the champion promotion boundary.

**Impact:** a sealed, internally consistent bundle can be accepted without
evidence that it meets the architecture's mandatory release thresholds.
Checksums establish integrity, not the sufficiency of the evidence.

**Fix plan:**

1. Define a versioned evidence schema covering required targets, fold windows,
   baseline comparisons, calibration, major slices, parity, and operations.
2. At promotion, validate those artifacts and evaluate the complete policy from
   their recorded metrics. Require the stored decision to agree with the
   recomputed decision. Retain clean-tree, shadow, and checksum gates.
3. Replace the permissive positive fixture with complete valid evidence.

**Regression tests:** omit each required evidence field in turn; submit a
regressing candidate with `eligible: true`; use placeholder fold records and
empty slices. All must fail before changing the champion pointer. A complete
passing fixture must still promote and support rollback.

## B2 — Reject invalid numbers and incomplete evidence mappings

**Evidence:** the complete-evidence evaluator returned `eligible=True` with
`target_improvements={"PTS": NaN}`, `contract_flags={}`, and NaN values in all
three major-slice mappings, while the other scalar gates passed. Comparisons
such as `value < minimum` and `abs(value - expected) > tolerance` do not reject
NaN. Empty contract flags also pass, and required target coverage is not
checked.

**Impact:** missing or numerically invalid evaluation results can count as
successful release evidence, including after B1 connects the evaluator.

**Fix plan:** require finite numeric metrics and valid domains; require actual
booleans for contract results; validate required target, contract, and slice
keys against the versioned policy. Reject empty mandatory mappings and
inconsistent slice coverage across scorecards with explicit reasons.

**Regression tests:** NaN, positive/negative infinity, empty mappings, missing
core targets, missing required contract checks, and reconciliation rates
outside `[0, 1]` must fail complete-evidence evaluation. Keep legitimate
boundary values covered.

## B3 — Make official forecast publication create-once under concurrency

**Evidence:** `write_forecast` checks whether a file exists, then
`_atomic_parquet` publishes with `os.replace`. Two writers can both observe no
file and overwrite one another. A temporary-directory reproduction synchronized
two writers immediately before publication, using the same request ID and
different means. Both calls succeeded and the retained mean was 21.0; one
payload silently replaced the other.

**Impact:** parallel jobs or retries can mutate an allegedly immutable official
forecast without reporting a contradiction.

**Fix plan:** use an exclusive publication primitive or cross-process lock
covering the existence check, comparison, and publication. If another writer
wins, read and compare its payload: accept identical content and reject a
conflicting payload. Retain atomic visibility and temporary-file cleanup.
Inspect the analogous check-then-replace helpers in bundle storage for the same
pattern, without rewriting existing sealed artifacts.

**Regression tests:** coordinate two independent writers with a barrier.
Identical writes must converge successfully; conflicting writes must produce
one success and one explicit conflict, with the first published bytes intact.
Also test interrupted publication and cleanup.

## B4 — Include actual-data identity in reconciliation revisions

**Evidence:** revision IDs hash only the date, aggregate replay score, and
forecast request IDs. Against the existing forecast fixture with mean 20,
actual PTS values of 19 and 21 produce the same MAE, RMSE, and interval coverage.
Reconciling these two different actuals returned exactly the same JSON path.
Neither the actual values nor an actual-source identity are stored.

**Impact:** corrected box scores can disappear from the audit trail when their
summary metrics happen to match; the scored actuals cannot be reconstructed
from the reconciliation record.

**Fix plan:** persist or reference immutable normalized actual rows, include
their content digest and the forecast payload identities in the reconciliation
payload, and derive the revision from that full provenance. Preserve prior
records and use a new schema version for richer records.

**Regression tests:** the 19-versus-21 example must create different revisions;
identical normalized actuals must remain idempotent; row reordering must not
create spurious revisions; each record must resolve to its exact scored inputs.

## B5 — Report push probability explicitly for integer lines

**Evidence:** for `ZERO_PROB=0.8`, zero P10/P25/P50/P75, and P90=2, querying
line 0 returns `under=0.8` and `over=0.2`, with no push field. For a nonnegative
count stat, the correct values are under=0, push=0.8, over=0.2. The current
function labels `P(X <= line)` as under. The architecture explicitly requires
`P(over) + P(push) + P(under) = 1` for integer lines.

**Fix plan:** define under as `P(X < line)`, push as `P(X = line)`, and over as
`P(X > line)`. Expose all three in `query_prob.py`. Use a saved calibrated
discrete distribution or declared sample representation for mass estimates;
the five quantiles alone cannot recover exact mass at arbitrary integer lines.
For artifacts without sufficient distribution evidence, report that limitation
explicitly instead of presenting interpolated mass as exact. Preserve exact
known zero mass.

**Regression tests:** zero line with a participation point mass; a known PMF at
an integer line; half-integer and negative lines; total probability equals one;
and artifacts with insufficient distribution evidence. Update the current
two-outcome test, which incorrectly requires over plus under to equal one at
every line.

## B6 — Require finite forecast outputs before storage or scoring

**Evidence:** replacing the existing forecast fixture's `MEAN` with positive
infinity still passes `validate_forecast_frame`. Numeric validation only checks
nulls and negativity. Ordered infinite quantiles can likewise pass comparisons.

**Impact:** invalid model outputs can enter the official ledger or produce
infinite replay metrics and nonstandard JSON numeric values.

**Fix plan:** require finiteness for every numeric forecast field after numeric
conversion, with errors identifying the affected column. Reuse this boundary
before ledger writes and replay; validate finite query lines as well.

**Regression tests:** positive and negative infinity and NaN in means, minute
summaries, and stat quantiles must fail. Finite valid forecasts must continue
to pass. Confirm rejected forecasts create no ledger file.

## Implementation order and verification

1. Fix B2 and B6 first to establish trustworthy numeric/evidence validation.
2. Fix B1 using the stricter evaluator; keep existing baseline candidates
   ineligible rather than fabricating evidence.
3. Fix B3 and B4 to protect immutable forecast and actual histories.
4. Fix B5 with the necessary distribution contract and CLI changes together.

For each fix, first add a failing focused regression test, implement the change,
and rerun the affected tests. Run the focused V2 command from `AGENTS.md` after
integration, plus the relevant forecast-contract tests and CLI import checks.
Do not run the full suite without the owner's explicit request. No data refresh,
training, production promotion, or deletion of snapshots/bundles is needed.

**Review validation completed:** the 11-file focused V2 selection prescribed in
`AGENTS.md` passed: **18 passed in 2.35 seconds**. Additional isolated Python
reproductions confirmed all six findings above; storage reproductions used
temporary directories and promotion validation did not change a real champion.
The passing tests demonstrate current coverage gaps, not that these bugs are
fixed. No full test suite was run.
