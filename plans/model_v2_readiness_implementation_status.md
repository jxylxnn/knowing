# Readiness implementation status

September 5, 2026. Integration base: `97ac1f9c5e276b8d424fffd4dab25d682e226b4f`.
The base commit does not describe the complete integration state: the workspace
already contained extensive modified and untracked v2 work. No existing changes
were discarded, staged, or committed. The readiness audit remains the backlog;
this record does not claim a release or a completed readiness plan.

## First repair increment

- B10: select every season year intersecting the October-based date interval;
  reject reversed intervals and handle leap-day starts without date replacement.
- B12: service output carries the request's game date; the query integration
  test selects that date directly.
- B15 (partial): external undated status tables now fail when a cutoff is
  supplied. Null statuses normalize to UNKNOWN. Context-specific status keys
  and validation of pre-materialized candidate lineage remain outstanding.
- B21 (partial): each fit resets fitted state; implicit identity has a defined
  tie order for insufficient or unusable evidence. Metadata records row count
  and insufficient-evidence status. Binding evidence to immutable partitions
  remains outstanding.
- B22 (partial): PMFs and samples require nonnegative integer support; duplicate
  numeric PMF support is rejected. Summary consistency and calibration evidence
  binding remain outstanding.
- B11 remains open: returning the original persisted payload must happen at the
  request execution boundary. Do not substitute the cutoff for generation time
  or ignore conflicting numerical predictions to make retries appear valid.

Verification: 18 targeted regression/opportunity/query/service checks passed;
43 prescribed focused v2 checks passed. These selections overlap and must not
be added together as a unique-test count. No full suite, live source refresh,
training, promotion, or production forecast was performed.

## Shared opportunity/distribution contract for W4

This specifies implementation requirements, not behavior already implemented.

For each team draw, sample a feasible participant set before allocating minutes.
Each draw must have at least five participants, with each participant receiving
positive regulation minutes no greater than 48 and the team receiving exactly
240. Reject impossible eligible rosters explicitly. Independent Bernoulli draws
followed by arbitrary rescue or resampling change participation marginals;
any constrained participation model must expose its actual resulting marginals.

Internal `UNCONDITIONAL_MINUTES` is E[M], including nonappearance zeroes.
Internal `CONDITIONAL_MINUTES` is E[M | appears]. Public `EXPECTED_MINUTES`
is the latter. For positive p, E[M] = p * E[M | appears]; for p = 0, use zero
as the documented public sentinel. Team sum of unconditional expectation is
240. Quantiles for public minutes use the conditional distribution.

Sample count rates after the common participation and minutes draw. Persist
full-game distributions including both nonappearance and participant zeroes.
Stat MEAN, ZERO_PROB, quantiles, queries, and simulation must derive from this
same declared law. If rates correlate with minutes, compute E[M*R] jointly;
a product of marginal means is not generally valid. Overtime requires a
separately declared component and cannot alter regulation conservation.

Acceptance includes unequal-probability fixtures, infeasible rosters, empirical
conditional moments, exact per-draw conservation and a preregistered Monte
Carlo tolerance. This contract deliberately does not authorize a single-factor
patch to the current allocator.

## Snapshot rebuild decision for W3

Preserve unsupported v1 snapshots unchanged. Build a new supported snapshot
with actual capture time and verified checksums; do not relabel a v1 manifest
or backdate a copy of current CSVs. Retain source ancestry and the old snapshot
identity in a migration record. A newly captured historical file establishes
availability now, not at the historical game cutoff. It cannot qualify as
native historical point-in-time evidence without independent publication
records. Existing forecasts continue to refer to their original identities.

The rebuild utility, canonical transformation identity, native roster/status
capture, input binding, and official evidence remain to be implemented. W3–W9
are not complete. All current promotion gates stay unchanged.
