# Model v2 input and evaluation execution

Activate Python 3.12 before commands:

```bash
source venv/bin/activate
```

## Capture and rebuild

Native official capture archives the response and receipt time. Schedule capture
uses the NBA's current public schedule JSON and retries bounded transient download
failures; roster capture uses bounded retries per team with throttling. Do not
repeat a capture just to inspect an existing archive.

```bash
python capture_official.py --native schedule --season 2026-27
python capture_official.py --native rosters --season 2026-27
python rebuild_snapshot.py --base-snapshot-id BASE_ID --capture CAPTURE_DIRECTORY
python canonicalize_data.py --snapshot-id NEW_ID
python check_data.py --snapshot-id NEW_ID
```

Use `--import-v1` only for an explicit forward import of a verified old source
archive. Runtime v2 readers still reject v1; this is not a compatibility fallback.
Rebuild records the import now and preserves the old manifest identity. A second
`--capture` may supply a different official table. Historical schedules use
`--historical-schedule CAPTURE_DIRECTORY`; they are retained separately from the
prospective runtime schedule and may reconcile exact historical venue identities.

Latest successfully rebuilt and canonicalized snapshot:
`20260907T141914Z_223db8d81aea`. `check_data.py` reports the union of player-log and
team-log games, adjusted for explicit resolved aliases, so quarantine cannot
erase an expected game. Passing core coverage does not mean official eligibility
or model readiness. Canonical directories are immutable; use a new snapshot
identity after further source corrections.

## Eligible populations and historical requests

Request manifests are written/read with
`src.evaluation.request_replay.write_requests` / `read_requests`. Each record
binds a `ForecastRequest`, its content-derived request ID, seed and draw count.
The request names the exact scheduled game, both teams, tip, horizon, cutoff,
source snapshot, model bundle and scenario. Native captures from September 2026
cannot be used for a cutoff in 2025.

The outcome snapshot must contain `player_game_eligibility.csv`, with explicit
`GAME_ID`, `TEAM_ID`, `PLAYER_ID` and known outcome labels. Unknown active or
appearance labels remain missing; a missing box-score row is not a DNP label.
Full-game minutes do not establish a regulation/overtime split.

```bash
python build_eligible_panels.py --requests REQUESTS_JSON --actuals-snapshot-id ACTUALS_ID
python evaluate_candidates.py --requests REQUESTS_JSON --actuals-snapshot-id ACTUALS_ID
```

The supplied real archives currently lack this complete historical population.
The evaluator fails if the chronological windows or required labels are absent;
do not reduce production minimums to make the data pass. Default evaluation
requires two nonoverlapping outer folds, 60 fit games and 20 games in each later
role, with normalization and resampling policy frozen before selection.

The baseline campaign is explicitly diagnostic and promotion-ineligible. It
records every rolling-5/10/20 trial, chooses on tune only, fits a mean scale on
independent calibration predictions, and scores the calibrated candidate on
outer games against all five declared point baselines. Availability and minutes
comparisons, empirical proper scores and missingness are retained separately.
Component experiments require a complete checksummed baseline campaign.

## Replay, retry, and release boundaries

```bash
python replay.py --requests REQUESTS_JSON --candidate SEALED_BUNDLE_DIR --actuals ACTUALS_CSV --json
```

Request replay regenerates the law through `execute_request`, the same entry
point used by the runner. `--predictions` remains scoring-only and cannot prove
historical execution. The campaign reads selection/calibration labels from the
earlier role-boundary snapshots, independently of final outer scoring labels.

`ForecastRunStore.get_or_create` returns the originally persisted request result,
including timestamps and samples. Changed or corrupt artifacts fail integrity
checks. Export destinations are content-addressed and never overwrite a prior
different forecast. `ShadowSlateStore.promotion_evidence` requires the exact
bundle, expected calendar, horizons and verified persisted requests; unbound
counts do not qualify a candidate.

No trained baseline here supports official release. Promotion still requires
all configured evidence and clean provenance. The final release attachment and
recomputation workflow and real shadow execution remain outstanding; do not
mistake a synthetic passing bundle fixture or an embedded success count for
qualification. See the [current execution status](../plans/model_v2_improvement_execution_status.md).

## Lightweight verification

The focused synthetic fold test uses two draws/request, two folds and one trial.
It verifies wiring without a real model benchmark. Run only affected files;
the owner has asked not to run a full or heavy test suite.

```bash
pytest tests/test_data/test_canonicalize.py tests/test_features/test_snapshot_inputs.py -q
pytest tests/test_training/test_fold_candidate.py tests/test_training/test_fold_campaign.py -q
pytest tests/test_forecasting/test_feasible_opportunity.py tests/test_operations/test_shadow.py -q
```
