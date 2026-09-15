# Model v2 CatBoost Challenger — Future Execution Plan

**Status:** Ready-to-execute specification. No implementation is authorized by this
document alone; each phase must earn its exit gate before the next begins.

**Prepared:** 2026-09-14

**Scope:** Port the strongest legacy ("OG") CatBoost feature families and modeling
choices into a **Model v2 challenger** that produces evidence through the
existing v2 contracts, then compare it to the rolling baseline through official
multi-fold replay.

**Design source of truth:** `plans/model_v2_full_architecture_plan.md`.
This document refines Phase 4/5/8 of that plan for one specific challenger. Where
the two disagree, the architecture plan wins.

**One-sentence decision:** keep the v2 prediction kernel, evidence bundle, and
promotion gates exactly as they are, and let a CatBoost challenger earn its way in
as one more backend behind `ForecastService` — never as a parallel runtime.

---

## 0. How to use this document

1. Read Section 2 first. It records what already exists so an agent does not
   rebuild working machinery.
2. Audit every prerequisite in Section 3 before implementation. Phases B and C
   may proceed with recorded data gaps because they are contract-only work, but
   do not fit a CatBoost candidate in Phase D or run a campaign in Phase E until
   the gates assigned to those phases pass.
3. Execute phases in order. Each phase lists exact files, interfaces, and an exit
   gate.
4. Treat Section 11 as the complete test obligation. Do not run the full suite on
   this machine unless the owner explicitly asks (`AGENTS.md`).
5. Section 15 is the checklist to tick as work lands.

---

## 1. Objective and non-goals

### 1.1 Objective

Produce a sealed, immutable Model v2 **candidate bundle** whose predictor
components are CatBoost models instead of the rolling baseline payloads, where:

- Participation, minutes, and conditional stat rates each have a learned,
  calibrated component.
- The challenger is trainable and replayable through the **same** code path as
  the live forecast (`src.pipeline.v2_execution.execute_request` ->
  `ForecastService.predict_game_distribution`).
- Its evidence — fold manifests, out-of-fold predictions, slice scorecards,
  calibration scorecards, paired bootstrap intervals — is produced by the
  existing v2 evaluation primitives, not by a new backtest.
- It is compared, on identical folds and identical eligible populations, against
  the declared rolling baseline bundle and the mandatory simple baselines.
- It may only become champion if it passes the **unmodified** promotion gates in
  `src/evaluation/promotion.py` and `src/models/bundle.py`.

### 1.2 Non-goals

- Re-enabling the retired flat runtime: `ModelManager`, `models/feature_cols.pkl`,
  `.cbm`/`.pkl` flat artifacts, `src/pipeline/prediction_service.py`,
  `src/pipeline/training_pipeline.py`, or the retired simulator. The quarantine
  recorded as F-01 stands permanently for those artifacts.
- Training the challenger on the leaked flat feature schema. The eight
  current-game team outcomes (`PTS_TEAM`, `REB_TEAM`, `AST_TEAM`, `DREB_TEAM`,
  `OREB_TEAM`, `FGA_TEAM`, `FTA_TEAM`, `TOV_TEAM`) documented in
  `plans/reconstruction_baseline_2026_07_14.json` are forbidden as features.
- Weakening, bypassing, reordering, or "temporarily disabling" any promotion gate.
- Making CatBoost the champion by default. This plan produces a **challenger**.
  Promotion is a separate, evidence-gated decision.
- A Transformer/neural component. That is architecture-plan Phase 8 and stays
  behind this work.
- Changing the canonical forecast contract, the horizon set, the 240-minute
  allocation rule, or the participation-before-minutes sampling order.

---

## 2. Current-state evidence

### 2.1 Assets that already exist and must be reused

| Concern | Location | What it gives the challenger |
|---|---|---|
| Immutable candidate/bundle contract | `src/models/bundle.py` (`V2_REQUIRED_FILES`, `V2_REQUIRED_MODEL_DIRS`, `write_v2_support_files`, `finalize_v2_bundle`, `validate_v2_bundle`) | Sealing, checksums, manifest, promotion-evidence recomputation |
| Bundle identity/registry | `src/models/versioning.py` (`ModelVersionRegistry`, `ModelBundleManifest`) | Content-addressed candidate directories and the champion pointer |
| Baseline candidate trainer | `src/training/v2_baseline.py` (`train_v2_baseline`, `_component_payloads`) | The exact four-component payload shape to mirror |
| Fold-scoped fit | `src/training/fold_candidate.py` (`fit_fold_candidate`) | Fit-only sealing from verified fit-game labels, no outer leakage |
| Multi-fold campaign | `src/training/fold_campaign.py` (`run_fold_campaign`, `_require_complete_baseline`) | Four-role folds, trial selection, calibration, outer scoring, bootstrap |
| Chronological policy | `src/evaluation/chronology.py` (`EvaluationPolicy`, `four_role_folds`, `digest_payload`, `validate_four_role_membership`) | Frozen roles `fit/tune/calibrate/outer_test` and their membership rules |
| Rolling folds | `src/evaluation/folds.py` (`rolling_origin_folds`, `partition_frame`, `fold_manifest`) | Game-atomic chronological splits and hash-addressable manifests |
| Mandatory baselines | `src/evaluation/baselines.py`, `src/evaluation/request_baselines.py` (`BASELINES`) | `rolling_5/10/20`, `ema_10`, `minutes_x_rate_10` on the replay population |
| Proper scores | `src/evaluation/distribution_scores.py` (`score_distributions`, `slice_scorecards`) | CRPS, discrete Brier, pinball, coverage, width, normalized MAE |
| Participation scores | `src/evaluation/opportunity_scores.py` (`score_opportunity`) | Active/appeared calibration and missingness denominators |
| Significance | `src/evaluation/significance.py` (`paired_macro_bootstrap`, `paired_game_bootstrap_many`) | Game-clustered intervals with multiplicity handling |
| Request replay | `src/evaluation/request_replay.py` (`request_record`, `replay_requests`) | Identical-path historical execution |
| Shared execution | `src/pipeline/v2_execution.py` (`execute_request`) | The one live/replay entry point |
| Runtime service | `src/pipeline/forecast_service.py` (`ForecastService`) | Backend-agnostic prediction, summary, and distribution |
| Baseline backend | `src/forecasting/baseline_backend.py` (`V2BaselineBackend`) | The backend interface a challenger must satisfy |
| Availability/participation | `src/forecasting/availability.py` (`status_rule_baseline`, `appearance_participation_baseline`, `ProbabilityCalibrator`, `CalibratedBinaryClassifier`, `ParticipationClassifier`) | Rule baselines plus isotonic/Platt calibration already implemented |
| Minutes | `src/forecasting/minutes.py` (`allocate_team_minutes`) | Constrained 240-minute allocation with cap saturation |
| Rates and distribution | `src/forecasting/rates.py`, `src/forecasting/distribution.py` | Rate-to-total propagation and the canonical long output |
| Joint sampler | `src/simulation/joint_sampler.py` (`sample_game`) | Participation -> minutes -> rates sampling already enforced |
| Feature materializer | `src/features/materializer.py` (`FeatureSpec`, `FeatureRegistry`, `FeatureMaterializer`) | Exact-name, point-in-time feature registration |
| Scheduled rows | `src/features/scheduled.py` (`materialize_scheduled_rows`) | Synthetic pregame rows: `PRIOR_{STAT}_{5,10,20}`, `REST_DAYS`, `PRIOR_APPEARANCES`, `PRIOR_TEAM_*`, `PRIOR_OPPONENT_*`, `PRIOR_TEAM_OUT_COUNT`, `STATUS` |
| Snapshot readers | `src/features/snapshot_inputs.py` | Verified bytes, cutoff enforcement, official-roster strictness |
| OG CatBoost trainer | `src/training/catboost_trainer.py` (`CatBoostTrainer`) | Feature to port: per-target profiles, multi-loss blend, quantile models, GPU->CPU fallback, `.cbm` plus joblib metadata |
| OG feature taxonomy | `src/preprocessing/features/registry.py` (`BUILTIN_SAFE_PREFIXES`), `src/preprocessing/features/*.py` | Group names and leak-safe prefixes to classify for migration |
| Feature safety contract | `src/contracts/features.py` (`FEATURE_SCHEMA_VERSION = "feature_schema_v4"`, `FORBIDDEN_PATTERNS`) | The forbidden-name policy the challenger schema must satisfy |
| Config | `config/model_v2.yaml`, `config/default.yaml` (`training_presets`) | Existing v2 keys and OG preset group lists |
| Colab entry | `train_colab.ipynb` | The 13-cell clone/capture/canonicalize/train/replay workflow to extend later |

An existing diagnostic bundle already exists at
`models/versions/preseason-baseline-20260913/` with a `feature_schema.json`
declaring only identity/participation/minutes fields and
`promotion_decision.eligible = false`. Treat it as the reference payload shape.

### 2.2 Constraints already proven in code

1. **Promotion evidence is recomputed, not trusted.** `validate_v2_bundle(...,
   promotion=True)` re-derives `checks` and `reasons` via
   `_validate_and_evaluate_promotion_evidence` and rejects any bundle whose
   stored decision disagrees. A challenger cannot hand-write a passing decision.
2. **Promotion requires two or more folds.** `training_folds.json` must contain
   `>= 2` folds passing `validate_four_role_membership`, each with populated
   `roles` and `evaluation`.
3. **Promotion requires seven consecutive shadow slates.**
   `decision["shadow_slates"]["consecutive_successes"] >= 7`.
4. **Required contract flags are `artifact_contract` and `calibration`**
   (`REQUIRED_CONTRACT_FLAGS`), plus `point_in_time`, `live_replay_parity`,
   `participation.beats_baseline`, and `minutes.beats_baseline` read from the
   candidate scorecard.
5. **The backend interface is already implicit.** `ForecastService` and
   `execute_request` depend on `model_version`, `training_cutoff` /
   `learning_cutoff`, `supports_official`, `targets`, `models`, `load_models`,
   `prepare_contexts`, `prepare_sampling_frame`, `predict_player_stats`, and
   optionally `overtime_probabilities`.
6. **Official forecasts are gated at runtime.** `predict_game_distribution`
   raises unless `backend.supports_official` is true for `scenario="official"`;
   `V2BaselineBackend` sets it `False` deliberately.
7. **Strict official roster requires same-day capture.**
   `load_official_roster` rejects non-official coverage and `SOURCE` containing
   `derived|appearance`.

### 2.3 Gaps the challenger must close (the prerequisites, restated)

- **G1 — No learned participation component.** `_component_payloads` writes
  `availability_model/baseline.json` with `kind:
  "declared_status_and_recent_appearance_baseline"` and
  `training_label_coverage: "appearances_only"`. There is no fitted
  `P_ACTIVE`/`P_PLAY_GIVEN_ACTIVE` model and no negative (DNP) examples.
- **G2 — No learned minutes/stat components.** Minutes are a rolling-20 mean and
  stats are rolling-20 per-minute rates, serialized as JSON dictionaries.
- **G3 — No backend selection by bundle.** `V2BaselineBackend` is hardcoded in
  `src/pipeline/forecast_service.py` (champion load), `src/simulation/v2_runner.py`
  (`run_scheduled_game` candidate branch), `src/evaluation/request_replay.py`
  (`replay_requests`), and `promote_model.py` (`_clean_process_validate`). A
  CatBoost bundle would load through the wrong class today.
- **G4 — No official multi-fold replay evidence for any candidate.**
  `candidate_scorecard.json` is `{"status": "not_evaluated"}` and
  `promotion_decision.checks.official_replay` is `False`.
- **G5 — No OG feature materialization inside v2.** `materialize_scheduled_rows`
  produces only the small `PRIOR_*` set; the OG groups in
  `BUILTIN_SAFE_PREFIXES` are not registered as v2 `FeatureSpec`s.

---

## 3. Prerequisites (phase-specific hard gates)

Audit all prerequisites in Phase A. P1-P6 are hard gates before fitting a
candidate in Phase D or starting the multi-fold campaign in Phase E. The clean
worktree part of P6 must also hold when any candidate is sealed. Backend-factory
and feature-contract work in Phases B-C may proceed against fixtures while a
data gap is documented; it must not be represented as official evidence.

**P1. Official point-in-time roster membership.**
`src/features/snapshot_inputs.py:load_official_roster` must succeed for the games
used in the campaign: `COVERAGE_STATUS == "official"` for every row, `SOURCE`
free of `derived|appearance`, and the roster capture on the same Eastern calendar
day as the forecast cutoff. Evidence: one campaign snapshot whose roster file
passes those checks.

**P2. Official participation labels (negative examples).**
A `player_game_eligibility.csv` snapshot with per-player `ACTIVE` and `APPEARED`
labels covering **both** appearances and DNPs. This is what `run_fold_campaign`
already requires (`required = {"GAME_ID","PLAYER_ID","TEAM_ID","ACTIVE",
"APPEARED","MIN", *TARGETS}`) and what `_require_complete_baseline` validates.
Without it, participation cannot beat the rule baseline and
`participation.beats_baseline` stays false.

**P3. Status snapshots with `AVAILABLE_AT`.**
`player_status_snapshots.csv` with publish timestamps so
`status_rule_baseline(..., cutoff=...)` and `load_request_status` are
point-in-time honest. Unknown status must remain explicitly `UNKNOWN`.

**P4. A completed, checksummed rolling-baseline campaign.**
`_require_complete_baseline(directory)` must pass: complete `checksums.json`,
`summary.json` and `campaign.json` present, fold count `>= policy.min_folds`, and
every fold with complete eligible-population evidence. This is the frozen
diagnostic sanity/provenance contract; a CatBoost experiment may not precede it.
The official-context rolling reference used for the promotion comparison is
generated separately in Phase E from the exact same materialized inputs as the
challenger.

**P5. `EvaluationPolicy` frozen before the challenger is trained.**
Record `policy.policy_id` (a hash of the policy payload) in the challenger
evidence. Thresholds and policy are versioned **before** candidate results are
seen, never after.

**P6. Clean, reproducible environment.**
`catboost` installed in the Python 3.12 venv, `git status --porcelain` clean at
candidate-seal time (promotion rejects dirty worktrees), and `environment.lock`
capturing the CatBoost version.

If a gate assigned to the next phase is unmet, report the specific gap and stop
at that gate. Do not substitute appearance-only history for P2, and do not relax
P1 to make a run pass.

---

## 4. Architecture boundaries

### 4.1 Invariants that must not change

1. `ForecastRequest` stays the sole identity of a forecast
   (`src/contracts/forecast.py`).
2. `CANONICAL_FORECAST_COLUMNS` and `validate_forecast_frame` stay as-is.
3. `execute_request` remains the only live/replay execution entry point.
4. Sampling order stays participation -> minutes -> rates, with each team
   allocated exactly 240 regulation minutes before rates are sampled.
5. `validate_v2_bundle(..., promotion=True)` and `evaluate_v2_promotion` remain
   the only promotion authorities.
6. The `feature_schema_v4` forbidden-name policy is enforced on every persisted
   feature schema (`src/contracts/features.py`).
7. Snapshots, canonical tables, bundles, forecasts, and reconciliation records
   stay immutable and content-addressed.
8. `models/champion.json` (via `ModelVersionRegistry`) stays the only deployment
   pointer.

### 4.2 New extension points this plan introduces

- **E1. Bundle-declared backend selection.** A `model_backend` block in
  `config_resolved.yaml` (for example `{kind: v2_rolling_baseline}` or
  `{kind: v2_catboost_challenger}`) plus a factory that returns the correct
  backend class. All four hardcoded call sites (G3) resolve through the factory.
- **E2. `src/forecasting/catboost_backend.py`.** A backend class implementing the
  interface in Section 6.2, loading per-target CatBoost artifacts from the bundle.
- **E3. `src/training/catboost_candidate.py`.** A fold-scoped CatBoost candidate
  trainer that mirrors `fit_fold_candidate`'s fit-only guarantees while producing
  CatBoost components.
- **E4. Registered v2 feature specs for the migrated OG families.**
  `FeatureSpec` entries in a dedicated module (for example
  `src/features/og_families.py`), registered through `FeatureRegistry`.
- **E5. A distinct official challenger campaign.** Add
  `run_official_challenger_campaign` that reuses chronology, scoring, and
  bootstrap primitives but has explicit Tier A/B request validation, a CatBoost
  trainer hook, fitted distribution calibration, and promotion-evidence
  assembly. Keep the current diagnostic `run_fold_campaign` ineligible; do not
  relabel or reuse its mean-scale calibration as official evidence.

### 4.3 Explicitly forbidden

- Any import of `src.models.model_manager`, `src.pipeline.training_pipeline`,
  `src.pipeline.prediction_service`, `src/evaluation/backtest_runner`, or the
  retired simulator from the v2 challenger path.
- A second inference implementation. If the CatBoost backend cannot be expressed
  behind `ForecastService`, the design is wrong; fix the design.
- Persisting `.pkl`/flat `.cbm` artifacts at the `models/` root. CatBoost
  artifacts live inside `availability_model/`, `minutes_models/`,
  `stat_rate_models/`, and `calibrators/` inside the sealed bundle.
- Using actual (future-known) minutes as a training input for a model that will
  receive predicted minutes at inference, unless the training inputs are
  out-of-fold predicted minutes.

---

## 5. Feature migration classification and leakage review

### 5.1 Classification scheme

Each OG group is assigned exactly one class:

- **PORT** — leak-safe as named, and the v2 materializer can produce it from
  snapshot inputs. Migrate as a registered `FeatureSpec`.
- **PORT-WITH-CHANGE** — valuable, but its OG column shape or computation must
  change to be point-in-time honest (for example it must consume `PRIOR_*` lags
  instead of same-game values).
- **DEFER** — needs a data source that is absent or below coverage (see F-04 in
  the architecture plan: `player_bios.csv`, `injury_history.csv`,
  `advanced_tracking.csv`).
- **DROP** — forbidden by `feature_schema_v4` or demonstrated leaky.

### 5.2 Group classification (from `BUILTIN_SAFE_PREFIXES`)

| OG group | Prefixes / keywords | Class | v2 disposition |
|---|---|---|---|
| `rolling` | `ROLL_`, `EWMA_` | PORT | Highest-value family. Register as prior-game rolling and EMA windows. `materialize_scheduled_rows` already emits lagged means for windows 5/10/20; extend to EMA and longer windows. |
| `recency_form` | `RECENCY_`, `IS_RECENT_` | PORT | Decay-weighted form from prior games only. |
| `minutes_confidence` | `MIN_CONF_` | PORT | Prior-minutes mean/std/start rate; feeds both participation and minutes components. |
| `rest_density` | `DAYS_SINCE_`, `GAMES_WITH_` | PORT | `REST_DAYS` already exists; add games-in-N-days and back-to-back flags from the prior schedule. |
| `team_role` | `ROLE_INDEX` | PORT | Start rate and rotation slot computed from prior games. |
| `season_phase` | `IS_SEASON_`, `SCHED_` | PORT | Calendar/phase flags with no outcome dependency. |
| `pace` | `TEAM_PACE_`, `PACE_FACTOR`, `EST_POSS` | PORT-WITH-CHANGE | Derive from prior completed team games, never the scheduled game's `team_games` row. |
| `matchup` / `opponent_strength` | `VS_OPP_`, `OPP_DEF` | PORT-WITH-CHANGE | Opponent context from prior games only; no same-game defensive outcome joins. |
| `defense_position` | `DEF_POS_`, `DEF_MATCHUP` | PORT-WITH-CHANGE | Historical opponent-by-position defense must be lagged. |
| `teammate_usage` | `TEAMMATE_` | PORT-WITH-CHANGE | Absent-teammate minutes/usage from prior games plus cutoff-known status. `PRIOR_TEAM_OUT_COUNT` already exists; extend it. |
| `lineup_stability` | `LINEUP_` | PORT-WITH-CHANGE | Requires `lineup_snapshots`; degrade visibly when absent. |
| `injury_opportunity` | `INJURY_OPP_` | PORT-WITH-CHANGE | Depends on P3 status snapshots; opportunity from confirmed `OUT` only. |
| `target_encoding` | `_SHARE` | PORT-WITH-CHANGE | Out-of-fold only. Fit inside each fit role; never on tune/calibrate/outer. |
| `league_rank` | `LEAGUE_PCT_`, `TEAM_CUMULATIVE_` | PORT-WITH-CHANGE | Prior-games-only cumulative team stats. |
| `efficiency` | `EFF_Z_SCORE` | DEFER | Needs enough same-season history per player; enable only after the coverage gate passes. |
| `momentum` | keyword family | DEFER | Evaluate only if it survives grouped ablation across folds. |
| `context` | keyword family | DEFER | Mostly schedule/season context; register only what is leak-safe and coverage-backed. |
| `fatigue` | `FATIGUE`, `B2B_IMPACT` | DEFER | Overlaps `rest_density`; consolidate rather than duplicate. |
| `archetype` | `ARCHETYPE_`, `SIMILARITY_TO_` | DEFER | Cluster priors must be fit inside fit folds only. |
| `team_motivation` | `IS_LATE_`, `IS_TANKING_` | DEFER | Requires standings history; not a core six-stat driver. |
| `postseason_context` | `IS_PLAYOFF_`, `PLAYOFF_PACE_` | DEFER | Playoff-only; the regular-season champion does not need it yet. |
| `injury_risk` | `INJURY_RISK_` | DEFER | No real injury history (F-04); would emit neutral defaults. |
| `aging_curve` / `kan_aging` | (none) | DEFER | Needs bio coverage; KAN caches are derived inputs needing invalidation. |
| `skill_development` | `POTENTIAL` | DEFER | Needs bio/experience coverage. |
| `advanced_tracking_features` (extension) | declares its own prefixes | DEFER | `advanced_tracking.csv` is absent (F-04). |

The first migration wave is therefore `rolling`, `recency_form`,
`minutes_confidence`, `rest_density`, `team_role`, and `season_phase` (PORT),
plus `pace`, `matchup`, `opponent_strength`, `defense_position`, and
`teammate_usage` (PORT-WITH-CHANGE). Everything else waits for coverage or
ablation evidence.

### 5.3 Leakage review protocol

For every migrated family, add tests that mirror
`plans/model_v2_full_architecture_plan.md` Section 7.3 and prove:

1. **Same-game target mutation is inert.** Mutating `PTS/REB/AST/STL/BLK/TOV/MIN`
   for the forecast game does not change the feature row. This is the exact
   failure that quarantined F-01.
2. **Future mutation is inert.** Mutating any game after a feature row's cutoff
   does not change that row.
3. **Cutoff enforcement.** Setting a row's `available_at` after
   `forecast_cutoff` removes its contribution
   (`FeatureMaterializer._filter_point_in_time` already implements this and must
   be exercised).
4. **Schema rejection.** `validate_feature_names` rejects every `*_TEAM`
   current-game column and matches `FORBIDDEN_PATTERNS`.
5. **Target encoding is out-of-fold.** Encodings fit in the fit role never touch
   tune/calibrate/outer rows (enforced by `validate_four_role_membership` plus an
   explicit hash check).
6. **Horizon separation.** Morning and pregame differ only through data available
   between their cutoffs.

Any family that fails one of these is DROPPED, not patched around.

---

## 6. Training backend and component artifact design

### 6.1 Bundle component layout (same directories, new payloads)

Keep the four required directories from `V2_REQUIRED_MODEL_DIRS` and populate them
with real artifacts:

```text
availability_model/
  model.json             # backend kind, calibration metadata, feature list, metrics
  active.cbm             # CatBoost classifier for P(active)
  play_given_active.cbm  # CatBoost classifier for P(plays | active)
  calibrators.json       # ProbabilityCalibrator metadata (method, evidence_kind)
minutes_models/
  model.json
  mean.cbm               # conditional minutes
  q10.cbm, q90.cbm       # minutes quantiles
  meta.joblib            # feature_cols, cat_features, config (mirrors CatBoostTrainer.save)
stat_rate_models/
  model.json
  {PTS,REB,AST,STL,BLK,TOV}.cbm        # direct-total or rate models
  {TARGET}_q10.cbm, {TARGET}_q90.cbm   # optional quantiles
calibrators/
  distribution.json      # per-target scale/quantile mapping from held-out residuals
```

Each of `availability_model/`, `minutes_models/`, and `stat_rate_models/` must
contain at least one file (validator requirement), plus the CatBoost artifacts.

### 6.2 Backend contract (`src/forecasting/catboost_backend.py`)

Implement `V2CatBoostBackend` matching what `ForecastService`, `execute_request`,
`replay_requests`, `run_scheduled_game`, and `promote_model.py` already assume:

```python
class V2CatBoostBackend:
    targets: tuple[str, ...]                      # the six canonical targets

    def __init__(self, bundle_dir): ...           # validate_v2_bundle(promotion=False)
                                                  # then load components

    model_version: str                            # manifest.bundle_id
    training_cutoff: str                          # manifest.cutoffs["training"]
    learning_cutoff: str                          # max(training, validation, calibration)
    supports_official: bool                       # True only when official-inference-ready
    overtime_probabilities: tuple[float, ...] | None

    models: dict                                  # truthy so ForecastService._loaded is set
    def load_models(self) -> None: ...

    def prepare_contexts(self, contexts: pd.DataFrame) -> pd.DataFrame:
        """Attach P_ACTIVE, P_PLAY_GIVEN_ACTIVE, PLAY_PROB, EXPECTED_MINUTES_RAW,
        MINUTES_STD, then call allocate_team_minutes so each team sums to 240."""

    def prepare_sampling_frame(self, contexts: pd.DataFrame) -> pd.DataFrame:
        """prepare_contexts plus {TARGET}_RATE, {TARGET}_RATE_STD and
        {TARGET}_MINUTES_RATE_SLOPE, matching what sample_game consumes today."""

    def predict_player_stats(
        self, context: pd.DataFrame, history_df=None, include_confidence=False
    ) -> dict[str, float]: ...
```

Non-negotiable behaviors:

- `prepare_contexts` must run `allocate_team_minutes`, which raises on infeasible
  rosters rather than inventing minutes.
- Separate **official inference readiness** from **promotion readiness**.
  `supports_official` may be `True` for an unpromoted candidate only after its
  point-in-time roster/status contract, artifact contract, canonical output,
  and live/replay parity checks pass. This permits strict candidate shadow runs.
  Seven successful shadow slates remain a promotion gate, and an unpromoted
  candidate must never publish to the official ledger.
- The backend is pure inference: no fitting, no writes, no network.

### 6.3 Feature porting into the v2 materializer

- Register each migrated family as a `FeatureSpec` with `name=<EXACT>`,
  `group=<family>`, `entity_keys=("GAME_ID","TEAM_ID","PLAYER_ID")`,
  `event_time_column`, `required_sources`, `allowed_horizons`, `missing_policy`,
  and `version`.
- Prefer producers that read only verified snapshot inputs
  (`load_request_history`, `load_official_roster`, `load_request_status`).
- Feature names must avoid `is_forbidden_feature` and follow the existing
  `PRIOR_`/lag convention. No same-game box-score column may appear.
- Reuse the OG computation code where it is already leak-safe, but call it
  through the materializer so training and live share one implementation.

### 6.4 Reusing `CatBoostTrainer`

`src/training/catboost_trainer.py` is the OG modeling asset to port. Reuse:

- `TARGET_PROFILES` per-target `depth`, `iterations`, `learning_rate`,
  `l2_leaf_reg`, `min_data_in_leaf`, and `grow_policy`.
- Multi-loss prediction blending (`multi_loss_rmse_weight` /
  `multi_loss_mae_weight`, defaults `0.6` / `0.4`).
- Quantile low/high models for interval outputs.
- `save`/`load` artifact naming (`{target}_catboost.cbm`,
  `{target.lower()}_metadata.joblib`, `validate_saved_artifacts`,
  `missing_runtime_artifacts`).
- GPU `task_type="GPU"` with CPU fallback; GPU runs must use one worker
  (`AGENTS.md` safety note).

Do **not** reuse flat `models/` output paths or any code path that reads
current-game outcomes.

### 6.5 Determinism

- Pin `random_seed` per component and record it in `model.json`.
- Record `random_strength`, `bagging_temperature`, and `thread_count`.
- Assert byte-stable reload: save, load, predict twice, compare arrays exactly.
- GPU/CPU divergence is allowed only within a declared tolerance recorded per
  component.

---

## 7. Shared live/replay integration

### 7.1 Backend factory (closes G3)

Add a single resolver, for example
`src/forecasting/backend_factory.py:load_backend(bundle_dir) -> backend`, that:

1. Validates the bundle with `validate_v2_bundle(bundle_dir, promotion=False)`.
2. Reads `config_resolved.yaml`'s `model_backend.kind` for every newly created
   bundle.
3. Returns `V2BaselineBackend` for `v2_rolling_baseline` and
   `V2CatBoostBackend` for `v2_catboost_challenger`, raising on unknown kinds.

Replace the four hardcoded constructions:

- `src/pipeline/forecast_service.py` champion load.
- `src/simulation/v2_runner.py` `run_scheduled_game` candidate branch.
- `src/evaluation/request_replay.py` `replay_requests`.
- `promote_model.py` `_clean_process_validate` (its embedded snippet must import
  the factory, not `V2BaselineBackend`).

`train_v2_baseline` and `fit_fold_candidate` must write
`model_backend: {kind: v2_rolling_baseline}` into `config_resolved.yaml` for all
future bundles. Immutable bundles must not be edited in place. For a v2 bundle
created before this discriminator existed, the factory may select
`V2BaselineBackend` only through an explicit historical-v2 schema mapping that
requires all of the following: `bundle_manifest.architecture == "v2"`; successful
v2 bundle validation; the exact four `*/baseline.json` component paths; and the
known rolling/status backend values already present under `models` in
`config_resolved.yaml`. Record the resolution as
`pre_discriminator_v2_rolling_baseline`. Any partial or unknown signature raises.
This is not permission to load legacy flat artifacts or infer a backend from an
arbitrary filename.

### 7.2 Parity obligations

- `execute_request` is unchanged. The challenger must work through it as-is.
- `replay_requests` and the live path must produce identical samples and
  forecasts modulo `GENERATED_AT`. `tests/test_training/test_fold_candidate.py`
  already asserts this shape and the challenger must satisfy the same pattern.
- Any change to `ForecastService` must be additive; do not branch behavior on
  bundle kind inside the service.

### 7.3 Files changed by this plan (implementation phase)

Expected new files: `src/forecasting/catboost_backend.py`,
`src/forecasting/backend_factory.py`, `src/training/catboost_candidate.py`,
`src/training/official_challenger_campaign.py`,
`src/evaluation/v2_rolling_reference.py`, `src/features/og_families.py`.
Expected edits: `src/pipeline/forecast_service.py`, `src/simulation/v2_runner.py`,
`src/evaluation/request_replay.py`, `promote_model.py`,
`src/training/v2_baseline.py` and `src/training/fold_candidate.py` (add the
`model_backend` block), `config/model_v2.yaml`, `train_colab.ipynb` (new cells),
plus focused tests.

Each file has one owner during implementation; do not run parallel edits on the
same file.

---

## 8. Phase-by-phase execution

### Phase A — Baseline campaign and prerequisites (owner: lead)

Tasks:

1. Audit P1-P6 and produce one machine-readable prerequisite report, including
   explicit pass/fail status and evidence paths.
2. Record whether the inputs needed to establish P4 are available; do not run
   the campaign before Phase B writes the backend discriminator into newly
   created fold bundles.
3. Record `policy.policy_id` and the intended fold boundaries.
4. Update `plans/model_v2_improvement_execution_status.md` with the readiness or
   blocked status.

Exit gate: the prerequisite report exists. P4 is established in Phase B. Phases
D and E require P1-P6 and
`_require_complete_baseline(baseline_evidence)` to pass.

### Phase B — Backend factory and baseline parity (owner: worker 1)

Tasks:

1. Add `model_backend` to both trainers' resolved config before creating any new
   fold bundle.
2. Implement `load_backend` and swap the four hardcoded call sites.
3. Keep `V2BaselineBackend` behavior byte-identical.
4. If P1-P3/P5 pass, run and freeze the diagnostic rolling-baseline campaign
   through the updated trainer, validate it with
   `_require_complete_baseline`, and mark P4 complete. This campaign is
   diagnostic evidence and must not be relabeled as official.

Exit gate: existing bundles load through the factory; the focused v2 subset still
passes; a baseline candidate replay equals its pre-change replay; and P4 is
either complete or recorded as the gate blocking Phase D/E.

### Phase C — Feature migration wave 1 (owner: worker 2)

Tasks:

1. Implement `src/features/og_families.py` for the PORT groups.
2. Register `FeatureSpec`s; extend `materialize_scheduled_rows` only where a
   producer needs the verified snapshot loaders.
3. Add the Section 5.3 leakage tests for each family.

Exit gate: every new family passes all six leakage tests; `validate_feature_names`
accepts the schema and rejects the forbidden columns.

### Phase D — CatBoost components and fold-fitted candidate (owner: worker 3)

Entry gate: P1-P6 pass. Fixture-only component tests may be written before
this gate, but no fitted candidate may be presented as replay evidence.

Tasks:

1. Add participation classifiers (active, play-given-active) with
   `ParticipationClassifier` plus `ProbabilityCalibrator` fit on fit-role OOF
   evidence.
2. Add minutes mean/P10/P90 with `allocate_team_minutes` unchanged.
3. Add conditional stat-rate models per target, plus optional quantiles.
4. Implement `src/training/catboost_candidate.py` mirroring
   `fit_fold_candidate`'s fit-only sealing (fit-game labels only, no outer
   labels).
5. Serialize per Section 6.1 and implement `V2CatBoostBackend`.

Exit gate: a sealed challenger candidate loads through the factory, produces a
canonical forecast frame via `execute_request`, satisfies all 240-minute
invariants, and its live/replay outputs match.

### Phase E — Multi-fold challenger campaign (owner: lead plus reviewer)

Entry gate: P1-P6 pass and the checksummed rolling-baseline campaign is complete.

Tasks:

1. Keep the current diagnostic `run_fold_campaign` honest: it rejects official
   requests, fits mean scales only, and always emits ineligible evidence. Do not
   relabel its output as official.
2. Implement `run_official_challenger_campaign` in
   `src/training/official_challenger_campaign.py`. Reuse its chronological fold
   and trial-selection primitives, but require Tier A/B point-in-time requests,
   a CatBoost candidate-trainer callback, fit-only component training, tune-only
   model selection, calibration-role probability/distribution calibration, and
   one frozen outer evaluation.
3. Require the baseline campaign directory; fail when absent or incomplete.
4. Generate the `v2_rolling_reference` from the same materialized official
   contexts as the challenger, and freeze its input-context digests before
   scoring.
5. Produce `training_folds.json`, `baseline_scorecard.json`,
   `candidate_scorecard.json`, `slice_scorecard.json`, and
   `calibration_scorecard.json` in the shapes
   `_validate_and_evaluate_promotion_evidence` recomputes.
6. Run `paired_macro_bootstrap` against the comparator selected by tune-window
   normalized MAE.
7. Add acceptance tests proving that the official campaign rejects diagnostic
   requests and Tier C evidence, never exposes outer labels before selection and
   calibration freeze, fits non-placeholder distribution calibrators, emits at
   least two complete four-role folds, and passes promotion-evidence
   recomputation without hand-editing stored decisions.

Exit gate: evidence files load and the recomputation accepts them; no gate is
edited.

### Phase F — Shadow operation and promotion decision (owner: lead)

Tasks:

1. Run daily shadow forecasts for at least seven consecutive slates through
   `run_scheduled_game(..., candidate_dir=...)`; record `shadow_slates`.
2. Build the final sealed candidate from a clean worktree.
3. Run `python promote_model.py --candidate <id> --dry-run`.
4. Promote only on a passing, recomputed decision; otherwise leave the deployment
   pointer unchanged and publish the honest result.

Exit gate: either a promoted challenger with rollback proven, or a documented
honest non-promotion with the prior deployment state retained. If no champion was
configured before the experiment, non-promotion means remaining without a
champion; the rolling baseline remains a comparator, not an implied deployment.

---

## 9. Evaluation experiment: rolling baseline vs challenger

### 9.1 Population

Do not compare the official challenger campaign to the diagnostic rolling-bundle
campaign as if they were two interchangeable request arms. They have different
scenario contracts, and the diagnostic evidence must never be relabeled
official.

For promotion evidence, materialize each official request once, then evaluate:

1. The CatBoost candidate through `execute_request`.
2. A pure `v2_rolling_reference` evaluator in
   `src/evaluation/v2_rolling_reference.py` that applies the declared rolling-v2
   participation/minutes/rate rules to the **same verified official roster,
   status, history, cutoff, and eligible population**. Its rows carry the
   candidate request's `REQUEST_ID`, an `INPUT_CONTEXT_DIGEST`, and
   `BASELINE_KIND=v2_rolling_reference`; they are evaluation evidence only, not a
   model bundle, forecast ledger entry, or claim that the diagnostic baseline is
   official-ready.

Score the candidate and reference one-to-one on
`(REQUEST_ID, INPUT_CONTEXT_DIGEST, GAME_ID, PLAYER_ID, STAT)`. Reject duplicate
keys, digest differences, population differences, or
`expected_rows != known_rows`. Preserve the frozen diagnostic rolling campaign
as separate sanity/provenance evidence. Any optional cross-bundle analysis is
non-promotional and must retain both original scenarios and request IDs rather
than mapping diagnostic execution to official execution.

### 9.2 Baselines in every scorecard

- Simple: `rolling_5`, `rolling_10`, `rolling_20`, `ema_10`,
  `minutes_x_rate_10` (`BASELINES`).
- V2 system comparator: `v2_rolling_reference` rows generated from the same
  official request materialization as the CatBoost candidate.
- Provenance-only reference: the separately sealed diagnostic rolling-baseline
  campaign; report it independently and never use it to fill missing official
  comparison rows.
- Optional separate reporting: `player_game_eligibility`-driven participation
  rules and `DEFAULT_STATUS_RULES`.

### 9.3 Metrics

Point: MAE, RMSE, bias, normalized MAE (macro mean of target MAE divided by the
frozen scale), per-target improvement.
Distributions: `score_distributions` CRPS, discrete Brier, zero Brier, pinball
10/50/90, coverage 80/90, width 80/90.
Participation/minutes: `score_opportunity` active/appeared calibration and
missingness; minutes MAE versus rolling-10 and role median.
Slices: `slice_scorecards` with predeclared denominators, preserving ineligible
and small slices rather than dropping them.

### 9.4 Decision rule

The challenger wins a fold only if it improves normalized MAE versus the
tune-selected comparator **and** the paired game-bootstrap interval excludes
baseline superiority (`paired_macro_bootstrap` -> `ci` with `ci[1] < 0`).
`paired_game_bootstrap_many` with `holm` is used for per-target claims. Practical
significance is required on top of statistical significance.

---

## 10. Promotion criteria (no gate weakening)

The challenger must satisfy, unchanged:

| Gate | Threshold | Source |
|---|---|---|
| Aggregate normalized MAE improvement | >= 1% | `PromotionThresholds.normalized_mae_improvement` |
| Core target regression (PTS/REB/AST) | <= 1% | `core_target_max_regression` |
| Secondary target regression (STL/BLK/TOV) | <= 2% | `secondary_target_max_regression` |
| Pairwise bootstrap | upper endpoint < 0 | `evaluate_v2_promotion(bootstrap_ci=...)` |
| Participation | beats rule baselines | `participation.beats_baseline = true` |
| Minutes | beats rolling-10 and role median | `minutes.beats_baseline = true` |
| Interval coverage | 80% and 90% within +/-3pp overall, +/-5pp on slices | `coverage_tolerance_*` |
| Point-in-time | all leakage tests pass | `contracts.point_in_time` |
| Live/replay parity | identical samples | `contracts.live_replay_parity` |
| Artifact contract | checksums, manifest, clean tree | `REQUIRED_CONTRACT_FLAGS` |
| Calibration | calibrated distributions present | `REQUIRED_CONTRACT_FLAGS` |
| Core reconciliation | >= 0.999 | `min_core_reconciliation` |
| Major-slice reconciliation | >= 0.95 | `min_major_slice_reconciliation` |
| Fallback / degraded rate | <= 5% / <= 10% | `max_fallback_rate`, `max_degraded_rate` |
| Folds | >= 2 four-role folds | `_validate_folds` |
| Shadow slates | >= 7 consecutive successes | `validate_v2_bundle(promotion=True)` |

Explicitly forbidden: editing `PromotionThresholds`, `REQUIRED_TARGETS`,
`REQUIRED_CONTRACT_FLAGS`, `PROMOTION_EVIDENCE_SCHEMA_VERSION`, or the
recomputation in `src/models/bundle.py` to accommodate a candidate. Threshold
changes are permitted only as a separately reviewed, pre-registered policy
version applied to **both** arms and landed before the challenger runs.

If the challenger fails, the correct outcome is to leave the deployment pointer
unchanged and record the failure, not to lower the bar. The current repository
has no configured champion, so the rolling baseline must not be described as the
deployed champion unless it independently becomes promotion-eligible.

---

## 11. Test matrix (focused tests only)

Per `AGENTS.md`, run the lightweight v2 subset, not the full suite.

**Existing suites that must stay green:**

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

Also keep the fold-machinery coverage green:
`tests/test_training/test_fold_candidate.py` and
`tests/test_training/test_fold_campaign.py`.

**New focused tests to add:**

| Test | File | Asserts |
|---|---|---|
| Backend factory resolves both kinds | `tests/test_forecasting/test_backend_factory.py` | baseline -> `V2BaselineBackend`, challenger -> `V2CatBoostBackend`, unknown -> raise |
| Baseline parity after factory | `tests/test_training/test_v2_baseline.py` | a genuinely pre-discriminator v2 bundle resolves through the explicit historical schema mapping and predicts identically; unknown/partial signatures raise |
| OG family leakage | `tests/test_features/test_og_family_leakage.py` | all six Section 5.3 checks per family |
| Forbidden schema rejection | `tests/test_contracts/test_feature_safety.py` | every `*_TEAM` name rejected |
| CatBoost component round-trip | `tests/test_training/test_catboost_components.py` | save/load predicts deterministically; artifacts listed in `model.json` |
| Challenger fold fit isolation | `tests/test_training/test_catboost_fold_candidate.py` | outer labels cannot enter; `fit_game_ids` recorded |
| Challenger live/replay parity | `tests/test_simulation/test_v2_catboost_execution.py` | `execute_request` equals `replay_requests` modulo `GENERATED_AT` |
| 240-minute invariant | existing joint-sampler tests plus a challenger variant | per-team sum equals 240 |
| Promotion recomputation honesty | `tests/test_models/test_v2_bundle.py` | hand-edited decisions are rejected |
| Campaign requires baseline | `tests/test_training/test_fold_campaign.py` | `_require_complete_baseline` blocks absent or incomplete evidence |
| Official campaign evidence | `tests/test_training/test_official_challenger_campaign.py` | Tier C/diagnostic requests rejected; selection/calibration/outer isolation; fitted calibration; >=2 folds; evidence recomputes |
| Rolling-v2 official-context reference | `tests/test_evaluation/test_v2_rolling_reference.py` | candidate and reference share request/context digest/population; reference cannot write ledger rows; any input or population mismatch raises |

Do not add slow/GPU/integration tests to the default selection; mark them
`slow`/`gpu`/`integration` as appropriate and run only on request.

---

## 12. Rollout and rollback

**Rollout**

1. Land the backend factory and feature families with no default change. The
   deployment pointer stays unchanged; at the time this plan was prepared, no
   `models/champion.json` existed and the rolling baseline was only a diagnostic
   comparator.
2. Seal the challenger candidate from a clean worktree; record commit, the dirty
   flag (`false`), snapshot IDs, and component versions.
3. Shadow-run seven consecutive slates with `candidate_dir` set (never writing
   the official ledger).
4. Run `promote_model.py --dry-run` and inspect the recomputed decision record.
5. Promote atomically via `models/champion.json`; run the clean-process golden
   load and one golden forecast.
6. Persist every official forecast to the ledger before tip.

**Rollback**

1. If a previous eligible champion exists, use `ModelVersionRegistry.rollback`
   to repoint `models/champion.json` to that immutable bundle; never copy model
   files over a bundle. Before a first-ever promotion, document and test the
   controlled restoration of the pre-promotion no-champion state because there
   is no previous model target to select.
2. Re-run the golden load and one golden forecast against the restored champion.
3. Record the rollback reason, the failed bundle ID, and the triggering evidence.
4. Automatic rollback is triggered if post-promotion validation fails; drift
   alone never triggers promotion or a new architecture.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Official roster/status coverage (P1/P2/P3) unavailable | Participation can never beat rule baselines; challenger stays ineligible | Do not substitute appearance-only history; report the gap and leave deployment unchanged |
| CatBoost overfits relative to the rolling baseline | No value added | Mandatory per-target comparisons, grouped ablation, and multi-fold evidence; retain the prior deployment state when CatBoost loses. The rolling baseline may be deployed only if it independently becomes promotion-eligible |
| Train/serve skew from actual minutes | Optimistic metrics that do not hold live | Train on out-of-fold predicted minutes only; parity test on the same path |
| Feature count and runtime blow-up | Misses the daily horizon | Coverage gates, grouped ablation, and per-component timing recorded in the bundle |
| GPU/CPU nondeterminism | Irreproducible evidence | Fixed seeds, one GPU worker, recorded tolerances, byte-stable reload test |
| Silent re-introduction of leaked `*_TEAM` features | Contaminated metrics | `feature_schema_v4` enforcement plus mutation tests on every family |
| Backend factory regression | Champion stops loading | Parity test before and after the swap; baseline bundle frozen as a fixture |
| Scope creep into Transformer/tracking work | Challenger never finishes | Explicitly deferred; architecture-plan Phase 8 owns that work |
| Dirty worktree at seal time | Promotion rejection | Clean-tree requirement; seal from a dedicated commit |

---

## 14. Definition of done

The work is done when all of the following are true:

1. A sealed challenger bundle exists with CatBoost components in all four
   required directories, a clean-worktree manifest, and an exact code version.
2. Its evidence files (`training_folds.json`, baseline/candidate/slice/
   calibration scorecards) satisfy `_validate_and_evaluate_promotion_evidence`
   with **unmodified** thresholds.
3. `execute_request` and `replay_requests` produce identical samples for the
   challenger (parity proven, not asserted).
4. All Section 5.3 leakage tests pass for every migrated family.
5. A written comparison against the rolling baseline and all declared simple
   baselines exists, with paired game-bootstrap intervals and complete
   eligible-population accounting.
6. The promotion decision is recorded: either a promoted challenger with rollback
   proven, or an honest non-promotion with the prior deployment state retained
   and the reason documented.
7. The focused test subset in Section 11 passes.
8. `plans/model_v2_improvement_execution_status.md` and
   `plans/model_v2_readiness_implementation_status.md` reflect the outcome.

---

## 15. Executable checklist

Prerequisites:

- [ ] P1 Official roster membership passes `load_official_roster` for campaign games.
- [ ] P2 `player_game_eligibility.csv` has `ACTIVE` and `APPEARED` including DNPs.
- [ ] P3 `player_status_snapshots.csv` has `AVAILABLE_AT`.
- [ ] P4 Rolling-baseline campaign passes `_require_complete_baseline`.
- [ ] P5 `EvaluationPolicy` frozen; `policy_id` recorded.
- [ ] P6 Clean worktree, CatBoost installed, `environment.lock` captured.

Phase A:

- [ ] Baseline campaign frozen with checksums and comparator selection.
- [ ] Status notes updated with the campaign path.

Phase B:

- [ ] `model_backend` written by both trainers.
- [ ] `load_backend` implemented; four hardcoded sites swapped.
- [ ] Baseline parity test green.

Phase C:

- [ ] PORT families registered as `FeatureSpec`s.
- [ ] PORT-WITH-CHANGE families lagged and registered.
- [ ] DEFER families documented as disabled with the missing source named.
- [ ] All six leakage checks per family green.

Phase D:

- [ ] Participation classifiers fitted and calibrated on fit-role OOF evidence.
- [ ] Minutes mean/P10/P90 fitted; allocation unchanged.
- [ ] Conditional rate models fitted per target.
- [ ] `V2CatBoostBackend` loads, predicts, and honors `supports_official`.
- [ ] Artifacts serialized per Section 6.1 with recorded seeds.

Phase E:

- [ ] Official campaign preserves four-role isolation while adding the required
  official request, fitted-distribution calibration, and promotion-evidence path.
- [ ] Evidence files recompute to the recorded decision.
- [ ] Macro bootstrap interval recorded against the tune-selected comparator.

Phase F:

- [ ] Seven consecutive shadow slates recorded.
- [ ] Clean-worktree candidate sealed.
- [ ] `promote_model.py --dry-run` run and inspected.
- [ ] Promotion or documented non-promotion completed.
- [ ] Rollback path exercised.
