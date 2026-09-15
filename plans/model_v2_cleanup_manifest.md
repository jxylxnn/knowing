# Model v2 cleanup manifest

This is the exact pending deletion set for the v2-only repository cleanup.
Generated caches, reports, experiments, and unsafe model artifacts have already
been removed with `clear_cache.py --all --yes`; raw CSV data was preserved.

The paths below are tracked project history or source. They remain recoverable
from Git, but deletion is intentionally paused until the owner explicitly
approves this manifest.

## Stale project/editor material

```text
.agent-teams/
.hermes/
.opencode/
.roo/
.webui_secret_key
cline_docs/
docs/build_polymarket_handoff.py
docs/polymarket_forecasting_handoff.docx
docs/polymarket_forecasting_handoff.md
project-brain/
DECISIONS.md
IMPROVEMENTS_SUMMARY.md
TASKS.md
NBA_Prediction_System_Explained.pdf
feature_pipeline_report.pdf
program_walkthrough.pdf
query_prob.ipynb
changes_dr-034_feature_cache.diff
worker-diff.patch
reconstruction_pr00_pr03.diff
```

## Superseded plans

Keep `model_v2_full_architecture_plan.md`, this cleanup manifest, and the current
v2 status plans: `model_v2_bug_fix_plan.md`, `model_v2_improvement_plan.md`,
`model_v2_improvement_execution_status.md`, and
`model_v2_readiness_implementation_status.md`. The claims, game-reconstruction,
and preseason process plans remain superseded/unshipped.

```text
plans/2026_27_architecture_rebuild.md
plans/audit_baseline_2026_07_13.md
plans/bugfix_groupby_series_keyerror.md
plans/catboost_gpu_callback_fix.md
plans/claims_reality_remediation_plan.md
plans/distributed_training_implementation.md
plans/fix_device_type_attribute_error.md
plans/game_reconstruction_mega_plan.md
plans/gpu_optimization.md
plans/preseason_subagent_01_data_readiness.md
plans/preseason_subagent_02_model_release_readiness.md
plans/reconstruction_baseline_2026_07_14.json
plans/repository_improvement_execution_plan.md
plans/self_optimizing_system.md
```

## Superseded root commands

These commands implement the flat-artifact, residual-correction, or old
backtest path. Model v2 uses `train.py`, `replay.py`, `promote_model.py`,
`simulate_season.py`, `query_prob.py`, and `reconcile_predictions.py` instead.

```text
backtest.py
build_residual_dataset.py
calibrate_residual_intervals.py
improve_models.py
monitor_residual_corrections.py
optimize_variance.py
optimize_weights.py
train_residual_models.py
```

## Superseded source packages

```text
src/correction/
src/reconstruction/
src/evaluation/backtest_runner.py
src/evaluation/continual_learning.py
src/evaluation/prediction_ledger.py
src/evaluation/residual_monitor.py
src/evaluation/residual_report.py
src/models/error_calibration.py
src/models/minutes_predictor.py
src/models/model_manager.py
src/pipeline/data_pipeline.py
src/pipeline/prediction_service.py
src/pipeline/training_pipeline.py
src/query/confidence_adjustment.py
src/query/distribution_fitter.py
src/query/empirical_covariance.py
src/query/interactive_cli.py
src/query/prob_formatter.py
src/query/probability_calculator.py
src/query/projection_loader.py
src/query/query_parser.py
src/simulation/archetype.py
src/simulation/four_factors_engine.py
src/simulation/game_context_engine.py
src/simulation/game_simulator.py
src/simulation/input_health.py
src/simulation/phase_simulator.py
src/simulation/player_correlation_engine.py
src/simulation/report_generator.py
src/simulation/role_sampler.py
src/simulation/season_simulator.py
src/simulation/sim_cache.py
src/simulation/sim_types.py
src/simulation/stat_utils.py
```

The feature registry, data scrapers, CatBoost trainer, Transformer challenger,
lifecycle features, bundle/versioning code, and v2 evaluation primitives are
retained because the authoritative plan explicitly reuses them.

## Tests removed with retired paths

```text
tests/test_correction/
tests/test_reconstruction/
tests/test_evaluation/test_backtest_json_output.py
tests/test_evaluation/test_continual_learning.py
tests/test_evaluation/test_prediction_ledger.py
tests/test_evaluation/test_residual_monitor.py
tests/test_query/test_interactive_cli.py
tests/test_query/test_probability_calculator.py
tests/test_query/test_projection_loader.py
tests/test_query/test_six_stat_contract.py
tests/test_simulation/test_game_simulator.py
tests/test_simulation/test_simulation_health_reporting.py
tests/test_pipeline/test_data_pipeline.py
tests/test_training/test_diagnostics.py
tests/test_training/test_laptop_quality_smoke.py
tests/test_training/test_train_entrypoint.py
```

After approval, imports and package `__init__.py` files must be normalized,
then the lightweight v2 test selection and compile check must be rerun.
