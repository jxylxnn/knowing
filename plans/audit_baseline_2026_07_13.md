# Repository Baseline Audit — 2026-07-13

## Scope

This report preserves the state that existed before the execution plan began
and records the first read-only audit. The preservation checkpoint is commit
`01762beba7782d1b251747c987c9e7256e7c95cb` on branch
`codex-model-training-cleanup`. The implementation branch is
`codex/baseline-audit`.

The checkpoint contains the owner’s existing worktree changes. No changes
were discarded. The audit entry point added for this PR is
`audit_project.py`; it only reads local files, never makes network requests,
and reports unsafe legacy artifacts without failing solely because they are
unsafe.

## Runtime

- Python: 3.12.0
- Platform: macOS 26.3, arm64
- Venv: `venv/`

Installed versions of direct dependencies discovered from `requirements.txt`
and `requirements-dev.txt`:

| Package | Version |
| --- | --- |
| PyYAML | 6.0.3 |
| beautifulsoup4 | 4.14.3 |
| catboost | 1.2.8 |
| coverage | 7.13.4 |
| efficient-kan | 0.1.0 |
| joblib | 1.5.3 |
| lightgbm | 4.6.0 |
| nba_api | 1.11.3 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| psutil | 7.2.2 |
| pytest | 9.0.2 |
| pytest-cov | 7.0.0 |
| requests | 2.32.5 |
| rich | 14.3.4 |
| scikit-learn | 1.8.0 |
| scipy | 1.16.3 |
| torch | 2.6.0 |
| tqdm | 4.67.1 |

`pyarrow==24.0.0` is installed in the venv but was not a declared direct
runtime dependency at this baseline.

## Verification baseline

Commands were run with the repository venv activated:

| Command | Result | Duration |
| --- | --- | --- |
| `python -m compileall -q src tests *.py` | PASS | <1 s |
| `git diff --check` | PASS | <1 s |
| `pytest -m "not slow" -q` | 479 passed, 7 deselected, 305 warnings | 152.00 s |
| `pytest -m "slow or gpu or integration" -q` | 13 passed, 473 deselected | 75.36 s |
| `python check_contracts.py --models-dir models` | PASS | <1 s |
| `python audit_project.py --data-dir data --models-dir models` | PASS; unsafe legacy artifacts reported | <1 s |

The warning count matches the audit: 304 scikit-learn feature-name warnings
from smart feature selection and one joblib physical-core warning.

## Data snapshot at baseline

The current data directory contains only the two core CSV files:

| File | Rows | Columns | Unique players | Unique games | Date range | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `data/nba_players.csv` | 52,583 | 71 | 686 | 2,442 | 2024-10-22 to 2026-04-10 | `7b22e4a14be48d933089336e950e95df068b4d515e2ea40bd3d92da23532bed1` |
| `data/nba_games.csv` | 4,882 | 57 | — | 2,441 | 2024-10-22 to 2026-04-10 | `dd9dc2cd286fe0ad6cb6d1e6bb8b2bc41fbed642bc0eec2137d02cb191a548cc` |

The player-only game ID is `0022501183`. No game-only ID was found. The
mismatch remains unresolved and is a PR 10 data-health task.

Missing optional sources:

- `data/player_bios.csv`
- `data/injury_history.csv`
- `data/advanced_tracking.csv`

## Active model artifacts

The active artifacts are a legacy flat CatBoost-only layout. There is no
`models/champion.json` and no `models/bundle_manifest.json`.

`feature_cols.pkl` and `feature_schema.pkl` each contain 309 features and
describe the same ordered schema. The following current-game team outcomes
are present and are unsafe:

```text
AST_TEAM
DREB_TEAM
FGA_TEAM
FTA_TEAM
OREB_TEAM
PTS_TEAM
REB_TEAM
TOV_TEAM
```

The complete baseline model file inventory and hashes is below.

| Path | SHA-256 |
| --- | --- |
| `ast_catboost.cbm` | `9f0fbd2e9769567fe6d7b9c6c4987ad62ec9b46e667688af8a94f8f4d17944ac` |
| `ast_catboost_qhigh.cbm` | `997699693f6262ee5c7d3c8eecf83509f4582627970bafc1747ea8ba833d3b35` |
| `ast_catboost_qlow.cbm` | `41a39091cb9a295267fcd92231686763d5f0d8412ab80384f8e3720ee71e305c` |
| `ast_metadata.joblib` | `0f740724eb193e81dde051fe580190ee5edd2a6294df1101d49c91a95b8696c7` |
| `blend_weights/current.json` | `a83e420f461f8f28f9e6ba574c3dc5ed09899bb08300aedeb8bf1c45b9a617d0` |
| `blend_weights/history.json` | `e06ca5a212842a94862641ec34584a2fe652047e0556b86e713aed00944f2504` |
| `blend_weights/v0001.json` | `a83e420f461f8f28f9e6ba574c3dc5ed09899bb08300aedeb8bf1c45b9a617d0` |
| `blend_weights.pkl` | `541472fdb1640b26096f20d8bb4670fafa77f284b2a60c40e731610865d74ffc` |
| `blk_catboost.cbm` | `c6596ce5eab5e3ea5b5ba8991fb0a4d8bd6870d6021d920c20b35f7384c61b3a` |
| `blk_catboost_qhigh.cbm` | `be130ae155ed484355fcaee86f475d15d4d7297ea405477ecc0f5572df2d1401` |
| `blk_catboost_qlow.cbm` | `9b3807d82201c058c224c8879f89267d1b4a90e1701a234332b8f3040433b60c` |
| `blk_metadata.joblib` | `6838be1f226c0f272de71f4e903a25f8e4c29da65b1dac2be78df96b6d3aecaf` |
| `feature_cols.pkl` | `b6b1e431ecda1909bb603db5fb8ba59b15ba883ce3821e15b1670f6309780771` |
| `feature_schema.pkl` | `295656744f4f8d7f7c9a39eaf4b57aaae1c6c912744d763034178e6112a936af` |
| `model_stack_metadata.pkl` | `058dc213c0abe00708d0e2518397a0ee28500563ae5c76b453cd7ad5fab29f7c` |
| `pts_catboost.cbm` | `007547b8015d35aedf30ce2bd55c257c30f6dec662c745d7227f0e3049dc089e` |
| `pts_catboost_qhigh.cbm` | `02cb7c2c636307c1f6859d6ab50978c5abfc81495e9c0de78a0225255904a5e1` |
| `pts_catboost_qlow.cbm` | `ed18ab7c935e5497763edafccafba609389f4a7cdbffb41e0bb7aaab5fb28b41` |
| `pts_metadata.joblib` | `e79420da8b4c8049813b90593296a34d55869fc94a4619368d5a39f54c067fa7` |
| `reb_catboost.cbm` | `3d511fc4c1119f6f14f2854599d2444e54f977eea016f9cb4a17567422d381c0` |
| `reb_catboost_qhigh.cbm` | `da6623f3cd9a9f3fa7fe7a7328c48f4ecac548fd62b534c595d8690a565903fe` |
| `reb_catboost_qlow.cbm` | `1313198363b6e8069e06419f4dc684a2471af2f7767eaf414828b4c3593c2d6f` |
| `reb_metadata.joblib` | `474546792df1d8270c76c85be987d93e3d796fc7930d2eebba30f3cf6acf6c1b` |
| `stl_catboost.cbm` | `38867a2c37568037c9c3da36e653035dd9b7328109054e9244f2683838a77a77` |
| `stl_catboost_qhigh.cbm` | `5f87e89f0562e22a1f795934f48cc16c0861447a3f4b128ef1ac53c1276a2341` |
| `stl_catboost_qlow.cbm` | `76df9c0d007768d3ebce2c222961d99dbe5473afb947fb5491ba87ca319fc391` |
| `stl_metadata.joblib` | `6b55a494a45a80f971de69399a80d122a7bdb024b9a4e45edc3c0714561c750e` |
| `tov_catboost.cbm` | `6664ef91f3d5ddfdb65d7256bc48ec4344ac8629d5114e1e5a3e12466de63b7c` |
| `tov_catboost_qhigh.cbm` | `f4de11aea347bbc0b039584177df06bbe16392e7ad7c860bfe24aa38e63645cf` |
| `tov_catboost_qlow.cbm` | `2de0dbb905a2dd1762ce823bb1d709366666d03be5dc6ae68e497ae572a29ce5` |
| `tov_metadata.joblib` | `5e2b8122e09254581ddcd06f7a36ed56d996cd56e20f2933ed6c056be9d588a5` |

## PR 00 acceptance

- The audit reproduces all eight unsafe `*_TEAM` features.
- It identifies all three absent optional data sources.
- It records the baseline commit, runtime, dependencies, data hashes, model
  hashes, schema count, and missing manifests.
- The audit and its tests are read-only with respect to data and model files.
- No production behavior was changed by PR 00.
