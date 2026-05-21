# AGENTS.md

Pure-Python CLI ML project (no web server, Docker, or database). Predicts NBA player stats via ensemble deep learning + GPU-accelerated Monte Carlo simulation. venv at project root (Python 3.12).

## Environment

Always activate venv before running anything:

```bash
source venv/bin/activate
```

No linter or formatter is installed (no ruff/flake8/pylint). Follow PEP 8.

## Tests (260+ total)

```bash
pytest tests/ -v                  # full suite (slow — several minutes)
pytest tests/test_config/ -v      # single package (fast)
pytest tests/test_query/ -v       # single package (fast)
pytest -m "not slow"              # skip slow-marked tests
```

Custom markers: `slow`, `gpu`, `integration` (registered in `tests/conftest.py`).
No `--timeout` flag (pytest-timeout not installed); some suites are slow and may exceed default 2-min tool timeout — increase timeout if needed.

## Pipeline (order matters: each step depends on prior output)

1. `python update_data.py` — fetch NBA data (requires internet; NBA.com API rate-limited at ~0.6s/request)
   - `--interactive` / `-i` for first-time setup (recommended)
   - `--all-seasons` for last 10 seasons
   - `--update` for incremental (only new games since last run)
   - `--force` to re-fetch even if data exists
2. `python train.py` — train models (requires `data/` from step 1)
   - Training preset: `--preset {small,full}` (default: `full`; `small` skips Transformer, fewer features)
   - Training mode: `--mode {quick,standard,full}`; defaults come from preset
   - Config: `config/default.yaml` → `training_presets`
   - Will fail if `data/nba_players.csv` is missing. Run `update_data.py` first.
3. `python simulate_season.py --today` — simulate games (requires `models/` from step 2)
   - Other modes: `--date YYYY-MM-DD`, `--week`, `--season`
   - `--sims N` for simulation count (default: 100)
   - `--workers 1` recommended when using GPU (avoid CUDA context contention)
4. `python query_prob.py` — interactive probability query CLI (requires models + data)

### Cleanup

```bash
python clear_cache.py --all --dry-run   # preview
python clear_cache.py --all --yes       # remove cache/, models/, data/cache/, data/sim_*, experiments/, __pycache__/
```

Raw CSV files in `data/` are preserved by clear_cache.

## Architecture

```
Root entry points:  update_data.py, train.py, simulate_season.py, query_prob.py
src/
  config/          — config loading (main config: config/default.yaml)
  data/            — scrapers (NBA API, ESPN injuries, Rotowire lineups, Action Network betting, Basketball Reference)
  preprocessing/    — 15-phase feature engineering pipeline → 150+ features (incl. lifecycle: injury risk, aging curves, KAN aging, skill dev)
  models/          — model_manager.py (live bridge), CatBoost (primary), Transformer (secondary)
  pipeline/        — training_pipeline.py, data_pipeline.py, prediction_service.py
  simulation/      — game_simulator.py (GPU Monte Carlo), season_simulator.py
  query/           — interactive_cli.py, probability_calculator.py
  training/        — training orchestration, presets, experiment tracking
  evaluation/      — model evaluation
  utils/           — logging, reproducibility, team_mappings
  lifecycle/       — B-Ianus Bayesian aging model, KAN age model, injury risk computation
tests/             — mirrors src/ structure; conftest.py injects project root into sys.path
```

## Gotchas

- **Disabled models**: LSTM and GNN are `enabled: false` in `config/default.yaml`. CatBoost + Transformer is the active stack.
- **CatBoost parallelism**: On GPU use `max_workers=1` (CUDA context contention); on CPU size workers to machine cores.
- **Gitignored at root only**: `.gitignore` uses `/data/` and `/models/` (with leading slash), so only root-level `data/` and `models/` are ignored — `src/data/` and `src/models/` are tracked.
- **PyTorch test shim**: `src/__init__.py` installs a NumPy-backed torch shim when running under pytest on machines without a working PyTorch. Tests import `src` and get the shim automatically; real runtime still requires real torch.
- **PyTorch CUDA**: installed with CUDA support but auto-detects CPU on machines without GPU. No setup needed.
- **NBA API rate limits**: `update_data.py` is slow for many seasons; use `--interactive` to select specific seasons.
- **Data must exist before train**: `train.py` will fail if `data/nba_players.csv` is missing. Run `update_data.py` first.
- **Player bio data required**: Lifecycle features need AGE and POSITION columns in `nba_players.csv`. Run `update_data.py` at least once after adding the PlayerBioScraper to populate them. Missing AGE defaults all aging features to neutral (1.0 factor).
- **Injury history is incremental**: The `injury_history.csv` file grows over time. First runs will have sparse data — injury risk features will be near-zero for most players until several update cycles accumulate.
- **KAN model precomputation**: KAN aging factors are pre-computed on CPU and cached to `data/cache/kan_aging_outputs.csv`. If you retrain with `--force`, delete this file to force recomputation. KAN always runs on CPU to avoid GPU contention with CatBoost/Transformer.
- **B-Ianus aging precomputation**: Aging curves are cached to `data/cache/aging_curves.csv`. Same rule — delete to force recomputation.