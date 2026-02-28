# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python CLI-based ML project (no web server, no Docker, no database). The virtual environment lives at `/workspace/venv`.

### Activating the environment

Always activate the venv before running anything:

```bash
source /workspace/venv/bin/activate
```

### Running tests

```bash
pytest tests/ -v
```

One pre-existing test (`test_registry_initialization`) fails due to a PosixPath vs string assertion mismatch — this is a known issue in the test, not an environment problem.

### Running the application

See `README.md` for full pipeline usage. Key entry points:

- `python query_prob.py` — interactive probability query CLI
- `python train.py` — train ML models (requires data from `update_data.py`)
- `python simulate_season.py --today` — simulate today's NBA games
- `python update_data.py` — fetch NBA data (requires internet; uses NBA.com Stats API with rate limits)

### Notes

- PyTorch is installed with CUDA support but runs on CPU in this environment (auto-detected). No GPU-specific setup needed.
- `update_data.py` calls the NBA.com Stats API which has rate limits; fetching many seasons can be slow.
- The `data/` and `models/` directories are gitignored; they are created on first use by the scripts.
- No linter is configured in the project (no flake8/ruff/pylint config files). PEP 8 conventions are expected per the README.
- Configuration is in `config/default.yaml`.
