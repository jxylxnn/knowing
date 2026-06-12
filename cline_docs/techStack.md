# Tech Stack

## Core Machine Learning

### Deep Learning Framework
- **PyTorch 2.0+** — Primary deep learning framework
  - Transformer attention model (active, secondary)
  - LSTM temporal model (disabled in config)
  - Graph Neural Network (disabled in config)
  - Custom Nexus model (optional, CRPS loss)
  - GPU acceleration support (CUDA 11.8+)

### Gradient Boosting
- **CatBoost 1.2.8** — Primary gradient boosting model (per-target)
  - Handles categorical features natively
  - GPU-accelerated training
  - Multi-loss training (RMSE + MAE + quantile regression)
- **LightGBM 4.6.0** — Installed, available for experimentation (not wired into active pipeline)
- **XGBoost 3.1.2** — Installed, available for experimentation (not wired into active pipeline)

### Classical ML
- **scikit-learn 1.8.0** — Traditional ML utilities
  - Ridge regression for ensemble blending
  - HistGradientBoosting for fast feature importance
  - Data preprocessing and model evaluation metrics

---

## Data Processing

### Core Libraries
- **pandas 2.3.3** — DataFrame operations, data manipulation
- **numpy 2.3.5** — Numerical computing, array operations
- **scipy 1.16.3** — Statistical functions, distributions, scipy.optimize for weight tuning

---

## Data Sources

### NBA Data APIs
- **nba_api 1.11.3** — Unofficial NBA Stats API wrapper
  - Player game logs
  - Team statistics
  - Matchup data

### Web Scraping
- **beautifulsoup4 4.14.3** — HTML parsing
  - Basketball Reference scraping
  - ESPN injury reports
  - Rotowire lineups

### HTTP Requests
- **requests 2.32.5** — HTTP client for API calls and scraping
- **aiohttp 3.13.2** — Async HTTP client

---

## Visualization

### Plotting Libraries
- **plotly 6.5.0** — Interactive visualizations
- **matplotlib 3.10.8** — Static plotting
- **seaborn 0.13.2** — Statistical visualization

---

## NLP/Transformers

### Transformers Library
- **transformers 4.57.3** — Hugging Face transformers (pre-trained model utilities)
- **tokenizers 0.22.1** — Fast tokenization
- **einops 0.8.1** — Tensor operations

---

## Testing & Quality

### Testing Framework
- **pytest 9.0.2** — Test framework (260+ test functions, 32 test files)
- **pytest-cov 7.0.0** — Code coverage measurement

---

## Architecture Decisions

### Project Type
- **Pure-Python CLI application**
- No web server (Flask/FastAPI not used)
- No Docker containerization
- No database (file-based storage with CSV/JSON)

### Storage
- **File-based**: CSV files for player/game data
- **CatBoost native format**: `.cbm` model files
- **Versioned JSON**: Ensemble blend weights in `models/blend_weights/` (atomic writes, rollback)
- **Pickle**: Serialized metadata (`.pkl`, `.joblib`)

### Configuration
- **YAML** configuration via `config/default.yaml`
- Centralized configuration loading in `src/config/`
- Preset system: `full` (default) and `small` (skips Transformer, fewer features)

---

## Directory Structure

```
knowing-master/
├── config/           # Configuration files (default.yaml)
├── data/             # Data storage (gitignored)
├── models/           # Trained models (gitignored)
├── cache/            # Scraped data cache
├── src/
│   ├── config/       # Configuration handling
│   ├── contracts/    # Inter-step artifact validation
│   ├── data/         # Data scrapers (10 modules)
│   ├── evaluation/   # Backtest, optimizer, drift detector, smart selector
│   ├── lifecycle/    # B-Ianus aging model, KAN age model
│   ├── models/       # ML models (model_manager, transformer, nexus, etc.)
│   ├── pipeline/     # Training & prediction pipelines
│   ├── preprocessing/# Feature engineering (25+ FeatureGroup modules)
│   ├── query/        # Probability query CLI (7 modules)
│   ├── simulation/   # Monte Carlo simulation (13 modules)
│   ├── training/     # Training orchestration (10 modules)
│   └── utils/        # Utilities
├── tests/            # Unit tests (mirrors src/ structure)
├── plans/            # Implementation plans
├── project-brain/    # Curated architectural brain
└── *.py              # Entry point scripts (9 root scripts)
```

---

## GPU Support

### CUDA Configuration
- PyTorch installed with CUDA support (auto-detects CPU on machines without GPU)
- Automatic CPU fallback when GPU unavailable
- Optional but recommended for high-volume simulations

### GPU-Accelerated Operations
- CatBoost training (GPU mode)
- PyTorch inference (CUDA tensors)
- Monte Carlo simulation vectorization

---

## Dependencies

All dependencies are specified in `requirements.txt`. Key packages:

```
torch (CUDA optional)
catboost==1.2.8
scikit-learn==1.8.0
lightgbm==4.6.0
xgboost==3.1.2
pandas==2.3.3
numpy==2.3.5
scipy==1.16.3
nba_api==1.11.3
beautifulsoup4==4.14.3
plotly==6.5.0
matplotlib==3.10.8
transformers==4.57.3
pytest==9.0.2
```

Install with:
```bash
source venv/bin/activate
pip install -r requirements.txt
```
