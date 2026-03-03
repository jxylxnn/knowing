# Tech Stack

## Core Machine Learning

### Deep Learning Framework
- **PyTorch 2.0+** — Primary deep learning framework
  - LSTM temporal model
  - Transformer attention model
  - Graph Neural Network
  - Joint multi-output neural network
  - GPU acceleration support (CUDA 11.8+)

### Gradient Boosting
- **CatBoost 1.2+** — Primary gradient boosting model
  - Handles categorical features natively
  - GPU-accelerated training
  - Multi-loss training (RMSE + MAE)
  - Quantile regression for uncertainty intervals
- **LightGBM 4.6+** — Additional boosting support

### Classical ML
- **scikit-learn 1.8+** — Traditional ML utilities
  - Ridge regression for ensemble blending
  - Data preprocessing
  - Model evaluation metrics

---

## Data Processing

### Core Libraries
- **pandas 2.3+** — DataFrame operations, data manipulation
- **numpy 2.3+** — Numerical computing, array operations
- **scipy 1.16+** — Statistical functions, distributions

---

## Data Sources

### NBA Data APIs
- **nba_api 1.11+** — Unofficial NBA Stats API wrapper
  - Player game logs
  - Team statistics
  - Matchup data

### Web Scraping
- **beautifulsoup4 4.14+** — HTML parsing
  - Basketball Reference scraping
  - ESPN injury reports
  - Rotowire lineups

### HTTP Requests
- **requests** — HTTP client for API calls and scraping

---

## Visualization

### Plotting Libraries
- **plotly 6.5+** — Interactive visualizations
- **matplotlib 3.10+** — Static plotting

---

## NLP/Transformers

### Transformers Library
- **transformers 4.57+** — Hugging Face transformers
  - Pre-trained model utilities
  - Tokenization support

---

## Testing & Quality

### Testing Framework
- **pytest 9.0+** — Test framework
- **coverage 7.13+** — Code coverage measurement

---

## Architecture Decisions

### Project Type
- **Pure-Python CLI application**
- No web server (Flask/FastAPI not used)
- No Docker containerization
- No database (file-based storage with CSV/JSON)

### Storage
- **File-based**: CSV files for player/game data
- **Pickle**: Serialized models
- **CatBoost native format**: `.cbm` model files
- **PyTorch format**: `.pt`/`.pkl` for neural networks

### Configuration
- **YAML** configuration via `config/default.yaml`
- Centralized configuration loading in `src/config/`

---

## Directory Structure

```
knowing-master/
├── config/           # Configuration files
├── data/             # Data storage (gitignored)
├── models/           # Trained models (gitignored)
├── cache/            # Scraped data cache
├── src/              # Source code
│   ├── config/       # Configuration handling
│   ├── data/         # Data scrapers
│   ├── models/       # ML models
│   ├── pipeline/     # Training & prediction pipelines
│   ├── preprocessing/# Feature engineering
│   ├── query/        # Probability query CLI
│   ├── simulation/   # Monte Carlo simulation
│   └── utils/        # Utilities
├── tests/            # Unit tests
└── *.py              # Entry point scripts
```

---

## GPU Support

### CUDA Configuration
- PyTorch installed with CUDA 11.8+ support
- Automatic CPU fallback when GPU unavailable
- Optional but recommended for high-volume simulations

### GPU-Accelerated Operations
- CatBoost training (GPU mode)
- PyTorch inference (CUDA tensors)
- Monte Carlo simulation vectorization

---

## Performance Characteristics

| Hardware | Sims/Game | Time/Game |
|----------|-----------|-----------|
| RTX 4090 | 10,000 | 0.3s |
| RTX 3080 | 10,000 | 0.5s |
| CPU (16-core) | 1,000 | 2.1s |

---

## Dependencies

All dependencies are specified in `requirements.txt`. Key packages:

```
pytorch>=2.0
catboost>=1.2
scikit-learn>=1.8
lightgbm>=4.6
pandas>=2.3
numpy>=2.3
scipy>=1.16
nba_api>=1.11
beautifulsoup4>=4.14
plotly>=6.5
matplotlib>=3.10
transformers>=4.57
pytest>=9.0
coverage>=7.13
```

Install with:
```bash
pip install -r requirements.txt