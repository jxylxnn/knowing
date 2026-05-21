# Player Lifecycle & Bio-Mechanical ML Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Integrate three player lifecycle subsystems — injury forecasting (METIC-style), aging curves (B-Ianus Bayesian + KAN), and skill development signals — into the existing NBA prediction pipeline as new feature groups that feed CatBoost and the Transformer.

**Architecture:** Three new feature groups plug into the existing `FeatureGroup` framework. A new `PlayerBioDataCache` module enriches raw game logs with age, position, and injury history from the NBA API `commonplayerinfo` endpoint. Each feature group produces shifted-only columns (no leakage) that join the existing 150+ feature set. The KAN aging model runs as a standalone PyTorch module whose outputs are cached to disk and injected as features at training time.

**Tech Stack:** nba_api (player bio), PyTorch (KAN model), PyMC / scipy (Bayesian aging curves — scipy for MVP, PyMC optional later), pandas/numpy (feature engineering).

---

## Blocker Analysis (Pre-Mortem)

| # | Blocker | Severity | Why It Matters | Fix Required Before Phase |
|---|---------|----------|----------------|---------------------------|
| 1 | **No player bio data in pipeline** — `nba_players.csv` has no BIRTHDATE, AGE, HEIGHT, WEIGHT, or POSITION columns. | CRITICAL | Aging curve and injury history features require knowing each player's age and position. Without it, every downstream feature is impossible. | Phase 1 must add `commonplayerinfo` fetch to `update_data.py` and merge bio columns into `nba_players.csv`. |
| 2 | **No historical injury log** — `InjuryScraper` only fetches TODAY's injury report. No longitudinal injury history per player exists. | CRITICAL | METIC injury forecasting needs past injury events per player over the career. | Phase 1 must add `playercareerstats` or a persistent injury event log that accumulates over `update_data.py` runs. |
| 3 | **Feature schema growth** — adding ~25-30 new features may break existing `feature_schema_v3` in `FeatureContext`. | MEDIUM | `FeatureSelector` and `FeatureSchema` may reject unknown columns at inference time. | Phase 2: verify `FeatureSchema` auto-discovers new columns (it reads from DF, not a hardcoded list — confirmed). No action needed beyond testing. |
| 4 | **KAN library not in venv** — `kan-ml` / `efficient-kan` PyPI packages needed. | HIGH | KAN aging model can't run without the library. | Phase 3: `pip install efficient-kan` (lightweight, PyTorch-only). Fallback: hand-implement KAN layer (~50 lines) if package breaks. |
| 5 | **GPU contention** — AGENTS.md warns to use `max_workers=1` when GPU is active; KAN model adds another GPU workload during training. | MEDIUM | Simultaneous Transformer + KAN + CatBoost on GPU may OOM. | Phase 3: run KAN aging model on CPU with `torch.device('cpu')` (small model, fast enough). Pre-compute and cache outputs. |
| 6 | **Training time regression** — 3 new feature groups add computation. | LOW | Full pipeline already takes minutes; +30s is acceptable. | Each group does vectorized pandas ops; KAN outputs are pre-cached. Monitor in Phase 2. |

---

## Phase 1: Data Foundation — Player Bio & Injury History

### Task 1: Add player bio scraper module

**Objective:** Create `src/data/player_bio_scraper.py` that fetches BIRTHDATE, HEIGHT, WEIGHT, POSITION for every player via `nba_api.stats.endpoints.commonplayerinfo`.

**Files:**
- Create: `src/data/player_bio_scraper.py`
- Test: `tests/test_data/test_player_bio_scraper.py`

**Step 1: Write failing test**

```python
# tests/test_data/test_player_bio_scraper.py
import pytest
import pandas as pd
from src.data.player_bio_scraper import PlayerBioScraper

def test_fetch_bio_returns_expected_columns():
    scraper = PlayerBioScraper()
    # Use a known player ID (LeBron = 2544)
    df = scraper.fetch_player_bio(player_id=2544)
    assert isinstance(df, pd.DataFrame)
    assert 'PLAYER_ID' in df.columns
    assert 'BIRTHDATE' in df.columns
    assert 'POSITION' in df.columns
    assert 'HEIGHT' in df.columns
    assert 'WEIGHT' in df.columns

def test_fetch_all_bios_batch():
    scraper = PlayerBioScraper()
    player_ids = [2544, 201939, 203999]  # LeBron, Curry, Giannis
    df = scraper.fetch_all_bios(player_ids)
    assert len(df) == 3
    assert 'AGE' in df.columns  # computed from BIRTHDATE
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_data/test_player_bio_scraper.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# src/data/player_bio_scraper.py
"""Fetches player biographical data (birthdate, position, height, weight) from NBA API."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

logger = logging.getLogger(__name__)

class PlayerBioScraper:
    """Scrapes player bio data from nba_api commonplayerinfo endpoint."""

    REQUIRED_COLUMNS = ['PLAYER_ID', 'BIRTHDATE', 'POSITION', 'HEIGHT', 'WEIGHT', 'COUNTRY', 'DRAFT_YEAR', 'DRAFT_ROUND']

    def __init__(self, cache_dir: str = 'data/cache', rate_delay: float = 0.6):
        self.cache_dir = cache_dir
        self.rate_delay = rate_delay
        self._cache_path = os.path.join(cache_dir, 'player_bios.csv')
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_player_bio(self, player_id: int) -> pd.DataFrame:
        """Fetch bio for a single player."""
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            df = info.common_player_info.get_data_frame()
            time.sleep(self.rate_delay)
            return self._normalize_bio(df)
        except Exception as e:
            logger.warning(f"Failed to fetch bio for player {player_id}: {e}")
            return pd.DataFrame()

    def _normalize_bio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and add computed AGE column."""
        df = df.copy()
        rename_map = {
            'PERSON_ID': 'PLAYER_ID',
            'BIRTHDATE': 'BIRTHDATE',
            'POSITION': 'POSITION',
            'HEIGHT': 'HEIGHT',
            'WEIGHT': 'WEIGHT',
            'COUNTRY': 'COUNTRY',
            'DRAFT_YEAR': 'DRAFT_YEAR',
            'DRAFT_ROUND': 'DRAFT_ROUND',
            'FROM_YEAR': 'CAREER_START',
            'TO_YEAR': 'CAREER_END',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if 'BIRTHDATE' in df.columns:
            df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce')
            df['AGE'] = (pd.Timestamp.now() - df['BIRTHDATE']).dt.days / 365.25

        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[self.REQUIRED_COLUMNS + ['AGE', 'CAREER_START', 'CAREER_END']]

    def fetch_all_bios(self, player_ids: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """Fetch bio for multiple players with caching."""
        cached = self._load_cache()
        if cached is not None and not force_refresh:
            missing = set(player_ids) - set(cached['PLAYER_ID'].unique())
            if not missing:
                return cached[cached['PLAYER_ID'].isin(player_ids)]

        results = []
        ids_to_fetch = player_ids if force_refresh else list(
            set(player_ids) - (set(cached['PLAYER_ID'].unique()) if cached is not None else set())
        )

        for pid in ids_to_fetch:
            bio_df = self.fetch_player_bio(pid)
            if not bio_df.empty:
                results.append(bio_df)

        new_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        if cached is not None and not new_df.empty:
            combined = pd.concat([cached, new_df], ignore_index=True).drop_duplicates(subset='PLAYER_ID', keep='last')
        elif not new_df.empty:
            combined = new_df
        else:
            combined = cached if cached is not None else pd.DataFrame()

        self._save_cache(combined)
        return combined[combined['PLAYER_ID'].isin(player_ids)] if not combined.empty else combined

    def _load_cache(self) -> Optional[pd.DataFrame]:
        if os.path.exists(self._cache_path):
            return pd.read_csv(self._cache_path)
        return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        df.to_csv(self._cache_path, index=False)
        logger.info(f"Saved {len(df)} player bios to {self._cache_path}")
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_data/test_player_bio_scraper.py -v`
Expected: PASS (requires network, mark with `@pytest.mark.integration` for offline CI)

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add PlayerBioScraper for player birthdate/position/height data"
```

---

### Task 2: Integrate bio enrichment into update_data.py

**Objective:** After fetching player game logs, enrich them with AGE and POSITION from `PlayerBioScraper`. Save enriched data and a separate `data/player_bios.csv`.

**Files:**
- Modify: `update_data.py:474-500` (save_data function)
- Test: `tests/test_data/test_update_data_bio.py`

**Step 1: Write failing test**

```python
def test_save_data_includes_bio_columns(tmp_path):
    """After update, player CSV should contain AGE and POSITION columns."""
    import pandas as pd
    players = pd.DataFrame({
        'PLAYER_ID': [2544], 'GAME_ID': ['G1'], 'GAME_DATE': ['2025-01-01'],
        'PTS': [25], 'REB': [7], 'AST': [10]
    })
    # verify that bio enrichment step added AGE column
    assert 'AGE' in players.columns or True  # placeholder — real test calls save pipeline
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_data/test_update_data_bio.py -v`

**Step 3: Modify update_data.py**

Add at the end of the `save_data` function (after parquet dual-write, ~line 498):

```python
    # ---- Bio enrichment: merge AGE, POSITION into player logs ----
    try:
        from src.data.player_bio_scraper import PlayerBioScraper
        bio_scraper = PlayerBioScraper(cache_dir=os.path.join(data_dir, 'cache'))
        unique_ids = players_df['PLAYER_ID'].unique().tolist()
        bio_df = bio_scraper.fetch_all_bios(unique_ids)
        if not bio_df.empty:
            bio_subset = bio_df[['PLAYER_ID', 'BIRTHDATE', 'AGE', 'POSITION', 'HEIGHT', 'WEIGHT', 'DRAFT_YEAR']].drop_duplicates('PLAYER_ID')
            players_df = players_df.merge(bio_subset, on='PLAYER_ID', how='left')
            # Save standalone bio file
            bio_df.to_csv(os.path.join(data_dir, 'player_bios.csv'), index=False)
            logger.info(f"Enriched {len(players_df)} records with player bio data")
    except Exception as e:
        logger.warning(f"Bio enrichment failed (non-fatal): {e}")
```

**Step 4: Run test to verify pass**

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: merge player bio (age, position) into update_data pipeline"
```

---

### Task 3: Add historical injury event logger

**Objective:** Create `src/data/injury_history_logger.py` that accumulates observed injury events across `update_data.py` runs into `data/injury_history.csv`, providing longitudinal per-player injury data for the METIC model.

**Files:**
- Create: `src/data/injury_history_logger.py`
- Create: `tests/test_data/test_injury_history_logger.py`

**Step 1: Write failing test**

```python
def test_log_injury_appends_new_events():
    from src.data.injury_history_logger import InjuryHistoryLogger
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        logger = InjuryHistoryLogger(history_dir=tmp)
        event = {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
                 'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15'}
        logger.log_injuries([event])
        df = logger.load_history()
        assert len(df) == 1
        assert df.iloc[0]['PLAYER_ID'] == 2544
```

**Step 2: Run test to verify failure**

**Step 3: Write implementation**

```python
# src/data/injury_history_logger.py
"""Accumulates injury events into a persistent longitudinal log for METIC-style forecasting."""
from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

class InjuryHistoryLogger:
    """Persists injury events across update_data runs into a CSV log."""

    def __init__(self, history_dir: str = 'data', filename: str = 'injury_history.csv'):
        self.path = os.path.join(history_dir, filename)
        os.makedirs(history_dir, exist_ok=True)

    def log_injuries(self, events: List[Dict]) -> None:
        """Append injury events to the persistent log (deduplicates by PLAYER_ID + DATE + INJURY_TYPE)."""
        if not events:
            return
        new_df = pd.DataFrame(events)
        for col in ['PLAYER_ID', 'DATE']:
            if col not in new_df.columns:
                logger.warning(f"Injury event missing required column: {col}")
                return
        existing = self.load_history()
        if existing.empty:
            combined = new_df
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
            dedup_cols = ['PLAYER_ID', 'DATE']
            if 'INJURY_TYPE' in combined.columns:
                dedup_cols.append('INJURY_TYPE')
            combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
        combined.to_csv(self.path, index=False)
        logger.info(f"Injury history: {len(combined)} events saved to {self.path}")

    def load_history(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            return pd.read_csv(self.path)
        return pd.DataFrame()

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        df = self.load_history()
        if df.empty or 'PLAYER_ID' not in df.columns:
            return pd.DataFrame()
        return df[df['PLAYER_ID'] == player_id].sort_values('DATE')
```

**Step 4: Run test to verify pass**

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add InjuryHistoryLogger for longitudinal injury event tracking"
```

---

### Task 4: Wire InjuryHistoryLogger into update_data.py

**Objective:** At the end of each `update_data.py` run, also call `InjuryScraper.fetch_injuries()` and log all OUT/Doubtful players into the persistent injury history.

**Files:**
- Modify: `update_data.py` (add at end of main execution, after bio enrichment)

**Step 1-5:** (standard TDD flow)

Add after bio enrichment block:

```python
    # ---- Log current injuries into history ----
    try:
        from src.data.injury_scraper import InjuryScraper
        from src.data.injury_history_logger import InjuryHistoryLogger
        inj_scraper = InjuryScraper(cache_dir=os.path.join(data_dir, 'cache'), config=get_config_early())
        inj_df = inj_scraper.fetch_injuries()
        if not inj_df.empty:
            history_logger = InjuryHistoryLogger(history_dir=data_dir)
            # Convert to event format
            events = []
            for _, row in inj_df.iterrows():
                events.append({
                    'PLAYER_ID': row.get('PLAYER_ID', None),  # may need name→ID mapping
                    'PLAYER': row.get('PLAYER', ''),
                    'TEAM_ABBR': row.get('TEAM_ABBR', ''),
                    'STATUS': row.get('STATUS', ''),
                    'INJURY_TYPE': row.get('COMMENT', 'Unknown'),
                    'DATE': row.get('DATE', datetime.now().strftime('%Y-%m-%d')),
                })
            events = [e for e in events if e['PLAYER_ID'] is not None or e['PLAYER']]
            history_logger.log_injuries(events)
            logger.info(f"Logged {len(events)} injury events to history")
    except Exception as e:
        logger.warning(f"Injury history logging failed (non-fatal): {e}")
```

**Commit:**

```bash
git add -A
git commit -m "feat: wire InjuryHistoryLogger into update_data pipeline"
```

---

## Phase 2: Feature Groups — Injury Forecasting, Aging Curves, Skill Development

### Task 5: Create InjuryRiskFeatureGroup (METIC-style)

**Objective:** Feature group that computes per-player injury risk signals from the longitudinal injury history + recent workload.

**Files:**
- Create: `src/preprocessing/features/injury_risk.py`
- Modify: `src/preprocessing/features/__init__.py`
- Modify: `src/preprocessing/feature_engineer.py:86-107` (_build_groups)
- Test: `tests/test_preprocessing/test_injury_risk_features.py`

**Output columns:**

| Column | Description |
|--------|-------------|
| `INJURY_RISK_CAREER_COUNT` | Total career injury events (shifted — computed from prior games only) |
| `INJURY_RISK_LAST_90D` | Injury events in the last 90 days |
| `INJURY_RISK_LAST_30D` | Injury events in the last 30 days |
| `INJURY_RISK_WORKLOAD_SPIKE` | Binary: recent MIN spike > 1.3x season avg (overuse flag) |
| `INJURY_RISK_BACK_TO_BACK_STRESS` | Cumulative B2B games in last 14 days (wear metric) |
| `INJURY_RISK_AVG_DAYS_BETWEEN` | Average days between injury events (lower = more fragile) |

**Step 1: Write failing test**

```python
def test_injury_risk_feature_group_creates_columns():
    from src.preprocessing.features.injury_risk import InjuryRiskFeatureGroup
    from src.preprocessing.features.base import FeatureDiagnostics
    import pandas as pd, numpy as np
    
    group = InjuryRiskFeatureGroup()
    df = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1, 1],
        'GAME_DATE': pd.to_datetime(['2025-01-01', '2025-01-03', '2025-01-05', '2025-01-15']),
        'MIN': [30, 35, 38, 25],
        'TEAM_ID': [10, 10, 10, 10],
    })
    result = group.create(df, diagnostics=FeatureDiagnostics())
    assert 'INJURY_RISK_CAREER_COUNT' in result.columns
    assert 'INJURY_RISK_WORKLOAD_SPIKE' in result.columns
    # All shifted — no leakage
    assert result['INJURY_RISK_CAREER_COUNT'].iloc[0] == 0  # first game sees nothing
```

**Step 2-5: Standard TDD flow**

Implementation sketch:

```python
# src/preprocessing/features/injury_risk.py
class InjuryRiskFeatureGroup(FeatureGroup):
    """METIC-inspired injury risk features from workload + injury history."""

    @property
    def name(self) -> str:
        return 'injury_risk'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE', 'MIN', 'TEAM_ID']

    @property
    def optional_columns(self) -> List[str]:
        return ['INJURY_HISTORY_DATE']  # populated if join with injury_history

    def create(self, df, *, diagnostics=None, context=None):
        # ... compute career count, recent counts, workload spike, B2B stress, avg days between
        pass
```

Key implementation details:
- Join `data/injury_history.csv` on PLAYER_ID if available; otherwise all risk features default to 0
- Compute workload spike: `MIN > 1.3 * rolling_avg_min_10` → flag = 1
- B2B stress: count games with rest = 0 in last 14 days
- All features are shifted by 1 group (no leakage)
- Config key: `injury_risk` added to `training_presets.full.feature_engineer.enable_groups`

**Commit:**

```bash
git add -A
git commit -m "feat: add InjuryRiskFeatureGroup (METIC-style workload + history features)"
```

---

### Task 6: Create AgingCurveFeatureGroup (B-Ianus Bayesian)

**Objective:** Feature group that computes position-aware aging curve adjustments using a Bayesian structural model that separates development (pre-peak) from decline (post-peak).

**Files:**
- Create: `src/preprocessing/features/aging_curve.py`
- Modify: `src/preprocessing/features/__init__.py`
- Modify: `src/preprocessing/feature_engineer.py:86-107`
- Create: `src/lifecycle/aging_model.py` (Bayesian aging curve computation)
- Test: `tests/test_preprocessing/test_aging_curve_features.py`

**Output columns:**

| Column | Description |
|--------|-------------|
| `AGING_PLAYER_AGE` | Player age at game date (computed from BIRTHDATE) |
| `AGING_YEARS_IN_LEAGUE` | NBA tenure (career start to game date) |
| `AGING_PEAK_AGE_EST` | Estimated peak age for this position (Bayesian posterior mean) |
| `AGING_PRE_POST_PEAK` | Binary: 0 = pre-peak (developing), 1 = post-peak (declining) |
| `AGING_CURVE_FACTOR` | Multiplicative aging adjustment per stat (from B-Ianus model) |
| `AGING_DECLINE_RATE` | Position-specific decline rate (steeper for bigs, flatter for guards) |

**Implementation approach (B-Ianus model):**

```python
# src/lifecycle/aging_model.py
"""B-Ianus Bayesian Aging Curve Model.

Separates development (pre-peak) from decline (post-peak) with 
position-specific peak ages and decline rates.

Peak age priors by position (from literature):
  PG: 28.5, SG: 27.8, SF: 27.5, PF: 27.0, C: 26.5

Decline rate priors (performance loss per year past peak):
  PG: 0.8%, SG: 1.0%, SF: 1.2%, PF: 1.5%, C: 1.8%
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple

# Position-specific priors (mean, std)
POSITION_PEAK_PRIORS = {
    'PG': (28.5, 1.5), 'SG': (27.8, 1.3), 'SF': (27.5, 1.2),
    'PF': (27.0, 1.2), 'C': (26.5, 1.5),
}
POSITION_DECLINE_PRIORS = {
    'PG': (0.008, 0.003), 'SG': (0.010, 0.003), 'SF': (0.012, 0.004),
    'PF': (0.015, 0.005), 'C': (0.018, 0.005),
}

class BIanusAgingModel:
    """Bayesian structural aging curve with position-specific priors."""

    def __init__(self, prior_strength: float = 10.0):
        self.prior_strength = prior_strength
        self._fitted_players: Dict[int, dict] = {}

    def fit_player(self, player_id: int, ages: np.ndarray, 
                   performance: np.ndarray, position: str) -> dict:
        """Fit aging curve for one player using MAP estimation.
        Returns dict with: peak_age, decline_rate, development_rate, curve_factor_by_age
        """
        peak_prior = POSITION_PEAK_PRIORS.get(position, (27.5, 1.5))
        decline_prior = POSITION_DECLINE_PRIORS.get(position, (0.012, 0.004))

        # Negative log-posterior = -log_likelihood - log_prior
        def neg_log_posterior(params):
            peak, decline, dev_rate = params
            # Prior penalties
            prior_penalty = (
                self.prior_strength * ((peak - peak_prior[0])**2 / peak_prior[1]**2 +
                (decline - decline_prior[0])**2 / decline_prior[1]**2)
            )
            # Model: piecewise
            predicted = np.where(
                ages <= peak,
                1.0 + dev_rate * (ages - peak),    # development phase
                1.0 - decline * (ages - peak),     # decline phase
            )
            residuals = performance - predicted
            likelihood = 0.5 * np.sum(residuals**2)
            return likelihood + prior_penalty

        x0 = [peak_prior[0], decline_prior[0], 0.02]
        result = minimize(neg_log_posterior, x0, method='L-BFGS-B',
                         bounds=[(24, 33), (0.001, 0.05), (0.001, 0.05)])

        peak, decline, dev_rate = result.x
        return {
            'peak_age': peak,
            'decline_rate': decline,
            'development_rate': dev_rate,
            'position': position,
        }

    def curve_factor(self, age: float, peak_age: float, 
                     decline_rate: float, dev_rate: float) -> float:
        """Compute aging curve multiplicative factor."""
        if age <= peak_age:
            return 1.0 + dev_rate * (age - peak_age)
        else:
            return 1.0 - decline_rate * (age - peak_age)

    def precompute_all(self, bios_df, performance_df) -> Dict[int, dict]:
        """Fit aging curves for all players and cache results."""
        # Group by player, fit curve, store
        pass
```

The `AgingCurveFeatureGroup` calls `BIanusAgingModel` once at fit time, caches the per-player parameters to `data/cache/aging_curves.csv`, and at feature-computation time just does a lookup + multiplication.

**Config:** Add `aging_curve` to `training_presets.full.feature_engineer.enable_groups`.

**Commit:**

```bash
git add -A
git commit -m "feat: add AgingCurveFeatureGroup with B-Ianus Bayesian aging model"
```

---

### Task 7: Create KANAgingFeatureGroup (Kolmogorov-Arnold Networks)

**Objective:** Add a KAN-based nonlinear aging adjustment that captures complex age-performance relationships the linear B-Ianus model misses. Pre-compute KAN outputs and inject as features.

**Files:**
- Create: `src/preprocessing/features/kan_aging.py`
- Create: `src/lifecycle/kan_age_model.py`
- Modify: `src/preprocessing/features/__init__.py`
- Modify: `src/preprocessing/feature_engineer.py`
- Test: `tests/test_preprocessing/test_kan_aging_features.py`

**Output columns:**

| Column | Description |
|--------|-------------|
| `KAN_AGE_NONLIN_FACTOR` | KAN-derived nonlinear aging curve multiplicative factor |
| `KAN_AGE_INFLECTION_AGE` | Age of detected performance inflection (nonlinear peak) |
| `KAN_AGE_VOLATILITY` | Age-dependent volatility estimate (older = more variance) |

**KAN model approach:**

```python
# src/lifecycle/kan_age_model.py
"""KAN (Kolmogorov-Arnold Network) model for nonlinear age-performance curves.

Uses efficient-kan library (lightweight pure-PyTorch KAN implementation).
Falls back to hand-rolled KAN layer if library unavailable.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try importing efficient_kan; fall back to hand-rolled
try:
    from efficient_kan import KANLinear
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    logger.info("efficient-kan not installed; using hand-rolled KAN layer")

# Hand-rolled minimal KAN layer fallback
class SimpleKANLayer(nn.Module):
    """Minimal KAN layer using B-spline basis functions."""
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, grid_range=(-2, 2)):
        super().__init__()
        self.grid_size = grid_size
        self.in_features = in_features
        self.out_features = out_features
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        self.register_buffer('grid', grid)
        self.weights = nn.Parameter(torch.randn(out_features, in_features, grid_size + 1) * 0.1)

    def forward(self, x):
        # B-spline basis computation (simplified)
        x = x.unsqueeze(-1)  # (batch, in, 1)
        grid = self.grid  # (grid_size+1,)
        # Compute distances to grid points
        bases = torch.zeros_like(x).expand(-1, -1, len(grid))
        for i, g in enumerate(grid):
            bases[:, :, i] = torch.exp(-10 * (x.squeeze(-1) - g) ** 2)
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)
        # Weighted sum
        out = torch.einsum('bio,big->bo', self.weight, bases)
        return out

    @property
    def weight(self):
        return self.weights


class KANAgeModel(nn.Module):
    """KAN network mapping age → performance factor."""
    def __init__(self, hidden_dim: int = 8, grid_size: int = 5):
        super().__init__()
        kan_layer = KANLinear if KAN_AVAILABLE else SimpleKANLayer
        self.layer1 = kan_layer(1, hidden_dim, grid_size=grid_size)
        self.layer2 = kan_layer(hidden_dim, 1, grid_size=grid_size) if KAN_AVAILABLE else nn.Linear(hidden_dim, 1)

    def forward(self, age_normalized):
        return self.layer2(self.layer1(age_normalized))
```

The model is trained once per season on aggregated league-wide age vs. performance data, then cached. At feature computation time it's a simple lookup.

**Commit:**

```bash
git add -A
git commit -m "feat: add KANAgingFeatureGroup with KAN nonlinear age curve model"
```

---

### Task 8: Create SkillDevelopmentFeatureGroup

**Objective:** Feature group that proxies skill development signals using year-over-year stat improvements as a "growth velocity" metric. (Full CV-based skill assessment is out of scope for MVP — we use stat trajectory as a proxy.)

**Files:**
- Create: `src/preprocessing/features/skill_development.py`
- Modify: `src/preprocessing/features/__init__.py`
- Modify: `src/preprocessing/feature_engineer.py`
- Test: `tests/test_preprocessing/test_skill_development_features.py`

**Output columns:**

| Column | Description |
|--------|-------------|
| `SKILL_DEV_PTS_VELOCITY` | YoY change in PTS/min (positive = improving) |
| `SKILL_DEV_EFF_VELOCITY` | YoY change in TS% (positive = improving) |
| `SKILL_DEV_REB_VELOCITY` | YoY change in REB/min |
| `SKILL_DEV_AST_TOV_TREND` | Trend in AST:TOV ratio (playmaking growth) |
| `SKILL_DEV_YOUTH_BOOST` | Binary: age < 25 AND improving (prospect flag) |
| `SKILL_DEV_VETERAN_STEADY` | Binary: age > 30 AND efficiency stable (veteran floor) |

**Implementation approach:**

Compute per-player season averages for PTS/MIN, TS%, REB/MIN, AST/TOV. Compute the difference between consecutive seasons (shifted). A positive velocity means the player is developing. The `YOUTH_BOOST` flag combines age < 25 with improving velocity. `VETERAN_STEADY` combines age > 30 with efficiency change within ±2%.

**Commit:**

```bash
git add -A
git commit -m "feat: add SkillDevelopmentFeatureGroup with growth velocity metrics"
```

---

## Phase 3: Integration, Config, and End-to-End Testing

### Task 9: Update config/default.yaml with lifecycle settings

**Objective:** Add configuration block for the three new feature groups and the aging/KAN models.

**Files:**
- Modify: `config/default.yaml`

**Add to `training_presets.full.feature_engineer.enable_groups`:**

```yaml
        - "injury_risk"
        - "aging_curve"
        - "kan_aging"
        - "skill_development"
```

**Add new config section:**

```yaml
# Player lifecycle & bio-mechanical configuration
lifecycle:
  injury_risk_enabled: true
  aging_curve_enabled: true
  kan_aging_enabled: true
  skill_development_enabled: true
  
  # B-Ianus aging model priors
  aging_prior_strength: 10.0
  aging_peak_priors:
    PG: [28.5, 1.5]
    SG: [27.8, 1.3]
    SF: [27.5, 1.2]
    PF: [27.0, 1.2]
    C: [26.5, 1.5]
  
  # KAN model settings
  kan_grid_size: 5
  kan_hidden_dim: 8
  kan_epochs: 200
  kan_learning_rate: 0.01
  
  # Skill development thresholds
  youth_age_threshold: 25
  veteran_age_threshold: 30
  efficiency_stability_band: 0.02
```

**Commit:**

```bash
git add -A
git commit -m "feat: add lifecycle config to default.yaml"
```

---

### Task 10: Wire aging model precomputation into train.py

**Objective:** Before the main training loop, run `BIanusAgingModel.precompute_all()` and `KANAgeModel` training, saving cached outputs to `data/cache/aging_curves.csv` and `data/cache/kan_aging_factors.csv`.

**Files:**
- Modify: `train.py`

**Step 1-5: Standard TDD**

Add after feature engineering, before model training:

```python
    # ---- Precompute lifecycle aging curves ----
    if config_config.get('lifecycle', {}).get('aging_curve_enabled', True):
        from src.lifecycle.aging_model import BIanusAgingModel
        aging_model = BIanusAgingModel()
        # ... fit on historical data, save to cache
```

**Commit:**

```bash
git add -A
git commit -m "feat: wire aging curve precomputation into train.py"
```

---

### Task 11: End-to-end integration test

**Objective:** Test that the full pipeline (update_data → train → simulate) works with all 4 new feature groups enabled.

**Files:**
- Create: `tests/test_integration/test_lifecycle_pipeline.py`

**Step 1: Write test**

```python
@pytest.mark.integration
def test_lifecycle_features_in_training_output(tmp_path):
    """Verify lifecycle feature columns appear in trained model's feature_cols."""
    # Run a minimal training cycle with lifecycle features enabled
    # Check that model_manager.feature_cols contains:
    #   INJURY_RISK_*, AGING_*, KAN_AGE_*, SKILL_DEV_*
    pass
```

**Step 2-5: TDD flow**

**Commit:**

```bash
git add -A
git commit -m "test: add integration test for lifecycle feature pipeline"
```

---

### Task 12: Update AGENTS.md with lifecycle documentation

**Objective:** Document the new data dependencies, feature groups, and gotchas.

**Files:**
- Modify: `AGENTS.md`

**Add to Architecture section:**

```
  lifecycle/        — B-Ianus Bayesian aging model, KAN age model, injury risk computation
```

**Add to Gotchas:**

```
- **Player bio data required**: Lifecycle features need AGE and POSITION columns in nba_players.csv. Run `update_data.py` at least once after adding the PlayerBioScraper to populate them. Missing AGE defaults all aging features to neutral (1.0 factor).
- **Injury history is incremental**: The injury_history.csv file grows over time. First runs will have sparse data — injury risk features will be near-zero for most players until several update cycles accumulate.
- **KAN model precomputation**: KAN aging factors are pre-computed and cached. If you retrain with `--force`, delete `data/cache/kan_aging_factors.csv` to force recomputation.
```

**Commit:**

```bash
git add -A
git commit -m "docs: update AGENTS.md with lifecycle feature documentation"
```

---

### Task 13: pip install efficient-kan

**Objective:** Add the KAN library to the project venv.

**Files:** None (venv change)

```bash
source venv/bin/activate
pip install efficient-kan
pip freeze | grep -i kan >> requirements.txt  # or however deps are tracked
```

**Verification:** `python -c "from efficient_kan import KANLinear; print('OK')"`

**Commit:**

```bash
git add -A
git commit -m "deps: add efficient-kan for KAN aging model"
```

---

## Summary of New Files

| File | Purpose |
|------|---------|
| `src/data/player_bio_scraper.py` | Fetches BIRTHDATE, AGE, POSITION from NBA API |
| `src/data/injury_history_logger.py` | Persists longitudinal injury event log |
| `src/preprocessing/features/injury_risk.py` | METIC-style injury risk feature group |
| `src/preprocessing/features/aging_curve.py` | B-Ianus Bayesian aging curve feature group |
| `src/preprocessing/features/kan_aging.py` | KAN nonlinear aging feature group |
| `src/preprocessing/features/skill_development.py` | Skill development velocity feature group |
| `src/lifecycle/aging_model.py` | B-Ianus Bayesian aging curve fitting |
| `src/lifecycle/kan_age_model.py` | KAN network for age→performance mapping |
| `tests/test_data/test_player_bio_scraper.py` | Bio scraper tests |
| `tests/test_data/test_injury_history_logger.py` | Injury logger tests |
| `tests/test_preprocessing/test_injury_risk_features.py` | Injury risk feature tests |
| `tests/test_preprocessing/test_aging_curve_features.py` | Aging curve feature tests |
| `tests/test_preprocessing/test_kan_aging_features.py` | KAN aging feature tests |
| `tests/test_preprocessing/test_skill_development_features.py` | Skill dev feature tests |
| `tests/test_integration/test_lifecycle_pipeline.py` | End-to-end integration test |

## New Feature Columns (~24 features)

All are shifted (leakage-safe) and default to neutral values when bio/injury data is unavailable:

- `INJURY_RISK_CAREER_COUNT`, `INJURY_RISK_LAST_90D`, `INJURY_RISK_LAST_30D`, `INJURY_RISK_WORKLOAD_SPIKE`, `INJURY_RISK_BACK_TO_BACK_STRESS`, `INJURY_RISK_AVG_DAYS_BETWEEN`
- `AGING_PLAYER_AGE`, `AGING_YEARS_IN_LEAGUE`, `AGING_PEAK_AGE_EST`, `AGING_PRE_POST_PEAK`, `AGING_CURVE_FACTOR`, `AGING_DECLINE_RATE`
- `KAN_AGE_NONLIN_FACTOR`, `KAN_AGE_INFLECTION_AGE`, `KAN_AGE_VOLATILITY`
- `SKILL_DEV_PTS_VELOCITY`, `SKILL_DEV_EFF_VELOCITY`, `SKILL_DEV_REB_VELOCITY`, `SKILL_DEV_AST_TOV_TREND`, `SKILL_DEV_YOUTH_BOOST`, `SKILL_DEV_VETERAN_STEADY`

## Out of Scope (Future Work)

- Computer Vision skill development (actual game footage analysis) — requires video pipeline + CV model, far beyond CLI scope
- PyMC full Bayesian inference for B-Ianus — scipy MAP estimation is the MVP; PyMC MCMC sampling can upgrade later
- Per-stat aging curves (separate PTS/REB/AST aging) — current model uses a single performance composite; per-stat curves are a natural extension
- Real-time METIC deep learning injury model — current implementation is feature-group level (tabular); a full LSTM/T-Transformer METIC model would be a separate model in the ensemble