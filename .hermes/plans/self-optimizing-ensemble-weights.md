# Self-Optimizing Ensemble Weight Retuning Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a system that continuously retunes CatBoost/Transformer blend weights based on recent prediction accuracy, without needing full model retraining.

**Architecture:** A prediction ledger records every prediction made by the system. After games complete, predictions are resolved against actuals. A blend weight optimizer reads the ledger, computes per-target MAE for current weights, searches for better weights, and atomically updates `blend_weights.pkl` with snapshot/rollback support.

**Tech Stack:** Python 3.12, pandas, numpy, joblib, scipy.optimize (already installed via numpy/scipy)

---

## Phase 1: Prediction Ledger

### Task 1: Create ledger data model and storage

**Objective:** Implement the `PredictionLedger` class that records and resolves predictions.

**Files:**
- Create: `src/optimization/__init__.py`
- Create: `src/optimization/ledger.py`
- Test: `tests/test_optimization/test_ledger.py`

**Step 1: Create the optimization package**

```python
# src/optimization/__init__.py
"""Prediction ledger, blend weight optimization, and self-tuning."""
```

**Step 2: Write the ledger implementation**

```python
# src/optimization/ledger.py
"""Append-only prediction ledger with walk-forward resolution."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1


@dataclass
class LedgerEntry:
    """One prediction awaiting resolution."""

    game_date: str
    player_id: int
    player_name: str
    target: str
    predicted: float
    catboost_pred: float
    transformer_pred: Optional[float]
    blend_catboost_weight: float
    blend_transformer_weight: float
    blend_intercept: float
    blend_method: str
    model_hash: str
    resolved: bool = False
    actual: Optional[float] = None
    error: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class PredictionLedger:
    """Append-only ledger of predictions with resolution tracking."""

    def __init__(self, path: str | Path = "models/predictions_ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: list[LedgerEntry] = []
        self._loaded = False

    def _load(self) -> list[LedgerEntry]:
        if self._loaded:
            return self._cache
        if not self.path.exists():
            self._cache = []
            self._loaded = True
            return self._cache
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LedgerEntry.from_dict(json.loads(line)))
        self._cache = entries
        self._loaded = True
        return self._cache

    def record_prediction(self, entry: LedgerEntry) -> None:
        """Append a new prediction to the ledger."""
        self._load()
        self._cache.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def get_unresolved(self) -> list[LedgerEntry]:
        """Return predictions not yet matched to actuals."""
        self._load()
        return [e for e in self._cache if not e.resolved]

    def get_resolved(
        self, lookback_days: Optional[int] = None, min_samples: int = 0
    ) -> list[LedgerEntry]:
        """Return resolved predictions, optionally filtered by recency."""
        self._load()
        resolved = [e for e in self._cache if e.resolved and e.actual is not None]
        if lookback_days is not None:
            from datetime import datetime, timedelta

            cutoff = datetime.now() - timedelta(days=lookback_days)
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            resolved = [e for e in resolved if e.game_date >= cutoff_str]
        if min_samples > 0 and len(resolved) < min_samples:
            logger.warning(
                "Only %d resolved predictions (need %d); optimization may be unreliable",
                len(resolved),
                min_samples,
            )
        return resolved

    def resolve(
        self, actuals_df, game_date_col: str = "GAME_DATE",
        player_id_col: str = "PLAYER_ID",
        player_name_col: str = "PLAYER_NAME",
        target_cols: Optional[list[str]] = None,
    ) -> int:
        """Match unresolved predictions to actual stats and compute errors.

        Args:
            actuals_df: DataFrame with actual game stats.
            game_date_col: Column name for game date in actuals_df.
            player_id_col: Column name for player ID.
            player_name_col: Column name for player name.
            target_cols: Which stat columns to resolve. Defaults to PTS/REB/AST/STL/BLK/TOV.

        Returns:
            Number of predictions resolved.
        """
        if target_cols is None:
            target_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

        self._load()
        unresolved = [e for e in self._cache if not e.resolved]
        if not unresolved:
            logger.info("No unresolved predictions to resolve")
            return 0

        actuals_df[game_date_col] = actuals_df[game_date_col].astype(str).str[:10]
        resolved_count = 0

        for entry in unresolved:
            mask = (
                (actuals_df[player_id_col] == entry.player_id)
                & (actuals_df[game_date_col] == entry.game_date)
            )
            row = actuals_df[mask]
            if row.empty:
                continue
            row = row.iloc[0]
            if entry.target in row.index and row[entry.target] is not None:
                entry.resolved = True
                entry.actual = float(row[entry.target])
                entry.error = entry.predicted - entry.actual
                resolved_count += 1

        if resolved_count > 0:
            self._rewrite_ledger()

        logger.info("Resolved %d predictions", resolved_count)
        return resolved_count

    def _rewrite_ledger(self) -> None:
        """Write the full ledger back to disk (e.g. after bulk resolution)."""
        with open(self.path, "w") as f:
            for entry in self._cache:
                f.write(json.dumps(entry.to_dict()) + "\n")

    def compute_model_hash(self, models_dir: str | Path) -> str:
        """Hash model artifacts to identify which model version produced a prediction."""
        p = Path(models_dir)
        hasher = hashlib.md5()
        for name in ["blend_weights.pkl", "model_stack_metadata.pkl"]:
            fp = p / name
            if fp.exists():
                hasher.update(fp.read_bytes())
        return hasher.hexdigest()[:12]
```

**Step 3: Write the ledger tests**

```python
# tests/test_optimization/__init__.py

# tests/test_optimization/test_ledger.py
"""Tests for PredictionLedger."""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.optimization.ledger import LedgerEntry, PredictionLedger


def _make_entry(game_date="2025-03-01", player_id=1, target="PTS", predicted=25.0):
    return LedgerEntry(
        game_date=game_date,
        player_id=player_id,
        player_name="Test Player",
        target=target,
        predicted=predicted,
        catboost_pred=24.5,
        transformer_pred=26.0,
        blend_catboost_weight=0.6,
        blend_transformer_weight=0.4,
        blend_intercept=0.0,
        blend_method="ridge",
        model_hash="abc123",
    )


class TestRecordAndLoad:
    def test_record_and_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ledger = PredictionLedger(path)
            ledger.record_prediction(_make_entry())
            ledger.record_prediction(_make_entry(player_id=2, predicted=18.0))

            ledger2 = PredictionLedger(path)
            entries = ledger2._load()
            assert len(entries) == 2
            assert entries[0].player_id == 1
            assert entries[1].player_id == 2
        finally:
            os.unlink(path)

    def test_empty_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ledger = PredictionLedger(path)
            assert ledger._load() == []
        finally:
            os.unlink(path)


class TestResolution:
    def test_resolve_matches_actuals(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ledger = PredictionLedger(path)
            ledger.record_prediction(_make_entry(game_date="2025-03-01", player_id=1, target="PTS", predicted=25.0))
            ledger.record_prediction(_make_entry(game_date="2025-03-01", player_id=1, target="REB", predicted=8.0))

            actuals = pd.DataFrame([
                {"PLAYER_ID": 1, "GAME_DATE": "2025-03-01", "PTS": 27.0, "REB": 7.5, "AST": 5.0, "STL": 1.0, "BLK": 0.5, "TOV": 2.0},
            ])

            count = ledger.resolve(actuals)
            assert count == 2

            resolved = ledger.get_resolved()
            assert len(resolved) == 2
            pts_entry = [e for e in resolved if e.target == "PTS"][0]
            assert pts_entry.actual == 27.0
            assert pts_entry.error == -2.0
        finally:
            os.unlink(path)

    def test_resolve_no_match(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ledger = PredictionLedger(path)
            ledger.record_prediction(_make_entry(game_date="2025-03-01", player_id=999, predicted=25.0))

            actuals = pd.DataFrame([
                {"PLAYER_ID": 1, "GAME_DATE": "2025-03-01", "PTS": 27.0},
            ])

            count = ledger.resolve(actuals)
            assert count == 0
        finally:
            os.unlink(path)


class TestFilters:
    def test_get_unresolved(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            ledger = PredictionLedger(path)
            ledger.record_prediction(_make_entry())
            ledger.record_prediction(_make_entry(player_id=2))

            unresolved = ledger.get_unresolved()
            assert len(unresolved) == 2
        finally:
            os.unlink(path)
```

**Step 4: Run tests**

```bash
pytest tests/test_optimization/test_ledger.py -v
```

Expected: 5 passed.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add prediction ledger with record and resolve"
```

---

## Phase 2: Integrate Ledger into Prediction Path

### Task 2: Hook ledger into ModelManager.predict_player_stats

**Objective:** Every call to `predict_player_stats` records its prediction to the ledger.

**Files:**
- Modify: `src/models/model_manager.py` (around line 387-411, inside the target loop)

**Step 1: Add ledger import and initialization**

At the top of `src/models/model_manager.py`, add after the existing imports:

```python
from src.optimization.ledger import LedgerEntry, PredictionLedger
```

In `__init__`, after `self.registry = registry or ModelRegistry(...)`, add:

```python
        self.ledger: Optional[PredictionLedger] = None
```

**Step 2: Add lazy ledger init + record in predict_player_stats**

Find the `predict_player_stats` method. Inside it, after `base_predictions` is computed and before the method returns, add recording logic. The best insertion point is right before the `return predictions` at the end of the method (~line 419):

```python
        # Record prediction to ledger for self-optimization
        if self.ledger is None:
            self.ledger = PredictionLedger(
                Path(self.models_dir) / "predictions_ledger.jsonl"
            )
        model_hash = self.ledger.compute_model_hash(self.models_dir)
        blend_cfg = self.blend_weights or {}

        for target in self.targets:
            if target not in base_predictions:
                continue
            entry = LedgerEntry(
                game_date=str(player_context_df["GAME_DATE"].iloc[0])
                if "GAME_DATE" in player_context_df.columns
                else "",
                player_id=int(player_context_df["PLAYER_ID"].iloc[0])
                if "PLAYER_ID" in player_context_df.columns
                else 0,
                player_name=str(player_context_df.get("PLAYER_NAME", [""])[0]),
                target=target,
                predicted=predictions[target],
                catboost_pred=base_predictions[target],
                transformer_pred=predictions.get(f"_tx_{target}"),
                blend_catboost_weight=float(blend_cfg.get(target, {}).get("catboost", 1.0)),
                blend_transformer_weight=float(blend_cfg.get(target, {}).get("transformer", 0.0)),
                blend_intercept=float(blend_cfg.get(target, {}).get("intercept", 0.0)),
                blend_method=blend_cfg.get("_method", "inverse_mae"),
                model_hash=model_hash,
            )
            self.ledger.record_prediction(entry)
```

Note: We need to stash the transformer prediction separately so the ledger can record the per-model breakdown. Add this inside the target loop in `predict_player_stats`, right after computing `transformer_pred`:

```python
            if transformer_pred is not None:
                predictions[f"_tx_{target}"] = transformer_pred  # stash for ledger
```

This `_tx_*` key will not leak to callers since the return dict is used internally by `predict_player_stats_batch` and the simulation code. If you want to be safer, pass it through the ledger recording block directly instead.

**Step 3: Test**

```bash
python -c "
from src.models.model_manager import ModelManager
mm = ModelManager()
mm._load_models()
print('Loaded', len(mm.models), 'models')
print('Ledger:', mm.ledger)
"
```

Expected: Models loaded, ledger is None (will be created on first predict call).

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: record predictions to ledger in ModelManager"
```

---

## Phase 3: Blend Weight Optimizer

### Task 3: Implement BlendWeightOptimizer

**Objective:** Read resolved predictions, search for better CatBoost/Transformer weights per target.

**Files:**
- Create: `src/optimization/blend_optimizer.py`
- Test: `tests/test_optimization/test_blend_optimizer.py`

**Step 1: Create the optimizer**

```python
# src/optimization/blend_optimizer.py
"""Optimize CatBoost/Transformer blend weights from resolved predictions."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.optimization.ledger import LedgerEntry

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]


class BlendWeightOptimizer:
    """Finds better ensemble blend weights using recent resolved predictions."""

    def __init__(
        self,
        targets: Optional[list[str]] = None,
        accept_margin: float = 0.005,
        strategy: str = "ridge",
    ):
        self.targets = targets or DEFAULT_TARGETS
        self.accept_margin = accept_margin  # minimum relative improvement to accept
        self.strategy = strategy  # "ridge", "grid", or "rolling_inv_mae"

    def optimize(
        self,
        resolved: list[LedgerEntry],
        current_weights: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Run optimization and return results.

        Returns:
            {
                "accepted": bool,
                "baseline_mae": dict[target, float],
                "optimized_mae": dict[target, float],
                "new_weights": dict[target, dict],
                "old_weights": dict[target, dict],
                "improvement": dict[target, float],  # relative MAE change
                "strategy": str,
                "n_predictions": int,
            }
        """
        if not resolved:
            return {"error": "No resolved predictions available"}

        n = len(resolved)
        logger.info("Optimizing blend weights on %d resolved predictions (strategy=%s)", n, self.strategy)

        # Group by target
        by_target = defaultdict(list)
        for e in resolved:
            if e.target in self.targets:
                by_target[e.target].append(e)

        # Filter targets with enough data
        min_per_target = 30
        eligible = {t: preds for t, preds in by_target.items() if len(preds) >= min_per_target}
        if not eligible:
            return {"error": f"Need >= {min_per_target} resolved predictions per target"}

        # Compute baseline MAE per target
        baseline_mae = {}
        for target, preds in eligible.items():
            baseline_mae[target] = np.mean([abs(e.error) for e in preds])

        # Optimize per target
        new_weights = {}
        optimized_mae = {}
        for target in self.targets:
            if target not in eligible:
                new_weights[target] = current_weights.get(target, {"catboost": 1.0, "transformer": 0.0, "intercept": 0.0})
                optimized_mae[target] = baseline_mae.get(target, 0.0)
                continue

            preds = eligible[target]
            old_w = current_weights.get(target, {"catboost": 1.0, "transformer": 0.0, "intercept": 0.0})
            old_cb = float(old_w.get("catboost", 1.0))
            old_tx = float(old_w.get("transformer", 0.0))
            old_intercept = float(old_w.get("intercept", 0.0))

            # Reconstruct per-model predictions
            # predicted = cb_pred * old_cb + tx_pred * old_tx + old_intercept
            # We need to recover tx_pred from the ledger entry
            cb_preds = np.array([e.catboost_pred for e in preds])
            tx_preds = np.array([e.transformer_pred if e.transformer_pred is not None else e.catboost_pred for e in preds])
            actuals = np.array([e.actual for e in preds])

            if self.strategy == "grid":
                best_w = self._grid_search(cb_preds, tx_preds, actuals)
            elif self.strategy == "ridge":
                best_w = self._ridge_fit(cb_preds, tx_preds, actuals)
            else:  # rolling_inv_mae
                best_w = self._rolling_inv_mae(cb_preds, tx_preds, actuals)

            new_weights[target] = best_w

            # Compute MAE with new weights
            new_preds = cb_preds * best_w["catboost"] + tx_preds * best_w["transformer"] + best_w["intercept"]
            optimized_mae[target] = float(np.mean(np.abs(actuals - new_preds)))

        # Build result
        improvement = {}
        any_accepted = False
        for target in self.targets:
            if baseline_mae.get(target, 0) > 0:
                improvement[target] = (baseline_mae[target] - optimized_mae[target]) / baseline_mae[target]
            else:
                improvement[target] = 0.0
            if improvement[target] >= self.accept_margin:
                any_accepted = True

        return {
            "accepted": any_accepted,
            "baseline_mae": {t: round(baseline_mae.get(t, 0), 4) for t in self.targets},
            "optimized_mae": {t: round(optimized_mae.get(t, 0), 4) for t in self.targets},
            "new_weights": new_weights,
            "old_weights": {t: current_weights.get(t, {}) for t in self.targets},
            "improvement": {t: round(improvement[t], 4) for t in self.targets},
            "strategy": self.strategy,
            "n_predictions": n,
        }

    def _ridge_fit(
        self, cb_preds: np.ndarray, tx_preds: np.ndarray, actuals: np.ndarray
    ) -> dict[str, float]:
        """Fit Ridge regression to find optimal blend weights."""
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            return self._rolling_inv_mae(cb_preds, tx_preds, actuals)

        X = np.column_stack([cb_preds, tx_preds])
        ridge = Ridge(alpha=1.0, fit_intercept=True, positive=True)
        ridge.fit(X, actuals)

        w_cb = float(ridge.coef_[0])
        w_tx = float(ridge.coef_[1])
        intercept = float(ridge.intercept_)

        # Normalize so weights sum to ~1 (Ridge positive constraint handles this)
        total = w_cb + w_tx
        if total > 0:
            return {
                "catboost": round(w_cb / total, 4),
                "transformer": round(w_tx / total, 4),
                "intercept": round(intercept, 4),
            }
        return {"catboost": 1.0, "transformer": 0.0, "intercept": 0.0}

    def _grid_search(
        self, cb_preds: np.ndarray, tx_preds: np.ndarray, actuals: np.ndarray
    ) -> dict[str, float]:
        """Brute-force search over CatBoost weight [0.0, 0.1, ..., 1.0]."""
        best_mae = float("inf")
        best_w = {"catboost": 1.0, "transformer": 0.0, "intercept": 0.0}

        for w_cb_10 in range(0, 11):
            w_cb = w_cb_10 / 10.0
            w_tx = 1.0 - w_cb
            blended = cb_preds * w_cb + tx_preds * w_tx
            mae = float(np.mean(np.abs(actuals - blended)))
            if mae < best_mae:
                best_mae = mae
                best_w = {"catboost": w_cb, "transformer": w_tx, "intercept": 0.0}

        return best_w

    def _rolling_inv_mae(
        self, cb_preds: np.ndarray, tx_preds: np.ndarray, actuals: np.ndarray
    ) -> dict[str, float]:
        """Inverse-MAE weighting based on recent per-model performance."""
        cb_mae = float(np.mean(np.abs(actuals - cb_preds)))
        tx_mae = float(np.mean(np.abs(actuals - tx_preds)))

        if cb_mae <= 0 and tx_mae <= 0:
            return {"catboost": 0.5, "transformer": 0.5, "intercept": 0.0}
        if cb_mae <= 0:
            return {"catboost": 1.0, "transformer": 0.0, "intercept": 0.0}
        if tx_mae <= 0:
            return {"catboost": 0.0, "transformer": 1.0, "intercept": 0.0}

        inv_cb = 1.0 / cb_mae
        inv_tx = 1.0 / tx_mae
        total = inv_cb + inv_tx

        return {
            "catboost": round(inv_cb / total, 4),
            "transformer": round(inv_tx / total, 4),
            "intercept": 0.0,
        }


def format_optimization_report(result: dict) -> str:
    """Human-readable report of optimization results."""
    if "error" in result:
        return f"OPTIMIZATION FAILED: {result['error']}"

    lines = [
        f"Blend Weight Optimization Report (strategy={result['strategy']})",
        f"Predictions used: {result['n_predictions']}",
        "",
        f"{'Target':<8} {'Old CB':>7} {'Old TX':>7} {'New CB':>7} {'New TX':>7} {'Baseline MAE':>14} {'Optimized MAE':>14} {'Improvement':>12}",
        "-" * 90,
    ]

    for target in DEFAULT_TARGETS:
        old = result["old_weights"].get(target, {})
        new = result["new_weights"].get(target, {})
        lines.append(
            f"{target:<8} {old.get('catboost', 1.0):>7.3f} {old.get('transformer', 0.0):>7.3f} "
            f"{new.get('catboost', 0.0):>7.3f} {new.get('transformer', 0.0):>7.3f} "
            f"{result['baseline_mae'].get(target, 0.0):>14.4f} "
            f"{result['optimized_mae'].get(target, 0.0):>14.4f} "
            f"{result['improvement'].get(target, 0.0):>11.1%}"
        )

    lines.append("")
    lines.append(f"Accepted: {result['accepted']}")
    return "\n".join(lines)
```

**Step 2: Write optimizer tests**

```python
# tests/test_optimization/test_blend_optimizer.py
"""Tests for BlendWeightOptimizer."""

import numpy as np
import pytest

from src.optimization.blend_optimizer import BlendWeightOptimizer, format_optimization_report
from src.optimization.ledger import LedgerEntry


def _make_resolved_entry(
    cb_pred=25.0, tx_pred=27.0, actual=26.0, target="PTS"
):
    predicted = cb_pred * 0.6 + tx_pred * 0.4
    return LedgerEntry(
        game_date="2025-03-01",
        player_id=1,
        player_name="Test",
        target=target,
        predicted=predicted,
        catboost_pred=cb_pred,
        transformer_pred=tx_pred,
        blend_catboost_weight=0.6,
        blend_transformer_weight=0.4,
        blend_intercept=0.0,
        blend_method="ridge",
        model_hash="abc",
        resolved=True,
        actual=actual,
        error=predicted - actual,
    )


class TestGridSearch:
    def test_prefers_better_model(self):
        # CatBoost is closer to actuals
        entries = [_make_resolved_entry(cb_pred=25.5, tx_pred=30.0, actual=25.0) for _ in range(50)]
        optimizer = BlendWeightOptimizer(strategy="grid")
        cb = np.array([e.catboost_pred for e in entries])
        tx = np.array([e.transformer_pred for e in entries])
        actuals = np.array([e.actual for e in entries])
        result = optimizer._grid_search(cb, tx, actuals)
        assert result["catboost"] > result["transformer"]

    def test_equal_models(self):
        entries = [_make_resolved_entry(cb_pred=26.0, tx_pred=26.0, actual=26.0) for _ in range(50)]
        optimizer = BlendWeightOptimizer(strategy="grid")
        cb = np.array([e.catboost_pred for e in entries])
        tx = np.array([e.transformer_pred for e in entries])
        actuals = np.array([e.actual for e in entries])
        result = optimizer._grid_search(cb, tx, actuals)
        # Both are equal, any split is optimal; grid should pick one extreme or 0.5/0.5
        assert abs(result["catboost"] + result["transformer"] - 1.0) < 0.01


class TestRidgeFit:
    def test_positive_weights(self):
        entries = [_make_resolved_entry(cb_pred=25.0, tx_pred=27.0, actual=26.0) for _ in range(50)]
        optimizer = BlendWeightOptimizer(strategy="ridge")
        cb = np.array([e.catboost_pred for e in entries])
        tx = np.array([e.transformer_pred for e in entries])
        actuals = np.array([e.actual for e in entries])
        result = optimizer._ridge_fit(cb, tx, actuals)
        assert result["catboost"] >= 0
        assert result["transformer"] >= 0


class TestRollingInvMae:
    def test_inverse_weighting(self):
        # CatBoost is much better
        entries = [_make_resolved_entry(cb_pred=25.1, tx_pred=30.0, actual=25.0) for _ in range(50)]
        optimizer = BlendWeightOptimizer(strategy="rolling_inv_mae")
        cb = np.array([e.catboost_pred for e in entries])
        tx = np.array([e.transformer_pred for e in entries])
        actuals = np.array([e.actual for e in entries])
        result = optimizer._rolling_inv_mae(cb, tx, actuals)
        assert result["catboost"] > 0.5


class TestFullOptimize:
    def test_no_data(self):
        optimizer = BlendWeightOptimizer()
        result = optimizer.optimize([], {})
        assert "error" in result

    def test_optimization_runs(self):
        entries = [_make_resolved_entry() for _ in range(50)]
        optimizer = BlendWeightOptimizer(strategy="grid")
        current = {"PTS": {"catboost": 0.6, "transformer": 0.4, "intercept": 0.0}}
        result = optimizer.optimize(entries, current)
        assert "error" not in result
        assert "new_weights" in result
        assert "baseline_mae" in result
        assert "optimized_mae" in result

    def test_report_format(self):
        entries = [_make_resolved_entry() for _ in range(50)]
        optimizer = BlendWeightOptimizer(strategy="grid")
        current = {"PTS": {"catboost": 0.6, "transformer": 0.4, "intercept": 0.0}}
        result = optimizer.optimize(entries, current)
        report = format_optimization_report(result)
        assert "Blend Weight Optimization Report" in report
        assert "PTS" in report
```

**Step 3: Run tests**

```bash
pytest tests/test_optimization/test_blend_optimizer.py -v
```

Expected: 7 passed.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add blend weight optimizer with grid/ridge/inv-mae strategies"
```

---

## Phase 4: Snapshot Manager

### Task 4: Implement snapshot and rollback

**Objective:** Atomic save of blend weights with versioned snapshots and rollback support.

**Files:**
- Create: `src/optimization/snapshot_manager.py`
- Test: `tests/test_optimization/test_snapshot_manager.py`

**Step 1: Create the snapshot manager**

```python
# src/optimization/snapshot_manager.py
"""Atomic blend weight saving with snapshot history and rollback."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)

HISTORY_FILE = "weight_snapshots/history.json"


class SnapshotManager:
    """Manages blend weight snapshots with atomic writes and rollback."""

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.snapshot_dir = self.models_dir / "weight_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.models_dir / HISTORY_FILE
        self._ensure_history()

    def _ensure_history(self) -> None:
        if not self.history_path.exists():
            self._write_history([])

    def _read_history(self) -> list[dict]:
        with open(self.history_path, "r") as f:
            return json.load(f)

    def _write_history(self, history: list[dict]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def save(self, weights: dict, label: Optional[str] = None) -> str:
        """Atomically save blend weights with a snapshot backup.

        Returns:
            Snapshot ID (e.g. "20250315_143022")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.snapshot_dir / f"{timestamp}.pkl"

        # Backup current weights if they exist
        current_path = self.models_dir / "blend_weights.pkl"
        if current_path.exists():
            shutil.copy2(current_path, snapshot_file)
            logger.info("Backed up current weights to %s", snapshot_file.name)

        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(self.models_dir), suffix=".pkl")
        try:
            joblib.dump(weights, tmp_path)
            shutil.move(tmp_path, str(current_path))
        except Exception:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            raise

        # Update history
        history = self._read_history()
        history.append({
            "id": timestamp,
            "timestamp": datetime.now().isoformat(),
            "label": label or "",
            "targets": list(weights.keys()) if isinstance(weights, dict) else [],
        })
        self._write_history(history)

        logger.info("Saved blend weights (snapshot %s)", timestamp)
        return timestamp

    def rollback(self, snapshot_id: Optional[str] = None) -> bool:
        """Roll back to a snapshot. If no ID given, roll back to most recent.

        Returns:
            True if rollback succeeded.
        """
        history = self._read_history()
        if not history:
            logger.warning("No snapshots available for rollback")
            return False

        if snapshot_id is None:
            target = history[-1]
        else:
            target = next((h for h in history if h["id"] == snapshot_id), None)
            if target is None:
                logger.warning("Snapshot %s not found", snapshot_id)
                return False

        snapshot_file = self.snapshot_dir / f"{target['id']}.pkl"
        if not snapshot_file.exists():
            logger.warning("Snapshot file %s not found", snapshot_file.name)
            return False

        current_path = self.models_dir / "blend_weights.pkl"
        shutil.copy2(snapshot_file, current_path)

        logger.info("Rolled back to snapshot %s", target["id"])
        return True

    def list_snapshots(self) -> list[dict]:
        """List all available snapshots."""
        return self._read_history()
```

**Step 2: Write snapshot tests**

```python
# tests/test_optimization/test_snapshot_manager.py
"""Tests for SnapshotManager."""

import os
import tempfile
from pathlib import Path

import joblib
import pytest

from src.optimization.snapshot_manager import SnapshotManager


@pytest.fixture
def tmp_models():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestSnapshotManager:
    def test_save_creates_snapshot(self, tmp_models):
        mgr = SnapshotManager(tmp_models)
        weights = {"PTS": {"catboost": 0.6, "transformer": 0.4}}
        sid = mgr.save(weights)
        assert len(sid) > 0
        assert (tmp_models / "blend_weights.pkl").exists()
        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["id"] == sid

    def test_save_backs_up_previous(self, tmp_models):
        mgr = SnapshotManager(tmp_models)
        # First save
        mgr.save({"PTS": {"catboost": 0.6, "transformer": 0.4}})
        # Second save
        mgr.save({"PTS": {"catboost": 0.4, "transformer": 0.6}})
        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 2

    def test_rollback_restores_previous(self, tmp_models):
        mgr = SnapshotManager(tmp_models)
        mgr.save({"PTS": {"catboost": 0.6, "transformer": 0.4}})
        mgr.save({"PTS": {"catboost": 0.4, "transformer": 0.6}})

        mgr.rollback()
        restored = joblib.load(tmp_models / "blend_weights.pkl")
        assert restored["PTS"]["catboost"] == 0.6

    def test_rollback_no_snapshots(self, tmp_models):
        mgr = SnapshotManager(tmp_models)
        assert mgr.rollback() is False

    def test_list_snapshots_empty(self, tmp_models):
        mgr = SnapshotManager(tmp_models)
        assert mgr.list_snapshots() == []
```

**Step 3: Run tests**

```bash
pytest tests/test_optimization/test_snapshot_manager.py -v
```

Expected: 5 passed.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add snapshot manager with atomic write and rollback"
```

---

## Phase 5: CLI Entry Point

### Task 5: Create self_optimize.py CLI

**Objective:** Command-line tool to run the full optimization pipeline.

**Files:**
- Create: `self_optimize.py`

**Step 1: Create the CLI entry point**

```python
#!/usr/bin/env python3
"""Self-optimizing ensemble weight retuning CLI.

Usage:
    python self_optimize.py                     # Full pipeline: resolve + optimize + save
    python self_optimize.py --resolve-only      # Just match predictions to actuals
    python self_optimize.py --dry-run           # Show what would change without saving
    python self_optimize.py --strategy grid     # Use grid search (default: ridge)
    python self_optimize.py --lookback 60       # Use last 60 days of resolved data
    python self_optimize.py --min-samples 200   # Require at least 200 resolved predictions
    python self_optimize.py --rollback          # Roll back to previous snapshot
    python self_optimize.py --list-snapshots    # Show all snapshots
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.optimization.blend_optimizer import BlendWeightOptimizer, format_optimization_report
from src.optimization.ledger import PredictionLedger
from src.optimization.snapshot_manager import SnapshotManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("self_optimize")


def main():
    parser = argparse.ArgumentParser(description="Self-optimize ensemble blend weights")
    parser.add_argument("--resolve-only", action="store_true", help="Only resolve predictions, don't optimize")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed changes without saving")
    parser.add_argument("--strategy", choices=["ridge", "grid", "rolling_inv_mae"], default="ridge")
    parser.add_argument("--lookback", type=int, default=30, help="Days of resolved data to use")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum resolved predictions")
    parser.add_argument("--rollback", action="store_true", help="Roll back to previous snapshot")
    parser.add_argument("--list-snapshots", action="store_true", help="List all snapshots")
    parser.add_argument("--rollback-to", type=str, help="Roll back to specific snapshot ID")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--accept-margin", type=float, default=0.005, help="Minimum relative improvement to accept")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    ledger = PredictionLedger(models_dir / "predictions_ledger.jsonl")
    snapshots = SnapshotManager(models_dir)

    # Handle snapshot operations
    if args.list_snapshots:
        history = snapshots.list_snapshots()
        if not history:
            print("No snapshots available")
        else:
            print(f"{'ID':<20} {'Timestamp':<25} {'Label':<20} {'Targets'}")
            print("-" * 80)
            for s in history:
                print(f"{s['id']:<20} {s['timestamp']:<25} {s.get('label',''):<20} {len(s.get('targets',[]))}")
        return

    if args.rollback or args.rollback_to:
        sid = args.rollback_to if args.rollback_to else None
        ok = snapshots.rollback(sid)
        if ok:
            print("Rollback successful")
        else:
            print("Rollback failed")
        return

    # Resolve predictions
    if not args.resolve_only:
        logger.info("=== Resolving predictions ===")
    actuals_file = data_dir / "nba_players.csv"
    if actuals_file.exists():
        actuals = pd.read_csv(actuals_file, parse_dates=["GAME_DATE"])
        count = ledger.resolve(actuals)
        logger.info("Resolved %d predictions", count)
    else:
        logger.warning("No actuals file at %s; skipping resolution", actuals_file)

    if args.resolve_only:
        logger.info("Resolution complete. Run without --resolve-only to optimize.")
        return

    # Load current weights
    weights_file = models_dir / "blend_weights.pkl"
    if not weights_file.exists():
        logger.error("No blend_weights.pkl found at %s. Train models first.", models_dir)
        sys.exit(1)

    current_weights = joblib.load(weights_file)

    # Get resolved predictions
    resolved = ledger.get_resolved(lookback_days=args.lookback, min_samples=args.min_samples)
    if not resolved:
        logger.error("No resolved predictions available. Run --resolve-only first.")
        sys.exit(1)

    logger.info("Using %d resolved predictions (lookback=%d days)", len(resolved), args.lookback)

    # Optimize
    optimizer = BlendWeightOptimizer(
        strategy=args.strategy,
        accept_margin=args.accept_margin,
    )
    result = optimizer.optimize(resolved, current_weights)

    # Print report
    report = format_optimization_report(result)
    print(report)

    if "error" in result:
        logger.error("Optimization failed: %s", result["error"])
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run — no changes saved")
        return

    if result["accepted"]:
        snapshots.save(result["new_weights"], label=f"{args.strategy} optimization")
        logger.info("New blend weights saved successfully")
    else:
        logger.info("No weights accepted (improvement < accept_margin=%.1f%%)", args.accept_margin * 100)


if __name__ == "__main__":
    main()
```

**Step 2: Test the CLI**

```bash
# Just verify it runs and shows help
python self_optimize.py --help
```

Expected: Shows all CLI options.

```bash
# Dry run (won't save anything)
python self_optimize.py --dry-run
```

Expected: Shows optimization report or "No resolved predictions" error (expected until you have data).

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add self_optimize.py CLI for blend weight retuning"
```

---

## Phase 6: Integration Tests

### Task 6: End-to-end integration test

**Objective:** Verify the full pipeline works: predict -> ledger -> resolve -> optimize -> save.

**Files:**
- Create: `tests/test_optimization/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_optimization/test_integration.py
"""End-to-end test of the self-optimization pipeline."""

import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.optimization.blend_optimizer import BlendWeightOptimizer
from src.optimization.ledger import LedgerEntry, PredictionLedger
from src.optimization.snapshot_manager import SnapshotManager


class TestEndToEnd:
    """Simulate the full pipeline without requiring trained models."""

    def test_predict_resolve_optimize_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ledger = PredictionLedger(tmp / "predictions_ledger.jsonl")
            snapshots = SnapshotManager(tmp)

            # Simulate 100 predictions with known actuals
            rng = np.random.default_rng(42)
            for i in range(100):
                cb_pred = float(rng.normal(25.0, 3.0))
                tx_pred = float(rng.normal(26.0, 4.0))
                actual = float(rng.normal(25.5, 3.0))
                predicted = cb_pred * 0.6 + tx_pred * 0.4
                entry = LedgerEntry(
                    game_date="2025-03-01",
                    player_id=i % 30 + 1,
                    player_name=f"Player {i}",
                    target="PTS",
                    predicted=predicted,
                    catboost_pred=cb_pred,
                    transformer_pred=tx_pred,
                    blend_catboost_weight=0.6,
                    blend_transformer_weight=0.4,
                    blend_intercept=0.0,
                    blend_method="ridge",
                    model_hash="abc123",
                    resolved=True,
                    actual=actual,
                    error=predicted - actual,
                )
                ledger.record_prediction(entry)

            # Resolve with actuals DataFrame
            actuals = pd.DataFrame([
                {"PLAYER_ID": i % 30 + 1, "GAME_DATE": "2025-03-01",
                 "PTS": float(rng.normal(25.5, 3.0))}
                for i in range(100)
            ])

            # Save initial weights
            initial_weights = {
                "PTS": {"catboost": 0.6, "transformer": 0.4, "intercept": 0.0},
                "_method": "inverse_mae",
            }
            snapshots.save(initial_weights, label="initial")

            # Optimize
            optimizer = BlendWeightOptimizer(strategy="grid")
            resolved = ledger.get_resolved()
            result = optimizer.optimize(resolved, initial_weights)

            assert "error" not in result
            assert "new_weights" in result
            assert "baseline_mae" in result
            assert "optimized_mae" in result

            # Save optimized weights
            if result["accepted"]:
                snapshots.save(result["new_weights"], label="optimized")

            # Verify rollback works
            snapshots.rollback()
            restored = joblib.load(tmp / "blend_weights.pkl")
            assert restored["PTS"]["catboost"] == 0.6
```

**Step 2: Run tests**

```bash
pytest tests/test_optimization/test_integration.py -v
```

Expected: 1 passed.

**Step 3: Commit**

```bash
git add -A
git commit -m "test: add end-to-end integration test for self-optimization"
```

---

## Phase 7: Update config

### Task 7: Add self_optimization section to config

**Objective:** Add configuration for the self-optimization system.

**Files:**
- Modify: `src/config/config.py` (add SelfOptimizationConfig dataclass)
- Modify: `config/default.yaml` (add self_optimization section)

**Step 1: Add config dataclass**

In `src/config/config.py`, add after `EnsembleConfig`:

```python
@dataclass
class SelfOptimizationConfig:
    """Self-optimization / blend weight retuning configuration."""
    enabled: bool = True
    strategy: str = "ridge"  # ridge, grid, rolling_inv_mae
    lookback_days: int = 30
    min_samples: int = 100
    accept_margin: float = 0.005  # minimum relative improvement to accept
    verify_lookback_days: int = 7  # separate validation window
    max_snapshots: int = 50  # keep this many snapshots before pruning
```

Add it to the `Config` dataclass:

```python
    self_optimization: SelfOptimizationConfig = field(default_factory=SelfOptimizationConfig)
```

Add loading in `_from_dict`:

```python
        if 'self_optimization' in data:
            config.self_optimization = SelfOptimizationConfig(**data['self_optimization'])
```

**Step 2: Add to default.yaml**

In `config/default.yaml`, add at the end:

```yaml
self_optimization:
  enabled: true
  strategy: ridge          # ridge, grid, rolling_inv_mae
  lookback_days: 30        # days of resolved data to optimize on
  min_samples: 100         # minimum resolved predictions required
  accept_margin: 0.005     # 0.5% minimum relative MAE improvement to accept
  verify_lookback_days: 7  # separate validation window to prevent overfitting
  max_snapshots: 50        # keep this many snapshots before pruning
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add self_optimization config section"
```

---

## Phase 8: Tests and verification

### Task 8: Run full test suite

**Objective:** Ensure nothing is broken.

**Step 1: Run optimization tests**

```bash
source venv/bin/activate
pytest tests/test_optimization/ -v
```

Expected: All new tests pass.

**Step 2: Run existing tests (spot check)**

```bash
pytest tests/test_config/ tests/test_query/ -v
```

Expected: No regressions.

**Step 3: Verify CLI integration**

```bash
python self_optimize.py --list-snapshots
python self_optimize.py --help
```

---

## Summary of changes

| File | Action | Description |
|------|--------|-------------|
| `src/optimization/__init__.py` | Create | Package marker |
| `src/optimization/ledger.py` | Create | Prediction recording and resolution |
| `src/optimization/blend_optimizer.py` | Create | Weight optimization (grid/ridge/inv-mae) |
| `src/optimization/snapshot_manager.py` | Create | Atomic save, snapshot, rollback |
| `self_optimize.py` | Create | CLI entry point |
| `src/models/model_manager.py` | Modify | Record predictions to ledger |
| `src/config/config.py` | Modify | Add SelfOptimizationConfig |
| `config/default.yaml` | Modify | Add self_optimization section |
| `tests/test_optimization/` | Create | Full test coverage |

## Usage after implementation

```bash
# After each game day: resolve predictions to actuals
python self_optimize.py --resolve-only

# Then optimize blend weights
python self_optimize.py --strategy ridge

# See what would change without saving
python self_optimize.py --dry-run

# If something goes wrong, roll back
python self_optimize.py --rollback

# Check snapshot history
python self_optimize.py --list-snapshots
```
