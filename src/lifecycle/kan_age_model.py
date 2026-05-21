"""KAN (Kolmogorov-Arnold Network) model for nonlinear age-performance curves.

Uses efficient-kan library (lightweight pure-PyTorch KAN implementation).
Falls back to hand-rolled KAN layer if library unavailable.
Always runs on CPU to avoid GPU contention.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try importing efficient_kan; fall back to hand-rolled
try:
    from efficient_kan import KANLinear
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    logger.info("efficient-kan not installed; using hand-rolled KAN layer")

# Lazy-import torch only when needed
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("PyTorch is required for KAN aging model")
    return _torch


# Hand-rolled minimal KAN layer fallback
class SimpleKANLayer(nn.Module):
    """Minimal KAN layer using radial basis functions.

    This is a numpy-only fallback when PyTorch or efficient-kan is unavailable.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, grid_range: tuple = (-2, 2)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        self.register_buffer('grid', grid)
        # Random init weights (will be overwritten by pre-trained or defaults)
        self.weights = nn.Parameter(torch.randn(out_features, in_features, grid_size + 1) * 0.1)

    def forward(self, x):
        # B-spline-like basis using RBF
        x_expanded = x.unsqueeze(-1)  # (batch, in, 1)
        grid = self.grid.to(x.device)  # (grid_size+1,)
        # Compute RBF basis
        diff = x_expanded - grid  # (batch, in, grid_size+1) via broadcast
        bases = torch.exp(-10 * diff ** 2)  # (batch, in, grid_size+1)
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)
        # Weighted sum
        out = torch.einsum('oig,big->bo', self.weights, bases)
        return out


class KANAgeModel:
    """KAN network mapping normalized age -> performance factor.

    Trained once per season on league-wide aggregated age vs. performance data.
    Outputs are cached to disk at data/cache/kan_aging_outputs.csv.
    """

    def __init__(self, hidden_dim: int = 8, grid_size: int = 5, device: str = 'cpu'):
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.device = device
        self.model = None

    def _build_model(self):
        nn = torch.nn

        if KAN_AVAILABLE:
            layer1 = KANLinear(1, self.hidden_dim, grid_size=self.grid_size)
            layer2 = KANLinear(self.hidden_dim, 1, grid_size=self.grid_size)
        else:
            layer1 = SimpleKANLayer(1, self.hidden_dim, grid_size=self.grid_size)
            layer2 = nn.Linear(self.hidden_dim, 1)

        class KANNet(nn.Module):
            def __init__(self, l1, l2):
                super().__init__()
                self.layer1 = l1
                self.layer2 = l2

            def forward(self, x):
                h = self.layer1(x)
                if not KAN_AVAILABLE and isinstance(self.layer2, nn.Linear):
                    # SimpleKANLayer returns raw tensor, apply activation
                    h = torch.relu(h)
                return self.layer2(h)

        self.model = KANNet(layer1, layer2).to(self.device)
        return self.model

    def train_on_age_performance(
        self,
        ages: np.ndarray,
        performance: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> dict:
        """Train KAN model on (age, performance) pairs.

        Ages are normalized to [0, 1] range (18-42).
        Returns training metrics dict.
        """

        model = self._build_model()
        model.train()

        # Normalize ages to [0, 1]
        age_min, age_max = 18.0, 42.0
        ages_norm = (ages - age_min) / (age_max - age_min + 1e-8)
        ages_t = torch.tensor(ages_norm, dtype=torch.float32).unsqueeze(1).to(self.device)
        perf_t = torch.tensor(performance, dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(ages_t)
            loss = criterion(pred, perf_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        self.model = model
        return {
            'final_loss': losses[-1],
            'epochs': epochs,
            'all_losses': losses,
        }

    def predict(self, age: float) -> dict:
        """Predict KAN output for a single age.

        Returns:
            factor: nonlinear aging curve factor
            inflection_age: age where slope changes most (derivative peak)
            volatility: age-dependent variance estimate
        """

        if self.model is None:
            return self._fallback_predict(age)

        self.model.eval()
        age_min, age_max = 18.0, 42.0
        age_norm = (age - age_min) / (age_max - age_min + 1e-8)

        with torch.no_grad():
            x = torch.tensor([[age_norm]], dtype=torch.float32).to(self.device)
            factor = float(self.model(x).item())

        # Detect inflection point: find where second derivative is largest
        inflection_age = self._find_inflection_point()

        # Volatility: older players have more variance
        # Model as increasing function past peak age (~28)
        peak_dist = abs(age - 28.0)
        volatility = 0.02 + 0.005 * peak_dist  # baseline + age-related

        return {
            'factor': factor,
            'inflection_age': inflection_age,
            'volatility': volatility,
        }

    def _find_inflection_point(self) -> float:
        """Find age where the KAN curve's second derivative is largest."""

        if self.model is None:
            return 28.0

        self.model.eval()
        age_min, age_max = 18.0, 42.0
        test_ages = np.linspace(0, 1, 50)

        first_derivatives = []
        with torch.no_grad():
            for i in range(len(test_ages) - 1):
                x1 = torch.tensor([[test_ages[i]]], dtype=torch.float32).to(self.device)
                x2 = torch.tensor([[test_ages[i + 1]]], dtype=torch.float32).to(self.device)
                y1 = self.model(x1).item()
                y2 = self.model(x2).item()
                first_derivatives.append(y2 - y1)

        # Second derivative = diff of first derivatives
        if len(first_derivatives) >= 2:
            second_derivs = [abs(first_derivatives[i + 1] - first_derivatives[i])
                             for i in range(len(first_derivatives) - 1)]
            max_idx = int(np.argmax(second_derivs))
            inflection_norm = test_ages[max_idx + 1]
            return inflection_norm * (age_max - age_min) + age_min
        return 28.0

    def _fallback_predict(self, age: float) -> dict:
        """Fallback when no trained model — use quadratic approximation."""
        peak_age = 28.0
        if age <= peak_age:
            factor = 1.0 + 0.015 * (age - peak_age)
        else:
            factor = 1.0 - 0.012 * (age - peak_age)
        return {
            'factor': factor,
            'inflection_age': 28.0,
            'volatility': 0.02 + 0.005 * abs(age - peak_age),
        }

    def precompute_all(
        self,
        bios_df: pd.DataFrame,
        performance_df: pd.DataFrame,
        cache_dir: str = 'data/cache',
    ) -> pd.DataFrame:
        """Train KAN model on league data and compute per-player outputs.

        Returns DataFrame with PLAYER_ID, KAN_AGE_NONLIN_FACTOR, etc.
        """
        if 'AGE' not in bios_df.columns:
            logger.warning("No AGE column in bios; skipping KAN precompute")
            return pd.DataFrame()

        # Aggregate league-wide age -> performance
        if 'PTS' in performance_df.columns and 'MIN' in performance_df.columns:
            perf_col = (performance_df['PTS'] / performance_df['MIN'].clip(lower=1))
        else:
            return pd.DataFrame()

        ages = performance_df['AGE'].values if 'AGE' in performance_df.columns else None
        if ages is None and 'PLAYER_ID' in performance_df.columns:
            # Merge age from bios
            merged = performance_df.merge(
                bios_df[['PLAYER_ID', 'AGE']].drop_duplicates('PLAYER_ID'),
                on='PLAYER_ID', how='left',
            )
            if 'AGE_y' in merged.columns:
                ages = merged['AGE_y'].values
            elif 'AGE' in merged.columns:
                ages = merged['AGE'].values

        if ages is None or len(ages) == 0:
            return pd.DataFrame()

        # Normalize performance to ~1.0
        perf_norm = perf_col.values / (np.nanmean(perf_col.values) + 1e-8)
        valid = np.isfinite(ages) & np.isfinite(perf_norm)
        if valid.sum() < 10:
            return pd.DataFrame()

        # Train KAN model
        metrics = self.train_on_age_performance(ages[valid], perf_norm[valid])
        logger.info(f"KAN training: loss={metrics['final_loss']:.4f}")

        # Predict for each unique age in bios
        results = []
        for pid in bios_df['PLAYER_ID'].unique():
            age_row = bios_df[bios_df['PLAYER_ID'] == pid]
            if 'AGE' in age_row.columns:
                age = float(age_row['AGE'].iloc[0])
            else:
                continue
            pred = self.predict(age)
            results.append({
                'PLAYER_ID': pid,
                'KAN_AGE_NONLIN_FACTOR': pred['factor'],
                'KAN_AGE_INFLECTION_AGE': pred['inflection_age'],
                'KAN_AGE_VOLATILITY': pred['volatility'],
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Cache to disk
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'kan_aging_outputs.csv')
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached KAN aging outputs for {len(df)} players to {cache_path}")

        return df

    def load_cached_outputs(self, cache_dir: str = 'data/cache') -> pd.DataFrame:
        """Load previously cached KAN aging outputs."""
        cache_path = os.path.join(cache_dir, 'kan_aging_outputs.csv')
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception:
                pass
        return pd.DataFrame()