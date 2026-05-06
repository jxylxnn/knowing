"""Statistical utility functions for simulation analysis.

Provides shared statistical computation helpers used across simulation
components (e.g., GameSimulator, ReportGenerator) to avoid duplication.
Includes robust mode estimation and summary statistics generation.
"""

import numpy as np


def compute_mode(values: np.ndarray) -> float:
    """Compute the mode of a distribution. Uses KDE for continuous data,
    histogram binning for discrete-like data.

    Args:
        values: Array of numeric values.

    Returns:
        The estimated mode as a float. Falls back to mean/median on edge cases.
    """
    values = np.asarray(values)
    if len(values) < 3:
        return float(np.mean(values))

    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float(np.mean(values))

    # Check if values are essentially discrete (integers or half-integers)
    rounded = np.round(values * 2) / 2
    if np.allclose(values, rounded, atol=0.1):
        bins = np.arange(values.min() - 0.25, values.max() + 0.75, 0.5)
        if len(bins) < 2:
            return float(np.mean(values))
        hist, bin_edges = np.histogram(values, bins=bins)
        mode_idx = np.argmax(hist)
        return float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)

    try:
        from scipy import stats as scipy_stats
        kde = scipy_stats.gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 200)
        densities = kde(x_grid)
        mode_idx = np.argmax(densities)
        return float(x_grid[mode_idx])
    except Exception:
        return float(np.median(values))


def compute_stats_summary(values: np.ndarray) -> dict:
    """Compute mean, mode, median, std, min, max, and key percentiles.

    Args:
        values: Array of numeric values.

    Returns:
        Dict with keys: mean, mode, median, std, min, max,
        p5, p10, p25, p50, p75, p90, p95.
    """
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'mode': compute_mode(arr),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p10': float(np.percentile(arr, 10)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
    }