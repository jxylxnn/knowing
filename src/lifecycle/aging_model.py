"""B-Ianus Bayesian Aging Curve Model.

Separates development (pre-peak) from decline (post-peak) with
position-specific peak ages and decline rates.

Peak age priors by position (from sports science literature):
  PG: 28.5, SG: 27.8, SF: 27.5, PF: 27.0, C: 26.5

Decline rate priors (performance loss per year past peak):
  PG: 0.8%, SG: 1.0%, SF: 1.2%, PF: 1.5%, C: 1.8%
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Position-specific priors (mean, std)
POSITION_PEAK_PRIORS: Dict[str, Tuple[float, float]] = {
    'PG': (28.5, 1.5), 'SG': (27.8, 1.3), 'SF': (27.5, 1.2),
    'PF': (27.0, 1.2), 'C': (26.5, 1.5),
    # Aliases
    'G': (28.2, 1.4), 'F': (27.3, 1.3), 'GF': (28.0, 1.4),
    'FC': (26.8, 1.4), 'CG': (27.5, 1.3),
}

POSITION_DECLINE_PRIORS: Dict[str, Tuple[float, float]] = {
    'PG': (0.008, 0.003), 'SG': (0.010, 0.003), 'SF': (0.012, 0.004),
    'PF': (0.015, 0.005), 'C': (0.018, 0.005),
    'G': (0.009, 0.003), 'F': (0.013, 0.004),
    'GF': (0.010, 0.004), 'FC': (0.016, 0.005), 'CG': (0.009, 0.003),
}

DEFAULT_PEAK_PRIOR = (27.5, 1.5)
DEFAULT_DECLINE_PRIOR = (0.012, 0.004)


def normalize_position(pos: str) -> str:
    """Map messy NBA position strings to canonical 5 positions."""
    if not pos or not isinstance(pos, str):
        return 'SF'  # default
    pos = pos.upper().strip()
    # Direct match
    if pos in POSITION_PEAK_PRIORS:
        return pos
    # Common NBA API position strings
    if 'POINT' in pos or pos == 'PG':
        return 'PG'
    if 'SHOOT' in pos or pos == 'SG':
        return 'SG'
    if 'SMALL' in pos or pos == 'SF':
        return 'SF'
    if 'POWER' in pos or pos == 'PF':
        return 'PF'
    if 'CENTER' in pos or pos == 'C':
        return 'C'
    # Hyphenated like "Guard-Forward"
    if 'GUARD' in pos and 'FORWARD' in pos:
        return 'GF'
    if 'FORWARD' in pos and 'CENTER' in pos:
        return 'FC'
    if 'GUARD' in pos:
        return 'G'
    if 'FORWARD' in pos:
        return 'F'
    return 'SF'


class BIanusAgingModel:
    """Bayesian structural aging curve with position-specific priors.

    Uses MAP (Maximum A Posteriori) estimation via scipy.optimize to fit
    a piecewise linear model:
      - Pre-peak:  performance = 1.0 + dev_rate * (age - peak_age)
      - Post-peak: performance = 1.0 - decline_rate * (age - peak_age)
    """

    def __init__(self, prior_strength: float = 10.0):
        self.prior_strength = prior_strength
        self._fitted_players: Dict[int, dict] = {}

    def fit_player(
        self,
        player_id: int,
        ages: np.ndarray,
        performance: np.ndarray,
        position: str,
    ) -> dict:
        """Fit aging curve for one player using MAP estimation.

        Returns dict with: peak_age, decline_rate, development_rate, position
        """
        norm_pos = normalize_position(position)
        peak_prior = POSITION_PEAK_PRIORS.get(norm_pos, DEFAULT_PEAK_PRIOR)
        decline_prior = POSITION_DECLINE_PRIORS.get(norm_pos, DEFAULT_DECLINE_PRIOR)

        def neg_log_posterior(params):
            peak, decline, dev_rate = params
            # Prior penalties (Gaussian)
            prior_penalty = self.prior_strength * (
                (peak - peak_prior[0]) ** 2 / peak_prior[1] ** 2 +
                (decline - decline_prior[0]) ** 2 / decline_prior[1] ** 2
            )
            # Piecewise linear model
            predicted = np.where(
                ages <= peak,
                1.0 + dev_rate * (ages - peak),
                1.0 - decline * (ages - peak),
            )
            residuals = performance - predicted
            likelihood = 0.5 * np.sum(residuals ** 2)
            return likelihood + prior_penalty

        x0 = [peak_prior[0], decline_prior[0], 0.02]
        bounds = [(24, 33), (0.001, 0.05), (0.001, 0.05)]

        try:
            result = minimize(
                neg_log_posterior, x0, method='L-BFGS-B', bounds=bounds,
            )
            peak, decline, dev_rate = result.x
        except Exception as e:
            logger.warning(f"BIanus fit failed for player {player_id}: {e}")
            peak, decline, dev_rate = peak_prior[0], decline_prior[0], 0.02

        player_params = {
            'peak_age': float(peak),
            'decline_rate': float(decline),
            'development_rate': float(dev_rate),
            'position': norm_pos,
        }
        self._fitted_players[player_id] = player_params
        return player_params

    def curve_factor(
        self,
        age: float,
        peak_age: float,
        decline_rate: float,
        dev_rate: float,
    ) -> float:
        """Compute aging curve multiplicative factor for a given age."""
        if age <= peak_age:
            return 1.0 + dev_rate * (age - peak_age)
        else:
            return 1.0 - decline_rate * (age - peak_age)

    def get_player_params(self, player_id: int) -> Optional[dict]:
        """Return previously fitted parameters for a player."""
        return self._fitted_players.get(player_id)

    def precompute_all(
        self,
        bios_df: pd.DataFrame,
        performance_df: pd.DataFrame,
        cache_dir: str = 'data/cache',
    ) -> pd.DataFrame:
        """Fit aging curves for all players and cache results.

        For players without enough data, uses position defaults.

        Returns DataFrame with columns:
            PLAYER_ID, peak_age, decline_rate, development_rate, position
        """
        results = []

        if 'PLAYER_ID' not in performance_df.columns or 'AGE' not in bios_df.columns:
            logger.warning("Missing required columns for aging curve precompute")
            return pd.DataFrame()

        # Merge bio data for position info
        if 'POSITION' in bios_df.columns:
            bio_pos = bios_df[['PLAYER_ID', 'POSITION']].drop_duplicates('PLAYER_ID')
        else:
            bio_pos = pd.DataFrame({'PLAYER_ID': [], 'POSITION': []})

        # Group performance by player
        for pid, group in performance_df.groupby('PLAYER_ID'):
            pos_row = bio_pos[bio_pos['PLAYER_ID'] == pid]
            position = pos_row['POSITION'].iloc[0] if len(pos_row) > 0 else 'SF'

            pos_norm = normalize_position(str(position))
            peak_prior = POSITION_PEAK_PRIORS.get(pos_norm, DEFAULT_PEAK_PRIOR)
            decline_prior = POSITION_DECLINE_PRIORS.get(pos_norm, DEFAULT_DECLINE_PRIOR)

            if 'AGE' in group.columns and len(group) >= 5:
                ages = group['AGE'].values.astype(float)
                # Use PTS/MIN as performance proxy
                if 'PTS' in group.columns and 'MIN' in group.columns:
                    perf = (group['PTS'] / group['MIN'].clip(lower=1)).values.astype(float)
                    # Normalize to ~1.0
                    perf = perf / (np.nanmean(perf) + 1e-8)
                else:
                    perf = np.ones_like(ages)
                params = self.fit_player(pid, ages, perf, position)
            else:
                # Not enough data — use position defaults
                params = {
                    'peak_age': peak_prior[0],
                    'decline_rate': decline_prior[0],
                    'development_rate': 0.02,
                    'position': pos_norm,
                }

            params['PLAYER_ID'] = pid
            results.append(params)

        if not results:
            return pd.DataFrame()

        curves_df = pd.DataFrame(results)

        # Cache to disk
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'aging_curves.csv')
        curves_df.to_csv(cache_path, index=False)
        logger.info(f"Cached {len(curves_df)} aging curves to {cache_path}")

        return curves_df

    def load_cached_curves(self, cache_dir: str = 'data/cache') -> pd.DataFrame:
        """Load previously cached aging curves."""
        cache_path = os.path.join(cache_dir, 'aging_curves.csv')
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached aging curves: {e}")
        return pd.DataFrame()

    def get_curve_factor_for_player(
        self,
        player_id: int,
        age: float,
        position: str = 'SF',
        cache_dir: str = 'data/cache',
    ) -> float:
        """Get the aging curve factor for a player at a given age.

        Uses cached curves if available, otherwise falls back to priors.
        """
        params = self._fitted_players.get(player_id)
        if params is None:
            # Try loading from cache
            curves = self.load_cached_curves(cache_dir)
            if not curves.empty and 'PLAYER_ID' in curves.columns:
                match = curves[curves['PLAYER_ID'] == player_id]
                if not match.empty:
                    row = match.iloc[0]
                    params = {
                        'peak_age': row.get('peak_age', 27.5),
                        'decline_rate': row.get('decline_rate', 0.012),
                        'development_rate': row.get('development_rate', 0.02),
                    }
                    self._fitted_players[player_id] = params

        if params is not None:
            return self.curve_factor(
                age, params['peak_age'],
                params['decline_rate'], params['development_rate'],
            )

        # Fallback: use position priors
        norm_pos = normalize_position(position)
        peak_prior = POSITION_PEAK_PRIORS.get(norm_pos, DEFAULT_PEAK_PRIOR)
        decline_prior = POSITION_DECLINE_PRIORS.get(norm_pos, DEFAULT_DECLINE_PRIOR)
        return self.curve_factor(age, peak_prior[0], decline_prior[0], 0.02)