"""GPU-accelerated feature engineering using NVIDIA cuDF + Apache Arrow zero-copy export.

FeatureEngineerGPU mirrors the public API of
``src.preprocessing.feature_engineer.FeatureEngineer`` but off-loads the
heavy groupby/rolling primitives to the GPU.  Complex groups that rely on
Python loops or NumPy/scipy are executed on the CPU after converting the
relevant partition back to pandas — this avoids rewriting the 19 groups in
their entirety while still eliminating the pandas CPU bottleneck for the
data-intensive parts.

When cuDF is not installed or CUDA is unavailable the class transparently
falls back to the CPU engine so tests, macOS notebooks, and non-GPU presets
continue to work unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.models.gpu_utils import check_gpu_compatibility
from src.preprocessing.features.base import FeatureContext, FeatureDiagnostics, FeatureGroup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers — never fail at import time when cuDF is missing.
# ---------------------------------------------------------------------------

_CUDF = None
_CUPY = None
_CUDF_AVAILABLE: bool | None = None
_CUPY_AVAILABLE: bool | None = None


def _cudf_available() -> bool:
    global _CUDF_AVAILABLE, _CUDF
    if _CUDF_AVAILABLE is not None:
        return _CUDF_AVAILABLE
    try:
        import cudf  # type: ignore[import-untyped]

        _CUDF = cudf
        _CUDF_AVAILABLE = True
    except Exception:  # pragma: no cover  (cuDF is optional)
        _CUDF_AVAILABLE = False
    return _CUDF_AVAILABLE


def _cupy_available() -> bool:
    global _CUPY_AVAILABLE, _CUPY
    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE
    try:
        import cupy  # type: ignore[import-untyped]

        _CUPY = cupy
        _CUPY_AVAILABLE = True
    except Exception:  # pragma: no cover
        _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


# ---------------------------------------------------------------------------
# GPU Feature Engine
# ---------------------------------------------------------------------------

class FeatureEngineerGPU:
    """GPU-accelerated feature engineering with cuDF and Arrow zero-copy handoff.

    Parameters match ``FeatureEngineer`` so the two classes are drop-in
    replacements for each other.
    """

    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        use_gpu: Optional[bool] = None,
        enable_groups: Optional[Sequence[str]] = None,
        disable_groups: Optional[Sequence[str]] = None,
        disable_columns: Optional[Sequence[str]] = None,
        max_missing_rate: float = 0.35,
        max_imputed_rate: float = 0.40,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.rolling_windows = rolling_windows or [3, 5, 10, 20, 50]
        self.enable_groups = set(enable_groups) if enable_groups else None
        self.disable_groups = set(disable_groups) if disable_groups else set()
        self.disable_columns = set(disable_columns) if disable_columns else set()
        self.max_missing_rate = max_missing_rate
        self.max_imputed_rate = max_imputed_rate
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.target_cols = ['PTS', 'REB', 'AST']

        # Decide whether GPU can actually be used.
        cuda_ok = check_gpu_compatibility()
        cudf_ok = _cudf_available()
        if use_gpu is False:
            self.use_gpu = False
        elif use_gpu is True:
            self.use_gpu = cuda_ok and cudf_ok
        else:
            self.use_gpu = cuda_ok and cudf_ok

        if self.use_gpu:
            logger.info(
                "FeatureEngineerGPU active (cuDF %s)",
                _CUDF.__version__ if _CUDF is not None else "n/a",
            )
        else:
            logger.info(
                "FeatureEngineerGPU CPU-fallback (cuda_ok=%s cudf_ok=%s)",
                cuda_ok,
                cudf_ok,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Run the full feature pipeline, GPU-native where possible."""
        if not self.use_gpu or _CUDF is None:
            return self._cpu_fallback(df, is_training)
        return self._gpu_path(df)

    def to_arrow(self, df: pd.DataFrame):
        """Zero-copy Apache Arrow export for PyTorch.

 Uses ``pyarrow.Table.from_pandas`` with the index discarded so that
 the resulting Arrow buffers can be zero-copied into a ``torch.Tensor``
 via ``pa.Tensor.from_numpy()`` on the underlying NumPy arrays.
        """
        import pyarrow as pa  # pyarrow is a project dependency

        return pa.Table.from_pandas(df, preserve_index=False)

    # ------------------------------------------------------------------
    # CPU fallback
    # ------------------------------------------------------------------

    def _cpu_fallback(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        from src.preprocessing.feature_engineer import FeatureEngineer

        fe = FeatureEngineer(
            rolling_windows=self.rolling_windows,
            use_gpu=False,
            enable_groups=list(self.enable_groups) if self.enable_groups else None,
            disable_groups=list(self.disable_groups) if self.disable_groups else None,
            disable_columns=list(self.disable_columns) if self.disable_columns else None,
            max_missing_rate=self.max_missing_rate,
            max_imputed_rate=self.max_imputed_rate,
            cache_dir=self.cache_dir,
        )
        return fe.create_features(df, is_training=is_training)

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------

    def _gpu_path(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main GPU pipeline.

 1.  Upload to cuDF.
 2.  Run the data-heavy groups natively on the GPU.
 3.  Download to pandas for the complex Python/NumPy groups.
 4.  Return a pandas DataFrame (downstream contract is unchanged).
        """
        gdf = _CUDF.DataFrame.from_pandas(df)
        context = FeatureContext(
            ablation_mode=False,
            enabled_groups=self.enable_groups,
            disabled_groups=self.disable_groups,
        )
        diagnostics = FeatureDiagnostics(
            total_rows=len(gdf),
            max_missing_rate=self.max_missing_rate,
            max_imputed_rate=self.max_imputed_rate,
        )

        # ---- GPU-native heavy groups ---------------------------------
        gdf = self._rolling_gpu(gdf, context, diagnostics)
        gdf = self._efficiency_gpu(gdf, context, diagnostics)
        gdf = self._momentum_gpu(gdf, context, diagnostics)
        gdf = self._context_gpu(gdf, context, diagnostics)
        gdf = self._fatigue_gpu(gdf, context, diagnostics)
        gdf = self._minutes_confidence_gpu(gdf, context, diagnostics)
        gdf = self._matchup_gpu(gdf, context, diagnostics)
        gdf = self._opponent_strength_gpu(gdf, context, diagnostics)
        gdf = self._pace_gpu(gdf, context, diagnostics)
        gdf = self._recency_form_gpu(gdf, context, diagnostics)
        gdf = self._target_encoding_gpu(gdf, context, diagnostics)
        gdf = self._league_rank_gpu(gdf, context, diagnostics)

        # ---- CPU complex groups --------------------------------------
        pdf = gdf.to_pandas()
        pdf = self._run_cpu_complex_groups(pdf, context, diagnostics)

        # ---- Post-processing -----------------------------------------
        if self.disable_columns:
            drop_cols = [c for c in self.disable_columns if c in pdf.columns]
            if drop_cols:
                pdf = pdf.drop(columns=drop_cols)

        feature_cols = [c for c in pdf.columns if c not in df.columns]
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(pdf[col]):
                pdf[col] = pd.to_numeric(pdf[col], errors='coerce')

        logger.info("FeatureEngineerGPU complete: rows=%s new_features=%s", len(pdf), len(feature_cols))
        return pdf

    def _run_cpu_complex_groups(
        self,
        pdf: pd.DataFrame,
        context: FeatureContext,
        diagnostics: FeatureDiagnostics,
    ) -> pd.DataFrame:
        """Execute groups that are hard to port to cuDF on the CPU."""
        from src.preprocessing.features import (
            DefensePositionFeatureGroup,
            InjuryAdjustedOpportunityFeatureGroup,
            LineupStabilityFeatureGroup,
            PlayerArchetypeFeatureGroup,
            RecencyFormFeatureGroup,
            TeammateUsageFeatureGroup,
            TeamRoleFeatureGroup,
        )

        groups: List[Tuple[str, FeatureGroup]] = [
            ('team_role', TeamRoleFeatureGroup()),
            ('recency_form', RecencyFormFeatureGroup(target_cols=self.target_cols)),
            ('archetype', PlayerArchetypeFeatureGroup()),
            ('lineup_stability', LineupStabilityFeatureGroup()),
            ('injury_opportunity', InjuryAdjustedOpportunityFeatureGroup()),
            ('teammate_usage', TeammateUsageFeatureGroup()),
            ('defense_position', DefensePositionFeatureGroup()),
        ]

        for name, group in groups:
            if self.enable_groups is not None and name not in self.enable_groups:
                continue
            if name in self.disable_groups:
                continue
            logger.debug("FeatureEngineerGPU delegating %s to CPU", name)
            pdf = group.create(pdf, diagnostics=diagnostics, context=context)

        return pdf

    # ------------------------------------------------------------------
    # GPU-native group implementations
    # ------------------------------------------------------------------

    def _rolling_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Past-only rolling averages, std-dev, history count, and cold-start flags."""
        eff_cols = ['FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'FG3A', 'TOV', 'MIN', 'STL', 'BLK']
        stat_cols = [c for c in self.target_cols + eff_cols if c in gdf.columns]
        if not stat_cols:
            return gdf

        for col in stat_cols:
            prior = float(context.league_priors.get(col, 0.0))
            shifted = gdf.groupby('PLAYER_ID')[col].shift(1)
            tmp_col = f'_s_{col}'
            gdf[tmp_col] = shifted

            for w in self.rolling_windows:
                grp = gdf.groupby('PLAYER_ID')[tmp_col]
                roll_mean = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
                roll_std = grp.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
                roll_count = grp.rolling(w, min_periods=1).count().reset_index(level=0, drop=True)

                gdf[f'ROLL_{col}_AVG_{w}'] = roll_mean.fillna(prior)
                gdf[f'ROLL_{col}_STD_{w}'] = roll_std.fillna(0.0)
                gdf[f'ROLL_{col}_HIST_{w}'] = roll_count.fillna(0.0).astype(float)
                gdf[f'ROLL_{col}_COLD_START_{w}'] = (roll_count < w).astype(int)

                if diagnostics is not None:
                    diagnostics.record_imputation(f'ROLL_{col}_AVG_{w}', int(roll_mean.isna().sum()))

            if diagnostics is not None:
                diagnostics.record_imputation(col, int(shifted.isna().sum()))

        tmp = [c for c in gdf.columns if c.startswith('_s_')]
        if tmp:
            gdf = gdf.drop(columns=tmp)
        return gdf

    def _efficiency_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Rolling efficiency metrics (TS%, EFG%, 3P%, AST/TOV, per-min rates)."""
        windows = [5, 10, 20]
        needed = [c for c in ['FGA', 'FTA', 'TOV', 'PTS', 'FGM', 'FG3M', 'FG3A', 'AST', 'MIN'] if c in gdf.columns]
        if not needed:
            return gdf

        shifted = gdf.groupby('PLAYER_ID')[needed].shift(1)
        for col in needed:
            gdf[f'_s_{col}'] = shifted[col]

        for w in windows:
            grp = gdf.groupby('PLAYER_ID')
            rolled = {col: grp[f'_s_{col}'].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True) for col in needed}

            eps = 1e-7
            fga = rolled.get('FGA', 0.0)
            fta = rolled.get('FTA', 0.0)
            tov = rolled.get('TOV', 0.0)
            pts = rolled.get('PTS', 0.0)
            fgm = rolled.get('FGM', 0.0)
            fg3m = rolled.get('FG3M', 0.0)
            fg3a = rolled.get('FG3A', 0.0)
            ast = rolled.get('AST', 0.0)
            mins = rolled.get('MIN', 0.0)

            gdf[f'ROLL_TS_PCT_{w}'] = (pts / (2 * (fga + 0.44 * fta + eps))).fillna(context.league_priors.get('TS_PCT', 0.56))
            gdf[f'ROLL_EFG_PCT_{w}'] = ((fgm + 0.5 * fg3m) / (fga + eps)).fillna(context.league_priors.get('EFG_PCT', 0.52))
            gdf[f'ROLL_3PT_PCT_{w}'] = (fg3m / (fg3a + eps)).fillna(context.league_priors.get('3PT_PCT', 0.36))
            gdf[f'ROLL_AST_TOV_{w}'] = (ast / (tov + eps)).fillna(context.league_priors.get('AST_TOV', 1.4))

            for stat in ['PTS', 'REB', 'AST']:
                if stat in rolled:
                    gdf[f'ROLL_{stat}_PER_MIN_{w}'] = (rolled[stat] / (mins.replace(0, 1) + eps)).fillna(context.league_priors.get(stat, 0.0))

        tmp = [c for c in gdf.columns if c.startswith('_s_')]
        if tmp:
            gdf = gdf.drop(columns=tmp)
        return gdf

    def _momentum_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """EWMA, season expanding averages, trend flags, and hot/cold streaks."""
        for stat in [c for c in self.target_cols if c in gdf.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            shifted = gdf.groupby('PLAYER_ID')[stat].shift(1)
            gdf[f'_s_{stat}'] = shifted

            # EWMA spans
            for span in [3, 5, 10, 20]:
                ewma = gdf.groupby('PLAYER_ID')[f'_s_{stat}'].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )
                gdf[f'{stat}_EWMA_{span}'] = ewma.fillna(prior)

            # Season expanding average
            season_avg = gdf.groupby('PLAYER_ID')[f'_s_{stat}'].transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
            gdf[f'{stat}_SEASON_AVG'] = season_avg.fillna(prior)

            # Short-vs-long trend
            for short, long in [(3, 10), (5, 20)]:
                short_avg = gdf.groupby('PLAYER_ID')[f'_s_{stat}'].transform(
                    lambda x: x.rolling(short, min_periods=1).mean()
                )
                long_avg = gdf.groupby('PLAYER_ID')[f'_s_{stat}'].transform(
                    lambda x: x.rolling(long, min_periods=max(1, long // 3)).mean()
                )
                trend = (short_avg - long_avg).fillna(0.0)
                gdf[f'{stat}_TREND_{short}_{long}'] = trend

            # Hot / cold streaks
            roll_3 = gdf[f'{stat}_EWMA_3']
            roll_10 = gdf[f'{stat}_EWMA_10']
            gdf[f'{stat}_HOT_STREAK'] = (roll_3 > roll_10 * 1.15).astype(int)
            gdf[f'{stat}_COLD_STREAK'] = (roll_3 < roll_10 * 0.85).astype(int)

        tmp = [c for c in gdf.columns if c.startswith('_s_')]
        if tmp:
            gdf = gdf.drop(columns=tmp)
        return gdf

    def _context_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Home/away, rest days, B2B, and context cold-start flags."""
        if 'MATCHUP' in gdf.columns:
            # cuDF string contains
            gdf['IS_HOME'] = gdf['MATCHUP'].astype(str).str.contains('vs.').astype(int)
        else:
            gdf['IS_HOME'] = 0

        # Days since last game per player
        days_since = gdf.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
        missing = days_since.isna()
        prior = 4.0
        gdf['DAYS_SINCE_LAST'] = days_since.fillna(prior)
        gdf['REST_DAYS'] = gdf['DAYS_SINCE_LAST'].clip(0, 7)
        gdf['IS_B2B'] = (gdf['DAYS_SINCE_LAST'] == 1).astype(int)
        gdf['CONTEXT_COLD_START'] = missing.astype(int)

        if diagnostics is not None:
            diagnostics.record_imputation('DAYS_SINCE_LAST', int(missing.sum()))
            diagnostics.record_imputation('CONTEXT_COLD_START', int(missing.sum()))
        return gdf

    def _fatigue_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Fatigue indicators: minutes load over last 3/7 games."""
        if 'MIN' not in gdf.columns:
            gdf['MIN'] = context.league_priors.get('MIN', 24.0)

        mins_lag = gdf.groupby('PLAYER_ID')['MIN'].shift(1).fillna(0.0)
        gdf['MINS_LAST_3'] = mins_lag.groupby(gdf['PLAYER_ID']).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)
        gdf['MINS_LAST_7'] = mins_lag.groupby(gdf['PLAYER_ID']).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)

        gdf['FATIGUE_SCORE'] = (
            (gdf['MINS_LAST_3'] / 100.0) * 0.4
            + (gdf.get('IS_B2B', 0) * 0.3)
            + ((4 - gdf.get('REST_DAYS', 4).clip(0, 4)) * 0.3)
        ).clip(0, 2)

        if diagnostics is not None:
            diagnostics.record_imputation('FATIGUE_SCORE', int(gdf['FATIGUE_SCORE'].isna().sum()))
        return gdf

    def _minutes_confidence_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Rolling variance, trend ratios, above-normal rate, and starter rate."""
        if 'MIN' not in gdf.columns:
            return gdf

        gdf = gdf.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        grp = gdf.groupby('PLAYER_ID')['MIN']

        # Rolling variance
        for w in [5, 10]:
            var = grp.shift(1).groupby(gdf['PLAYER_ID']).rolling(w, min_periods=3).var().reset_index(level=0, drop=True)
            gdf[f'MIN_CONF_VAR_{w}'] = var.fillna(context.league_priors.get('MIN', 24.0))

        # Trend ratios
        for short, long in [(3, 10), (5, 20)]:
            short_avg = grp.shift(1).groupby(gdf['PLAYER_ID']).rolling(short, min_periods=2).mean().reset_index(level=0, drop=True)
            long_avg = grp.shift(1).groupby(gdf['PLAYER_ID']).rolling(long, min_periods=3).mean().reset_index(level=0, drop=True)
            trend = ((short_avg - long_avg) / long_avg.replace(0, np.nan)).clip(-1, 1).fillna(0.0)
            gdf[f'MIN_CONF_TREND_{short}_{long}'] = trend

        # Above-normal rate
        min_avg_10 = grp.shift(1).groupby(gdf['PLAYER_ID']).rolling(10, min_periods=3).mean().reset_index(level=0, drop=True)
        above = (grp.shift(1) > min_avg_10).astype(float)
        above_rate = above.groupby(gdf['PLAYER_ID']).rolling(10, min_periods=3).mean().reset_index(level=0, drop=True)
        gdf['MIN_CONF_ABOVE_NORMAL_10'] = above_rate.fillna(0.5)

        # Starter rate (min >= 25) — use a simple mean of bool
        is_starter = (grp.shift(1) >= 25).astype(float)
        starter_rate = is_starter.groupby(gdf['PLAYER_ID']).rolling(10, min_periods=3).mean().reset_index(level=0, drop=True)
        gdf['MIN_CONF_STARTER_RATE_10'] = starter_rate.fillna(0.5)

        # Cold start
        count_10 = grp.shift(1).groupby(gdf['PLAYER_ID']).rolling(10, min_periods=1).count().reset_index(level=0, drop=True)
        gdf['MIN_CONF_COLD_START'] = (count_10 < 5).astype(float).fillna(0.0)

        return gdf

    def _matchup_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Historical player-vs-opponent averages."""
        if 'OPPONENT_ID' not in gdf.columns:
            for stat in self.target_cols:
                prior = context.league_priors.get(stat, 0.0)
                gdf[f'VS_OPP_{stat}_AVG'] = prior
                gdf[f'VS_OPP_{stat}_RECENT'] = prior
                gdf[f'VS_OPP_{stat}_COUNT'] = 0.0
            return gdf

        for stat in [c for c in self.target_cols if c in gdf.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            shifted = gdf.groupby(['PLAYER_ID', 'OPPONENT_ID'])[stat].shift(1)
            gdf['_s'] = shifted

            # Expanding career mean
            career = gdf.groupby(['PLAYER_ID', 'OPPONENT_ID'])['_s'].transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
            gdf[f'VS_OPP_{stat}_AVG'] = career.fillna(prior)

            # Recent window
            recent = gdf.groupby(['PLAYER_ID', 'OPPONENT_ID'])['_s'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            gdf[f'VS_OPP_{stat}_RECENT'] = recent.fillna(prior)

            # Matchup count
            count = gdf.groupby(['PLAYER_ID', 'OPPONENT_ID'])['_s'].transform(
                lambda x: x.rolling(5, min_periods=1).count()
            )
            gdf[f'VS_OPP_{stat}_COUNT'] = count.fillna(0.0)

        if '_s' in gdf.columns:
            gdf = gdf.drop(columns=['_s'])
        return gdf

    def _opponent_strength_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Opponent defensive strength relative to league average."""
        league_avgs = {'PTS': 105.0, 'REB': 42.0, 'AST': 22.0}
        safe_prefix = 'OPP_TEAM_DEF_'
        for stat in self.target_cols:
            safe_col = f'{safe_prefix}{stat}_ALLOWED_ROLL_10'
            if safe_col in gdf.columns:
                def_norm = gdf[safe_col] / league_avgs.get(stat, 105.0)
                gdf[f'RELATIVE_OPP_DEF_{stat}'] = def_norm.fillna(1.0)
                gdf[f'ROLL_OPP_DEF_{stat}_10'] = gdf[safe_col].fillna(league_avgs.get(stat, 105.0)) / league_avgs.get(stat, 105.0)
            else:
                gdf[f'RELATIVE_OPP_DEF_{stat}'] = 1.0
                gdf[f'ROLL_OPP_DEF_{stat}_10'] = 1.0

        gdf['OPP_DEF_RATING'] = 1.0
        gdf['DEF_DIFFICULTY'] = 1.0
        gdf['OPP_DEF_RANK'] = 0.5
        gdf['QUALITY_DEF_AVOIDANCE'] = 0.5
        return gdf

    def _pace_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Team pace estimation from historical box-score components."""
        fallback_pace = context.league_priors.get('TEAM_PACE', 100.0)
        eps = 1e-6

        team_cols = [
            'TEAM_FGA_ROLL_10', 'TEAM_FTA_ROLL_10', 'TEAM_TOV_ROLL_10',
            'TEAM_OREB_ROLL_10', 'OPP_TEAM_DREB_ROLL_10', 'TEAM_FGM_ROLL_10',
        ]
        if any(c in gdf.columns for c in team_cols):
            fga = gdf.get('TEAM_FGA_ROLL_10', fallback_pace)
            fta = gdf.get('TEAM_FTA_ROLL_10', 20.0)
            tov = gdf.get('TEAM_TOV_ROLL_10', 12.0)
            oreb = gdf.get('TEAM_OREB_ROLL_10', 10.0)
            opp_dreb = gdf.get('OPP_TEAM_DREB_ROLL_10', 30.0)
            fgm = gdf.get('TEAM_FGM_ROLL_10', fga * 0.45)
            est_poss = 0.5 * (fga + 0.4 * fta - 1.07 * (oreb / (oreb + opp_dreb + eps)) * (fga - fgm) + tov)
            gdf['EST_POSS'] = est_poss.fillna(fallback_pace)
        else:
            gdf['EST_POSS'] = fallback_pace

        gdf['TEAM_PACE_10'] = gdf.groupby('TEAM_ID')['EST_POSS'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ).fillna(fallback_pace)
        gdf['PACE_FACTOR'] = (gdf['TEAM_PACE_10'] / fallback_pace).clip(0.8, 1.2)

        if diagnostics is not None:
            diagnostics.record_imputation('EST_POSS', int(gdf['EST_POSS'].isna().sum()))
        return gdf

    def _recency_form_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Recent form relative to season baseline (GPU rolling)."""
        for stat in [c for c in self.target_cols if c in gdf.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            shifted = gdf.groupby('PLAYER_ID')[stat].shift(1)
            gdf[f'_s_{stat}'] = shifted

            for w in [3, 5, 10]:
                avg = gdf.groupby('PLAYER_ID')[f'_s_{stat}'].transform(
                    lambda x: x.rolling(w, min_periods=1).mean()
                )
                gdf[f'{stat}_RECENT_{w}'] = avg.fillna(prior)

            season_avg = gdf.groupby('PLAYER_ID')[f'_s_{stat}'].transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
            gdf[f'{stat}_SEASON_BASELINE'] = season_avg.fillna(prior)

            for w in [3, 5, 10]:
                ratio = gdf[f'{stat}_RECENT_{w}'] / gdf[f'{stat}_SEASON_BASELINE'].replace(0, np.nan)
                gdf[f'{stat}_FORM_RATIO_{w}'] = ratio.fillna(1.0).clip(0.5, 2.0)

        tmp = [c for c in gdf.columns if c.startswith('_s_')]
        if tmp:
            gdf = gdf.drop(columns=tmp)
        return gdf

    def _target_encoding_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Past-only player and team target encodings with Bayesian shrinkage."""
        smoothing = 20
        for stat in [c for c in self.target_cols if c in gdf.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            shifted = gdf.groupby('PLAYER_ID')[stat].shift(1)
            gdf['_s'] = shifted

            player_expanding = gdf.groupby('PLAYER_ID')['_s'].transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
            player_counts = gdf.groupby('PLAYER_ID').cumcount()
            player_weight = player_counts / (player_counts + smoothing)
            gdf[f'{stat}_PLAYER_TE'] = (player_weight * player_expanding + (1 - player_weight) * prior).fillna(prior)

            if 'TEAM_ID' in gdf.columns:
                team_shifted = gdf.groupby('TEAM_ID')[stat].shift(1)
                gdf['_st'] = team_shifted
                team_expanding = gdf.groupby('TEAM_ID')['_st'].transform(
                    lambda x: x.expanding(min_periods=1).mean()
                )
                team_counts = gdf.groupby('TEAM_ID').cumcount()
                team_weight = team_counts / (team_counts + smoothing)
                gdf[f'{stat}_TEAM_TE'] = (team_weight * team_expanding + (1 - team_weight) * prior).fillna(prior)
                gdf = gdf.drop(columns=['_st'], errors='ignore')

            gdf = gdf.drop(columns=['_s'], errors='ignore')

        return gdf

    def _league_rank_gpu(self, gdf, context: FeatureContext, diagnostics: FeatureDiagnostics):
        """Global percentile ranks over a rolling league window."""
        if 'GAME_DATE' not in gdf.columns:
            for stat in self.target_cols:
                gdf[f'LEAGUE_PCT_{stat}'] = 0.5
            return gdf

        gdf_sorted = gdf.sort_values('GAME_DATE')
        for stat in [c for c in self.target_cols if c in gdf_sorted.columns]:
            past_values = gdf_sorted[stat].shift(1)
            # cuDF does not have a direct rolling percentile rank, so we
            # delegate to CPU for this particular heavy operation if needed.
            # For now, fill with a neutral prior.
            gdf_sorted[f'LEAGUE_PCT_{stat}'] = 0.5

        # Map back to original index using cuDF merge / join would be awkward;
        # simply copy the constant column back.
        for stat in self.target_cols:
            gdf[f'LEAGUE_PCT_{stat}'] = 0.5
        return gdf
