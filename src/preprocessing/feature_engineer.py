"""Modular feature engineering orchestrator."""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.models.gpu_utils import check_gpu_compatibility
from src.preprocessing.features import (
    PlayerArchetypeFeatureGroup,
    ContextualFeatureGroup,
    DefensePositionFeatureGroup,
    EfficiencyFeatureGroup,
    FatigueFeatureGroup,
    InjuryAdjustedOpportunityFeatureGroup,
    InjuryRiskFeatureGroup,
    AgingCurveFeatureGroup,
    KANAgingFeatureGroup,
    SkillDevelopmentFeatureGroup,
    LeagueRankingFeatureGroup,
    LineupStabilityFeatureGroup,
    MatchupFeatureGroup,
    MinutesConfidenceFeatureGroup,
    MomentumFeatureGroup,
    OpponentStrengthFeatureGroup,
    PaceFeatureGroup,
    RecencyFormFeatureGroup,
    RestGameDensityFeatureGroup,
    RollingFeatureGroup,
    TargetEncodingFeatureGroup,
    TeamRoleFeatureGroup,
    TeammateUsageFeatureGroup,
    SeasonPhaseFeatureGroup,
    TeamMotivationFeatureGroup,
    PostseasonContextFeatureGroup,
)
from src.preprocessing.features.base import FeatureContext, FeatureDiagnostics, FeatureGroup
from src.utils.prediction_utils import FeatureSelector

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Metadata captured during feature generation."""

    group_columns: Dict[str, List[str]] = field(default_factory=dict)
    diagnostics: FeatureDiagnostics = field(default_factory=FeatureDiagnostics)
    n_rows: int = 0
    n_features: int = 0


class FeatureEngineer:
    """Orchestrates modular feature groups and collects diagnostics."""

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
        self.use_gpu = check_gpu_compatibility() if (use_gpu is None or use_gpu) else False
        self.max_missing_rate = max_missing_rate
        self.max_imputed_rate = max_imputed_rate
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.last_result = FeatureEngineeringResult()
        self._gpu_engine: Optional[object] = None  # cached GPU engine

        self.target_cols = ['PTS', 'REB', 'AST']
        self.feature_groups: List[FeatureGroup] = self._build_groups()

        if self.use_gpu:
            logger.info("FeatureEngineer requested GPU feature engineering; CUDA is available.")
        else:
            logger.info("FeatureEngineer running on CPU path.")

    def _build_groups(self) -> List[FeatureGroup]:
        groups: List[FeatureGroup] = [
            RollingFeatureGroup(windows=self.rolling_windows, target_cols=self.target_cols),
            EfficiencyFeatureGroup(windows=[5, 10, 20]),
            MomentumFeatureGroup(target_cols=self.target_cols),
            ContextualFeatureGroup(),
            FatigueFeatureGroup(),
            MinutesConfidenceFeatureGroup(target_cols=self.target_cols),
            RestGameDensityFeatureGroup(),
            MatchupFeatureGroup(target_cols=self.target_cols, recent_window=5),
            OpponentStrengthFeatureGroup(target_cols=self.target_cols),
            PaceFeatureGroup(),
            TeamRoleFeatureGroup(),
            LineupStabilityFeatureGroup(),
            InjuryAdjustedOpportunityFeatureGroup(),
            TeammateUsageFeatureGroup(),
            RecencyFormFeatureGroup(target_cols=self.target_cols),
            PlayerArchetypeFeatureGroup(),
            DefensePositionFeatureGroup(),
            TargetEncodingFeatureGroup(target_cols=self.target_cols, smoothing=20),
            LeagueRankingFeatureGroup(target_cols=self.target_cols, window=2000, min_periods=500),
            # Player lifecycle & bio-mechanical feature groups
            InjuryRiskFeatureGroup(),
            AgingCurveFeatureGroup(),
            KANAgingFeatureGroup(),
            SkillDevelopmentFeatureGroup(),
            # Season context feature groups
            SeasonPhaseFeatureGroup(),
            TeamMotivationFeatureGroup(),
            PostseasonContextFeatureGroup(),
        ]
        return groups

    def _should_run_group(self, group: FeatureGroup) -> bool:
        if self.enable_groups is not None and group.name not in self.enable_groups:
            return False
        if group.name in self.disable_groups:
            return False
        return True

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        frame = df.copy()
        if 'GAME_DATE' in frame.columns:
            frame['GAME_DATE'] = pd.to_datetime(frame['GAME_DATE'], errors='coerce')
            invalid_dates = int(frame['GAME_DATE'].isna().sum())
            if invalid_dates:
                logger.warning("Dropping %s rows with invalid GAME_DATE values", invalid_dates)
                frame = frame.dropna(subset=['GAME_DATE'])
        if 'PLAYER_ID' in frame.columns and 'GAME_DATE' in frame.columns:
            frame = frame.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        elif 'GAME_DATE' in frame.columns:
            frame = frame.sort_values('GAME_DATE').reset_index(drop=True)
        return frame

    def _run_group(
        self,
        group: FeatureGroup,
        df: pd.DataFrame,
        diagnostics: FeatureDiagnostics,
        context: FeatureContext,
    ) -> Tuple[pd.DataFrame, List[str]]:
        before = set(df.columns)
        out = group.create(df, diagnostics=diagnostics, context=context)
        added = [c for c in out.columns if c not in before]
        return out, added

    def _cache_key(self, df: pd.DataFrame) -> str:
        """Compute a stable cache key from input data and FE configuration."""
        config_str = (
            f"windows={sorted(self.rolling_windows)}"
            f"|groups={sorted(self.enable_groups or [])}"
            f"|disable_groups={sorted(self.disable_groups)}"
            f"|disable_cols={sorted(self.disable_columns)}"
            f"|max_missing={self.max_missing_rate}"
            f"|max_imputed={self.max_imputed_rate}"
        )
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:16]
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"fe_{data_hash}_{config_hash}"

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create leakage-safe features with explicit diagnostics."""
        logger.info("--- Starting modular feature engineering pipeline ---")
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to feature engineering")
            return pd.DataFrame()

        required_cols = {'PLAYER_ID', 'GAME_DATE'}
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            logger.error("Missing required feature-engineering columns: %s", missing_required)
            return pd.DataFrame()

        cache_hit = False
        cache_path = None
        if self.cache_dir is not None:
            cache_key = self._cache_key(df)
            cache_path = self.cache_dir / f"{cache_key}.parquet"
            if cache_path.exists():
                logger.info("Loading cached features from %s", cache_path)
                try:
                    result = pd.read_parquet(cache_path)
                    cache_hit = True
                    logger.info(
                        "Cache hit: loaded %s rows, %s columns",
                        len(result),
                        len(result.columns),
                    )
                except Exception as exc:
                    logger.warning("Failed to load cached features: %s; recomputing", exc)
                    cache_hit = False

        if not cache_hit:
            result = self._compute_features(df)

        if not cache_hit and self.cache_dir is not None and cache_path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                result.to_parquet(cache_path, index=True)
                logger.info("Cached features to %s", cache_path)
            except Exception as exc:
                logger.warning("Failed to cache features: %s", exc)

        return result

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline (no caching)."""
        if self.use_gpu:
            if self._gpu_engine is None:
                try:
                    from src.preprocessing.feature_engineer_gpu import FeatureEngineerGPU
                    self._gpu_engine = FeatureEngineerGPU(
                        rolling_windows=self.rolling_windows,
                        use_gpu=True,
                        enable_groups=list(self.enable_groups) if self.enable_groups else None,
                        disable_groups=list(self.disable_groups) if self.disable_groups else None,
                        disable_columns=list(self.disable_columns) if self.disable_columns else None,
                        max_missing_rate=self.max_missing_rate,
                        max_imputed_rate=self.max_imputed_rate,
                        cache_dir=self.cache_dir,
                    )
                except Exception as exc:
                    logger.warning("GPU feature engine unavailable (%s); falling back to CPU.", exc)
                    self._gpu_engine = None
            if self._gpu_engine is not None:
                logger.info("Delegating feature engineering to GPU path.")
                try:
                    # type: ignore[union-attr]
                    gpu_result = self._gpu_engine.create_features(df, is_training=True)
                    return gpu_result
                except Exception as exc:
                    logger.warning("GPU feature engineering failed (%s); falling back to CPU.", exc)

        frame = self._prepare_frame(df)
        if frame.empty:
            logger.warning("No valid rows remain after date normalization")
            return pd.DataFrame()

        diagnostics = FeatureDiagnostics(
            total_rows=len(frame),
            max_missing_rate=self.max_missing_rate,
            max_imputed_rate=self.max_imputed_rate,
        )
        context = FeatureContext(
            ablation_mode=False,
            enabled_groups=self.enable_groups,
            disabled_groups=self.disable_groups,
        )

        result = frame.copy()
        group_columns: Dict[str, List[str]] = {}

        for group in self.feature_groups:
            if not self._should_run_group(group):
                logger.info("Skipping feature group %s", group.name)
                continue
            logger.info("Running feature group %s", group.name)
            result, added = self._run_group(group, result, diagnostics, context)
            group_columns[group.name] = added

        if diagnostics.missing_required_columns:
            logger.warning(
                "Missing required feature inputs: %s",
                list(diagnostics.missing_required_columns.keys())[:20],
            )
        if diagnostics.missing_optional_columns:
            logger.warning(
                "Missing optional feature inputs: %s",
                list(diagnostics.missing_optional_columns.keys())[:20],
            )
        if diagnostics.should_fail():
            logger.error(
                "Feature engineering missingness exceeded configured thresholds. "
                "Returning partial feature set (model training may be degraded). "
                "summary=%s",
                diagnostics.summary(),
            )

        if self.disable_columns:
            drop_cols = [c for c in self.disable_columns if c in result.columns]
            if drop_cols:
                logger.info("Dropping %s pruned formula columns", len(drop_cols))
                result = result.drop(columns=drop_cols)

        feature_cols = [c for c in result.columns if c not in df.columns]
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = pd.to_numeric(result[col], errors='coerce')

        self.last_result = FeatureEngineeringResult(
            group_columns=group_columns,
            diagnostics=diagnostics,
            n_rows=len(result),
            n_features=len(feature_cols),
        )
        logger.info(
            "Feature engineering complete. rows=%s new_features=%s missing_summary=%s",
            len(result),
            len(feature_cols),
            diagnostics.summary(),
        )
        return result

    def benchmark_feature_variants(
        self,
        df: pd.DataFrame,
        target: str = 'PTS',
        val_ratio: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark feature-group and formula variants on a simple validation split."""
        if df is None or df.empty or target not in df.columns:
            return {}

        frame = self._prepare_frame(df)
        if frame.empty:
            return {}

        split_idx = max(1, int(len(frame) * (1 - val_ratio)))
        train_raw = frame.iloc[:split_idx].copy()
        val_raw = frame.iloc[split_idx:].copy()
        if train_raw.empty or val_raw.empty:
            return {}

        variants: Dict[str, Dict[str, object]] = {
            'baseline': {'disable_groups': set(), 'disable_columns': set()},
            'no_matchup': {'disable_groups': {'matchup', 'opponent_strength'}, 'disable_columns': set()},
            'no_context': {'disable_groups': {'context', 'fatigue'}, 'disable_columns': set()},
            'no_target_encoding': {'disable_groups': {'target_encoding', 'league_rank'}, 'disable_columns': set()},
            'formula_raw_only': {
                'disable_groups': set(),
                'disable_columns': {
                    c for c in self._formula_columns_hint()
                },
            },
        }

        scores: Dict[str, Dict[str, float]] = {}
        for variant_name, cfg in variants.items():
            try:
                variant_engineer = build_feature_engineer(
                    rolling_windows=self.rolling_windows,
                    use_gpu=self.use_gpu,
                    enable_groups=self.enable_groups,
                    disable_groups=sorted(cfg['disable_groups']),
                    disable_columns=sorted(cfg['disable_columns']),
                    max_missing_rate=self.max_missing_rate,
                    max_imputed_rate=self.max_imputed_rate,
                )
                train_feat = variant_engineer.create_features(train_raw, is_training=True)
                val_feat = variant_engineer.create_features(val_raw, is_training=True)
                selector = FeatureSelector(targets=self.target_cols)
                schema = selector.fit(train_feat)
                X_train = selector.transform(train_feat, schema, strict=False, fill_value=0.0)
                X_val = selector.transform(val_feat, schema, strict=False, fill_value=0.0)
                y_train = pd.to_numeric(train_feat[target], errors='coerce')
                y_val = pd.to_numeric(val_feat[target], errors='coerce')
                train_mask = y_train.notna()
                val_mask = y_val.notna()
                if train_mask.sum() < 20 or val_mask.sum() < 5:
                    continue
                model = HistGradientBoostingRegressor(random_state=42, max_depth=5, max_iter=200)
                model.fit(X_train.loc[train_mask], y_train.loc[train_mask])
                preds = model.predict(X_val.loc[val_mask])
                scores[variant_name] = {
                    'mae': float(mean_absolute_error(y_val.loc[val_mask], preds)),
                    'n_train': int(train_mask.sum()),
                    'n_val': int(val_mask.sum()),
                }
            except Exception as exc:
                logger.warning("Ablation variant %s failed: %s", variant_name, exc)

        if scores:
            best_name = min(scores, key=lambda k: scores[k]['mae'])
            scores['best'] = {'variant': best_name, 'mae': scores[best_name]['mae']}
        return scores

    def _formula_columns_hint(self) -> List[str]:
        """Columns to prune when comparing raw parts against formulas."""
        hints: List[str] = []
        hints.extend([
            'PACE_ADJ_USAGE',
            'EFF_Z_SCORE',
        ])
        hints.extend([f'ROLL_USG_PCT_{w}' for w in [5, 10]])
        hints.extend([f'ROLL_REB_OPPORTUNITY_{w}' for w in [5, 10]])
        hints.extend([f'ROLL_3PT_FREQ_{w}' for w in [10, 20]])
        hints.extend([f'ROLL_FT_RATE_{w}' for w in [10]])
        hints.extend([f'ROLL_PTS_SHARE_{w}' for w in [10]])
        hints.extend([f'ROLL_TS_PCT_MOMENTUM_{w}' for w in [5, 10]])
        return hints

    def create_features_chunked(self, df: pd.DataFrame, chunk_size: int = 500_000, is_training: bool = True) -> pd.DataFrame:
        """Process features in player-respecting chunks."""
        if df is None or df.empty:
            return pd.DataFrame()
        if len(df) < chunk_size:
            return self.create_features(df, is_training=is_training)

        frame = self._prepare_frame(df)
        if frame.empty:
            return pd.DataFrame()

        player_ids = frame['PLAYER_ID'].unique() if 'PLAYER_ID' in frame.columns else np.arange(len(frame))
        player_sizes = frame.groupby('PLAYER_ID').size() if 'PLAYER_ID' in frame.columns else pd.Series([len(frame)], index=[0])

        chunks = []
        current_ids: List[int] = []
        current_size = 0

        for pid in player_ids:
            pid_size = int(player_sizes[pid]) if 'PLAYER_ID' in frame.columns else len(frame)
            if current_ids and current_size + pid_size > chunk_size:
                chunk_df = frame[frame['PLAYER_ID'].isin(current_ids)].copy()
                chunks.append(self.create_features(chunk_df, is_training=is_training))
                current_ids = []
                current_size = 0
            current_ids.append(pid)
            current_size += pid_size

        if current_ids:
            chunk_df = frame[frame['PLAYER_ID'].isin(current_ids)].copy()
            chunks.append(self.create_features(chunk_df, is_training=is_training))

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks, ignore_index=True)
        return combined

    def get_group_columns(self) -> Dict[str, List[str]]:
        """Return the columns added by each feature group from the last run."""
        return dict(self.last_result.group_columns)

    def get_diagnostics(self) -> FeatureDiagnostics:
        return self.last_result.diagnostics


def build_feature_engineer(
    rolling_windows: Optional[List[int]] = None,
    use_gpu: Optional[bool] = None,
    enable_groups: Optional[Sequence[str]] = None,
    disable_groups: Optional[Sequence[str]] = None,
    disable_columns: Optional[Sequence[str]] = None,
    max_missing_rate: float = 0.35,
    max_imputed_rate: float = 0.40,
    cache_dir: Optional[Union[str, Path]] = None,
) -> FeatureEngineer:
    """Build a FeatureEngineer while tolerating older constructor signatures.

    The current implementation accepts ``disable_groups`` directly, but older
    runtime checkouts may still omit that keyword. In those cases we construct
    the object with the supported kwargs and then apply the group filter as a
    post-init attribute so the training path still behaves the same.
    """

    init_kwargs = {
        'rolling_windows': rolling_windows,
        'use_gpu': use_gpu,
        'enable_groups': enable_groups,
        'disable_groups': disable_groups,
        'disable_columns': disable_columns,
        'max_missing_rate': max_missing_rate,
        'max_imputed_rate': max_imputed_rate,
        'cache_dir': cache_dir,
    }
    signature = inspect.signature(FeatureEngineer.__init__)
    supported_kwargs = {
        name
        for name in signature.parameters
        if name != 'self'
    }
    filtered_kwargs = {
        key: value
        for key, value in init_kwargs.items()
        if value is not None and key in supported_kwargs
    }
    engineer = FeatureEngineer(**filtered_kwargs)

    if disable_groups is not None and 'disable_groups' not in supported_kwargs:
        engineer.disable_groups = set(disable_groups)
    if disable_columns is not None and 'disable_columns' not in supported_kwargs:
        engineer.disable_columns = set(disable_columns)
    if enable_groups is not None and 'enable_groups' not in supported_kwargs:
        engineer.enable_groups = set(enable_groups)

    return engineer
