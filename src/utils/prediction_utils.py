"""Shared prediction and training utilities.

This module contains utility classes extracted from ModelManager and DataPipeline
to reduce code duplication and improve maintainability.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    'FeatureSchema',
    'TemporalWeightCalculator',
    'FallbackPredictor',
    'FeatureSelector',
    'TargetPreprocessor',
]


@dataclass
class FeatureSchema:
    """Persisted feature schema shared by training and inference."""

    feature_cols: List[str]
    categorical_cols: List[str] = field(default_factory=list)
    group_columns: Dict[str, List[str]] = field(default_factory=dict)
    dtype_map: Dict[str, str] = field(default_factory=dict)
    version: str = 'feature_schema_v3'
    schema_hash: str = ''

    def __post_init__(self) -> None:
        if not self.schema_hash:
            payload = json.dumps(
                {
                    'feature_cols': self.feature_cols,
                    'categorical_cols': self.categorical_cols,
                    'group_columns': self.group_columns,
                    'version': self.version,
                },
                sort_keys=True,
                default=str,
            ).encode('utf-8')
            self.schema_hash = hashlib.sha256(payload).hexdigest()


class TemporalWeightCalculator:
    """Calculate temporal decay weights for training samples.
    
    Games in the last 30 days get weight 1.0.
    Games 6 months ago get weight ~0.2.
    This teaches models to focus on recent performance.
    
    Attributes:
        lambda_decay: Decay rate for exponential weighting.
        min_weight: Minimum weight to avoid ignoring old data entirely.
    """
    
    def __init__(self, lambda_decay: float = 0.023, min_weight: float = 0.1):
        """Initialize the weight calculator.
        
        Args:
            lambda_decay: Exponential decay rate (default ~0.5 weight after 30 days).
            min_weight: Minimum weight floor for old games.
        """
        self.lambda_decay = lambda_decay
        self.min_weight = min_weight
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate sample weights based on recency.
        
        Args:
            df: DataFrame with 'GAME_DATE' column.
            
        Returns:
            Array of weights for each sample.
        """
        if 'GAME_DATE' not in df.columns:
            return np.ones(len(df))
        
        max_date = df['GAME_DATE'].max()
        days_ago = (max_date - df['GAME_DATE']).dt.days
        
        weights = np.exp(-self.lambda_decay * days_ago)
        weights = np.clip(weights, self.min_weight, 1.0)
        
        return weights


class FallbackPredictor:
    """Provides fallback predictions when models are unavailable.
    
    Uses historical averages from rolling statistics when available,
    falling back to league averages when no history exists.
    
    Attributes:
        league_averages: Default values for each stat when no history is available.
    """
    
    LEAGUE_AVERAGES: Dict[str, float] = {
        'PTS': 10.0,
        'REB': 4.5,
        'AST': 2.5,
        'STL': 0.8,
        'BLK': 0.6,
        'TOV': 1.5
    }
    
    FALLBACK_COLUMNS: Dict[str, List[str]] = {
        'PTS': ['ROLL_PTS_AVG_10', 'ROLL_PTS_AVG_20', 'PTS_EWMA_5', 'PTS'],
        'REB': ['ROLL_REB_AVG_10', 'ROLL_REB_AVG_20', 'REB_EWMA_5', 'REB'],
        'AST': ['ROLL_AST_AVG_10', 'ROLL_AST_AVG_20', 'AST_EWMA_5', 'AST'],
        'STL': ['ROLL_STL_AVG_10', 'ROLL_STL_AVG_20', 'STL_EWMA_5', 'STL'],
        'BLK': ['ROLL_BLK_AVG_10', 'ROLL_BLK_AVG_20', 'BLK_EWMA_5', 'BLK'],
        'TOV': ['ROLL_TOV_AVG_10', 'ROLL_TOV_AVG_20', 'TOV_EWMA_5', 'TOV']
    }
    
    def __init__(self, league_averages: Optional[Dict[str, float]] = None):
        """Initialize with optional custom league averages.
        
        Args:
            league_averages: Custom default values. Uses defaults if None.
        """
        self.league_averages = league_averages or self.LEAGUE_AVERAGES.copy()
    
    def predict(self, df: pd.DataFrame, targets: Optional[List[str]] = None) -> Dict[str, float]:
        """Generate fallback predictions for all targets.
        
        Args:
            df: DataFrame with player context (single row).
            targets: List of targets to predict. Uses all if None.
            
        Returns:
            Dictionary mapping target names to predicted values.
        """
        targets = targets or list(self.league_averages.keys())
        predictions = {}
        
        for target in targets:
            predictions[target] = self._get_fallback_value(df, target)
        
        return predictions
    
    def predict_batch(
        self, 
        df: pd.DataFrame, 
        targets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate fallback predictions for multiple players.
        
        Args:
            df: DataFrame with player contexts (multiple rows).
            targets: List of targets to predict. Uses all if None.
            
        Returns:
            DataFrame with predictions for each target.
        """
        targets = targets or list(self.league_averages.keys())
        predictions = {}
        
        for target in targets:
            predictions[target] = self._get_fallback_values_batch(df, target)
        
        return pd.DataFrame(predictions, index=df.index)
    
    def _get_fallback_value(self, df: pd.DataFrame, target: str) -> float:
        """Get fallback value for a single target from DataFrame.
        
        Args:
            df: Single-row DataFrame with player context.
            target: Target stat name.
            
        Returns:
            Fallback prediction value.
        """
        fallback_cols = self.FALLBACK_COLUMNS.get(target, [f'ROLL_{target}_AVG_10', target])
        
        for col in fallback_cols:
            if col in df.columns:
                val = df[col].iloc[0] if len(df) > 0 else None
                if pd.notna(val) and val > 0:
                    return float(val)
        
        return self.league_averages.get(target, 0.0)
    
    def _get_fallback_values_batch(self, df: pd.DataFrame, target: str) -> np.ndarray:
        """Get fallback values for a target across all rows.
        
        Args:
            df: Multi-row DataFrame with player contexts.
            target: Target stat name.
            
        Returns:
            Array of fallback prediction values.
        """
        fallback_cols = self.FALLBACK_COLUMNS.get(target, [f'ROLL_{target}_AVG_10', target])
        league_avg = self.league_averages.get(target, 0.0)
        
        values = np.full(len(df), league_avg)
        
        for col in fallback_cols:
            if col in df.columns:
                mask = (values == league_avg) & df[col].notna() & (df[col] > 0)
                values[mask] = df.loc[mask, col].values
        
        return values


class FeatureSelector:
    """Master selector shared by training and inference."""

    SAFE_PREFIXES = (
        'ROLL_',
        'EWMA_',
        'VS_OPP_',
        'RAW_',
        'LEAGUE_PCT_',
        'ARCHETYPE_',
        'SIMILARITY_TO_',
        'MIN_CONF_',
        'RECENCY_',
        'LINEUP_',
        'INJURY_OPP_',
        'TEAMMATE_',
        'DEF_POS_',
        'SCHED_',
    )
    SAFE_KEYWORDS = (
        'TREND',
        'BAYESIAN',
        'PACE',
        '_TE',
        '_SHARE',
        'ROLE_INDEX',
        'MISSING_',
        'IMPUTED_',
        'COLD_START',
        'FATIGUE',
        'EFF_Z_SCORE',
        'HOT_STREAK',
        'COLD_STREAK',
        'POTENTIAL',
        'B2B_IMPACT',
        'USG_PCT',
        'REB_OPPORTUNITY',
        'FT_RATE',
        'TS_PCT_MOMENTUM',
        'DEF_MATCHUP',
        'OPP_DEF',
        'EST_POSS',
        'PACE_FACTOR',
    )
    SAFE_EXACT = (
        'IS_HOME',
        'REST_DAYS',
        'IS_B2B',
        'FATIGUE_SCORE',
        'DAYS_SINCE_LAST',
        'MINS_LAST_3',
        'MINS_LAST_7',
        'CONTEXT_COLD_START',
        'TEAM_PACE_10',
        'PACE_FACTOR',
        'STAR_TEAMMATE_OUT',
    )
    EXCLUDE_ALWAYS = {
        'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
        'GAME_ID', 'GAME_DATE', 'MATCHUP', 'OPPONENT_ID', 'OPPONENT_ABBR',
        'WL', 'SEASON_ID', 'VIDEO_AVAILABLE', 'REST_BUCKET',
    }
    RAW_BOX_SCORE = {
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FGA', 'FGM', 'FTA',
        'FTM', 'FG3A', 'FG3M', 'OREB', 'DREB',
    }

    def __init__(self, targets: Optional[List[str]] = None):
        self.targets = targets or ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        self.feature_schema: Optional[FeatureSchema] = None

    def _is_numeric(self, series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    def _is_safe_feature(self, col: str) -> bool:
        if col in self.EXCLUDE_ALWAYS or col in self.targets or col in self.RAW_BOX_SCORE:
            return False
        if col in self.SAFE_EXACT:
            return True
        if col.startswith(self.SAFE_PREFIXES):
            return True
        if any(keyword in col for keyword in self.SAFE_KEYWORDS):
            return True
        if (col.startswith('TEAM_') or col.startswith('OPP_TEAM_')) and (
            'ROLL_' in col or 'PACE' in col or 'DEF' in col
        ):
            return True
        return False

    def select_features(self, df: pd.DataFrame) -> List[str]:
        schema = self.fit(df)
        return schema.feature_cols

    def fit(
        self,
        df: pd.DataFrame,
        group_columns: Optional[Dict[str, List[str]]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
    ) -> FeatureSchema:
        """Build and cache the canonical feature schema."""
        candidate_cols = [
            c for c in df.columns
            if c not in self.EXCLUDE_ALWAYS and c not in self.targets and self._is_numeric(df[c])
        ]

        safe_features = [c for c in candidate_cols if self._is_safe_feature(c)]

        if group_columns:
            ordered: List[str] = []
            seen = set()
            for cols in group_columns.values():
                for col in cols:
                    if col in safe_features and col not in seen:
                        ordered.append(col)
                        seen.add(col)
            for col in safe_features:
                if col not in seen:
                    ordered.append(col)
            safe_features = ordered

        cat_cols = [c for c in (categorical_columns or ['PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID']) if c in df.columns]
        dtype_map = {col: str(df[col].dtype) for col in safe_features if col in df.columns}

        self.feature_schema = FeatureSchema(
            feature_cols=safe_features,
            categorical_cols=cat_cols,
            group_columns=group_columns or {},
            dtype_map=dtype_map,
        )
        logger.info("Selected %s canonical features", len(safe_features))
        return self.feature_schema

    def save_schema(self, schema: FeatureSchema, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(schema, path)

    def load_schema(self, path: Path) -> Optional[FeatureSchema]:
        if not Path(path).exists():
            return None
        schema = joblib.load(path)
        if isinstance(schema, list):
            schema = FeatureSchema(feature_cols=list(schema))
        self.feature_schema = schema
        return schema

    def validate_exact_match(self, df: pd.DataFrame, schema: Optional[FeatureSchema] = None) -> None:
        schema = schema or self.feature_schema
        if schema is None:
            raise ValueError("No feature schema available for validation")

        missing = [c for c in schema.feature_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in schema.feature_cols and c not in self.EXCLUDE_ALWAYS]
        if missing or extra:
            raise ValueError(
                f"Feature schema mismatch. missing={missing[:10]} extra={extra[:10]} schema_hash={schema.schema_hash}"
            )

    def align_frame(
        self,
        df: pd.DataFrame,
        schema: Optional[FeatureSchema] = None,
        *,
        strict: bool = False,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        """Align a frame to the canonical schema."""
        schema = schema or self.feature_schema
        if schema is None:
            raise ValueError("No feature schema available")

        frame = df.copy()
        missing = [c for c in schema.feature_cols if c not in frame.columns]
        extra = [c for c in frame.columns if c not in schema.feature_cols]

        if missing:
            logger.warning("Missing feature columns: %s", missing[:20])
            if strict:
                raise ValueError(f"Missing feature columns: {missing}")
            for col in missing:
                frame[col] = fill_value
        if extra:
            logger.info("Dropping %s non-schema columns", len(extra))

        aligned = frame.reindex(columns=list(schema.feature_cols), fill_value=fill_value)
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors='coerce').fillna(fill_value)
        return aligned

    def transform(
        self,
        df: pd.DataFrame,
        schema: Optional[FeatureSchema] = None,
        *,
        strict: bool = False,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        return self.align_frame(df, schema=schema, strict=strict, fill_value=fill_value)


class TargetPreprocessor:
    """Preprocesses target variables to handle outliers.
    
    Caps target values at player-specific percentiles to prevent
    extreme values from skewing model training.
    """
    
    def __init__(self, percentile: float = 0.99, targets: Optional[List[str]] = None):
        """Initialize with capping parameters.
        
        Args:
            percentile: Percentile to use for capping (e.g., 0.99 for 99th).
            targets: List of target column names.
        """
        self.percentile = percentile
        self.targets = targets or ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap target values at player-specific percentiles.
        
        Args:
            df: DataFrame with target columns.
            
        Returns:
            DataFrame with capped targets (adds {TARGET}_CLEAN columns).
        """
        df = df.copy()
        
        for stat in self.targets:
            if stat not in df.columns:
                continue
            
            player_caps = df.groupby('PLAYER_ID')[stat].quantile(self.percentile)
            player_caps_mapped = df['PLAYER_ID'].map(player_caps)
            
            global_cap = df[stat].quantile(self.percentile)
            player_caps_mapped = player_caps_mapped.fillna(global_cap)
            
            df[f'{stat}_CLEAN'] = df[stat].clip(upper=player_caps_mapped)
        
        return df
