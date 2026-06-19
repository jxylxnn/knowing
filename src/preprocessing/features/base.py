"""Shared primitives for modular feature engineering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureContext:
    """Shared state passed between feature groups."""

    league_priors: Dict[str, float] = field(
        default_factory=lambda: {
            'PTS': 10.0,
            'REB': 4.5,
            'AST': 2.5,
            'STL': 0.8,
            'BLK': 0.6,
            'TOV': 1.5,
            'MIN': 24.0,
            'FGA': 9.0,
            'FGM': 4.5,
            'FTA': 3.0,
            'FTM': 2.2,
            'FG3A': 4.0,
            'FG3M': 1.4,
            'OREB': 1.5,
            'DREB': 3.5,
            'TEAM_PACE': 100.0,
            'TS_PCT': 0.56,
            'EFG_PCT': 0.52,
            '3PT_PCT': 0.36,
            'AST_TOV': 1.4,
            'USAGE': 0.18,
            'REB_OPP': 0.48,
            'PTS_SHARE': 0.22,
        }
    )
    enabled_groups: Optional[Set[str]] = None
    disabled_groups: Optional[Set[str]] = None
    ablation_mode: bool = False
    schema_version: str = 'feature_schema_v3'


@dataclass
class FeatureDiagnostics:
    """Tracks missing inputs and imputed outputs explicitly."""

    total_rows: int = 0
    missing_required_columns: Dict[str, int] = field(default_factory=dict)
    missing_optional_columns: Dict[str, int] = field(default_factory=dict)
    imputed_values: Dict[str, int] = field(default_factory=dict)
    group_missing_rows: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    max_missing_rate: float = 0.35
    max_imputed_rate: float = 0.40

    def record_column_missing(self, group: str, column: str, required: bool, count: int = 1) -> None:
        key = f'{group}.{column}'
        if required:
            self.missing_required_columns[key] = self.missing_required_columns.get(key, 0) + count
        else:
            self.missing_optional_columns[key] = self.missing_optional_columns.get(key, 0) + count

    def record_imputation(self, column: str, count: int) -> None:
        self.imputed_values[column] = self.imputed_values.get(column, 0) + int(count)

    def record_group_missing(self, group: str, missing_rows: int) -> None:
        self.group_missing_rows[group] = self.group_missing_rows.get(group, 0) + int(missing_rows)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(message)

    def summary(self) -> Dict[str, int]:
        return {
            'total_rows': self.total_rows,
            'missing_required_columns': len(self.missing_required_columns),
            'missing_optional_columns': len(self.missing_optional_columns),
            'imputed_columns': len(self.imputed_values),
            'groups_with_missing': len(self.group_missing_rows),
        }

    def should_fail(self) -> bool:
        if self.total_rows <= 0:
            return False
        missing_ratio = sum(self.group_missing_rows.values()) / float(self.total_rows)
        if self.missing_required_columns and missing_ratio > self.max_missing_rate:
            return True
        return False


class FeatureGroup(ABC):
    """Base class for a single-purpose feature group."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable feature group name."""

    @property
    def required_columns(self) -> List[str]:
        return []

    @property
    def optional_columns(self) -> List[str]:
        return []

    @abstractmethod
    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        """Create this feature group."""

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Best-effort feature name discovery for schema bookkeeping."""
        return [c for c in df.columns if c.startswith(self.name.upper())]

    def external_files(self) -> List[str]:
        """Declare on-disk files this group reads that are NOT in the input DataFrame.

        The FeatureEngineer folds these into its feature cache key (by path +
        size + mtime) so that cached features are invalidated whenever the
        external data changes. Groups with no external dependencies return an
        empty list (the default).
        """
        return []

    def _check_columns(self, df: pd.DataFrame, diagnostics: Optional[FeatureDiagnostics]) -> None:
        missing_required = [c for c in self.required_columns if c not in df.columns]
        missing_optional = [c for c in self.optional_columns if c not in df.columns]
        if diagnostics is not None:
            for col in missing_required:
                diagnostics.record_column_missing(self.name, col, required=True, count=len(df))
            for col in missing_optional:
                diagnostics.record_column_missing(self.name, col, required=False, count=len(df))
            if missing_required:
                diagnostics.record_group_missing(self.name, len(df))
                diagnostics.warn(
                    f'{self.name}: missing required columns {missing_required}; using safe fallbacks.'
                )
            elif missing_optional:
                diagnostics.warn(
                    f'{self.name}: missing optional columns {missing_optional}; using safe fallbacks.'
                )


def add_missing_flag(df: pd.DataFrame, flag_name: str, mask: pd.Series) -> pd.DataFrame:
    """Attach an integer missingness flag."""
    df = df.copy()
    df[flag_name] = mask.astype(int)
    return df


def fill_series_with_prior(series: pd.Series, prior: float, diagnostics: Optional[FeatureDiagnostics] = None, column_name: Optional[str] = None) -> pd.Series:
    """Fill missing values with a deterministic prior and record the imputation."""
    missing = series.isna()
    if diagnostics is not None and column_name is not None:
        diagnostics.record_imputation(column_name, int(missing.sum()))
    return series.fillna(prior)


def normalize_output_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure output columns exist and are numeric."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
