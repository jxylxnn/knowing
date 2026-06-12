from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd

from src.contracts.errors import FeatureSchemaContractError


@dataclass(frozen=True)
class FeatureSchemaReport:
    missing_features: list[str]
    extra_features: list[str]
    non_numeric_features: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_features and not self.non_numeric_features


def load_expected_feature_cols(models_dir: Path) -> list[str]:
    path = Path(models_dir) / "feature_cols.pkl"

    if not path.exists():
        raise FeatureSchemaContractError(f"Missing feature_cols.pkl: {path}")

    try:
        with path.open("rb") as f:
            feature_cols = pickle.load(f)
    except Exception as exc:
        raise FeatureSchemaContractError(f"Could not load feature_cols.pkl: {path}") from exc

    if not isinstance(feature_cols, list) or not all(isinstance(c, str) for c in feature_cols):
        raise FeatureSchemaContractError("feature_cols.pkl must contain list[str]")

    return feature_cols


def validate_feature_frame(df: pd.DataFrame, expected_feature_cols: list[str], *, allow_extra: bool = True) -> FeatureSchemaReport:
    actual = set(df.columns)
    expected = set(expected_feature_cols)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected) if not allow_extra else []

    non_numeric = []
    for col in expected_feature_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    report = FeatureSchemaReport(
        missing_features=missing,
        extra_features=extra,
        non_numeric_features=sorted(non_numeric),
    )

    if not report.ok:
        raise FeatureSchemaContractError(
            "Feature schema contract failed:\n"
            f"Missing features: {report.missing_features[:30]}\n"
            f"Non-numeric features: {report.non_numeric_features[:30]}\n"
            f"Extra features: {report.extra_features[:30]}"
        )

    return report


def align_feature_frame(df: pd.DataFrame, expected_feature_cols: list[str]) -> pd.DataFrame:
    validate_feature_frame(df, expected_feature_cols, allow_extra=True)
    return df.loc[:, expected_feature_cols].copy()
