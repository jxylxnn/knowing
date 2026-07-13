from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any, Iterable, Sequence

import pandas as pd

from src.contracts.errors import FeatureSchemaContractError


FEATURE_SCHEMA_VERSION = "feature_schema_v4"

# Targets are kept in the training frame as labels, but they can never be
# members of a persisted feature schema.
TARGET_COLUMNS = frozenset({"PTS", "REB", "AST", "STL", "BLK", "TOV"})

# Same-game player outcomes other than the six canonical targets.  These are
# useful labels/reporting values, not point-in-time features.
RAW_CURRENT_GAME_COLUMNS = frozenset(
    {
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "BLKA",
        "PF",
        "PFD",
        "PLUS_MINUS",
        "NBA_FANTASY_PTS",
        "DD2",
        "TD3",
        "WNBA_FANTASY_PTS",
        "MIN_SEC",
        "FANTASY_PTS",
        "AVAILABLE_FLAG",
        "WL",
    }
)

CURRENT_GAME_TEAM_OUTCOMES = frozenset(
    f"{column}_TEAM" for column in TARGET_COLUMNS | RAW_CURRENT_GAME_COLUMNS
)
FORBIDDEN_EXACT_COLUMNS = TARGET_COLUMNS | RAW_CURRENT_GAME_COLUMNS | CURRENT_GAME_TEAM_OUTCOMES

# Keep this policy stricter than the current producer list.  A future data
# source must not bypass it by inventing a variant such as PTS_TEAM_CURRENT or
# BOX_PTS_TEAM.  Shifted/rolling context remains valid because it is named
# TEAM_PTS_ROLL_10 rather than PTS_TEAM.
FORBIDDEN_PATTERNS = (
    re.compile(
        r"^(?:[A-Z0-9]+_)*(?:"
        + "|".join(
            sorted(
                TARGET_COLUMNS | RAW_CURRENT_GAME_COLUMNS,
                key=len,
                reverse=True,
            )
        )
        + r")_TEAM(?:$|_[A-Z0-9_]+$)"
    ),
    re.compile(
        r"^(?:CURRENT|LIVE|GAME)_(?:"
        + "|".join(sorted(TARGET_COLUMNS | RAW_CURRENT_GAME_COLUMNS, key=len, reverse=True))
        + r")(?:$|_[A-Z0-9_]+$)"
    ),
)


def is_forbidden_feature(name: str) -> bool:
    """Return whether ``name`` can contain an unsafe same-game outcome."""

    if not isinstance(name, str):
        return False
    return name in FORBIDDEN_EXACT_COLUMNS or any(
        pattern.fullmatch(name) for pattern in FORBIDDEN_PATTERNS
    )


def validate_feature_names(names: Sequence[str], *, context: str = "feature schema") -> None:
    """Reject malformed, duplicate, or semantically forbidden feature names."""

    values = list(names)
    non_strings = [value for value in values if not isinstance(value, str)]
    duplicates = sorted({value for value in values if values.count(value) > 1 and isinstance(value, str)})
    forbidden = sorted({value for value in values if is_forbidden_feature(value)})
    if non_strings or duplicates or forbidden:
        details: list[str] = []
        if non_strings:
            details.append(f"non-string names={non_strings[:10]}")
        if duplicates:
            details.append(f"duplicate names={duplicates[:20]}")
        if forbidden:
            details.append(f"forbidden current-game outcomes={forbidden}")
        raise FeatureSchemaContractError(f"{context} is unsafe: " + "; ".join(details))


def _load_pickle(path: Path) -> Any:
    try:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise FeatureSchemaContractError(f"Could not load feature schema: {path}") from exc


def _extract_feature_columns(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    if isinstance(value, dict):
        for key in ("feature_cols", "features", "columns"):
            columns = _extract_feature_columns(value.get(key))
            if columns is not None:
                return columns
    for attribute in ("feature_cols", "features", "columns"):
        columns = _extract_feature_columns(getattr(value, attribute, None))
        if columns is not None:
            return columns
    return None


def _schema_version(value: Any) -> str | None:
    if isinstance(value, dict):
        version = value.get("version") or value.get("schema_version")
    else:
        version = getattr(value, "version", None) or getattr(value, "schema_version", None)
    return str(version) if version is not None else None


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

    feature_cols = _extract_feature_columns(_load_pickle(path))
    if feature_cols is None:
        raise FeatureSchemaContractError("feature_cols.pkl must contain list[str]")
    validate_feature_names(feature_cols, context=str(path))

    return feature_cols


def load_feature_schema(path: Path) -> tuple[list[str], str | None]:
    """Load the inspectable feature schema columns and version from a pickle."""

    schema_path = Path(path)
    if not schema_path.exists():
        raise FeatureSchemaContractError(f"Missing feature schema: {schema_path}")
    payload = _load_pickle(schema_path)
    columns = _extract_feature_columns(payload)
    if columns is None:
        raise FeatureSchemaContractError(
            f"{schema_path} must contain a feature_cols list or schema object"
        )
    validate_feature_names(columns, context=str(schema_path))
    return columns, _schema_version(payload)


def validate_feature_frame(df: pd.DataFrame, expected_feature_cols: list[str], *, allow_extra: bool = True) -> FeatureSchemaReport:
    validate_feature_names(expected_feature_cols)
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
