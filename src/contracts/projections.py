from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.contracts.errors import ProjectionSchemaContractError

BASE_COLUMNS = {"PLAYER_NAME", "TEAM", "OPPONENT", "DATA_QUALITY"}
STATS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")

REQUIRED_STAT_COLUMNS = set()
for stat in STATS:
    REQUIRED_STAT_COLUMNS.update(
        {
            stat,
            f"{stat}_P10",
            f"{stat}_P50",
            f"{stat}_P90",
            f"{stat}_STD",
            f"{stat}_SKEW",
            f"{stat}_ZERO_PROB",
            f"{stat}_LAMBDA",
        }
    )

REQUIRED_PROJECTION_COLUMNS = BASE_COLUMNS | REQUIRED_STAT_COLUMNS
OPTIONAL_INTERVAL_COLUMNS = {
    column
    for stat in STATS
    for column in (
        f"{stat}_INTERVAL_80_LOW",
        f"{stat}_INTERVAL_80_HIGH",
        f"{stat}_INTERVAL_90_LOW",
        f"{stat}_INTERVAL_90_HIGH",
        f"{stat}_CONFIDENCE_SCORE",
        f"{stat}_CONFIDENCE",
    )
}
OPTIONAL_CORRECTION_COLUMNS = {
    f"{stat}_{suffix}"
    for stat in STATS
    for suffix in ("CORRECTED", "BASE", "RESIDUAL_CORRECTION")
}
REQUIRED_CONFIDENCE_COLUMNS = {f"{stat}_CONFIDENCE" for stat in STATS}


def validate_projection_frame(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PROJECTION_COLUMNS - set(df.columns))

    if missing:
        raise ProjectionSchemaContractError(
            "Projection export schema is missing required columns:\n"
            + "\n".join(f"- {col}" for col in missing)
        )

    quality_values = set(df["DATA_QUALITY"].dropna().astype(str).unique())
    invalid_quality = quality_values - {"FULL", "DEGRADED_FALLBACK", "DEGRADED_MISSING"}
    if invalid_quality:
        raise ProjectionSchemaContractError(f"Invalid DATA_QUALITY values: {sorted(invalid_quality)}")

    confidence_values = set()
    for col in REQUIRED_CONFIDENCE_COLUMNS & set(df.columns):
        confidence_values.update(df[col].dropna().astype(str).unique())
    invalid_confidence = confidence_values - {"HIGH", "MEDIUM", "LOW", "NO_EDGE"}
    if invalid_confidence:
        raise ProjectionSchemaContractError(
            f"Invalid confidence labels: {sorted(invalid_confidence)}"
        )

    numeric_cols = sorted(REQUIRED_STAT_COLUMNS | (OPTIONAL_INTERVAL_COLUMNS - REQUIRED_CONFIDENCE_COLUMNS) | OPTIONAL_CORRECTION_COLUMNS)
    non_numeric = [
        col for col in numeric_cols
        if col in df.columns
        and df[col].notna().any()
        and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ProjectionSchemaContractError(
            "Projection column is not numeric:\n"
            + "\n".join(f"- {col}" for col in non_numeric)
        )

    # Optional interval columns may be omitted by older projection exports,
    # but when present their bounds must be ordered for every complete row.
    for stat in STATS:
        for confidence in (80, 90):
            low_col = f"{stat}_INTERVAL_{confidence}_LOW"
            high_col = f"{stat}_INTERVAL_{confidence}_HIGH"
            if low_col not in df.columns or high_col not in df.columns:
                continue
            mask = df[low_col].notna() & df[high_col].notna()
            if (df.loc[mask, low_col] > df.loc[mask, high_col]).any():
                raise ProjectionSchemaContractError(
                    f"{low_col} must not exceed {high_col}"
                )


def validate_projection_csv(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        raise ProjectionSchemaContractError(f"Projection CSV does not exist: {path}")

    df = pd.read_csv(path)
    validate_projection_frame(df)
