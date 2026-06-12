"""Residual dataset structures and helpers."""

from dataclasses import dataclass, asdict
from typing import Optional, List

import pandas as pd


@dataclass
class ResidualTrainingRow:
    """A single row in the residual correction training set."""

    game_id: str
    game_date: str
    player_id: str
    player_name: str
    team_id: str
    opponent: str
    stat: str

    base_prediction: float
    actual: float
    error: float

    model_fold: str
    model_version: Optional[str]
    data_quality: Optional[str]
    feature_cutoff_date: str


def build_residual_dataframe(rows: List[ResidualTrainingRow]) -> pd.DataFrame:
    """Convert a list of residual rows into a canonical DataFrame."""
    if not rows:
        return pd.DataFrame(columns=[
            "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
            "OPPONENT", "STAT", "BASE_PREDICTION", "ACTUAL", "ERROR",
            "MODEL_FOLD", "FEATURE_CUTOFF_DATE", "MODEL_VERSION", "DATA_QUALITY",
        ])
    records = []
    for row in rows:
        records.append({
            "GAME_ID": row.game_id,
            "GAME_DATE": row.game_date,
            "PLAYER_ID": row.player_id,
            "PLAYER_NAME": row.player_name,
            "TEAM_ID": row.team_id,
            "OPPONENT": row.opponent,
            "STAT": row.stat,
            "BASE_PREDICTION": row.base_prediction,
            "ACTUAL": row.actual,
            "ERROR": row.error,
            "MODEL_FOLD": row.model_fold,
            "FEATURE_CUTOFF_DATE": row.feature_cutoff_date,
            "MODEL_VERSION": row.model_version,
            "DATA_QUALITY": row.data_quality,
        })
    return pd.DataFrame(records)
