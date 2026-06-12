from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.contracts.errors import ScheduleContractError

REQUIRED_SCHEDULE_COLUMNS = {"GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM"}


@dataclass(frozen=True)
class ScheduleGame:
    game_id: str
    game_date: str
    home_team: str
    away_team: str


def validate_schedule_frame(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SCHEDULE_COLUMNS - set(df.columns))
    if missing:
        raise ScheduleContractError("Schedule output missing required columns:\n" + "\n".join(f"- {col}" for col in missing))

    if df.empty:
        raise ScheduleContractError("Schedule output is empty.")

    null_columns = [col for col in REQUIRED_SCHEDULE_COLUMNS if df[col].isna().any()]
    if null_columns:
        raise ScheduleContractError("Schedule output has nulls in required columns:\n" + "\n".join(f"- {col}" for col in null_columns))


def normalize_schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_schedule_frame(df)

    normalized = df.copy()
    normalized["GAME_ID"] = normalized["GAME_ID"].astype(str)
    normalized["HOME_TEAM"] = normalized["HOME_TEAM"].astype(str).str.upper().str.strip()
    normalized["AWAY_TEAM"] = normalized["AWAY_TEAM"].astype(str).str.upper().str.strip()

    try:
        normalized["GAME_DATE"] = pd.to_datetime(normalized["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    except Exception as exc:
        raise ScheduleContractError("GAME_DATE could not be parsed as dates.") from exc

    validate_schedule_frame(normalized)
    return normalized


def schedule_rows_to_games(df: pd.DataFrame) -> list[ScheduleGame]:
    normalized = normalize_schedule_frame(df)

    return [
        ScheduleGame(
            game_id=str(row.GAME_ID),
            game_date=str(row.GAME_DATE),
            home_team=str(row.HOME_TEAM),
            away_team=str(row.AWAY_TEAM),
        )
        for row in normalized.itertuples(index=False)
    ]
