"""Append-only pregame prediction ledger and postgame reconciliation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


KEY_COLUMNS = ("MODEL_VERSION", "GAME_ID", "PLAYER_ID", "STAT")
TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


class PredictionLedger:
    """Persist predictions without allowing postgame values into the snapshot."""

    def __init__(self, path: str | Path = "data/evaluation/prediction_history.parquet") -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)

    def append(self, rows: Iterable[Dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if frame.empty:
            return self.load()
        missing = [column for column in KEY_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Prediction ledger rows missing key columns: {missing}")
        if "GENERATED_AT" not in frame.columns:
            frame["GENERATED_AT"] = pd.Timestamp.utcnow().isoformat()
        frame["STAT"] = frame["STAT"].astype(str).str.upper()
        frame = frame[frame["STAT"].isin(TARGETS)].copy()
        existing = self.load()
        combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame
        # The first pregame snapshot is immutable; a later duplicate cannot
        # replace it with a post-hoc value.
        combined = combined.drop_duplicates(list(KEY_COLUMNS), keep="first")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        return combined

    def reconcile_actuals(
        self,
        actuals: pd.DataFrame,
        *,
        actual_columns: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Fill ACTUAL values by key; never overwrite pregame predictions."""
        ledger = self.load()
        if ledger.empty:
            return ledger
        actuals = actuals.copy()
        for column in ("GAME_ID", "PLAYER_ID"):
            if column not in actuals.columns:
                raise ValueError(f"Actuals missing required column: {column}")
        actual_columns = actual_columns or {stat: stat for stat in TARGETS}
        long_rows = []
        for stat, column in actual_columns.items():
            if column not in actuals.columns:
                continue
            subset = actuals[["GAME_ID", "PLAYER_ID", column]].copy()
            subset["STAT"] = stat.upper()
            subset = subset.rename(columns={column: "ACTUAL"})
            long_rows.append(subset)
        if not long_rows:
            return ledger
        actual_long = pd.concat(long_rows, ignore_index=True)
        actual_long["STAT"] = actual_long["STAT"].astype(str).str.upper()
        actual_long = actual_long.drop_duplicates(["GAME_ID", "PLAYER_ID", "STAT"], keep="last")
        merged = ledger.merge(
            actual_long,
            on=["GAME_ID", "PLAYER_ID", "STAT"],
            how="left",
            suffixes=("", "_NEW"),
        )
        if "ACTUAL_NEW" in merged:
            merged["ACTUAL"] = merged.get("ACTUAL", pd.Series(index=merged.index)).where(
                merged.get("ACTUAL", pd.Series(index=merged.index)).notna(),
                merged["ACTUAL_NEW"],
            )
            merged = merged.drop(columns=["ACTUAL_NEW"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return merged
