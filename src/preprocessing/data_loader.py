import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic cleaning of NBA datasets."""
    
    def __init__(self, players_path: str, games_path: str):
        self.players_path = players_path
        self.games_path = games_path
        self.players_df = None
        self.games_df = None

    def _ensure_columns(self, df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
        """Ensure a DataFrame has all required columns with sensible defaults."""
        df = df.copy()
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        return df

    def _parse_minutes(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)): return float(val)
        s = str(val).strip()
        if s == '' or s.lower() in {'nan','none'}: return np.nan
        if ':' in s:
            parts = s.split(':')
            if len(parts) == 2: return float(parts[0]) + float(parts[1])/60.0
        try: return float(s)
        except ValueError: return np.nan

    def _add_team_rolling_features(self) -> None:
        """Compute safe team-level rolling context from prior games only."""
        if self.games_df is None or self.games_df.empty:
            return

        self.games_df = self.games_df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        for stat in ['PTS', 'REB', 'AST', 'FGA', 'FTA', 'FGM', 'OREB', 'DREB', 'TOV']:
            if stat not in self.games_df.columns:
                continue
            self.games_df[f'TEAM_{stat}_ROLL_10'] = self.games_df.groupby('TEAM_ID')[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            self.games_df[f'TEAM_{stat}_ROLL_5'] = self.games_df.groupby('TEAM_ID')[stat].transform(
                lambda x: x.shift(1).rolling(5, min_periods=2).mean()
            )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data from {self.players_path} and {self.games_path}")
        self.players_df = pd.read_csv(self.players_path)
        self.games_df = pd.read_csv(self.games_path)
        
        num_cols = ['PTS','REB','AST','STL','BLK','FGA','FGM','FTA','FTM','FG3A','FG3M','TOV']
        for c in num_cols:
            if c in self.players_df.columns: self.players_df[c] = pd.to_numeric(self.players_df[c], errors='coerce')

        if 'MIN' in self.players_df.columns:
            self.players_df['MIN'] = self.players_df['MIN'].apply(self._parse_minutes).clip(0, 60)

        self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
        self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
        self.players_df = self.players_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        self._add_team_rolling_features()

        missing_stats_cols = [c for c in ['PTS', 'REB', 'AST', 'MIN'] if c in self.players_df.columns]
        missing_players = self.players_df[missing_stats_cols].isna().sum().to_dict() if missing_stats_cols else {}
        if any(v > 0 for v in missing_players.values()):
            logger.warning("Player rows retain missing base stats: %s", missing_players)
        
        logger.info(f"Loaded {len(self.players_df)} player records and {len(self.games_df)} game records")
        return self.players_df, self.games_df

    def merge_datasets(self) -> pd.DataFrame:
        """Merge player rows with pregame context only.

        Player box-score columns remain in the returned frame as labels for
        training.  Team box-score outcomes are deliberately *not* merged into
        that frame.  Only opponent identity and shifted/rolling team context
        may cross the player/team boundary.
        """
        if self.players_df is None or self.games_df is None:
            self.load_data()

        assert self.players_df is not None
        assert self.games_df is not None

        base_games = self.games_df.copy()
        game_keys = base_games[["GAME_ID", "TEAM_ID", "GAME_DATE"]].drop_duplicates()
        outcome_columns = [
            column
            for column in [
                "PTS",
                "REB",
                "AST",
                "FGA",
                "FTA",
                "FGM",
                "OREB",
                "DREB",
                "TOV",
            ]
            if column in base_games.columns
        ]

        # Build opponent identity from keys only.  This map contains no
        # outcomes and therefore cannot leak a same-game score into a player
        # row even if a caller mutates the game totals.
        opponent_keys = game_keys[["GAME_ID", "TEAM_ID"]].rename(
            columns={"TEAM_ID": "OPPONENT_ID"}
        )
        team_pairs = game_keys[["GAME_ID", "TEAM_ID"]].merge(
            opponent_keys, on="GAME_ID", how="left"
        )
        team_pairs = team_pairs[team_pairs["TEAM_ID"] != team_pairs["OPPONENT_ID"]]
        team_pairs = team_pairs.drop_duplicates(["GAME_ID", "TEAM_ID"])

        # Opponent outcomes are used only to construct shifted defensive
        # history.  The current game's opponent row is shifted out before the
        # rolling window is calculated, so changing either team's current
        # outcome cannot change its current feature row.
        opponent_outcomes = base_games[["GAME_ID", "TEAM_ID"] + outcome_columns].rename(
            columns={
                "TEAM_ID": "OPPONENT_ID",
                **{column: f"OPP_{column}_ALLOWED" for column in outcome_columns},
            }
        )
        defensive = game_keys.merge(opponent_outcomes, on="GAME_ID", how="left")
        defensive = defensive[defensive["TEAM_ID"] != defensive["OPPONENT_ID"]]
        defensive = defensive.sort_values(
            ["TEAM_ID", "GAME_DATE", "GAME_ID"], kind="mergesort"
        )
        defensive_columns: list[str] = []
        for stat in ("PTS", "REB", "AST"):
            source = f"OPP_{stat}_ALLOWED"
            if source not in defensive.columns:
                continue
            output = f"TEAM_DEF_OPP_{stat}_ALLOWED_ROLL_10"
            defensive[output] = defensive.groupby("TEAM_ID")[source].transform(
                lambda values: values.shift(1).rolling(10, min_periods=3).mean()
            )
            defensive_columns.append(output)

        safe_context = base_games[["GAME_ID", "TEAM_ID"]].copy()
        rolling_columns = [
            column
            for column in base_games.columns
            if column.startswith("TEAM_")
            and column.endswith(("_ROLL_10", "_ROLL_5"))
        ]
        safe_context = safe_context.merge(
            base_games[["GAME_ID", "TEAM_ID"] + rolling_columns],
            on=["GAME_ID", "TEAM_ID"],
            how="left",
        )
        if defensive_columns:
            safe_context = safe_context.merge(
                defensive[["GAME_ID", "TEAM_ID"] + defensive_columns].drop_duplicates(
                    ["GAME_ID", "TEAM_ID"]
                ),
                on=["GAME_ID", "TEAM_ID"],
                how="left",
            )

        # Only pregame context is selected here.  In particular, do not add
        # WL or any raw team stat; those are current-game outcomes.
        merged_df = self.players_df.merge(
            safe_context.drop_duplicates(["GAME_ID", "TEAM_ID"]),
            on=["GAME_ID", "TEAM_ID"],
            how="left",
            suffixes=("", "_TEAM_CONTEXT"),
        )

        # Carry the same shifted context for the opponent without carrying
        # its current outcome columns.
        opponent_roll = safe_context.rename(columns={"TEAM_ID": "OPPONENT_ID"})
        opponent_roll = opponent_roll.rename(
            columns={
                column: f"OPP_TEAM_{column.replace('TEAM_', '', 1).replace('DEF_OPP_', 'DEF_')}"
                for column in opponent_roll.columns
                if column.startswith("TEAM_")
            }
        )
        if not opponent_roll.empty:
            merged_df = merged_df.merge(
                team_pairs[["GAME_ID", "TEAM_ID", "OPPONENT_ID"]],
                on=["GAME_ID", "TEAM_ID"],
                how="left",
                suffixes=("", "_FROM_GAME"),
            )
            if "OPPONENT_ID_FROM_GAME" in merged_df.columns:
                merged_df["OPPONENT_ID"] = merged_df["OPPONENT_ID"].fillna(
                    merged_df.pop("OPPONENT_ID_FROM_GAME")
                )
            merged_df = merged_df.merge(
                opponent_roll,
                on=["GAME_ID", "OPPONENT_ID"],
                how="left",
            )
        else:
            merged_df = merged_df.merge(
                team_pairs[["GAME_ID", "TEAM_ID", "OPPONENT_ID"]],
                on=["GAME_ID", "TEAM_ID"],
                how="left",
            )

        logger.info("Merged dataset shape: %s", merged_df.shape)
        return merged_df
