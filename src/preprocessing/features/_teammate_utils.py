"""Shared teammate/roster precomputations for feature groups that need team context.

This module centralizes the repeated (TEAM_ID, GAME_DATE) → roster mapping logic
that was previously duplicated across `lineup_stability`, `injury_opportunity`,
and `teammate_usage`.  Centralising guarantees consistency and makes the maps
cheap to reuse.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def build_game_roster_map(df: pd.DataFrame, min_col: str = "MIN") -> Dict[Tuple, Set[int]]:
    """Return mapping (TEAM_ID, GAME_DATE) → set of PLAYER_IDs who played (MIN > 0).

    Handles empty rosters gracefully (the key simply won't exist).
    """
    played_mask = df[min_col] > 0
    roster_map: Dict[Tuple, Set[int]] = {}
    for (tid, gdate), group in df[played_mask].groupby(["TEAM_ID", "GAME_DATE"]):
        roster_map[(tid, gdate)] = set(group["PLAYER_ID"].values)
    return roster_map


def build_team_games_map(df: pd.DataFrame, min_col: str = "MIN") -> Dict[int, List[pd.Timestamp]]:
    """Return mapping TEAM_ID → sorted unique GAME_DATEs for that team.

    Only considers games where at least one player had MIN > 0 so that
    completely empty rosters don't pollute the timeline.
    """
    played_mask = df[min_col] > 0
    team_games: Dict[int, List[pd.Timestamp]] = {}
    for tid, group in df[played_mask].groupby("TEAM_ID"):
        team_games[tid] = sorted(group["GAME_DATE"].unique())
    return team_games


def build_regular_teammates_map(
    df: pd.DataFrame,
    game_roster_map: Optional[Dict[Tuple, Set[int]]] = None,
    team_games_map: Optional[Dict[int, List[pd.Timestamp]]] = None,
    *,
    window: int = 20,
    threshold_ratio: float = 0.5,
    min_col: str = "MIN",
) -> Dict[Tuple, Set[int]]:
    """Return mapping (TEAM_ID, GAME_DATE) → set of "regular" PLAYER_IDs.

    A player is regular for a given team/game if they appeared in >
    `threshold_ratio` of the team's last `window` games (including the
    current game date).

    Parameters
    ----------
    df:
        Input DataFrame with at least TEAM_ID, GAME_DATE, PLAYER_ID, MIN.
    game_roster_map:
        Precomputed roster map.  If None, built from `df`.
    team_games_map:
        Precomputed team-games map.  If None, built from `df`.
    window:
        Number of recent team games to look back.
    threshold_ratio:
        Fraction of those games a player must appear in to be "regular".
    min_col:
        Column used to determine whether a player "played".

    Returns
    -------
    regular_teammates : dict
        Keys are (TEAM_ID, GAME_DATE); values are sets of PLAYER_IDs.
    """
    if game_roster_map is None:
        game_roster_map = build_game_roster_map(df, min_col=min_col)
    if team_games_map is None:
        team_games_map = build_team_games_map(df, min_col=min_col)

    regular_teammates: Dict[Tuple, Set[int]] = {}

    for tid, dates in team_games_map.items():
        for i, gdate in enumerate(dates):
            window_dates = dates[max(0, i - (window - 1)) : i + 1]
            player_counts: Dict[int, int] = {}
            for wd in window_dates:
                roster = game_roster_map.get((tid, wd), set())
                for pid in roster:
                    player_counts[pid] = player_counts.get(pid, 0) + 1

            threshold = len(window_dates) * threshold_ratio
            regulars = {pid for pid, cnt in player_counts.items() if cnt > threshold}
            regular_teammates[(tid, gdate)] = regulars

    return regular_teammates


def build_high_usage_teammates_map(
    df: pd.DataFrame,
    game_roster_map: Optional[Dict[Tuple, Set[int]]] = None,
    team_games_map: Optional[Dict[int, List[pd.Timestamp]]] = None,
    *,
    window: int = 20,
    top_n: int = 3,
    usage_col: Optional[str] = None,
    min_col: str = "MIN",
) -> Dict[Tuple, Set[int]]:
    """Return mapping (TEAM_ID, GAME_DATE) → set of top-N high-usage PLAYER_IDs.

    Usage is summed over the team's last `window` games.  If `usage_col`
    is not present in `df`, falls back to ``PTS`` and then ``MIN``.

    Parameters
    ----------
    df:
        Input DataFrame.
    game_roster_map:
        Precomputed roster map.  If None, built from `df`.
    team_games_map:
        Precomputed team-games map.  If None, built from `df`.
    window:
        Number of recent team games to sum usage over.
    top_n:
        How many players to mark as high-usage.
    usage_col:
        Column to treat as the usage metric.  Auto-detected if None.
    min_col:
        Column used to determine whether a player played.

    Returns
    -------
    high_usage_teammates : dict
        Keys are (TEAM_ID, GAME_DATE); values are sets of PLAYER_IDs.
    """
    if game_roster_map is None:
        game_roster_map = build_game_roster_map(df, min_col=min_col)
    if team_games_map is None:
        team_games_map = build_team_games_map(df, min_col=min_col)

    # Resolve usage column
    if usage_col is None or usage_col not in df.columns:
        if "FGA" in df.columns:
            usage_col = "FGA"
        elif "PTS" in df.columns:
            usage_col = "PTS"
        else:
            usage_col = min_col

    high_usage: Dict[Tuple, Set[int]] = {}

    for tid, dates in team_games_map.items():
        for i, gdate in enumerate(dates):
            window_dates = dates[max(0, i - (window - 1)) : i + 1]
            player_usage: Dict[int, float] = {}
            for wd in window_dates:
                roster = game_roster_map.get((tid, wd), set())
                mask = (df["TEAM_ID"] == tid) & (df["GAME_DATE"] == wd) & (df["PLAYER_ID"].isin(roster))
                game_rows = df.loc[mask]
                for _, row in game_rows.iterrows():
                    pid = row["PLAYER_ID"]
                    val = row.get(usage_col, 0)
                    if pd.isna(val):
                        val = 0
                    player_usage[pid] = player_usage.get(pid, 0) + val

            sorted_players = sorted(player_usage.items(), key=lambda x: x[1], reverse=True)
            top = {pid for pid, _ in sorted_players[:top_n]}
            high_usage[(tid, gdate)] = top

    return high_usage


def build_team_totals_map(
    df: pd.DataFrame,
    cols: List[str],
    min_col: str = "MIN",
) -> Dict[Tuple, Dict[str, float]]:
    """Return mapping (TEAM_ID, GAME_DATE) → dict of team totals for `cols`.

    Missing columns are omitted from the inner dict.  The caller should
    apply safe defaults when a key is missing.
    """
    available_cols = [c for c in cols if c in df.columns]
    played_mask = df[min_col] > 0
    totals_map: Dict[Tuple, Dict[str, float]] = {}
    for (tid, gdate), group in df[played_mask].groupby(["TEAM_ID", "GAME_DATE"]):
        totals: Dict[str, float] = {}
        for c in available_cols:
            totals[c] = max(group[c].sum(), 1.0)
        totals_map[(tid, gdate)] = totals
    return totals_map


class TeammateContext:
    """Container for all precomputed teammate/roster maps.

    Instantiate once per feature-engineering pass and share across
    feature groups that need roster context.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        regular_window: int = 20,
        regular_threshold_ratio: float = 0.5,
        usage_window: int = 20,
        usage_top_n: int = 3,
        usage_col: Optional[str] = None,
        min_col: str = "MIN",
    ):
        self.game_roster_map = build_game_roster_map(df, min_col=min_col)
        self.team_games_map = build_team_games_map(df, min_col=min_col)
        self.regular_teammates_map = build_regular_teammates_map(
            df,
            self.game_roster_map,
            self.team_games_map,
            window=regular_window,
            threshold_ratio=regular_threshold_ratio,
            min_col=min_col,
        )
        self.high_usage_teammates_map = build_high_usage_teammates_map(
            df,
            self.game_roster_map,
            self.team_games_map,
            window=usage_window,
            top_n=usage_top_n,
            usage_col=usage_col,
            min_col=min_col,
        )

    def roster_for(self, team_id: int, game_date: pd.Timestamp) -> Set[int]:
        """Return the set of PLAYER_IDs who played for `team_id` on `game_date`."""
        return self.game_roster_map.get((team_id, game_date), set())

    def regulars_for(self, team_id: int, game_date: pd.Timestamp) -> Set[int]:
        """Return the set of regular teammates for `team_id` on `game_date`."""
        return self.regular_teammates_map.get((team_id, game_date), set())

    def high_usage_for(self, team_id: int, game_date: pd.Timestamp) -> Set[int]:
        """Return the set of high-usage teammates for `team_id` on `game_date`."""
        return self.high_usage_teammates_map.get((team_id, game_date), set())
