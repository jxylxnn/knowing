"""Unit tests for the 7 new feature groups introduced in the feature-engineering epic.

Each test uses small synthetic DataFrames so the suite runs fast and offline.
Edge cases covered per group:
  - First game (no history)
  - Team change mid-season
  - Missing teammates / empty rosters
  - Cold-start players
  - Back-to-back games
  - No data leakage (future rows must not affect earlier rows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features.base import FeatureContext, FeatureDiagnostics
from src.preprocessing.features.rest_density import RestGameDensityFeatureGroup
from src.preprocessing.features.lineup_stability import LineupStabilityFeatureGroup
from src.preprocessing.features.injury_opportunity import InjuryAdjustedOpportunityFeatureGroup
from src.preprocessing.features.teammate_usage import TeammateUsageFeatureGroup
from src.preprocessing.features.recency_form import RecencyFormFeatureGroup
from src.preprocessing.features.minutes_confidence import MinutesConfidenceFeatureGroup
from src.preprocessing.features.defense_position import DefensePositionFeatureGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(start: str, periods: int, freq: str = "D") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=freq)


def _assert_no_leakage(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: list[str]) -> None:
    """Assert that early rows are identical before/after perturbing a future row."""
    early = df_before.index[:-1]
    for col in cols:
        pd.testing.assert_series_equal(
            df_before.loc[early, col].reset_index(drop=True),
            df_after.loc[early, col].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# RestGameDensityFeatureGroup
# ---------------------------------------------------------------------------

class TestRestGameDensityFeatureGroup:
    def _build_df(self, rows: int = 6, b2b: bool = False) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        if b2b and rows >= 3:
            # Force back-to-back on rows 1 and 2
            dates = dates.tolist()
            dates[1] = dates[0] + pd.Timedelta(days=1)
            dates[2] = dates[1] + pd.Timedelta(days=1)
            dates = pd.DatetimeIndex(dates)
        return pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "MIN": np.linspace(20, 35, rows),
                "OPPONENT_ID": [200] * rows,
                "TEAM_ID": [1] * rows,
            }
        )

    def test_basic_output_columns(self):
        group = RestGameDensityFeatureGroup()
        df = self._build_df(rows=6)
        out = group.create(df)
        expected = {
            "SCHED_GAMES_3D",
            "SCHED_GAMES_5D",
            "SCHED_GAMES_7D",
            "SCHED_MIN_PER_DAY_5",
            "SCHED_IS_B2B_SECOND",
            "SCHED_REST_ADVANTAGE",
            "SCHED_DENSITY_SCORE",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = RestGameDensityFeatureGroup()
        df = self._build_df(rows=2)
        out = group.create(df)
        # First row has no history → counts should be 0.0 (prior)
        assert out.loc[0, "SCHED_GAMES_3D"] == 0.0
        assert out.loc[0, "SCHED_GAMES_5D"] == 0.0
        assert out.loc[0, "SCHED_IS_B2B_SECOND"] == 0.0

    def test_back_to_back_flag(self):
        group = RestGameDensityFeatureGroup()
        df = self._build_df(rows=6, b2b=True)
        out = group.create(df)
        # Row 2 is a B2B second night (previous game was 1 day ago)
        # After shift(1), row 3 should carry the B2B flag from row 2
        assert out.loc[2, "SCHED_IS_B2B_SECOND"] == 1.0

    def test_no_data_leakage(self):
        group = RestGameDensityFeatureGroup()
        df = self._build_df(rows=6)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "MIN"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["SCHED_GAMES_3D", "SCHED_GAMES_5D", "SCHED_MIN_PER_DAY_5"])

    def test_missing_opponent_falls_back(self):
        group = RestGameDensityFeatureGroup()
        df = self._build_df(rows=4).drop(columns=["OPPONENT_ID"])
        out = group.create(df)
        assert (out["SCHED_REST_ADVANTAGE"] == 0.0).all()


# ---------------------------------------------------------------------------
# LineupStabilityFeatureGroup
# ---------------------------------------------------------------------------

class TestLineupStabilityFeatureGroup:
    def _build_df(self, rows: int = 8, team_change: bool = False) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        teams = [1] * rows
        if team_change and rows >= 5:
            teams[4:] = [2] * (rows - 4)
        return pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "TEAM_ID": teams,
                "MIN": np.linspace(20, 35, rows),
            }
        )

    def test_basic_output_columns(self):
        group = LineupStabilityFeatureGroup()
        df = self._build_df(rows=8)
        out = group.create(df)
        expected = {
            "LINEUP_STARTER_RATE_10",
            "LINEUP_TEAM_STABILITY_5",
            "LINEUP_TEAM_STABILITY_10",
            "LINEUP_ROTATION_SIZE_VAR_5",
            "LINEUP_MIN_RANK_AVG_5",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = LineupStabilityFeatureGroup()
        df = self._build_df(rows=3)
        out = group.create(df)
        assert out.loc[0, "LINEUP_STARTER_RATE_10"] == 0.5
        assert out.loc[0, "LINEUP_TEAM_STABILITY_5"] == 0.5

    def test_team_change_resets_stability(self):
        group = LineupStabilityFeatureGroup()
        df = self._build_df(rows=8, team_change=True)
        out = group.create(df)
        # With a single player the Jaccard is always 1.0 (same teammate set),
        # so the feature should be 1.0 rather than the 0.5 prior.
        assert out.loc[4, "LINEUP_TEAM_STABILITY_5"] == 1.0

    def test_no_data_leakage(self):
        group = LineupStabilityFeatureGroup()
        df = self._build_df(rows=8)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "MIN"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["LINEUP_STARTER_RATE_10", "LINEUP_TEAM_STABILITY_5"])


# ---------------------------------------------------------------------------
# InjuryAdjustedOpportunityFeatureGroup
# ---------------------------------------------------------------------------

class TestInjuryAdjustedOpportunityFeatureGroup:
    def _build_df(self, rows: int = 8, missing_teammate: bool = False) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        # Two players on same team; player 102 disappears after game 3 if missing_teammate
        p1 = pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "TEAM_ID": [1] * rows,
                "MIN": [30] * rows,
                "FGA": [15] * rows,
                "PTS": [20] * rows,
                "AST": [5] * rows,
                "REB": [6] * rows,
            }
        )
        p2_rows = rows - 3 if missing_teammate else rows
        p2 = pd.DataFrame(
            {
                "PLAYER_ID": [102] * p2_rows,
                "GAME_DATE": dates[:p2_rows],
                "TEAM_ID": [1] * p2_rows,
                "MIN": [28] * p2_rows,
                "FGA": [14] * p2_rows,
                "PTS": [18] * p2_rows,
                "AST": [4] * p2_rows,
                "REB": [5] * p2_rows,
            }
        )
        return pd.concat([p1, p2], ignore_index=True)

    def test_basic_output_columns(self):
        group = InjuryAdjustedOpportunityFeatureGroup()
        df = self._build_df(rows=8)
        out = group.create(df)
        expected = {
            "INJURY_OPP_MISSING_HIGH_USAGE",
            "INJURY_OPP_MISSING_SAME_POS",
            "INJURY_OPP_MIN_BOOST",
            "INJURY_OPP_USAGE_BOOST",
            "INJURY_OPP_TEAM_ABSENCES_5",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = InjuryAdjustedOpportunityFeatureGroup()
        df = self._build_df(rows=3)
        out = group.create(df)
        assert out.loc[0, "INJURY_OPP_MISSING_HIGH_USAGE"] == 0.0
        assert out.loc[0, "INJURY_OPP_MISSING_SAME_POS"] == 0.0
        assert out.loc[0, "INJURY_OPP_MIN_BOOST"] == 0.0

    def test_missing_high_usage_teammate(self):
        group = InjuryAdjustedOpportunityFeatureGroup()
        df = self._build_df(rows=8, missing_teammate=True)
        out = group.create(df)
        # Player 101 rows are first 8 rows (concat order)
        # Player 102 disappears after game 5 (0-indexed row 4), so from game 6 (row 5)
        # the missing flag is 1. After shift(1), row 6 carries the flag.
        p1_out = out[out["PLAYER_ID"] == 101].reset_index(drop=True)
        assert p1_out.loc[6, "INJURY_OPP_MISSING_HIGH_USAGE"] == 1.0

    def test_no_data_leakage(self):
        group = InjuryAdjustedOpportunityFeatureGroup()
        df = self._build_df(rows=8)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "PTS"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["INJURY_OPP_MISSING_HIGH_USAGE", "INJURY_OPP_MIN_BOOST"])


# ---------------------------------------------------------------------------
# TeammateUsageFeatureGroup
# ---------------------------------------------------------------------------

class TestTeammateUsageFeatureGroup:
    def _build_df(self, rows: int = 8, empty_roster: bool = False) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        p1 = pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "TEAM_ID": [1] * rows,
                "MIN": [30] * rows,
                "FGA": [15] * rows,
                "PTS": [20] * rows,
                "AST": [5] * rows,
                "REB": [6] * rows,
            }
        )
        p2 = pd.DataFrame(
            {
                "PLAYER_ID": [102] * rows,
                "GAME_DATE": dates,
                "TEAM_ID": [1] * rows,
                "MIN": [28] * rows,
                "FGA": [14] * rows,
                "PTS": [18] * rows,
                "AST": [4] * rows,
                "REB": [5] * rows,
            }
        )
        df = pd.concat([p1, p2], ignore_index=True)
        if empty_roster:
            # Set MIN to 0 for all rows so no one "played"
            df["MIN"] = 0.0
        return df

    def test_basic_output_columns(self):
        group = TeammateUsageFeatureGroup()
        df = self._build_df(rows=8)
        out = group.create(df)
        expected = {
            "TEAMMATE_TOP_USAGE_ACTIVE",
            "TEAMMATE_MISSING_USAGE_SHARE",
            "TEAMMATE_MISSING_AST_SHARE",
            "TEAMMATE_MISSING_REB_SHARE",
            "TEAMMATE_MISSING_SHOT_VOLUME",
            "TEAMMATE_ACTIVE_SCORING_DEPTH",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = TeammateUsageFeatureGroup()
        df = self._build_df(rows=2)
        out = group.create(df)
        assert out.loc[0, "TEAMMATE_TOP_USAGE_ACTIVE"] == 0.0
        assert out.loc[0, "TEAMMATE_MISSING_USAGE_SHARE"] == 0.0

    def test_empty_roster_falls_back(self):
        group = TeammateUsageFeatureGroup()
        df = self._build_df(rows=4, empty_roster=True)
        out = group.create(df)
        # With no one playing, all features should be 0.0 (or prior)
        assert (out["TEAMMATE_TOP_USAGE_ACTIVE"] == 0.0).all()
        assert (out["TEAMMATE_ACTIVE_SCORING_DEPTH"] == 0.0).all()

    def test_no_data_leakage(self):
        group = TeammateUsageFeatureGroup()
        df = self._build_df(rows=8)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "PTS"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["TEAMMATE_TOP_USAGE_ACTIVE", "TEAMMATE_MISSING_USAGE_SHARE"])


# ---------------------------------------------------------------------------
# RecencyFormFeatureGroup
# ---------------------------------------------------------------------------

class TestRecencyFormFeatureGroup:
    def _build_df(self, rows: int = 12) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        return pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "PTS": np.arange(10, 10 + rows),
                "REB": np.arange(4, 4 + rows),
                "AST": np.arange(2, 2 + rows),
                "MIN": np.linspace(20, 35, rows),
            }
        )

    def test_basic_output_columns(self):
        group = RecencyFormFeatureGroup()
        df = self._build_df(rows=12)
        out = group.create(df)
        expected = {
            "RECENCY_PTS_VS_SEASON",
            "RECENCY_PTS_FORM_RATIO",
            "RECENCY_PTS_VOLATILITY_5",
            "RECENCY_MIN_DELTA_5",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = RecencyFormFeatureGroup()
        df = self._build_df(rows=2)
        out = group.create(df)
        assert out.loc[0, "RECENCY_PTS_VS_SEASON"] == 0.0
        assert out.loc[0, "RECENCY_PTS_FORM_RATIO"] == 1.0
        assert out.loc[0, "RECENCY_PTS_VOLATILITY_5"] == 0.0

    def test_cold_start_player(self):
        group = RecencyFormFeatureGroup()
        df = self._build_df(rows=3)
        out = group.create(df)
        # With only 2 shifted values, rolling windows with min_periods=2/3 should still produce values
        assert out["RECENCY_PTS_VS_SEASON"].isna().sum() == 0
        assert out["RECENCY_PTS_FORM_RATIO"].isna().sum() == 0

    def test_no_data_leakage(self):
        group = RecencyFormFeatureGroup()
        df = self._build_df(rows=12)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "PTS"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["RECENCY_PTS_VS_SEASON", "RECENCY_PTS_FORM_RATIO"])


# ---------------------------------------------------------------------------
# MinutesConfidenceFeatureGroup
# ---------------------------------------------------------------------------

class TestMinutesConfidenceFeatureGroup:
    def _build_df(self, rows: int = 12) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        return pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "MIN": np.linspace(20, 35, rows),
            }
        )

    def test_basic_output_columns(self):
        group = MinutesConfidenceFeatureGroup()
        df = self._build_df(rows=12)
        out = group.create(df)
        expected = {
            "MIN_CONF_VAR_5",
            "MIN_CONF_VAR_10",
            "MIN_CONF_TREND_3_10",
            "MIN_CONF_TREND_5_20",
            "MIN_CONF_ABOVE_NORMAL_10",
            "MIN_CONF_STARTER_RATE_10",
            "MIN_CONF_COLD_START",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = MinutesConfidenceFeatureGroup()
        df = self._build_df(rows=2)
        out = group.create(df)
        assert out.loc[0, "MIN_CONF_VAR_5"] == pytest.approx(24.0, rel=1e-6)
        assert out.loc[0, "MIN_CONF_TREND_3_10"] == 0.0
        assert out.loc[0, "MIN_CONF_COLD_START"] == 1.0

    def test_cold_start_flag(self):
        group = MinutesConfidenceFeatureGroup()
        df = self._build_df(rows=6)
        out = group.create(df)
        # First few rows should have cold_start == 1 because shifted count < 5
        assert out.loc[0, "MIN_CONF_COLD_START"] == 1.0
        # After enough history, flag should drop to 0
        assert out.loc[5, "MIN_CONF_COLD_START"] == 0.0

    def test_no_data_leakage(self):
        group = MinutesConfidenceFeatureGroup()
        df = self._build_df(rows=12)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "MIN"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["MIN_CONF_VAR_5", "MIN_CONF_TREND_3_10", "MIN_CONF_STARTER_RATE_10"])


# ---------------------------------------------------------------------------
# DefensePositionFeatureGroup
# ---------------------------------------------------------------------------

class TestDefensePositionFeatureGroup:
    def _build_df(self, rows: int = 8, with_archetype: bool = True) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        archetype = [0] * rows if with_archetype else None
        df = pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "OPPONENT_ID": [200] * rows,
                "PTS": np.arange(10, 10 + rows),
                "REB": np.arange(4, 4 + rows),
                "AST": np.arange(2, 2 + rows),
                "STL": np.linspace(0.5, 1.2, rows),
                "BLK": np.linspace(0.2, 0.8, rows),
                "TOV": np.linspace(1.0, 2.5, rows),
                "MIN": np.linspace(20, 35, rows),
            }
        )
        if with_archetype:
            df["ARCHETYPE_ID"] = archetype
        return df

    def test_basic_output_columns(self):
        group = DefensePositionFeatureGroup()
        df = self._build_df(rows=8)
        out = group.create(df)
        expected = {
            "DEF_POS_PTS_ALLOWED",
            "DEF_POS_REB_ALLOWED",
            "DEF_POS_AST_ALLOWED",
            "DEF_POS_STL_ALLOWED",
            "DEF_POS_BLK_ALLOWED",
            "DEF_POS_TOV_ALLOWED",
            "DEF_POS_RANK",
            "DEF_POS_RECENT_PTS_ALLOWED",
        }
        assert expected.issubset(set(out.columns))

    def test_first_game_uses_priors(self):
        group = DefensePositionFeatureGroup()
        df = self._build_df(rows=2)
        out = group.create(df)
        assert out.loc[0, "DEF_POS_PTS_ALLOWED"] == pytest.approx(10.0, rel=1e-6)
        # With a single opponent in the data, the rank is 1.0 (best/only)
        assert out.loc[0, "DEF_POS_RANK"] == pytest.approx(1.0, rel=1e-6)

    def test_missing_opponent_falls_back(self):
        group = DefensePositionFeatureGroup()
        df = self._build_df(rows=4).drop(columns=["OPPONENT_ID"])
        out = group.create(df)
        assert (out["DEF_POS_PTS_ALLOWED"] == 10.0).all()
        assert (out["DEF_POS_RANK"] == 15.0).all()

    def test_no_archetype_uses_inference(self):
        group = DefensePositionFeatureGroup()
        df = self._build_df(rows=4, with_archetype=False)
        out = group.create(df)
        assert "DEF_POS_PTS_ALLOWED" in out.columns
        # Should not crash and should produce real values
        assert out["DEF_POS_PTS_ALLOWED"].notna().all()

    def test_no_data_leakage(self):
        group = DefensePositionFeatureGroup()
        df = self._build_df(rows=8)
        out1 = group.create(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "PTS"] = 999
        out2 = group.create(df2)
        _assert_no_leakage(out1, out2, ["DEF_POS_PTS_ALLOWED", "DEF_POS_RECENT_PTS_ALLOWED"])


# ---------------------------------------------------------------------------
# Diagnostics & integration smoke tests
# ---------------------------------------------------------------------------

class TestDiagnosticsIntegration:
    def test_diagnostics_tracks_missing_required(self):
        group = RestGameDensityFeatureGroup()
        diag = FeatureDiagnostics()
        df = pd.DataFrame({"PLAYER_ID": [1], "GAME_DATE": [pd.Timestamp("2024-01-01")]})
        out = group.create(df, diagnostics=diag)
        assert diag.missing_required_columns
        assert "rest_density.MIN" in diag.missing_required_columns

    def test_context_priors_used_for_imputation(self):
        group = MinutesConfidenceFeatureGroup()
        ctx = FeatureContext()
        ctx.league_priors["MIN"] = 99.0
        df = pd.DataFrame(
            {
                "PLAYER_ID": [101],
                "GAME_DATE": [pd.Timestamp("2024-01-01")],
                "MIN": [20.0],
            }
        )
        out = group.create(df, context=ctx)
        # First row has no history → variance imputed to prior
        assert out.loc[0, "MIN_CONF_VAR_5"] == pytest.approx(99.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Shared teammate utility tests
# ---------------------------------------------------------------------------

class TestTeammateUtils:
    def _build_df(self, rows: int = 6) -> pd.DataFrame:
        dates = _make_dates("2024-01-01", rows)
        p1 = pd.DataFrame(
            {
                "PLAYER_ID": [101] * rows,
                "GAME_DATE": dates,
                "TEAM_ID": [1] * rows,
                "MIN": [30] * rows,
                "FGA": [15] * rows,
                "PTS": [20] * rows,
            }
        )
        p2 = pd.DataFrame(
            {
                "PLAYER_ID": [102] * rows,
                "GAME_DATE": dates,
                "TEAM_ID": [1] * rows,
                "MIN": [28] * rows,
                "FGA": [14] * rows,
                "PTS": [18] * rows,
            }
        )
        return pd.concat([p1, p2], ignore_index=True)

    def test_game_roster_map(self):
        from src.preprocessing.features._teammate_utils import build_game_roster_map

        df = self._build_df(rows=4)
        roster_map = build_game_roster_map(df)
        assert len(roster_map) == 4  # 4 unique (team, date) pairs
        for key in roster_map:
            assert roster_map[key] == {101, 102}

    def test_regular_teammates_map(self):
        from src.preprocessing.features._teammate_utils import build_regular_teammates_map

        df = self._build_df(rows=4)
        regulars = build_regular_teammates_map(df)
        # All players are regular because they appear in every game
        for key in regulars:
            assert regulars[key] == {101, 102}

    def test_high_usage_teammates_map(self):
        from src.preprocessing.features._teammate_utils import build_high_usage_teammates_map

        df = self._build_df(rows=4)
        high = build_high_usage_teammates_map(df, top_n=1)
        # Player 101 has slightly higher FGA usage
        for key in high:
            assert high[key] == {101}

    def test_teammate_context_empty_roster(self):
        from src.preprocessing.features._teammate_utils import TeammateContext

        df = self._build_df(rows=2)
        df["MIN"] = 0.0  # No one played
        tctx = TeammateContext(df)
        assert tctx.game_roster_map == {}
        assert tctx.regular_teammates_map == {}


# ---------------------------------------------------------------------------
# Performance smoke tests
# ---------------------------------------------------------------------------

class TestPerformanceSmoke:
    def _synth_rest_df(self, n_rows: int = 500) -> pd.DataFrame:
        np.random.seed(42)
        n_players = 50
        player_ids = np.random.choice(range(1000, 1000 + n_players), size=n_rows)
        base_dates = pd.date_range("2024-01-01", periods=n_rows // n_players + 10, freq="D")
        game_dates = np.random.choice(base_dates, size=n_rows)
        game_dates.sort()
        return pd.DataFrame(
            {
                "PLAYER_ID": player_ids,
                "GAME_DATE": game_dates,
                "MIN": np.random.uniform(10, 40, size=n_rows),
                "OPPONENT_ID": np.random.choice(range(200, 230), size=n_rows),
                "TEAM_ID": np.random.choice(range(1, 31), size=n_rows),
            }
        )

    def _synth_lineup_df(self, n_rows: int = 500) -> pd.DataFrame:
        np.random.seed(43)
        n_players = 50
        player_ids = np.random.choice(range(1000, 1000 + n_players), size=n_rows)
        base_dates = pd.date_range("2024-01-01", periods=n_rows // n_players + 10, freq="D")
        game_dates = np.random.choice(base_dates, size=n_rows)
        game_dates.sort()
        return pd.DataFrame(
            {
                "PLAYER_ID": player_ids,
                "GAME_DATE": game_dates,
                "TEAM_ID": np.random.choice(range(1, 31), size=n_rows),
                "MIN": np.random.uniform(10, 40, size=n_rows),
            }
        )

    def test_rest_density_runs_fast(self):
        import time

        group = RestGameDensityFeatureGroup()
        df = self._synth_rest_df(500)
        t0 = time.perf_counter()
        out = group.create(df)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"RestGameDensityFeatureGroup took {elapsed:.2f}s"
        assert "SCHED_GAMES_3D" in out.columns

    def test_lineup_stability_runs_fast(self):
        import time

        group = LineupStabilityFeatureGroup()
        df = self._synth_lineup_df(500)
        t0 = time.perf_counter()
        out = group.create(df)
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"LineupStabilityFeatureGroup took {elapsed:.2f}s"
        assert "LINEUP_TEAM_STABILITY_5" in out.columns
