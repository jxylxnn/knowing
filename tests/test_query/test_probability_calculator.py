"""Tests for the stat-aware probability calculator."""

import pytest

from src.query.probability_calculator import ProbabilityCalculator
from src.query.projection_loader import ProjectionLoader


class TestProbabilityCalculator:
    def test_count_stats_use_count_aware_model(self):
        calculator = ProbabilityCalculator()

        result = calculator.calculate_from_projection(
            player_name="Test Player",
            stat="stl",
            line=1.5,
            mean=0.8,
            std=0.7,
            play_probability=1.0,
        )

        assert result.probability_method in {
            "poisson",
            "negative_binomial",
            "zero_inflated_poisson",
            "empirical_bootstrap",
            "degenerate_zero",
        }
        assert result.fallback_used is True
        assert result.prob_over + result.prob_under == pytest.approx(1.0, rel=1e-6)
        assert result.prob_under > result.prob_over

    def test_empirical_bootstrap_used_for_continuous_stats(self):
        calculator = ProbabilityCalculator()
        recent_games = [
            {"pts": 21, "min": 30},
            {"pts": 24, "min": 32},
            {"pts": 26, "min": 34},
            {"pts": 19, "min": 28},
            {"pts": 23, "min": 31},
        ]

        result = calculator.calculate_from_projection(
            player_name="Test Player",
            stat="pts",
            line=22.5,
            mean=23.0,
            std=4.0,
            recent_games=recent_games,
            recent_avg={"min": 31.0},
        )

        assert result.probability_method == "empirical_bootstrap"
        assert result.fallback_used is False
        assert result.sample_count >= 5
        assert 0.0 <= result.prob_over <= 1.0
        assert 0.0 <= result.prob_under <= 1.0

    def test_play_probability_adds_dnp_mass(self):
        calculator = ProbabilityCalculator()

        result = calculator.calculate_from_projection(
            player_name="Test Player",
            stat="pts",
            line=18.5,
            mean=20.0,
            std=4.0,
            play_probability=0.5,
        )

        assert result.prob_over + result.prob_under == pytest.approx(1.0, rel=1e-6)
        assert result.prob_under > 0.4

    def test_calibration_metrics_report_brier_and_bins(self):
        calculator = ProbabilityCalculator()

        report = calculator.evaluate_calibration(
            predicted_probabilities=[0.1, 0.3, 0.8, 0.9],
            actual_outcomes=[0, 0, 1, 1],
            n_bins=4,
        )

        assert report["sample_size"] == 4
        assert report["brier_score"] >= 0.0
        assert report["log_loss"] >= 0.0
        assert len(report["calibration_bins"]) == 4
        assert len(report["confidence_buckets"]) == 4


class TestProjectionLoaderContext:
    def test_context_includes_count_stats(self):
        loader = ProjectionLoader(data_dir="does-not-matter", players_data_path="does-not-matter")
        loader.get_recent_games = lambda player_name, n=5: [
            {"pts": 18, "reb": 6, "ast": 4, "stl": 1, "blk": 0, "tov": 2, "min": 31, "date_short": "Jan 01"}
        ]
        loader.get_matchup_history = lambda player_name, opponent, n=5: [
            {"pts": 20, "reb": 5, "ast": 3, "stl": 2, "blk": 1, "tov": 1, "min": 33, "date_short": "Jan 02"}
        ]
        loader.get_opponent_defense_profile = lambda team: {"league_rank": 8}
        loader._calculate_trend = lambda games, stat: {"direction": "neutral", "pct_change": 0, "description": "Flat"}

        context = loader.get_player_context("Test Player", "BOS", "stl")

        assert context["recent_avg"]["stl"] == pytest.approx(1.0)
        assert context["recent_avg"]["blk"] == pytest.approx(0.0)
        assert context["recent_avg"]["tov"] == pytest.approx(2.0)
        assert context["matchup_avg"]["stl"] == pytest.approx(2.0)
        assert context["matchup_sample_size"] == 1
        assert context["recent_sample_size"] == 1
