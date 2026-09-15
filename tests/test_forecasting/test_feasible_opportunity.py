"""Feasible regulation opportunity and count-distribution regressions."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.minutes import allocate_team_minutes
from src.simulation.joint_sampler import sample_game, sample_participants


def roster(probabilities):
    return pd.DataFrame({
        "PLAYER_ID": range(len(probabilities)), "TEAM_ID": 1,
        "PLAY_PROB": probabilities,
        "EXPECTED_MINUTES_RAW": [40] + [20] * (len(probabilities) - 1),
        "PTS_RATE": .01, "PTS_RATE_STD": 0.0, "MINUTES_STD": 4.0,
    })


def test_conditional_expectations_and_caps():
    source = roster([1, 1, 1, 1, .8, .6, .4, .2, 0])
    result = allocate_team_minutes(source)
    assert np.allclose(result.UNCONDITIONAL_MINUTES,
                       result.PLAY_PROB * result.EXPECTED_MINUTES)
    assert result.UNCONDITIONAL_MINUTES.sum() == pytest.approx(240)
    assert result.EXPECTED_MINUTES.between(0, 48).all()
    assert result.iloc[-1].EXPECTED_MINUTES == 0
    assert "UNCONDITIONAL_MINUTES" not in source


def test_five_players_and_impossible_rosters():
    assert allocate_team_minutes(roster([1] * 5)).EXPECTED_MINUTES.eq(48).all()
    for probabilities in ([1] * 4, [.9] * 5, [0] * 6):
        with pytest.raises(ValueError, match="Infeasible"):
            allocate_team_minutes(roster(probabilities))


def test_dependent_participation_preserves_marginals():
    probabilities = np.array([1, 1, 1, .9, .8, .7, .6, .3, 0])
    rng = np.random.default_rng(123)
    draws = np.array([sample_participants(probabilities, rng)
                      for _ in range(10000)])
    assert (draws.sum(axis=1) >= 5).all()
    # Predeclared absolute tolerance exceeds 5 binomial standard errors.
    assert np.allclose(draws.mean(axis=0), probabilities, atol=.025)


def test_count_zeroes_uncertain_minutes_and_reproducibility():
    source = roster([1] * 7 + [0])
    result = sample_game(source, targets=("PTS",), simulations=50, seed=3)
    pd.testing.assert_frame_equal(result, sample_game(
        source, targets=("PTS",), simulations=50, seed=3))
    assert np.allclose(result.groupby("SIMULATION").MINUTES.sum(), 240)
    assert result.MINUTES.between(0, 48).all()
    assert result.loc[result.PLAYER_ID == 7, "VALUE"].eq(0).all()
    assert ((result.MINUTES > 0) & (result.VALUE == 0)).any()
    assert result.loc[result.PLAYER_ID == 1, "MINUTES"].std() > 0


def test_service_summary_and_query_share_seeded_samples():
    import json

    from src.contracts.forecast import ForecastRequest
    from src.pipeline.forecast_service import ForecastService
    from src.query.v2_probability import probability_at_line

    class Backend:
        targets = ("PTS",)
        models = {"PTS": object()}
        model_version = "test-bundle"

        def prepare_sampling_frame(self, frame):
            return frame

        def prepare_contexts(self, frame):
            return allocate_team_minutes(frame)

        def predict_player_stats(self, context, **kwargs):
            return {"PTS": 1.0}

    request = ForecastRequest(
        game_id="test-game", game_date="2026-10-20",
        scheduled_tip="2026-10-20T19:00:00+00:00",
        home_team_id=1, away_team_id=2,
        forecast_cutoff="2026-10-20T12:00:00+00:00",
        horizon="morning", source_snapshot_id="test-snapshot",
        model_bundle_id="test-bundle", scenario="diagnostic",
    )
    source = roster([1] * 7 + [0])
    service = ForecastService(model_backend=Backend())
    forecast = service.predict_forecast_frame(
        request, source, simulations=30, seed=7)
    draws = sample_game(source, targets=("PTS",), simulations=30, seed=7)
    for _, row in forecast.iterrows():
        values = draws.loc[draws.PLAYER_ID.eq(row.PLAYER_ID), "VALUE"]
        assert json.loads(row.DISTRIBUTION_SAMPLES) == values.tolist()
        assert row.MEAN == pytest.approx(values.mean())
        assert row.ZERO_PROB == pytest.approx(values.eq(0).mean())
        assert probability_at_line(row, 0)["push"] == pytest.approx(row.ZERO_PROB)
        assert row.UNCONDITIONAL_MINUTES == pytest.approx(
            row.PLAY_PROB * row.EXPECTED_MINUTES)
    assert forecast.UNCONDITIONAL_MINUTES.sum() == pytest.approx(240)
    from dataclasses import replace

    with pytest.raises(ValueError, match="does not match"):
        service.predict_forecast_frame(
            replace(request, model_bundle_id="other-bundle"), source)
    with pytest.raises(ValueError, match="Regulation-only"):
        service.predict_forecast_frame(replace(request, scenario="official"), source)


def test_overtime_is_shared_and_separate_from_regulation():
    source = roster([1] * 5)
    result = sample_game(source, targets=("PTS",), simulations=3, seed=4,
                         overtime_probabilities=(0.0, 1.0))
    assert result.MINUTES.eq(48).all()
    assert result.OVERTIME_MINUTES.eq(5).all()
    assert result.FULL_GAME_MINUTES.eq(53).all()
    assert result.groupby("SIMULATION").MINUTES.sum().eq(240).all()


def test_conditional_poisson_moments_and_minutes_rate_dependence():
    source = roster([1] * 5).assign(MINUTES_STD=0.0)
    draws = sample_game(source, targets=("PTS",), simulations=40, seed=12)
    # 200 independent Poisson(.48) counts; tolerances predeclared above 4 SE.
    assert abs(draws.VALUE.mean() - .48) < .20
    assert abs(draws.VALUE.eq(0).mean() - np.exp(-.48)) < .15
    varying = roster([1] * 7).assign(PTS_RATE=.5, MINUTES_STD=10.0)
    independent = sample_game(varying, targets=("PTS",), simulations=20, seed=9)
    dependent = sample_game(varying.assign(PTS_MINUTES_RATE_SLOPE=.04),
                            targets=("PTS",), simulations=20, seed=9)
    # Changing the rate model cannot change sampled opportunity under the same seed.
    assert independent.MINUTES.equals(dependent.MINUTES)
    assert not independent.VALUE.equals(dependent.VALUE)
