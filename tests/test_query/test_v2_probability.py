from datetime import datetime, timezone

import pandas as pd
import pytest

from src.contracts.forecast import ForecastRequest
from src.pipeline.forecast_service import ForecastService
from src.query.v2_probability import probability_at_line, select_forecast


class _Backend:
    targets = ["PTS"]
    models = {"PTS": object()}
    model_version = "bundle-test"

    def predict_player_stats(self, context, history_df=None, include_confidence=False):
        return {"PTS": 20.0, "PTS_STD": 4.0}


def test_select_and_query_v2_forecast():
    request = ForecastRequest(
        game_id="g1",
        game_date="2026-10-20",
        scheduled_tip=datetime(2026, 10, 20, 23, tzinfo=timezone.utc),
        home_team_id=1,
        away_team_id=2,
        forecast_cutoff=datetime(2026, 10, 20, 12, tzinfo=timezone.utc),
        horizon="morning",
        source_snapshot_id="snapshot",
        model_bundle_id="bundle",
    )
    forecast = ForecastService(manager=_Backend()).predict_forecast_frame(
        request,
        pd.DataFrame([{"PLAYER_ID": 10, "TEAM_ID": 1, "OPPONENT_ID": 2}]),
    )
    row = select_forecast(forecast, player="10", stat="pts", game_date="2026-10-20")
    probability = probability_at_line(row, 20.0)
    assert 0 <= probability["over"] <= 1
    assert probability["over"] + probability["push"] + probability["under"] == 1


def _quantile_row(**extra):
    return pd.Series({
        "ZERO_PROB": 0.8,
        "P10": 0.0,
        "P25": 0.0,
        "P50": 0.0,
        "P75": 0.0,
        "P90": 2.0,
        **extra,
    })


def test_integer_line_reports_known_zero_push_probability():
    result = probability_at_line(_quantile_row(), 0)
    assert result["under"] == 0.0
    assert result["push"] == 0.8
    assert result["over"] == 0.2
    assert result["probability_exact"] is True


def test_declared_pmf_reports_exact_mass_at_integer_line():
    result = probability_at_line(
        _quantile_row(PMF={"0": 0.5, "1": 0.25, "2": 0.25}), 1
    )
    assert result["under"] == 0.5
    assert result["push"] == 0.25
    assert result["over"] == 0.25
    assert result["probability_method"] == "declared_discrete_pmf"


def test_negative_and_half_integer_lines_preserve_nonnegative_integer_support():
    negative = probability_at_line(_quantile_row(), -1)
    half = probability_at_line(_quantile_row(), 0.5)
    assert (negative["under"], negative["push"], negative["over"]) == (0.0, 0.0, 1.0)
    assert (half["under"], half["push"], half["over"]) == (0.8, 0.0, 0.2)


def test_quantile_only_query_discloses_missing_discrete_evidence():
    result = probability_at_line(_quantile_row(), 1)
    assert result["probability_exact"] is False
    assert result["probability_method"] == "quantile_interpolation_estimate"
    assert "cannot recover exact mass" in result["probability_limitation"]


def test_query_rejects_nonfinite_lines():
    with pytest.raises(ValueError, match="finite"):
        probability_at_line(_quantile_row(), float("inf"))


@pytest.mark.parametrize("support", [-1, 1.5])
@pytest.mark.parametrize("field", ["PMF", "SAMPLES"])
def test_count_support_rejected(support, field):
    payload = {str(support): 1.0} if field == "PMF" else [support]
    with pytest.raises(ValueError, match="nonnegative integers"):
        probability_at_line(pd.Series({field: payload}), 0)
