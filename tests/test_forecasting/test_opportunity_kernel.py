import pandas as pd

from src.forecasting.availability import participation_baseline
from src.forecasting.minutes import allocate_team_minutes
from src.forecasting.rates import totals_from_minutes_and_rates


def test_opportunity_kernel_respects_play_and_team_minute_constraints():
    players = pd.DataFrame({"PLAYER_ID": [1, 2, 3, 4, 5], "TEAM_ID": [1] * 5, "EXPECTED_MINUTES_RAW": [35, 32, 28, 25, 20], "PLAY_PROB": [1] * 5})
    minutes = allocate_team_minutes(players)
    assert minutes["EXPECTED_MINUTES"].sum() == 240
    totals = totals_from_minutes_and_rates(minutes.assign(PTS_RATE=.5, PTS_RATE_STD=.1), targets=("PTS",))
    assert totals["PTS_P10"].le(totals["PTS_P50"]).all()


def test_participation_baseline_has_cold_start_prior():
    result = participation_baseline(pd.DataFrame({"PLAYER_ID": [1], "APPEARED": [1]}), pd.DataFrame({"PLAYER_ID": [1, 2]}))
    assert result["PLAY_PROB"].between(0, 1).all()
