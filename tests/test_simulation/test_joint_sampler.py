import pandas as pd

from src.simulation.joint_sampler import sample_game


def test_joint_sampler_preserves_zeroes_and_team_minutes():
    source = pd.DataFrame({"PLAYER_ID": [1, 2, 3, 4, 5, 6], "TEAM_ID": [1] * 6,
                           "EXPECTED_MINUTES_RAW": [35, 33, 30, 28, 25, 20],
                           "PLAY_PROB": [1, 1, 1, 1, 1, 0], "PTS_RATE": [.5] * 6,
                           "PTS_RATE_STD": [0] * 6})
    result = sample_game(source, targets=("PTS",), simulations=2, seed=1)
    assert result.groupby("SIMULATION")["MINUTES"].first().shape[0] == 2
    assert (result[result["PLAYER_ID"] == 6]["VALUE"] == 0).all()
    assert result.groupby("SIMULATION")["MINUTES"].sum().eq(240).all()
