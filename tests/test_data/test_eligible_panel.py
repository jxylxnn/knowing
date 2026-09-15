import pandas as pd
import pytest

from src.contracts.errors import ContractError
from src.data.eligible_panel import normalize_outcomes


def test_unknown_labels_stay_unknown_and_overtime_is_not_invented():
    frame = pd.DataFrame({"GAME_ID": ["g"] * 3, "TEAM_ID": [1] * 3,
                          "PLAYER_ID": [1, 2, 3], "MIN": [20, 0, None],
                          "APPEARED": [None, 0, None], "ACTIVE": [None, 1, None]})
    result = normalize_outcomes(frame)
    assert result.APPEARED.iloc[0] == 1
    assert result.APPEARED.iloc[1] == 0
    assert pd.isna(result.APPEARED.iloc[2])
    assert result.REGULATION_MINUTES.isna().all()
    assert pd.isna(result.ACTIVE.iloc[0])
    assert result.PTS.isna().all()
    with pytest.raises(ContractError, match="contradict"):
        normalize_outcomes(frame.assign(PTS=[1, 2, None]))
