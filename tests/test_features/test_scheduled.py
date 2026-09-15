from datetime import date, datetime, timezone

import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.features.scheduled import materialize_scheduled_rows


def test_scheduled_rows_use_future_fixture_and_prior_history_only():
    request = ForecastRequest(game_id="future", game_date=date(2026, 10, 20), scheduled_tip=datetime(2026, 10, 20, 19, tzinfo=timezone.utc), home_team_id=1, away_team_id=2, forecast_cutoff=datetime(2026, 10, 20, 12, tzinfo=timezone.utc), horizon="morning", source_snapshot_id="s", model_bundle_id="b")
    roster = pd.DataFrame({"PLAYER_ID": [10, 20], "TEAM_ID": [1, 2]})
    history = pd.DataFrame({"PLAYER_ID": [10, 10], "GAME_DATE": ["2026-10-19", "2026-10-21"]})
    rows = materialize_scheduled_rows(request, roster, history)
    assert rows["GAME_ID"].eq("future").all()
    assert rows.loc[0, "REST_DAYS"] == 0
    assert rows.loc[1, "REST_DAYS"] == 7
