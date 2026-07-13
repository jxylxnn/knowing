import pandas as pd

from src.evaluation.prediction_ledger import PredictionLedger


def test_prediction_ledger_deduplicates_and_reconciles(tmp_path):
    ledger = PredictionLedger(tmp_path / "history.parquet")
    rows = [
        {
            "MODEL_VERSION": "v1",
            "GAME_ID": "g1",
            "PLAYER_ID": 1,
            "STAT": "PTS",
            "PREDICTION": 25.0,
        },
        {
            "MODEL_VERSION": "v1",
            "GAME_ID": "g1",
            "PLAYER_ID": 1,
            "STAT": "PTS",
            "PREDICTION": 25.0,
        },
    ]
    frame = ledger.append(rows)
    assert len(frame) == 1
    reconciled = ledger.reconcile_actuals(
        pd.DataFrame([{"GAME_ID": "g1", "PLAYER_ID": 1, "PTS": 27.0}])
    )
    assert reconciled.loc[0, "ACTUAL"] == 27.0
    assert reconciled.loc[0, "PREDICTION"] == 25.0
