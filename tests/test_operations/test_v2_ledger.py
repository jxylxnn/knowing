from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.contracts.errors import ContractError
from src.operations.ledger import OfficialForecastLedger
from tests.test_evaluation.test_v2_replay import _forecast


def test_official_ledger_is_idempotent_and_reconciles(tmp_path):
    ledger = OfficialForecastLedger(tmp_path / "ledger")
    forecast = _forecast()
    first = ledger.write_forecast(forecast)
    second = ledger.write_forecast(forecast)
    assert first == second
    assert_frame_equal(
        ledger.load_forecasts().reset_index(drop=True),
        forecast.reset_index(drop=True),
        check_dtype=False,
    )


def test_concurrent_identical_forecasts_converge(tmp_path):
    forecast = _forecast()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: OfficialForecastLedger(tmp_path / "ledger").write_forecast(forecast),
            (1, 2),
        ))
    assert results[0] == results[1]
    assert_frame_equal(pd.read_parquet(results[0]), forecast, check_dtype=False)


def test_concurrent_conflicting_forecasts_report_one_conflict(tmp_path):
    forecast = _forecast()
    conflicting = forecast.copy()
    conflicting.loc[:, "MEAN"] = conflicting["MEAN"] + 1.0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(OfficialForecastLedger(tmp_path / "ledger").write_forecast, frame)
            for frame in (forecast, conflicting)
        ]
    results = []
    for future in futures:
        try:
            results.append(("ok", future.result()))
        except ValueError as exc:
            results.append(("conflict", str(exc)))
    assert [result[0] for result in results].count("ok") == 1
    assert [result[0] for result in results].count("conflict") == 1
    assert "Conflicting immutable forecast payload" in next(
        result[1] for result in results if result[0] == "conflict"
    )


def test_ledger_rejects_close_float_retry(tmp_path):
    ledger = OfficialForecastLedger(tmp_path / "ledger")
    forecast = _forecast()
    ledger.write_forecast(forecast)

    changed = forecast.copy()
    changed.loc[:, "MEAN"] = changed["MEAN"] + 0.00001

    with pytest.raises(ValueError, match="Conflicting immutable forecast payload"):
        ledger.write_forecast(changed)


def test_reconciliation_revision_includes_actual_identity_and_is_order_stable(tmp_path):
    ledger = OfficialForecastLedger(tmp_path / "ledger")
    forecast = _forecast()
    second = forecast.copy()
    second.loc[:, "PLAYER_ID"] = 11
    forecast = pd.concat([forecast, second], ignore_index=True)
    ledger.write_forecast(forecast)
    actuals = pd.DataFrame({
        "GAME_ID": ["g1", "g1"],
        "PLAYER_ID": [10, 11],
        "PTS": [19, 21],
    })
    first = ledger.reconcile(actuals, game_date="2026-10-20")
    reordered = ledger.reconcile(actuals.iloc[::-1].reset_index(drop=True), game_date="2026-10-20")
    changed = ledger.reconcile(
        actuals.assign(PTS=[21, 21]), game_date="2026-10-20"
    )
    assert first == reordered
    assert first != changed
    record = json.loads(first.read_text(encoding="utf-8"))
    assert record["schema_version"] == "reconciliation_v2"
    assert record["actuals_digest"] == record["actuals"]["digest"]
    assert record["actuals"]["rows"]
    assert record["forecast_payloads"][0]["digest"]


def test_invalid_forecast_is_rejected_before_ledger_publication(tmp_path):
    forecast = _forecast()
    forecast.loc[:, "MEAN"] = np.inf
    with pytest.raises(ContractError, match="MEAN|finite"):
        OfficialForecastLedger(tmp_path / "ledger").write_forecast(forecast)
    assert not list((tmp_path / "ledger" / "forecasts").glob("*.parquet"))


def test_forecast_missing_game_date_is_rejected_before_ledger_publication(tmp_path):
    forecast = _forecast().drop(columns=["GAME_DATE"])

    with pytest.raises(ContractError, match="GAME_DATE"):
        OfficialForecastLedger(tmp_path / "ledger").write_forecast(forecast)

    assert not list((tmp_path / "ledger" / "forecasts").glob("*.parquet"))


def test_reconciliation_rejects_persisted_forecast_missing_game_date(tmp_path):
    root = tmp_path / "ledger"
    forecast_path = root / "forecasts" / "request.parquet"
    forecast_path.parent.mkdir(parents=True)
    _forecast().drop(columns=["GAME_DATE"]).to_parquet(forecast_path, index=False)

    with pytest.raises(ContractError, match="GAME_DATE"):
        OfficialForecastLedger(root).reconcile(
            pd.DataFrame({"GAME_ID": ["g1"], "PLAYER_ID": [10], "PTS": [19]}),
            game_date="2026-10-20",
        )

    assert not list((root / "reconciliations").rglob("*.json"))


def test_reconciliation_rejects_persisted_forecast_with_invalid_game_date(tmp_path):
    root = tmp_path / "ledger"
    forecast_path = root / "forecasts" / "request.parquet"
    forecast_path.parent.mkdir(parents=True)
    _forecast().assign(GAME_DATE="not-a-date").to_parquet(
        forecast_path, index=False
    )

    with pytest.raises(ContractError, match="GAME_DATE.*date-like.*non-null"):
        OfficialForecastLedger(root).reconcile(
            pd.DataFrame({"GAME_ID": ["g1"], "PLAYER_ID": [10], "PTS": [19]}),
            game_date="2026-10-20",
        )

    assert not list((root / "reconciliations").rglob("*.json"))


@pytest.mark.parametrize("value", [np.inf, -1, "NaN", "inf"])
def test_invalid_actuals_are_rejected_before_reconciliation_publication(tmp_path, value):
    ledger = OfficialForecastLedger(tmp_path / "ledger")
    ledger.write_forecast(_forecast())
    actuals = pd.DataFrame({"GAME_ID": ["g1"], "PLAYER_ID": [10], "PTS": [value]})

    with pytest.raises(ValueError, match="ACTUAL|finite|nonnegative"):
        ledger.reconcile(actuals, game_date="2026-10-20")

    assert not list((tmp_path / "ledger" / "reconciliations").rglob("*.json"))
