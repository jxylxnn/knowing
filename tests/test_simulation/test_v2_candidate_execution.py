"""Focused coverage for explicit sealed-candidate execution."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import simulate_season
import src.simulation.v2_runner as v2_runner
from src.contracts.forecast import ForecastRequest, validate_forecast_frame
from src.pipeline.forecast_service import ForecastService
from src.simulation.v2_runner import run_scheduled_game


def _game_row():
    return pd.Series({
        "GAME_ID": "g1",
        "GAME_DATE": "2026-10-20",
        "SCHEDULED_TIP": pd.Timestamp("2026-10-20T19:30:00-04:00"),
        "HOME_TEAM_ID": 10,
        "AWAY_TEAM_ID": 20,
    })


def _forecast_frame(*, bundle_id="bundle-1", scenario="official"):
    """Build one canonical long-form forecast row that passes validation."""
    return pd.DataFrame([{
        "REQUEST_ID": "request-1",
        "MODEL_BUNDLE_ID": bundle_id,
        "SOURCE_SNAPSHOT_ID": "snapshot-1",
        "GENERATED_AT": "2026-10-20T13:00:00+00:00",
        "FORECAST_CUTOFF": "2026-10-20T13:00:00+00:00",
        "GAME_ID": "g1",
        "GAME_DATE": "2026-10-20",
        "SCHEDULE_VERSION": "schedule_v1",
        "PLAYER_ID": 10,
        "TEAM_ID": 10,
        "OPPONENT_ID": 20,
        "HORIZON": "morning",
        "P_ACTIVE": 1.0,
        "P_PLAY_GIVEN_ACTIVE": 1.0,
        "PLAY_PROB": 1.0,
        "EXPECTED_MINUTES": 30.0,
        "MIN_P10": 25.0,
        "MIN_P50": 30.0,
        "MIN_P90": 35.0,
        "STAT": "PTS",
        "MEAN": 20.0,
        "P10": 15.0,
        "P25": 17.0,
        "P50": 20.0,
        "P75": 23.0,
        "P90": 25.0,
        "ZERO_PROB": 0.0,
        "DATA_QUALITY": "OFFICIAL_ROSTER",
        "CALIBRATION_VERSION": "calibration-1",
        "SCENARIO": scenario,
    }])


def _patch_execution(monkeypatch, *, candidate_bundle_id="bundle-candidate"):
    """Patch the runner collaborators and capture the injected backend."""
    captured: dict = {}
    calls: list = []

    class FakeBackend:
        def __init__(self, bundle_dir):
            captured["bundle_dir"] = Path(bundle_dir)
            self.model_version = candidate_bundle_id
            self.training_cutoff = "2026-01-01"

    class FakeForecastService:
        def __init__(self, *args, **kwargs):
            captured["service_args"] = args
            captured["service_kwargs"] = kwargs
            self.backend = kwargs.get("model_backend") or SimpleNamespace(
                model_version="champion-bundle"
            )

    def fake_execute_request(service, request, **kwargs):
        calls.append({"service": service, "request": request, **kwargs})
        frame = _forecast_frame(
            bundle_id=request.model_bundle_id, scenario=request.scenario
        )
        return SimpleNamespace(forecasts=frame, samples=frame.copy())

    monkeypatch.setattr(v2_runner, "V2BaselineBackend", FakeBackend)
    monkeypatch.setattr(v2_runner, "ForecastService", FakeForecastService)
    monkeypatch.setattr(
        v2_runner, "validate_source_snapshot", lambda *a, **k: None
    )
    monkeypatch.setattr(v2_runner, "execute_request", fake_execute_request)
    return captured, calls


def _ledger_spy(monkeypatch):
    instances: list = []

    class LedgerSpy:
        def __init__(self, root):
            self.root = Path(root)
            self.writes: list = []
            instances.append(self)

        def write_forecast(self, frame):
            self.writes.append(frame)
            return self.root / "forecasts" / "spy.parquet"

    monkeypatch.setattr(v2_runner, "OfficialForecastLedger", LedgerSpy)
    return instances


def test_candidate_dir_injects_explicit_v2_backend(monkeypatch, tmp_path):
    captured, calls = _patch_execution(monkeypatch)
    candidate = tmp_path / "candidates" / "sealed-1"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"

    run_scheduled_game(
        _game_row(),
        source_snapshot_id="snapshot-1",
        data_dir=data_dir,
        models_dir=models_dir,
        candidate_dir=candidate,
    )

    assert captured["bundle_dir"] == Path(candidate)
    backend = captured["service_kwargs"]["model_backend"]
    assert backend.model_version == "bundle-candidate"
    assert captured["service_kwargs"]["data_dir"] == str(data_dir)
    assert captured["service_kwargs"]["models_dir"] == str(models_dir)
    assert calls[0]["request"].model_bundle_id == "bundle-candidate"


def test_absent_candidate_keeps_the_champion_loading_path(monkeypatch, tmp_path):
    captured, calls = _patch_execution(monkeypatch)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"

    run_scheduled_game(
        _game_row(),
        source_snapshot_id="snapshot-1",
        data_dir=data_dir,
        models_dir=models_dir,
    )

    assert "model_backend" not in captured["service_kwargs"]
    assert captured["service_kwargs"]["models_dir"] == str(models_dir)
    assert captured["service_kwargs"]["data_dir"] == str(data_dir)
    assert captured.get("bundle_dir") is None
    assert calls[0]["request"].model_bundle_id == "champion-bundle"


def test_strict_candidate_persists_recovery_evidence_without_publishing(
    monkeypatch, tmp_path
):
    captured, calls = _patch_execution(monkeypatch)
    ledger_calls = _ledger_spy(monkeypatch)

    forecast, _ = run_scheduled_game(
        _game_row(),
        source_snapshot_id="snapshot-1",
        data_dir=tmp_path / "data",
        candidate_dir=tmp_path / "candidates" / "sealed-1",
        strict=True,
        persist=True,
    )

    validate_forecast_frame(forecast)
    assert calls[0]["persist"] is True
    assert calls[0]["request"].scenario == "official"
    assert ledger_calls == []


def test_champion_strict_run_still_writes_the_official_ledger(monkeypatch, tmp_path):
    _patch_execution(monkeypatch)
    ledger_calls = _ledger_spy(monkeypatch)

    forecast, _ = run_scheduled_game(
        _game_row(),
        source_snapshot_id="snapshot-1",
        data_dir=tmp_path / "data",
    )

    assert len(ledger_calls) == 1
    assert ledger_calls[0].root == Path(tmp_path / "data") / "ledger"
    assert ledger_calls[0].writes[0].equals(forecast)


@pytest.mark.parametrize("with_candidate", [False, True])
def test_degraded_run_neither_persists_nor_publishes(
    monkeypatch, tmp_path, with_candidate
):
    _, calls = _patch_execution(monkeypatch)
    ledger_calls = _ledger_spy(monkeypatch)
    candidate = (
        tmp_path / "candidates" / "sealed-1" if with_candidate else None
    )

    run_scheduled_game(
        _game_row(),
        source_snapshot_id="snapshot-1",
        data_dir=tmp_path / "data",
        candidate_dir=candidate,
        strict=False,
        persist=False,
    )

    assert calls[0]["persist"] is False
    assert calls[0]["request"].scenario == "degraded_observed_roster"
    assert ledger_calls == []


class _UnofficialBackend:
    """Baseline-shaped backend whose full-game forecasts are not official."""

    targets = ("PTS",)
    models = {"PTS": object()}
    model_version = "bundle-candidate"
    supports_official = False

    def prepare_sampling_frame(self, contexts):
        raise AssertionError("the official gate must fail before sampling")

    def predict_player_stats(self, *args, **kwargs):
        raise AssertionError("the official gate must fail before any prediction")


def _request(*, scenario, bundle_id="bundle-candidate"):
    return ForecastRequest(
        game_id="g1",
        game_date="2026-10-20",
        scheduled_tip="2026-10-20T23:00:00+00:00",
        home_team_id=10,
        away_team_id=20,
        forecast_cutoff="2026-10-20T13:00:00+00:00",
        horizon="morning",
        source_snapshot_id="snapshot-1",
        model_bundle_id=bundle_id,
        scenario=scenario,
    )


def test_supports_official_gate_still_blocks_an_unofficial_backend():
    service = ForecastService(model_backend=_UnofficialBackend())

    with pytest.raises(ValueError, match="official"):
        service.predict_game_distribution(
            _request(scenario="official"), pd.DataFrame({"PLAYER_ID": [10]})
        )


def test_parser_accepts_a_candidate_bundle(tmp_path):
    candidate = tmp_path / "candidates" / "sealed-1"

    args = simulate_season.build_parser().parse_args([
        "--today", "--snapshot-id", "snapshot-1", "--candidate", str(candidate),
    ])

    assert args.candidate == str(candidate)
    assert simulate_season._execution_mode(args) == "shadow"


def _patch_cli(monkeypatch, *, bundle_ids=("bundle-1",)):
    games = pd.DataFrame([
        {
            "GAME_ID": f"g{index + 1}",
            "GAME_DATE": "2026-10-20",
            "SCHEDULED_TIP": "2026-10-20T23:00:00+00:00",
            "HOME_TEAM_ID": 10 + index,
            "AWAY_TEAM_ID": 20 + index,
        }
        for index in range(len(bundle_ids))
    ])
    monkeypatch.setattr(
        simulate_season, "_schedule", lambda args, dates: games
    )
    produced: list = []

    def fake_run(game, **kwargs):
        index = len(produced)
        produced.append({"game": game, **kwargs})
        return _forecast_frame(bundle_id=bundle_ids[index]), pd.DataFrame(
            {"GAME_ID": [str(game["GAME_ID"])], "SIMULATIONS": [100]}
        )

    monkeypatch.setattr(simulate_season, "run_scheduled_game", fake_run)
    exported: list = []

    def fake_export(root, forecasts, samples):
        exported.append({"root": Path(root), "forecasts": forecasts})
        return Path(root) / "forecasts.parquet", Path(root) / "samples.parquet"

    monkeypatch.setattr(simulate_season, "export_forecast_run", fake_export)
    return produced, exported


def _run_cli(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["simulate_season.py", *argv])
    code = simulate_season.main()
    return code, json.loads(capsys.readouterr().out)


def test_cli_reports_official_publication_for_the_champion(monkeypatch, tmp_path, capsys):
    produced, exported = _patch_cli(monkeypatch)

    code, payload = _run_cli(monkeypatch, capsys, [
        "--date", "2026-10-20", "--snapshot-id", "snapshot-1", "--json",
        "--output-dir", str(tmp_path / "out"),
    ])

    assert code == 0
    assert payload["status"] == "complete"
    assert payload["mode"] == "official"
    assert payload["official"] is True
    assert payload["published_to_official_ledger"] is True
    assert payload["model_bundle_id"] == "bundle-1"
    assert payload["candidate"] is None
    assert payload["games"] == 1
    output = tmp_path / "out"
    assert Path(payload["forecast_path"]) == output / "forecasts.parquet"
    assert Path(payload["samples_path"]) == output / "samples.parquet"
    assert produced[0]["candidate_dir"] is None
    assert produced[0]["strict"] is True
    assert produced[0]["persist"] is True
    assert len(exported) == 1


def test_cli_reports_shadow_for_a_candidate(monkeypatch, tmp_path, capsys):
    produced, exported = _patch_cli(monkeypatch)
    candidate = tmp_path / "candidates" / "sealed-1"

    code, payload = _run_cli(monkeypatch, capsys, [
        "--date", "2026-10-20", "--snapshot-id", "snapshot-1", "--json",
        "--candidate", str(candidate), "--output-dir", str(tmp_path / "out"),
    ])

    assert code == 0
    assert payload["mode"] == "shadow"
    assert payload["official"] is False
    assert payload["published_to_official_ledger"] is False
    assert payload["model_bundle_id"] == "bundle-1"
    assert payload["candidate"] == str(candidate)
    assert produced[0]["candidate_dir"] == str(candidate)
    assert produced[0]["strict"] is True
    assert produced[0]["persist"] is True
    assert len(exported) == 1


def test_cli_reports_degraded_for_an_explicit_diagnostic(monkeypatch, tmp_path, capsys):
    produced, _ = _patch_cli(monkeypatch)

    code, payload = _run_cli(monkeypatch, capsys, [
        "--date", "2026-10-20", "--snapshot-id", "snapshot-1", "--json",
        "--candidate", str(tmp_path / "candidates" / "sealed-1"),
        "--allow-degraded", "--output-dir", str(tmp_path / "out"),
    ])

    assert code == 0
    assert payload["mode"] == "degraded"
    assert payload["official"] is False
    assert payload["published_to_official_ledger"] is False
    assert payload["model_bundle_id"] == "bundle-1"
    assert produced[0]["strict"] is False
    assert produced[0]["persist"] is False


def test_cli_fails_closed_on_mixed_bundle_ids(monkeypatch, tmp_path, capsys):
    _, exported = _patch_cli(monkeypatch, bundle_ids=("bundle-1", "bundle-2"))

    code, payload = _run_cli(monkeypatch, capsys, [
        "--date", "2026-10-20", "--snapshot-id", "snapshot-1",
        "--output-dir", str(tmp_path / "out"),
    ])

    assert code == 3
    assert payload["status"] == "failed"
    assert payload["mode"] == "official"
    assert "MODEL_BUNDLE_ID" in payload["error"]
    assert exported == []


def test_single_model_bundle_id_accepts_one_stripped_value():
    frame = pd.DataFrame({"MODEL_BUNDLE_ID": [" bundle-1 ", "bundle-1"]})

    assert simulate_season._single_model_bundle_id(frame) == "bundle-1"


@pytest.mark.parametrize(
    "values",
    [
        ["", "  "],
        [None, float("nan")],
        ["bundle-1", "bundle-2"],
        ["bundle-1", ""],
        ["bundle-1", None],
        ["bundle-1", pd.NA],
    ],
)
def test_single_model_bundle_id_fails_closed_on_empty_or_mixed_values(values):
    frame = pd.DataFrame({"MODEL_BUNDLE_ID": values})

    with pytest.raises(ValueError, match="MODEL_BUNDLE_ID"):
        simulate_season._single_model_bundle_id(frame)


@pytest.mark.parametrize("sentinel", ["nan", "NaN", "none", "None", "<NA>", "<na>"])
def test_single_model_bundle_id_rejects_missing_value_sentinels(sentinel):
    frame = pd.DataFrame({"MODEL_BUNDLE_ID": ["bundle-1", sentinel]})

    with pytest.raises(ValueError, match="MODEL_BUNDLE_ID"):
        simulate_season._single_model_bundle_id(frame)


def test_single_model_bundle_id_requires_the_canonical_column():
    frame = pd.DataFrame({"BUNDLE": ["bundle-1"]})

    with pytest.raises(ValueError, match="missing MODEL_BUNDLE_ID"):
        simulate_season._single_model_bundle_id(frame)


def test_horizon_cutoff_is_still_timezone_aware_for_candidate_runs():
    tip = pd.Timestamp("2026-10-20T19:30:00-04:00").to_pydatetime()
    cutoff = v2_runner.horizon_cutoff(tip, "morning")
    assert cutoff.tzinfo == ZoneInfo("America/New_York")
    assert cutoff < tip
