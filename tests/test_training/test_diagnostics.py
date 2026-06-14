import sys

import pandas as pd
import pytest

from src.training.diagnostics import (
    DIAG_PREFIX,
    DiagnosticConfig,
    DiagnosticStageFailed,
    DiagnosticStop,
    diagnostic_noop,
    diagnostic_stage,
    print_data_summary,
    print_selection_summary,
    STAGES_ORDERED,
)


class TestDiagnosticConfig:
    def test_default_not_enabled(self):
        c = DiagnosticConfig()
        assert not c.enabled
        assert c.stop_after is None

    def test_should_not_stop_when_disabled(self):
        c = DiagnosticConfig(enabled=False, stop_after="preflight")
        assert not c.should_stop_after("preflight")

    def test_should_stop_on_exact_match(self):
        c = DiagnosticConfig(enabled=True, stop_after="data_load")
        assert c.should_stop_after("data_load")
        assert not c.should_stop_after("preflight")

    def test_should_not_stop_when_no_stop_after(self):
        c = DiagnosticConfig(enabled=True)
        assert not c.should_stop_after("preflight")


class TestDiagnosticStage:
    def test_success_prints_markers(self, capsys):
        config = DiagnosticConfig(enabled=True)
        with diagnostic_stage("test_stage", config):
            pass
        captured = capsys.readouterr()
        assert f"{DIAG_PREFIX} START test_stage" in captured.out
        assert f"{DIAG_PREFIX} OK test_stage" in captured.out

    def test_failure_raises_diagnostic_stage_failed(self, capsys):
        config = DiagnosticConfig(enabled=True)
        with pytest.raises(DiagnosticStageFailed) as exc_info:
            with diagnostic_stage("fail_stage", config):
                raise ValueError("something broke")
        captured = capsys.readouterr()
        assert f"{DIAG_PREFIX} START fail_stage" in captured.out
        assert f"{DIAG_PREFIX} FAILED fail_stage" in captured.out
        assert "ValueError" in captured.out
        assert "something broke" in captured.out
        assert "Stage 'fail_stage' failed" in str(exc_info.value)

    def test_silent_when_disabled(self, capsys):
        config = DiagnosticConfig(enabled=False)
        with diagnostic_stage("silent_stage", config):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_stops_after_requested_stage_raises_diagnostic_stop(self, capsys):
        config = DiagnosticConfig(enabled=True, stop_after="stop_here")
        with pytest.raises(DiagnosticStop):
            with diagnostic_stage("stop_here", config):
                pass
        captured = capsys.readouterr()
        assert f"{DIAG_PREFIX} Stopping after 'stop_here' as requested" in captured.out

    def test_continues_past_non_matching_stage(self, capsys):
        config = DiagnosticConfig(enabled=True, stop_after="later_stage")
        with diagnostic_stage("earlier_stage", config):
            pass
        captured = capsys.readouterr()
        assert f"{DIAG_PREFIX} OK earlier_stage" in captured.out
        assert "Stopping after" not in captured.out

    def test_elapsed_time_in_ok(self, capsys):
        config = DiagnosticConfig(enabled=True)
        with diagnostic_stage("timed_stage", config):
            pass
        captured = capsys.readouterr()
        assert f"{DIAG_PREFIX} OK timed_stage (" in captured.out
        assert "s)" in captured.out

    def test_preserves_diagnostic_stop_through(self, capsys):
        config = DiagnosticConfig(enabled=True)
        with pytest.raises(DiagnosticStop):
            with diagnostic_stage("inner_stage", config):
                raise DiagnosticStop("already stopped")

    def test_preserves_diagnostic_stage_failed_through(self, capsys):
        config = DiagnosticConfig(enabled=True)
        with pytest.raises(DiagnosticStageFailed):
            with diagnostic_stage("inner_stage", config):
                raise DiagnosticStageFailed("already failed")


class TestDiagnosticNoop:
    def test_prints_skip_when_enabled(self, capsys):
        config = DiagnosticConfig(enabled=True)
        diagnostic_noop("some_stage", config)
        captured = capsys.readouterr()
        assert f"{DIAG_PREFIX} START some_stage" in captured.out
        assert f"{DIAG_PREFIX} SKIP some_stage disabled" in captured.out
        assert f"{DIAG_PREFIX} OK some_stage" in captured.out

    def test_silent_when_disabled(self, capsys):
        config = DiagnosticConfig(enabled=False)
        diagnostic_noop("some_stage", config)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_stops_when_stage_matches(self, capsys):
        config = DiagnosticConfig(enabled=True, stop_after="stop_stage")
        with pytest.raises(DiagnosticStop):
            diagnostic_noop("stop_stage", config)
        captured = capsys.readouterr()
        assert "Stopping after 'stop_stage' as requested" in captured.out

    def test_accepts_custom_reason(self, capsys):
        config = DiagnosticConfig(enabled=True)
        diagnostic_noop("custom_stage", config, reason="not applicable")
        captured = capsys.readouterr()
        assert "SKIP custom_stage not applicable" in captured.out


class TestPrintDataSummary:
    def test_basic_summary(self, capsys):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        print_data_summary(df)
        captured = capsys.readouterr()
        assert "merged rows: 3" in captured.out
        assert "columns:" in captured.out
        assert "'a'" in captured.out
        assert "'b'" in captured.out

    def test_engineered_summary_with_targets(self, capsys):
        merged = pd.DataFrame({"x": [1]})
        full = pd.DataFrame({
            "x": [1],
            "PTS": [10.0],
            "REB": [5.0],
            "AST": [3.0],
            "STL": [1.0],
            "BLK": [0.5],
            "TOV": [2.0],
        })
        print_data_summary(merged, full)
        captured = capsys.readouterr()
        assert "engineered rows: 1" in captured.out
        assert "engineered cols: 7" in captured.out
        assert "PTS" in captured.out
        assert "missing target columns: none" in captured.out

    def test_missing_targets_reported(self, capsys):
        merged = pd.DataFrame({"x": [1]})
        full = pd.DataFrame({"x": [1], "PTS": [10.0]})
        print_data_summary(merged, full)
        captured = capsys.readouterr()
        assert "PTS" in captured.out
        assert "REB" in captured.out
        assert "missing target columns: REB, AST, STL, BLK, TOV" in captured.out


class TestPrintSelectionSummary:
    def test_no_manifest(self, capsys):
        print_selection_summary(None)
        captured = capsys.readouterr()
        assert "feature selection enabled: false" in captured.out

    def test_with_manifest(self, capsys):
        manifest = {
            "selected_features_by_target": {
                "PTS": ["a", "b", "c"],
                "REB": ["d", "e"],
            }
        }
        print_selection_summary(manifest)
        captured = capsys.readouterr()
        assert "feature selection enabled: true" in captured.out
        assert "PTS: 3" in captured.out
        assert "REB: 2" in captured.out


class TestStagesOrdered:
    def test_contains_expected_stages(self):
        assert "preflight" in STAGES_ORDERED
        assert "data_load" in STAGES_ORDERED
        assert "feature_engineering" in STAGES_ORDERED
        assert "feature_selection" in STAGES_ORDERED
        assert "prepare_data" in STAGES_ORDERED
        assert "artifact_check" in STAGES_ORDERED

    def test_preflight_is_first(self):
        assert STAGES_ORDERED[0] == "preflight"

    def test_artifact_check_is_last(self):
        assert STAGES_ORDERED[-1] == "artifact_check"


class TestStopAfterPreflightViaSubprocess:
    @pytest.mark.slow
    def test_diagnose_stop_after_preflight_no_data_exits_one(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "train.py", "--diagnose", "--stop-after", "preflight",
             "--data-dir", "/tmp/nonexistent_data_dir_for_diag_test",
             "--models-dir", "/tmp/nonexistent_models_dir_for_diag_test",
             "--cache-dir", "/tmp/nonexistent_cache_dir_for_diag_test",
             "--no-gpu"],
            capture_output=True, text=True, timeout=30,
        )
        assert f"{DIAG_PREFIX} START preflight" in result.stdout
        assert f"{DIAG_PREFIX} FAILED preflight" in result.stdout
        assert "FileNotFoundError" in result.stdout
        assert result.returncode == 1

    @pytest.mark.slow
    def test_diagnose_stop_after_no_stage_prints_diagnostic_message(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "train.py", "--diagnose",
             "--data-dir", "/tmp/nonexistent_data_dir_for_diag_test",
             "--models-dir", "/tmp/nonexistent_models_dir_for_diag_test",
             "--cache-dir", "/tmp/nonexistent_cache_dir_for_diag_test",
             "--no-gpu"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1
        assert f"{DIAG_PREFIX} FAILED preflight" in result.stdout


class TestArtifactCheckViaSubprocess:
    @pytest.mark.slow
    def test_artifact_check_exits_zero_when_artifacts_present(self, tmp_path):
        import subprocess
        import joblib
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)

        # Seed minimal valid artifacts so the contract passes.
        for target in ["pts", "reb", "ast", "stl", "blk", "tov"]:
            (models_dir / f"{target}_catboost.cbm").write_text("{}")
            joblib.dump({}, models_dir / f"{target}_metadata.joblib")
        for fname in ["feature_schema.pkl", "feature_cols.pkl", "blend_weights.pkl"]:
            joblib.dump({}, models_dir / fname)
        joblib.dump(
            {"targets": ["PTS", "REB", "AST", "STL", "BLK", "TOV"]},
            models_dir / "model_stack_metadata.pkl",
        )

        result = subprocess.run(
            [sys.executable, "train.py", "--diagnose", "--stop-after", "artifact_check",
             "--preset", "small", "--models-dir", str(models_dir),
             "--data-dir", str(tmp_path / "data"),
             "--cache-dir", str(tmp_path / "cache"),
             "--no-gpu"],
            capture_output=True, text=True, timeout=30,
        )
        assert f"{DIAG_PREFIX} START artifact_check" in result.stdout
        assert f"{DIAG_PREFIX} OK artifact_check" in result.stdout
        assert result.returncode == 0

    @pytest.mark.slow
    def test_artifact_check_exits_one_when_artifacts_missing(self, tmp_path):
        import subprocess
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir(parents=True)

        result = subprocess.run(
            [sys.executable, "train.py", "--diagnose", "--stop-after", "artifact_check",
             "--preset", "small", "--models-dir", str(empty_dir),
             "--data-dir", str(tmp_path / "data"),
             "--cache-dir", str(tmp_path / "cache"),
             "--no-gpu"],
            capture_output=True, text=True, timeout=30,
        )
        assert f"{DIAG_PREFIX} START artifact_check" in result.stdout
        assert f"{DIAG_PREFIX} FAILED artifact_check" in result.stdout
        assert result.returncode == 1
