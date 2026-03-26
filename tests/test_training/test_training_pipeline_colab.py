import numpy as np
import pandas as pd


def _build_recent_training_frame(rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "PLAYER_ID": np.arange(rows),
            "TEAM_ID": np.full(rows, 1610612747),
            "OPPONENT_ID": np.full(rows, 1610612738),
            "GAME_DATE": dates,
            "PTS": np.linspace(10, 25, rows),
            "REB": np.linspace(4, 11, rows),
            "AST": np.linspace(3, 9, rows),
            "STL": np.linspace(0.5, 2.0, rows),
            "BLK": np.linspace(0.2, 1.2, rows),
            "TOV": np.linspace(1.0, 4.0, rows),
            "ROLL_USAGE": np.linspace(0.2, 0.8, rows),
            "PACE_FACTOR": np.linspace(95, 103, rows),
        }
    )


def test_training_pipeline_falls_back_to_cpu_when_gpu_unavailable(monkeypatch, tmp_path):
    from src.training import pipeline as training_pipeline

    monkeypatch.setattr(training_pipeline, "check_gpu_compatibility", lambda: False)

    pipeline = training_pipeline.TrainingPipeline(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        parallel=False,
        use_gpu=True,
    )

    assert pipeline.use_gpu is False
    assert pipeline.gpu_settings["gpu_available"] is False


def test_prepare_data_uses_temporal_fallback_when_split_date_misses_history(tmp_path):
    from src.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        parallel=False,
        use_gpu=False,
    )

    fit_df, val_df, test_df = pipeline.prepare_data(
        _build_recent_training_frame(),
        test_date="2024-03-01",
        val_ratio=0.2,
    )

    assert len(fit_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(test_df) == 3
    assert fit_df["GAME_DATE"].max() < test_df["GAME_DATE"].min()
    assert pipeline.feature_cols
