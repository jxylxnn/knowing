from types import SimpleNamespace

from src.training.catboost_trainer import CatBoostProgressCallback


class _CaptureLogger:
    def __init__(self):
        self.metrics = None

    def log_iteration(self, metrics):
        self.metrics = metrics


def test_progress_callback_reads_current_catboost_metrics_shape():
    callback = CatBoostProgressCallback(target="PTS", total_iterations=10)
    capture = _CaptureLogger()
    callback.training_logger = capture
    info = SimpleNamespace(
        iteration=1,
        metrics={
            "learn": {"RMSE": [6.2, 5.8]},
            "validation": {"RMSE": [6.8, 6.4]},
        },
    )

    assert callback.after_iteration(info) is True
    assert capture.metrics.train_loss == 5.8
    assert capture.metrics.val_loss == 6.4
    assert capture.metrics.best_val_loss == 6.4
    assert capture.metrics.best_iteration == 1


def test_progress_callback_preserves_legacy_metric_fallback():
    callback = CatBoostProgressCallback(target="PTS", total_iterations=10)
    capture = _CaptureLogger()
    callback.training_logger = capture
    info = SimpleNamespace(
        iteration=1,
        learn_error=[5.8],
        test_error=[6.4],
    )

    assert callback.after_iteration(info) is True
    assert capture.metrics.train_loss == 5.8
    assert capture.metrics.val_loss == 6.4
