from dataclasses import dataclass

from src.evaluation.metrics import backtest_result_to_json_dict


@dataclass
class FakeTargetMetrics:
    mae: float
    rmse: float
    r2: float
    calibration_error: float
    interval_coverage: float
    sample_size: int


@dataclass
class FakeBacktestResult:
    target_metrics: dict


def test_backtest_result_to_json_dict_serializes_target_metrics():
    result = FakeBacktestResult(
        target_metrics={
            "PTS": FakeTargetMetrics(
                mae=3.5,
                rmse=4.8,
                r2=0.62,
                calibration_error=0.04,
                interval_coverage=0.81,
                sample_size=100,
            )
        }
    )

    payload = backtest_result_to_json_dict(result)

    assert payload["targets"]["PTS"]["mae"] == 3.5
    assert payload["targets"]["PTS"]["rmse"] == 4.8
    assert payload["targets"]["PTS"]["r2"] == 0.62
    assert payload["targets"]["PTS"]["calibration_error"] == 0.04
    assert payload["targets"]["PTS"]["interval_coverage"] == 0.81
    assert payload["targets"]["PTS"]["sample_size"] == 100
    assert payload["overall"]["mean_mae"] == 3.5
    assert payload["overall"]["mean_rmse"] == 4.8
    assert payload["overall"]["mean_r2"] == 0.62
