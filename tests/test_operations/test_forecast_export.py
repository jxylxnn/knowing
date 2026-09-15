import pandas as pd
import pytest

from src.operations.forecast_export import export_forecast_run


def test_export_is_immutable_atomic_pair_and_detects_corruption(tmp_path):
    forecasts = pd.DataFrame({"REQUEST_ID": ["r"], "MEAN": [1.0]})
    samples = pd.DataFrame({"VALUE": [0, 2]})
    first = export_forecast_run(tmp_path, forecasts, samples)
    assert export_forecast_run(tmp_path, forecasts, samples) == first
    second = export_forecast_run(tmp_path, forecasts.assign(MEAN=2.0), samples)
    assert second != first
    assert pd.read_parquet(first[0]).MEAN.tolist() == [1.0]
    first[1].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        export_forecast_run(tmp_path, forecasts, samples)
    assert not list(tmp_path.glob(".forecast_export_*"))
