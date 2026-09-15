"""Offline regressions from the September 2026 readiness audit."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.availability import ProbabilityCalibrator, status_rule_baseline
from update_data import get_seasons_between_dates


def test_incremental_range_crosses_october_boundary():
    assert get_seasons_between_dates('2026-04-10', '2026-10-25') == [
        '2025-26', '2026-27'
    ]
    assert get_seasons_between_dates('2024-02-29', '2025-03-01') == [
        '2023-24', '2024-25'
    ]
    with pytest.raises(ValueError, match='start_date'):
        get_seasons_between_dates('2026-10-25', '2026-04-10')


def test_undated_status_rejected_with_cutoff():
    with pytest.raises(ValueError, match='AVAILABLE_AT'):
        status_rule_baseline(
            pd.DataFrame({'PLAYER_ID': [1]}),
            pd.DataFrame({'PLAYER_ID': [1], 'STATUS': ['OUT']}),
            cutoff='2026-10-20T13:00:00Z',
        )


@pytest.mark.parametrize('methods', [('platt',), ('isotonic',), ('identity',)])
def test_small_calibration_and_refit(methods):
    calibrator = ProbabilityCalibrator(methods=methods)
    calibrator.fit(np.linspace(.1, .9, 40), [0, 1] * 20,
                   evidence_kind='heldout')
    calibrator.fit([.2, .3, .4, .5], [0, 1, 0, 1], evidence_kind='heldout')
    assert set(calibrator.method_scores_) == {'identity'}
    assert calibrator.metadata()['evidence_rows'] == 4
    assert calibrator.metadata()['insufficient_evidence']
    assert np.isfinite(calibrator.predict([0, .5, 1])).all()
    with pytest.raises(ValueError):
        calibrator.fit([.2], [0], evidence_kind='heldout')
    with pytest.raises(RuntimeError):
        calibrator.predict([.5])
