"""Complete eligible populations and explicitly known player-game labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.canonical_data import normalize_id_series, require_columns
from src.contracts.errors import ContractError
from src.contracts.forecast import ForecastRequest
from src.features.snapshot_inputs import assert_snapshot_game, load_official_roster


TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")
LABELS = ("ACTIVE", "APPEARED", "STARTED", "REGULATION_MINUTES", "OVERTIME_MINUTES",
          "MIN", *TARGETS)


def build_eligible_panel(data_dir, request: ForecastRequest,
                         outcomes: pd.DataFrame | None = None):
    """Left-join labels to the official cutoff roster, retaining unknowns.

    Absence from a box score never supplies a negative participation label.
    Explicit DNPs may carry APPEARED=0; unknown active status stays unknown.
    Full-game minutes alone cannot reveal a player's overtime allocation.
    """
    assert_snapshot_game(data_dir, request)
    panel = load_official_roster(data_dir, request)
    panel["REQUEST_ID"] = request.request_id
    panel["GAME_ID"] = request.game_id
    panel["GAME_DATE"] = request.game_date.isoformat()
    panel["SOURCE_SNAPSHOT_ID"] = request.source_snapshot_id
    panel["FORECAST_CUTOFF"] = request.forecast_cutoff.isoformat()
    panel["HORIZON"] = request.horizon
    panel["ROSTER_ELIGIBLE"] = 1
    truth = normalize_outcomes(outcomes) if outcomes is not None else pd.DataFrame()
    keys = ["GAME_ID", "TEAM_ID", "PLAYER_ID"]
    if not truth.empty:
        truth = truth.loc[truth.GAME_ID.eq(request.game_id)].copy()
        eligible_keys = pd.MultiIndex.from_frame(panel[keys])
        truth_keys = pd.MultiIndex.from_frame(truth[keys])
        quarantined = truth.loc[~truth_keys.isin(eligible_keys)].copy()
        matched = truth.loc[truth_keys.isin(eligible_keys)].copy()
        panel = panel.merge(matched, on=keys, how="left", validate="one_to_one")
    else:
        quarantined = truth
    for label in LABELS:
        if label not in panel:
            panel[label] = np.nan
    panel["LABEL_KNOWN"] = panel["APPEARED"].notna()
    report = {
        "request_id": request.request_id,
        "expected_players": len(panel),
        "known_participation": int(panel.APPEARED.notna().sum()),
        "unknown_participation": int(panel.APPEARED.isna().sum()),
        "known_active": int(panel.ACTIVE.notna().sum()),
        "quarantined_outcomes": len(quarantined),
        "targets": {target: {"known": int(panel[target].notna().sum()),
                             "missing": int(panel[target].isna().sum())}
                    for target in TARGETS},
    }
    return panel, quarantined, report


def normalize_outcomes(outcomes: pd.DataFrame):
    """Validate labels without converting missing values into zeroes."""
    frame = outcomes.copy()
    keys = ["GAME_ID", "TEAM_ID", "PLAYER_ID"]
    require_columns(frame, keys, source="official outcomes")
    for key in keys:
        frame[key] = normalize_id_series(frame[key], field=key)
    if frame.duplicated(keys).any():
        raise ContractError("Duplicate player-team-game outcome labels")
    for label in LABELS:
        if label not in frame:
            frame[label] = np.nan
        original = frame[label]
        values = pd.to_numeric(original, errors="coerce")
        if (original.notna() & values.isna()).any():
            raise ContractError(f"Malformed outcome label: {label}")
        if ((values.notna() & ~np.isfinite(values)) | (values < 0)).any():
            raise ContractError(f"Invalid outcome label: {label}")
        if label in ("ACTIVE", "APPEARED", "STARTED"):
            if not values.dropna().isin([0, 1]).all():
                raise ContractError(f"Binary outcome label required: {label}")
        elif label in TARGETS and not values.dropna().mod(1).eq(0).all():
            raise ContractError(f"Integer count outcome required: {label}")
        frame[label] = values
    if frame.REGULATION_MINUTES.gt(48).any():
        raise ContractError("Regulation minutes cannot exceed 48")
    if (frame.MIN.notna() & frame.REGULATION_MINUTES.notna()
            & frame.OVERTIME_MINUTES.notna()
            & ~np.isclose(frame.MIN, frame.REGULATION_MINUTES + frame.OVERTIME_MINUTES,
                          atol=1e-4)).any():
        raise ContractError("Full-game minutes differ from regulation plus overtime")
    appeared = frame.APPEARED.mask(frame.APPEARED.isna() & frame.MIN.gt(0), 1.0)
    positive_minutes = frame[["MIN", "REGULATION_MINUTES", "OVERTIME_MINUTES"]].gt(0).any(axis=1)
    if ((appeared.eq(0) & (positive_minutes | frame[list(TARGETS)].gt(0).any(axis=1)
                          | frame.STARTED.eq(1)))
            | (frame.ACTIVE.eq(0) & appeared.eq(1))).any():
        raise ContractError("Participation labels contradict known outcomes")
    frame["APPEARED"] = appeared
    return frame[keys + list(LABELS)]
