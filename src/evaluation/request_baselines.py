"""Declared point baselines on exactly the replay request/player population."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.eligible_panel import TARGETS
from src.features.snapshot_inputs import load_request_history


BASELINES = ("rolling_5", "rolling_10", "rolling_20", "ema_10", "minutes_x_rate_10")


def request_baselines(records, forecasts, *, data_dir):
    output = []
    for record in records:
        request = ForecastRequest(**record["request"])
        history = load_request_history(data_dir, request).sort_values(["GAME_DATE", "GAME_ID"])
        players = forecasts.loc[forecasts.REQUEST_ID.eq(request.request_id), "PLAYER_ID"].unique()
        for player in players:
            prior = history.loc[history.PLAYER_ID.astype(str).eq(str(player))]
            for stat in TARGETS:
                values = pd.to_numeric(prior[stat], errors="raise")
                minutes = pd.to_numeric(prior.MIN, errors="raise")
                recent, exposure = values.tail(10), minutes.tail(10)
                row = {"REQUEST_ID": request.request_id, "GAME_ID": request.game_id,
                       "PLAYER_ID": str(player), "STAT": stat}
                for window in (5, 10, 20):
                    row[f"rolling_{window}"] = float(values.tail(window).mean())
                row["ema_10"] = float(values.ewm(span=10, adjust=False).mean().iloc[-1]) if len(values) else np.nan
                valid = recent.notna() & exposure.notna() & exposure.gt(0)
                row["minutes_x_rate_10"] = (
                    float(exposure[valid].mean() * recent[valid].sum() / exposure[valid].sum())
                    if valid.any() else np.nan
                )
                output.append(row)
    return pd.DataFrame(output)


def paired_baseline_scores(detail, baselines, *, target_scales):
    keys = ["REQUEST_ID", "GAME_ID", "PLAYER_ID", "STAT"]
    frame = detail.copy()
    frame["PLAYER_ID"] = frame.PLAYER_ID.astype(str)
    joined = frame.merge(baselines, on=keys, how="left", validate="one_to_one")
    eligible = joined.KNOWN & joined[list(BASELINES)].notna().all(axis=1)
    selected = joined.loc[eligible].copy()
    scores = {}
    for name in ("candidate", *BASELINES):
        prediction = selected.MEAN if name == "candidate" else selected[name]
        errors = (prediction - selected.ACTUAL).abs()
        targets = {stat: (float(errors.loc[selected.STAT.eq(stat)].mean())
                          if selected.STAT.eq(stat).any() else None)
                   for stat in TARGETS}
        scores[name] = {
            "targets": targets,
            "normalized_mae": (float(np.mean([targets[stat] / target_scales[stat] for stat in TARGETS]))
                               if all(value is not None for value in targets.values()) else None),
            "rows": len(selected), "expected_rows": len(joined),
            "missing_rows": len(joined) - len(selected),
        }
    return scores, selected
