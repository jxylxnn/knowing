"""Participation and minutes comparisons on the complete eligible population."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.forecast import ForecastRequest
from src.data.snapshots import validate_source_snapshot
from src.features.snapshot_inputs import load_request_history, read_snapshot_csv
from src.forecasting.availability import appearance_participation_baseline, status_rule_baseline


def score_opportunity(records, forecasts, actuals, *, data_dir, minimum_rows):
    results = []
    for record in records:
        request = ForecastRequest(**record["request"])
        predicted = forecasts.loc[forecasts.REQUEST_ID.eq(request.request_id)].drop_duplicates("PLAYER_ID").copy()
        predicted["PLAYER_ID"] = predicted.PLAYER_ID.astype(str)
        rule = status_rule_baseline(predicted[["PLAYER_ID", "TEAM_ID", "STATUS"]])
        manifest = validate_source_snapshot(data_dir, request.source_snapshot_id,
                                            forecast_cutoff=request.forecast_cutoff)
        if any(item.relative_path == "player_game_eligibility.csv" for item in manifest.files):
            eligibility = read_snapshot_csv(data_dir, request.source_snapshot_id,
                                             ("player_game_eligibility.csv",), cutoff=request.forecast_cutoff)
            dates = pd.to_datetime(eligibility.GAME_DATE, errors="raise").dt.date
            eligibility = eligibility.loc[dates < request.game_date]
        else:
            eligibility = pd.DataFrame(columns=["PLAYER_ID"])
        frequency = appearance_participation_baseline(eligibility, predicted[["PLAYER_ID"]])
        history = load_request_history(data_dir, request).sort_values(["GAME_DATE", "GAME_ID"])
        history["MIN"] = pd.to_numeric(history.MIN, errors="raise")
        recent = history.groupby("PLAYER_ID").MIN.apply(lambda group: group.tail(10).mean())
        # Roles are explicitly defined from prior workload, never current minutes.
        bands = pd.cut(recent, [-np.inf, 15, 30, np.inf], labels=["low", "mid", "high"])
        medians = recent.groupby(bands, observed=True).median()
        role_minutes = bands.map(medians).astype(float)
        predicted["STATUS_ACTIVE"] = rule.P_ACTIVE.to_numpy()
        predicted["STATUS_PLAY"] = rule.PLAY_PROB.to_numpy()
        predicted["FREQUENCY_ACTIVE"] = frequency.P_ACTIVE.to_numpy()
        predicted["FREQUENCY_PLAY"] = frequency.PLAY_PROB.to_numpy()
        predicted["ROLLING_MINUTES"] = predicted.PLAYER_ID.map(recent)
        predicted["ROLE_MINUTES"] = predicted.PLAYER_ID.map(role_minutes)
        predicted["CANDIDATE_MINUTES"] = np.divide(
            predicted.EXPECTED_FULL_GAME_MINUTES, predicted.PLAY_PROB,
            out=np.zeros(len(predicted)), where=predicted.PLAY_PROB.to_numpy() > 0,
        )
        results.append(predicted)
    combined = pd.concat(results, ignore_index=True)
    labels = actuals[["GAME_ID", "PLAYER_ID", "ACTIVE", "APPEARED", "MIN"]].copy()
    labels["PLAYER_ID"] = labels.PLAYER_ID.astype(str)
    for column in ("ACTIVE", "APPEARED", "MIN"):
        labels[column] = pd.to_numeric(labels[column], errors="raise")
    joined = combined.merge(labels, on=["GAME_ID", "PLAYER_ID"], how="left", validate="many_to_one")
    probabilities = {}
    wins = []
    for label, columns in {
        "ACTIVE": ("P_ACTIVE", "STATUS_ACTIVE", "FREQUENCY_ACTIVE"),
        "APPEARED": ("PLAY_PROB", "STATUS_PLAY", "FREQUENCY_PLAY"),
    }.items():
        known = joined[label].notna()
        truth = joined.loc[known, label]
        if not truth.isin([0, 1]).all():
            raise ValueError("Opportunity outcome labels must be binary")
        metrics = {"rows": int(known.sum()), "missing": int((~known).sum())}
        for name, column in zip(("candidate", "status_rule", "appearance_rule"), columns):
            predicted = joined.loc[known, column]
            clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
            metrics[name] = {
                "brier": float(np.square(predicted - truth).mean()) if known.any() else None,
                "log_loss": float(-(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)).mean()) if known.any() else None,
            }
        enough = known.sum() >= minimum_rows and truth.nunique() == 2
        wins.append(bool(enough and all(metrics["candidate"]["brier"] < metrics[baseline]["brier"]
                                        for baseline in ("status_rule", "appearance_rule"))))
        probabilities[label.lower()] = metrics
    known_minutes = (joined.APPEARED.eq(1) & joined.MIN.notna()
                     & joined.ROLLING_MINUTES.notna() & joined.ROLE_MINUTES.notna())
    minutes = {"rows": int(known_minutes.sum()), "missing": int((~known_minutes).sum()),
               "role_definition": "prior_rolling_10_workload_band"}
    for name, column in (("candidate", "CANDIDATE_MINUTES"), ("rolling_10", "ROLLING_MINUTES"),
                         ("role_median", "ROLE_MINUTES")):
        minutes[name] = (float((joined.loc[known_minutes, column] - joined.loc[known_minutes, "MIN"]).abs().mean())
                         if known_minutes.any() else None)
    minutes["beats_baseline"] = bool(known_minutes.sum() >= minimum_rows and all(
        minutes["candidate"] < minutes[baseline] for baseline in ("rolling_10", "role_median")
    ))
    probabilities["beats_baseline"] = all(wins)
    return {"participation": probabilities, "minutes": minutes,
            "minimum_rows": minimum_rows, "expected_rows": len(joined)}
