"""One empirical law for service summaries, simulation, and probability queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from src.contracts.forecast import ForecastRequest, validate_forecast_frame


@dataclass(frozen=True)
class GameDistribution:
    request: ForecastRequest
    forecasts: pd.DataFrame
    samples: pd.DataFrame
    seed: int
    simulations: int


def summarize_game(request, contexts, samples, *, targets, seed, simulations,
                   full_game: bool):
    """Derive every advertised moment from the bound samples, without refitting."""
    samples = samples.copy()
    samples["REQUEST_ID"] = request.request_id
    samples["GAME_ID"] = request.game_id
    samples["MODEL_BUNDLE_ID"] = request.model_bundle_id
    samples["SOURCE_SNAPSHOT_ID"] = request.source_snapshot_id
    digest = hashlib.sha256(samples.to_json(orient="split", index=False).encode()).hexdigest()
    generated = datetime.now(timezone.utc).isoformat()
    records = []
    for _, context in contexts.iterrows():
        for stat in targets:
            draws = samples.loc[samples.PLAYER_ID.eq(context.PLAYER_ID)
                                & samples.TEAM_ID.eq(context.TEAM_ID)
                                & samples.STAT.eq(stat)].sort_values("SIMULATION")
            if (len(draws) != simulations
                    or draws.SIMULATION.tolist() != list(range(simulations))):
                raise ValueError("Incomplete or duplicate player distribution draws")
            values = draws.VALUE.to_numpy()
            appeared = draws.APPEARED.to_numpy(dtype=bool)
            regulation = draws.MINUTES.to_numpy()
            conditional = regulation[appeared]
            p_active = float(draws.ACTIVE.mean())
            p_play = float(appeared.mean())
            quantiles = np.quantile(values, [.05, .1, .25, .5, .75, .9, .95],
                                    method="inverted_cdf")
            minutes_quantiles = (np.quantile(conditional, [.1, .5, .9])
                                 if len(conditional) else np.zeros(3))
            records.append({
                "REQUEST_ID": request.request_id,
                "MODEL_BUNDLE_ID": request.model_bundle_id,
                "SOURCE_SNAPSHOT_ID": request.source_snapshot_id,
                "GENERATED_AT": generated, "FORECAST_CUTOFF": request.forecast_cutoff.isoformat(),
                "GAME_ID": request.game_id, "GAME_DATE": request.game_date.isoformat(),
                "SCHEDULE_VERSION": request.schedule_version,
                "PLAYER_ID": context.PLAYER_ID, "TEAM_ID": context.TEAM_ID,
                "OPPONENT_ID": context.get("OPPONENT_ID", ""), "HORIZON": request.horizon,
                "P_ACTIVE": p_active, "P_PLAY_GIVEN_ACTIVE": p_play / p_active if p_active else 0.0,
                "PLAY_PROB": p_play,
                "INPUT_P_ACTIVE": float(context.get("P_ACTIVE", 1.0)),
                "INPUT_P_PLAY_GIVEN_ACTIVE": float(context.get("P_PLAY_GIVEN_ACTIVE", context.PLAY_PROB)),
                "INPUT_PLAY_PROB": float(context.PLAY_PROB),
                "EXPECTED_MINUTES": float(conditional.mean()) if len(conditional) else 0.0,
                "UNCONDITIONAL_MINUTES": float(regulation.mean()),
                "EXPECTED_OVERTIME_MINUTES": float(draws.OVERTIME_MINUTES.mean()),
                "EXPECTED_FULL_GAME_MINUTES": float(draws.FULL_GAME_MINUTES.mean()),
                "MIN_P10": minutes_quantiles[0], "MIN_P50": minutes_quantiles[1],
                "MIN_P90": minutes_quantiles[2], "STAT": stat, "MEAN": float(values.mean()),
                "P05": quantiles[0], "P10": quantiles[1], "P25": quantiles[2],
                "P50": quantiles[3], "P75": quantiles[4], "P90": quantiles[5],
                "P95": quantiles[6], "ZERO_PROB": float((values == 0).mean()),
                "STATUS": str(context.get("STATUS", "UNKNOWN")),
                "COLD_START": bool(context.get("PRIOR_APPEARANCES", 0) == 0),
                "DATA_QUALITY": str(context.get("DATA_QUALITY", "DIAGNOSTIC")),
                "CALIBRATION_VERSION": request.model_bundle_id, "SCENARIO": request.scenario,
                "DISTRIBUTION_KIND": ("empirical_full_game_v2" if full_game
                                      else "empirical_regulation_v2"),
                "DISTRIBUTION_SEED": seed, "DISTRIBUTION_DRAW_COUNT": simulations,
                "DISTRIBUTION_DIGEST": digest,
                "DISTRIBUTION_SAMPLES": json.dumps(values.tolist()),
            })
    forecasts = pd.DataFrame(records)
    validate_forecast_frame(forecasts)
    return GameDistribution(request, forecasts, samples, seed, simulations)
