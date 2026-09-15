"""Runtime backend for an immutable Model v2 rolling baseline bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.forecasting.minutes import allocate_team_minutes
from src.forecasting.availability import status_rule_baseline
from src.models.bundle import validate_v2_bundle
from src.models.versioning import ModelBundleManifest


class V2BaselineBackend:
    targets = ("PTS", "REB", "AST", "STL", "BLK", "TOV")

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        validate_v2_bundle(self.bundle_dir, promotion=False)
        manifest = ModelBundleManifest.load(
            self.bundle_dir / ModelBundleManifest.FILE_NAME
        )
        self.model_version = manifest.bundle_id
        self.training_cutoff = manifest.cutoffs.get("training")
        learned_dates = [manifest.cutoffs.get(role) for role in ("training", "validation", "calibration")]
        self.learning_cutoff = max((value for value in learned_dates if value), default=None)
        self.availability = _load(
            self.bundle_dir / "availability_model" / "baseline.json"
        )
        self.minutes = _load(
            self.bundle_dir / "minutes_models" / "baseline.json"
        )
        self.rates = _load(
            self.bundle_dir / "stat_rate_models" / "baseline.json"
        )
        self.calibration = _load(self.bundle_dir / "calibrators" / "baseline.json")
        self.simulation_parameters = _load(self.bundle_dir / "simulation_parameters.json")
        overtime = self.simulation_parameters.get("overtime")
        self.overtime_probabilities = (
            tuple(overtime["probabilities"]) if overtime and overtime.get("fitted") else None
        )
        # Baseline bundles remain diagnostic until their official evidence is qualified.
        self.supports_official = False
        self.models = {target: self.rates for target in self.targets}

    def load_models(self) -> None:
        return None

    def prepare_contexts(self, contexts: pd.DataFrame) -> pd.DataFrame:
        result = contexts.copy()
        if "STATUS" in result and "P_ACTIVE" not in result:
            result = status_rule_baseline(
                result, default_active_probability=self.availability["default_p_active"],
                default_play_given_active=self.availability["default_p_play_given_active"],
            )
        result["P_ACTIVE"] = result.get(
            "P_ACTIVE", self.availability["default_p_active"]
        )
        result["P_PLAY_GIVEN_ACTIVE"] = result.get(
            "P_PLAY_GIVEN_ACTIVE",
            self.availability["default_p_play_given_active"],
        )
        result["PLAY_PROB"] = (
            pd.to_numeric(result["P_ACTIVE"], errors="raise")
            * pd.to_numeric(result["P_PLAY_GIVEN_ACTIVE"], errors="raise")
        )
        player_minutes = self.minutes.get("players", {})
        default = float(self.minutes.get("global_minutes", 24.0))
        result["EXPECTED_MINUTES_RAW"] = [
            float(player_minutes.get(str(player_id), {}).get("mean", default))
            for player_id in result["PLAYER_ID"]
        ]
        result["MINUTES_STD"] = [
            float(player_minutes.get(str(player_id), {}).get("std", 4.0))
            for player_id in result["PLAYER_ID"]
        ]
        result = allocate_team_minutes(result)
        result["DATA_QUALITY"] = result.get("DATA_QUALITY", "DEGRADED_BASELINE")
        return result

    def predict_player_stats(
        self,
        context: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
        include_confidence: bool = False,
    ) -> dict[str, float]:
        row = context.iloc[0]
        player_id = str(row["PLAYER_ID"])
        minutes = float(row.get("EXPECTED_MINUTES", row.get("EXPECTED_MINUTES_RAW", 0)))
        player_rates = self.rates.get("players", {}).get(player_id, {})
        global_rates = self.rates.get("global_rates", {})
        output: dict[str, float] = {}
        for target in self.targets:
            rate = player_rates.get(target, global_rates.get(target, {}))
            mean_rate = float(rate.get("mean", 0.0))
            std_rate = float(rate.get("std", 0.0))
            output[target] = max(0.0, minutes * mean_rate)
            if include_confidence:
                output[f"{target}_STD"] = max(0.0, minutes * std_rate)
        return output

    def prepare_sampling_frame(self, contexts: pd.DataFrame) -> pd.DataFrame:
        """Attach conditional rate parameters used by the joint sampler."""

        result = self.prepare_contexts(contexts)
        player_rates = self.rates.get("players", {})
        global_rates = self.rates.get("global_rates", {})
        for target in self.targets:
            result[f"{target}_RATE"] = [
                float(
                    player_rates.get(str(player_id), {})
                    .get(target, global_rates.get(target, {}))
                    .get("mean", 0.0)
                )
                for player_id in result["PLAYER_ID"]
            ]
            result[f"{target}_RATE_STD"] = [
                float(
                    player_rates.get(str(player_id), {})
                    .get(target, global_rates.get(target, {}))
                    .get("std", 0.0)
                )
                for player_id in result["PLAYER_ID"]
            ]
            result[f"{target}_MINUTES_RATE_SLOPE"] = [
                float(player_rates.get(str(player_id), {}).get(target, {}).get("minutes_slope", 0.0))
                for player_id in result["PLAYER_ID"]
            ]
            if self.calibration.get("status") == "independent_mean_scale":
                scale = float(self.calibration["scales"][target])
                result[f"{target}_RATE"] *= scale
                result[f"{target}_RATE_STD"] *= scale
        return result


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model v2 component must be a JSON object: {path}")
    return payload


__all__ = ["V2BaselineBackend"]
