"""Four independent chronological roles with concrete game/request membership."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math

import pandas as pd


ROLES = ("fit", "tune", "calibrate", "outer_test")


@dataclass(frozen=True)
class EvaluationPolicy:
    min_fit_games: int = 60
    min_tune_games: int = 20
    min_calibrate_games: int = 20
    min_outer_games: int = 20
    min_folds: int = 2
    min_slice_rows: int = 100
    simulations: int = 1000
    seed: int = 42
    bootstrap_samples: int = 2000
    target_scales: tuple[tuple[str, float], ...] = (
        ("PTS", 20.0), ("REB", 8.0), ("AST", 5.0),
        ("STL", 1.0), ("BLK", 1.0), ("TOV", 2.0),
    )

    def __post_init__(self):
        for name in ("min_fit_games", "min_tune_games", "min_calibrate_games",
                     "min_outer_games", "min_folds", "min_slice_rows", "simulations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.bootstrap_samples < 100:
            raise ValueError("At least 100 game bootstrap samples are required")
        if (set(dict(self.target_scales)) != {"PTS", "REB", "AST", "STL", "BLK", "TOV"}
                or len(self.target_scales) != 6
                or any(not math.isfinite(value) or value <= 0 for _, value in self.target_scales)):
            raise ValueError("Declare six positive target normalization scales")

    def to_dict(self):
        payload = asdict(self)
        payload["normalization"] = "macro_mean_of_target_MAE_divided_by_frozen_scale"
        payload["bootstrap_unit"] = "game"
        payload["comparator_selection"] = "lowest_tune_normalized_mae_then_name"
        return payload

    @property
    def policy_id(self):
        return digest_payload(self.to_dict())


def digest_payload(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def four_role_folds(requests: pd.DataFrame, *, policy: EvaluationPolicy,
                    min_train_days=90, tune_days=14, calibrate_days=14,
                    outer_days=28, step_days=None):
    """Skip sparse calendar windows, never silently overlap or reuse roles."""
    for value in (min_train_days, tune_days, calibrate_days, outer_days):
        if type(value) is not int or value < 1:
            raise ValueError("Chronological windows must be positive integer days")
    step = outer_days if step_days is None else step_days
    if type(step) is not int or step < outer_days:
        raise ValueError("Outer-test windows must not overlap")
    required = {"GAME_ID", "GAME_DATE", "REQUEST_ID", "SOURCE_SNAPSHOT_ID"}
    if required - set(requests):
        raise ValueError("Persisted requests require game/date/request/snapshot identities")
    frame = requests.copy()
    if frame[list(required)].isna().any().any() or frame.REQUEST_ID.duplicated().any():
        raise ValueError("Request identities must be complete and unique")
    frame["GAME_DATE"] = pd.to_datetime(frame.GAME_DATE, errors="raise").dt.date
    if frame.groupby("GAME_ID").GAME_DATE.nunique().gt(1).any():
        raise ValueError("One game cannot span chronological dates")
    if frame.empty:
        raise ValueError("No historical requests available")
    start, end = frame.GAME_DATE.min(), frame.GAME_DATE.max()
    cursor = start + timedelta(days=min_train_days - 1)
    accepted, skipped = [], []
    minima = dict(zip(ROLES, (policy.min_fit_games, policy.min_tune_games,
                              policy.min_calibrate_games, policy.min_outer_games)))
    while cursor + timedelta(days=tune_days + calibrate_days + outer_days) <= end:
        tune_end = cursor + timedelta(days=tune_days)
        calibration_end = tune_end + timedelta(days=calibrate_days)
        outer_end = calibration_end + timedelta(days=outer_days)
        windows = {
            "fit": (start, cursor),
            "tune": (cursor + timedelta(days=1), tune_end),
            "calibrate": (tune_end + timedelta(days=1), calibration_end),
            "outer_test": (calibration_end + timedelta(days=1), outer_end),
        }
        roles = {}
        for role, (first, last) in windows.items():
            block = frame.loc[frame.GAME_DATE.between(first, last)].sort_values("REQUEST_ID")
            roles[role] = {
                "start": first.isoformat(), "end": last.isoformat(),
                "game_ids": sorted(block.GAME_ID.astype(str).unique()),
                "request_ids": block.REQUEST_ID.astype(str).tolist(),
                "input_hash": digest_payload(block.to_dict("records")),
                "request_records": block.to_dict("records"),
                "snapshot_ids": sorted(block.SOURCE_SNAPSHOT_ID.astype(str).unique()),
            }
        sparse = [role for role in ROLES if len(roles[role]["game_ids"]) < minima[role]]
        record = {"roles": roles, "policy_id": policy.policy_id}
        record["fold_id"] = digest_payload(record)
        if sparse:
            skipped.append({**record, "reason": "insufficient_games", "sparse_roles": sparse})
        else:
            accepted.append(record)
        cursor += timedelta(days=step)
    return {"schema_version": "four_role_folds_v1", "policy": policy.to_dict(),
            "policy_id": policy.policy_id, "folds": accepted, "skipped": skipped,
            "sufficient_folds": len(accepted) >= policy.min_folds}


def validate_four_role_membership(folds):
    """Recompute memberships and reject reused outer games or mixed roles."""
    outer_seen, identifiers = set(), set()
    for fold in folds:
        if fold.get("fold_id") in identifiers:
            raise ValueError("Duplicate evaluation fold identity")
        identifiers.add(fold.get("fold_id"))
        roles = fold.get("roles", {})
        if set(roles) != set(ROLES):
            raise ValueError("Promotion requires fit, tune, calibrate, and outer_test roles")
        previous_end, game_seen, request_seen = None, set(), set()
        for role in ROLES:
            record = roles[role]
            start, end = pd.Timestamp(record["start"]).date(), pd.Timestamp(record["end"]).date()
            if start > end or (previous_end is not None and start <= previous_end):
                raise ValueError("Fold roles overlap or violate chronology")
            previous_end = end
            rows = record.get("request_records")
            if not rows or digest_payload(rows) != record.get("input_hash"):
                raise ValueError("Fold request membership hash mismatch")
            games = {str(row["GAME_ID"]) for row in rows}
            requests = [str(row["REQUEST_ID"]) for row in rows]
            if (games != set(record["game_ids"]) or set(requests) != set(record["request_ids"])
                    or len(requests) != len(set(requests)) or games & game_seen
                    or set(requests) & request_seen):
                raise ValueError("Fold game/request membership is inconsistent")
            if any(not start <= pd.Timestamp(row["GAME_DATE"]).date() <= end for row in rows):
                raise ValueError("A request is outside its declared chronological role")
            game_seen |= games
            request_seen |= set(requests)
            if role == "outer_test":
                if outer_seen & games:
                    raise ValueError("Outer-test games are reused across folds")
                outer_seen |= games
