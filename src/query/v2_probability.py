"""Probability queries over canonical Model v2 forecast distributions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


def probability_at_line(row: pd.Series, line: float) -> dict[str, object]:
    """Return under, push, and over probabilities for one betting line.

    A calibrated discrete PMF or declared sample representation is used when
    present. Quantile-only artifacts retain a clearly marked interpolation
    estimate; they cannot identify the point mass at an arbitrary integer.
    """

    try:
        checked_line = float(line)
    except (TypeError, ValueError) as exc:
        raise ValueError("Query line must be a finite number") from exc
    if not math.isfinite(checked_line):
        raise ValueError("Query line must be a finite number")

    distribution = _distribution_from_row(row)
    if distribution is not None:
        values, masses, method, exact = distribution
        under = float(masses[values < checked_line].sum())
        push = float(masses[values == checked_line].sum())
        over = float(masses[values > checked_line].sum())
        return _probability_result(
            under,
            push,
            over,
            method=method,
            exact=exact,
        )

    zero_prob = _finite_probability(row.get("ZERO_PROB", 0.0), "ZERO_PROB")
    # All canonical Model v2 stat outputs are nonnegative count quantities.
    # The zero mass therefore remains exact even when no full distribution was
    # persisted, including for negative and sub-one lines.
    if checked_line < 0:
        return _probability_result(
            0.0,
            0.0,
            1.0,
            method="known_zero_mass_only",
            exact=True,
            limitation="Only nonnegative support and ZERO_PROB are known for this artifact.",
        )
    if checked_line == 0:
        return _probability_result(
            0.0,
            zero_prob,
            1.0 - zero_prob,
            method="known_zero_mass_only",
            exact=True,
            limitation="Only ZERO_PROB is exact; no calibrated discrete distribution is stored.",
        )
    if 0 < checked_line < 1:
        return _probability_result(
            zero_prob,
            0.0,
            1.0 - zero_prob,
            method="known_zero_mass_only",
            exact=True,
            limitation=(
                "Only nonnegative integer support and ZERO_PROB are exact "
                "for this artifact."
            ),
        )

    under = _interpolated_cdf(row, checked_line)
    return _probability_result(
        under,
        0.0,
        1.0 - under,
        method="quantile_interpolation_estimate",
        exact=False,
        limitation=(
            "Quantiles alone cannot recover exact mass at an arbitrary integer "
            "line; store a calibrated discrete distribution or samples."
        ),
    )


def _probability_result(
    under: float,
    push: float,
    over: float,
    *,
    method: str,
    exact: bool,
    limitation: str | None = None,
) -> dict[str, object]:
    probabilities = np.array([under, push, over], dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Probability calculation produced an invalid total")
    probabilities /= total
    under_value = round(float(probabilities[0]), 12)
    push_value = round(float(probabilities[1]), 12)
    over_value = max(0.0, round(1.0 - under_value - push_value, 12))
    result: dict[str, object] = {
        "under": under_value,
        "push": push_value,
        "over": over_value,
        "probability_method": method,
        "probability_exact": exact,
    }
    if limitation:
        result["probability_limitation"] = limitation
    return result


def _finite_probability(value: object, name: str) -> float:
    try:
        checked = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not math.isfinite(checked) or not 0 <= checked <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return checked


def _interpolated_cdf(row: pd.Series, line: float) -> float:
    points = [
        (0.0, _finite_probability(row.get("ZERO_PROB", 0.0), "ZERO_PROB")),
        (float(row["P10"]), 0.10),
        (float(row["P25"]), 0.25),
        (float(row["P50"]), 0.50),
        (float(row["P75"]), 0.75),
        (float(row["P90"]), 0.90),
    ]
    if any(not math.isfinite(value) for value, _ in points):
        raise ValueError("Forecast quantiles must be finite")
    collapsed: dict[float, float] = {}
    for value, probability in points:
        collapsed[value] = max(collapsed.get(value, 0.0), probability)
    values = np.array(sorted(collapsed), dtype=float)
    probabilities = np.maximum.accumulate(
        np.array([collapsed[value] for value in values], dtype=float)
    )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    if line < values[0]:
        return 0.0
    if line >= values[-1]:
        spread = max(1.0, values[-1] - values[0])
        upper = values[-1] + spread
        return float(np.interp(line, [values[-1], upper], [probabilities[-1], 1.0]))
    return float(np.clip(np.interp(line, values, probabilities), 0.0, 1.0))


def _distribution_from_row(
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, str, bool] | None:
    for name in (
        "DISCRETE_PMF", "CALIBRATED_PMF", "DISTRIBUTION_PMF", "PMF",
        "FORECAST_PMF", "PROBABILITY_MASS_FUNCTION",
    ):
        if name in row and not _is_missing(row[name]):
            return _parse_pmf(row[name], name)

    for name in ("DISTRIBUTION", "DISCRETE_DISTRIBUTION", "CALIBRATED_DISTRIBUTION"):
        if name not in row or _is_missing(row[name]):
            continue
        payload = _decode_json(row[name])
        if isinstance(payload, Mapping) and "pmf" in payload:
            return _parse_pmf(payload["pmf"], f"{name}.pmf")
        if isinstance(payload, Mapping) and "samples" in payload:
            return _parse_samples(payload["samples"], f"{name}.samples")
        if isinstance(payload, Mapping):
            return _parse_pmf(payload, name)
        if isinstance(payload, (list, tuple)):
            return _parse_samples(payload, name)

    for name in (
        "DISTRIBUTION_SAMPLES",
        "CALIBRATED_SAMPLES",
        "SIMULATION_SAMPLES",
        "SAMPLES", "SAMPLE_VALUES",
    ):
        if name in row and not _is_missing(row[name]):
            return _parse_samples(row[name], name)
    return None


def _parse_pmf(
    payload: object, name: str
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    decoded = _decode_json(payload)
    if isinstance(decoded, Mapping):
        items = decoded.items()
    elif isinstance(decoded, list):
        items = []
        for item in decoded:
            if not isinstance(item, Mapping) or "value" not in item:
                raise ValueError(f"{name} list entries require value and probability")
            probability = item.get("probability", item.get("mass"))
            items.append((item["value"], probability))
    else:
        raise ValueError(f"{name} must be a value-to-probability mapping")
    values: list[float] = []
    masses: list[float] = []
    for value, mass in items:
        try:
            checked_value = float(value)
            checked_mass = float(mass)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a nonnumeric value or mass") from exc
        if not math.isfinite(checked_value) or not math.isfinite(checked_mass) or checked_mass < 0:
            raise ValueError(f"{name} must contain finite nonnegative masses")
        if checked_value < 0 or not checked_value.is_integer():
            raise ValueError(f"{name} support must be nonnegative integers")
        if checked_value in values:
            raise ValueError(f"{name} contains duplicate support")
        values.append(checked_value)
        masses.append(checked_mass)
    if not values:
        raise ValueError(f"{name} must not be empty")
    total = float(sum(masses))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} masses must sum to one")
    return (
        np.asarray(values, dtype=float),
        np.asarray(masses, dtype=float) / total,
        "declared_discrete_pmf",
        True,
    )


def _parse_samples(
    payload: object, name: str
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    decoded = _decode_json(payload)
    if not isinstance(decoded, (list, tuple)) or not decoded:
        raise ValueError(f"{name} must be a non-empty sample list")
    try:
        values = np.asarray([float(value) for value in decoded], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric samples") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite samples")
    if (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"{name} support must be nonnegative integers")
    unique, counts = np.unique(values, return_counts=True)
    return (
        unique,
        counts.astype(float) / len(values),
        "declared_empirical_sample_mass",
        False,
    )


def _decode_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Distribution evidence is not valid JSON") from exc
    return value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def select_forecast(
    forecasts: pd.DataFrame,
    *,
    player: str,
    stat: str,
    game_date: str | None = None,
    player_names: pd.DataFrame | None = None,
) -> pd.Series:
    """Select the latest matching immutable forecast row."""

    if forecasts.empty:
        raise ValueError("No Model v2 forecasts are available")
    frame = forecasts.copy()
    if game_date is not None:
        if "GAME_DATE" not in frame:
            raise ValueError("Forecast rows do not contain GAME_DATE")
        dates = pd.to_datetime(frame["GAME_DATE"], errors="raise").dt.date.astype(str)
        frame = frame.loc[dates.eq(str(game_date))]
    requested = str(player).strip()
    id_match = frame["PLAYER_ID"].astype(str).eq(requested)
    if "PLAYER_NAME" in frame:
        name_match = frame["PLAYER_NAME"].astype(str).str.casefold().eq(requested.casefold())
    elif player_names is not None and {"PLAYER_ID", "PLAYER_NAME"}.issubset(player_names):
        names = (
            player_names[["PLAYER_ID", "PLAYER_NAME"]]
            .dropna()
            .drop_duplicates("PLAYER_ID", keep="last")
            .assign(PLAYER_ID=lambda item: item["PLAYER_ID"].astype(str))
        )
        name_ids = set(
            names.loc[
                names["PLAYER_NAME"].astype(str).str.casefold().eq(requested.casefold()),
                "PLAYER_ID",
            ]
        )
        name_match = frame["PLAYER_ID"].astype(str).isin(name_ids)
    else:
        name_match = pd.Series(False, index=frame.index)
    frame = frame.loc[(id_match | name_match) & frame["STAT"].astype(str).str.upper().eq(stat.upper())]
    if frame.empty:
        raise ValueError(f"No {stat.upper()} forecast found for {player!r}")
    generated = pd.to_datetime(frame["GENERATED_AT"], errors="coerce", utc=True)
    return frame.loc[generated.sort_values(kind="stable").index[-1]]


__all__ = ["probability_at_line", "select_forecast"]
