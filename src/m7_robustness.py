"""Module M7: robustness, reliability, sensitivity, and decision analysis.

Run with ``python src/m7_robustness.py``. M7 consumes M5's ``simulate_days``
function and M6's ranked candidates when they are available. Missing upstream
modules are reported explicitly; this module never substitutes fabricated
simulation metrics.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
OUT_DIR = PROJ_DIR / "outputs" / "m7"
M6_CANDIDATES_PATH = PROJ_DIR / "outputs" / "m6" / "m6_top3_candidates.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from data_loader import load_all_data

try:
    from m5_simulator import simulate_days
except ModuleNotFoundError as exc:
    if exc.name != "m5_simulator":
        raise
    simulate_days = None

METRICS = ("S45", "em_p95_min", "road_fallback_share")
WEEK_NAMES = ("Calm Week", "Typical Week", "Worst Monsoon Week")
DEFAULT_SEEDS = 20
DEFAULT_MC_TRIALS = 100


def _fallback_candidates() -> list[dict[str, Any]]:
    """Use the first three M1 designs as non-ranked placeholders."""
    m1_path = PROJ_DIR / "outputs" / "m1" / "m1_designs.json"
    if m1_path.exists():
        with m1_path.open(encoding="utf-8") as file:
            designs = json.load(file)
        if isinstance(designs, list) and designs:
            chosen = designs[:3]
        else:
            chosen = []
    else:
        chosen = []

    if not chosen:
        chosen = [
            {"design_id": "placeholder-1", "R_km": 18, "E_Wh": 599.4},
            {"design_id": "placeholder-2", "R_km": 24, "E_Wh": 1198.8},
            {"design_id": "placeholder-3", "R_km": 20, "E_Wh": 710.4},
        ]

    return [
        {
            **design,
            "candidate_id": str(design.get("candidate_id", design.get("design_id", f"placeholder-{i}"))),
            "is_placeholder": True,
            "network_choice": design.get("network_choice", "not specified (M6 pending)"),
            "vehicle_choice": design.get("vehicle_choice", design.get("design_id", "not specified")),
        }
        for i, design in enumerate(chosen[:3], start=1)
    ]


def _load_candidates(path: Path = M6_CANDIDATES_PATH) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        warnings.warn(
            f"M6 candidates not found at {path}; using three M1 design placeholders.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fallback_candidates(), True

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = payload.get("candidates", payload.get("top3", payload.get("top_3", [])))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty candidate list.")
    candidates = [dict(candidate) for candidate in payload[:3]]
    for index, candidate in enumerate(candidates, start=1):
        candidate.setdefault(
            "candidate_id",
            candidate.get("design_id", candidate.get("name", f"candidate-{index}")),
        )
        candidate.setdefault("is_placeholder", False)
    return candidates, False


def _daily_wind(wind: pd.DataFrame, gust_limit_ms: float) -> pd.DataFrame:
    required = {"day", "month", "mean_wind_ms", "gust_ms"}
    missing = required.difference(wind.columns)
    if missing:
        raise ValueError(f"wind_days.csv is missing columns: {sorted(missing)}")
    daily = (
        wind.groupby(["month", "day"], sort=False)
        .agg(
            mean_wind_ms=("mean_wind_ms", "mean"),
            gust_max_ms=("gust_ms", "max"),
            gust_mean_ms=("gust_ms", "mean"),
            grounded_hours=("gust_ms", lambda values: int((values > gust_limit_ms).sum())),
        )
        .reset_index()
    )
    return daily


def _wind_weeks(wind: pd.DataFrame, gust_limit_ms: float) -> dict[str, dict[str, Any]]:
    daily = _daily_wind(wind, gust_limit_ms)
    if len(daily) < 7:
        raise ValueError("At least seven distinct days are required for rolling wind-week analysis.")

    records: list[dict[str, Any]] = []
    daily_rows = daily.to_dict("records")
    count = len(daily_rows)
    for start in range(count):
        window = [daily_rows[(start + offset) % count] for offset in range(7)]
        window_dates = {(int(row["month"]), int(row["day"])) for row in window}
        records.append(
            {
                "start_month": int(window[0]["month"]),
                "start_day_of_year": int(window[0]["day"]),
                "grounded_hours": int(sum(row["grounded_hours"] for row in window)),
                "mean_wind_ms": float(np.mean([row["mean_wind_ms"] for row in window])),
                "gust_max_ms": float(max(row["gust_max_ms"] for row in window)),
                "wind_days": wind[
                    wind[["month", "day"]].apply(tuple, axis=1).isin(window_dates)
                ].copy(),
            }
        )

    weeks = pd.DataFrame(
        [{key: value for key, value in record.items() if key != "wind_days"} for record in records]
    )
    median_wind = float(weeks["mean_wind_ms"].median())
    median_grounded = float(weeks["grounded_hours"].median())
    wind_scale = max(float(weeks["mean_wind_ms"].std()), 1e-9)
    grounded_scale = max(float(weeks["grounded_hours"].std()), 1.0)
    typical_distance = (
        ((weeks["mean_wind_ms"] - median_wind) / wind_scale) ** 2
        + ((weeks["grounded_hours"] - median_grounded) / grounded_scale) ** 2
    )
    calm_idx = min(
        range(len(records)),
        key=lambda i: (weeks.iloc[i]["grounded_hours"], weeks.iloc[i]["mean_wind_ms"]),
    )
    typical_idx = int(typical_distance.idxmin())
    monsoon_indices = [i for i, record in enumerate(records) if record["start_month"] in (6, 7, 8, 9)]
    if not monsoon_indices:
        monsoon_indices = list(range(len(records)))
    worst_idx = max(
        monsoon_indices,
        key=lambda i: (weeks.iloc[i]["grounded_hours"], weeks.iloc[i]["gust_max_ms"]),
    )

    chosen: dict[str, dict[str, Any]] = {}
    for name, index in zip(WEEK_NAMES, (calm_idx, typical_idx, worst_idx)):
        record = records[index]
        chosen[name] = {
            **{key: value for key, value in record.items() if key != "wind_days"},
            "wind_days": record["wind_days"],
        }
    return chosen


def _result_value(result: Any, keys: tuple[str, ...]) -> float:
    if isinstance(result, pd.Series):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise TypeError("simulate_days must return a mapping with simulation metrics.")
    candidates = [result]
    for nested_key in ("metrics", "summary", "results", "service_metrics"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for mapping in candidates:
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if value is None or pd.isna(value):
                    break
                number = float(value)
                if keys[0] == "S45" and 0 <= number <= 1:
                    return number * 100.0
                return number
    raise KeyError(f"simulate_days result is missing metric; expected one of {keys}.")


def _invoke_simulator(
    simulator: Callable[..., Any],
    candidate: dict[str, Any],
    wind_days: pd.DataFrame,
    data: dict[str, Any],
    seed: int,
    demand_multiplier: float = 1.0,
    drone_availability: float = 1.0,
    scenario: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    signature = inspect.signature(simulator)
    values: dict[str, Any] = {
        "candidate": candidate,
        "candidate_config": candidate,
        "config": candidate,
        "design": candidate,
        "wind_days": wind_days,
        "wind_data": wind_days,
        "wind": wind_days,
        "days": wind_days,
        "costs": data["costs"],
        "cost": data["costs"],
        "data": data,
        "seed": seed,
        "random_seed": seed,
        "demand_multiplier": demand_multiplier,
        "demand_scale": demand_multiplier,
        "drone_availability": drone_availability,
        "availability": drone_availability,
        "availability_probability": drone_availability,
        "scenario": scenario or {},
        "overrides": scenario or {},
    }
    parameters = signature.parameters
    if "network_cfg" in parameters:
        import m5_simulator

        network_cfg = candidate.get("network_cfg")
        if not isinstance(network_cfg, dict):
            network_cfg = m5_simulator.build_network_config(data)
        vehicle_cfg = candidate.get("vehicle_cfg")
        if not isinstance(vehicle_cfg, dict):
            vehicle_cfg = candidate.get("vehicle", candidate)
        policy_cfg = candidate.get("policy_cfg")
        if not isinstance(policy_cfg, dict):
            policy_cfg = m5_simulator._default_policy(network_cfg)
            for key in ("drones", "bays", "spare_batteries", "emergency_reserve_frac", "r_em"):
                if key in candidate:
                    policy_cfg[key] = candidate[key]
        day_indices = sorted(int(day) - 1 for day in wind_days["day"].drop_duplicates())
        values.update(
            {
                "network_cfg": network_cfg,
                "vehicle_cfg": vehicle_cfg,
                "policy_cfg": policy_cfg,
                "day_indices": day_indices,
            }
        )
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    missing: list[str] = []
    for name, parameter in parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in values:
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                positional.append(values[name])
            else:
                kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(
            "Cannot call simulate_days; unsupported required parameter(s): "
            + ", ".join(missing)
        )
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs.setdefault("candidate", candidate)
        kwargs.setdefault("wind_days", wind_days)
        kwargs.setdefault("costs", data["costs"])
        kwargs.setdefault("seed", seed)
        kwargs.setdefault("demand_multiplier", demand_multiplier)
        kwargs.setdefault("drone_availability", drone_availability)
        kwargs.setdefault("scenario", scenario or {})

    result = simulator(*positional, **kwargs)
    metrics = {
        "S45": _result_value(result, ("S45", "s45", "s45_pct", "service_45_pct")),
        "em_p95_min": _result_value(
            result, ("em_p95_min", "emergency_p95_min", "emergency_p95_minutes")
        ),
        "road_fallback_share": _result_value(
            result, ("road_fallback_share", "fallback_share", "road_share")
        ),
    }
    metadata = result if isinstance(result, dict) else {}
    return metrics, metadata


def _candidate_cost(candidate: dict[str, Any], result: dict[str, Any] | None = None) -> float:
    keys = (
        "annualized_system_cost_inr",
        "total_cost_inr",
        "total_capex_inr",
        "total_system_cost_inr",
        "lifecycle_cost_inr",
        "annual_cost_inr",
        "total_cost",
        "system_cost_inr",
        "cost_inr",
        "estimated_capex_inr",
        "drone_capex_inr",
        "capex_inr",
        "cost",
    )
    for mapping in (result or {}, candidate):
        pending = [mapping]
        while pending:
            current = pending.pop(0)
            for key in keys:
                if key in current and current[key] is not None:
                    return float(current[key])
            pending.extend(value for value in current.values() if isinstance(value, dict))
    return math.nan


def _fleet_size(candidate: dict[str, Any]) -> int | None:
    count_keys = ("drone_count", "n_drones", "num_drones", "fleet_size", "number_of_drones")

    def find_count(value: Any) -> int | None:
        if isinstance(value, dict):
            for key in count_keys:
                if value.get(key) is not None:
                    return int(value[key])
            if isinstance(value.get("drones"), list):
                return len(value["drones"])
            for nested in value.values():
                found = find_count(nested)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = find_count(nested)
                if found is not None:
                    return found
        return None

    return find_count(candidate)


def _metric_rows(
    candidates: list[dict[str, Any]],
    weeks: dict[str, dict[str, Any]],
    data: dict[str, Any],
    simulator: Callable[..., Any] | None,
    seeds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if simulator is None:
        for candidate in candidates:
            for week_name, week in weeks.items():
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "week_regime": week_name,
                        "start_month": week["start_month"],
                        "start_day_of_year": week["start_day_of_year"],
                        "grounded_hours": week["grounded_hours"],
                        "mean_wind_ms": week["mean_wind_ms"],
                        "status": "pending_m5_simulator",
                        **{f"{metric}_{stat}": math.nan for metric in METRICS for stat in ("mean", "worst_case")},
                    }
                )
        return pd.DataFrame(rows)

    for candidate in candidates:
        for week_name, week in weeks.items():
            trial_metrics = [
                _invoke_simulator(
                    simulator,
                    candidate,
                    week["wind_days"],
                    data,
                    seed=seed,
                )[0]
                for seed in range(seeds)
            ]
            row: dict[str, Any] = {
                "candidate_id": candidate["candidate_id"],
                "week_regime": week_name,
                "start_month": week["start_month"],
                "start_day_of_year": week["start_day_of_year"],
                "grounded_hours": week["grounded_hours"],
                "mean_wind_ms": week["mean_wind_ms"],
                "status": "simulated",
            }
            for metric in METRICS:
                values = [trial[metric] for trial in trial_metrics]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_worst_case"] = (
                    float(min(values)) if metric == "S45" else float(max(values))
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _stratified_week(wind: pd.DataFrame, trial: int, rng: np.random.Generator) -> pd.DataFrame:
    daily = _daily_wind(wind, float("inf"))
    months = sorted(daily["month"].unique())
    month = months[trial % len(months)]
    month_days = daily.loc[daily["month"] == month, "day"].to_numpy()
    start_day = int(rng.choice(month_days))
    dates = []
    current_month = month
    current_day = start_day
    for _ in range(7):
        dates.append((current_month, current_day))
        idx = daily.index[
            (daily["month"] == current_month) & (daily["day"] == current_day)
        ].tolist()
        current_position = daily.index.get_loc(idx[0])
        next_position = (current_position + 1) % len(daily)
        current_month = int(daily.iloc[next_position]["month"])
        current_day = int(daily.iloc[next_position]["day"])
    mask = [
        (int(row.month), int(row.day)) in dates
        for row in wind[["month", "day"]].itertuples(index=False)
    ]
    return wind.loc[mask].copy()


def _reliability(
    candidates: list[dict[str, Any]],
    wind: pd.DataFrame,
    data: dict[str, Any],
    simulator: Callable[..., Any] | None,
    trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if simulator is None:
        summary = pd.DataFrame(
            [
                {
                    "candidate_id": candidate["candidate_id"],
                    "trials": trials,
                    "probability_S45_ge_90_pct": math.nan,
                    "S45_mean": math.nan,
                    "S45_median": math.nan,
                    "S45_p05": math.nan,
                    "S45_p95": math.nan,
                    "status": "pending_m5_simulator",
                }
                for candidate in candidates
            ]
        )
        return summary, pd.DataFrame(columns=["candidate_id", "trial", "S45"])

    rng = np.random.default_rng(2026)
    trial_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for trial in range(trials):
            sampled_wind = _stratified_week(wind, trial, rng)
            demand_multiplier = float(rng.uniform(0.75, 1.25))
            availability = float(rng.uniform(0.90, 0.98))
            sampled_candidate = copy.deepcopy(candidate)
            fleet_size = _fleet_size(sampled_candidate)
            if fleet_size is not None:
                unavailable = np.flatnonzero(rng.random(fleet_size) > availability).tolist()
                sampled_candidate["unavailable_drone_indices"] = unavailable
                sampled_candidate["available_drone_count"] = fleet_size - len(unavailable)
                sampled_candidate["drone_availability"] = availability
                fleet = sampled_candidate
                while isinstance(fleet, dict) and "drones" not in fleet:
                    nested = next(
                        (
                            value
                            for value in fleet.values()
                            if isinstance(value, dict) and _fleet_size(value) is not None
                        ),
                        None,
                    )
                    if nested is None:
                        break
                    fleet = nested
                if isinstance(fleet, dict) and isinstance(fleet.get("drones"), list):
                    unavailable_set = set(unavailable)
                    for index, drone in enumerate(fleet["drones"]):
                        if isinstance(drone, dict):
                            drone["available"] = index not in unavailable_set
            metrics, _ = _invoke_simulator(
                simulator,
                sampled_candidate,
                sampled_wind,
                data,
                seed=trial,
                demand_multiplier=demand_multiplier,
                drone_availability=availability,
            )
            trial_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "trial": trial + 1,
                    "S45": metrics["S45"],
                    "em_p95_min": metrics["em_p95_min"],
                    "road_fallback_share": metrics["road_fallback_share"],
                    "demand_multiplier": demand_multiplier,
                    "drone_availability": availability,
                }
            )
    trial_frame = pd.DataFrame(trial_rows)
    summaries = []
    for candidate_id, group in trial_frame.groupby("candidate_id", sort=False):
        s45 = group["S45"]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "trials": len(group),
                "probability_S45_ge_90_pct": float((s45 >= 90.0).mean()),
                "S45_mean": float(s45.mean()),
                "S45_median": float(s45.median()),
                "S45_p05": float(s45.quantile(0.05)),
                "S45_p95": float(s45.quantile(0.95)),
                "status": "simulated",
            }
        )
    return pd.DataFrame(summaries), trial_frame


def _sensitivity(
    candidate: dict[str, Any],
    wind: pd.DataFrame,
    data: dict[str, Any],
    simulator: Callable[..., Any] | None,
) -> pd.DataFrame:
    scenarios = [
        ("Battery specific energy +20%", {"battery_specific_energy_Wh_kg": 180.0}, 1),
        ("Drone capex -30%", {"drone_capex_multiplier": 0.70}, 1),
        ("Demand growth +15%/year", {"demand_growth_per_year": 0.15}, 1),
        ("Demand growth +15%/year", {"demand_growth_per_year": 0.15}, 3),
        ("Demand growth +15%/year", {"demand_growth_per_year": 0.15}, 5),
    ]
    if simulator is None:
        return pd.DataFrame(
            [
                {
                    "candidate_id": candidate["candidate_id"],
                    "scenario": label,
                    "year": year,
                    **overrides,
                    "S45": math.nan,
                    "em_p95_min": math.nan,
                    "road_fallback_share": math.nan,
                    "cost_inr": math.nan,
                    "network_choice": "pending M5",
                    "vehicle_choice": "pending M5",
                    "choice_changed": pd.NA,
                    "status": "pending_m5_simulator",
                }
                for label, overrides, year in scenarios
            ]
        )

    rows = []
    baseline_metrics, baseline_result = _invoke_simulator(
        simulator, candidate, wind, data, seed=0
    )
    baseline_network = baseline_result.get("network_choice", candidate.get("network_choice"))
    baseline_vehicle = baseline_result.get("vehicle_choice", candidate.get("vehicle_choice"))
    for label, overrides, year in scenarios:
        scenario_candidate = copy.deepcopy(candidate)
        scenario_candidate.update(overrides)
        if "demand_growth_per_year" in overrides:
            demand_multiplier = 1.0
        else:
            demand_multiplier = 1.0
        scenario_costs = copy.deepcopy(data["costs"])
        if "drone_capex_multiplier" in overrides:
            factor = overrides["drone_capex_multiplier"]
            scenario_costs["drone"]["capex_fixed"] *= factor
            scenario_costs["drone"]["capex_per_kg_mtow"] *= factor
        scenario_data = {**data, "costs": scenario_costs}
        metrics, result = _invoke_simulator(
            simulator,
            scenario_candidate,
            wind,
            scenario_data,
            seed=year,
            demand_multiplier=demand_multiplier,
            scenario={**overrides, "year": year},
        )
        network = result.get("network_choice", scenario_candidate.get("network_choice"))
        vehicle = result.get("vehicle_choice", scenario_candidate.get("vehicle_choice"))
        
        # Actual recomputed annual system cost
        actual_cost = math.nan
        if isinstance(result, dict):
            if "annualized_system_cost_inr" in result and result["annualized_system_cost_inr"] is not None:
                actual_cost = float(result["annualized_system_cost_inr"])
            elif "metrics" in result and isinstance(result["metrics"], dict) and "annualized_system_cost_inr" in result["metrics"]:
                actual_cost = float(result["metrics"]["annualized_system_cost_inr"])
        if math.isnan(actual_cost):
            actual_cost = _candidate_cost(scenario_candidate, result)

        fallback_val = float(metrics.get("road_fallback_share", 0.0))
        s45_val = float(metrics.get("S45", 0.0))
        base_s45_val = float(baseline_metrics.get("S45", 0.0))

        # Check if capacity expansion needed or choice changed
        if fallback_val > 0.10 or (base_s45_val - s45_val) > 5.0:
            changed = "True (Capacity expansion needed)"
        elif network != baseline_network or vehicle != baseline_vehicle:
            changed = "True"
        else:
            changed = "False"

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "scenario": label,
                "year": year,
                **overrides,
                **metrics,
                "cost_inr": actual_cost,
                "network_choice": network,
                "vehicle_choice": vehicle,
                "choice_changed": changed,
                "baseline_S45": baseline_metrics["S45"],
                "status": "simulated",
            }
        )
    return pd.DataFrame(rows)


def _decision_matrix(
    candidates: list[dict[str, Any]],
    wind_results: pd.DataFrame,
) -> pd.DataFrame:
    if wind_results.empty or wind_results["S45_mean"].isna().all():
        return pd.DataFrame(
            [
                {
                    "rule": rule,
                    "candidate_id": candidate["candidate_id"],
                    "rank": pd.NA,
                    "decision_metric": math.nan,
                    "plain_english": "Decision pending M5 simulation results.",
                    "status": "pending_m5_simulator",
                }
                for rule in ("Knee Point", "Minimax Regret", "Best Service within Budget")
                for candidate in candidates
            ]
        )

    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    costs = {
        candidate["candidate_id"]: _candidate_cost(candidate)
        for candidate in candidates
    }
    available_costs = [value for value in costs.values() if math.isfinite(value)]
    budget = max(available_costs) if available_costs else math.nan
    regime_means = wind_results.pivot_table(
        index="candidate_id", columns="week_regime", values="S45_mean", aggfunc="mean"
    )
    service = regime_means.mean(axis=1).reindex(candidate_ids)

    cost_low, cost_high = min(available_costs) if available_costs else math.nan, max(available_costs) if available_costs else math.nan
    service_low, service_high = float(service.min()), float(service.max())
    cost_span = cost_high - cost_low if math.isfinite(cost_high - cost_low) else 0.0
    service_span = service_high - service_low
    normalized_cost = {
        key: (value - cost_low) / cost_span if cost_span > 0 else 0.0
        for key, value in costs.items()
    }
    normalized_service = {
        key: (float(service.get(key, math.nan)) - service_low) / service_span
        if service_span > 0
        else 1.0
        for key in candidate_ids
    }
    knee_score = {
        key: math.hypot(1.0 - normalized_service[key], normalized_cost[key])
        for key in candidate_ids
    }
    knee_rank = {
        key: rank
        for rank, (key, _) in enumerate(sorted(knee_score.items(), key=lambda item: item[1]), start=1)
    }

    regret_scores: dict[str, float] = {}
    for candidate_id in candidate_ids:
        regime_regrets = []
        for regime in regime_means.columns:
            values = regime_means[regime]
            best_s45 = float(values.max())
            s45_range = max(float(values.max() - values.min()), 1e-9)
            s45_regret = (best_s45 - float(values.get(candidate_id, math.nan))) / s45_range
            candidate_cost = costs[candidate_id]
            best_cost = min(available_costs) if available_costs else math.nan
            cost_regret = (
                max(0.0, candidate_cost - best_cost) / max(cost_span, 1e-9)
                if math.isfinite(candidate_cost)
                else 0.0
            )
            regime_regrets.append(s45_regret + cost_regret)
        regret_scores[candidate_id] = max(regime_regrets, default=math.nan)
    regret_rank = {
        key: rank
        for rank, (key, _) in enumerate(sorted(regret_scores.items(), key=lambda item: item[1]), start=1)
    }

    in_budget = [
        candidate_id
        for candidate_id in candidate_ids
        if math.isfinite(costs[candidate_id]) and costs[candidate_id] <= budget
    ]
    best_service = max(in_budget, key=lambda key: float(service[key])) if in_budget else None
    rows = []
    decisions = [
        ("Knee Point", knee_rank, knee_score, min(knee_score, key=knee_score.get)),
        ("Minimax Regret", regret_rank, regret_scores, min(regret_scores, key=regret_scores.get)),
        (
            "Best Service within Budget",
            {key: (1 if key == best_service else 2) for key in candidate_ids},
            {key: float(service.get(key, math.nan)) for key in candidate_ids},
            best_service,
        ),
    ]
    for rule, ranks, scores, winner in decisions:
        for candidate_id in candidate_ids:
            value = scores[candidate_id]
            if rule == "Knee Point":
                metric_text = f"normalized knee distance {value:.3f}"
            elif rule == "Minimax Regret":
                metric_text = f"worst normalized combined regret {value:.3f}"
            else:
                metric_text = f"mean S45 {value:.1f}% at budget INR {budget:,.0f}"
            plain = (
                f"{candidate_id} is ranked {ranks[candidate_id]} under {rule}; "
                f"{metric_text}."
            )
            if winner == candidate_id:
                plain += " Recommended by this rule."
            rows.append(
                {
                    "rule": rule,
                    "candidate_id": candidate_id,
                    "rank": ranks[candidate_id],
                    "decision_metric": value,
                    "budget_inr": budget,
                    "mean_S45_across_wind_regimes": float(service.get(candidate_id, math.nan)),
                    "estimated_cost_inr": costs[candidate_id],
                    "plain_english": plain,
                    "status": "simulated",
                }
            )
    return pd.DataFrame(rows)


def _save_plots(
    wind_results: pd.DataFrame,
    reliability_summary: pd.DataFrame,
    reliability_trials: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    if not wind_results.empty and wind_results["S45_mean"].notna().any():
        for metric, axis in zip(METRICS, axes):
            pivot = wind_results.pivot(index="week_regime", columns="candidate_id", values=f"{metric}_mean")
            pivot.plot(kind="bar", ax=axis)
            axis.set_title(metric.replace("_", " "))
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=25)
    else:
        for axis, title in zip(axes, METRICS):
            axis.set_title(title.replace("_", " "))
            axis.text(0.5, 0.5, "Awaiting M5 simulator", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle("Wind-week robustness: mean over random seeds")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m7_wind_robustness.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if not reliability_trials.empty:
        reliability_trials.boxplot(column="S45", by="candidate_id", ax=axes[0])
        axes[0].set_title("Monte Carlo S45 distribution")
        reliability_summary.set_index("candidate_id")["probability_S45_ge_90_pct"].plot(
            kind="bar", ax=axes[1], ylim=(0, 1)
        )
        axes[1].set_title("P(S45 >= 90%)")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "Awaiting M5 simulator", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle("Monte Carlo reliability")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m7_reliability_mc.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    if not sensitivity.empty and "baseline_S45" in sensitivity:
        sensitivity = sensitivity.copy()
        sensitivity["S45_delta_pp"] = sensitivity["S45"] - sensitivity["baseline_S45"]
        labels = sensitivity["scenario"] + sensitivity["year"].apply(
            lambda year: f" (Year {year})" if year in (3, 5) else ""
        )
        axis.barh(labels, sensitivity["S45_delta_pp"])
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Change in S45 (percentage points vs baseline)")
    else:
        axis.text(0.5, 0.5, "Awaiting M5 simulator", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Sensitivity of recommended candidate")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m7_sensitivity.png", dpi=160)
    plt.close(fig)


def run_m7(
    data: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    simulator: Callable[..., Any] | None = None,
    n_seeds: int = DEFAULT_SEEDS,
    n_mc_trials: int = DEFAULT_MC_TRIALS,
) -> dict[str, pd.DataFrame]:
    """Run M7 and write its requested CSV and PNG outputs."""
    data = load_all_data() if data is None else data
    simulator = simulate_days if simulator is None else simulator
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1.")
    if n_mc_trials < 1:
        raise ValueError("n_mc_trials must be at least 1.")
    placeholder_candidates = False
    if candidates is None:
        candidates, placeholder_candidates = _load_candidates()
    if len(candidates) != 3:
        warnings.warn(f"Expected 3 candidates; received {len(candidates)}.", RuntimeWarning, stacklevel=2)
    if simulator is None:
        warnings.warn(
            "M5 simulator not found; M7 will create pending-status outputs without fabricated metrics.",
            RuntimeWarning,
            stacklevel=2,
        )

    costs = data["costs"]
    gust_limit = float(costs["operations"]["gust_grounding_limit_ms"])
    weeks = _wind_weeks(data["wind_days"], gust_limit)
    wind_results = _metric_rows(candidates, weeks, data, simulator, n_seeds)
    wind_results.to_csv(OUT_DIR / "m7_wind_robustness.csv", index=False)

    reliability_summary, reliability_trials = _reliability(
        candidates, data["wind_days"], data, simulator, n_mc_trials
    )
    reliability_summary.to_csv(OUT_DIR / "m7_reliability_summary.csv", index=False)

    # M6 writes the Top 3 in rank order, so its first candidate is the sensitivity baseline.
    recommended = candidates[0]
    sensitivity = _sensitivity(recommended, data["wind_days"], data, simulator)
    sensitivity.to_csv(OUT_DIR / "m7_sensitivity_table.csv", index=False)

    decision_matrix = _decision_matrix(candidates, wind_results)
    decision_matrix.to_csv(OUT_DIR / "m7_decision_matrix.csv", index=False)
    print("\nM7 Decision Rule Matrix")
    if decision_matrix.empty:
        print("No decision results available.")
    else:
        print(decision_matrix.to_string(index=False))

    _save_plots(
        wind_results,
        reliability_summary,
        reliability_trials,
        sensitivity,
    )
    if placeholder_candidates:
        print("M7 used placeholder candidate configurations; replace them with M6 Top 3 when available.")
    if simulator is None:
        print(f"M7 outputs are marked pending; no simulation metrics were fabricated. See {OUT_DIR}.")
    return {
        "wind_robustness": wind_results,
        "reliability_summary": reliability_summary,
        "reliability_trials": reliability_trials,
        "sensitivity": sensitivity,
        "decision_matrix": decision_matrix,
    }


if __name__ == "__main__":
    run_m7()
