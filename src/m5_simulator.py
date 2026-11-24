"""M5: discrete-event drone logistics simulation and validation.

Run ``python src/m5_simulator.py`` to produce verification outputs and a
representative-day dashboard under ``outputs/m5``.

``day_indices`` in :func:`simulate_days` are zero-based day-of-year indices
(0 = January 1). The public simulator accepts a network, vehicle, and policy
configuration so it can also be used by later project modules.
"""

from __future__ import annotations

import heapq
import json
import math
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon

SRC_DIR = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
OUT_DIR = PROJ_DIR / "outputs" / "m5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from data_loader import load_all_data

EMERGENCY = "emergency"
LAB = "lab"
ROUTINE = "routine"
ORDER_PRIORITY = {EMERGENCY: 0, LAB: 1, ROUTINE: 2}


@dataclass
class Order:
    order_id: int
    phc_id: str
    hub_id: str | None
    kind: str
    arrival_min: float
    distance_km: float
    road_min: float
    road_cost_inr: float
    status: str = "queued"
    delivery_min: float | None = None


@dataclass
class ChargeJob:
    pack_id: int
    remaining_kwh: float
    order_id: int


@dataclass
class HubState:
    hub_id: str
    drone_count: int
    bay_count: int
    spare_count: int
    idle_drones: int
    charged_packs: deque[int]
    queue: list[tuple[int, float, int, Order]] = field(default_factory=list)
    charge_jobs: deque[ChargeJob] = field(default_factory=deque)
    charging_wakeup: simpy.Event | None = None
    dispatch_wakeup: simpy.Event | None = None
    active_flights: int = 0
    active_chargers: int = 0
    max_queue: int = 0
    energy_used_kwh: float = 0.0
    grid_kwh: float = 0.0
    genset_kwh: float = 0.0
    battery_cycles: float = 0.0


def _parse_hhmm(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _is_outage(day_index: int, minute_of_day: float, costs: dict[str, Any]) -> bool:
    outage = costs["grid_outage"]
    weekday = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[day_index % 7]
    start = _parse_hhmm(str(outage["start"]))
    end = _parse_hhmm(str(outage["end"]))
    return weekday in outage["days"] and start <= minute_of_day < end


def _load_distances(data: dict[str, Any]) -> pd.DataFrame:
    path = PROJ_DIR / "outputs" / "m4" / "distance_matrix_km.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)

    warnings.warn(
        "M4 distance matrix not found; calculating straight-line distances as fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    sites = data["sites"].set_index("site_id")
    points = sites[["x_km", "y_km"]].to_numpy(dtype=float)
    matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return pd.DataFrame(matrix, index=sites.index, columns=sites.index)


def _choose_m4_solution() -> dict[str, Any] | None:
    path = PROJ_DIR / "outputs" / "m4" / "m4_solutions.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        solutions = payload.get("solutions", payload.get("candidates", []))
    else:
        solutions = payload
    if not isinstance(solutions, list) or not solutions:
        raise ValueError(f"{path} does not contain a non-empty M4 solution list.")
    valid = [item for item in solutions if item.get("hubs_open")]
    if not valid:
        raise ValueError(f"{path} contains no solution with open hubs.")
    return min(valid, key=lambda item: float(item.get("milp_obj", math.inf)))


def build_network_config(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build assignments and obstacle-avoiding ``L_ij`` distances from M4."""
    data = load_all_data() if data is None else data
    sites = data["sites"].set_index("site_id")
    demand = data["demand"].set_index("phc_id")
    distances = _load_distances(data)
    solution = _choose_m4_solution()
    if solution is None:
        warnings.warn(
            "M4 solutions are unavailable; using the realistic four-hub fallback H1/H2/H4/H6.",
            RuntimeWarning,
            stacklevel=2,
        )
        open_hubs = ["H1", "H2", "H4", "H6"]
        service_radius_km = 24.0
        design_id = "fallback-4-hub"
    else:
        open_hubs = [str(hub) for hub in solution["hubs_open"]]
        service_radius_km = float(solution.get("R_km", 24.0))
        design_id = str(solution.get("design_id", "m4-selected"))

    assignments: dict[str, str | None] = {}
    distances_by_phc: dict[str, dict[str, float]] = {}
    for phc_id in demand.index.astype(str):
        phc_distances = {
            hub_id: float(distances.loc[phc_id, hub_id])
            for hub_id in open_hubs
            if phc_id in distances.index and hub_id in distances.columns
        }
        distances_by_phc[phc_id] = phc_distances
        nearest = min(phc_distances, key=phc_distances.get) if phc_distances else None
        assignments[phc_id] = (
            nearest
            if nearest is not None and phc_distances[nearest] <= service_radius_km
            else None
        )

    return {
        "design_id": design_id,
        "open_hubs": open_hubs,
        "assignments": assignments,
        "L_ij": distances_by_phc,
        "service_radius_km": service_radius_km,
        "hub_locations": {
            hub_id: {
                "x_km": float(sites.loc[hub_id, "x_km"]),
                "y_km": float(sites.loc[hub_id, "y_km"]),
            }
            for hub_id in open_hubs
        },
        "phc_locations": {
            str(phc_id): {
                "x_km": float(row["x_km"]),
                "y_km": float(row["y_km"]),
            }
            for phc_id, row in sites[sites["type"] == "phc"].iterrows()
        },
    }


def _default_vehicle_config(data: dict[str, Any]) -> dict[str, Any]:
    designs_path = PROJ_DIR / "outputs" / "m1" / "m1_designs.json"
    if designs_path.exists():
        with designs_path.open(encoding="utf-8") as file:
            designs = json.load(file)
        if designs:
            chosen = next((item for item in designs if item.get("design_id") == "R24km"), designs[0])
            return dict(chosen)
    return {
        "design_id": "default-vehicle",
        "R_km": 24.0,
        "MTOW_kg": 14.15,
        "V_ms": 18.0,
        "E_Wh": 1200.0,
        "P_hover_loaded_W": 900.0,
        "P_hover_empty_W": 650.0,
        "P_cruise_loaded_W": 800.0,
        "P_cruise_empty_W": 600.0,
        "drone_capex_inr": 836669.0,
    }


def _default_policy(network_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "drones": {hub: 4 for hub in network_cfg["open_hubs"]},
        "bays": {hub: 4 for hub in network_cfg["open_hubs"]},
        "spare_batteries": {hub: 2 for hub in network_cfg["open_hubs"]},
        "emergency_reserve_frac": 0.25,
    }


def _hub_count(policy_cfg: dict[str, Any], key: str, hub_id: str, default: int) -> int:
    value = policy_cfg.get(key, default)
    if isinstance(value, dict):
        value = value.get(hub_id, default)
    result = int(value)
    if result < 0:
        raise ValueError(f"{key}[{hub_id}] must be non-negative.")
    return result


def itemized_annual_cost(
    network_cfg: dict[str, Any],
    vehicle_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    costs: dict[str, Any],
    daily: pd.DataFrame,
) -> dict[str, float]:
    """Itemized annual cost breakdown according to costs.yaml specifications."""
    hubs = network_cfg["open_hubs"]
    hub_count = len(hubs)
    drone_count = sum(_hub_count(policy_cfg, "drones", hub, 0) for hub in hubs)
    bay_count = sum(_hub_count(policy_cfg, "bays", hub, 0) for hub in hubs)
    spare_count = sum(_hub_count(policy_cfg, "spare_batteries", hub, 0) for hub in hubs)

    r = float(costs.get("discount_rate", 0.10))

    def crf(rate: float, n: int) -> float:
        if rate <= 0:
            return 1.0 / n
        return rate * (1.0 + rate) ** n / ((1.0 + rate) ** n - 1.0)

    hub_life = int(costs["hub"]["life_years"])
    drone_life = int(costs["drone"]["life_years"])
    bay_life = int(costs["charging_bay"]["life_years"])

    hub_annual = hub_count * (
        costs["hub"]["capex"] * crf(r, hub_life) + costs["hub"]["opex_per_year"]
    )
    drone_capex = float(
        vehicle_cfg.get(
            "drone_capex_inr",
            costs["drone"]["capex_fixed"]
            + costs["drone"]["capex_per_kg_mtow"] * float(vehicle_cfg.get("MTOW_kg", 0)),
        )
    )
    drone_annual = drone_count * (
        drone_capex * crf(r, drone_life)
        + drone_capex * costs["drone"]["maintenance_frac_of_capex_per_year"]
    )
    bay_annual = bay_count * costs["charging_bay"]["capex"] * crf(r, bay_life)

    staffing_annual = 0.0
    for hub in hubs:
        drones = _hub_count(policy_cfg, "drones", hub, 0)
        operators_per_shift = max(1, math.ceil(drones / costs["staff"]["drones_per_operator"]))
        staffing_annual += (
            operators_per_shift
            * costs["staff"]["shifts_per_day"]
            * costs["staff"]["relief_factor"]
            * costs["staff"]["salary_per_person_year"]
        )

    pack_capex = float(vehicle_cfg.get("E_Wh", 0)) * costs["battery_pack"]["cost_per_Wh"]
    # Spare batteries bought separately; capex annualized over drone life_years (3 years) via CRF
    spare_batteries_annual = spare_count * pack_capex * crf(r, drone_life)

    average_daily_cycles = float(daily["battery_cycles"].mean()) if not daily.empty else 0.0
    battery_cycles_annual = (
        average_daily_cycles
        * 365.0
        / max(1, costs["battery_pack"]["cycle_life"])
        * pack_capex
    )

    grid_kwh_mean = (
        float(daily["grid_kWh"].mean())
        if ("grid_kWh" in daily.columns and not daily.empty)
        else 0.0
    )
    energy_grid_annual = grid_kwh_mean * 365.0 * costs["energy"]["grid_per_kWh"]

    genset_kwh_mean = (
        float(daily["genset_kWh"].mean())
        if ("genset_kWh" in daily.columns and not daily.empty)
        else 0.0
    )
    energy_genset_annual = genset_kwh_mean * 365.0 * costs["energy"]["genset_per_kWh"]

    road_em_mean = (
        float(daily["road_emergency_trips"].mean())
        if ("road_emergency_trips" in daily.columns and not daily.empty)
        else 0.0
    )
    road_emergency_annual = (
        road_em_mean * 365.0 * costs["road_fallback"]["cost_per_trip"]["emergency"]
    )

    road_rl_mean = (
        float(daily["road_routine_lab_trips"].mean())
        if ("road_routine_lab_trips" in daily.columns and not daily.empty)
        else 0.0
    )
    road_routine_lab_annual = (
        road_rl_mean * 365.0 * costs["road_fallback"]["cost_per_trip"]["routine"]
    )

    total = (
        hub_annual
        + staffing_annual
        + drone_annual
        + bay_annual
        + spare_batteries_annual
        + energy_grid_annual
        + energy_genset_annual
        + battery_cycles_annual
        + road_emergency_annual
        + road_routine_lab_annual
    )
    return {
        "hubs": hub_annual,
        "staff": staffing_annual,
        "drones": drone_annual,
        "bays": bay_annual,
        "spare_batteries": spare_batteries_annual,
        "energy_grid": energy_grid_annual,
        "energy_genset": energy_genset_annual,
        "battery_cycles": battery_cycles_annual,
        "road_emergency": road_emergency_annual,
        "road_routine_lab": road_routine_lab_annual,
        "total": total,
    }


def _annualized_system_cost(
    network_cfg: dict[str, Any],
    vehicle_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    costs: dict[str, Any],
    daily: pd.DataFrame,
) -> float:
    return itemized_annual_cost(network_cfg, vehicle_cfg, policy_cfg, costs, daily)["total"]


def simulate_days(
    network_cfg: dict[str, Any],
    vehicle_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    day_indices: Iterable[int],
    seed: int = 42,
    demand_multiplier: float = 1.0,
    drone_availability: float = 1.0,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate one or more zero-based days-of-year using SimPy.

    Returns ``metrics`` (aggregates), ``daily_metrics`` (one row per day), and
    an ``event_trace`` for dashboarding. Wind/day records are reloaded once via
    the shared cleaned data loader for each public call.
    """
    data = load_all_data()
    wind = data["wind_days"]
    costs = data["costs"]
    demand = data["demand"]
    hourly = data["hourly_profile"]
    days = [int(day) for day in day_indices]
    if not days:
        raise ValueError("day_indices must contain at least one zero-based day index.")
    if min(days) < 0 or max(days) >= wind["day"].nunique():
        raise ValueError(f"day_indices must be between 0 and {wind['day'].nunique() - 1}.")
    if not network_cfg.get("open_hubs"):
        raise ValueError("network_cfg.open_hubs must contain at least one hub.")
    for required in ("V_ms", "E_Wh"):
        if float(vehicle_cfg.get(required, 0)) <= 0:
            raise ValueError(f"vehicle_cfg.{required} must be positive.")
    reserve_fraction = float(
        policy_cfg.get("emergency_reserve_frac", policy_cfg.get("r_em", 0.0))
    )
    if not 0 <= reserve_fraction <= 1:
        raise ValueError("emergency_reserve_frac must be in [0, 1].")
    if demand_multiplier <= 0:
        raise ValueError("demand_multiplier must be positive.")
    if not 0 < drone_availability <= 1:
        raise ValueError("drone_availability must be in (0, 1].")
    scenario = {} if scenario is None else dict(scenario)
    vehicle_cfg = dict(vehicle_cfg)
    if "battery_specific_energy_Wh_kg" in scenario:
        base_specific_energy = float(scenario.get("baseline_battery_specific_energy_Wh_kg", 150.0))
        new_specific_energy = float(scenario["battery_specific_energy_Wh_kg"])
        if base_specific_energy <= 0 or new_specific_energy <= 0:
            raise ValueError("Battery specific energy values must be positive.")
        # Reduce battery mass by 1/1.2 (-16.7%)
        battery_mass_reduction = float(vehicle_cfg["E_Wh"]) / base_specific_energy - float(
            vehicle_cfg["E_Wh"]
        ) / new_specific_energy
        old_mtow = float(vehicle_cfg.get("MTOW_kg", 0.0))
        new_mtow = max(0.0, old_mtow - battery_mass_reduction)
        vehicle_cfg["MTOW_kg"] = new_mtow
        # Scale drone capex via mass-growth scaling
        vehicle_cfg["drone_capex_inr"] = (
            costs["drone"]["capex_fixed"] + costs["drone"]["capex_per_kg_mtow"] * new_mtow
        )
        # Scale motor and cruise power with lighter weight
        if old_mtow > 0:
            p_scale = (new_mtow / old_mtow) ** 1.2
            for p_key in ("P_hover_loaded_W", "P_hover_empty_W", "P_cruise_loaded_W", "P_cruise_empty_W"):
                if p_key in vehicle_cfg:
                    vehicle_cfg[p_key] = float(vehicle_cfg[p_key]) * p_scale
    if "drone_capex_multiplier" in scenario:
        mult = float(scenario["drone_capex_multiplier"])
        base_capex = float(vehicle_cfg.get(
            "drone_capex_inr",
            costs["drone"]["capex_fixed"] + costs["drone"]["capex_per_kg_mtow"] * float(vehicle_cfg.get("MTOW_kg", 0.0))
        ))
        vehicle_cfg["drone_capex_inr"] = base_capex * mult
    growth = float(scenario.get("demand_growth_per_year", 0.0))
    year = int(scenario.get("year", 1))
    if growth:
        demand_multiplier *= (1.0 + growth) ** max(0, year - 1)

    wind_by_day = {
        int(day_num) - 1: group.sort_values("hour").reset_index(drop=True)
        for day_num, group in wind.groupby("day", sort=False)
    }
    demand_map = demand.set_index("phc_id").to_dict("index")
    hourly_map = hourly.set_index("hour")
    daily_rows: list[dict[str, Any]] = []
    combined_trace: list[dict[str, Any]] = []
    all_emergency_times: list[float] = []
    all_emergency_success: list[bool] = []
    totals = defaultdict(float)
    order_seq = 0
    operation = costs["operations"]
    road_costs = costs["road_fallback"]["cost_per_trip"]
    road_times = costs["road_fallback"]["delivery_time_min"]
    max_emergency_wait = float(operation["emergency_max_wait_before_road_min"])
    nonemergency_wait = float(policy_cfg.get("nonemergency_max_wait_min", 60.0))

    for day_index in days:
        env = simpy.Environment()
        day_seed = np.random.SeedSequence([int(seed), int(day_index)])
        demand_seed, availability_seed = day_seed.spawn(2)
        rng = np.random.default_rng(demand_seed)
        availability_rng = np.random.default_rng(availability_seed)
        day_wind = wind_by_day[day_index]
        if len(day_wind) != 24:
            raise ValueError(f"Wind day index {day_index} has {len(day_wind)} hourly rows, expected 24.")
        daily_orders: list[Order] = []
        day_trace: list[dict[str, Any]] = []
        day_metrics = {
            "day_index": day_index,
            "day_of_year": day_index + 1,
            "emergency_orders": 0,
            "emergency_within_45": 0,
            "emergency_delivery_times": [],
            "total_orders": 0,
            "road_orders": 0,
            "road_cost_inr": 0.0,
            "road_emergency_trips": 0,
            "road_routine_lab_trips": 0,
            "fallback_gust_grounding_14ms": 0,
            "fallback_headwind_range_limit": 0,
            "fallback_emergency_wait_30min": 0,
            "fallback_day_end_routine_lab": 0,
            "drone_missions": 0,
            "drone_busy_min": 0.0,
            "grid_kwh": 0.0,
            "genset_kwh": 0.0,
            "battery_cycles": 0.0,
        }
        hub_states: dict[str, HubState] = {}
        dispatch_events: dict[str, simpy.Event] = {}
        charging_events: dict[str, simpy.Event] = {}
        for hub_id in network_cfg["open_hubs"]:
            installed_drones = _hub_count(policy_cfg, "drones", hub_id, 0)
            drones = int(availability_rng.binomial(installed_drones, drone_availability))
            bays = _hub_count(policy_cfg, "bays", hub_id, 0)
            spares = _hub_count(policy_cfg, "spare_batteries", hub_id, 0)
            if drones == 0:
                warnings.warn(f"Hub {hub_id} has zero drones; all assigned orders use road fallback.")
            if bays == 0 and spares < drones:
                warnings.warn(
                    f"Hub {hub_id} has no charging bays and fewer than one pack per drone.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            hub_states[hub_id] = HubState(
                hub_id=hub_id,
                drone_count=drones,
                bay_count=bays,
                spare_count=spares,
                idle_drones=drones,
                charged_packs=deque(range(drones + spares)),
                charging_wakeup=env.event(),
                dispatch_wakeup=env.event(),
            )
            dispatch_events[hub_id] = hub_states[hub_id].dispatch_wakeup
            charging_events[hub_id] = hub_states[hub_id].charging_wakeup

        def log(event: str, **fields: Any) -> None:
            record = {"time_min": float(env.now), "day_index": day_index, "event": event, **fields}
            day_trace.append(record)

        def wake_dispatch(hub_id: str) -> None:
            state = hub_states[hub_id]
            if state.dispatch_wakeup is not None and not state.dispatch_wakeup.triggered:
                state.dispatch_wakeup.succeed()
            state.dispatch_wakeup = env.event()
            dispatch_events[hub_id] = state.dispatch_wakeup

        def wake_charging(hub_id: str) -> None:
            state = hub_states[hub_id]
            if state.charging_wakeup is not None and not state.charging_wakeup.triggered:
                state.charging_wakeup.succeed()
            state.charging_wakeup = env.event()
            charging_events[hub_id] = state.charging_wakeup

        def charging_controller(hub_id: str):
            state = hub_states[hub_id]
            while True:
                if not state.charge_jobs:
                    yield state.charging_wakeup
                    continue
                minute = float(env.now % 1440)
                charging_day = day_index + int(env.now // 1440)
                outage = _is_outage(charging_day, minute, costs)
                allowed_bays = min(state.bay_count, 2) if outage else state.bay_count
                if allowed_bays <= 0:
                    yield env.timeout(1)
                    continue
                active = list(state.charge_jobs)[:allowed_bays]
                state.active_chargers = len(active)
                power_kw = float(costs["charging_bay"]["power_kW"])
                efficiency = float(costs["charging_bay"]["charge_efficiency"])
                delivered_per_minute = power_kw * efficiency / 60.0
                input_per_minute = power_kw / 60.0
                yield env.timeout(1)
                now_minute = float(env.now % 1440)
                charging_day = day_index + int((env.now - 0.5) // 1440)
                outage_during_minute = _is_outage(
                    charging_day, (now_minute - 0.5) % 1440, costs
                )
                for job in active:
                    delivered = min(job.remaining_kwh, delivered_per_minute)
                    job.remaining_kwh -= delivered
                    input_kwh = delivered / efficiency
                    if outage_during_minute:
                        state.genset_kwh += input_kwh
                    else:
                        state.grid_kwh += input_kwh
                completed = [job for job in state.charge_jobs if job.remaining_kwh <= 1e-12]
                for job in completed:
                    state.charge_jobs.remove(job)
                    state.charged_packs.append(job.pack_id)
                    log("charge_complete", hub_id=hub_id, pack_id=job.pack_id, order_id=job.order_id)
                    wake_dispatch(hub_id)
                state.active_chargers = 0

        def road_fallback(order: Order, reason: str) -> None:
            if order.status in ("airborne", "delivered", "road"):
                return
            order.status = "road"
            order.delivery_min = max(0.0, float(env.now) - order.arrival_min) + order.road_min
            day_metrics["road_orders"] += 1
            day_metrics["road_cost_inr"] += order.road_cost_inr
            if order.kind == EMERGENCY:
                day_metrics["road_emergency_trips"] += 1
                day_metrics["emergency_delivery_times"].append(order.delivery_min)
                all_emergency_times.append(order.delivery_min)
                all_emergency_success.append(order.delivery_min <= float(operation["emergency_target_min"]))
            else:
                day_metrics["road_routine_lab_trips"] += 1

            if reason == "gust_grounding_14ms":
                day_metrics["fallback_gust_grounding_14ms"] += 1
            elif reason == "headwind_range_limit":
                day_metrics["fallback_headwind_range_limit"] += 1
            elif reason == "emergency_wait_30min":
                day_metrics["fallback_emergency_wait_30min"] += 1
            elif reason == "day_end_routine_lab":
                day_metrics["fallback_day_end_routine_lab"] += 1
            else:
                if "gust" in reason:
                    day_metrics["fallback_gust_grounding_14ms"] += 1
                elif "headwind" in reason or "range" in reason or "singularity" in reason:
                    day_metrics["fallback_headwind_range_limit"] += 1
                elif "wait" in reason or "emergency" in reason:
                    day_metrics["fallback_emergency_wait_30min"] += 1
                else:
                    day_metrics["fallback_day_end_routine_lab"] += 1

            log(
                "road_fallback",
                order_id=order.order_id,
                phc_id=order.phc_id,
                hub_id=order.hub_id,
                kind=order.kind,
                reason=reason,
                delivery_min=order.delivery_min,
            )
            if order.hub_id in hub_states:
                wake_dispatch(order.hub_id)

        def charge_completion(hub_id: str, pack_id: int, energy_kwh: float, order_id: int):
            state = hub_states[hub_id]
            if energy_kwh <= 0:
                state.charged_packs.append(pack_id)
                wake_dispatch(hub_id)
                return
            if state.bay_count <= 0:
                log("battery_unavailable", hub_id=hub_id, pack_id=pack_id, order_id=order_id)
                return
            state.charge_jobs.append(ChargeJob(pack_id, energy_kwh, order_id))
            log(
                "charge_start",
                hub_id=hub_id,
                pack_id=pack_id,
                order_id=order_id,
                active_chargers=state.active_chargers,
                outage=_is_outage(
                    day_index + int(env.now // 1440),
                    float(env.now % 1440),
                    costs,
                ),
            )
            wake_charging(hub_id)

        def order_watchdog(order: Order):
            if order.kind == EMERGENCY:
                yield env.timeout(max_emergency_wait)
                if order.status == "queued":
                    hour = min(int(env.now // 60), 23)
                    row = day_wind.iloc[hour]
                    gust = float(row["gust_ms"])
                    mean_wind = float(row["mean_wind_ms"])
                    v_ms = float(vehicle_cfg["V_ms"])
                    if gust >= float(operation["gust_grounding_limit_ms"]):
                        road_fallback(order, "gust_grounding_14ms")
                    elif mean_wind >= 0.95 * v_ms:
                        road_fallback(order, "headwind_range_limit")
                    else:
                        road_fallback(order, "emergency_wait_30min")
            # Routine and lab orders do not divert after 30 min.
            # They wait until drones/batteries become available or until day-end (minute 1440).

        def order_arrival(order: Order):
            daily_orders.append(order)
            day_metrics["total_orders"] += 1
            if order.kind == EMERGENCY:
                day_metrics["emergency_orders"] += 1
            log(
                "order_arrival",
                order_id=order.order_id,
                phc_id=order.phc_id,
                hub_id=order.hub_id,
                kind=order.kind,
                queue_length=(
                    len(hub_states[order.hub_id].queue)
                    if order.hub_id in hub_states
                    else 0
                ),
            )
            if order.hub_id not in hub_states or order.distance_km > float(vehicle_cfg.get("R_km", 999.0)):
                road_fallback(order, "headwind_range_limit")
                return
            state = hub_states[order.hub_id]
            heapq.heappush(
                state.queue,
                (ORDER_PRIORITY[order.kind], order.arrival_min, order.order_id, order),
            )
            state.max_queue = max(state.max_queue, len(state.queue))
            wake_dispatch(order.hub_id)
            env.process(order_watchdog(order))

        def dispatcher(hub_id: str):
            state = hub_states[hub_id]
            reserve = int(math.ceil(reserve_fraction * state.drone_count))
            while True:
                dispatched_any = False
                while state.queue and state.idle_drones > 0 and state.charged_packs:
                    wind_row = day_wind.iloc[min(int(env.now // 60), 23)]
                    mean_wind = float(wind_row["mean_wind_ms"])
                    gust = float(wind_row["gust_ms"])
                    grounded = (
                        gust >= float(operation["gust_grounding_limit_ms"])
                        or mean_wind >= 0.95 * float(vehicle_cfg["V_ms"])
                    )
                    if grounded:
                        break
                    eligible_index = None
                    for index, (_, _, _, queued_order) in enumerate(state.queue):
                        if queued_order.status != "queued":
                            continue
                        if (
                            queued_order.kind != EMERGENCY
                            and state.idle_drones <= reserve
                        ):
                            continue
                        eligible_index = index
                        break
                    if eligible_index is None:
                        break
                    _, _, _, order = state.queue.pop(eligible_index)
                    heapq.heapify(state.queue)
                    if order.status != "queued":
                        continue
                    order.status = "airborne"
                    state.idle_drones -= 1
                    pack_id = state.charged_packs.popleft()
                    state.active_flights += 1
                    dist_km = max(0.0, float(order.distance_km))
                    speed = float(vehicle_cfg["V_ms"])
                    if dist_km > 0:
                        flight_min = 2 * dist_km * 1000.0 * speed / (
                            speed**2 - mean_wind**2
                        ) / 60.0
                    else:
                        flight_min = 0.0
                    prep_min = float(operation["load_and_takeoff_min"])
                    drop_min = float(operation["drop_or_pickup_min"])
                    turn_min = float(operation["battery_swap_turnaround_min"])
                    mission_min = prep_min + flight_min + drop_min
                    loaded_power = float(
                        vehicle_cfg.get("P_cruise_loaded_W", vehicle_cfg.get("cruise_power_W", 800.0))
                    )
                    empty_power = float(
                        vehicle_cfg.get("P_cruise_empty_W", vehicle_cfg.get("cruise_power_W", 600.0))
                    )
                    loaded_hover = float(vehicle_cfg.get("P_hover_loaded_W", 750.0))
                    empty_hover = float(vehicle_cfg.get("P_hover_empty_W", 500.0))
                    energy_wh = (
                        (loaded_power + empty_power) * flight_min * 60.0 / 2.0 / 3600.0
                        + (loaded_hover + empty_hover) * 60.0 / 3600.0
                    )
                    capacity_wh = float(vehicle_cfg["E_Wh"])
                    if energy_wh > capacity_wh:
                        state.idle_drones += 1
                        state.charged_packs.appendleft(pack_id)
                        state.active_flights -= 1
                        calm_energy_wh = (
                            (loaded_power + empty_power) * (2 * dist_km * 1000.0 / speed) / 2.0 / 3600.0
                            + (loaded_hover + empty_hover) * 60.0 / 3600.0
                        )
                        if calm_energy_wh > capacity_wh or mean_wind >= 0.95 * speed:
                            road_fallback(order, "headwind_range_limit")
                        else:
                            order.status = "queued"
                            heapq.heappush(
                                state.queue,
                                (ORDER_PRIORITY[order.kind], order.arrival_min, order.order_id, order),
                            )
                        break
                        heapq.heappush(
                            state.queue,
                            (ORDER_PRIORITY[order.kind], order.arrival_min, order.order_id, order),
                        )
                        break
                    route_distance = dist_km
                    phc_location = network_cfg.get("phc_locations", {}).get(order.phc_id, {})
                    hub_location = network_cfg.get("hub_locations", {}).get(hub_id, {})
                    log(
                        "drone_depart",
                        order_id=order.order_id,
                        phc_id=order.phc_id,
                        hub_id=hub_id,
                        kind=order.kind,
                        distance_km=route_distance,
                        start_x=float(hub_location.get("x_km", 0.0)),
                        start_y=float(hub_location.get("y_km", 0.0)),
                        end_x=float(phc_location.get("x_km", 0.0)),
                        end_y=float(phc_location.get("y_km", 0.0)),
                        duration_min=mission_min,
                        energy_kwh=energy_wh / 1000.0,
                        gust_ms=gust,
                        mean_wind_ms=mean_wind,
                    )
                    env.process(
                        flight_process(
                            order,
                            hub_id,
                            pack_id,
                            mission_min,
                            flight_min,
                            energy_wh / 1000.0,
                            turn_min,
                        )
                    )
                    dispatched_any = True
                if dispatched_any:
                    yield env.timeout(0)
                    continue
                yield state.dispatch_wakeup

        def flight_process(
            order: Order,
            hub_id: str,
            pack_id: int,
            mission_min: float,
            flight_min: float,
            energy_kwh: float,
            turn_min: float,
        ):
            state = hub_states[hub_id]
            yield env.timeout(mission_min)
            order.status = "delivered"
            order.delivery_min = float(env.now) - order.arrival_min
            if order.kind == EMERGENCY:
                day_metrics["emergency_delivery_times"].append(order.delivery_min)
                all_emergency_times.append(order.delivery_min)
                success = order.delivery_min <= float(operation["emergency_target_min"])
                all_emergency_success.append(success)
                if success:
                    day_metrics["emergency_within_45"] += 1
            day_metrics["drone_missions"] += 1
            day_metrics["drone_busy_min"] += mission_min + turn_min
            usable_capacity_wh = 0.80 * float(vehicle_cfg.get("E_Wh", 1.0))
            frac_cycles = (energy_kwh * 1000.0) / usable_capacity_wh if usable_capacity_wh > 0 else 0.0
            day_metrics["battery_cycles"] += frac_cycles
            state.battery_cycles += frac_cycles
            state.energy_used_kwh += energy_kwh
            state.active_flights -= 1
            log(
                "drone_return",
                order_id=order.order_id,
                phc_id=order.phc_id,
                hub_id=hub_id,
                kind=order.kind,
                delivery_min=order.delivery_min,
                flight_min=flight_min,
                energy_kwh=energy_kwh,
                drone_ready_min=float(env.now) + turn_min,
            )
            env.process(drone_turnaround(hub_id, pack_id, energy_kwh, order.order_id, turn_min))
            wake_dispatch(hub_id)

        def drone_turnaround(hub_id: str, pack_id: int, energy_kwh: float, order_id: int, turn_min: float):
            state = hub_states[hub_id]
            yield env.timeout(turn_min)
            charge_completion(hub_id, pack_id, energy_kwh, order_id)
            state.idle_drones += 1
            log("drone_ready", hub_id=hub_id, order_id=order_id)
            wake_dispatch(hub_id)

        def hourly_weather():
            for hour in range(24):
                row = day_wind.iloc[hour]
                outage = _is_outage(day_index, hour * 60, costs)
                grounded = (
                    float(row["gust_ms"]) >= float(operation["gust_grounding_limit_ms"])
                    or float(row["mean_wind_ms"]) >= 0.95 * float(vehicle_cfg["V_ms"])
                )
                log(
                    "weather",
                    hour=hour,
                    gust_ms=float(row["gust_ms"]),
                    mean_wind_ms=float(row["mean_wind_ms"]),
                    grounded=grounded,
                    outage=outage,
                )
                for hub in network_cfg["open_hubs"]:
                    wake_dispatch(hub)
                    wake_charging(hub)
                yield env.timeout(60)

        def generate_orders():
            nonlocal order_seq
            arrivals: list[tuple[float, str, str | None, float, str]] = []
            for phc_id, demand_row in demand_map.items():
                hub_id = network_cfg.get("assignments", {}).get(phc_id)
                distance = (
                    float(
                        network_cfg.get("L_ij", {})
                        .get(phc_id, {})
                        .get(hub_id, 0.0)
                    )
                    if hub_id is not None
                    else 0.0
                )
                for hour in range(24):
                    profile = hourly_map.loc[hour]
                    for kind in (EMERGENCY, ROUTINE, LAB):
                        rate = (
                            float(demand_row[f"{kind}_per_day"])
                            * float(profile[f"{kind}_frac"])
                            * demand_multiplier
                        )
                        count = int(rng.poisson(rate))
                        arrivals.extend(
                            (
                                hour * 60 + float(rng.uniform(0, 60)),
                                str(phc_id),
                                hub_id,
                                distance,
                                kind,
                            )
                            for _ in range(count)
                        )
            for arrival_min, phc_id, hub_id, distance, kind in sorted(arrivals):
                yield env.timeout(max(0.0, arrival_min - float(env.now)))
                order = Order(
                    order_id=order_seq,
                    phc_id=phc_id,
                    hub_id=hub_id,
                    kind=kind,
                    arrival_min=float(env.now),
                    distance_km=distance,
                    road_min=float(road_times[kind]),
                    road_cost_inr=float(road_costs[kind]),
                )
                order_arrival(order)
                order_seq += 1

        for hub_id in network_cfg["open_hubs"]:
            env.process(dispatcher(hub_id))
            env.process(charging_controller(hub_id))
        env.process(hourly_weather())
        env.process(generate_orders())
        env.run(until=1440)

        # Any outstanding orders are explicitly transferred to road before the day closes.
        for state in hub_states.values():
            for _, _, _, order in state.queue:
                if order.status == "queued":
                    if order.kind == EMERGENCY:
                        hour = min(int(env.now // 60), 23)
                        row = day_wind.iloc[hour]
                        if float(row["gust_ms"]) >= float(operation["gust_grounding_limit_ms"]):
                            road_fallback(order, "gust_grounding_14ms")
                        elif float(row["mean_wind_ms"]) >= 0.95 * float(vehicle_cfg["V_ms"]):
                            road_fallback(order, "headwind_range_limit")
                        else:
                            road_fallback(order, "emergency_wait_30min")
                    else:
                        road_fallback(order, "day_end_routine_lab")
            state.queue.clear()
        for order in daily_orders:
            if order.status == "queued":
                if order.kind == EMERGENCY:
                    hour = min(int(env.now // 60), 23)
                    row = day_wind.iloc[hour]
                    if float(row["gust_ms"]) >= float(operation["gust_grounding_limit_ms"]):
                        road_fallback(order, "gust_grounding_14ms")
                    elif float(row["mean_wind_ms"]) >= 0.95 * float(vehicle_cfg["V_ms"]):
                        road_fallback(order, "headwind_range_limit")
                    else:
                        road_fallback(order, "emergency_wait_30min")
                else:
                    road_fallback(order, "day_end_routine_lab")
        # Complete missions already airborne at midnight so their deliveries
        # are included in this day's service and utilization statistics.
        env.run()

        day_emergencies = day_metrics["emergency_orders"]
        s45 = (
            100.0 * day_metrics["emergency_within_45"] / day_emergencies
            if day_emergencies
            else math.nan
        )
        em_times = day_metrics["emergency_delivery_times"]
        em_p95 = float(np.percentile(em_times, 95)) if em_times else math.nan
        installed_drone_min = sum(state.drone_count for state in hub_states.values()) * 1440.0
        utilization = (
            day_metrics["drone_busy_min"] / installed_drone_min
            if installed_drone_min
            else 0.0
        )
        road_share = (
            day_metrics["road_orders"] / day_metrics["total_orders"]
            if day_metrics["total_orders"]
            else 0.0
        )
        grid_kwh = sum(state.grid_kwh for state in hub_states.values())
        genset_kwh = sum(state.genset_kwh for state in hub_states.values())
        battery_cycles = day_metrics["battery_cycles"]
        row = {
            "day_index": day_index,
            "day_of_year": day_index + 1,
            "S45": s45,
            "em_p95_min": em_p95,
            "drone_utilization": utilization,
            "road_fallback_share": road_share,
            "grid_kWh": grid_kwh,
            "genset_kWh": genset_kwh,
            "battery_cycles": battery_cycles,
            "emergency_orders": day_emergencies,
            "total_orders": day_metrics["total_orders"],
            "drone_missions": day_metrics["drone_missions"],
            "road_orders": day_metrics["road_orders"],
            "road_cost_inr": day_metrics["road_cost_inr"],
            "road_emergency_trips": day_metrics["road_emergency_trips"],
            "road_routine_lab_trips": day_metrics["road_routine_lab_trips"],
            "fallback_gust_grounding_14ms": day_metrics["fallback_gust_grounding_14ms"],
            "fallback_headwind_range_limit": day_metrics["fallback_headwind_range_limit"],
            "fallback_emergency_wait_30min": day_metrics["fallback_emergency_wait_30min"],
            "fallback_day_end_routine_lab": day_metrics["fallback_day_end_routine_lab"],
            "max_hub_queue": max((state.max_queue for state in hub_states.values()), default=0),
        }
        daily_rows.append(row)
        for key in (
            "grid_kWh",
            "genset_kWh",
            "battery_cycles",
            "emergency_orders",
            "total_orders",
            "drone_missions",
            "road_orders",
            "road_cost_inr",
            "road_emergency_trips",
            "road_routine_lab_trips",
            "fallback_gust_grounding_14ms",
            "fallback_headwind_range_limit",
            "fallback_emergency_wait_30min",
            "fallback_day_end_routine_lab",
        ):
            totals[key] += float(row[key])
        combined_trace.extend(day_trace)

    daily_frame = pd.DataFrame(daily_rows)
    metrics = {
        "S45": (
            100.0 * sum(all_emergency_success) / len(all_emergency_success)
            if all_emergency_success
            else math.nan
        ),
        "em_p95_min": float(np.percentile(all_emergency_times, 95)) if all_emergency_times else math.nan,
        "drone_utilization": float(daily_frame["drone_utilization"].mean()),
        "road_fallback_share": (
            totals["road_orders"] / totals["total_orders"] if totals["total_orders"] else 0.0
        ),
        "grid_kWh": float(totals["grid_kWh"]),
        "genset_kWh": float(totals["genset_kWh"]),
        "battery_cycles": float(totals["battery_cycles"]),
        "annualized_system_cost_inr": _annualized_system_cost(
            network_cfg, vehicle_cfg, policy_cfg, costs, daily_frame
        ),
        "itemized_annual_cost_inr": itemized_annual_cost(
            network_cfg, vehicle_cfg, policy_cfg, costs, daily_frame
        ),
        "days_simulated": len(days),
        "total_orders": int(totals["total_orders"]),
        "emergency_orders": int(totals["emergency_orders"]),
        "drone_missions": int(totals["drone_missions"]),
        "road_orders": int(totals["road_orders"]),
        "road_emergency_trips": int(totals["road_emergency_trips"]),
        "road_routine_lab_trips": int(totals["road_routine_lab_trips"]),
        "fallback_gust_grounding_14ms": int(totals["fallback_gust_grounding_14ms"]),
        "fallback_headwind_range_limit": int(totals["fallback_headwind_range_limit"]),
        "fallback_emergency_wait_30min": int(totals["fallback_emergency_wait_30min"]),
        "fallback_day_end_routine_lab": int(totals["fallback_day_end_routine_lab"]),
    }
    return {
        **metrics,
        "metrics": metrics,
        "daily_metrics": daily_frame,
        "event_trace": pd.DataFrame(combined_trace),
        "recommended_N_days": len(days),
    }


def _stratified_day_indices(data: dict[str, Any], n_days: int, seed: int) -> list[int]:
    day_month = (
        data["wind_days"][["day", "month"]]
        .drop_duplicates()
        .sort_values("day")
        .reset_index(drop=True)
    )
    if n_days > len(day_month):
        raise ValueError(f"n_days cannot exceed {len(day_month)} wind days.")
    rng = np.random.default_rng(seed)
    month_groups = {
        int(month): group["day"].to_numpy(dtype=int) - 1
        for month, group in day_month.groupby("month")
    }
    raw_allocation = {
        month: n_days * len(population) / len(day_month)
        for month, population in month_groups.items()
    }
    allocation = {month: int(value) for month, value in raw_allocation.items()}
    remainder = n_days - sum(allocation.values())
    remainder_months = sorted(
        month_groups,
        key=lambda month: raw_allocation[month] - allocation[month],
        reverse=True,
    )
    for month in remainder_months[:remainder]:
        allocation[month] += 1
    chosen: list[int] = []
    for month, population in month_groups.items():
        size = min(allocation[month], len(population))
        if size:
            chosen.extend(rng.choice(population, size=size, replace=False).tolist())
    return sorted(int(index) for index in chosen)


def erlang_c_probability(c: int, rho: float) -> float:
    """Exact Erlang-C probability of waiting for an M/M/c queue."""
    if c < 1 or not 0 < rho < 1:
        raise ValueError("c must be positive and rho must be in (0, 1).")
    offered_load = c * rho
    terms = [offered_load**n / math.factorial(n) for n in range(c)]
    last = offered_load**c / (math.factorial(c) * (1.0 - rho))
    return last / (sum(terms) + last)


def erlang_c_mean_wait_min(c: int, rho: float, mean_service_min: float) -> float:
    """Exact M/M/c mean queue wait from Erlang C."""
    if mean_service_min <= 0:
        raise ValueError("mean_service_min must be positive.")
    mu_per_min = 1.0 / mean_service_min
    lam_per_min = rho * c * mu_per_min
    probability_wait = erlang_c_probability(c, rho)
    return probability_wait / (c * mu_per_min - lam_per_min)


def _simulate_mm_c_wait(
    c: int,
    rho: float,
    mean_service_min: float,
    seed: int,
    arrivals: int = 60000,
    warmup: int = 6000,
) -> float:
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=c)
    rng = np.random.default_rng(seed)
    waits: list[float] = []
    lam = rho * c / mean_service_min

    def customer():
        arrival_time = float(env.now)
        with resource.request() as request:
            yield request
            wait = float(env.now) - arrival_time
            waits.append(wait)
            yield env.timeout(float(rng.exponential(mean_service_min)))

    def arrivals_process():
        for index in range(arrivals):
            if index:
                yield env.timeout(float(rng.exponential(1.0 / lam)))
            if index >= warmup:
                env.process(customer())
            else:
                env.process(customer())

    env.process(arrivals_process())
    env.run()
    # The initial warmup customers are discarded to reduce transient bias.
    steady_waits = waits[warmup:]
    if not steady_waits:
        raise RuntimeError("Erlang-C simulation produced no steady-state observations.")
    return float(np.mean(steady_waits))


def run_erlang_c_verification(seed: int = 42) -> pd.DataFrame:
    rows = []
    service_mean = 10.0
    for index, rho in enumerate((0.5, 0.7, 0.85)):
        simulated = _simulate_mm_c_wait(3, rho, service_mean, seed + index)
        analytical = erlang_c_mean_wait_min(3, rho, service_mean)
        relative_error = abs(simulated - analytical) / analytical
        rows.append(
            {
                "servers": 3,
                "rho": rho,
                "lambda_per_min": rho * 3 / service_mean,
                "mu_per_min": 1 / service_mean,
                "simulated_mean_wait_min": simulated,
                "erlang_c_mean_wait_min": analytical,
                "absolute_error_min": abs(simulated - analytical),
                "relative_error_pct": 100 * relative_error,
                "passed_10pct_tolerance": relative_error < 0.10,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "erlang_c_verification.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(frame["rho"], frame["simulated_mean_wait_min"], "o-", label="SimPy simulated")
    axis.plot(frame["rho"], frame["erlang_c_mean_wait_min"], "s--", label="Erlang-C exact")
    axis.set(
        xlabel="Traffic intensity ρ = λ/(cμ)",
        ylabel="Mean waiting time (minutes)",
        title="M/M/3 analytical verification",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "erlang_c_verification.png", dpi=160)
    plt.close(fig)
    return frame


def run_ci_convergence(
    network_cfg: dict[str, Any],
    vehicle_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int = 42,
) -> tuple[pd.DataFrame, int | None]:
    data = load_all_data()
    rows = []
    selected_n: int | None = None
    sample_sizes = [10, 20, 40, 60, 100]
    for n_days in sample_sizes:
        indices = _stratified_day_indices(data, n_days, seed)
        result = simulate_days(network_cfg, vehicle_cfg, policy_cfg, indices, seed=seed + n_days)
        values = result["daily_metrics"]["S45"].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            half_width = math.nan
            mean_s45 = float(np.mean(values)) if len(values) else math.nan
            sd_s45 = math.nan
        else:
            mean_s45 = float(np.mean(values))
            sd_s45 = float(np.std(values, ddof=1))
            t_critical = _t_critical_95(len(values) - 1)
            half_width = t_critical * sd_s45 / math.sqrt(len(values))
        rows.append(
            {
                "N_days": n_days,
                "S45_mean_pct": mean_s45,
                "S45_sd_pct": sd_s45,
                "ci95_half_width_pp": half_width,
                "meets_3pp_target": bool(half_width < 3.0),
                "status": "passes" if half_width < 3 else "does_not_meet_target",
            }
        )
        if selected_n is None and half_width < 3:
            selected_n = n_days

    # If the largest mandatory sample size is insufficient, continue sampling
    # (up to a full-year census) rather than claiming the ±3 pp target was met.
    total_days = len(data["wind_days"]["day"].unique())
    n_days = 120
    while selected_n is None and n_days < total_days:
        indices = _stratified_day_indices(data, n_days, seed)
        result = simulate_days(network_cfg, vehicle_cfg, policy_cfg, indices, seed=seed + n_days)
        values = result["daily_metrics"]["S45"].dropna().to_numpy(dtype=float)
        mean_s45 = float(np.mean(values)) if len(values) else math.nan
        sd_s45 = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        half_width = (
            _t_critical_95(len(values) - 1) * sd_s45 / math.sqrt(len(values))
            if len(values) > 1
            else math.nan
        )
        rows.append(
            {
                "N_days": n_days,
                "S45_mean_pct": mean_s45,
                "S45_sd_pct": sd_s45,
                "ci95_half_width_pp": half_width,
                "meets_3pp_target": bool(half_width < 3.0),
                "status": "passes" if half_width < 3 else "does_not_meet_target",
            }
        )
        if half_width < 3:
            selected_n = n_days
        n_days += 20
    if selected_n is None and n_days != total_days:
        indices = _stratified_day_indices(data, total_days, seed)
        result = simulate_days(network_cfg, vehicle_cfg, policy_cfg, indices, seed=seed + total_days)
        values = result["daily_metrics"]["S45"].dropna().to_numpy(dtype=float)
        mean_s45 = float(np.mean(values)) if len(values) else math.nan
        sd_s45 = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        half_width = (
            _t_critical_95(len(values) - 1) * sd_s45 / math.sqrt(len(values))
            if len(values) > 1
            else math.nan
        )
        rows.append(
            {
                "N_days": total_days,
                "S45_mean_pct": mean_s45,
                "S45_sd_pct": sd_s45,
                "ci95_half_width_pp": half_width,
                "meets_3pp_target": bool(half_width < 3.0),
                "status": "passes" if half_width < 3 else "does_not_meet_target",
            }
        )
        if half_width < 3:
            selected_n = total_days

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "ci_convergence.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(frame["N_days"], frame["ci95_half_width_pp"], "o-", color="#187b78")
    axis.axhline(3.0, color="#bd3d3a", linestyle="--", label="±3 percentage-point target")
    if selected_n is not None:
        axis.axvline(selected_n, color="#555555", linestyle=":", label=f"Chosen N = {selected_n}")
    axis.set(
        xlabel="Stratified simulated days",
        ylabel="95% CI half-width for S45 (percentage points)",
        title="S45 sample-size convergence",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ci_convergence.png", dpi=160)
    plt.close(fig)
    return frame, selected_n


def _t_critical_95(degrees_freedom: int) -> float:
    if degrees_freedom < 1:
        raise ValueError("A confidence interval needs at least two observations.")
    known = {9: 2.262, 19: 2.093, 39: 2.023, 59: 2.001, 99: 1.984}
    if degrees_freedom in known:
        return known[degrees_freedom]
    z = 1.959963984540054
    df = float(degrees_freedom)
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def _draw_dashboard(
    data: dict[str, Any],
    network_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    day_result: dict[str, Any],
    day_index: int,
) -> None:
    trace = day_result["event_trace"]
    if trace.empty:
        raise RuntimeError("Representative-day simulation produced no event trace.")
    sites = data["sites"]
    route_events = trace[trace["event"] == "drone_depart"].copy()
    weather = trace[trace["event"] == "weather"].copy()
    x_bounds = (float(sites["x_km"].min()) - 3, float(sites["x_km"].max()) + 3)
    y_bounds = (float(sites["y_km"].min()) - 3, float(sites["y_km"].max()) + 3)

    def base_figure():
        fig, axis = plt.subplots(figsize=(9, 7))
        for polygon in data["nofly_polygons"]:
            points = np.asarray(polygon.exterior.coords)
            axis.add_patch(
                MplPolygon(
                    points,
                    closed=True,
                    facecolor="#e66b5b",
                    alpha=0.22,
                    edgecolor="#a52a2a",
                    linewidth=0.8,
                )
            )
        hubs = sites[sites["site_id"].isin(network_cfg["open_hubs"])]
        phcs = sites[sites["type"] == "phc"]
        axis.scatter(phcs["x_km"], phcs["y_km"], c="#4f81bd", s=24, label="PHC")
        axis.scatter(hubs["x_km"], hubs["y_km"], marker="s", c="#183a37", s=95, label="Open hub")
        for _, row in phcs.iterrows():
            axis.annotate(row["site_id"], (row["x_km"], row["y_km"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        axis.set(xlim=x_bounds, ylim=y_bounds, xlabel="District X (km)", ylabel="District Y (km)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper left")
        return fig, axis

    fig, axis = base_figure()
    snapshot_min = 12 * 60
    _draw_dashboard_frame(axis, data, network_cfg, policy_cfg, trace, weather, route_events, snapshot_min, day_index)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "live_day_snapshot.png", dpi=170)
    plt.close(fig)

    fig, axis = base_figure()
    frame_minutes = np.linspace(0, 24 * 60 - 1, 49)

    def animate(frame_index: int):
        axis.clear()
        # Rebuild stable map layers as axis.clear() removes patches and annotations.
        for polygon in data["nofly_polygons"]:
            points = np.asarray(polygon.exterior.coords)
            axis.add_patch(
                MplPolygon(points, closed=True, facecolor="#e66b5b", alpha=0.22, edgecolor="#a52a2a", linewidth=0.8)
            )
        hubs = sites[sites["site_id"].isin(network_cfg["open_hubs"])]
        phcs = sites[sites["type"] == "phc"]
        axis.scatter(phcs["x_km"], phcs["y_km"], c="#4f81bd", s=24, label="PHC")
        axis.scatter(hubs["x_km"], hubs["y_km"], marker="s", c="#183a37", s=95, label="Open hub")
        for _, row in phcs.iterrows():
            axis.annotate(row["site_id"], (row["x_km"], row["y_km"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        axis.set(xlim=x_bounds, ylim=y_bounds, xlabel="District X (km)", ylabel="District Y (km)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper left")
        _draw_dashboard_frame(
            axis,
            data,
            network_cfg,
            policy_cfg,
            trace,
            weather,
            route_events,
            float(frame_minutes[frame_index]),
            day_index,
        )
        return axis.collections

    animation = FuncAnimation(fig, animate, frames=len(frame_minutes), interval=240, blit=False)
    animation.save(OUT_DIR / "live_day_dashboard.gif", writer=PillowWriter(fps=4))
    plt.close(fig)


def _draw_dashboard_frame(
    axis: Any,
    data: dict[str, Any],
    network_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    trace: pd.DataFrame,
    weather: pd.DataFrame,
    route_events: pd.DataFrame,
    minute: float,
    day_index: int,
) -> None:
    sites = data["sites"].set_index("site_id")
    latest_weather = weather[weather["time_min"] <= minute].tail(1)
    if latest_weather.empty:
        gust, mean_wind, grounded, outage = 0.0, 0.0, False, False
    else:
        state = latest_weather.iloc[0]
        gust, mean_wind = float(state["gust_ms"]), float(state["mean_wind_ms"])
        grounded, outage = bool(state["grounded"]), bool(state["outage"])

    # Draw flights active at the selected dashboard time.
    active_count = 0
    for _, flight in route_events.iterrows():
        start = float(flight["time_min"])
        end = start + float(flight["duration_min"])
        if start <= minute <= end:
            fraction = (minute - start) / max(end - start, 1e-9)
            x = float(flight["start_x"]) + fraction * (float(flight["end_x"]) - float(flight["start_x"]))
            y = float(flight["start_y"]) + fraction * (float(flight["end_y"]) - float(flight["start_y"]))
            axis.scatter([x], [y], marker="^", c="#f28e2b", s=80, zorder=5)
            active_count += 1

    recent = trace[trace["time_min"] <= minute]
    queue_lengths: dict[str, int] = defaultdict(int)
    for _, row in recent.iterrows():
        if pd.notna(row.get("hub_id")) and row["event"] == "order_arrival":
            queue_lengths[str(row["hub_id"])] += 1
        elif pd.notna(row.get("hub_id")) and row["event"] in ("drone_depart", "road_fallback"):
            queue_lengths[str(row["hub_id"])] = max(0, queue_lengths[str(row["hub_id"])] - 1)
    charging = trace[
        (trace["event"] == "charge_start") & (trace["time_min"] <= minute)
    ].groupby("hub_id").size().to_dict()
    finished = trace[
        (trace["event"] == "charge_complete") & (trace["time_min"] <= minute)
    ].groupby("hub_id").size().to_dict()

    for hub_id, location in network_cfg["hub_locations"].items():
        queue = queue_lengths[hub_id]
        bays = min(
            _hub_count(policy_cfg, "bays", hub_id, 0),
            int(charging.get(hub_id, 0) - finished.get(hub_id, 0)),
        )
        axis.annotate(
            f"Q:{queue}  charge:{max(0, bays)}",
            (float(location["x_km"]), float(location["y_km"])),
            fontsize=7,
            color="#183a37",
            xytext=(6, 8),
            textcoords="offset points",
        )

    time_label = f"{int(minute // 60):02d}:{int(minute % 60):02d}"
    banners = []
    if outage:
        banners.append("GRID OUTAGE")
    if grounded:
        banners.append("GUST / HEADWIND GROUNDING")
    banner = " | ".join(banners) if banners else "Operations normal"
    axis.set_title(
        f"Live operations — day {day_index + 1}, {time_label}  |  {banner}\n"
        f"Wind {mean_wind:.1f} m/s · gust {gust:.1f} m/s · active drones {active_count}"
    )
    axis.text(
        0.99,
        0.02,
        "Red polygons: no-fly zones  |  Orange triangles: drones",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def run_m5() -> dict[str, Any]:
    """Run analytical verification, convergence study, and live-day dashboard."""
    data = load_all_data()
    network_cfg = build_network_config(data)
    vehicle_cfg = _default_vehicle_config(data)
    policy_cfg = _default_policy(network_cfg)

    erlang = run_erlang_c_verification()
    if not erlang["passed_10pct_tolerance"].all():
        raise AssertionError("M/M/c simulation did not match Erlang-C within 10%.")
    print("M/M/3 Erlang-C verification: all three traffic levels passed 10% relative error.")

    convergence, selected_n = run_ci_convergence(network_cfg, vehicle_cfg, policy_cfg)
    if selected_n is None:
        convergence.to_csv(OUT_DIR / "ci_convergence.csv", index=False)
        raise AssertionError("A full-year census did not reach the ±3 percentage-point CI target.")
    print(f"S45 convergence: N_days={selected_n} is the smallest tested size below ±3 percentage points.")
    chosen_indices = _stratified_day_indices(data, selected_n, 42)
    run_result = simulate_days(network_cfg, vehicle_cfg, policy_cfg, chosen_indices, seed=2026)
    run_result["daily_metrics"].to_csv(OUT_DIR / "daily_metrics.csv", index=False)
    summary_dict = dict(run_result["metrics"])
    # Convert nested dict for CSV output
    if "itemized_annual_cost_inr" in summary_dict:
        for k, v in summary_dict["itemized_annual_cost_inr"].items():
            summary_dict[f"cost_{k}_inr"] = v
        del summary_dict["itemized_annual_cost_inr"]
    summary = pd.DataFrame([summary_dict])
    summary["network_design_id"] = network_cfg["design_id"]
    summary["vehicle_design_id"] = vehicle_cfg.get("design_id", "unspecified")
    summary["recommended_N_days"] = selected_n
    summary.to_csv(OUT_DIR / "simulation_summary.csv", index=False)

    # Print Itemized Annual Cost Breakdown
    itemized = run_result["metrics"]["itemized_annual_cost_inr"]
    print("\n" + "=" * 70)
    print("M5 ANNUAL COST BREAKDOWN AUDIT (Consistent with costs.yaml)")
    print("=" * 70)
    print(f"  1. Hubs Capex + Opex:            INR {itemized['hubs']:>12,.2f}")
    print(f"  2. Staffing (shifts + relief):   INR {itemized['staff']:>12,.2f}")
    print(f"  3. Drone Fleet (Capex + Maint):  INR {itemized['drones']:>12,.2f}  (1 pack included, not double-charged)")
    print(f"  4. Charging Bays:                INR {itemized['bays']:>12,.2f}")
    print(f"  5. Spare Battery Packs:          INR {itemized['spare_batteries']:>12,.2f}")
    print(f"  6. Grid Energy:                  INR {itemized['energy_grid']:>12,.2f}")
    print(f"  7. Genset Energy (outages):      INR {itemized['energy_genset']:>12,.2f}")
    print(f"  8. Battery Cycle Degradation:    INR {itemized['battery_cycles']:>12,.2f}")
    print(f"  9. Road Fallback (Emergency):    INR {itemized['road_emergency']:>12,.2f}")
    print(f" 10. Road Fallback (Routine/Lab):  INR {itemized['road_routine_lab']:>12,.2f}")
    print("  " + "-" * 66)
    print(f"  TOTAL ANNUAL COST:               INR {itemized['total']:>12,.2f}")
    print("=" * 70)

    # Print Road Fallback Audit Breakdown
    m = run_result["metrics"]
    total_ord = m["total_orders"]
    road_ord = m["road_orders"]
    fb_gust = m["fallback_gust_grounding_14ms"]
    fb_hw = m["fallback_headwind_range_limit"]
    fb_wait = m["fallback_emergency_wait_30min"]
    fb_dayend = m["fallback_day_end_routine_lab"]

    print("\n" + "=" * 70)
    print("M5 ROAD FALLBACK ROOT-CAUSE AUDIT")
    print("=" * 70)
    print(f"Total Orders Simulated:        {total_ord:,}")
    print(f"Total Road Fallback Orders:    {road_ord:,} ({m['road_fallback_share']*100:.2f}%)")
    print(f"Emergency Orders Served <=45m: {m['S45']:.2f}% (Target: >=90%)")
    print(f"Emergency P95 Delivery Time:   {m['em_p95_min']:.2f} min")
    print("\nFallback Root-Cause Decomposition:")
    print(f"  - Gust Grounding (>=14 m/s):   {fb_gust:>5} orders ({fb_gust/max(1, road_ord)*100:>5.1f}% of fallback, {fb_gust/max(1, total_ord)*100:>5.2f}% of all)")
    print(f"  - Headwind / Range Limit:      {fb_hw:>5} orders ({fb_hw/max(1, road_ord)*100:>5.1f}% of fallback, {fb_hw/max(1, total_ord)*100:>5.2f}% of all)")
    print(f"  - Emergency Wait >30m (Busy):  {fb_wait:>5} orders ({fb_wait/max(1, road_ord)*100:>5.1f}% of fallback, {fb_wait/max(1, total_ord)*100:>5.2f}% of all)")
    print(f"  - Day-End Routine/Lab Unserved:{fb_dayend:>5} orders ({fb_dayend/max(1, road_ord)*100:>5.1f}% of fallback, {fb_dayend/max(1, total_ord)*100:>5.2f}% of all)")
    print("=" * 70 + "\n")

    representative_day = _stratified_day_indices(data, 1, 15)[0]
    representative = simulate_days(
        network_cfg,
        vehicle_cfg,
        policy_cfg,
        [representative_day],
        seed=15,
    )
    representative["event_trace"].to_csv(OUT_DIR / "live_day_event_trace.csv", index=False)
    _draw_dashboard(data, network_cfg, policy_cfg, representative, representative_day)
    print(f"Live-day dashboard generated for wind day {representative_day + 1}.")
    print(f"M5 results saved under {OUT_DIR}.")
    return {
        "network_cfg": network_cfg,
        "vehicle_cfg": vehicle_cfg,
        "policy_cfg": policy_cfg,
        "erlang_c": erlang,
        "ci_convergence": convergence,
        "selected_n_days": selected_n,
        "summary": summary,
        "representative": representative,
    }


if __name__ == "__main__":
    run_m5()
