"""M6: surrogate-assisted multi-objective optimization using common random numbers.

Run ``python src/m6_sim_opt.py`` to generate 55 Latin-hypercube M5 evaluations,
Gaussian-process surrogates, an NSGA-II Pareto front, direct rechecks, and
weighted-sum verification under ``outputs/m6``.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut, cross_val_predict

SRC_DIR = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
OUT_DIR = PROJ_DIR / "outputs" / "m6"
M1_PATH = PROJ_DIR / "outputs" / "m1" / "m1_designs.json"
M4_PATH = PROJ_DIR / "outputs" / "m4" / "m4_solutions.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from data_loader import load_all_data

try:
    import m5_simulator
    from m5_simulator import simulate_days
except ModuleNotFoundError as exc:
    if exc.name != "m5_simulator":
        raise
    m5_simulator = None
    simulate_days = None

USE_STUB = False
N_LHS_SAMPLES = 55
N_RECHECKS = 15
N_COMMON_DAYS = 20  # Stratified 16–24 CRN days for GP training sweep
N_RECHECK_DAYS = 120  # Full 120 stratified days for final Pareto rechecks & Top 3 export
COMMON_SEED = 6206
OBJECTIVES = ("annual_cost_inr", "S45", "road_fallback_share", "em_p95_min")
FEATURE_NAMES = (
    "vehicle_idx",
    "drone_delta",
    "bay_delta",
    "spares_tenths",
    "reserve_pct",
)


def _read_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file is required but missing: {path}")
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = payload.get("solutions", payload.get("designs", payload.get("candidates", [])))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty list of {label}.")
    return [dict(row) for row in payload]


def _load_upstream() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vehicles = _read_json_list(M1_PATH, "M1 vehicle designs")
    m4_solutions = _read_json_list(M4_PATH, "M4 solutions")
    if not any(solution.get("hubs_open") for solution in m4_solutions):
        raise ValueError(f"{M4_PATH} contains no solution with hubs_open.")
    if not all(float(vehicle.get("V_ms", 0)) > 0 for vehicle in vehicles):
        raise ValueError("Every M1 vehicle design must include positive V_ms.")
    return vehicles, m4_solutions


def _default_policy(
    network_cfg: dict[str, Any],
    decision: dict[str, int],
) -> dict[str, Any]:
    drones_per_hub = 4 + decision["drone_delta"]
    bays_per_hub = 4 + decision["bay_delta"]
    spare_packs_per_hub = int(math.ceil(drones_per_hub * decision["spares_tenths"] / 10.0))
    return {
        "drones": {hub: drones_per_hub for hub in network_cfg["open_hubs"]},
        "bays": {hub: bays_per_hub for hub in network_cfg["open_hubs"]},
        "spare_batteries": {hub: spare_packs_per_hub for hub in network_cfg["open_hubs"]},
        "emergency_reserve_frac": decision["reserve_pct"] / 100.0,
    }


def _stub_network(data: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    sites = data["sites"].set_index("site_id")
    open_hubs = [str(hub) for hub in solution.get("hubs_open", ["H1", "H2", "H4", "H6"])]
    phcs = sites[sites["type"] == "phc"]
    hub_locations = {
        hub: {"x_km": float(sites.loc[hub, "x_km"]), "y_km": float(sites.loc[hub, "y_km"])}
        for hub in open_hubs
    }
    phc_locations = {
        phc: {"x_km": float(row["x_km"]), "y_km": float(row["y_km"])}
        for phc, row in phcs.iterrows()
    }
    assignments = {}
    distances = {}
    radius = float(solution.get("R_km", 24.0))
    for phc, location in phc_locations.items():
        from math import hypot

        by_hub = {
            hub: hypot(location["x_km"] - coords["x_km"], location["y_km"] - coords["y_km"])
            for hub, coords in hub_locations.items()
        }
        distances[phc] = by_hub
        nearest = min(by_hub, key=by_hub.get)
        assignments[phc] = nearest if by_hub[nearest] <= radius else None
    return {
        "design_id": str(solution.get("design_id", "stub-m4")),
        "open_hubs": open_hubs,
        "assignments": assignments,
        "L_ij": distances,
        "hub_locations": hub_locations,
        "phc_locations": phc_locations,
        "service_radius_km": radius,
    }


def _decision_from_vector(vector: np.ndarray | list[float]) -> dict[str, int]:
    values = np.asarray(vector, dtype=float)
    return {
        "vehicle_idx": int(np.clip(np.rint(values[0]), 0, 5)),
        "drone_delta": int(np.clip(np.rint(values[1]), 0, 4)),
        "bay_delta": int(np.clip(np.rint(values[2]), 0, 3)),
        "spares_tenths": int(np.clip(np.rint(values[3]), 5, 20)),
        "reserve_pct": int(np.clip(np.rint(values[4]), 0, 40)),
    }


def _decision_features(decisions: list[dict[str, int]]) -> np.ndarray:
    # One-hot encoding avoids imposing an artificial ordinal relationship among vehicles.
    features = np.zeros((len(decisions), 10), dtype=float)
    for row_index, decision in enumerate(decisions):
        features[row_index, decision["vehicle_idx"]] = 1.0
        features[row_index, 6] = decision["drone_delta"] / 4.0
        features[row_index, 7] = decision["bay_delta"] / 3.0
        features[row_index, 8] = (decision["spares_tenths"] - 5) / 15.0
        features[row_index, 9] = decision["reserve_pct"] / 40.0
    return features


def _vector_features(vector: np.ndarray | list[float]) -> np.ndarray:
    return _decision_features([_decision_from_vector(vector)])


def _policy_evaluation(
    decision: dict[str, int],
    vehicles: list[dict[str, Any]],
    network_cfg: dict[str, Any],
    common_days: list[int],
    seed: int,
    simulator_available: bool,
    data: dict[str, Any],
) -> dict[str, Any]:
    vehicle = dict(vehicles[decision["vehicle_idx"]])
    policy = _default_policy(network_cfg, decision)
    if simulator_available and not USE_STUB:
        result = simulate_days(
            network_cfg=network_cfg,
            vehicle_cfg=vehicle,
            policy_cfg=policy,
            day_indices=common_days,
            seed=seed,
        )
        return {
            "annual_cost_inr": float(result["annualized_system_cost_inr"]),
            "S45": float(result["S45"]),
            "road_fallback_share": float(result["road_fallback_share"]),
            "em_p95_min": float(result["em_p95_min"]),
            "source": "m5",
        }
    return _analytical_stub(decision, vehicle, network_cfg, policy, data)


def _analytical_stub(
    decision: dict[str, int],
    vehicle: dict[str, Any],
    network_cfg: dict[str, Any],
    policy: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Fast transparent surrogate when M5/M4 upstream artifacts are unavailable."""
    hubs = len(network_cfg["open_hubs"])
    drone_count = sum(policy["drones"].values())
    bay_count = sum(policy["bays"].values())
    assigned_distances = [
        distances[assignment]
        for assignment, distances in zip(
            network_cfg.get("assignments", {}).values(),
            network_cfg.get("L_ij", {}).values(),
        )
        if assignment is not None and assignment in distances
    ]
    mean_route = float(np.mean(assigned_distances)) if assigned_distances else 12.0
    vehicle_range = float(vehicle.get("R_km", 18.0))
    speed = float(vehicle.get("V_ms", 15.0))
    energy = float(vehicle.get("E_Wh", 600.0))
    capacity_factor = min(1.0, max(0.35, vehicle_range / max(mean_route * 2.0, 1.0)))
    reliability = (
        90.5
        + 1.55 * decision["drone_delta"]
        + 0.62 * decision["bay_delta"]
        + 0.30 * (decision["spares_tenths"] - 5) / 5
        + 0.17 * decision["reserve_pct"]
        + 0.18 * (speed - 15.0)
        + 0.004 * (energy - 600.0)
        - 3.2 * max(0.0, mean_route - vehicle_range * 0.42)
    )
    s45 = float(np.clip(reliability * capacity_factor, 25.0, 99.5))
    fallback = float(
        np.clip(
            0.28
            - 0.028 * decision["drone_delta"]
            - 0.010 * decision["bay_delta"]
            - 0.006 * (decision["spares_tenths"] - 5) / 5
            - (s45 - 90.0) * 0.003,
            0.01,
            0.65,
        )
    )
    p95 = float(max(35.0, 82.0 - 0.55 * (s45 - 75.0) + max(0.0, mean_route - 10.0) * 1.2))
    costs = data["costs"]
    drone_capex = float(
        vehicle.get(
            "drone_capex_inr",
            costs["drone"]["capex_fixed"]
            + costs["drone"]["capex_per_kg_mtow"] * float(vehicle.get("MTOW_kg", 0.0)),
        )
    )
    r = float(costs.get("discount_rate", 0.10))

    def crf(rate: float, n: int) -> float:
        if rate <= 0:
            return 1.0 / n
        return rate * (1.0 + rate) ** n / ((1.0 + rate) ** n - 1.0)

    hub_annual = hubs * (
        costs["hub"]["capex"] * crf(r, int(costs["hub"]["life_years"])) + costs["hub"]["opex_per_year"]
    )
    drone_annual = drone_count * (
        drone_capex * crf(r, int(costs["drone"]["life_years"]))
        + drone_capex * costs["drone"]["maintenance_frac_of_capex_per_year"]
    )
    bay_annual = bay_count * costs["charging_bay"]["capex"] * crf(r, int(costs["charging_bay"]["life_years"]))
    staff_annual = sum(
        max(1, math.ceil(count / costs["staff"]["drones_per_operator"]))
        * costs["staff"]["shifts_per_day"]
        * costs["staff"]["relief_factor"]
        * costs["staff"]["salary_per_person_year"]
        for count in policy["drones"].values()
    )
    annual_cost = (
        hub_annual
        + drone_annual
        + bay_annual
        + staff_annual
        + float(data["demand"]["emergency_per_day"].sum()) * 365 * fallback * 1250
    )
    return {
        "annual_cost_inr": float(annual_cost),
        "S45": s45,
        "road_fallback_share": fallback,
        "em_p95_min": p95,
        "source": "analytical_stub",
    }


def _lhs_decisions(n_samples: int = N_LHS_SAMPLES, seed: int = 2026) -> list[dict[str, int]]:
    sampler = qmc.LatinHypercube(d=len(FEATURE_NAMES), seed=seed, optimization="random-cd")
    unit_samples = sampler.random(n=n_samples)
    lower = np.array([0, 0, 0, 5, 0], dtype=float)
    upper = np.array([5, 4, 3, 20, 40], dtype=float)
    samples = qmc.scale(unit_samples, lower, upper)
    decisions: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for sample in samples:
        decision = _decision_from_vector(sample)
        key = tuple(decision[name] for name in FEATURE_NAMES)
        if key in seen:
            continue
        decisions.append(decision)
        seen.add(key)
    # Quantization can collide; deterministic random fill preserves the exact 55 count.
    rng = np.random.default_rng(seed + 1)
    while len(decisions) < n_samples:
        decision = {
            "vehicle_idx": int(rng.integers(0, 6)),
            "drone_delta": int(rng.integers(0, 5)),
            "bay_delta": int(rng.integers(0, 4)),
            "spares_tenths": int(rng.integers(5, 21)),
            "reserve_pct": int(rng.integers(0, 41)),
        }
        key = tuple(decision[name] for name in FEATURE_NAMES)
        if key not in seen:
            decisions.append(decision)
            seen.add(key)
    return decisions


def _fit_surrogates(
    decisions: list[dict[str, int]],
    observations: pd.DataFrame,
) -> tuple[dict[str, GaussianProcessRegressor], pd.DataFrame]:
    x = _decision_features(decisions)
    models: dict[str, GaussianProcessRegressor] = {}
    loo_rows = []
    for objective in OBJECTIVES:
        y = observations[objective].to_numpy(dtype=float)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=2026,
        )
        model.fit(x, y)
        models[objective] = model
        predicted = cross_val_predict(
            GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                optimizer=None,
                random_state=2026,
            ),
            x,
            y,
            cv=LeaveOneOut(),
            n_jobs=1,
        )
        residual = predicted - y
        loo_rows.append(
            {
                "objective": objective,
                "loocv_mae": float(np.mean(np.abs(residual))),
                "loocv_rmse": float(np.sqrt(np.mean(residual**2))),
                "loocv_r2": float(1 - np.sum(residual**2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
                "fitted_kernel": str(model.kernel_),
            }
        )
    return models, pd.DataFrame(loo_rows)


def _predict_objectives(models: dict[str, GaussianProcessRegressor], vector: np.ndarray) -> np.ndarray:
    feature = _vector_features(vector)
    cost = float(models["annual_cost_inr"].predict(feature)[0])
    s45 = float(models["S45"].predict(feature)[0])
    road = float(models["road_fallback_share"].predict(feature)[0])
    p95 = float(models["em_p95_min"].predict(feature)[0])
    return np.array([cost, -s45, road, p95], dtype=float)


def _select_representatives(
    front: pd.DataFrame,
    count: int = N_RECHECKS,
    baseline_decision: dict[str, int] | None = None,
) -> pd.DataFrame:
    if front.empty:
        raise RuntimeError("NSGA-II returned an empty Pareto front.")
    objectives = front[["annual_cost_inr", "S45", "road_fallback_share"]].to_numpy(dtype=float)
    minima = objectives.min(axis=0)
    spans = np.maximum(objectives.max(axis=0) - minima, 1e-12)
    normalized = (objectives - minima) / spans
    utopia_distance = np.linalg.norm(normalized - np.array([0.0, 1.0, 0.0]), axis=1)
    chosen: list[int] = [
        int(np.argmin(objectives[:, 0])),
        int(np.argmax(objectives[:, 1])),
        int(np.argmin(objectives[:, 2])),
        int(np.argmin(utopia_distance)),
    ]
    if baseline_decision is not None:
        base_tuple = tuple(int(baseline_decision[name]) for name in FEATURE_NAMES)
        for idx, (_, row) in enumerate(front.iterrows()):
            if tuple(int(row[name]) for name in FEATURE_NAMES) == base_tuple:
                if idx not in chosen:
                    chosen.append(idx)
                break
    extremes = {
        int(np.argmin(objectives[:, 0])),
        int(np.argmax(objectives[:, 1])),
        int(np.argmin(objectives[:, 2])),
    }
    while len(set(chosen)) < min(count, len(front)):
        selected = sorted(set(chosen))
        distances = cdist(normalized, normalized[selected]).min(axis=1)
        for index in selected:
            distances[index] = -1
        chosen.append(int(np.argmax(distances)))
    chosen = list(dict.fromkeys(chosen))[:count]
    if len(chosen) < count:
        # A low-dimensional stub can have fewer than 15 nondominated points.
        # Repeat its extreme/knee representatives to exercise the full pipeline.
        chosen.extend(chosen[index % len(chosen)] for index in range(count - len(chosen)))
    return front.iloc[chosen].copy().reset_index(drop=True)


def _run_two_objective_nsga(
    models: dict[str, GaussianProcessRegressor],
    vehicle_max: int,
) -> pd.DataFrame:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    class CostServiceSlice(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=5,
                n_obj=2,
                n_ieq_constr=0,
                xl=np.array([0, 0, 0, 5, 0]),
                xu=np.array([vehicle_max, 4, 3, 20, 40]),
                vtype=int,
            )

        def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            prediction = _predict_objectives(models, x)
            out["F"] = prediction[:2]

    result = minimize(
        CostServiceSlice(),
        NSGA2(
            pop_size=256,
            n_offsprings=192,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=0.2, eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        ),
        get_termination("n_gen", 140),
        seed=6202,
        verbose=False,
    )
    if result.X is None:
        raise RuntimeError("Two-objective NSGA-II returned no cost-service points.")
    rows = []
    seen = set()
    for vector in np.atleast_2d(result.X):
        decision = _decision_from_vector(vector)
        key = tuple(decision[name] for name in FEATURE_NAMES)
        if key in seen:
            continue
        seen.add(key)
        prediction = _predict_objectives(
            models, np.array([decision[name] for name in FEATURE_NAMES])
        )
        rows.append(
            {
                **decision,
                "annual_cost_inr": float(prediction[0]),
                "S45": float(-prediction[1]),
            }
        )
    return pd.DataFrame(rows)


def _run_weighted_sum_checks(
    models: dict[str, GaussianProcessRegressor],
    front: pd.DataFrame,
    two_objective_front: pd.DataFrame,
) -> pd.DataFrame:
    cost_scale = max(float(front["annual_cost_inr"].max() - front["annual_cost_inr"].min()), 1.0)
    s45_scale = max(float(front["S45"].max() - front["S45"].min()), 1.0)
    service_min = float(front["S45"].min())
    rows = []
    weights = np.linspace(0.0, 1.0, 7)
    for index, weight in enumerate(weights):
        def objective(vector: np.ndarray) -> float:
            decision = _decision_from_vector(vector)
            predicted = _predict_objectives(models, np.array([decision[name] for name in FEATURE_NAMES]))
            cost, neg_s45 = predicted[:2]
            normalized_cost = (cost - float(front["annual_cost_inr"].min())) / cost_scale
            normalized_service = (-neg_s45 - service_min) / s45_scale
            return float(weight * normalized_cost - (1.0 - weight) * normalized_service)

        result = differential_evolution(
            objective,
            [(0, 5), (0, 4), (0, 3), (5, 20), (0, 40)],
            seed=700 + index,
            maxiter=60,
            popsize=12,
            polish=False,
            updating="immediate",
            workers=1,
            integrality=[True, True, True, True, True],
        )
        decision = _decision_from_vector(result.x)
        prediction = _predict_objectives(
            models, np.array([decision[name] for name in FEATURE_NAMES])
        )
        cost, s45 = float(prediction[0]), float(-prediction[1])
        cost_tolerance = max(abs(cost) * 0.001, 10.0)
        service_tolerance_pp = 0.05
        dominating = (
            (two_objective_front["annual_cost_inr"].to_numpy() <= cost + cost_tolerance)
            & (two_objective_front["S45"].to_numpy() >= s45 - service_tolerance_pp)
        )
        behind_front = bool(dominating.any())
        rows.append(
            {
                "weight_cost": float(weight),
                "weight_service": float(1 - weight),
                **decision,
                "predicted_cost_inr": cost,
                "predicted_S45": s45,
                "weighted_objective": float(result.fun),
                "on_or_behind_nsga2_front": behind_front,
                "front_comparison": "two-objective NSGA-II cost-service slice",
                "front_cost_tolerance_inr": cost_tolerance,
                "front_service_tolerance_pp": service_tolerance_pp,
            }
        )
    return pd.DataFrame(rows)


def _create_plots(
    front: pd.DataFrame,
    two_objective_front: pd.DataFrame,
    weighted: pd.DataFrame,
    rechecks: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(14, 5.5))
    axis3d = fig.add_subplot(121, projection="3d")
    scatter = axis3d.scatter(
        front["annual_cost_inr"] / 1e6,
        front["S45"],
        front["road_fallback_share"],
        c=front["em_p95_min"],
        cmap="viridis_r",
        s=24,
    )
    axis3d.set(
        xlabel="Annual cost (INR millions)",
        ylabel="S45 (%)",
        zlabel="Road fallback share",
        title="NSGA-II 3-objective Pareto front",
    )
    fig.colorbar(scatter, ax=axis3d, shrink=0.65, label="Emergency p95 (min)")
    axis2d = fig.add_subplot(122)
    axis2d.scatter(
        two_objective_front["annual_cost_inr"] / 1e6,
        two_objective_front["S45"],
        c="#2b7a78",
        s=25,
        label="NSGA-II cost-service slice",
    )
    if not weighted.empty:
        axis2d.scatter(
            weighted["predicted_cost_inr"] / 1e6,
            weighted["predicted_S45"],
            marker="x",
            c="#d1495b",
            s=80,
            label="Weighted-sum checks",
        )
    if not rechecks.empty:
        axis2d.scatter(
            rechecks["true_annual_cost_inr"] / 1e6,
            rechecks["true_S45"],
            marker="D",
            facecolors="none",
            edgecolors="#173f5f",
            label="Direct M5 rechecks",
        )
    axis2d.set(
        xlabel="Annualized cost (INR millions)",
        ylabel="S45 (%)",
        title="Cost–service Pareto slice",
    )
    axis2d.grid(alpha=0.25)
    axis2d.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m6_pareto_front.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(front["annual_cost_inr"] / 1e6, front["em_p95_min"], c="#4c78a8", alpha=0.75, label="NSGA-II front")
    if not rechecks.empty:
        axis.scatter(
            rechecks["true_annual_cost_inr"] / 1e6,
            rechecks["true_em_p95_min"],
            marker="D",
            facecolors="none",
            edgecolors="#e45756",
            label="Direct M5 rechecks",
        )
    axis.set(
        xlabel="Annualized cost (INR millions)",
        ylabel="Emergency p95 delivery time (minutes)",
        title="Emergency p95 behavior vs annualized cost",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m6_p95_behavior.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        two_objective_front["annual_cost_inr"] / 1e6,
        two_objective_front["S45"],
        s=25,
        label="NSGA-II Pareto slice",
    )
    axis.scatter(
        weighted["predicted_cost_inr"] / 1e6,
        weighted["predicted_S45"],
        marker="x",
        s=85,
        color="#d1495b",
        label="7 weighted-sum solutions",
    )
    axis.set(xlabel="Annualized cost (INR millions)", ylabel="S45 (%)", title="Weighted-sum verification")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m6_weighted_sum_verification.png", dpi=170)
    plt.close(fig)


def _export_top3(
    representatives: pd.DataFrame,
    vehicles: list[dict[str, Any]],
    network_cfg: dict[str, Any],
    m4_solutions: list[dict[str, Any]],
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    m4_solution_by_design = {item.get("design_id"): item for item in m4_solutions}
    for design in m4_solutions:
        if design.get("design_id") == network_cfg.get("design_id"):
            m4_solution_by_design.setdefault("selected", design)
    subset = representatives.copy()
    features = list(FEATURE_NAMES)

    # 1. Low-Cost: lowest-cost feasible point
    low_idx = subset["true_annual_cost_inr"].idxmin()
    low_row = subset.loc[low_idx]
    low_cfg = tuple(int(low_row[f]) for f in features)

    # 2. High-Service: highest-S45 point (tie-break by lower road fallback share, then higher cost)
    high_candidates = subset[subset["true_S45"] == subset["true_S45"].max()]
    high_idx = high_candidates.sort_values(
        by=["true_road_fallback_share", "true_annual_cost_inr"], ascending=[True, False]
    ).index[0]
    high_row = subset.loc[high_idx]
    high_cfg = tuple(int(high_row[f]) for f in features)

    # 3. Knee/Balanced: among remaining distinct non-dominated points strictly between Low-Cost and High-Service
    # in both cost and S45, pick minimum normalized utopia-distance point
    between = subset[
        (subset["true_annual_cost_inr"] > low_row["true_annual_cost_inr"])
        & (subset["true_annual_cost_inr"] < high_row["true_annual_cost_inr"])
        & (subset["true_S45"] > low_row["true_S45"])
        & (subset["true_S45"] < high_row["true_S45"])
    ].copy()

    # Filter out configurations identical to low or high
    between["_cfg"] = between[features].apply(lambda r: tuple(int(x) for x in r), axis=1)
    between = between[~between["_cfg"].isin([low_cfg, high_cfg])]

    # Fallback to any distinct configuration if strictly-between is empty
    if between.empty:
        distinct = subset.copy()
        distinct["_cfg"] = distinct[features].apply(lambda r: tuple(int(x) for x in r), axis=1)
        between = distinct[~distinct["_cfg"].isin([low_cfg, high_cfg])]

    c_min, c_max = subset["true_annual_cost_inr"].min(), subset["true_annual_cost_inr"].max()
    s_min, s_max = subset["true_S45"].min(), subset["true_S45"].max()
    f_min, f_max = subset["true_road_fallback_share"].min(), subset["true_road_fallback_share"].max()

    c_norm = (between["true_annual_cost_inr"] - c_min) / max(c_max - c_min, 1e-9)
    s_norm = (between["true_S45"] - s_min) / max(s_max - s_min, 1e-9)
    f_norm = (between["true_road_fallback_share"] - f_min) / max(f_max - f_min, 1e-9)
    dists = np.sqrt(c_norm**2 + (1.0 - s_norm) ** 2 + f_norm**2)
    knee_idx = dists.idxmin()

    selections = (("Low-Cost", low_idx), ("Knee/Balanced", knee_idx), ("High-Service", high_idx))
    exported = []
    seen_ids = set()
    for label, index in selections:
        row = subset.loc[index]
        decision = {key: int(row[key]) for key in FEATURE_NAMES}
        vehicle = dict(vehicles[decision["vehicle_idx"]])
        policy = _default_policy(network_cfg, decision)
        cid = f"M6-{label.replace('/', '-')}-{vehicle.get('design_id', 'vehicle')}"
        if cid in seen_ids:
            cid = f"{cid}-v{decision['vehicle_idx']}d{decision['drone_delta']}s{decision['spares_tenths']}"
        seen_ids.add(cid)
        candidate = {
            "candidate_id": cid,
            "decision_label": label,
            "is_placeholder": bool(USE_STUB or simulate_days is None),
            "network_design_id": network_cfg["design_id"],
            "network_choice": network_cfg["design_id"],
            "vehicle_choice": vehicle.get("design_id"),
            "vehicle_cfg": vehicle,
            "network_cfg": network_cfg,
            "policy_cfg": policy,
            "decision_variables": decision,
            "annual_cost_inr": float(row["true_annual_cost_inr"]),
            "S45": float(row["true_S45"]),
            "road_fallback_share": float(row["true_road_fallback_share"]),
            "em_p95_min": float(row["true_em_p95_min"]),
            "n_days_evaluated": N_RECHECK_DAYS,
            "m4_solution": m4_solution_by_design.get(network_cfg.get("design_id"), {}),
        }
        exported.append(candidate)
    path = OUT_DIR / "m6_top3_candidates.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(exported, file, indent=2)
    return exported


def run_m6(use_cached_rechecks: bool = False) -> dict[str, Any]:
    global USE_STUB
    if "--cached" in sys.argv or "--use-cached" in sys.argv:
        use_cached_rechecks = True

    data = load_all_data()
    upstream_files_ready = M1_PATH.exists() and M4_PATH.exists() and m5_simulator is not None
    USE_STUB = not upstream_files_ready
    if USE_STUB:
        warnings.warn(
            "M1/M4/M5 upstream components are incomplete; USE_STUB=True enables the "
            "analytical surrogate pipeline. It will switch to M5 automatically when ready.",
            RuntimeWarning,
            stacklevel=2,
        )
        vehicles = _read_json_list(M1_PATH, "M1 vehicle designs") if M1_PATH.exists() else [
            {"design_id": f"stub-{i}", "R_km": 14 + 2 * i, "MTOW_kg": 9 + i, "V_ms": 14 + i, "E_Wh": 500 + i * 120}
            for i in range(6)
        ]
        m4_solutions = _read_json_list(M4_PATH, "M4 solutions") if M4_PATH.exists() else [
            {"design_id": "stub-m4", "hubs_open": ["H1", "H2", "H4", "H6"], "R_km": 24}
        ]
        network_cfg = (
            m5_simulator.build_network_config(data)
            if m5_simulator is not None
            else _stub_network(data, m4_solutions[0])
        )
    else:
        vehicles, m4_solutions = _load_upstream()
        network_cfg = m5_simulator.build_network_config(data)

    cached_rechecks_path = OUT_DIR / "m6_surrogate_recheck.csv"
    if use_cached_rechecks and cached_rechecks_path.exists():
        print(f"M6: Using cached rechecks from {cached_rechecks_path} for fast Top-3 selection.")
        rechecks = pd.read_csv(cached_rechecks_path)
        pareto = pd.read_csv(OUT_DIR / "m6_surrogate_pareto.csv") if (OUT_DIR / "m6_surrogate_pareto.csv").exists() else rechecks
        two_objective_front = pd.read_csv(OUT_DIR / "m6_cost_service_nsga2.csv") if (OUT_DIR / "m6_cost_service_nsga2.csv").exists() else pareto
        weighted = pd.read_csv(OUT_DIR / "m6_weighted_sum_results.csv") if (OUT_DIR / "m6_weighted_sum_results.csv").exists() else pd.DataFrame()
        top3 = _export_top3(rechecks, vehicles, network_cfg, m4_solutions, data)
        _create_plots(pareto, two_objective_front, weighted, rechecks)
        print(f"Top 3 candidates successfully exported to {OUT_DIR / 'm6_top3_candidates.json'}")
        return {
            "pareto": pareto,
            "rechecks": rechecks,
            "weighted_sum": weighted,
            "top3": top3,
            "use_stub": USE_STUB,
        }

    if len(vehicles) != 6:
        warnings.warn(
            f"M1 contains {len(vehicles)} designs; decision bounds will use the available designs.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Keep decision conversion aligned with the actual number of vehicle configurations.
    vehicle_max = len(vehicles) - 1

    # M5 baseline configuration: R24km, drone_delta=0, bay_delta=0, spares_tenths=5 (2 spares/hub), reserve_pct=25 (0.25)
    baseline_vehicle_idx = next(
        (i for i, v in enumerate(vehicles) if v.get("design_id") == "R24km" or abs(v.get("R_km", 0) - 24.0) < 0.1),
        min(4, vehicle_max),
    )
    baseline_decision = {
        "vehicle_idx": baseline_vehicle_idx,
        "drone_delta": 0,
        "bay_delta": 0,
        "spares_tenths": 5,
        "reserve_pct": 25,
    }

    common_days = (
        m5_simulator._stratified_day_indices(data, N_COMMON_DAYS, COMMON_SEED)
        if m5_simulator is not None
        else list(range(N_COMMON_DAYS))
    )
    # Exact same 120-day stratified day set as M5 (selected_n=120, seed=42)
    recheck_days = (
        m5_simulator._stratified_day_indices(data, N_RECHECK_DAYS, 42)
        if m5_simulator is not None
        else list(range(N_RECHECK_DAYS))
    )
    sample_decisions = _lhs_decisions()
    sample_decisions = [
        {**decision, "vehicle_idx": min(decision["vehicle_idx"], vehicle_max)}
        for decision in sample_decisions
    ]
    # Ensure baseline configuration is evaluated in LHS training sweep
    baseline_tup = tuple(baseline_decision[name] for name in FEATURE_NAMES)
    if baseline_tup not in {tuple(item[name] for name in FEATURE_NAMES) for item in sample_decisions}:
        sample_decisions[0] = dict(baseline_decision)
    sample_decisions = list(
        {
            tuple(decision[name] for name in FEATURE_NAMES): decision
            for decision in sample_decisions
        }.values()
    )
    rng_fill = np.random.default_rng(2027)
    while len(sample_decisions) < N_LHS_SAMPLES:
        decision = {
            "vehicle_idx": int(rng_fill.integers(0, len(vehicles))),
            "drone_delta": int(rng_fill.integers(0, 5)),
            "bay_delta": int(rng_fill.integers(0, 4)),
            "spares_tenths": int(rng_fill.integers(5, 21)),
            "reserve_pct": int(rng_fill.integers(0, 41)),
        }
        if tuple(decision[name] for name in FEATURE_NAMES) not in {
            tuple(item[name] for name in FEATURE_NAMES) for item in sample_decisions
        }:
            sample_decisions.append(decision)

    observation_rows = []
    for evaluation_id, decision in enumerate(sample_decisions, start=1):
        metrics = _policy_evaluation(
            decision,
            vehicles,
            network_cfg,
            common_days,
            COMMON_SEED,
            simulator_available=simulate_days is not None,
            data=data,
        )
        observation_rows.append(
            {
                "evaluation_id": evaluation_id,
                **decision,
                **metrics,
                "common_day_indices": json.dumps(common_days),
                "common_seed": COMMON_SEED,
            }
        )
        if evaluation_id % 10 == 0:
            print(f"M6 evaluated {evaluation_id}/{N_LHS_SAMPLES} common-random-number samples.")
    observations = pd.DataFrame(observation_rows)
    observations.to_csv(OUT_DIR / "m6_lhs_evaluations.csv", index=False)
    if len(observations) != N_LHS_SAMPLES:
        raise AssertionError(f"Expected {N_LHS_SAMPLES} LHS evaluations, got {len(observations)}.")

    models, loocv = _fit_surrogates(sample_decisions, observations)
    loocv["training_samples"] = len(observations)
    loocv.to_csv(OUT_DIR / "m6_surrogate_loocv.csv", index=False)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    class SurrogateParetoProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=5,
                n_obj=3,
                n_ieq_constr=0,
                xl=np.array([0, 0, 0, 5, 0]),
                xu=np.array([vehicle_max, 4, 3, 20, 40]),
                vtype=int,
            )

        def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            prediction = _predict_objectives(models, x)
            out["F"] = prediction[:3]

    algorithm = NSGA2(
        pop_size=128,
        n_offsprings=96,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.2, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    result = minimize(
        SurrogateParetoProblem(),
        algorithm,
        get_termination("n_gen", 90),
        seed=2026,
        verbose=False,
    )
    if result.X is None or result.F is None:
        raise RuntimeError("NSGA-II failed to return Pareto solutions.")
    front_decisions = [_decision_from_vector(vector) for vector in np.atleast_2d(result.X)]
    # Explicitly inject the M5 baseline configuration into the candidate pool
    front_decisions.append(baseline_decision)

    unique: dict[tuple[int, ...], dict[str, int]] = {}
    for decision in front_decisions:
        unique[tuple(decision[name] for name in FEATURE_NAMES)] = decision

    candidate_decisions = list(unique.values())
    F_list = []
    for decision in candidate_decisions:
        prediction = _predict_objectives(
            models, np.array([decision[name] for name in FEATURE_NAMES])
        )
        # 3 objectives to minimize: [annual_cost, -S45, road_fallback_share]
        F_list.append(prediction[:3])

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds_indices = NonDominatedSorting().do(np.array(F_list), only_non_dominated_front=True)

    front_rows = []
    for idx in nds_indices:
        decision = candidate_decisions[idx]
        prediction = _predict_objectives(
            models, np.array([decision[name] for name in FEATURE_NAMES])
        )
        front_rows.append(
            {
                **decision,
                "annual_cost_inr": float(prediction[0]),
                "S45": float(-prediction[1]),
                "road_fallback_share": float(prediction[2]),
                "em_p95_min": float(prediction[3]),
            }
        )
    pareto = pd.DataFrame(front_rows)
    pareto.to_csv(OUT_DIR / "m6_surrogate_pareto.csv", index=False)
    representative = _select_representatives(pareto, count=N_RECHECKS, baseline_decision=baseline_decision)

    recheck_rows = []
    for recheck_id, (_, row) in enumerate(representative.iterrows(), start=1):
        decision = {name: int(row[name]) for name in FEATURE_NAMES}
        true_metrics = _policy_evaluation(
            decision,
            vehicles,
            network_cfg,
            recheck_days,
            COMMON_SEED,
            simulator_available=simulate_days is not None,
            data=data,
        )
        recheck_rows.append(
            {
                "recheck_id": recheck_id,
                **decision,
                "predicted_annual_cost_inr": float(row["annual_cost_inr"]),
                "true_annual_cost_inr": true_metrics["annual_cost_inr"],
                "annual_cost_abs_error_inr": abs(
                    true_metrics["annual_cost_inr"] - float(row["annual_cost_inr"])
                ),
                "predicted_S45": float(row["S45"]),
                "true_S45": true_metrics["S45"],
                "S45_abs_error_pp": abs(true_metrics["S45"] - float(row["S45"])),
                "predicted_road_fallback_share": float(row["road_fallback_share"]),
                "true_road_fallback_share": true_metrics["road_fallback_share"],
                "road_fallback_abs_error": abs(
                    true_metrics["road_fallback_share"] - float(row["road_fallback_share"])
                ),
                "predicted_em_p95_min": float(row["em_p95_min"]),
                "true_em_p95_min": true_metrics["em_p95_min"],
                "em_p95_abs_error_min": abs(
                    true_metrics["em_p95_min"] - float(row["em_p95_min"])
                ),
                "verification_source": true_metrics["source"],
                "recheck_day_indices": json.dumps(recheck_days),
                "common_seed": COMMON_SEED,
                "n_days": len(recheck_days),
            }
        )
        print(f"M6 directly rechecked Pareto point {recheck_id}/{len(representative)} (on {len(recheck_days)} stratified days).")
    rechecks = pd.DataFrame(recheck_rows)
    rechecks.to_csv(OUT_DIR / "m6_surrogate_recheck.csv", index=False)

    two_objective_front = _run_two_objective_nsga(models, vehicle_max)
    two_objective_front.to_csv(OUT_DIR / "m6_cost_service_nsga2.csv", index=False)
    weighted = _run_weighted_sum_checks(models, pareto, two_objective_front)
    weighted.to_csv(OUT_DIR / "m6_weighted_sum_results.csv", index=False)
    if not weighted["on_or_behind_nsga2_front"].all():
        raise AssertionError(
            "Weighted-sum points must lie on or behind the NSGA-II 2D front within "
            "the recorded surrogate tolerances."
        )

    representatives_for_export = representative.merge(
        rechecks[
            [
                *FEATURE_NAMES,
                "true_annual_cost_inr",
                "true_S45",
                "true_road_fallback_share",
                "true_em_p95_min",
            ]
        ].drop_duplicates(subset=list(FEATURE_NAMES), keep="first"),
        on=list(FEATURE_NAMES),
        how="left",
        validate="many_to_one",
    )
    top3 = _export_top3(
        representatives_for_export,
        vehicles,
        network_cfg,
        m4_solutions,
        data,
    )
    _create_plots(pareto, two_objective_front, weighted, rechecks)

    if len(weighted) != 7:
        raise AssertionError("Weighted-sum verification must contain 7 solutions.")
    if len(rechecks) != N_RECHECKS:
        raise AssertionError(f"Expected {N_RECHECKS} direct rechecks, got {len(rechecks)}.")
    if len(top3) != 3:
        raise AssertionError("M7 requires exactly 3 exported candidates.")
    print(f"M6 complete: {len(observations)} LHS evaluations, {len(pareto)} Pareto points, {len(rechecks)} rechecks.")
    print(f"Optimization mode: {'analytical stub' if USE_STUB else 'real M5 simulator'}")
    print(f"Top 3 candidates saved to {OUT_DIR / 'm6_top3_candidates.json'}")
    return {
        "lhs_evaluations": observations,
        "loocv": loocv,
        "pareto": pareto,
        "rechecks": rechecks,
        "weighted_sum": weighted,
        "top3": top3,
        "use_stub": USE_STUB,
    }


if __name__ == "__main__":
    run_m6()
