"""
Module M4 – Network Hub-and-Spoke MILP
=======================================
Obstacle-avoiding visibility graph + PuLP/CBC MILP to optimise hub selection,
fleet sizing, and PHC assignment over the Skylink district network.

Outputs (all in outputs/m4/):
  distance_matrix_km.csv
  network_nofly_paths.png
  m4_bruteforce_verification.csv
  m4_solutions.json
  m4_cost_vs_vehicle.csv
  m4_cost_vs_vehicle.png
  m4_best_network_map.png
"""

from __future__ import annotations
import itertools, json, math, sys, warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
import pulp
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
DATA_DIR = PROJ_DIR / "datasets"
OUT_DIR  = PROJ_DIR / "outputs" / "m4"
M1_JSON  = PROJ_DIR / "outputs" / "m1" / "m1_designs.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load project data ─────────────────────────────────────────────────────────
sys.path.insert(0, str(SRC_DIR))
from data_loader import load_all_data

_data      = load_all_data()
sites_df   = _data["sites"]
demand_df  = _data["demand"]
costs_cfg  = _data["costs"]
nofly_geo  = _data["nofly_geojson"]

# ── Constants from costs.yaml ────────────────────────────────────────────────
DISC_RATE      = costs_cfg["discount_rate"]           # 0.10
HUB_CAPEX      = costs_cfg["hub"]["capex"]            # 1 800 000
HUB_OPEX       = costs_cfg["hub"]["opex_per_year"]    # 400 000
HUB_LIFE       = costs_cfg["hub"]["life_years"]       # 10
DRONE_FIX      = costs_cfg["drone"]["capex_fixed"]    # 200 000
DRONE_KG       = costs_cfg["drone"]["capex_per_kg_mtow"]  # 45 000
DRONE_LIFE     = costs_cfg["drone"]["life_years"]     # 3
DRONE_MAINT    = costs_cfg["drone"]["maintenance_frac_of_capex_per_year"]  # 0.15
BAY_CAPEX      = costs_cfg["charging_bay"]["capex"]   # 90 000
BAY_LIFE       = costs_cfg["charging_bay"]["life_years"]  # 5
BAY_KW         = costs_cfg["charging_bay"]["power_kW"]    # 1.5
BAY_EFF        = costs_cfg["charging_bay"]["charge_efficiency"]  # 0.90
STAFF_SALARY   = costs_cfg["staff"]["salary_per_person_year"]  # 420 000
DRONES_PER_OP  = costs_cfg["staff"]["drones_per_operator"]     # 4
SHIFTS         = costs_cfg["staff"]["shifts_per_day"]          # 3
RELIEF         = costs_cfg["staff"]["relief_factor"]           # 1.2
GRID_KWH       = costs_cfg["energy"]["grid_per_kWh"]           # 9.0
BATT_COST_Wh   = costs_cfg["battery_pack"]["cost_per_Wh"]      # 22
BATT_CYCLES    = costs_cfg["battery_pack"]["cycle_life"]        # 500
ROAD_EMERG     = costs_cfg["road_fallback"]["cost_per_trip"]["emergency"]   # 2500
ROAD_ROUT      = costs_cfg["road_fallback"]["cost_per_trip"]["routine"]     # 500
ROAD_LAB       = costs_cfg["road_fallback"]["cost_per_trip"]["lab"]         # 500
LOAD_TAKEOFF   = costs_cfg["operations"]["load_and_takeoff_min"]           # 3
DROP_PICKUP    = costs_cfg["operations"]["drop_or_pickup_min"]             # 1
BATT_SWAP      = costs_cfg["operations"]["battery_swap_turnaround_min"]    # 6
GROUND_MIN     = LOAD_TAKEOFF + DROP_PICKUP + BATT_SWAP                    # 10 min

# ── Demand & Road Fallback lookup from demand.csv ────────────────────────────
demand_idx = demand_df.set_index("phc_id")
DEMAND_LAMBDA = {
    p: float(demand_idx.loc[p, "emergency_per_day"] + demand_idx.loc[p, "routine_per_day"] + demand_idx.loc[p, "lab_per_day"])
    for p in demand_idx.index
}
DEMAND_ROAD_ANNUAL = {
    p: 365.0 * (float(demand_idx.loc[p, "emergency_per_day"]) * ROAD_EMERG +
                (float(demand_idx.loc[p, "routine_per_day"]) + float(demand_idx.loc[p, "lab_per_day"])) * ROAD_ROUT)
    for p in demand_idx.index
}

def crf(r: float, n: int) -> float:
    """Capital Recovery Factor."""
    return r * (1 + r)**n / ((1 + r)**n - 1)

CRF_HUB   = crf(DISC_RATE, HUB_LIFE)
CRF_DRONE = crf(DISC_RATE, DRONE_LIFE)
CRF_BAY   = crf(DISC_RATE, BAY_LIFE)

# ── Site tables ───────────────────────────────────────────────────────────────
HUB_IDS = ["H0","H1","H2","H3","H4","H5","H6","P09"]
PHC_IDS = [f"P{i:02d}" for i in range(1, 25)]   # P01..P24

sites_idx = sites_df.set_index("site_id")

def xy(sid: str):
    r = sites_idx.loc[sid]
    return float(r["x_km"]), float(r["y_km"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: Visibility Graph & Distance Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_nofly_polygons(buf_km: float = 0.05):
    """Return (original_union, buffered_union, list_of_buffered_polys)."""
    raw_polys = []
    for feat in nofly_geo["features"]:
        raw_polys.append(shape(feat["geometry"]))
    orig   = unary_union(raw_polys)
    bufs   = [p.buffer(buf_km) for p in raw_polys]
    buf_u  = unary_union(bufs)
    return orig, buf_u, bufs


def _polygon_vertices(poly: Polygon) -> list[tuple]:
    """Return exterior + interior ring vertices of a polygon."""
    verts = list(poly.exterior.coords[:-1])          # drop repeated last
    for ring in poly.interiors:
        verts += list(ring.coords[:-1])
    return verts


def _all_buffered_vertices(bufs: list) -> list[tuple]:
    verts = []
    for p in bufs:
        if isinstance(p, Polygon):
            verts += _polygon_vertices(p)
        elif isinstance(p, MultiPolygon):
            for sub in p.geoms:
                verts += _polygon_vertices(sub)
    return verts


def _seg_clear(p1, p2, blocker) -> bool:
    """True if segment p1→p2 does not cross the INTERIOR of blocker."""
    seg = LineString([p1, p2])
    # Allow touches (boundary) but not interior crossing
    return not seg.crosses(blocker) and not blocker.contains(seg)


def build_visibility_graph(buf_km: float = 0.05):
    """
    Build visibility graph over all 31 site positions + no-fly boundary vertices.
    Returns (graph G with edge weight=euclidean km, node list).
    """
    orig, buf_union, bufs = _build_nofly_polygons(buf_km)

    # Node positions: all sites + buffered polygon boundary vertices
    site_nodes = {sid: xy(sid) for sid in sites_df["site_id"].tolist()}
    buf_verts  = _all_buffered_vertices(bufs)
    buf_nodes  = {f"BV{i}": v for i, v in enumerate(buf_verts)}

    all_nodes  = {**site_nodes, **buf_nodes}
    node_ids   = list(all_nodes.keys())
    pos        = [all_nodes[n] for n in node_ids]

    G = nx.Graph()
    for nid, (x, y) in all_nodes.items():
        G.add_node(nid, x=x, y=y)

    n = len(node_ids)
    print(f"  Visibility graph: {n} nodes ({len(site_nodes)} sites + {len(buf_nodes)} boundary verts)")
    checked = 0
    for i in range(n):
        for j in range(i+1, n):
            p1 = pos[i]; p2 = pos[j]
            if _seg_clear(p1, p2, orig):
                dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                G.add_edge(node_ids[i], node_ids[j], weight=dist)
            checked += 1

    print(f"  Edges checked: {checked}, edges added: {G.number_of_edges()}")
    return G, orig, bufs, site_nodes


def compute_distance_matrix(G: nx.Graph, orig, site_nodes: dict) -> pd.DataFrame:
    """Dijkstra shortest paths between all hub/PHC pairs (deduplicated index)."""
    # P09 is in both HUB_IDS and PHC_IDS — build deduped list preserving order
    seen = set(); all_ids = []
    for sid in HUB_IDS + PHC_IDS:
        if sid not in seen:
            seen.add(sid); all_ids.append(sid)

    n   = len(all_ids)
    mat = np.full((n, n), np.inf)
    np.fill_diagonal(mat, 0.0)

    for i, src in enumerate(all_ids):
        try:
            lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        except nx.NodeNotFound:
            continue
        for j, dst in enumerate(all_ids):
            if src == dst:
                mat[i, j] = 0.0
            else:
                mat[i, j] = lengths.get(dst, np.inf)

    df = pd.DataFrame(mat, index=all_ids, columns=all_ids)
    return df


def verify_no_crossing(G: nx.Graph, dist_df: pd.DataFrame, orig) -> bool:
    """Spot-check that reconstructed paths never cross no-fly interiors."""
    ok = True
    for src in HUB_IDS[:3]:
        for dst in PHC_IDS[:6]:
            if src == dst: continue
            try:
                path = nx.shortest_path(G, src, dst, weight="weight")
                for a, b in zip(path[:-1], path[1:]):
                    pa = (G.nodes[a]["x"], G.nodes[a]["y"])
                    pb = (G.nodes[b]["x"], G.nodes[b]["y"])
                    if not _seg_clear(pa, pb, orig):
                        print(f"  [WARN] Segment {a}→{b} crosses no-fly zone!")
                        ok = False
            except nx.NetworkXNoPath:
                pass
    if ok:
        print("  [OK] All sampled path segments clear of no-fly interiors.")
    return ok


def plot_network(G: nx.Graph, dist_df: pd.DataFrame, orig, bufs,
                 site_nodes: dict, out_path: Path):
    """Plot no-fly zones, sites, and sample deflected vs straight paths."""
    fig, ax = plt.subplots(figsize=(13, 10))

    # Draw no-fly zones
    colors = ["#FF9999", "#99CCFF"]
    labels = ["Civil airstrip", "Reserve forest"]
    for poly, col, lbl in zip(bufs, colors, labels):
        if isinstance(poly, Polygon):
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.35, color=col, label=lbl)
            ax.plot(xs, ys, color=col, lw=1.2)

    # Sites
    hub_xy  = [(xy(h)[0], xy(h)[1]) for h in HUB_IDS]
    phc_xy  = [(xy(p)[0], xy(p)[1]) for p in PHC_IDS]
    ax.scatter([x for x,y in hub_xy], [y for x,y in hub_xy],
               c="blue", s=120, zorder=5, label="Candidate Hubs", marker="^")
    ax.scatter([x for x,y in phc_xy], [y for x,y in phc_xy],
               c="green", s=60, zorder=5, label="PHCs", marker="o")
    for sid in HUB_IDS + PHC_IDS:
        ax.annotate(sid, xy(sid), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    # Sample paths: H0→P15, H0→P16, H0→P17 (likely deflected by Civil airstrip)
    sample_pairs = [("H0","P15"),("H0","P16"),("H0","P17"),("H2","P01"),("H5","P01")]
    path_colors  = ["darkred","purple","darkorange","teal","navy"]
    for (src, dst), col in zip(sample_pairs, path_colors):
        try:
            path = nx.shortest_path(G, src, dst, weight="weight")
            px   = [G.nodes[n]["x"] for n in path]
            py   = [G.nodes[n]["y"] for n in path]
            ax.plot(px, py, "-", color=col, lw=1.6, alpha=0.85,
                    label=f"{src}→{dst} bypass")
            # Straight dashed line
            sx, sy = xy(src); dx, dy = xy(dst)
            ax.plot([sx, dx], [sy, dy], "--", color=col, lw=0.9, alpha=0.4)
        except Exception:
            pass

    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title("Skylink District – Obstacle-Avoiding Flight Paths")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: Vehicle designs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_vehicle_designs() -> list[dict]:
    if M1_JSON.exists():
        with open(M1_JSON) as f:
            designs = json.load(f)
        # Compute average cruise power
        for d in designs:
            d["P_cruise_avg_W"] = (d.get("P_cruise_loaded_W", 0) +
                                   d.get("P_cruise_empty_W",  0)) / 2.0
        print(f"  Loaded {len(designs)} vehicle designs from {M1_JSON.name}")
        return designs
    else:
        print("  [WARN] m1_designs.json not found — using placeholder designs.")
        placeholders = [
            {"design_id":"PH14","R_km":14,"MTOW_kg":7.5, "V_ms":16.0,"E_Wh":450, "drone_capex_inr":537500,  "P_cruise_avg_W":1400},
            {"design_id":"PH18","R_km":18,"MTOW_kg":9.5, "V_ms":16.0,"E_Wh":700, "drone_capex_inr":627500,  "P_cruise_avg_W":1900},
            {"design_id":"PH20","R_km":20,"MTOW_kg":11.0,"V_ms":16.0,"E_Wh":900, "drone_capex_inr":695000,  "P_cruise_avg_W":2300},
            {"design_id":"PH22","R_km":22,"MTOW_kg":13.0,"V_ms":16.0,"E_Wh":1200,"drone_capex_inr":785000,  "P_cruise_avg_W":2900},
            {"design_id":"PH26","R_km":26,"MTOW_kg":17.5,"V_ms":16.0,"E_Wh":1800,"drone_capex_inr":987500,  "P_cruise_avg_W":4200},
        ]
        return placeholders


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: MILP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_sortie_energy_Wh(vehicle: dict, L_km: float) -> float:
    """
    Compute route-specific sortie energy:
    E_sortie_Wh(i, j) = ((P_cruise_loaded_W + P_cruise_empty_W) * (L_ij*1000 / V_ms) +
                         60 * (P_hover_loaded_W + P_hover_empty_W)) / 3600.0
    Fallback to E_Wh if power fields are missing.
    """
    P_cr_L = vehicle.get("P_cruise_loaded_W")
    P_cr_E = vehicle.get("P_cruise_empty_W")
    P_hov_L = vehicle.get("P_hover_loaded_W")
    P_hov_E = vehicle.get("P_hover_empty_W")
    V_ms = vehicle.get("V_ms", 16.0)
    E_Wh = vehicle.get("E_Wh", 1000.0)
    if None in (P_cr_L, P_cr_E, P_hov_L, P_hov_E):
        return float(E_Wh)
    t_cruise_s = (float(L_km) * 1000.0) / float(V_ms)
    E_joules = (float(P_cr_L) + float(P_cr_E)) * t_cruise_s + 60.0 * (float(P_hov_L) + float(P_hov_E))
    return float(E_joules / 3600.0)


def build_and_solve_milp(
    dist_df: pd.DataFrame,
    vehicle: dict,
    hubs: list[str] = None,
    phcs:  list[str] = None,
    relax: bool      = False,
    verbose: bool    = True,
) -> Optional[dict]:
    """
    Build and solve the MILP for a given vehicle design.
    If relax=True, solve LP relaxation.
    Returns result dict or None if infeasible.
    """
    hubs = hubs or HUB_IDS
    phcs = phcs or PHC_IDS
    J    = hubs
    I    = phcs

    R_km    = vehicle["R_km"]
    V_ms    = vehicle["V_ms"]
    E_Wh    = vehicle["E_Wh"]
    MTOW    = vehicle["MTOW_kg"]
    D_CAPEX = DRONE_FIX + DRONE_KG * MTOW
    P_cr    = vehicle["P_cruise_avg_W"]

    # Precompute route-specific sortie energy for all (i, j) pairs
    E_sortie = {}
    for j in J:
        for i in I:
            L = float(dist_df.at[j, i]) if j != i else 0.0
            E_sortie[(i, j)] = calc_sortie_energy_Wh(vehicle, L)

    # Reachability: x_ij = 0 if L_ij > R_km
    reach = {}
    for j in J:
        for i in I:
            L = float(dist_df.at[j, i]) if j != i else 0.0
            reach[(i, j)] = (L <= R_km + 1e-9) and not np.isinf(L)

    # ---- Big-M values -------------------------------------------------------
    total_demand = sum(DEMAND_LAMBDA[i] for i in I)
    L_max        = float(dist_df.loc[J, I].replace(np.inf, 0).values.max())
    
    reachable_busy = [
        GROUND_MIN + 2.0 * float(dist_df.at[j, i]) * 1000.0 / (V_ms * 60.0)
        for j in J for i in I if reach[(i, j)]
    ]
    t_max_min = max(reachable_busy) if reachable_busy else (GROUND_MIN + 2.0 * L_max * 1000.0 / (V_ms * 60.0))
    M_n_raw   = math.ceil(total_demand * t_max_min / (1440.0 * 0.70))
    M_n       = max(M_n_raw, 10)

    # Charging: grid energy per sortie in kWh = E_sortie / 1000.0 / BAY_EFF
    reachable_energies = [E_sortie[(i, j)] for j in J for i in I if reach[(i, j)]]
    E_sortie_max_Wh   = max(reachable_energies) if reachable_energies else E_Wh
    E_day_max_kWh     = total_demand * (E_sortie_max_Wh / 1000.0 / BAY_EFF)
    bay_daily_cap_kWh = 0.80 * 24.0 * BAY_KW  # 28.8 kWh/day per bay
    M_b_raw   = math.ceil(E_day_max_kWh / bay_daily_cap_kWh)
    M_b       = max(M_b_raw, 5)
    M_op      = max(M_n, 10)

    if verbose:
        print(f"\n  Big-M: M_n={M_n} (from total_demand={total_demand:.2f} trips/day, t_max={t_max_min:.1f} min)")
        print(f"         M_b={M_b} (from E_day_max={E_day_max_kWh:.2f} kWh, bay_cap={bay_daily_cap_kWh:.2f} kWh/day)")
        print(f"  Formula: M_n = ceil(sum_i(lambda_i)*t_max / (1440*0.70)) = {M_n_raw}")
        print(f"           M_b = ceil(sum_i(lambda_i)*(E_sortie_max/1000/{BAY_EFF}) / (0.80*24*{BAY_KW})) = {M_b_raw}")

    # ---- PuLP model ---------------------------------------------------------
    prob = pulp.LpProblem("Skylink_M4", pulp.LpMinimize)

    cat_b = "Continuous" if relax else "Binary"
    cat_i = "Continuous" if relax else "Integer"

    y  = {j: pulp.LpVariable(f"y_{j}",  0, 1, cat=cat_b) for j in J}
    x  = {(i,j): pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat=cat_b)
          for j in J for i in I}
    z  = {i: pulp.LpVariable(f"z_{i}", 0, 1, cat=cat_b) for i in I}
    nv = {j: pulp.LpVariable(f"n_{j}", 0, M_n, cat=cat_i) for j in J}
    bv = {j: pulp.LpVariable(f"b_{j}", 0, M_b, cat=cat_i) for j in J}
    op = {j: pulp.LpVariable(f"op_{j}", 0, M_op, cat=cat_i) for j in J}

    # ---- Constraints --------------------------------------------------------
    # 1. Each PHC served exactly once (drone hub OR road fallback)
    for i in I:
        prob += (pulp.lpSum(x[(i,j)] for j in J) + z[i] == 1,
                 f"serve_{i}")

    # 2. Assignment only to open hub & reachable
    for j in J:
        for i in I:
            if reach[(i, j)]:
                prob += (x[(i,j)] <= y[j], f"open_{i}_{j}")
            else:
                prob += (x[(i,j)] == 0, f"unreach_{i}_{j}")

    # 3. At least one hub open
    prob += (pulp.lpSum(y[j] for j in J) >= 1, "min_one_hub")

    # 4. Capacity: total daily busy minutes at hub j <= 0.70 * 1440 * n_j
    for j in J:
        flight_busy = pulp.lpSum(
            x[(i,j)] * DEMAND_LAMBDA[i] *
            (GROUND_MIN + 2 * float(dist_df.at[j, i]) * 1000 / (V_ms * 60))
            for i in I if reach[(i, j)]
        )
        prob += (flight_busy <= 0.70 * 1440 * nv[j], f"cap_drone_{j}")
        prob += (nv[j] <= M_n * y[j], f"bigM_n_{j}")

    # 5. Charging bays: daily grid energy <= 0.80 * 24 * BAY_KW * b_j (kWh)
    for j in J:
        daily_grid_kWh = pulp.lpSum(
            x[(i,j)] * DEMAND_LAMBDA[i] * (E_sortie[(i, j)] / 1000.0 / BAY_EFF)
            for i in I if reach[(i, j)]
        )
        prob += (daily_grid_kWh <= bay_daily_cap_kWh * bv[j], f"cap_bay_{j}")
        prob += (bv[j] <= M_b * y[j], f"bigM_b_{j}")

    # 6. Operators: op_j >= n_j / 4 and op_j >= y_j
    for j in J:
        prob += (op[j] * DRONES_PER_OP >= nv[j], f"op_min_drones_{j}")
        prob += (op[j] >= y[j], f"op_min_one_{j}")
        prob += (op[j] <= M_op * y[j], f"bigM_op_{j}")

    # ---- Objective ----------------------------------------------------------
    # Annualised hub costs
    hub_ann = pulp.lpSum(
        y[j] * (CRF_HUB * HUB_CAPEX + HUB_OPEX) for j in J
    )

    # Drone capex + maintenance (annualised per drone)
    drone_ann_per = CRF_DRONE * D_CAPEX + DRONE_MAINT * D_CAPEX
    drone_ann = pulp.lpSum(nv[j] * drone_ann_per for j in J)

    # Bay capex
    bay_ann = pulp.lpSum(bv[j] * CRF_BAY * BAY_CAPEX for j in J)

    # Staff
    staff_ann = pulp.lpSum(
        op[j] * SHIFTS * RELIEF * STAFF_SALARY for j in J
    )

    # Energy (grid) per year
    # kWh/year = sum_ij x_ij * lambda_i * (E_sortie_Wh / 1000.0 / 0.90) * 365
    energy_ann = pulp.lpSum(
        x[(i,j)] * DEMAND_LAMBDA[i] * (E_sortie[(i, j)] / 1000.0 / BAY_EFF) * 365 * GRID_KWH
        for j in J for i in I if reach[(i, j)]
    )

    # Battery cycle wear
    # Fractional cycles = E_sortie_Wh / (0.80 * E_Wh) at cycle_life = 500
    # Annual wear = sum_ij x_ij * lambda_i * (E_sortie_Wh / (0.80 * E_Wh)) * (BATT_COST_Wh * E_Wh / BATT_CYCLES) * 365
    #             = sum_ij x_ij * lambda_i * (E_sortie_Wh * BATT_COST_Wh / (0.80 * BATT_CYCLES)) * 365
    batt_ann = pulp.lpSum(
        x[(i,j)] * DEMAND_LAMBDA[i] * (E_sortie[(i, j)] * BATT_COST_Wh / (0.80 * BATT_CYCLES)) * 365
        for j in J for i in I if reach[(i, j)]
    )

    # Road fallback
    road_ann = pulp.lpSum(z[i] * DEMAND_ROAD_ANNUAL[i] for i in I)

    total_cost = hub_ann + drone_ann + bay_ann + staff_ann + energy_ann + batt_ann + road_ann
    prob += total_cost

    # ---- Solve --------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
    status = prob.solve(solver)

    if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
        if verbose:
            print(f"  [WARN] Solver status: {pulp.LpStatus[prob.status]}")
        return None

    # ---- Extract solution ---------------------------------------------------
    obj_val = float(pulp.value(prob.objective))
    hubs_open  = [j for j in J if pulp.value(y[j]) > 0.5]
    assignments = {i: next((j for j in J if pulp.value(x[(i,j)]) > 0.5), None)
                   for i in I}
    road_phcs  = [i for i in I if pulp.value(z[i]) > 0.5]
    n_drones   = {j: int(round(pulp.value(nv[j]))) for j in J}
    n_bays     = {j: int(round(pulp.value(bv[j]))) for j in J}
    n_ops      = {j: int(round(pulp.value(op[j]))) for j in J}

    hub_cost_val    = float(pulp.value(hub_ann))
    drone_cost_val  = float(pulp.value(drone_ann))
    bay_cost_val    = float(pulp.value(bay_ann))
    staff_cost_val  = float(pulp.value(staff_ann))
    energy_cost_val = float(pulp.value(energy_ann))
    batt_cost_val   = float(pulp.value(batt_ann))
    road_cost_val   = float(pulp.value(road_ann))

    cost_breakdown = {
        "hubs":          hub_cost_val,
        "staff":         staff_cost_val,
        "drones":        drone_cost_val,
        "bays":          bay_cost_val,
        "energy":        energy_cost_val,
        "battery_wear":  batt_cost_val,
        "road_fallback": road_cost_val,
        "total":         obj_val,
    }

    return {
        "status":         pulp.LpStatus[prob.status],
        "objective":      obj_val,
        "hubs_open":      hubs_open,
        "assignments":    assignments,
        "road_phcs":      road_phcs,
        "n_drones":       n_drones,
        "n_bays":         n_bays,
        "n_ops":          n_ops,
        "cost_breakdown": cost_breakdown,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: Brute-Force Verification (sub-instance)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def brute_force_subinstance(dist_df: pd.DataFrame, vehicle: dict) -> pd.DataFrame:
    """
    Enumerate all 2^4=16 hub subsets for 4-hub, 8-PHC sub-instance.
    For each subset solve exact assignment / fleet and return cost breakdown.
    Uses actual demand lambda_i from demand.csv and route-specific sortie energy.
    """
    sub_hubs = ["H0","H1","H2","H4"]
    sub_phcs = [f"P{i:02d}" for i in range(1, 9)]   # P01..P08

    R_km = vehicle["R_km"]
    V_ms = vehicle["V_ms"]
    E_Wh = vehicle["E_Wh"]
    MTOW = vehicle["MTOW_kg"]
    D_CAPEX = DRONE_FIX + DRONE_KG * MTOW
    drone_ann_per = CRF_DRONE * D_CAPEX + DRONE_MAINT * D_CAPEX
    bay_daily_cap_kWh = 0.80 * 24.0 * BAY_KW

    rows = []
    for k in range(1, 2**len(sub_hubs)):   # skip 0 (no hub open)
        open_set = [sub_hubs[b] for b in range(len(sub_hubs)) if (k >> b) & 1]
        hub_cost = len(open_set) * (CRF_HUB * HUB_CAPEX + HUB_OPEX)

        # Assign each PHC to nearest reachable open hub (greedy optimal for fixed hubs)
        total_drone_min = {j: 0.0 for j in open_set}
        total_grid_kWh  = {j: 0.0 for j in open_set}
        total_batt_wear = 0.0
        road    = []
        for i in sub_phcs:
            lam = DEMAND_LAMBDA[i]
            best_j, best_L = None, np.inf
            for j in open_set:
                L = float(dist_df.at[j, i]) if j != i else 0.0
                if L <= R_km + 1e-9 and L < best_L:
                    best_j, best_L = j, L
            if best_j is not None:
                t_min = GROUND_MIN + 2 * best_L * 1000 / (V_ms * 60)
                total_drone_min[best_j] += lam * t_min
                E_s = calc_sortie_energy_Wh(vehicle, best_L)
                total_grid_kWh[best_j]  += lam * (E_s / 1000.0 / BAY_EFF)
                wear_sortie = E_s * BATT_COST_Wh / (0.80 * BATT_CYCLES)
                total_batt_wear += lam * wear_sortie * 365
            else:
                road.append(i)

        # Fleet sizing
        n_drones = {j: max(1, math.ceil(total_drone_min[j] / (1440 * 0.70)))
                    for j in open_set}
        n_bays   = {j: max(1, math.ceil(total_grid_kWh[j] / bay_daily_cap_kWh))
                    for j in open_set}
        n_ops    = {j: max(1, math.ceil(n_drones[j] / DRONES_PER_OP))
                    for j in open_set}

        drone_cost  = sum(n_drones[j] * drone_ann_per for j in open_set)
        bay_cost    = sum(n_bays[j] * CRF_BAY * BAY_CAPEX for j in open_set)
        staff_cost  = sum(n_ops[j] * SHIFTS * RELIEF * STAFF_SALARY for j in open_set)
        energy_cost = sum(total_grid_kWh[j] * 365 * GRID_KWH for j in open_set)
        batt_cost   = total_batt_wear
        road_cost   = sum(DEMAND_ROAD_ANNUAL[i] for i in road)
        total       = hub_cost + drone_cost + bay_cost + staff_cost + energy_cost + batt_cost + road_cost

        rows.append({
            "hub_subset":  str(open_set),
            "n_hubs":      len(open_set),
            "hub_cost":    hub_cost,
            "drone_cost":  drone_cost,
            "bay_cost":    bay_cost,
            "staff_cost":  staff_cost,
            "energy_cost": energy_cost,
            "batt_cost":   batt_cost,
            "road_cost":   road_cost,
            "total_cost":  total,
            "road_phcs":   str(road),
        })

    df = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_cost_vs_vehicle(results: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ids   = [r["design_id"] for r in results if r.get("milp_obj") is not None]
    costs = [r["milp_obj"] / 1e6 for r in results if r.get("milp_obj") is not None]
    rmkm  = [r["R_km"] for r in results if r.get("milp_obj") is not None]

    ax = axes[0]
    ax.bar(ids, costs, color="steelblue")
    ax.set_xlabel("Vehicle Design"); ax.set_ylabel("Annual Cost (M INR)")
    ax.set_title("MILP Annual Cost vs Vehicle Design")
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", ls="--", alpha=0.5)

    ax = axes[1]
    ax.plot(rmkm, costs, "b-o")
    ax.set_xlabel("Range (km)"); ax.set_ylabel("Annual Cost (M INR)")
    ax.set_title("Cost vs Vehicle Range")
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_best_network(G: nx.Graph, sol: dict, dist_df: pd.DataFrame,
                      bufs, out_path: Path):
    fig, ax = plt.subplots(figsize=(13, 10))

    colors = ["#FF9999","#99CCFF"]
    labels = ["Civil airstrip","Reserve forest"]
    for poly, col, lbl in zip(bufs, colors, labels):
        if isinstance(poly, Polygon):
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.35, color=col, label=lbl)

    # All hubs
    for j in HUB_IDS:
        xj, yj = xy(j)
        col = "blue" if j in sol["hubs_open"] else "lightgrey"
        mk  = "^" if j in sol["hubs_open"] else "v"
        ax.scatter(xj, yj, c=col, s=140, zorder=5, marker=mk)
        ax.annotate(j, (xj, yj), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", color=col)

    # PHCs and assignment lines
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(sol["hubs_open"]))
    hub_col = {h: cmap(k) for k, h in enumerate(sol["hubs_open"])}
    for i in PHC_IDS:
        xi, yi = xy(i)
        j = sol["assignments"].get(i)
        if j is not None and j in hub_col:
            ax.scatter(xi, yi, c=[hub_col[j]], s=55, zorder=4, marker="o")
            xj, yj = xy(j)
            ax.plot([xi, xj], [yi, yj], "-", color=hub_col[j], lw=0.8, alpha=0.5)
        else:
            ax.scatter(xi, yi, c="red", s=55, zorder=4, marker="x")
        ax.annotate(i, (xi, yi), fontsize=5.5, ha="center", va="top",
                    xytext=(0,-5), textcoords="offset points")

    # Legend
    patches = [mpatches.Patch(color=hub_col[h], label=f"Hub {h}") for h in sol["hubs_open"]]
    patches += [mpatches.Patch(color="red", label="Road fallback")]
    ax.legend(handles=patches, fontsize=7, loc="upper left")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title("Optimal Network – Best Vehicle Design")
    ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "="*70)
    print("  SKYLINK M4 – Network Hub-and-Spoke MILP")
    print("="*70)

    # ── Step 1: Visibility graph & distance matrix ────────────────────────────
    print("\n[STEP 1] Building visibility graph & distance matrix...")
    G, orig, bufs, site_nodes = build_visibility_graph(buf_km=0.05)
    dist_df = compute_distance_matrix(G, orig, site_nodes)
    dist_df.to_csv(OUT_DIR / "distance_matrix_km.csv")
    h0_p15  = float(dist_df.at["H0",  "P15"])
    p09_p09 = float(dist_df.at["P09", "P09"])
    print(f"  Distance matrix saved. Sample H0→P15: {h0_p15:.3f} km")
    print(f"  P09→P09: {p09_p09:.3f} km")

    # Verify no crossing
    verify_no_crossing(G, dist_df, orig)

    # Network plot
    plot_network(G, dist_df, orig, bufs, site_nodes,
                 OUT_DIR / "network_nofly_paths.png")

    # ── Step 2: Vehicle designs ───────────────────────────────────────────────
    print("\n[STEP 2] Loading vehicle designs...")
    designs = load_vehicle_designs()

    # ── Step 3: Brute-force verification (sub-instance) ───────────────────────
    print("\n[STEP 3] Brute-force verification on 4-hub, 8-PHC sub-instance...")
    # Use first design for brute-force verification
    veh_bf = designs[0]   # R14 design is most restrictive – most educational
    bf_df  = brute_force_subinstance(dist_df, veh_bf)
    bf_min = bf_df["total_cost"].iloc[0]
    print(f"  Brute-force optimal: {bf_min:,.0f} INR (subset={bf_df['hub_subset'].iloc[0]})")

    # Also solve MILP on same sub-instance
    milp_sub = build_and_solve_milp(
        dist_df, veh_bf,
        hubs=["H0","H1","H2","H4"],
        phcs=[f"P{i:02d}" for i in range(1,9)],
        verbose=True,
    )
    milp_sub_obj = milp_sub["objective"] if milp_sub else None
    if milp_sub_obj is not None:
        pct_diff = abs(milp_sub_obj - bf_min) / max(bf_min, 1) * 100
        print(f"  MILP sub-instance optimal: {milp_sub_obj:,.0f} INR")
        print(f"  Difference vs brute-force: {pct_diff:.6f}%")
        passed = pct_diff < 1e-4
        print(f"  Verification: {'PASS ✓' if passed else 'FAIL ✗'} (< 1e-4% threshold)")
        # Annotate brute-force table
        bf_df["milp_optimal"] = milp_sub_obj
        bf_df["pct_diff_vs_milp"] = abs(bf_df["total_cost"] - milp_sub_obj) / milp_sub_obj * 100
    bf_df.to_csv(OUT_DIR / "m4_bruteforce_verification.csv", index=False)
    print(f"  Saved m4_bruteforce_verification.csv")

    # ── Step 4: LP relaxation gap ─────────────────────────────────────────────
    print("\n[STEP 4] LP relaxation gap (first design)...")
    lp_sol   = build_and_solve_milp(dist_df, veh_bf, relax=True,  verbose=False)
    milp_sol = build_and_solve_milp(dist_df, veh_bf, relax=False, verbose=False)
    if lp_sol and milp_sol:
        lp_opt   = lp_sol["objective"]
        milp_opt = milp_sol["objective"]
        gap_pct  = 100 * (milp_opt - lp_opt) / max(milp_opt, 1)
        print(f"  LP_opt   = {lp_opt:>14,.2f} INR")
        print(f"  MILP_opt = {milp_opt:>14,.2f} INR")
        print(f"  Gap      = {gap_pct:.4f}%")

    # ── Step 5: Sweep over all designs ────────────────────────────────────────
    print("\n[STEP 5] Sweeping over all vehicle designs...")
    all_results = []
    best_obj    = np.inf
    best_sol    = None
    best_design = None

    for veh in designs:
        print(f"\n  Design: {veh['design_id']}  R={veh['R_km']} km  MTOW={veh['MTOW_kg']} kg")
        sol = build_and_solve_milp(dist_df, veh, verbose=True)
        if sol is not None:
            obj = sol["objective"]
            print(f"  Optimal cost: {obj:,.0f} INR/yr | Hubs open: {sol['hubs_open']}")
            print(f"  Drones: { {j:sol['n_drones'][j] for j in sol['hubs_open']} } (total: {sum(sol['n_drones'][j] for j in sol['hubs_open'])})")
            print(f"  Bays:   { {j:sol['n_bays'][j] for j in sol['hubs_open']} } (total: {sum(sol['n_bays'][j] for j in sol['hubs_open'])})")
            print(f"  Road fallback PHCs: {sol['road_phcs']}")
            row = {
                "design_id":   veh["design_id"],
                "R_km":        veh["R_km"],
                "MTOW_kg":     veh["MTOW_kg"],
                "V_ms":        veh["V_ms"],
                "E_Wh":        veh["E_Wh"],
                "milp_obj":    obj,
                "hubs_open":   str(sol["hubs_open"]),
                "n_road_phcs": len(sol["road_phcs"]),
            }
            all_results.append({**row, **sol})
            if obj < best_obj:
                best_obj    = obj
                best_sol    = sol
                best_design = veh
        else:
            row = {
                "design_id": veh["design_id"],
                "R_km":      veh["R_km"],
                "MTOW_kg":   veh["MTOW_kg"],
                "milp_obj":  None,
            }
            all_results.append(row)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # JSON
    json_out = []
    for r in all_results:
        jr = {k: v for k, v in r.items()
              if k in ["design_id","R_km","MTOW_kg","V_ms","E_Wh",
                        "milp_obj","hubs_open","n_road_phcs","cost_breakdown"]}
        json_out.append(jr)
    with open(OUT_DIR / "m4_solutions.json", "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\n  Saved m4_solutions.json")

    # CSV
    csv_rows = [{k: v for k, v in r.items()
                 if k in ["design_id","R_km","MTOW_kg","V_ms","E_Wh",
                            "milp_obj","hubs_open","n_road_phcs"]}
                for r in all_results]
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "m4_cost_vs_vehicle.csv", index=False)
    print(f"  Saved m4_cost_vs_vehicle.csv")

    # Cost vs vehicle plot
    plot_cost_vs_vehicle(all_results, OUT_DIR / "m4_cost_vs_vehicle.png")

    # Best network map & Itemized Cost Breakdown
    if best_sol is not None:
        print(f"\n  Best design: {best_design['design_id']}"
              f"  Cost: {best_obj:,.0f} INR/yr")
        cb = best_sol.get("cost_breakdown", {})
        print("\n" + "="*70)
        print(f"  ITEMIZED ANNUAL COST BREAKDOWN (Best Design: {best_design['design_id']})")
        print("="*70)
        print(f"  1. Hub Capex & Opex (10 yr):      INR {cb.get('hubs', 0):>12,.2f}")
        print(f"  2. Staff (Operators, 1.2 relief): INR {cb.get('staff', 0):>12,.2f}")
        print(f"  3. Drone Capex & Maint (3 yr):    INR {cb.get('drones', 0):>12,.2f}")
        print(f"  4. Charging Bays (5 yr):          INR {cb.get('bays', 0):>12,.2f}")
        print(f"  5. Grid Energy (90% eff):         INR {cb.get('energy', 0):>12,.2f}")
        print(f"  6. Battery Cycle Wear (500 cyc):  INR {cb.get('battery_wear', 0):>12,.2f}")
        print(f"  7. Road Fallback:                 INR {cb.get('road_fallback', 0):>12,.2f}")
        print("  " + "-"*66)
        print(f"  TOTAL ANNUALIZED COST:            INR {cb.get('total', best_obj):>12,.2f}")
        print("="*70)

        plot_best_network(G, best_sol, dist_df, bufs,
                          OUT_DIR / "m4_best_network_map.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  M4 COMPLETE")
    print("="*70)
    print(f"  Outputs in {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
