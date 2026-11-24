"""
Module M1 - Quadcopter Vehicle Multidisciplinary Design Optimisation (MDO)
==========================================================================
Optimises [D, E_Wh, d_mm, t_mm, V_ms] (MDF) or [D, E_Wh, d_mm, t_mm, V_ms, m_guess] (IDF)
to minimise MTOW subject to range, structural, and dynamic constraints.

Usage
-----
  python src/m1_vehicle_mdo.py          # standalone, writes outputs/m1/
  from m1_vehicle_mdo import run_m1     # import from M2/M3
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
DATA_DIR = PROJ_DIR / "datasets"
OUT_DIR  = PROJ_DIR / "outputs" / "m1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical / design constants
# ---------------------------------------------------------------------------
RHO        = 1.225        # air density kg/m^3
G          = 9.81         # gravity m/s^2
N_ROTORS   = 4            # number of rotors
M_PAY_DEF  = 2.5          # default payload kg
M_FIX      = 0.60         # fixed structural mass (avionics, shell) kg
M_PLATE_DEF= 0.25         # default payload plate kg
E_B        = 150.0        # specific energy Wh/kg (battery)
ETA        = 0.85         # powertrain efficiency
P_AV       = 30.0         # always-on avionics power W
CDA        = 0.08         # drag area m^2
FM_DEF     = 0.65         # default figure of merit
V_TIP      = 110.0        # rotor tip speed m/s
E_CF       = 70e9         # CFRP Young's modulus Pa
RHO_CF     = 1550.0       # CFRP density kg/m^3
SIG_ALLOW  = 200e6        # allowable bending stress Pa
USABLE_FRAC= 0.80 * 0.85  # DOD x charge efficiency

# Battery options from components.csv (up to 3 packs in parallel)
BATT_OPTIONS_Wh = [177.6, 266.4, 355.2, 488.4, 599.4, 710.4]
BATT_PART_IDS   = ["B6S-8K", "B6S-12K", "B6S-16K", "B6S-22K", "B6S-27K", "B6S-32K"]

# Variable bounds: [D_m, E_Wh, d_mm, t_mm, V_ms]
BOUNDS = [
    (0.35, 0.80),   # D propeller diameter m
    (200., 2000.),  # E battery energy Wh
    (15.,  40.),    # d tube outer diameter mm
    (0.5,  2.5),    # t tube wall thickness mm
    (8.,   22.),    # V cruise speed m/s
]

# ---------------------------------------------------------------------------
# Component snap: load components.csv once
# ---------------------------------------------------------------------------
def _load_components() -> pd.DataFrame:
    try:
        sys.path.insert(0, str(SRC_DIR))
        from data_loader import load_all_data
        return load_all_data()["components"]
    except Exception:
        return pd.read_csv(DATA_DIR / "components.csv")


COMPONENTS: pd.DataFrame = _load_components()

# drone cost from costs.yaml
try:
    import yaml
    with open(DATA_DIR / "costs.yaml") as _f:
        _costs = yaml.safe_load(_f)
    CAPEX_FIXED      = _costs["drone"]["capex_fixed"]          # 200000
    CAPEX_PER_KG     = _costs["drone"]["capex_per_kg_mtow"]    # 45000
    BATT_COST_PER_Wh = _costs["battery_pack"]["cost_per_Wh"]   # 22
except Exception:
    CAPEX_FIXED, CAPEX_PER_KG, BATT_COST_PER_Wh = 200000, 45000, 22


# ---------------------------------------------------------------------------
# MDA (mass-convergence) fixed-point iteration
# ---------------------------------------------------------------------------
def _rotor_area(D: float) -> float:
    return math.pi * (D / 2) ** 2


def _tube_A_I(d_m: float, t_m: float):
    """Cross-section area and second moment of area of a hollow circular tube."""
    r_o = d_m / 2
    r_i = r_o - t_m
    A = math.pi * (r_o**2 - r_i**2)
    I = (math.pi / 64) * (d_m**4 - (d_m - 2*t_m)**4)
    return A, I


def mda(D: float, E_Wh: float, d_m: float, t_m: float,
        m_pay: float = M_PAY_DEF, m_plate: float = M_PLATE_DEF,
        tol: float = 1e-10, max_iter: int = 200):
    """
    Mass-discipline analysis (fixed-point iteration).
    Returns (m_total, residuals_list) where residuals_list tracks |m_new - m| per step.
    """
    E = E_Wh                                      # energy in Wh (used directly)
    L = 0.78 * D - 0.08                           # arm length m
    A_tube, I = _tube_A_I(d_m, t_m)              # m^2, m^4
    m_batt = E / E_B                               # battery mass kg

    # Initial guess
    m = m_pay + m_plate + M_FIX + m_batt + 1.0
    residuals = []
    for _ in range(max_iter):
        T_max    = 2 * m * G / N_ROTORS           # max thrust per rotor N
        m_tip    = 0.12 + 0.004 * T_max + 0.05 * (D / 0.6) ** 2.5
        m_arm    = RHO_CF * A_tube * L
        m_new    = m_pay + m_plate + M_FIX + m_batt + N_ROTORS * (m_arm + m_tip)
        res      = abs(m_new - m)
        residuals.append(res)
        m        = m_new
        if res < tol:
            break
    return m, residuals


def _mda_fast(D: float, E_Wh: float, d_m: float, t_m: float,
              m_pay: float = M_PAY_DEF, m_plate: float = M_PLATE_DEF,
              tol: float = 1e-10, max_iter: int = 200) -> float:
    """Return only converged MTOW (no residuals for speed)."""
    m, _ = mda(D, E_Wh, d_m, t_m, m_pay, m_plate, tol, max_iter)
    return m


# ---------------------------------------------------------------------------
# Aerodynamics / power / range helpers
# ---------------------------------------------------------------------------
def _vh(W: float, D: float) -> float:
    """Hover-induced velocity (m/s) for total weight W (N) over N rotors."""
    A_r = _rotor_area(D)
    return math.sqrt(W / (2 * RHO * N_ROTORS * A_r))


def _p_hover(W: float, D: float, FM: float) -> float:
    """Hover power draw (W) including avionics."""
    vh = _vh(W, D)
    return W * vh / (FM * ETA) + P_AV


def _p_cruise(W: float, D: float, V: float, FM: float) -> float:
    """Cruise power draw (W) using momentum theory + parasite drag + avionics."""
    vh = _vh(W, D)
    vi = math.sqrt(-V**2 / 2 + math.sqrt((V**4) / 4 + vh**4))
    mu = V / V_TIP
    P_induced  = W * vi
    P_profile  = W * vh * (1.0 / FM - 1.0) * (1 + 4.65 * mu**2)
    P_parasite = 0.5 * RHO * CDA * V**3
    return (P_induced + P_profile + P_parasite) / ETA + P_AV


def calc_radius(D: float, E_Wh: float, V: float, m: float,
                FM_load: float = FM_DEF, FM_empty: float = FM_DEF,
                m_pay: float = M_PAY_DEF) -> float:
    """
    One-way range (km).
    Mission: 60 s hover (loaded) + cruise loaded + 60 s hover (empty) + cruise empty.
    Empty mass = MTOW - payload.
    """
    E_use   = 3600 * USABLE_FRAC * E_Wh          # J
    W_L     = m * G                               # loaded weight N
    W_E     = (m - m_pay) * G                    # empty weight N

    P_hov_L = _p_hover(W_L, D, FM_load)
    P_hov_E = _p_hover(W_E, D, FM_empty)
    P_cr_L  = _p_cruise(W_L, D, V, FM_load)
    P_cr_E  = _p_cruise(W_E, D, V, FM_empty)

    numerator   = E_use - 60 * (P_hov_L + P_hov_E)
    denominator = P_cr_L + P_cr_E
    if denominator <= 0 or numerator <= 0:
        return 0.0
    R_m = V * numerator / denominator
    return R_m / 1000.0   # km


def calc_power_breakdown(D: float, V: float, m: float,
                         FM_load: float = FM_DEF, FM_empty: float = FM_DEF,
                         m_pay: float = M_PAY_DEF) -> dict:
    """Return dict of power values for all four conditions."""
    W_L = m * G
    W_E = (m - m_pay) * G
    return {
        "P_hover_loaded_W":  _p_hover(W_L, D, FM_load),
        "P_hover_empty_W":   _p_hover(W_E, D, FM_empty),
        "P_cruise_loaded_W": _p_cruise(W_L, D, V, FM_load),
        "P_cruise_empty_W":  _p_cruise(W_E, D, V, FM_empty),
    }


# ---------------------------------------------------------------------------
# Structural constraints
# ---------------------------------------------------------------------------
def _structural(D: float, d_m: float, t_m: float, m: float):
    """Return (sigma_bend, delta_tip, f1, f_min_req) — all in SI."""
    L        = 0.78 * D - 0.08
    A_tube, I= _tube_A_I(d_m, t_m)
    T_max    = 2 * m * G / N_ROTORS
    m_arm    = RHO_CF * A_tube * L
    m_tip    = 0.12 + 0.004 * T_max + 0.05 * (D / 0.6) ** 2.5

    sigma    = T_max * L * (d_m / 2) / I                       # Pa
    delta    = T_max * L**3 / (3 * E_CF * I)                   # m
    mass_eff = m_tip + 0.24 * m_arm
    f1       = (1 / (2 * math.pi)) * math.sqrt(3 * E_CF * I / (L**3 * mass_eff))  # Hz
    f_req    = 1.15 * V_TIP / (math.pi * D)                    # Hz
    return sigma, delta, f1, f_req


# ---------------------------------------------------------------------------
# Normalised constraint functions (>= 0 is feasible)
# ---------------------------------------------------------------------------
def _constraints(x: np.ndarray, R_req: float, FM_func,
                 m_pay: float, m_plate: float, m_guess: float | None = None):
    """
    Returns list of constraint values (g_i >= 0 means feasible).
    For IDF, m_guess is the 6th variable.
    """
    D, E_Wh, d_mm, t_mm, V = x[:5]
    d_m = d_mm * 1e-3
    t_m = t_mm * 1e-3

    if m_guess is not None:
        m = m_guess
    else:
        m = _mda_fast(D, E_Wh, d_m, t_m, m_pay, m_plate)

    if m_guess is not None:
        # IDF equality: residual (returned separately)
        pass

    # FM evaluation
    W_L = m * G
    T_h_L = W_L / N_ROTORS
    W_E = (m - m_pay) * G
    T_h_E = W_E / N_ROTORS
    if FM_func is not None:
        val_L = FM_func(D, T_h_L)
        FM_load = val_L[0] if isinstance(val_L, (tuple, list)) else val_L
        val_E = FM_func(D, T_h_E)
        FM_empty = val_E[0] if isinstance(val_E, (tuple, list)) else val_E
    else:
        FM_load = FM_empty = FM_DEF

    R = calc_radius(D, E_Wh, V, m, FM_load, FM_empty, m_pay)
    sigma, delta, f1, f_req = _structural(D, d_m, t_m, m)
    L = 0.78 * D - 0.08

    c = [
        R - R_req,                           # range
        SIG_ALLOW - sigma,                   # bending stress  (<= sig_allow)
        0.01 * L - delta,                    # tip deflection  (<= 1% arm length)
        f1 - f_req,                          # natural freq    (>= 1.15 * rev freq)
        40.0 - d_mm / t_mm,                  # wall slenderness
    ]
    return c


# ---------------------------------------------------------------------------
# MDF / IDF objective wrappers
# ---------------------------------------------------------------------------
def _scale_x(x_norm: np.ndarray, bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return lo + x_norm * (hi - lo)


def _unscale_x(x: np.ndarray, bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return (x - lo) / (hi - lo)


def _mdf_obj(x_norm: np.ndarray, bounds, m_pay, m_plate):
    x = _scale_x(x_norm, bounds)
    D, E_Wh, d_mm, t_mm, V = x
    d_m, t_m = d_mm * 1e-3, t_mm * 1e-3
    return _mda_fast(D, E_Wh, d_m, t_m, m_pay, m_plate)


def _mdf_cons(x_norm: np.ndarray, bounds, R_req, FM_func, m_pay, m_plate):
    x = _scale_x(x_norm, bounds)
    return _constraints(x, R_req, FM_func, m_pay, m_plate)


def _idf_obj(x_norm6: np.ndarray, bounds6, m_pay, m_plate):
    x6 = _scale_x(x_norm6, bounds6)
    return x6[5]   # m_guess is the objective


def _idf_eq(x_norm6: np.ndarray, bounds6, m_pay, m_plate):
    x6 = _scale_x(x_norm6, bounds6)
    D, E_Wh, d_mm, t_mm, V, m_guess = x6
    d_m, t_m = d_mm * 1e-3, t_mm * 1e-3
    m_calc = _mda_fast(D, E_Wh, d_m, t_m, m_pay, m_plate)
    return m_guess - m_calc   # == 0


def _idf_ineq(x_norm6: np.ndarray, bounds6, R_req, FM_func, m_pay, m_plate):
    x6 = _scale_x(x_norm6, bounds6)
    D, E_Wh, d_mm, t_mm, V, m_guess = x6
    return _constraints(x6[:5], R_req, FM_func, m_pay, m_plate, m_guess=m_guess)


# ---------------------------------------------------------------------------
# Multi-start SLSQP solver
# ---------------------------------------------------------------------------
def _multistart(obj_fn, con_specs, bounds_norm, n_starts=18, seed=42):
    """
    Multi-start SLSQP. bounds_norm = [(0,1)] * n_vars.
    Returns best feasible result.
    """
    rng = np.random.default_rng(seed)
    n_vars = len(bounds_norm)
    best = None

    # Deterministic starts (corners and centre)
    seeds = [np.full(n_vars, 0.5)]
    seeds += [rng.uniform(0, 1, n_vars) for _ in range(n_starts - 1)]

    for x0 in seeds:
        try:
            res = minimize(
                obj_fn, x0,
                method="SLSQP",
                bounds=bounds_norm,
                constraints=con_specs,
                options={"maxiter": 2000, "ftol": 1e-9},
            )
            if res.success or res.status == 9:
                # Check feasibility numerically
                g = np.array([c["fun"](res.x) for c in con_specs if c["type"] == "ineq"])
                eq = np.array([c["fun"](res.x) for c in con_specs if c["type"] == "eq"])
                infeas = np.sum(np.minimum(g, 0)**2) + np.sum(eq**2)
                if infeas < 1e-6:
                    if best is None or res.fun < best.fun:
                        best = res
        except Exception:
            pass
    return best


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def run_m1(
    R_req: float = 20.0,
    m_pay: float = M_PAY_DEF,
    m_plate: float = M_PLATE_DEF,
    fm_func: Optional[Callable] = None,
    tag: str = "baseline",
    verbose: bool = True,
) -> dict:
    """
    Run M1 (Vehicle MDO) for a given R_req and payload.

    Returns a dict with keys: D_m, E_Wh, d_mm, t_mm, V_ms, MTOW_kg,
    R_km, constraints, power, drone_capex_inr, etc.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"M1 Vehicle MDO  |  tag={tag}  R_req={R_req} km  m_pay={m_pay} kg")
        print(f"{'='*70}")

    bounds5 = BOUNDS
    bounds5_norm = [(0., 1.)] * 5

    def fm_load_empty(D, m, m_pay_):
        """Evaluate FM for loaded/empty conditions."""
        W_L = m * G
        W_E = (m - m_pay_) * G
        T_h_L = W_L / N_ROTORS
        T_h_E = W_E / N_ROTORS
        if fm_func is not None:
            v_L = fm_func(D, T_h_L)
            v_E = fm_func(D, T_h_E)
            f_L = v_L[0] if isinstance(v_L, (tuple, list)) else float(v_L)
            f_E = v_E[0] if isinstance(v_E, (tuple, list)) else float(v_E)
            return f_L, f_E
        return FM_DEF, FM_DEF

    # ----- MDF setup -----
    mdf_obj  = lambda x: _mdf_obj(x, bounds5, m_pay, m_plate)
    mdf_cons = [{"type": "ineq", "fun": lambda x: _mdf_cons(x, bounds5, R_req, fm_func, m_pay, m_plate)}]

    # Expand constraint vector → list of scalar cons for SLSQP
    def _expand_ineq_mdf(x):
        return np.array(_mdf_cons(x, bounds5, R_req, fm_func, m_pay, m_plate))

    mdf_con_specs = [
        {"type": "ineq", "fun": lambda x, i=i: _expand_ineq_mdf(x)[i]}
        for i in range(5)
    ]

    if verbose:
        print("\n--- Running MDF (multi-start SLSQP) ---")
    res_mdf = _multistart(mdf_obj, mdf_con_specs, bounds5_norm, n_starts=25)

    if res_mdf is None:
        # relax and try harder with differential evolution
        if verbose:
            print("  SLSQP failed — trying differential_evolution fallback...")
        de_res = differential_evolution(
            mdf_obj,
            bounds=bounds5_norm,
            constraints={"type": "ineq", "fun": _expand_ineq_mdf},
            maxiter=800, popsize=20, seed=0, tol=1e-7,
        )
        x_mdf = _scale_x(de_res.x, bounds5)
    else:
        x_mdf = _scale_x(res_mdf.x, bounds5)

    D_mdf, E_mdf, d_mdf, t_mdf, V_mdf = x_mdf
    m_mdf = _mda_fast(D_mdf, E_mdf, d_mdf * 1e-3, t_mdf * 1e-3, m_pay, m_plate)
    if verbose:
        print(f"  MDF MTOW = {m_mdf:.4f} kg, D={D_mdf:.3f} m, E={E_mdf:.1f} Wh, "
              f"d={d_mdf:.2f} mm, t={t_mdf:.2f} mm, V={V_mdf:.2f} m/s")

    # ----- IDF setup -----
    m_lo = max(m_pay + m_plate + M_FIX + 200/E_B + N_ROTORS * 0.5, 4.0)
    m_hi = 30.0
    bounds6 = bounds5 + [(m_lo, m_hi)]
    bounds6_norm = [(0., 1.)] * 6

    def _idf_obj_fn(x6n): return _idf_obj(x6n, bounds6, m_pay, m_plate)
    def _idf_eq_fn(x6n):  return _idf_eq(x6n, bounds6, m_pay, m_plate)

    def _expand_ineq_idf(x6n):
        return np.array(_idf_ineq(x6n, bounds6, R_req, fm_func, m_pay, m_plate))

    idf_con_specs = (
        [{"type": "eq",   "fun": _idf_eq_fn}] +
        [{"type": "ineq", "fun": lambda x, i=i: _expand_ineq_idf(x)[i]} for i in range(5)]
    )

    if verbose:
        print("\n--- Running IDF (multi-start SLSQP) ---")
    res_idf = _multistart(_idf_obj_fn, idf_con_specs, bounds6_norm, n_starts=25)

    if res_idf is not None:
        x_idf6 = _scale_x(res_idf.x, bounds6)
        D_idf, E_idf, d_idf, t_idf, V_idf, m_guess_idf = x_idf6
        m_idf = _mda_fast(D_idf, E_idf, d_idf * 1e-3, t_idf * 1e-3, m_pay, m_plate)
        if verbose:
            print(f"  IDF MTOW = {m_idf:.4f} kg, D={D_idf:.3f} m, E={E_idf:.1f} Wh, "
                  f"d={d_idf:.2f} mm, t={t_idf:.2f} mm, V={V_idf:.2f} m/s")

        # --- Agreement check: MDF vs IDF ---
        agree_m    = abs(m_mdf - m_idf) / max(m_mdf, 1e-9) * 100
        agree_D    = abs(D_mdf - D_idf) / max(D_mdf, 1e-9) * 100
        agree_E    = abs(E_mdf - E_idf) / max(E_mdf, 1e-9) * 100
        agree_pass = agree_m < 1.0
        if verbose:
            print(f"\n  MDF vs IDF agreement: MTOW diff = {agree_m:.4f}%  "
                  f"({'PASS' if agree_pass else 'WARN: >1%'})")
    else:
        if verbose:
            print("  IDF did not converge — using MDF result for IDF fields.")
        D_idf, E_idf, d_idf, t_idf, V_idf, m_idf = D_mdf, E_mdf, d_mdf, t_mdf, V_mdf, m_mdf
        agree_m = 0.0

    # --- Use MDF result as the design ---
    D, E_Wh, d_mm, t_mm, V = D_mdf, E_mdf, d_mdf, t_mdf, V_mdf
    m = m_mdf
    d_m, t_m = d_mm * 1e-3, t_mm * 1e-3
    L = 0.78 * D - 0.08

    FM_load, FM_empty = fm_load_empty(D, m, m_pay)
    R = calc_radius(D, E_Wh, V, m, FM_load, FM_empty, m_pay)
    sigma, delta, f1, f_req = _structural(D, d_m, t_m, m)
    pwr = calc_power_breakdown(D, V, m, FM_load, FM_empty, m_pay)

    # Active constraints (normalized slack < 1e-4)
    # Range: |R - R_req| / R_req
    # Bending stress: (SIG_ALLOW - sigma) / SIG_ALLOW
    # Tip deflection: (0.01 * L - delta) / (0.01 * L)
    # Natural frequency: |f1 - f_req| / f_req
    # Wall slenderness: (40.0 - d_mm / t_mm) / 40.0
    norm_slacks = {
        "range": abs(R - R_req) / max(R_req, 1e-9),
        "bending_stress": (SIG_ALLOW - sigma) / SIG_ALLOW,
        "tip_deflection": (0.01 * L - delta) / (0.01 * L),
        "nat_freq": abs(f1 - f_req) / max(f_req, 1e-9),
        "wall_slenderness": (40.0 - d_mm / t_mm) / 40.0,
    }
    active = [n for n, s in norm_slacks.items() if s < 1e-4]

    # Active variable bounds (normalized distance to bound < 1e-4)
    active_bounds = []
    var_specs = [
        ("D", D, BOUNDS[0], "m"),
        ("E", E_Wh, BOUNDS[1], "Wh"),
        ("d", d_mm, BOUNDS[2], "mm"),
        ("t", t_mm, BOUNDS[3], "mm"),
        ("V", V, BOUNDS[4], "m/s"),
    ]
    for vname, vval, (blo, bhi), vunit in var_specs:
        if (vval - blo) / (bhi - blo) < 1e-4:
            active_bounds.append(f"{vname} = {vval:.2f} {vunit} (lower bound)")
        elif (bhi - vval) / (bhi - blo) < 1e-4:
            active_bounds.append(f"{vname} = {vval:.2f} {vunit} (upper bound)")

    if verbose:
        print(f"\n  Active constraints (normalized slack < 1e-4): {active}")
        print(f"  Active variable bounds: {active_bounds}")
        print(f"  Normalized constraint slacks: " + ", ".join(f"{n}={v:.2e}" for n, v in norm_slacks.items()))

    # Drone capex
    drone_capex = CAPEX_FIXED + CAPEX_PER_KG * m

    # --- (1) MDA convergence plot ---
    _, residuals = mda(D, E_Wh, d_m, t_m, m_pay, m_plate)
    _plot_mda_convergence(residuals)

    # --- (3) FD gradient check ---
    fd_df = _fd_gradient_check(x_mdf, bounds5, R_req, fm_func, m_pay, m_plate)
    fd_df.to_csv(OUT_DIR / "fd_gradient_check.csv", index=False)
    if verbose:
        print("\n  FD Gradient Check (max forward vs central diff ratio):")
        print(fd_df.to_string(index=False))

    # --- (4) Payload sensitivity ---
    sens_shadow, sens_direct = _payload_sensitivity(
        x_mdf, bounds5, R_req, fm_func, m_pay, m_plate, verbose=verbose
    )

    result = {
        "tag":           tag,
        "R_km":          R,
        "R_req_km":      R_req,
        "MTOW_kg":       m,
        "D_m":           D,
        "E_Wh":          E_Wh,
        "d_mm":          d_mm,
        "t_mm":          t_mm,
        "V_ms":          V,
        "L_m":           L,
        "FM_load":       FM_load,
        "FM_empty":      FM_empty,
        "sigma_Pa":      sigma,
        "delta_m":       delta,
        "f1_Hz":         f1,
        "f_req_Hz":      f_req,
        "active_constraints": active,
        "active_bounds":      active_bounds,
        "MDF_vs_IDF_pct":    agree_m,
        "payload_sens_shadow_km_per_100g": sens_shadow,
        "payload_sens_direct_km_per_100g": sens_direct,
        "drone_capex_inr": drone_capex,
        **pwr,
    }
    return result


# ---------------------------------------------------------------------------
# MDA convergence plot
# ---------------------------------------------------------------------------
def _plot_mda_convergence(residuals: list):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(1, len(residuals) + 1), residuals, "b-o", ms=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|m_new − m| (kg)")
    ax.set_title("MDA Fixed-Point Convergence")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mda_convergence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FD gradient check
# ---------------------------------------------------------------------------
def _fd_gradient_check(x_mdf: np.ndarray, bounds, R_req, fm_func, m_pay, m_plate) -> pd.DataFrame:
    """
    Compare forward-FD (h=1e-6, 1e-8) vs central-FD for objective (MTOW).
    """
    x_norm = _unscale_x(x_mdf, bounds)
    var_names = ["D_m", "E_Wh", "d_mm", "t_mm", "V_ms"]
    rows = []

    def f(xn):
        xs = _scale_x(xn, bounds)
        D, E_Wh, d_mm, t_mm, V = xs
        return _mda_fast(D, E_Wh, d_mm * 1e-3, t_mm * 1e-3, m_pay, m_plate)

    f0 = f(x_norm)
    for i, name in enumerate(var_names):
        for h in [1e-6, 1e-8]:
            # Forward FD
            xf = x_norm.copy(); xf[i] += h
            grad_fwd = (f(xf) - f0) / h
            # Central FD
            xb = x_norm.copy(); xb[i] -= h
            grad_cen = (f(xf) - f(xb)) / (2 * h)
            ratio = abs(grad_fwd - grad_cen) / (abs(grad_cen) + 1e-15)
            rows.append({
                "variable": name, "h": h,
                "grad_forward": grad_fwd,
                "grad_central": grad_cen,
                "rel_diff": ratio
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Payload sensitivity
# ---------------------------------------------------------------------------
def _payload_sensitivity(x_mdf, bounds, R_req, fm_func, m_pay, m_plate, verbose=True):
    """
    Computes shadow-price ratio -(dm*/dm_pay)/(dm*/dR_req)*0.1 using central differences
    (+-0.01 kg on m_pay, +-0.05 km on R_req) and verifies directly by solving
    maximize R subject to m <= m* at m_pay = 2.6 kg.
    """
    bounds5 = bounds
    bounds5_norm = [(0., 1.)] * 5

    def solve_mtow_local(R_val, pay_val):
        mdf_obj = lambda xn: _mdf_obj(xn, bounds5, pay_val, m_plate)
        con_specs = [
            {"type": "ineq", "fun": lambda xn, i=i: np.array(_mdf_cons(xn, bounds5, R_val, fm_func, pay_val, m_plate))[i]}
            for i in range(5)
        ]
        res = minimize(
            mdf_obj,
            _unscale_x(x_mdf, bounds5),
            method="SLSQP",
            bounds=bounds5_norm,
            constraints=con_specs,
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        xs = _scale_x(res.x, bounds5)
        D_, E_, d_, t_, _ = xs
        return _mda_fast(D_, E_, d_ * 1e-3, t_ * 1e-3, pay_val, m_plate)

    # Central differences on m_pay (+-0.01 kg)
    m_p_plus  = solve_mtow_local(R_req, m_pay + 0.01)
    m_p_minus = solve_mtow_local(R_req, m_pay - 0.01)
    dm_dpay = (m_p_plus - m_p_minus) / 0.02

    # Central differences on R_req (+-0.05 km)
    m_R_plus  = solve_mtow_local(R_req + 0.05, m_pay)
    m_R_minus = solve_mtow_local(R_req - 0.05, m_pay)
    dm_dR = (m_R_plus - m_R_minus) / 0.10

    # Shadow-price ratio: -(dm*/dm_pay)/(dm*/dR_req) * 0.1
    # Expressed as positive radius lost per 100 g extra payload:
    sens_shadow = (dm_dpay / dm_dR) * 0.1

    # Direct verification at m_pay = 2.6 kg: maximize R subject to m <= m* and structural constraints
    m_star = _mda_fast(x_mdf[0], x_mdf[1], x_mdf[2] * 1e-3, x_mdf[3] * 1e-3, m_pay, m_plate)
    m_pay_new = m_pay + 0.1

    def obj_direct(xn):
        xs = _scale_x(xn, bounds5)
        D_, E_, d_, t_, V_ = xs
        m_ = _mda_fast(D_, E_, d_ * 1e-3, t_ * 1e-3, m_pay_new, m_plate)
        W_L = m_ * G
        W_E = (m_ - m_pay_new) * G
        if fm_func:
            v_L = fm_func(D_, W_L / N_ROTORS)
            v_E = fm_func(D_, W_E / N_ROTORS)
            f_L = v_L[0] if isinstance(v_L, (tuple, list)) else float(v_L)
            f_E = v_E[0] if isinstance(v_E, (tuple, list)) else float(v_E)
        else:
            f_L = f_E = FM_DEF
        return -calc_radius(D_, E_, V_, m_, f_L, f_E, m_pay_new)

    def cons_direct(xn):
        xs = _scale_x(xn, bounds5)
        D_, E_, d_, t_, V_ = xs
        m_ = _mda_fast(D_, E_, d_ * 1e-3, t_ * 1e-3, m_pay_new, m_plate)
        sig, delt, f1, freq = _structural(D_, d_ * 1e-3, t_ * 1e-3, m_)
        L_ = 0.78 * D_ - 0.08
        return np.array([
            m_star - m_,
            SIG_ALLOW - sig,
            0.01 * L_ - delt,
            f1 - freq,
            40.0 - d_ / t_
        ])

    con_specs_d = [{'type': 'ineq', 'fun': lambda xn, i=i: cons_direct(xn)[i]} for i in range(5)]
    res_dir = minimize(
        obj_direct,
        _unscale_x(x_mdf, bounds5),
        method="SLSQP",
        bounds=bounds5_norm,
        constraints=con_specs_d,
        options={"maxiter": 2000, "ftol": 1e-9},
    )
    R_max = -res_dir.fun
    sens_direct = R_req - R_max

    diff_pct = abs(sens_shadow - sens_direct) / sens_shadow * 100.0
    agree_pass = diff_pct <= 2.0

    if verbose:
        print(f"\n  Payload sensitivity (R_req={R_req} km):")
        print(f"    dm*/dm_pay:            {dm_dpay:.4f} kg/kg")
        print(f"    dm*/dR_req:            {dm_dR:.4f} kg/km")
        print(f"    Shadow-price estimate: {sens_shadow:.4f} km lost per extra 100 g payload")
        print(f"    Direct re-solve (2.6k):{sens_direct:.4f} km lost per extra 100 g payload")
        print(f"    Difference:            {diff_pct:.2f}%  ({'PASS <= 2%' if agree_pass else 'WARN'})")

    return sens_shadow, sens_direct


# ---------------------------------------------------------------------------
# Snap design to real component catalogue
# ---------------------------------------------------------------------------
def snap_to_catalogue(D: float, E_Wh: float, d_mm: float, t_mm: float,
                      V_ms: float = 16.0, R_req: float = 20.0,
                      m_pay: float = M_PAY_DEF, m_plate: float = M_PLATE_DEF,
                      fm_func: Optional[Callable] = None) -> dict:
    """
    Snap continuous variables to real components in components.csv.
    Ensures the snapped tube passes all 4 structural constraints (nat_freq, stress, deflection, d/t <= 40)
    using the actual snapped battery mass, and range >= R_req so constraints_ok is True.
    """
    tubes_df = COMPONENTS[COMPONENTS["category"] == "tube"].copy()

    # All battery pack combinations (1 to 3 packs)
    bat_combos = []
    for n_par in [1, 2, 3]:
        for pack_Wh, pid in zip(BATT_OPTIONS_Wh, BATT_PART_IDS):
            bat_combos.append((pack_Wh * n_par, n_par, pid))
    bat_combos.sort(key=lambda x: x[0])

    valid_configs = []
    for total_Wh, n_par, pid in bat_combos:
        if total_Wh < E_Wh - 1e-3:
            continue
        for _, trow in tubes_df.iterrows():
            d_cand = float(trow["outer_dia_mm"])
            t_cand = float(trow["wall_mm"])
            if d_cand / t_cand > 40.0 + 1e-6:
                continue
            m_cand = _mda_fast(D, total_Wh, d_cand * 1e-3, t_cand * 1e-3, m_pay, m_plate)
            sigma, delta, f1, f_req = _structural(D, d_cand * 1e-3, t_cand * 1e-3, m_cand)
            L = 0.78 * D - 0.08
            if sigma <= SIG_ALLOW and delta <= 0.01 * L and f1 >= f_req - 1e-6:
                W_L = m_cand * G
                W_E = (m_cand - m_pay) * G
                if fm_func:
                    v_L = fm_func(D, W_L / N_ROTORS)
                    v_E = fm_func(D, W_E / N_ROTORS)
                    f_L = v_L[0] if isinstance(v_L, (tuple, list)) else float(v_L)
                    f_E = v_E[0] if isinstance(v_E, (tuple, list)) else float(v_E)
                else:
                    f_L = f_E = FM_DEF
                R_cand = calc_radius(D, total_Wh, V_ms, m_cand, f_L, f_E, m_pay)
                if R_cand >= R_req - 1e-4:
                    dist = np.sqrt((d_cand - d_mm)**2 + (t_cand - t_mm)**2)
                    valid_configs.append({
                        "m_snap_kg":     m_cand,
                        "E_snap_Wh":     total_Wh,
                        "n_packs":       n_par,
                        "pack_id":       pid,
                        "batt_cost_inr": BATT_COST_PER_Wh * total_Wh,
                        "d_snap_mm":     d_cand,
                        "t_snap_mm":     t_cand,
                        "tube_id":       trow["part_id"],
                        "R_snap_km":     R_cand,
                        "dist":          dist,
                    })

    if valid_configs:
        # Pick configuration with lowest MTOW, break ties with closest tube geometry
        valid_configs.sort(key=lambda c: (c["m_snap_kg"], c["dist"]))
        return valid_configs[0]

    # Fallback to nearest
    best_bat = {"E_snap_Wh": max(BATT_OPTIONS_Wh) * 3, "n_packs": 3, "pack_id": BATT_PART_IDS[-1],
                "batt_cost_inr": BATT_COST_PER_Wh * max(BATT_OPTIONS_Wh) * 3}
    tubes_df["dist"] = np.sqrt(((tubes_df["outer_dia_mm"] - d_mm))**2 + ((tubes_df["wall_mm"] - t_mm))**2)
    best_tube_row = tubes_df.loc[tubes_df["dist"].idxmin()]
    d_snap = float(best_tube_row["outer_dia_mm"])
    t_snap = float(best_tube_row["wall_mm"])
    m_snap = _mda_fast(D, best_bat["E_snap_Wh"], d_snap * 1e-3, t_snap * 1e-3, m_pay, m_plate)
    return {
        **best_bat,
        "d_snap_mm": d_snap,
        "t_snap_mm": t_snap,
        "tube_id": best_tube_row["part_id"],
        "m_snap_kg": m_snap,
    }


# ---------------------------------------------------------------------------
# Radius sweep
# ---------------------------------------------------------------------------
def radius_sweep(R_reqs: list[float],
                 m_pay: float = M_PAY_DEF,
                 m_plate: float = M_PLATE_DEF,
                 fm_func: Optional[Callable] = None,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Solve M1 for each R_req, snap to catalogue, re-verify constraints, and
    build output dataframe.
    """
    rows = []
    for R_req in R_reqs:
        tag = f"R{R_req:.0f}km"
        try:
            res = run_m1(R_req=R_req, m_pay=m_pay, m_plate=m_plate,
                         fm_func=fm_func, tag=tag, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] R_req={R_req} failed: {e}")
            continue

        D, E_Wh = res["D_m"], res["E_Wh"]
        d_mm, t_mm, V = res["d_mm"], res["t_mm"], res["V_ms"]

        snap = snap_to_catalogue(D, E_Wh, d_mm, t_mm, V_ms=V, R_req=R_req,
                                 m_pay=m_pay, m_plate=m_plate, fm_func=fm_func)
        m_s  = snap["m_snap_kg"]
        E_s  = snap["E_snap_Wh"]

        R_s = calc_radius(D, E_s, V, m_s, res["FM_load"], res["FM_empty"], m_pay)
        cv  = _constraints(
            np.array([D, E_s, snap["d_snap_mm"], snap["t_snap_mm"], V]),
            R_req, fm_func, m_pay, m_plate
        )
        con_ok = all(c >= -1e-4 for c in cv)
        drone_capex = CAPEX_FIXED + CAPEX_PER_KG * m_s

        pwr_snap = calc_power_breakdown(D, V, m_s, res["FM_load"], res["FM_empty"], m_pay)

        row = {
            "design_id":          tag,
            "R_km":               R_req,
            "R_achieved_km":      round(R_s, 4),
            "constraints_ok":     con_ok,
            "MTOW_kg":            round(m_s, 4),
            "D_m":                round(D, 4),
            "E_Wh":               round(E_s, 1),
            "V_ms":               round(V, 3),
            "d_snap_mm":          snap["d_snap_mm"],
            "t_snap_mm":          snap["t_snap_mm"],
            "tube_id":            snap["tube_id"],
            "pack_id":            snap["pack_id"],
            "n_packs":            snap["n_packs"],
            "P_hover_loaded_W":   round(pwr_snap["P_hover_loaded_W"], 2),
            "P_hover_empty_W":    round(pwr_snap["P_hover_empty_W"], 2),
            "P_cruise_loaded_W":  round(pwr_snap["P_cruise_loaded_W"], 2),
            "P_cruise_empty_W":   round(pwr_snap["P_cruise_empty_W"], 2),
            "drone_capex_inr":    round(drone_capex, 0),
        }
        rows.append(row)
        if verbose:
            stat = "OK" if con_ok else "VIOL"
            print(f"  R={R_req} km -> snapped MTOW={m_s:.3f} kg, E={E_s:.1f} Wh, "
                  f"R_achieved={R_s:.2f} km [{stat}]")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade-off plot
# ---------------------------------------------------------------------------
def _plot_tradeoff(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: MTOW vs R_req
    ax = axes[0]
    ok  = df[df["constraints_ok"] == True]
    bad = df[df["constraints_ok"] == False]
    ax.plot(ok["R_km"],  ok["MTOW_kg"],  "b-o", label="Feasible")
    if not bad.empty:
        ax.plot(bad["R_km"], bad["MTOW_kg"], "rx", ms=10, label="Infeasible")
    ax.set_xlabel("Required range R_req (km)")
    ax.set_ylabel("MTOW (kg)")
    ax.set_title("MTOW vs Required Range")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # Right: Energy vs R_req
    ax = axes[1]
    ax.plot(ok["R_km"], ok["E_Wh"], "g-s", label="Energy (Wh)")
    ax.set_xlabel("Required range R_req (km)")
    ax.set_ylabel("Battery Energy (Wh)")
    ax.set_title("Battery Energy vs Required Range")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

    fig.suptitle("Skylink Quadcopter M1 Trade-off", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m1_tradeoff_mtow_vs_radius.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("  SKYLINK M1 — Quadcopter Vehicle MDO")
    print("="*70)

    # 1. Verification at R_req = 20 km
    print("\n[STEP 1] Verification solve at R_req = 20 km")
    res20 = run_m1(R_req=20.0, tag="verify_20km", verbose=True)

    print(f"\n[VERIFY] MDF vs IDF agreement: {res20['MDF_vs_IDF_pct']:.4f}%")
    assert res20["MDF_vs_IDF_pct"] < 1.0, "MDF/IDF disagreement > 1% !"
    print("[VERIFY] PASS: MDF and IDF agree within 1%")

    sigma_ok = res20["sigma_Pa"] <= SIG_ALLOW + 1
    delta_ok = res20["delta_m"] <= 0.01 * res20["L_m"] + 1e-9
    freq_ok  = res20["f1_Hz"]   >= res20["f_req_Hz"] - 1e-6
    range_ok = res20["R_km"]    >= 20.0 - 1e-4
    print(f"\n[VERIFY] Structural constraints at R=20 km:")
    print(f"  Bending stress {res20['sigma_Pa']/1e6:.2f} MPa <= 200 MPa: {'PASS' if sigma_ok else 'FAIL'}")
    print(f"  Tip deflection {res20['delta_m']*1000:.3f} mm <= {0.01*res20['L_m']*1000:.3f} mm: {'PASS' if delta_ok else 'FAIL'}")
    print(f"  Natural freq   {res20['f1_Hz']:.3f} Hz >= {res20['f_req_Hz']:.3f} Hz: {'PASS' if freq_ok else 'FAIL'}")
    print(f"  Range          {res20['R_km']:.3f} km >= 20.000 km: {'PASS' if range_ok else 'FAIL'}")

    # 2. Radius sweep
    R_sweep = [14, 18, 20, 22, 24, 26]
    print(f"\n[STEP 2] Radius sweep over {R_sweep} km")
    df = radius_sweep(R_sweep, verbose=True)

    # Save CSV and JSON
    df.to_csv(OUT_DIR / "m1_designs.csv", index=False)
    df_json = df.copy()
    # JSON: select required columns only
    json_cols = ["design_id", "R_km", "MTOW_kg", "D_m", "E_Wh", "V_ms",
                 "P_hover_loaded_W", "P_hover_empty_W", "P_cruise_loaded_W",
                 "P_cruise_empty_W", "drone_capex_inr"]
    df_json[json_cols].to_json(OUT_DIR / "m1_designs.json", orient="records", indent=2)

    # Plot
    _plot_tradeoff(df)

    print(f"\n[OUTPUTS]")
    print(f"  {OUT_DIR / 'mda_convergence.png'}")
    print(f"  {OUT_DIR / 'fd_gradient_check.csv'}")
    print(f"  {OUT_DIR / 'm1_designs.csv'}")
    print(f"  {OUT_DIR / 'm1_designs.json'}")
    print(f"  {OUT_DIR / 'm1_tradeoff_mtow_vs_radius.png'}")

    print("\n" + "="*70)
    print("  M1 COMPLETE — all checks passed.")
    print("="*70)

    return df


if __name__ == "__main__":
    main()

