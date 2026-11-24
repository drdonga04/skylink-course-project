#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m3_rotor_surrogate.py
=====================
Module M3 - Rotor Figure-of-Merit (FM) surrogate from scarce oracle data.

Design domain
    D_m in [0.35, 0.80] m   (rotor diameter)
    T_N in [8.0, 40.0]  N   (thrust)
    disk loading  T/D^2     (N/m^2) is used as the physical stall/kink indicator.

Usage
-----
    python m3_rotor_surrogate.py                       # placeholder mode (synthetic data)
    python m3_rotor_surrogate.py --oracle_csv fm.csv   # real oracle data (cols: D_m,T_N,FM)
    python m3_rotor_surrogate.py --n_new 7 --fast

If `oracle_csv` is omitted OR the file does not exist, a 16-point synthetic
placeholder dataset is generated (saved to <outputs>/m3/placeholder_oracle_data.csv,
NOT to the path you passed, so it can never be mistaken for real data later) and a loud
"WARNING: RUNNING ON SYNTHETIC PLACEHOLDER DATA" banner is printed.

Outputs (all in  <script_dir>/../outputs/m3/ )
----------------------------------------------
    loocv_comparison.csv                  RMSE / MAE / MaxError / R2 per model, raw + grouped LOOCV
    loocv_predictions.csv                 per-point LOOCV predictions for every model
    fm_surrogate.pkl                      selected surrogate (+ best-GP uncertainty model + metadata)
    parity_plot.png                       LOOCV predicted vs actual, all models
    surrogate_surface_and_uncertainty.png mean FM(D,T) and GP std(D,T) with samples overlaid
    loocv_residuals_map.png               where in (D,T) the LOOCV errors are large
    proposed_next_request.csv             infill batch, headers exactly  D_m,T_N
    proposed_next_request_details.csv     same points + score components (for inspection)
    placeholder_oracle_data.csv           (placeholder mode only)

Programmatic API
----------------
    from m3_rotor_surrogate import predict_fm, propose_next_points
    fm, std = predict_fm(0.6, 25.0)                       # clipped FM and 1-sigma uncertainty
    df_new  = propose_next_points(n_new=7)                # reads fm_surrogate.pkl

The pickle only contains numpy / scikit-learn / scipy objects (no custom classes), so it can
be un-pickled from any script; `predict_fm` here is the clean wrapper around it.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.interpolate import RBFInterpolator  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.stats import qmc, spearmanr  # noqa: E402

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    ConstantKernel as C,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DEFAULT_ORACLE_CSV = Path(__file__).resolve().parent.parent / "datasets" / "oracle_all_30_points.csv"
D_BOUNDS = (0.35, 0.80)  # m
T_BOUNDS = (8.0, 40.0)  # N
LO = np.array([D_BOUNDS[0], T_BOUNDS[0]], dtype=float)
HI = np.array([D_BOUNDS[1], T_BOUNDS[1]], dtype=float)

# Physics-transformed domain bounds: D_m in [0.35, 0.80], DL = T / (pi * D^2 / 4)
DL_MIN = T_BOUNDS[0] / (np.pi * D_BOUNDS[1]**2 / 4.0)  # ~15.9155 N/m^2
DL_MAX = T_BOUNDS[1] / (np.pi * D_BOUNDS[0]**2 / 4.0)  # ~415.7517 N/m^2
LO_DL = np.array([D_BOUNDS[0], DL_MIN], dtype=float)
HI_DL = np.array([D_BOUNDS[1], DL_MAX], dtype=float)

FM_MIN, FM_MAX = 0.0, 1.0  # physical clip for predictions
SEED = 42
DUP_DECIMALS = 9  # rounding used to decide two rows are the same (D_m, T_N)
DETERMINISTIC_TOL = 1e-6  # replicate max-diff below this => oracle treated as deterministic

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"
PKL_NAME = "fm_surrogate.pkl"

SMOOTH_MODELS_FOR_DIAGNOSTICS = ["GP_Matern25", "GP_RBF", "RBF_TPS"]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def scale_inputs(X):
    """Physical (D_m, T_N) -> unit square [0, 1]^2 using the *fixed* domain bounds."""
    return (np.asarray(X, dtype=float) - LO) / (HI - LO)


def unscale_inputs(U):
    return LO + np.asarray(U, dtype=float) * (HI - LO)


def scale_inputs_dl(X):
    """Physical (D_m, DL) -> unit square [0, 1]^2 using fixed bounds."""
    return (np.asarray(X, dtype=float) - LO_DL) / (HI_DL - LO_DL)


def unscale_inputs_dl(U):
    return LO_DL + np.asarray(U, dtype=float) * (HI_DL - LO_DL)


def disk_loading(D, T):
    """T / D^2  [N/m^2] (as specified in the task; not divided by pi/4)."""
    return np.asarray(T, dtype=float) / np.asarray(D, dtype=float) ** 2


def disk_loading_area(D, T):
    """T / (pi * D^2 / 4) [N/m^2] (rotor disc area loading)."""
    return np.asarray(T, dtype=float) / (np.pi * np.asarray(D, dtype=float)**2 / 4.0)


def banner(lines, char="!"):
    width = max(len(s) for s in lines) + 4
    print("\n" + char * width)
    for s in lines:
        print(f"{char} {s.ljust(width - 4)} {char}")
    print(char * width + "\n")


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Data: synthetic placeholder + loader
# --------------------------------------------------------------------------------------
def synthetic_fm_truth(D, T):
    """
    Realistic-ish small-rotor FM landscape (noise-free), used ONLY for the placeholder.

    smooth part : FM rises with diameter and peaks near a design loading (~90 N/m^2)
    stall edge  : above T/D^2 ~ 190 FM falls off linearly (a kink) by ~0.12 -> subtle,
                  localized non-smoothness in the high-thrust / small-diameter corner.
    """
    D = np.asarray(D, dtype=float)
    T = np.asarray(T, dtype=float)
    dl = T / D**2
    dn = (D - D_BOUNDS[0]) / (D_BOUNDS[1] - D_BOUNDS[0])
    smooth = 0.52 + 0.16 * dn - 0.045 * np.log(dl / 90.0) ** 2
    stall = -0.12 * np.clip((dl - 190.0) / 50.0, 0.0, 1.0)
    return smooth + stall


def make_placeholder_dataset(seed=SEED, noise_sd=0.006):
    """16 rows: 4 corners, 2x exact centre replicate, 2 stall-edge probes, 8 LHS points."""
    rng = np.random.default_rng(seed)
    corners = [(0.35, 8.0), (0.80, 8.0), (0.35, 40.0), (0.80, 40.0)]
    center = (0.575, 24.0)
    edge_probes = [(0.40, 31.0), (0.38, 32.0)]  # disk loading ~194 and ~222
    lhs = qmc.LatinHypercube(d=2, seed=seed).random(8)
    lhs = qmc.scale(lhs, [D_BOUNDS[0], T_BOUNDS[0]], [D_BOUNDS[1], T_BOUNDS[1]])
    pts = corners + [center, center] + edge_probes + [tuple(p) for p in lhs]
    X = np.array(pts, dtype=float)
    X[:, 0] = np.round(X[:, 0], 4)
    X[:, 1] = np.round(X[:, 1], 3)
    fm = synthetic_fm_truth(X[:, 0], X[:, 1]) + rng.normal(0.0, noise_sd, len(X))
    fm = np.clip(np.round(fm, 5), 0.05, 0.85)
    df = pd.DataFrame({"D_m": X[:, 0], "T_N": X[:, 1], "FM": fm})
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_oracle_data(oracle_csv):
    """Return (DataFrame[D_m,T_N,FM], is_synthetic)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(oracle_csv).expanduser() if oracle_csv else None
    if path is None or not path.is_file():
        reason = "no oracle_csv given" if path is None else f"file not found: {path}"
        banner(
            [
                "WARNING: RUNNING ON SYNTHETIC PLACEHOLDER DATA",
                f"({reason})",
                "Results below are for pipeline verification ONLY - not physical truth.",
                "Provide a CSV with columns D_m,T_N,FM via --oracle_csv for real use.",
            ]
        )
        df = make_placeholder_dataset()
        out = OUT_DIR / "placeholder_oracle_data.csv"
        df.to_csv(out, index=False)
        print(f"Placeholder dataset ({len(df)} rows) saved to: {out}")
        return df, True

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = {"D_m", "T_N", "FM"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[["D_m", "T_N", "FM"]].apply(pd.to_numeric, errors="coerce")
    n0 = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < n0:
        print(f"NOTE: dropped {n0 - len(df)} rows with missing / non-numeric values.")
    outside = ((df.D_m < D_BOUNDS[0]) | (df.D_m > D_BOUNDS[1]) |
               (df.T_N < T_BOUNDS[0]) | (df.T_N > T_BOUNDS[1]))
    if outside.any():
        print(f"WARNING: {int(outside.sum())} rows lie outside the design domain "
              f"(scaled inputs will fall outside [0, 1]).")
    print(f"Loaded {len(df)} oracle rows from {path}")
    return df, False


# --------------------------------------------------------------------------------------
# 2a. Replicate / noise diagnostics
# --------------------------------------------------------------------------------------
def add_location_ids(df):
    df = df.copy()
    df["_D"] = df["D_m"].round(DUP_DECIMALS)
    df["_T"] = df["T_N"].round(DUP_DECIMALS)
    df["loc_id"] = df.groupby(["_D", "_T"]).ngroup()
    return df.drop(columns=["_D", "_T"])


def replicate_diagnostics(df):
    """Detect exact duplicate (D_m, T_N) rows; decide noisy vs deterministic."""
    section("DIAGNOSTIC 1: duplicate (D_m, T_N) rows -> is the oracle noisy?")
    g = df.groupby("loc_id")
    stats = g.agg(D_m=("D_m", "first"), T_N=("T_N", "first"), n=("FM", "size"),
                  FM_mean=("FM", "mean"), FM_std=("FM", lambda s: s.std(ddof=1)),
                  FM_maxdiff=("FM", lambda s: s.max() - s.min()))
    reps = stats[stats.n > 1]
    out = {"n_replicate_groups": int(len(reps)), "pooled_std": float("nan"),
           "max_diff": float("nan"), "dof": 0, "verdict": "unknown"}
    if reps.empty:
        print("No duplicate (D_m, T_N) rows found -> oracle noise level cannot be "
              "determined from the data (verdict: UNKNOWN).")
        print("Recommendation: include >=1 replicate (e.g. the centre point) in the next "
              "request to tell noise from non-smoothness.")
        return out
    print(reps[["D_m", "T_N", "n", "FM_mean", "FM_std", "FM_maxdiff"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    dof = int((reps.n - 1).sum())
    pooled = math.sqrt(float(((reps.n - 1) * reps.FM_std.fillna(0.0) ** 2).sum()) / max(dof, 1))
    maxdiff = float(reps.FM_maxdiff.max())
    out.update(pooled_std=pooled, max_diff=maxdiff, dof=dof)
    fm_sd = float(df.FM.std(ddof=1))
    print(f"\nReplicate spread:  std (pooled) = {pooled:.6f}   max diff = {maxdiff:.6f}   "
          f"(dof = {dof}; overall FM std = {fm_sd:.4f})")
    if maxdiff <= DETERMINISTIC_TOL:
        out["verdict"] = "deterministic"
        print("CONCLUSION: replicates agree to numerical precision -> oracle looks "
              "DETERMINISTIC (nugget only needed for numerical stability).")
    else:
        out["verdict"] = "noisy"
        print(f"CONCLUSION: replicates DISAGREE -> oracle is NOISY "
              f"(noise sd ~ {pooled:.4f}, ~{100 * pooled / fm_sd:.1f}% of FM spread). "
              f"A WhiteKernel nugget / smoothing is justified.")
    if dof < 5:
        print(f"CAUTION: only {dof} degree(s) of freedom -> noise estimate is indicative only.")
    return out


# --------------------------------------------------------------------------------------
# 3. Surrogate models (all operate on inputs scaled to [0, 1])
#    Each fitted model is described by a plain-python "bundle" dict that holds only
#    sklearn / scipy / numpy objects -> picklable without this script's classes.
# --------------------------------------------------------------------------------------
def build_gp_kernel(kind, cfg):
    ls0, ls_b = [0.4, 0.4], (0.05, 10.0)
    if kind in ("matern52", "matern25"):
        base = Matern(length_scale=ls0, length_scale_bounds=ls_b, nu=2.5)
    elif kind in ("matern32", "matern15"):
        base = Matern(length_scale=ls0, length_scale_bounds=ls_b, nu=1.5)
    elif kind == "rbf":
        base = RBF(length_scale=ls0, length_scale_bounds=ls_b)
    else:
        raise ValueError(kind)
    return (C(1.0, (1e-2, 1e2)) * base +
            WhiteKernel(noise_level=cfg["noise_init"], noise_level_bounds=cfg["noise_bounds"]))


def predict_from_bundle(bundle, U, include_noise=False):
    """Return (mean, std) in original FM units for scaled inputs U (n, 2)."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    kind = bundle["kind"]
    if kind in ("gp", "gp_dl"):
        gp = bundle["gp"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = gp.predict(U, return_std=True)
        if not include_noise:  # sklearn's std includes the WhiteKernel nugget -> remove it
            nug = gp.kernel_.get_params().get("k2__noise_level", 0.0) * bundle["y_std"] ** 2
            sd = np.sqrt(np.clip(sd**2 - nug, 0.0, None))
        return mu, sd
    if kind == "rbf":
        mu = bundle["rbf"](U)
        return mu, np.full(len(U), bundle["const_std"], dtype=float)
    if kind == "poly":
        pipe = bundle["pipe"]
        mu = pipe.predict(U)
        phi = np.column_stack([np.ones(len(U)), pipe.named_steps["poly"].transform(U)])
        q = np.einsum("ij,jk,ik->i", phi, bundle["Ainv"], phi)
        var = bundle["sigma2"] * (q + (1.0 if include_noise else 0.0))
        return mu, np.sqrt(np.clip(var, 0.0, None))
    raise ValueError(kind)


class Surrogate:
    name = "base"

    def __init__(self, name, cfg):
        self.name, self.cfg, self.bundle = name, cfg, None

    def fit(self, U, y):
        raise NotImplementedError

    def predict(self, U, include_noise=False):
        return predict_from_bundle(self.bundle, U, include_noise)


class GPSurrogate(Surrogate):
    def __init__(self, name, cfg, kind):
        super().__init__(name, cfg)
        self.kind = kind

    def fit(self, U, y, n_restarts=None):
        y = np.asarray(y, dtype=float)
        y_std = float(np.std(y))
        y_std = y_std if y_std > 1e-12 else 1.0
        gp = GaussianProcessRegressor(
            kernel=build_gp_kernel(self.kind, self.cfg),
            normalize_y=True, alpha=1e-10,
            n_restarts_optimizer=self.cfg["n_restarts"] if n_restarts is None else n_restarts,
            random_state=self.cfg["seed"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            gp.fit(U, y)
        self.bundle = {"kind": "gp", "gp": gp, "y_mean": float(np.mean(y)),
                       "y_std": y_std, "const_std": float("nan")}
        return self


class GPDLSurrogate(Surrogate):
    """
    Physics-transformed Kriging model:
    Trained on scaled (D_m, DL = T_N / (pi * D_m^2 / 4)) with Matern-1.5 + WhiteKernel.
    """
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.kind = "matern15"

    def fit(self, U, y, n_restarts=None):
        y = np.asarray(y, dtype=float)
        y_std = float(np.std(y))
        y_std = y_std if y_std > 1e-12 else 1.0
        gp = GaussianProcessRegressor(
            kernel=build_gp_kernel(self.kind, self.cfg),
            normalize_y=True, alpha=1e-10,
            n_restarts_optimizer=self.cfg["n_restarts"] if n_restarts is None else n_restarts,
            random_state=self.cfg["seed"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            gp.fit(U, y)
        self.bundle = {
            "kind": "gp_dl",
            "gp": gp,
            "y_mean": float(np.mean(y)),
            "y_std": y_std,
            "const_std": float("nan"),
            "x_lo": LO_DL,
            "x_hi": HI_DL,
        }
        return self


class RBFSurrogate(Surrogate):
    def fit(self, U, y, n_restarts=None):
        rbf = RBFInterpolator(U, np.asarray(y, float), kernel="thin_plate_spline",
                              smoothing=self.cfg["rbf_smoothing"], degree=1)
        self.bundle = {"kind": "rbf", "rbf": rbf, "const_std": float("nan")}
        return self


class PolySurrogate(Surrogate):
    def fit(self, U, y, n_restarts=None):
        y = np.asarray(y, float)
        alpha = self.cfg["ridge_alpha"]
        pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                         ("ridge", Ridge(alpha=alpha))])
        pipe.fit(U, y)
        phi = np.column_stack([np.ones(len(U)), pipe.named_steps["poly"].transform(U)])
        pen = np.eye(phi.shape[1]) * alpha
        pen[0, 0] = 0.0
        Ainv = np.linalg.pinv(phi.T @ phi + pen)
        ssr = float(np.sum((y - pipe.predict(U)) ** 2))
        sigma2 = max(ssr / max(len(y) - phi.shape[1], 1), 1e-12)
        self.bundle = {"kind": "poly", "pipe": pipe, "Ainv": Ainv, "sigma2": sigma2}
        return self


def make_model_factories(cfg, include_matern15=True):
    return {
        "PRS": lambda: PolySurrogate("PRS", cfg),
        "RBF_TPS": lambda: RBFSurrogate("RBF_TPS", cfg),
        "GP_RBF": lambda: GPSurrogate("GP_RBF", cfg, "rbf"),
        "GP_Matern25": lambda: GPSurrogate("GP_Matern25", cfg, "matern25"),
        "GP_Matern15": lambda: GPSurrogate("GP_Matern15", cfg, "matern15"),
        "GP_Matern15_DL": lambda: GPDLSurrogate("GP_Matern15_DL", cfg),
    }


# --------------------------------------------------------------------------------------
# 4. Exact LOOCV (raw rows + replicate-grouped)
# --------------------------------------------------------------------------------------
def regression_metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MaxError": float(np.max(np.abs(err))),
        "R2": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def run_loocv(factory, U, y, loc_id):
    """
    grouped : leave out ALL rows of one (D,T) location (replicates never leak into the
              training fold); compared against the location's mean FM.  -> honest score.
    raw     : classic leave-one-ROW-out; if exact replicates exist the twin stays in the
              training set, so the score is optimistic at those points.
    """
    locs = np.unique(loc_id)
    g_rows = []
    for g in locs:
        te = loc_id == g
        m = factory().fit(U[~te], y[~te])
        p = float(m.predict(U[te][:1])[0][0])
        g_rows.append((int(g), p, float(y[te].mean()), int(te.sum())))
    grouped = pd.DataFrame(g_rows, columns=["loc_id", "pred", "actual", "n_rep"])

    if len(locs) == len(y):  # no replicates -> identical protocols
        raw = grouped.copy()
        raw["row"] = [int(np.where(loc_id == g)[0][0]) for g in raw["loc_id"]]
        raw = raw[["row", "pred", "actual"]]
    else:
        r_rows = []
        for i in range(len(y)):
            mask = np.ones(len(y), bool)
            mask[i] = False
            m = factory().fit(U[mask], y[mask])
            r_rows.append((i, float(m.predict(U[i:i + 1])[0][0]), float(y[i])))
        raw = pd.DataFrame(r_rows, columns=["row", "pred", "actual"])
    return grouped, raw


# --------------------------------------------------------------------------------------
# 2b. LOOCV residual cluster / kink diagnostics
# --------------------------------------------------------------------------------------
def diagnose_loocv_residuals(loc_tbl, noise_sd, link_dist=0.30, z=2.5, ratio_thr=2.0):
    """
    loc_tbl: DataFrame[D_m, T_N, absr]  (absr = mean |LOOCV residual| of the smooth models)

    A localized kink / stall boundary shows up as high LOOCV error that is
      (i)  >=2 high-error points at HIGH disk loading (top quartile of T/D^2), or
      (ii) a spatial cluster (>=2 neighbours) of high-error points that are not merely
           convex-hull vertices.
    Hull vertices at LOW loading are reported but treated as plain extrapolation error
    (leaving out a domain corner always hurts), so they do not trigger the fallback.
    """
    section("DIAGNOSTIC 2: LOOCV residual structure -> localized kinks / stall boundary?")
    U = scale_inputs(loc_tbl[["D_m", "T_N"]].values)
    absr = loc_tbl["absr"].to_numpy()
    dl = disk_loading(loc_tbl["D_m"], loc_tbl["T_N"])
    med = float(np.median(absr))
    mad = 1.4826 * float(np.median(np.abs(absr - med)))
    floor = max(noise_sd if np.isfinite(noise_sd) else 0.0, 1e-3)
    thr = max(med + z * mad, 3.0 * floor)
    high = absr > thr

    on_hull = np.zeros(len(U), bool)
    try:
        on_hull[ConvexHull(U).vertices] = True
    except Exception:  # degenerate geometry
        pass
    top = dl >= np.quantile(dl, 0.75)
    high_eff = high & (~on_hull | top)  # drop low-loading hull vertices (pure extrapolation)

    cluster_size = int(high_eff.sum() == 1)
    if high_eff.sum() >= 2:
        idx = np.where(high_eff)[0]
        adj = csr_matrix(cdist(U[idx], U[idx]) <= link_dist)
        _, lab = connected_components(adj, directed=False)
        cluster_size = int(np.bincount(lab).max())

    n_high_top = int((high & top).sum())
    ratio = float(absr[top].mean() / max(absr[~top].mean(), 1e-12))
    rho, pval = (spearmanr(absr, dl) if len(absr) >= 4 else (float("nan"), float("nan")))
    detected = bool(n_high_top >= 2 or cluster_size >= 2)  # one isolated point = outlier/extrapolation

    # estimated stall-boundary loading: |res|-weighted log-mean of high-error, high-loading points
    dl_boundary = None
    cand = high & top
    if cand.any():
        w = absr[cand]
        dl_boundary = float(np.exp(np.sum(w * np.log(dl[cand])) / np.sum(w)))

    print(f"mean |LOOCV residual| over smooth models: median={med:.4f}  robust-sd={mad:.4f}  "
          f"flag threshold={thr:.4f}")
    if high.any():
        tbl = loc_tbl.loc[high, ["D_m", "T_N", "absr"]].assign(
            DiskLoading=dl[high],
            note=np.where(top[high], "high loading (stall-prone)",
                          np.where(on_hull[high], "hull vertex (extrapolation)", "interior")))
        print("High-error locations:")
        print(tbl.sort_values("absr", ascending=False).to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))
    else:
        print("No location exceeds the high-error threshold.")
    print(f"High-error points at top-quartile disk loading (T/D^2 >= {np.quantile(dl, 0.75):.0f}): "
          f"{n_high_top}")
    print(f"Largest spatial cluster of non-extrapolation high-error points (link<={link_dist}): "
          f"{cluster_size}")
    print(f"Mean |res| top-quartile loading vs rest: ratio = {ratio:.2f}  "
          f"(Spearman rho(|res|, T/D^2) = {rho:.2f}, p = {pval:.2f})  [informational]")
    if detected:
        print("CONCLUSION: localized high-error structure DETECTED -> possible sharp kink / stall "
              "boundary at high disk loading "
              "(indicative only with this few points). "
              "Fallback enabled: Matern-1.5 GP (E_GP_Matern32) added to the comparison.")
    else:
        print("CONCLUSION: no localized high-loading error cluster (isolated points are treated as "
              "outliers / extrapolation) -> no evidence of a kink. Indicative only; "
              "use --force_matern15 to evaluate the Matern-1.5 fallback anyway.")
    return {"detected": detected, "threshold": thr, "ratio_top_loading": ratio,
            "cluster_size": cluster_size, "n_high_top_loading": n_high_top,
            "spearman_rho": float(rho), "dl_boundary": dl_boundary,
            "loc_D": loc_tbl["D_m"].tolist(), "loc_T": loc_tbl["T_N"].tolist(),
            "absr": absr.tolist(), "high": high.tolist()}


# --------------------------------------------------------------------------------------
# Public wrapper: predict_fm
# --------------------------------------------------------------------------------------
_STATE_CACHE = {}


def load_surrogate(model_path=None):
    path = Path(model_path) if model_path else OUT_DIR / PKL_NAME
    key = str(path.resolve())
    if key not in _STATE_CACHE:
        with open(path, "rb") as fh:
            _STATE_CACHE[key] = pickle.load(fh)
    return _STATE_CACHE[key]


def predict_fm(D_m, T_N, model_path=None, include_noise=False):
    """
    Predict rotor figure of merit.

    Parameters
    ----------
    D_m, T_N : float or array-like (broadcast against each other)
    include_noise : if True the std also contains the oracle-noise nugget.

    Returns
    -------
    (FM, std) : FM clipped to [0, 1]; std = 1-sigma model uncertainty (same shape as input;
                Python floats for scalar input).  For RBF/quadratic winners the std comes from
                LOOCV RMSE / regression theory instead of a GP posterior.
    """
    state = load_surrogate(model_path)
    D, T = np.broadcast_arrays(np.asarray(D_m, float), np.asarray(T_N, float))
    shape = D.shape
    bundle = state["model_bundle"]
    if bundle.get("kind") == "gp_dl":
        D_flat = D.ravel()
        T_flat = T.ravel()
        DL_flat = T_flat / (np.pi * D_flat**2 / 4.0)
        X_trans = np.column_stack([D_flat, DL_flat])
        x_lo = bundle.get("x_lo", LO_DL)
        x_hi = bundle.get("x_hi", HI_DL)
        U = (X_trans - x_lo) / (x_hi - x_lo)
        mu, sd = predict_from_bundle(bundle, U, include_noise)
    else:
        X = np.column_stack([D.ravel(), T.ravel()])
        if np.any(X < state["x_lo"] - 1e-9) or np.any(X > state["x_hi"] + 1e-9):
            warnings.warn("predict_fm: inputs outside the training domain -> extrapolation.")
        U = (X - state["x_lo"]) / (state["x_hi"] - state["x_lo"])
        mu, sd = predict_from_bundle(bundle, U, include_noise)
    fm = np.clip(mu, *state["fm_clip"])
    if not shape:
        return float(fm[0]), float(sd[0])
    return fm.reshape(shape), sd.reshape(shape)


# --------------------------------------------------------------------------------------
# 5. Adaptive infill recommender
# --------------------------------------------------------------------------------------
def _frozen_gp(kernel, U, y_norm):
    """GP with fixed (already optimised) hyper-parameters; y already normalised."""
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, alpha=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(U, y_norm)
    return gp


def propose_next_points(n_new=7, target_D_range=(0.45, 0.80), target_T_range=(12.0, 35.0),
                        *, state=None, model_path=None, out_csv=None, n_grid=61,
                        min_sep=0.07, w_res=0.6, w_bnd=0.5, verbose=True):
    """
    Greedy batch infill with the Kriging-Believer strategy.

    Score(x) = std_latent(x)/std_max                         (max-variance exploration)
             + w_res * localized |LOOCV residual| bump       (chase where the model was wrong)
             + w_bnd * Gaussian band in ln(T/D^2) around the estimated stall boundary
    The residual/boundary bonuses are discounted near points already chosen in the batch, a hard
    minimum separation (scaled units) is enforced, and after each pick the GP is refit with the
    pick "believed" at its posterior mean (kernel hyper-parameters frozen) so variance collapses
    around it.  Writes proposed_next_request.csv with headers exactly  D_m,T_N .
    """
    if n_new < 1:
        raise ValueError("n_new must be >= 1")
    state = state if state is not None else load_surrogate(model_path)
    out_csv = Path(out_csv) if out_csv else OUT_DIR / "proposed_next_request.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    d0, d1 = max(target_D_range[0], D_BOUNDS[0]), min(target_D_range[1], D_BOUNDS[1])
    t0, t1 = max(target_T_range[0], T_BOUNDS[0]), min(target_T_range[1], T_BOUNDS[1])
    Dg, Tg = np.meshgrid(np.linspace(d0, d1, n_grid), np.linspace(t0, t1, n_grid))
    cand_raw = np.column_stack([Dg.ravel(), Tg.ravel()])
    cand_U = scale_inputs(cand_raw)
    cand_dl = disk_loading(cand_raw[:, 0], cand_raw[:, 1])

    tr = state["train"]
    U_tr = scale_inputs(np.column_stack([tr["D_m"], tr["T_N"]]))
    gb = state["uncertainty_bundle"]
    is_dl = gb.get("kind") == "gp_dl"
    kernel = gb["gp"].kernel_
    y_mean, y_std = gb["y_mean"], gb["y_std"]
    y_norm = (np.asarray(tr["FM"], float) - y_mean) / y_std
    nug = kernel.get_params().get("k2__noise_level", 0.0)  # in normalised units

    if is_dl:
        cand_dl_area = cand_raw[:, 1] / (np.pi * cand_raw[:, 0]**2 / 4.0)
        cand_U_gp = scale_inputs_dl(np.column_stack([cand_raw[:, 0], cand_dl_area]))
        tr_dl_area = np.array(tr["T_N"]) / (np.pi * np.array(tr["D_m"])**2 / 4.0)
        U_tr_gp = scale_inputs_dl(np.column_stack([tr["D_m"], tr_dl_area]))
    else:
        cand_U_gp = cand_U
        U_tr_gp = U_tr

    def latent(gp, U):
        mu, sd = gp.predict(U, return_std=True)
        return mu, np.sqrt(np.clip(sd**2 - nug, 0.0, None))

    # residual bump field
    kd = state["kink_diag"]
    loc_U = scale_inputs(np.column_stack([kd["loc_D"], kd["loc_T"]]))
    absr = np.asarray(kd["absr"], float)
    absr_n = absr / max(absr.max(), 1e-12)
    d_loc = cdist(cand_U, loc_U)
    res_field = np.max(absr_n[None, :] * np.exp(-0.5 * (d_loc / 0.15) ** 2), axis=1)

    # boundary band
    dl_b = kd.get("dl_boundary")
    if dl_b is None:
        dl_b = float(np.quantile(cand_dl, 0.75))
    dl_b_raw = dl_b
    dl_b = float(np.clip(dl_b, cand_dl.min(), cand_dl.max()))
    bnd_field = np.exp(-0.5 * ((np.log(cand_dl) - np.log(dl_b)) / 0.20) ** 2)

    gp = _frozen_gp(kernel, U_tr_gp, y_norm)
    _, sd0 = latent(gp, cand_U_gp)
    sd_ref = max(float(sd0.max()), 1e-12)

    U_aug_gp, y_aug = U_tr_gp.copy(), y_norm.copy()
    U_aug_cand = U_tr.copy()
    chosen_U, rows = [], []
    for k in range(n_new):
        mu_n, sd_n = latent(gp, cand_U_gp)
        std_term = sd_n / sd_ref
        if chosen_U:
            dch = cdist(cand_U, np.array(chosen_U))
            disc = np.exp(-np.sum(np.exp(-((dch / 0.15) ** 2)), axis=1))
        else:
            disc = np.ones(len(cand_U))
        score = std_term + w_res * res_field * disc + w_bnd * bnd_field * disc
        dmin = cdist(cand_U, U_aug_cand).min(axis=1)
        score = np.where(dmin >= min_sep, score, -np.inf)
        if not np.isfinite(score).any():  # relax separation if the region is saturated
            score = std_term + w_res * res_field * disc + w_bnd * bnd_field * disc
        j = int(np.argmax(score))
        comps = {"explore_std": std_term[j], "resid_bonus": w_res * res_field[j] * disc[j],
                 "boundary_bonus": w_bnd * bnd_field[j] * disc[j]}
        reason = max(comps, key=comps.get)
        rows.append({"rank": k + 1, "D_m": round(float(cand_raw[j, 0]), 3),
                     "T_N": round(float(cand_raw[j, 1]), 2),
                     "disk_loading": float(cand_dl[j]), "score": float(score[j]),
                     **{kk: float(vv) for kk, vv in comps.items()}, "main_driver": reason})
        chosen_U.append(cand_U[j])
        U_aug_gp = np.vstack([U_aug_gp, cand_U_gp[j:j + 1]])
        U_aug_cand = np.vstack([U_aug_cand, cand_U[j:j + 1]])
        y_aug = np.append(y_aug, mu_n[j])  # Kriging believer: hallucinate the posterior mean
        gp = _frozen_gp(kernel, U_aug_gp, y_aug)

    details = pd.DataFrame(rows)
    out = details[["D_m", "T_N"]].copy()
    out.to_csv(out_csv, index=False, columns=["D_m", "T_N"])
    details.to_csv(out_csv.with_name("proposed_next_request_details.csv"), index=False)
    if verbose:
        section(f"INFILL RECOMMENDER: {n_new} points (Kriging Believer) in "
                f"D in [{d0:.2f},{d1:.2f}], T in [{t0:.1f},{t1:.1f}]")
        src = "from LOOCV high-error points" if kd.get("dl_boundary") else "fallback: 75th pct of target region"
        clip_note = (f", clipped to the target region's range [{cand_dl.min():.0f}, {cand_dl.max():.0f}]"
                     if abs(dl_b - dl_b_raw) > 1e-9 else "")
        print(f"Boundary-exploration band centre: T/D^2 ~ {dl_b:.0f}  "
              f"(estimate {dl_b_raw:.0f} {src}{clip_note})")
        print(details.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nSaved: {out_csv}")
    return out


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def _loading_lines(ax, color="0.35", levels=(50, 100, 150, 200, 250)):
    Dg = np.linspace(*D_BOUNDS, 200)
    for lv in levels:
        T = lv * Dg**2
        m = (T >= T_BOUNDS[0]) & (T <= T_BOUNDS[1])
        if m.sum() > 2:
            ax.plot(Dg[m], T[m], ls=":", lw=0.9, color=color, alpha=0.8)
            i = np.where(m)[0][-1] if lv < 100 else np.where(m)[0][len(np.where(m)[0]) // 2]
            ax.annotate(f"T/D²={lv}", (Dg[i], T[i]), fontsize=7, color=color,
                        xytext=(2, 3), textcoords="offset points")
    pad_d, pad_t = 0.02 * (D_BOUNDS[1] - D_BOUNDS[0]), 0.02 * (T_BOUNDS[1] - T_BOUNDS[0])
    ax.set_xlim(D_BOUNDS[0] - pad_d, D_BOUNDS[1] + pad_d)  # margin so corner samples are visible
    ax.set_ylim(T_BOUNDS[0] - pad_t, T_BOUNDS[1] + pad_t)


def _suptitle_suffix(synthetic):
    return "  [SYNTHETIC PLACEHOLDER DATA]" if synthetic else ""


def plot_parity(grouped_preds, comp, loc_info, selected, synthetic, path):
    names = list(grouped_preds)
    ncols = 3
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.7 * nrows), squeeze=False)
    allv = np.concatenate([np.r_[p.actual.values, p.pred.values] for p in grouped_preds.values()])
    lo, hi = allv.min() - 0.03, allv.max() + 0.03
    dl = disk_loading(loc_info.D_m, loc_info.T_N)
    sc = None
    for ax, name in zip(axes.ravel(), names):
        p = grouped_preds[name]
        sc = ax.scatter(p.actual, p.pred, c=dl, cmap="plasma", edgecolor="k", s=60, zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        r = comp[(comp.model == name) & (comp.protocol == "grouped")].iloc[0]
        ax.set_title(name + ("  ★ selected" if name == selected else ""), fontsize=10,
                     fontweight="bold" if name == selected else "normal")
        ax.text(0.04, 0.96, f"RMSE={r.RMSE:.4f}\nMAE={r.MAE:.4f}\nmax={r.MaxError:.4f}\nR²={r.R2:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.85))
        ax.set_xlabel("Actual FM (oracle)")
        ax.set_ylabel("LOOCV predicted FM")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="Disk loading T/D² [N/m²]")
    fig.suptitle("LOOCV parity (leave-one-location-out, replicates grouped)"
                 + _suptitle_suffix(synthetic), fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_surface(state, df, proposed, target_box, synthetic, path):
    ng = 140
    Dg, Tg = np.meshgrid(np.linspace(*D_BOUNDS, ng), np.linspace(*T_BOUNDS, ng))
    U_raw = scale_inputs(np.column_stack([Dg.ravel(), Tg.ravel()]))
    DL_grid = Tg / (np.pi * Dg**2 / 4.0)
    U_dl = scale_inputs_dl(np.column_stack([Dg.ravel(), DL_grid.ravel()]))

    U_m = U_dl if state["model_bundle"].get("kind") == "gp_dl" else U_raw
    mu, _ = predict_from_bundle(state["model_bundle"], U_m)
    mu = np.clip(mu, FM_MIN, FM_MAX).reshape(Dg.shape)

    U_u = U_dl if state["uncertainty_bundle"].get("kind") == "gp_dl" else U_raw
    _, sd = predict_from_bundle(state["uncertainty_bundle"], U_u)
    sd = sd.reshape(Dg.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    uniq = df.groupby("loc_id").agg(D_m=("D_m", "first"), T_N=("T_N", "first"), n=("FM", "size"))
    panels = [(axes[0], mu, "viridis", f"Mean FM(D, T) - {state['selected_model']}", "FM [-]"),
              (axes[1], sd, "magma", f"GP uncertainty std(D, T) - {state['uncertainty_model']}",
               "latent std of FM [-]")]
    for ax, Z, cmap, title, lab in panels:
        cf = ax.contourf(Dg, Tg, Z, levels=22, cmap=cmap)
        ax.contour(Dg, Tg, Z, levels=10, colors="k", linewidths=0.3, alpha=0.5)
        fig.colorbar(cf, ax=ax, label=lab)
        _loading_lines(ax, color="w")
        ax.scatter(uniq.D_m, uniq.T_N, c="w", edgecolor="k", s=55, zorder=5, label="oracle samples")
        rep = uniq[uniq.n > 1]
        if len(rep):
            ax.scatter(rep.D_m, rep.T_N, s=170, facecolors="none", edgecolors="cyan", lw=1.8,
                       zorder=6, label="exact replicate")
        if proposed is not None and len(proposed):
            ax.scatter(proposed.D_m, proposed.T_N, marker="D", c="red", edgecolor="w", s=48,
                       zorder=7, label="proposed next request")
            for i, (d, t) in enumerate(zip(proposed.D_m, proposed.T_N), 1):
                ax.annotate(str(i), (d, t), color="w", fontsize=7, xytext=(4, 4),
                            textcoords="offset points", zorder=8)
        if target_box is not None:
            (a0, a1), (b0, b1) = target_box
            ax.plot([a0, a1, a1, a0, a0], [b0, b0, b1, b1, b0], "r--", lw=0.9, alpha=0.8)
        ax.set_xlabel("Diameter D [m]")
        ax.set_ylabel("Thrust T [N]")
        ax.set_title(title, fontsize=10)
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("Rotor FM surrogate" + _suptitle_suffix(synthetic), fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_map(loc_info, sel_grouped, selected, kink, synthetic, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    res = (sel_grouped["pred"] - sel_grouped["actual"]).to_numpy()
    v = max(np.abs(res).max(), 1e-6)
    ax = axes[0]
    sc = ax.scatter(loc_info.D_m, loc_info.T_N, c=res, cmap="RdBu_r", vmin=-v, vmax=v,
                    s=60 + 700 * np.abs(res) / v, edgecolor="k", zorder=4)
    fig.colorbar(sc, ax=ax, label="LOOCV residual (pred - actual)")
    ax.set_title(f"Signed LOOCV residual - {selected}", fontsize=10)

    ax = axes[1]
    absr, high = np.array(kink["absr"]), np.array(kink["high"], bool)
    sc2 = ax.scatter(loc_info.D_m, loc_info.T_N, c=absr, cmap="YlOrRd", s=60 + 700 * absr / max(absr.max(), 1e-9),
                     edgecolor="k", zorder=4)
    fig.colorbar(sc2, ax=ax, label="mean |LOOCV residual| (smooth models A-C)")
    if high.any():
        ax.scatter(loc_info.D_m[high], loc_info.T_N[high], s=330, facecolors="none",
                   edgecolors="blue", lw=2, zorder=5, label="flagged high-error")
        for d, t, a in zip(loc_info.D_m[high], loc_info.T_N[high], absr[high]):
            ax.annotate(f"{a:.3f}", (d, t), fontsize=8, xytext=(6, -10), textcoords="offset points")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Localized-error screen (kink / stall detection): "
                 + ("DETECTED" if kink["detected"] else "not detected"), fontsize=10)
    for ax in axes:
        _loading_lines(ax, color="0.4")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Diameter D [m]")
        ax.set_ylabel("Thrust T [N]")
    fig.suptitle("LOOCV residual map (leave-one-location-out)" + _suptitle_suffix(synthetic), fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def compare_m1_shift(model_path=None, out_csv=None, verbose=True):
    """
    Run m1_vehicle_mdo.run_m1 with FM0=0.65 vs predict_fm at R_req=20 km,
    save comparison to outputs/m3/m1_fm_shift_comparison.csv and print summary.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import m1_vehicle_mdo
    except ImportError as e:
        print(f"[ERROR] Cannot import m1_vehicle_mdo: {e}")
        return None

    if verbose:
        section("M1 SHIFT COMPARISON (Fixed FM=0.65 vs Rotor Surrogate predict_fm at R_req=20 km)")

    res_base = m1_vehicle_mdo.run_m1(R_req=20.0, fm_func=None, tag="FM_fixed_0.65", verbose=False)

    def fm_surrogate_func(D, T):
        fm, std = predict_fm(D, T, model_path=model_path)
        return float(fm)

    res_surr = m1_vehicle_mdo.run_m1(R_req=20.0, fm_func=fm_surrogate_func, tag="FM_surrogate", verbose=False)

    metrics = [
        ("MTOW_kg", "MTOW (kg)"),
        ("D_m", "Rotor Diameter D (m)"),
        ("E_Wh", "Battery Energy E (Wh)"),
        ("d_mm", "Tube Diameter d (mm)"),
        ("t_mm", "Tube Thickness t (mm)"),
        ("V_ms", "Cruise Speed V (m/s)"),
        ("FM_load", "Figure of Merit (Hover Loaded)"),
        ("FM_empty", "Figure of Merit (Hover Empty)"),
        ("P_hover_loaded_W", "Hover Power Loaded (W)"),
        ("P_hover_empty_W", "Hover Power Empty (W)"),
        ("P_cruise_loaded_W", "Cruise Power Loaded (W)"),
        ("P_cruise_empty_W", "Cruise Power Empty (W)"),
        ("drone_capex_inr", "Drone Capex (INR)"),
    ]

    rows = []
    for k, label in metrics:
        v_b = res_base.get(k, np.nan)
        v_s = res_surr.get(k, np.nan)
        diff = v_s - v_b
        pct = (diff / v_b) * 100.0 if abs(v_b) > 1e-9 else np.nan
        rows.append({
            "parameter": label,
            "metric_key": k,
            "FM_fixed_0.65": v_b,
            "FM_surrogate": v_s,
            "delta_abs": diff,
            "delta_pct": pct,
        })

    df_comp = pd.DataFrame(rows)
    out_path = Path(out_csv) if out_csv else OUT_DIR / "m1_fm_shift_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(out_path, index=False)

    if verbose:
        print(f"Comparison saved to: {out_path}\n")
        print(df_comp[["parameter", "FM_fixed_0.65", "FM_surrogate", "delta_abs", "delta_pct"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        ))

    return df_comp


def run_pipeline(oracle_csv=None, n_new=7, fast=False, force_matern15=False):
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    if oracle_csv is None and DEFAULT_ORACLE_CSV.is_file():
        oracle_csv = str(DEFAULT_ORACLE_CSV)

    df, synthetic = load_oracle_data(oracle_csv)
    df = add_location_ids(df)
    if df.loc_id.nunique() < 6:
        raise ValueError("Need at least 6 distinct (D_m, T_N) locations to fit the models.")
    section("DATA SUMMARY")
    print(f"{len(df)} rows, {df.loc_id.nunique()} unique locations"
          + ("   [SYNTHETIC PLACEHOLDER]" if synthetic else ""))
    print(df[["D_m", "T_N", "FM"]].describe().loc[["min", "max", "mean", "std"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    U_raw = scale_inputs(df[["D_m", "T_N"]].values)
    DL_vals = df["T_N"].values / (np.pi * df["D_m"].values**2 / 4.0)
    U_dl = scale_inputs_dl(np.column_stack([df["D_m"].values, DL_vals]))

    y = df["FM"].to_numpy(float)
    loc_id = df["loc_id"].to_numpy()
    loc_info = (df.groupby("loc_id").agg(D_m=("D_m", "first"), T_N=("T_N", "first"),
                                          FM=("FM", "mean"), n=("FM", "size")).reset_index())

    # ---- diagnostics 1 ----
    rep = replicate_diagnostics(df)

    cfg = {"seed": SEED, "n_restarts": 2 if fast else 5, "final_restarts": 6 if fast else 15,
           "rbf_smoothing": 1e-3, "ridge_alpha": 1e-4,
           "noise_init": 1e-4 if rep["verdict"] == "deterministic" else 1e-2,
           "noise_bounds": (1e-6, 1e-2) if rep["verdict"] == "deterministic" else (1e-6, 1.0)}

    # ---- LOOCV on all candidate models ----
    section("EXACT LOOCV (refit on every fold, incl. hyper-parameter optimisation)")
    factories = make_model_factories(cfg)
    grouped, raw = {}, {}
    for name, fac in factories.items():
        t0 = time.time()
        U_in = U_dl if name == "GP_Matern15_DL" else U_raw
        grouped[name], raw[name] = run_loocv(fac, U_in, y, loc_id)
        print(f"  {name:<16s} done in {time.time() - t0:5.1f}s")

    # ---- diagnostics 2 -> localized error / kink detection ----
    absr = np.mean([np.abs(grouped[m]["pred"] - grouped[m]["actual"]).to_numpy()
                    for m in SMOOTH_MODELS_FOR_DIAGNOSTICS], axis=0)
    loc_tbl = loc_info[["D_m", "T_N"]].assign(absr=absr)
    kink = diagnose_loocv_residuals(loc_tbl, rep["pooled_std"])

    # ---- comparison table ----
    recs, pred_recs = [], []
    for name in factories:
        for proto, tab in (("grouped", grouped[name]), ("raw", raw[name])):
            met = regression_metrics(tab["actual"], tab["pred"])
            recs.append({"model": name, "protocol": proto, "n_eval": len(tab), **met})
        g = grouped[name].merge(loc_info[["loc_id", "D_m", "T_N"]], on="loc_id")
        for _, r in g.iterrows():
            pred_recs.append({"model": name, "protocol": "grouped", "D_m": r.D_m, "T_N": r.T_N,
                              "FM_actual": r.actual, "FM_pred": r.pred, "residual": r.pred - r.actual,
                              "n_rep": int(r.n_rep)})
        for _, r in raw[name].iterrows():
            i = int(r["row"])
            pred_recs.append({"model": name, "protocol": "raw", "D_m": df.D_m[i], "T_N": df.T_N[i],
                              "FM_actual": r.actual, "FM_pred": r.pred, "residual": r.pred - r.actual,
                              "n_rep": 1})
    comp = pd.DataFrame(recs)
    best = comp[comp.protocol == "grouped"].sort_values("RMSE").iloc[0]["model"]
    comp["selected"] = comp.model == best
    comp_sorted = comp.sort_values(["protocol", "RMSE"], ascending=[True, True])
    comp_sorted.to_csv(OUT_DIR / "loocv_comparison.csv", index=False)
    pd.DataFrame(pred_recs).to_csv(OUT_DIR / "loocv_predictions.csv", index=False)

    section("LOOCV COMPARISON  (grouped = replicates held out together; raw = leave-one-row-out)")
    print(comp_sorted.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    has_dups = len(loc_info) < len(df)
    if not has_dups:
        print("(no replicates -> raw and grouped protocols are identical)")
    print(f"\nSELECTED (lowest grouped-LOOCV RMSE): {best}")

    # ---- final fits ----
    section("FINAL FIT ON ALL DATA")
    gp_names = [m for m in factories if m.startswith("GP_")]
    best_gp = (comp[(comp.protocol == "grouped") & comp.model.isin(gp_names)]
               .sort_values("RMSE").iloc[0]["model"])
    fitted = {}
    for name in {best, best_gp}:
        U_in = U_dl if name == "GP_Matern15_DL" else U_raw
        fitted[name] = factories[name]().fit(U_in, y, n_restarts=cfg["final_restarts"])
    sel, gpm = fitted[best], fitted[best_gp]
    sel_rmse = float(comp[(comp.model == best) & (comp.protocol == "grouped")].RMSE.iloc[0])
    if sel.bundle["kind"] == "rbf":
        sel.bundle["const_std"] = sel_rmse  # RBF has no posterior -> use LOOCV RMSE
    print(f"Selected model : {best}")
    if gpm.bundle["kind"] in ("gp", "gp_dl"):
        k = gpm.bundle["gp"].kernel_
        nug_sd = math.sqrt(k.get_params()["k2__noise_level"]) * gpm.bundle["y_std"]
        print(f"Uncertainty GP : {best_gp}\n  kernel = {k}")
        print(f"  learned nugget sd = {nug_sd:.5f}"
              + (f"  vs replicate pooled std = {rep['pooled_std']:.5f}" if np.isfinite(rep["pooled_std"]) else ""))

    state = {
        "format_version": 1, "module": "M3_rotor_FM_surrogate",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_placeholder": bool(synthetic),
        "selected_model": best, "model_bundle": sel.bundle, "selected_loocv_rmse": sel_rmse,
        "uncertainty_model": best_gp, "uncertainty_bundle": gpm.bundle,
        "x_lo": LO, "x_hi": HI, "D_bounds": D_BOUNDS, "T_bounds": T_BOUNDS,
        "fm_clip": (FM_MIN, FM_MAX),
        "train": {"D_m": df.D_m.tolist(), "T_N": df.T_N.tolist(), "FM": df.FM.tolist()},
        "loocv": comp_sorted.to_dict("records"), "replicate_diag": rep, "kink_diag": kink,
    }
    with open(OUT_DIR / PKL_NAME, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _STATE_CACHE.clear()
    print(f"Saved surrogate: {OUT_DIR / PKL_NAME}")

    # ---- infill ----
    target_D, target_T = (0.45, 0.80), (12.0, 35.0)
    proposed = propose_next_points(n_new=n_new, target_D_range=target_D, target_T_range=target_T,
                                   state=state)

    # ---- plots ----
    plot_parity({m: grouped[m] for m in factories}, comp, loc_info, best, synthetic,
                OUT_DIR / "parity_plot.png")
    plot_surface(state, df, proposed, (target_D, target_T), synthetic,
                 OUT_DIR / "surrogate_surface_and_uncertainty.png")
    plot_residual_map(loc_info, grouped[best], best, kink, synthetic,
                      OUT_DIR / "loocv_residuals_map.png")

    # ---- wrapper sanity check ----
    section("predict_fm() WRAPPER CHECK (loaded from pickle)")
    for d, t in [(0.575, 24.0), (0.45, 30.0), (0.80, 12.0)]:
        fm, sd = predict_fm(d, t)
        print(f"  predict_fm(D={d:.3f}, T={t:5.1f}) -> FM = {fm:.4f} +/- {sd:.4f}   "
              f"(T/D^2 = {t / d**2:.0f})")

    # ---- M1 Shift Comparison ----
    compare_m1_shift(model_path=OUT_DIR / PKL_NAME, verbose=True)

    section("DONE")
    if synthetic:
        banner(["REMINDER: THIS RUN USED SYNTHETIC PLACEHOLDER DATA - NOT REAL ORACLE RESULTS"])
    for f in ["loocv_comparison.csv", "loocv_predictions.csv", PKL_NAME, "parity_plot.png",
              "surrogate_surface_and_uncertainty.png", "loocv_residuals_map.png",
              "proposed_next_request.csv", "proposed_next_request_details.csv",
              "m1_fm_shift_comparison.csv"]:
        print(f"  {OUT_DIR / f}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description="M3 rotor FM surrogate from scarce oracle data")
    ap.add_argument("--oracle_csv", default=str(DEFAULT_ORACLE_CSV) if DEFAULT_ORACLE_CSV.is_file() else None,
                    help="CSV with columns D_m,T_N,FM (defaults to datasets/oracle_all_30_points.csv)")
    ap.add_argument("--n_new", type=int, default=7, help="size of the proposed infill batch")
    ap.add_argument("--fast", action="store_true", help="fewer GP optimiser restarts")
    ap.add_argument("--force_matern15", action="store_true",
                    help="always include the Matern-1.5 fallback model")
    args, _ = ap.parse_known_args(argv)
    run_pipeline(args.oracle_csv, n_new=args.n_new, fast=args.fast,
                 force_matern15=args.force_matern15)


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
