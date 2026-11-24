#!/usr/bin/env python3
"""
m2_plate_simp.py  --  2D SIMP topology optimisation (Q4 plane stress + OC)

Dependencies: numpy, scipy (sparse, sparse.linalg.spsolve, ndimage), matplotlib.

Part 1  MBB-beam verification (3 cases -> mbb_verification.png)
Part 2  SkyLink hub plate (160x160x3 mm, 6061-T6) volume-fraction sweep,
        Pareto plot, topology grid, water-jet outline, JSON + CSV results.

Outputs go to  <project root>/outputs/m2/  where project root is the parent of
the folder containing this script:
    Path(__file__).resolve().parent.parent / "outputs" / "m2"

Conventions
-----------
* Node n = ix*(nely+1) + iy, iy = 0 is the TOP row; physical y points UP.
* Element e = ex*nely + ey (column-major, same as the 88-line code).
* Element dof order is counter-clockwise: BL, BR, TR, TL (x,y per node).
* Q4 stiffness is independent of element size for square elements, so
  KE = thickness * KE_unit(nu) (times E).
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Q4 element, filter, FE assembly (vectorised)
# --------------------------------------------------------------------------
def q4_unit_stiffness(nu):
    """8x8 plane-stress Q4 stiffness for E = 1, thickness = 1 (square element)."""
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    idx = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 0, 7, 6, 5, 4, 3, 2],
                    [2, 7, 0, 5, 6, 3, 4, 1],
                    [3, 6, 5, 0, 7, 2, 1, 4],
                    [4, 5, 6, 7, 0, 1, 2, 3],
                    [5, 4, 3, 2, 1, 0, 7, 6],
                    [6, 3, 4, 1, 2, 7, 0, 5],
                    [7, 2, 1, 4, 3, 6, 5, 0]])
    return k[idx] / (1.0 - nu**2)


def build_density_filter(nelx, nely, rmin, enabled=True):
    """Linear-cone density filter H (CSR) and row sums Hs; identity if disabled."""
    nel = nelx * nely
    if not enabled:
        return sp.identity(nel, format="csr"), np.ones(nel)
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    e = (ex * nely + ey).ravel()
    ex, ey = ex.ravel(), ey.ravel()
    R = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            d = np.hypot(dx, dy)
            if d >= rmin:
                continue
            nx, ny = ex + dx, ey + dy
            ok = (nx >= 0) & (nx < nelx) & (ny >= 0) & (ny < nely)
            rows.append(e[ok])
            cols.append((nx * nely + ny)[ok])
            vals.append(np.full(ok.sum(), rmin - d))
    H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nel, nel)).tocsr()
    return H, np.asarray(H.sum(axis=1)).ravel()


class Q4FE:
    """Vectorised Q4 assembler. The sparse pattern (free dofs only) is built once;
    each iteration only re-sums element data with np.add.reduceat."""

    def __init__(self, nelx, nely, nu, thickness, fixed_dofs, F):
        self.nelx, self.nely = nelx, nely
        self.nel = nelx * nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.KE = thickness * q4_unit_stiffness(nu)
        ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        tl = (ex * (nely + 1) + ey).ravel()          # top-left node of each element
        bl, br, tr = tl + 1, tl + 1 + (nely + 1), tl + (nely + 1)
        nodes = np.stack([bl, br, tr, tl], axis=1)   # CCW
        self.edof = np.stack([2 * nodes, 2 * nodes + 1], axis=2).reshape(self.nel, 8)
        F = F.reshape(self.ndof, -1)
        fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)
        self.F, self.nl = F, F.shape[1]
        self.Ff = np.ascontiguousarray(F[self.free])
        fmap = -np.ones(self.ndof, dtype=np.int64)
        fmap[self.free] = np.arange(self.free.size)
        nf = self.free.size
        ik = fmap[np.repeat(self.edof, 8, axis=1).ravel()]
        jk = fmap[np.tile(self.edof, (1, 8)).ravel()]
        sel = np.flatnonzero((ik >= 0) & (jk >= 0))
        key = jk[sel] * nf + ik[sel]                  # column-major ordering -> CSC
        order = np.argsort(key, kind="stable")
        skey = key[order]
        self.perm = sel[order]
        self.starts = np.flatnonzero(np.r_[True, skey[1:] != skey[:-1]])
        ukey = skey[self.starts]
        self.rows = (ukey % nf).astype(np.int32)
        cols = ukey // nf
        self.indptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=nf))].astype(np.int32)
        self.nf = nf

    def solve(self, Efield):
        """Efield: Young's modulus per element. Returns U (ndof,nl), ce (nel,)."""
        sK = (Efield[:, None] * self.KE.ravel()[None, :]).ravel()
        data = np.add.reduceat(sK[self.perm], self.starts)
        K = sp.csc_matrix((data, self.rows, self.indptr), shape=(self.nf, self.nf))
        # symmetric-pattern ordering is ~2x faster than COLAMD here; one factorisation serves all load cases
        Uf = spsolve(K, self.Ff, permc_spec="MMD_AT_PLUS_A").reshape(self.nf, -1)
        U = np.zeros((self.ndof, self.nl))
        U[self.free] = Uf
        Ue = U[self.edof]                                 # (nel, 8, nl)
        ce = np.einsum("eik,ij,ejk->e", Ue, self.KE, Ue)  # unit-E element energy, summed over cases
        return U, ce

    def compliance(self, Efield):
        U, ce = self.solve(Efield)
        return float(np.sum(self.F * U)), U, ce


# --------------------------------------------------------------------------
# SIMP / OC optimiser
# --------------------------------------------------------------------------
def simp_oc(fe, volfrac, rmin, penal, use_filter, E0, Emin, passive=None,
            maxit=200, tol=0.01, move=0.2, x0=None, verbose=False):
    nel = fe.nel
    H, Hs = build_density_filter(fe.nelx, fe.nely, rmin, use_filter)
    passive = np.zeros(nel, bool) if passive is None else passive
    x = np.full(nel, volfrac) if x0 is None else x0.copy()
    x[passive] = 1.0

    def filt(v):
        out = (H @ v) / Hs
        out[passive] = 1.0
        return out

    dv_phys = np.ones(nel)
    dv_phys[passive] = 0.0
    dv = np.maximum(H @ (dv_phys / Hs), 1e-12)
    change, it, hist = 1.0, 0, []
    while change > tol and it < maxit:
        it += 1
        xPhys = filt(x)
        c, _, ce = fe.compliance(Emin + xPhys**penal * (E0 - Emin))
        dc = -penal * xPhys**(penal - 1) * (E0 - Emin) * ce
        dc[passive] = 0.0
        dc = H @ (dc / Hs)
        hist.append(c)
        # --- optimality criteria update (bisection on the Lagrange multiplier) ---
        l1, l2 = 0.0, 1e9
        ratio = np.sqrt(np.maximum(-dc, 1e-30) / dv)
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-4:
            lm = 0.5 * (l1 + l2)
            xn = np.clip(x * ratio / np.sqrt(lm), np.maximum(x - move, 0.0), np.minimum(x + move, 1.0))
            xn[passive] = 1.0
            if filt(xn).mean() > volfrac:
                l1 = lm
            else:
                l2 = lm
        change = np.max(np.abs(xn - x))
        x = xn
        if verbose and (it % 10 == 0 or it == 1):
            print(f"    it {it:3d}  C = {c:.5e}  vol = {xPhys.mean():.3f}  ch = {change:.3f}")
    return dict(x=x, xPhys=xPhys, c=c, iters=it, hist=hist)


def grayness(xPhys, passive=None):
    """Measure of non-discreteness  4*mean(x(1-x)) in [0,1] (0 = perfectly 0/1)."""
    v = xPhys if passive is None else xPhys[~passive]
    return float(np.mean(4.0 * v * (1.0 - v)))


def to_grid(v, nelx, nely):
    """Element vector -> image array [row = ey (top first), col = ex]."""
    return v.reshape(nelx, nely).T


# --------------------------------------------------------------------------
# PART 1 : MBB verification
# --------------------------------------------------------------------------
def run_mbb():
    nelx, nely, vf, rmin = 60, 20, 0.5, 2.4
    nn = (nelx + 1) * (nely + 1)
    F = np.zeros(2 * nn)
    F[1] = -1.0                                           # unit downward load, top-left node
    fixed = np.r_[2 * np.arange(nely + 1),                # symmetry plane: ux = 0 on left edge
                  2 * nn - 1]                             # roller: uy = 0 at bottom-right node
    fe = Q4FE(nelx, nely, 0.3, 1.0, fixed, F)
    cases = [("(a) Standard SIMP  p=3, filter ON", 3.0, True),
             ("(b) No filter  p=3, filter OFF", 3.0, False),
             ("(c) Linear  p=1, filter ON", 1.0, True)]
    res = []
    fig, axs = plt.subplots(3, 1, figsize=(8, 8.5))
    print("\n=== PART 1: half-MBB verification (60x20, vf=0.5, rmin=2.4) ===")
    for ax, (name, p, filt_on) in zip(axs, cases):
        r = simp_oc(fe, vf, rmin, p, filt_on, 1.0, 1e-9, maxit=200, tol=0.01)
        g = grayness(r["xPhys"])
        res.append((name, r["c"], g, r["iters"]))
        print(f"  {name:<40s} C = {r['c']:10.3f}   grayness = {g:.4f}   iters = {r['iters']}")
        ax.imshow(1 - to_grid(r["xPhys"], nelx, nely), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{name}\nC = {r['c']:.2f}, grayness = {g:.3f}, {r['iters']} it", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Half-MBB beam verification (nelx=60, nely=20, volfrac=0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mbb_verification.png", dpi=150)
    plt.close(fig)
    return res


# --------------------------------------------------------------------------
# PART 2 : SkyLink hub plate
# --------------------------------------------------------------------------
L_MM, T_MM, NEL = 160.0, 3.0, 80
H_MM = L_MM / NEL                                         # 2 mm / element
E0, NU, RHO = 68.9e9, 0.33, 2700.0                        # 6061-T6
EMIN = 1e-9 * E0
RMIN_PLATE = 2.5                                          # elements = 5 mm
F_ARM = 100.0                                             # N per corner per load case (scale-free ratio)
VF_LIST = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
RATIO_LIMIT = 2.5
FASTENER_KG = 0.04
SWEEP_MAXIT, SWEEP_TOL = 60, 0.02                       # warm-started sweep: runtime cap (<60 s total)


def plate_mass(vf):
    return vf * 0.16 * 0.16 * 0.003 * RHO + FASTENER_KG


def build_plate_problem():
    n = NEL
    nn = (n + 1) * (n + 1)

    def node(ix, iy):
        return ix * (n + 1) + iy

    # ---- Boundary conditions -------------------------------------------------
    # Central 40x40 mm battery/hub mounting region, x,y in [60,100] mm:
    #   nodes ix,iy = 30..50 (21x21 nodes) are fixed in BOTH UX and UY.
    ii, jj = np.meshgrid(np.arange(30, 51), np.arange(30, 51), indexing="ij")
    hub_nodes = node(ii, jj).ravel()
    fixed = np.r_[2 * hub_nodes, 2 * hub_nodes + 1]

    # ---- Passive solid elements (rho = 1) ------------------------------------
    # (i) central mounting zone 40x40 mm = elements 30..49 in both directions
    # (ii) four 12x12 mm corner arm-clamp pads = 6x6 elements in each corner
    passive2d = np.zeros((n, n), bool)                    # [ex, ey]
    passive2d[30:50, 30:50] = True
    pads = [(0, 6), (n - 6, n)]
    for xs in pads:
        for ys in pads:
            passive2d[xs[0]:xs[1], ys[0]:ys[1]] = True
    passive = passive2d.ravel()

    # ---- Loads: two load cases, solved together, compliances summed ----------
    # Each corner pad (7x7 nodes) receives F_ARM spread evenly over its nodes
    # (avoids a point-load singularity). Corner (sx,sy) = (+-1,+-1), y points up.
    #   LC1 radial outward pull : f = F_ARM * ( sx, sy)/sqrt2   (arm pull / bending)
    #   LC2 in-plane twist      : f = F_ARM * (-sy, sx)/sqrt2   (CCW tangential shear,
    #                                                            yaw torque, 4*F*r net torque)
    # Both load cases are self-equilibrating in force (sum Fx = sum Fy = 0); the
    # fixed hub region supplies the reaction moment for LC2.
    F = np.zeros((2 * nn, 2))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ixs = np.arange(0, 7) if sx < 0 else np.arange(n - 6, n + 1)
            iys = np.arange(n - 6, n + 1) if sy < 0 else np.arange(0, 7)   # iy=0 is top (y = +L)
            gi, gj = np.meshgrid(ixs, iys, indexing="ij")
            nd = node(gi, gj).ravel()
            w = F_ARM / nd.size / np.sqrt(2.0)
            F[2 * nd, 0] += w * sx
            F[2 * nd + 1, 0] += w * sy
            F[2 * nd, 1] += w * (-sy)
            F[2 * nd + 1, 1] += w * sx
    fe = Q4FE(n, n, NU, T_MM * 1e-3, fixed, F)
    return fe, passive


def morph_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def pad_morph(op, mask, se, r):
    """Binary morphology with edge-replicated padding so the plate border is not eroded."""
    p = np.pad(mask, r + 1, mode="edge")
    return op(p, structure=se)[r + 1:-(r + 1), r + 1:-(r + 1)]


def make_waterjet_mask(xPhys, passive2d_img, up=4, min_width_mm=6.0):
    """Threshold at 0.5 on a 4x upsampled field, close then open with a disk of
    diameter >= min_width_mm, re-add passive regions, drop floating islands."""
    px = H_MM / up                                        # mm per pixel
    r = int(np.ceil(min_width_mm / 2.0 / px))             # disk radius in pixels (6 px = 3 mm)
    se = morph_disk(r)
    fine = ndimage.zoom(xPhys, up, order=1, mode="nearest", grid_mode=True)
    m = fine >= 0.5
    m = pad_morph(ndimage.binary_closing, m, se, r)       # fill gaps/holes narrower than 6 mm
    m = pad_morph(ndimage.binary_opening, m, se, r)       # remove members narrower than 6 mm
    pas = np.repeat(np.repeat(passive2d_img, up, 0), up, 1)
    m = m | pas                                           # passive blocks are >= 12 mm wide
    lab, nlab = ndimage.label(m)
    keep = np.unique(lab[pas & (lab > 0)])
    m = np.isin(lab, keep)
    ok = not np.any(pad_morph(ndimage.binary_opening, m, se, r) ^ m)   # opening-invariance == min width
    return m, px, r, ok


def run_plate():
    print("\n=== PART 2: SkyLink hub plate (160x160x3 mm Al 6061-T6, 80x80 mesh) ===")
    fe, passive = build_plate_problem()
    passive_img = to_grid(passive, NEL, NEL)

    # ---- solid baseline ------------------------------------------------------
    C_solid, _, _ = fe.compliance(np.full(fe.nel, E0))
    print(f"  Solid plate compliance (vf=1.0): C_solid = {C_solid:.5e} J  (2 load cases summed)")

    # ---- volume-fraction sweep ----------------------------------------------
    rows, fields = [], []
    x_prev = None
    print(f"  {'vf':>5} {'mass[kg]':>9} {'C[J]':>12} {'C/Csolid':>9} {'gray':>6} {'it':>4}  feasible(<=2.5)")
    for vf in VF_LIST:
        t0 = time.time()
        r = simp_oc(fe, vf, RMIN_PLATE, 3.0, True, E0, EMIN, passive=passive,
                    maxit=SWEEP_MAXIT, tol=SWEEP_TOL, x0=x_prev)
        x_prev = r["x"]
        ratio = r["c"] / C_solid
        feas = ratio <= RATIO_LIMIT
        m = plate_mass(vf)
        g = grayness(r["xPhys"], passive)
        rows.append(dict(volfrac=vf, m_plate_kg=m, compliance=r["c"], compliance_ratio=ratio,
                         feasible=bool(feas), grayness=g, iterations=r["iters"]))
        fields.append(r["xPhys"])
        print(f"  {vf:5.2f} {m:9.4f} {r['c']:12.5e} {ratio:9.3f} {g:6.3f} {r['iters']:4d}  "
              f"{'YES' if feas else 'no'}   ({time.time()-t0:.1f}s)")

    # ---- CSV -----------------------------------------------------------------
    with open(OUT_DIR / "m2_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- selection: lightest design with C/C_solid <= 2.5 --------------------
    feas_idx = [i for i, r in enumerate(rows) if r["feasible"]]
    if feas_idx:
        sel = min(feas_idx, key=lambda i: rows[i]["m_plate_kg"])
    else:
        sel = int(np.argmin([r["compliance_ratio"] for r in rows]))
        print("  WARNING: no design meets C/C_solid <= 2.5; selecting the stiffest.")
    print(f"  Selected: vf = {rows[sel]['volfrac']:.2f}, m = {rows[sel]['m_plate_kg']:.4f} kg, "
          f"C/C_solid = {rows[sel]['compliance_ratio']:.3f}")

    # ---- (a) Pareto plot -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ms = [r["m_plate_kg"] for r in rows]
    cr = [r["compliance_ratio"] for r in rows]
    ax.plot(ms, cr, "-", color="0.6", zorder=1)
    for r in rows:
        ax.scatter(r["m_plate_kg"], r["compliance_ratio"], s=60, zorder=3,
                   color="tab:green" if r["feasible"] else "tab:red")
        ax.annotate(f"vf={r['volfrac']:.2f}", (r["m_plate_kg"], r["compliance_ratio"]),
                    textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax.axhline(RATIO_LIMIT, color="k", ls="--", label=f"C/C_solid = {RATIO_LIMIT}")
    ax.scatter(ms[sel], cr[sel], s=220, facecolors="none", edgecolors="b", lw=2, zorder=4, label="selected")
    ax.set_xlabel("Plate mass incl. 0.04 kg fasteners [kg]")
    ax.set_ylabel("Compliance ratio  C / C_solid")
    ax.set_title("SkyLink hub plate: stiffness-mass Pareto sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plate_sweep_pareto.png", dpi=150)
    plt.close(fig)

    # ---- (b) topology grid ---------------------------------------------------
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.4))
    for k, ax in enumerate(axs.ravel()):
        if k < len(rows):
            r = rows[k]
            ax.imshow(1 - to_grid(fields[k], NEL, NEL), cmap="gray", vmin=0, vmax=1,
                      extent=[0, L_MM, 0, L_MM], interpolation="nearest")
            ax.set_title(f"vf={r['volfrac']:.2f} | {r['m_plate_kg']*1000:.0f} g | "
                         f"C/Cs={r['compliance_ratio']:.2f}" + (" ✓" if r["feasible"] else " ✗"),
                         fontsize=10, color="darkgreen" if r["feasible"] else "darkred")
            ax.set_xticks([0, 80, 160]); ax.set_yticks([0, 80, 160])
        else:
            ax.axis("off")
    fig.suptitle("Optimised density fields (black = aluminium); ✓ = C/C_solid ≤ 2.5", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plate_topologies_grid.png", dpi=130)
    plt.close(fig)

    # ---- (c) water-jet outline -------------------------------------------------
    mask, px, rpx, ok = make_waterjet_mask(to_grid(fields[sel], NEL, NEL), passive_img)
    vol_frac_outline = float(mask.mean())
    # re-check stiffness of the cleaned outline (block-average to FE mesh, linear interpolation)
    up = int(round(H_MM / px))
    xb = mask.reshape(NEL, up, NEL, up).mean(axis=(1, 3))          # [ey, ex]
    c_out, _, _ = fe.compliance(EMIN + xb.T.ravel() * (E0 - EMIN))
    ratio_out = c_out / C_solid
    print(f"  Water-jet outline: area fraction = {vol_frac_outline:.3f}, "
          f"min-width test (opening-invariant) = {'PASS' if ok else 'FAIL'}, "
          f"re-analysed C/C_solid = {ratio_out:.3f}")

    ext = [0, L_MM, 0, L_MM]
    xs = (np.arange(mask.shape[1]) + 0.5) * px
    ys = (np.arange(mask.shape[0]) + 0.5) * px
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.contourf(xs, ys, np.flipud(mask).astype(float), levels=[0.5, 1.5], colors=["#c9d3dc"])
    ax.contour(xs, ys, np.flipud(mask).astype(float), levels=[0.5], colors="k", linewidths=1.6)
    ax.add_patch(plt.Rectangle((60, 60), 40, 40, fill=False, ls="--", ec="tab:red", lw=1.2))
    for cx in (0, L_MM - 12):
        for cy in (0, L_MM - 12):
            ax.add_patch(plt.Rectangle((cx, cy), 12, 12, fill=False, ls="--", ec="tab:blue", lw=1.2))
    ax.plot([], [], "--", c="tab:red", label="40x40 mm hub mount (fixed)")
    ax.plot([], [], "--", c="tab:blue", label="12x12 mm arm-clamp pads")
    ax.set_xlim(0, L_MM); ax.set_ylim(0, L_MM); ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Water-jet outline  (vf={rows[sel]['volfrac']:.2f}, {rows[sel]['m_plate_kg']*1000:.0f} g, "
                 f"min member ≥ 6 mm)", fontsize=10)
    ax.legend(loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plate_waterjet_outline.png", dpi=170)
    plt.close(fig)

    # ---- (d) JSON --------------------------------------------------------------
    result = dict(
        selected_vf=rows[sel]["volfrac"],
        m_plate_kg=rows[sel]["m_plate_kg"],
        compliance_ratio=rows[sel]["compliance_ratio"],
        feasible=rows[sel]["feasible"],
        ratio_limit=RATIO_LIMIT,
        C_solid_J=C_solid,
        mesh=f"{NEL}x{NEL}", element_mm=H_MM, rmin_elements=RMIN_PLATE, rmin_mm=RMIN_PLATE * H_MM,
        material=dict(E0_Pa=E0, nu=NU, rho_kg_m3=RHO, Emin_Pa=EMIN),
        waterjet_outline=dict(area_fraction=vol_frac_outline, opening_radius_mm=rpx * px,
                              min_width_check_passed=bool(ok), reanalysed_compliance_ratio=ratio_out),
        sweep=rows,
    )
    with open(OUT_DIR / "m2_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    t0 = time.time()
    run_mbb()
    run_plate()
    print(f"\nAll outputs written to {OUT_DIR}\nTotal runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
