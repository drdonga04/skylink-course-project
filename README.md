# SkyLink: Autonomous Medical Drone Delivery Network (ED6002)

**SkyLink** is an integrated multidisciplinary optimization, simulation, and decision framework for an autonomous medical drone logistics network serving 24 Primary Health Centres (PHCs) across a challenging rural district with no-fly airspace zones, variable monsoon winds, scheduled grid outages, and urgent emergency response constraints.

---

## Table of Contents
1. [Architecture & Pipeline Overview](#architecture--pipeline-overview)
2. [Module Descriptions & Verification Highlights](#module-descriptions--verification-highlights)
3. [Repository Structure](#repository-structure)
4. [Installation & Requirements](#installation--requirements)
5. [Quickstart: Running the Pipeline](#quickstart-running-the-pipeline)
6. [Itemized Cost & Simulation Audit](#itemized-cost--simulation-audit)
7. [Verification & Validation Matrix](#verification--validation-matrix)

---

## Architecture & Pipeline Overview

The project is structured as a sequential, multi-disciplinary engineering design pipeline where each module feeds high-fidelity physics and operational data to downstream stages:

```
[Raw Datasets] (7 files in datasets/)
       │
       ▼
 [00_data_audit.py] ──> Data audit & consistency verification
       │
       ▼
 [M1 Baseline MDO] (FM=0.65, m_plate=0.25 kg)
       │
       ▼
 [M2 Plate SIMP] ──> FEA topology optimization (250g -> 165g)
       │
       ▼
 [M1 Re-run] (with optimized M2 plate mass)
       │
       ▼
 [M3 Rotor Surrogate] ──> 30-pt Oracle Kriging (GP_Matern15_DL wins LOOCV)
       │
       ▼
 [M1 Final Integrated MDO] ──> Snaps sweep to catalog, exports m1_designs.json
       │
       ▼
 [M4 Network MILP] ──> Obstacle bypass visibility graph + district demand sizing
       │
       ▼
 [M5 Simulator] ──> SimPy DES, Erlang-C M/M/3, 120-day CI (<±3pp), itemized cost audit
       │
       ▼
 [M6 Sim-Opt] ──> 55 LHS sweep (20 CRN days) + NSGA-II + 120-day rechecks (Top 3)
       │
       ▼
 [M7 Robustness] ──> Wind regimes, 100 MC trials, sensitivity, decision matrix
```

---

## Module Descriptions & Verification Highlights

### Stage 1: Data Audit (`src/00_data_audit.py`)
- Reads all 7 datasets relative to the script: `sites.csv`, `demand.csv`, `hourly_profile.csv`, `wind_days.csv`, `costs.yaml`, `components.csv`, and `nofly.geojson`.
- Audits and verifies:
  - Column sums of `emergency_frac`, `routine_frac`, `lab_frac` in `hourly_profile.csv` sum to $1.0 \pm 10^{-3}$.
  - Site matching: 24 PHCs, 8 candidate hubs (H0–H6, P09), noting P09 serves as both PHC and hub.
  - No-fly polygon closure and confirms no site lies inside restricted airspace.
  - CFRP tubes ($d/t \le 40$, density $1550\text{ kg/m}^3$) and motor scaling law ($0.12 + 0.004 \times T_{\text{max}}$).
  - Wind dataset: 8,760 hourly records, non-negative values, flyable hours.

### Stage 2 & 4 & 6: Vehicle MDO (`src/m1_vehicle_mdo.py`)
- **Physics**: Coupled multidisciplinary design analysis (MDA) fixed-point iteration until $|\Delta m| < 10^{-10}\text{ kg}$.
- **Structural Constraints**: Cantilever beam under maximum thrust $T_{\text{max}} = 2mg/n$:
  - Bending stress $\sigma \le 200\text{ MPa}$.
  - Tip deflection $\delta \le 0.01 L$.
  - First bending natural frequency $f_1 \ge 3 \times f_{\text{bladepass}}$ ($f_{\text{bladepass}} = 2 \times V_{\text{tip}} / (\pi D)$).
  - Tube thin-walled buckling constraint $d/t \le 40$.
- **Active Constraint Detection**: Checked with normalized slack $< 10^{-4}$. At $R_{\text{req}} = 20\text{ km}$, active constraints are strictly `['range', 'nat_freq']`; tip deflection has $>80\%$ margin and is non-active. Active bounds reported: $D = 0.80\text{ m}, d = 40.0\text{ mm}$.
- **Payload Sensitivity**: Dual verification:
  - Central-difference shadow-price ratio: $-(\partial m^*/\partial m_{\text{pay}}) / (\partial m^*/\partial R_{\text{req}}) \times 0.1 = 0.3430\text{ km lost per 100 g}$.
  - Direct re-solve with fixed optimal MTOW at $m_{\text{pay}} = 2.6\text{ kg}$: $0.3449\text{ km lost per 100 g}$.
  - Both methods agree within $0.56\% \le 2.0\%$.
- **Catalog Snapping**: All sweep designs ($14\text{ to }26\text{ km}$) snapped to real components in `components.csv` with `constraints_ok = True`.

### Stage 3: Center Plate Topology Optimization (`src/m2_plate_simp.py`)
- Solid Isotropic Material with Penalisation (SIMP) FEA on an $80 \times 80$ mesh ($160 \times 160 \times 3\text{ mm}$ Aluminium 6061-T6).
- Quadcopter load cases: arm shear and moment couplings around the central $40 \times 40\text{ mm}$ electronics mount.
- Reduces plate mass from default $250\text{ g}$ to $\approx 165.7\text{ g}$ while satisfying compliance ratio $C/C_{\text{solid}} \le 2.5$ and minimum manufacturing member width $\ge 6.0\text{ mm}$ for water-jet cutting.

### Stage 5: Rotor Aerodynamic Surrogate (`src/m3_rotor_surrogate.py`)
- Trains on scarce 30-point Oracle CFD rotor dataset (`datasets/oracle_all_30_points.csv`).
- Evaluates 6 surrogate formulations via grouped Leave-One-Out Cross-Validation (LOOCV):
  1. Polynomial Response Surface (PRS - Quadratic)
  2. Radial Basis Function (RBF Thin-Plate Spline)
  3. Gaussian Process RBF
  4. Gaussian Process Matérn-2.5
  5. Gaussian Process Matérn-1.5 (Raw $(D_m, T_N)$)
  6. **Physics-Transformed Kriging (`GP_Matern15_DL`)**: Input feature transformed to $(D_m, \text{Disk Loading} = T / (\pi D^2 / 4))$.
- **Winner**: `GP_Matern15_DL` wins LOOCV with grouped $R^2 = 0.9270$ and $\text{RMSE} = 0.0322$.
- Exports `fm_surrogate.pkl` and evaluates MTOW shift ($+7.28\%$ at $20\text{ km}$ due to realistic rotor stall at high disk loading).

### Stage 7: Network Obstacle Bypass & MILP (`src/m4_network_milp.py`)
- **Obstacle Bypass Visibility Graph**: Inflates no-fly polygons by $+0.05\text{ km}$ buffer; uses Dijkstra's algorithm to compute exact obstacle-avoiding shortest paths $L_{ij}$ between all 8 candidate hubs and 24 PHCs.
- **Demand Integration**: Uses exact daily demand $\lambda_i = \text{emergency} + \text{routine} + \text{lab}$ from `demand.csv` ($195.66\text{ orders/day}$ district-wide).
- **Energy & Wear**: Evaluates route-specific sortie energy $E_{\text{sortie}}(i, j)$ with $90\%$ charger efficiency and 500-cycle fractional battery wear.
- **Optimal Sizing**: Selects 3 open hubs (`H1`, `H2`, `H4`), sizes a fleet of 8 drones (`H1: 3, H2: 2, H4: 3`) and 5 charging bays, achieving zero unreachable PHCs at annual cost $\approx ₹12.68\text{M}$.

### Stage 8: Discrete-Event Simulator (`src/m5_simulator.py`)
- **Analytical Verification**: Validates M/M/3 queueing against exact Erlang-C formulas; all traffic intensities ($\rho = 0.5, 0.7, 0.85$) pass within $<10\%$ relative error.
- **Sample-Size Convergence**: Stratified sampling demonstrates $N = 120\text{ days}$ is required to achieve a $95\%$ confidence interval half-width $< \pm 3.0$ percentage points on $S_{45}$.
- **Itemized Cost Audit**: Itemizes all 10 annual cost components in exact accordance with `costs.yaml`, verifying that `one_battery_pack_included: true` is respected (only spare packs $s_j$ are charged as capex).
- **Fallback Root-Cause Audit**: Decomposes road fallback into 4 distinct categories:
  1. Gust grounding ($\ge 14.0\text{ m/s}$)
  2. Headwind range limit ($w \ge 0.95 V_{\text{ms}}$ or flight singularity)
  3. Emergency wait $> 30\text{ min}$ (resource congestion)
  4. Day-end unserved routine/lab orders
- Generates live dashboard snapshot and 24-hour animated operational GIF (`live_day_dashboard.gif`).

### Stage 9: Simulation-Optimization (`src/m6_sim_opt.py`)
- **Surrogate Training Sweep**: Evaluates 55 Latin Hypercube Sampling (LHS) designs across 20 stratified Common Random Number (CRN) days.
- **Multi-Objective Surrogates**: Fits Gaussian Process models for `annual_cost_inr`, `S45`, `road_fallback_share`, and `em_p95_min`.
- **Pareto Optimization**: Runs NSGA-II to extract the 3-objective Pareto front.
- **Verification & Re-Evaluation**:
  - Re-checks 7 weighted-sum solutions confirming they lie on or behind the NSGA-II front.
  - Re-simulates the 15 Pareto representative points and exported Top 3 candidates on the **full 120 stratified days** to guarantee tight confidence intervals ($< \pm 3\text{ pp}$).

### Stage 10: Robustness, Reliability & Decision Analysis (`src/m7_robustness.py`)
- **Wind-Week Regimes**: Evaluates performance across 3 distinct 7-day rolling windows from `wind_days.csv`: Calm Week, Typical Week, and Worst Monsoon Week.
- **Monte Carlo Reliability**: 100 trials with randomized demand multipliers ($0.75\text{--}1.25$) and drone availability ($0.90\text{--}0.98$).
- **Stress Sensitivity**: Tests battery energy density ($+20\%$), drone capex ($-30\%$), and annual demand growth ($+15\%/\text{year}$ over 1, 3, and 5 years).
- **Decision Matrix**: Computes multi-criteria rankings under 3 rules:
  1. Knee Point (minimum normalized distance to Utopia)
  2. Minimax Regret
  3. Best Service within Budget

---

## Repository Structure

```
Skylink Course Project/
├── datasets/                        # Raw, immutable problem datasets
│   ├── components.csv               # Batteries, CFRP tubes, motors, avionics
│   ├── costs.yaml                   # Economic parameters & operating rules
│   ├── demand.csv                   # Daily order demand per PHC
│   ├── hourly_profile.csv           # Diurnal distribution of order arrivals
│   ├── nofly.geojson                # Restricted airspace polygons
│   ├── oracle_all_30_points.csv     # 30-point CFD rotor oracle evaluations
│   ├── sites.csv                    # Geographic coordinates of PHCs and hubs
│   └── wind_days.csv                # 8,760 hourly wind speed & gust records
├── src/                             # Core engineering & optimization source code
│   ├── 00_data_audit.py             # Data integrity validation
│   ├── data_loader.py               # Shared immutable data loader
│   ├── m1_vehicle_mdo.py            # Quadcopter vehicle MDO (MDF/IDF)
│   ├── m2_plate_simp.py             # Center plate topology optimization (SIMP)
│   ├── m3_rotor_surrogate.py        # Rotor aerodynamic surrogate modeling
│   ├── m4_network_milp.py           # Shortest bypass path & network MILP
│   ├── m5_simulator.py              # Discrete-event simulation & cost audit
│   ├── m6_sim_opt.py                # Surrogate-assisted multi-objective sim-opt
│   └── m7_robustness.py             # Robustness, reliability & decision analysis
├── outputs/                         # Generated artifacts, figures, and summaries
│   ├── data_audit_report.txt
│   ├── m1/                          # Vehicle MDO plots, summaries, designs.json
│   ├── m2/                          # Density topologies, waterjet mask, results.json
│   ├── m3/                          # Surrogate surface, LOOCV parity, fm_surrogate.pkl
│   ├── m4/                          # Obstacle bypass paths, solutions.json
│   ├── m5/                          # Live dashboard GIF, CI convergence, summary
│   ├── m6/                          # Pareto front, top 3 candidates.json
│   └── m7/                          # Wind robustness, MC reliability, decision matrix
├── run_all.py                       # Master end-to-end execution pipeline
└── README.md                        # Project documentation
```

---

## Installation & Requirements

The project runs on standard Python 3.10+ (tested on Python 3.11).

```bash
# Clone or navigate to the workspace
cd "Skylink Course Project"

# Install required dependencies
pip install numpy scipy pandas shapely matplotlib pyyaml scikit-learn simpy pymoo pulp
```

---

## Quickstart: Running the Pipeline

To run the complete verified chain end-to-end:

```bash
python run_all.py
```

This executes all 10 stages in order and prints an aggregated verification table upon completion.

### Running Individual Modules
```bash
python src/00_data_audit.py      # Run data audit
python src/m1_vehicle_mdo.py     # Run vehicle MDO
python src/m2_plate_simp.py      # Run center plate SIMP
python src/m3_rotor_surrogate.py # Run rotor surrogate pipeline
python src/m4_network_milp.py    # Run network MILP
python src/m5_simulator.py       # Run DES simulator & cost audit
python src/m6_sim_opt.py         # Run multi-objective sim-opt
python src/m7_robustness.py      # Run robustness & decision rules
```

---

## Itemized Cost & Simulation Audit

Under `outputs/m5/simulation_summary.csv`, the annualized cost is audited across 10 distinct categories:

| Cost Item | Description | Accounting Basis |
|---|---|---|
| **Hubs** | Landing pad, shelter, genset, communications | Capex amortized over 10 yr + annual opex |
| **Staffing** | 3 shifts/day, 1.2 relief factor, 1 operator per 4 drones | ₹420,000 / person-year |
| **Drones** | Fixed + MTOW-dependent capex + 15% annual maintenance | Capex amortized over 3 yr (1 battery pack included) |
| **Charging Bays** | 1.5 kW chargers with 90% charge efficiency | Capex amortized over 5 yr |
| **Spare Batteries** | Dedicated spare packs bought separately | Capex amortized over 3 yr drone life |
| **Grid Energy** | Regular electric charging | ₹9.0 / kWh delivered |
| **Genset Energy** | Outage operation (Mon/Wed/Fri 14:00–17:00) | ₹28.0 / kWh delivered |
| **Battery Wear** | Replacement packs for cycle degradation | 500-cycle life throughput wear |
| **Road Emergency** | Diverted emergency deliveries (120 min) | ₹2,500 / trip |
| **Road Routine/Lab** | Diverted routine/lab deliveries (180 min) | ₹500 / trip |

---

## Verification & Validation Matrix

| Stage | Module | Status | Verification Criteria & Key Findings |
|---|---|:---:|---|
| **1** | `00_audit` | **PASS** | 7 raw datasets verified; column sums $= 1.0 \pm 10^{-3}$; no site in no-fly zone |
| **2** | `M1_base` | **PASS** | MDF/IDF diff $< 0.01\%$; active constraints: `['range', 'nat_freq']`; shadow price agree within $0.56\%$ |
| **3** | `M2_plate` | **PASS** | Plate mass reduced from $250\text{ g}$ to $\approx 165.7\text{ g}$; $C/C_{\text{solid}} = 2.05 \le 2.5$; width $\ge 6\text{ mm}$ |
| **4** | `M1_plate` | **PASS** | Vehicle MDO re-converged with M2 mass; MTOW reduced |
| **5** | `M3_rotor` | **PASS** | Physics-transformed `GP_Matern15_DL` wins LOOCV ($R^2 = 0.9270$, $\text{RMSE} = 0.0322$) |
| **6** | `M1_final` | **PASS** | Final sweep ($14\text{ to }26\text{ km}$) snapped to real catalog with all 4 structural constraints passed |
| **7** | `M4_milp` | **PASS** | Visibility graph bypasses no-fly zones; solves district demand ($195.66\text{ orders/day}$); sizes 3 hubs & 8 drones |
| **8** | `M5_sim` | **PASS** | Erlang-C verified ($<10\%$ err); $N=120\text{ days}$ meets $\pm 3\text{ pp}$ CI; itemized audit & fallback decomposed |
| **9** | `M6_opt` | **PASS** | 55 LHS designs + NSGA-II Pareto front + 15 points re-evaluated on 120 stratified days |
| **10** | `M7_robust` | **PASS** | Evaluated on Calm, Typical & Monsoon weeks; 100 MC trials; decision matrix selected winner |
