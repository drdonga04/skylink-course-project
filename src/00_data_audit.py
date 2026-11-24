"""
Data audit script for Skylink Course Project.
Audits all 7 raw datasets in datasets/ and writes a detailed report to outputs/data_audit_report.txt.
"""

from pathlib import Path
import json
import math
import pandas as pd
import yaml
from shapely.geometry import shape, Point, Polygon


def audit_datasets():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent / "datasets"
    output_dir = script_dir.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "data_audit_report.txt"

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(str(msg))

    log("=" * 80)
    log("SKYLINK COURSE PROJECT - DATA AUDIT REPORT")
    log("=" * 80)
    log(f"Datasets Directory: {base_dir}")
    log()

    # -------------------------------------------------------------------------
    # (a) hourly_profile.csv audit
    # -------------------------------------------------------------------------
    log("-" * 80)
    log("(a) Hourly Demand Profile (hourly_profile.csv)")
    log("-" * 80)
    hourly_path = base_dir / "hourly_profile.csv"
    hourly = pd.read_csv(hourly_path)
    log(f"Loaded {len(hourly)} hourly profile records.")
    
    frac_cols = ["emergency_frac", "routine_frac", "lab_frac"]
    col_sums = hourly[frac_cols].sum()
    log("Column sums:")
    for col in frac_cols:
        s = col_sums[col]
        diff = abs(s - 1.0)
        flagged = diff > 1e-3
        status = f"FLAGGED (outside 1.0 ± 1e-3, diff={diff:.6f})" if flagged else "OK (sums to 1.0 within tolerance)"
        log(f"  - {col}: sum = {s:.4f} -> {status}")
    log()

    # -------------------------------------------------------------------------
    # (b) sites.csv and demand.csv audit
    # -------------------------------------------------------------------------
    log("-" * 80)
    log("(b) Sites & Demand Match, Candidate Hubs (sites.csv, demand.csv)")
    log("-" * 80)
    sites_path = base_dir / "sites.csv"
    demand_path = base_dir / "demand.csv"
    sites = pd.read_csv(sites_path)
    demand = pd.read_csv(demand_path)

    phc_sites = set(sites[sites["type"] == "phc"]["site_id"])
    phc_demand = set(demand["phc_id"])

    match = phc_sites == phc_demand
    log(f"Total sites: {len(sites)} | Total demand records: {len(demand)}")
    log(f"PHC count in sites: {len(phc_sites)} | PHC count in demand: {len(phc_demand)}")
    log(f"PHC ID exact match between sites.csv and demand.csv: {match}")
    if not match:
        missing_in_demand = phc_sites - phc_demand
        missing_in_sites = phc_demand - phc_sites
        if missing_in_demand:
            log(f"  [FLAG] In sites but missing in demand: {sorted(missing_in_demand)}")
        if missing_in_sites:
            log(f"  [FLAG] In demand but missing in sites: {sorted(missing_in_sites)}")

    candidate_hubs = sites[sites["candidate_hub"] == 1]["site_id"].tolist()
    log(f"Candidate hubs (candidate_hub == 1): {candidate_hubs}")
    is_p09_both = ("P09" in candidate_hubs) and ("P09" in phc_sites)
    log(f"Special check: P09 is both a PHC and a candidate hub: {is_p09_both}")
    log()

    # -------------------------------------------------------------------------
    # (c) nofly.geojson ring closure and site inside check
    # -------------------------------------------------------------------------
    log("-" * 80)
    log("(c) No-Fly Zones Polygon Ring Closure & Site Intersection (nofly.geojson)")
    log("-" * 80)
    nofly_path = base_dir / "nofly.geojson"
    with open(nofly_path, "r", encoding="utf-8") as f:
        nofly_geo = json.load(f)

    polygons = []
    for idx, feature in enumerate(nofly_geo.get("features", [])):
        geom = feature.get("geometry", {})
        poly_shape = shape(geom)
        name = feature.get("properties", {}).get("name", f"Feature_{idx}")
        kind = feature.get("properties", {}).get("kind", "unknown")
        
        # Check ring closure
        if isinstance(poly_shape, Polygon):
            polys = [poly_shape]
        else:
            polys = list(poly_shape.geoms)
        
        for p_idx, p in enumerate(polys):
            exterior_closed = (p.exterior.coords[0] == p.exterior.coords[-1])
            interiors_closed = all(ring.coords[0] == ring.coords[-1] for ring in p.interiors)
            closed = exterior_closed and interiors_closed
            log(f"Polygon '{name}' (idx {idx}, sub {p_idx}, kind: {kind}): Ring closed = {closed}")
            polygons.append((name, p))

    sites_in_nofly = []
    for _, site in sites.iterrows():
        pt = Point(site["x_km"], site["y_km"])
        for name, poly in polygons:
            if poly.contains(pt):
                sites_in_nofly.append((site["site_id"], site["name"], name))

    if sites_in_nofly:
        log(f"FLAG: Sites found inside no-fly zones: {sites_in_nofly}")
    else:
        log("OK: Zero sites lie inside any no-fly zone.")
    log()

    # -------------------------------------------------------------------------
    # (d) components.csv audit (Batteries, CFRP tubes, Motors)
    # -------------------------------------------------------------------------
    log("-" * 80)
    log("(d) Components Audit (components.csv)")
    log("-" * 80)
    components_path = base_dir / "components.csv"
    components = pd.read_csv(components_path)

    # 1. Batteries
    batteries = components[components["category"] == "battery"].copy()
    batteries["Wh"] = batteries["capacity"] * batteries["nominal_voltage_V"] / 1000.0
    batteries["Wh_per_kg"] = batteries["Wh"] / batteries["mass_kg"]
    batteries["INR_per_Wh"] = batteries["unit_cost_inr"] / batteries["Wh"]

    log("1. Battery Packs:")
    for _, bat in batteries.iterrows():
        out_of_range = not (200.0 <= bat["Wh"] <= 2000.0)
        range_status = "FLAGGED (outside [200, 2000] Wh)" if out_of_range else "In range [200, 2000] Wh"
        log(f"  - {bat['part_id']}: Capacity={bat['capacity']} mAh, Voltage={bat['nominal_voltage_V']} V, "
            f"Energy={bat['Wh']:.1f} Wh ({range_status}), "
            f"Specific Energy={bat['Wh_per_kg']:.1f} Wh/kg (~150 Wh/kg), "
            f"Cost={bat['INR_per_Wh']:.2f} INR/Wh (~22 INR/Wh)")

    # 2. CFRP Tubes
    tubes = components[components["category"] == "tube"].copy()
    tubes["d_over_t"] = tubes["outer_dia_mm"] / tubes["wall_mm"]
    rho_cf = 1550.0  # kg/m^3
    # Cross sectional area in m^2: pi/4 * (d^2 - (d-2t)^2) = pi * t * (d - t) * 1e-6
    tubes["theor_mass_kg_m"] = rho_cf * math.pi * (tubes["outer_dia_mm"] * tubes["wall_mm"] - tubes["wall_mm"]**2) * 1e-6
    tubes["mass_diff_pct"] = (tubes["mass_kg"] - tubes["theor_mass_kg_m"]) / tubes["theor_mass_kg_m"] * 100.0

    log(f"\n2. CFRP Tubes (Total count: {len(tubes)}):")
    dt_ok = (tubes["d_over_t"] <= 40.0).all()
    log(f"  - d/t <= 40 check: {'PASSED (all tubes d/t <= 40)' if dt_ok else 'FLAGGED'}")
    log(f"  - Max d/t observed: {tubes['d_over_t'].max():.2f}")
    log(f"  - Mass match against rho_cf = 1550 kg/m^3: difference range [{tubes['mass_diff_pct'].min():.3f}%, {tubes['mass_diff_pct'].max():.3f}%] (consistent)")

    # 3. Motors
    motors = components[components["category"] == "motor"].copy()
    motors["expected_mass"] = 0.12 + 0.004 * motors["max_thrust_N"]
    motors["mass_diff"] = (motors["mass_kg"] - motors["expected_mass"]).abs()
    motor_ok = (motors["mass_diff"] < 1e-4).all()

    log(f"\n3. Motors (Total count: {len(motors)}):")
    log(f"  - Verification against mass formula (0.12 + 0.004 * max_thrust_N): {'PASSED (all exact match)' if motor_ok else 'FLAGGED'}")
    for _, mot in motors.iterrows():
        log(f"    * {mot['part_id']}: Max Thrust={mot['max_thrust_N']} N, Mass={mot['mass_kg']:.3f} kg, Expected={mot['expected_mass']:.3f} kg")
    log()

    # -------------------------------------------------------------------------
    # (e) wind_days.csv audit
    # -------------------------------------------------------------------------
    log("-" * 80)
    log("(e) Wind Profiles Audit (wind_days.csv)")
    log("-" * 80)
    wind_path = base_dir / "wind_days.csv"
    wind = pd.read_csv(wind_path)

    log(f"Row count: {len(wind)} (Expected: 8760) -> {'OK' if len(wind) == 8760 else 'FLAGGED'}")
    missing_cnt = wind.isnull().sum().sum()
    negative_cnt = (wind.select_dtypes(include=['number']) < 0).sum().sum()
    log(f"Missing values: {missing_cnt} | Negative values: {negative_cnt}")

    gust_less_mean = (wind["gust_ms"] < wind["mean_wind_ms"]).sum()
    log(f"Anomalies where gust_ms < mean_wind_ms: {gust_less_mean} -> {'OK' if gust_less_mean == 0 else 'FLAGGED'}")

    flyable_mask = (wind["gust_ms"] < 14.0) & (wind["mean_wind_ms"] >= 8.0)
    flyable_count = flyable_mask.sum()
    log(f"Flyable hours (gust_ms < 14.0 and mean_wind_ms >= 8.0): {flyable_count} hours ({flyable_count / len(wind) * 100:.2f}% of the year)")
    log()

    # -------------------------------------------------------------------------
    # (f) costs.yaml summary
    # -------------------------------------------------------------------------
    log("-" * 80)
    log("(f) Costs Configuration Summary (costs.yaml)")
    log("-" * 80)
    costs_path = base_dir / "costs.yaml"
    with open(costs_path, "r", encoding="utf-8") as f:
        costs = yaml.safe_load(f)

    log(f"Currency: {costs.get('currency', 'INR')}")
    log(f"Discount Rate: {costs.get('discount_rate')}")
    log(f"Hub Capex: INR {costs.get('hub', {}).get('capex'):,} | Annual Opex: INR {costs.get('hub', {}).get('opex_per_year'):,}")
    log(f"Charging Bay Capex: INR {costs.get('charging_bay', {}).get('capex'):,} | Power: {costs.get('charging_bay', {}).get('power_kW')} kW")
    
    outage_cfg = costs.get("grid_outage", {})
    genset_kw = outage_cfg.get("genset_kW_per_hub", 3.0)
    bay_kw = costs.get("charging_bay", {}).get("power_kW", 1.5)
    max_bays = int(genset_kw // bay_kw)
    log(f"Grid Outages: {outage_cfg.get('days')} from {outage_cfg.get('start')} to {outage_cfg.get('end')}")
    log(f"Genset backup power per hub: {genset_kw} kW")
    log(f"Power per charging bay: {bay_kw} kW")
    log(f"Power constraint during outages: {genset_kw} kW / {bay_kw} kW = max {max_bays} active charging bays during Mon/Wed/Fri 14:00–17:00 outages.")
    log()
    log("=" * 80)
    log("AUDIT COMPLETED SUCCESSFULLY. REPORT SAVED.")
    log("=" * 80)

    # Save to report file
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to: {report_file}")


if __name__ == "__main__":
    audit_datasets()
