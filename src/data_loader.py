"""
Data loader module for Skylink Course Project.
Exposes load_all_data() returning a dictionary of all 7 cleaned datasets without modifying raw files.
"""

from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import yaml
from shapely.geometry import shape, Polygon


def load_all_data():
    """
    Loads and cleans all 7 project datasets from datasets/:
      1. hourly_profile: emergency_frac, routine_frac, lab_frac normalized to sum to 1.0
      2. sites: site coordinates, types, candidate_hub flags
      3. demand: PHC demand data
      4. nofly: parsed into shapely Polygons
      5. components: battery rows augmented with energy_Wh = capacity * nominal_voltage_V / 1000.0
      6. wind_days: 8760 hourly wind and gust records
      7. costs: system parameters and outage constraints
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent / "datasets"

    # 1. hourly_profile.csv - normalize fraction columns to each sum to 1.0
    hourly = pd.read_csv(base_dir / "hourly_profile.csv")
    frac_cols = ["emergency_frac", "routine_frac", "lab_frac"]
    for col in frac_cols:
        col_sum = hourly[col].sum()
        if col_sum > 0:
            hourly[col] = hourly[col] / col_sum

    # 2. sites.csv
    sites = pd.read_csv(base_dir / "sites.csv")

    # 3. demand.csv
    demand = pd.read_csv(base_dir / "demand.csv")

    # 4. nofly.geojson - parse into shapely polygons
    with open(base_dir / "nofly.geojson", "r", encoding="utf-8") as f:
        nofly_geo = json.load(f)

    nofly_polygons = []
    for feat in nofly_geo.get("features", []):
        geom = feat.get("geometry", {})
        shp = shape(geom)
        if isinstance(shp, Polygon):
            nofly_polygons.append(shp)
        else:
            # Handle MultiPolygon if present
            nofly_polygons.extend(list(shp.geoms))

    # 5. components.csv - add energy_Wh to battery rows
    components = pd.read_csv(base_dir / "components.csv")
    battery_mask = components["category"] == "battery"
    components.loc[battery_mask, "energy_Wh"] = (
        components.loc[battery_mask, "capacity"]
        * components.loc[battery_mask, "nominal_voltage_V"]
        / 1000.0
    )

    # 6. wind_days.csv
    wind = pd.read_csv(base_dir / "wind_days.csv")

    # 7. costs.yaml
    with open(base_dir / "costs.yaml", "r", encoding="utf-8") as f:
        costs = yaml.safe_load(f)

    return {
        "sites": sites,
        "demand": demand,
        "hourly_profile": hourly,
        "wind_days": wind,
        "costs": costs,
        "components": components,
        "nofly_polygons": nofly_polygons,
        "nofly_geojson": nofly_geo,
    }


def verify():
    """
    Verifies data loader outputs and runs 00_data_audit.py to ensure zero errors.
    """
    print("Loading all datasets via load_all_data()...")
    data = load_all_data()

    # Verification checks on cleaned data
    hourly = data["hourly_profile"]
    for col in ["emergency_frac", "routine_frac", "lab_frac"]:
        col_sum = hourly[col].sum()
        assert abs(col_sum - 1.0) < 1e-6, f"{col} did not sum to 1.0 (got {col_sum})"

    components = data["components"]
    battery_rows = components[components["category"] == "battery"]
    assert "energy_Wh" in components.columns, "energy_Wh column missing in components"
    assert not battery_rows["energy_Wh"].isnull().any(), "energy_Wh missing in battery rows"

    nofly_polys = data["nofly_polygons"]
    assert len(nofly_polys) >= 2, f"Expected at least 2 no-fly polygons, got {len(nofly_polys)}"

    print("All load_all_data() programmatic assertions passed.")

    # Run 00_data_audit.py
    script_dir = Path(__file__).resolve().parent
    audit_script = script_dir / "00_data_audit.py"
    print(f"\nRunning audit script: {audit_script.name}...")
    result = subprocess.run([sys.executable, str(audit_script)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"00_data_audit.py failed with return code {result.returncode}")
    
    print("Verification completed with zero errors.")


if __name__ == "__main__":
    verify()
