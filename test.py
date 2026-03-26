import argparse
import os
import numpy as np
import rasterio
from pyproj import Transformer

os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["static", "dynamic"],
    default="static",
)
args = parser.parse_args()

map_path = f"outputs/{args.model}_risk_map.npy"

risk_map = np.load(map_path)

risk_norm = risk_map.astype(np.float32) / 2.0



REGIONS = {
    
    "Apollo 17 (Taurus-Littrow)": (20.19,   30.77,  1, "Moderate"),
    "Apollo 11 (Sea of Tranquility)":( 0.67,  23.47,  0, "Low"),
    "Mare Imbrium":               (32.80,  344.00,  0, "Low "),
    "Oceanus Procellarum":        (18.40,  303.00,  0, "Low "),
    "Tycho Crater":               (-43.31, 348.68,  2, "High "),
    "Copernicus Crater":          (  9.62, 340.00,  2, "High "),
    "Shackleton (South Pole)":    (-89.90,   0.00,  2, "High "),
}


def _fix_lon(lon: float) -> float:
    return lon - 360.0 if lon > 180.0 else lon


with rasterio.open("data/dem.tif") as src:
    crs_dem = src.crs
    n_rows, n_cols = src.shape

transformer = Transformer.from_crs("EPSG:4326", crs_dem, always_xy=True)


print(f"  LUNAR REGION HAZARD VALIDATION  ({args.model.upper()} model)")


passed = 0
total  = len(REGIONS)

for name, (lat, lon, expected, desc) in REGIONS.items():
    lon = _fix_lon(lon)
    try:
        with rasterio.open("data/dem.tif") as src:
            x, y = transformer.transform(lon, lat)
            row, col = src.index(x, y)
            row = int(np.clip(row, 0, n_rows - 1))
            col = int(np.clip(col, 0, n_cols - 1))

        
        r1 = max(0, row - 10);  r2 = min(n_rows, row + 10)
        c1 = max(0, col - 10);  c2 = min(n_cols, col + 10)
        window   = risk_map[r1:r2, c1:c2]
        avg_risk = float(window.mean())

        if avg_risk < 0.5:
            label = "SAFE (0)"
        elif avg_risk < 1.5:
            label = "MODERATE (1)"
        else:
            label = "HIGH RISK (2)"

        
        rounded = round(avg_risk)
        ok = abs(rounded - expected) <= 1
        passed += ok
        status = "Correct" if ok else "Incorrect"

        print(f"\n{name}")
        print(f"  {desc}")
        print(f"  Lat , Lon        : {lat:.2f}, {lon:.2f}")
        print(f"  Risk score : {avg_risk:.2f} , {label}")
        print(f"  Expected class : {expected}   {status}")

    except Exception as e:
        print(f"\n{name} → ERROR: {e}")

print(f"  Result: {passed}/{total} regions passed.")

