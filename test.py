
import numpy as np
import rasterio
import os
from pyproj import Transformer


os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"
risk_map=np.load("outputs/static_risk_map.npy")
risk_map=np.clip(risk_map, 0, 1)
print("Risk map shape:", risk_map.shape)


with rasterio.open("data/dem.tif") as src:
    rows, cols=risk_map.shape
    print("\nDEM Information")
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    
    transformer=Transformer.from_crs(
        "EPSG:4326",
        src.crs,
        always_xy=True
    )
    regions = {
        "Apollo 17 (Taurus-Littrow Valley)": (20.19, 30.77),
        "Tycho Crater": (-43.31, 348.68),
        "Mare Imbrium": (32.8, 344.0),
        "Shackleton Crater (South Pole)": (-89.9, 0.0),
        "Copernicus Crater": (9.62, 340.0),
        "Oceanus Procellarum": (18.4, 303.0)
    }
    print("LUNAR REGION HAZARD VALIDATION")
    
    for name, (lat, lon) in regions.items():
        try:
            if lon>180:
                lon=lon - 360
            x,y=transformer.transform(lon, lat)
            row,col=src.index(x, y)
            row=np.clip(row, 0, rows - 1)
            col=np.clip(col, 0, cols - 1)

            point_risk=risk_map[row, col]
            r1=max(0, row - 10)
            r2=min(rows, row + 10)
            c1=max(0, col - 10)
            c2=min(cols, col + 10)

            window=risk_map[r1:r2, c1:c2]
            avg_risk=window.mean()
            if avg_risk<0.33:
                risk_label="SAFE"
            elif avg_risk<0.66:
                risk_label="MODERATE RISK"
            else:
                risk_label="HIGH RISK"

            print(f"\n{name}")
            print("Coordinates:", lat, lon)
            print("Projected coords:", round(x), round(y))
            print("Pixel location:", row, col)
            print("Point risk:", round(float(point_risk), 3))
            print("Average risk:", round(float(avg_risk), 3))
            print("Interpreted hazard:", risk_label)

        except Exception as e:

            print(f"\n{name} → Error:", e)



