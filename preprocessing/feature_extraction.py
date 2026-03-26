import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

def extract_features(dem_path: str,
                     temp_day_path: str = None,
                     temp_night_path: str = None) -> dict:

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)

    dy, dx = np.gradient(dem)

    slope = np.sqrt(dx**2 + dy**2)
    roughness = np.abs(dx) + np.abs(dy)

    dxx, _ = np.gradient(dx)
    _, dyy = np.gradient(dy)
    curvature = dxx + dyy

    tri = np.sqrt(uniform_filter((dem - uniform_filter(dem, 3)) ** 2, 3))
    tpi = dem - uniform_filter(dem, 15)

    features = {
        "dem":       dem,
        "slope":     slope,
        "roughness": roughness,
        "curvature": curvature,
        "tri":       tri,
        "tpi":       tpi,
    }

    if temp_day_path and temp_night_path:
        with rasterio.open(temp_day_path) as src:
            temp_day = src.read(1).astype(np.float32)

        with rasterio.open(temp_night_path) as src:
            temp_night = src.read(1).astype(np.float32)

        temp_variation = temp_day - temp_night

        dy_t, dx_t = np.gradient(temp_day)
        temp_gradient = np.sqrt(dx_t**2 + dy_t**2)

        features.update({
            "temp_day":       temp_day,
            "temp_night":     temp_night,
            "temp_variation": temp_variation,
            "temp_gradient":  temp_gradient,
        })

    return features
