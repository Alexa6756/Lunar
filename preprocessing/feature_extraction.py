import rasterio
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(dem_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    dy, dx = np.gradient(dem)
    slope = np.sqrt(dx**2 + dy**2)
    roughness = np.abs(dx) + np.abs(dy)
    dxx, _ = np.gradient(dx)
    _, dyy = np.gradient(dy)
    curvature = dxx + dyy
    tri = np.sqrt(uniform_filter((dem - uniform_filter(dem,3))**2,3))
    tpi = dem - uniform_filter(dem,15)

    return {
        "dem": dem,
        "slope": slope,
        "roughness": roughness,
        "curvature": curvature,
        "tri": tri,
        "tpi": tpi
    }