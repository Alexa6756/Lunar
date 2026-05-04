import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from models.static_model import StaticHazardModel
from models.dynamic_model import DynamicHazardModel
from preprocessing.feature_extraction import extract_features


class AppState:

    ready: bool = False
    static_map: np.ndarray | None = None
    dynamic_map: np.ndarray | None = None

    features: dict | None = None
    static_model: StaticHazardModel | None = None
    dynamic_model: DynamicHazardModel | None = None

    @classmethod
    def load(cls) -> None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        static_map_path = os.path.join(base_dir, "outputs", "static_risk_map.npy")
        dynamic_map_path = os.path.join(base_dir, "outputs", "dynamic_risk_map.npy")

        if os.path.exists(static_map_path):
            cls.static_map = np.load(static_map_path)

        if os.path.exists(dynamic_map_path):
            cls.dynamic_map = np.load(dynamic_map_path)

        dem_path = os.path.join(base_dir, "data", "dem.tif")
        if os.path.exists(dem_path):
            cls.features = extract_features(dem_path)

        static_model_path = os.path.join(base_dir, "outputs", "static_model.pkl")
        static_scaler_path = os.path.join(base_dir, "outputs", "static_scaler.pkl")

        if os.path.exists(static_model_path) and os.path.exists(static_scaler_path):
            cls.static_model = StaticHazardModel.load(os.path.join(base_dir, "outputs"))

        dynamic_model_path = os.path.join(base_dir, "outputs", "dynamic_model.pkl")
        dynamic_scaler_path = os.path.join(base_dir, "outputs", "dynamic_scaler.pkl")

        if os.path.exists(dynamic_model_path) and os.path.exists(dynamic_scaler_path):
            cls.dynamic_model = DynamicHazardModel.load(os.path.join(base_dir, "outputs"))

        cls.ready = True

    @classmethod
    def get_risk_map(cls, model: str) -> np.ndarray | None:
        return cls.static_map if model == "static" else cls.dynamic_map

    @classmethod
    def get_model(cls, model: str):
        return cls.static_model if model == "static" else cls.dynamic_model
