import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
import numpy as np
from models.static_model  import StaticHazardModel
from models.dynamic_model import DynamicHazardModel
from preprocessing.feature_extraction import extract_features
 
 
class AppState:
 
    ready: bool = False
    static_map:  np.ndarray | None = None
    dynamic_map: np.ndarray | None = None
 
    features: dict | None = None
    static_model:  StaticHazardModel  | None = None
    dynamic_model: DynamicHazardModel | None = None
 
    @classmethod
    def load(cls) -> None:

        for attr, path in [
            ("static_map",  "outputs/static_risk_map.npy"),
            ("dynamic_map", "outputs/dynamic_risk_map.npy"),
        ]:
            if os.path.exists(path):
                setattr(cls, attr, np.load(path))
 
        
        dem_path = "data/dem.tif"
        if os.path.exists(dem_path):
            cls.features = extract_features(dem_path)
 
        
        if (os.path.exists("outputs/static_model.pkl") and
                os.path.exists("outputs/static_scaler.pkl")):
            cls.static_model = StaticHazardModel.load("outputs")
 
        
        if (os.path.exists("outputs/dynamic_model.pkl") and
                os.path.exists("outputs/dynamic_scaler.pkl")):
            cls.dynamic_model = DynamicHazardModel.load("outputs")
 
        cls.ready = True
 
    @classmethod
    def get_risk_map(cls, model: str) -> np.ndarray | None:
        return cls.static_map if model == "static" else cls.dynamic_map
 
    @classmethod
    def get_model(cls, model: str):
        return cls.static_model if model == "static" else cls.dynamic_model
