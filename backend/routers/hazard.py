import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
 
from backend.schemas import (
    ModelType,
    PixelRiskRequest,
    PixelRiskResponse,
    PredictRequest,
    PredictResponse,
    RiskStatsResponse,
)
from backend.state import AppState
 
router   = APIRouter()
LABEL_MAP = {0: "safe", 1: "moderate", 2: "danger"}
 
 
def _get_map(model: ModelType) -> np.ndarray:
    risk_map = AppState.get_risk_map(model)
    if risk_map is None:
        raise HTTPException(
            status_code=503,
            detail=f"Risk map for '{model}' not loaded. "
                   f"Run {model}.py.",
        )
    return risk_map
 
 
@router.get("/stats/{model}", response_model=RiskStatsResponse)
def risk_stats(model: ModelType):
    risk_map = _get_map(model)
    total = risk_map.size
    return RiskStatsResponse(
        model=model,
        shape=list(risk_map.shape),
        safe_pct=    round(100 * np.sum(risk_map == 0) / total, 2),
        moderate_pct=round(100 * np.sum(risk_map == 1) / total, 2),
        danger_pct=  round(100 * np.sum(risk_map == 2) / total, 2),
    )
 
 
@router.post("/pixel", response_model=PixelRiskResponse)
def pixel_risk(req: PixelRiskRequest):
    risk_map    = _get_map(req.model)
    rows, cols  = risk_map.shape
    if req.row >= rows or req.col >= cols:
        raise HTTPException(
            status_code=422,
            detail=f"({req.row},{req.col}) out of bounds {risk_map.shape}."
        )
    cls = int(risk_map[req.row, req.col])
    return PixelRiskResponse(
        row=req.row, col=req.col, model=req.model,
        risk_class=cls, risk_label=LABEL_MAP[cls],
        risk_norm=round(cls / 2.0, 4),
    )
 
 
@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_obj = AppState.get_model(req.model)
    if model_obj is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model}' not loaded. Run {req.model}.py.",
        )
 
    base = {
        "elevation": req.elevation, "slope":     req.slope,
        "roughness": req.roughness, "curvature": req.curvature,
        "tpi":       req.tpi,       "tri":       req.tri,
    }
 
    if req.model == "dynamic":
        missing = [f for f in
                   ["temp_day","temp_night","temp_variation","temp_gradient"]
                   if getattr(req, f) is None]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Dynamic model need thermal features: {missing}",
            )
        base.update({
            "temp_day":       req.temp_day,
            "temp_night":     req.temp_night,
            "temp_variation": req.temp_variation,
            "temp_gradient":  req.temp_gradient,
        })
 
    X    = pd.DataFrame([base])
    cls  = int(model_obj.predict(X)[0])
    prob = float(model_obj.predict_proba(X)[0][cls])
 
    return PredictResponse(
        model=req.model, risk_class=cls,
        risk_label=LABEL_MAP[cls], confidence=round(prob, 4),
    )
 
 
@router.get("/map/{model}")
def risk_map_array(model: ModelType, downsample: int = 20):
    if not 1 <= downsample <= 100:
        raise HTTPException(status_code=422, detail="downsample")
    risk_map = _get_map(model)
    ds_map   = risk_map[::downsample, ::downsample]
    return {
        "model": model, "downsample": downsample,
        "shape": list(ds_map.shape), "data": ds_map.tolist(),
    }
 
