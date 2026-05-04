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

TERRAIN_FEATURES = ["elevation", "slope", "roughness", "curvature", "tpi", "tri"]
THERMAL_FEATURES = ["temp_day", "temp_night", "temp_variation", "temp_gradient"]
from fastapi.responses import StreamingResponse
import io, matplotlib.pyplot as plt, matplotlib.colors as mc

@router.get("/map_image")
def map_image(model: str = "static"):
    m = AppState.get_risk_map(model)
    if m is None:
        raise HTTPException(503, "Map not loaded")
    cmap = mc.LinearSegmentedColormap.from_list("lrisk",["#1D9E75","#BA7517","#E24B4A"])
    fig, ax = plt.subplots(figsize=(8,8), dpi=100)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=2, aspect='auto')
    ax.axis('off'); fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def _get_map(model: ModelType) -> np.ndarray:
    risk_map = AppState.get_risk_map(model)
    if risk_map is None:
        raise HTTPException(
            status_code=503,
            detail=f"Risk map for '{model}' not loaded. Run {model}.py.",
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
                detail=f"Dynamic model needs thermal features: {missing}",
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
        raise HTTPException(status_code=422, detail="downsample must be 1-100")
    risk_map = _get_map(model)
    ds_map   = risk_map[::downsample, ::downsample]
    return {
        "model": model, "downsample": downsample,
        "shape": list(ds_map.shape), "data": ds_map.tolist(),
    }


@router.get("/features/{model}")
def pixel_features(model: ModelType, x: int, y: int):
    """Return terrain (and optionally thermal) feature values at pixel (x, y)."""
    features = AppState.features
    if features is None:
        raise HTTPException(status_code=503, detail="Terrain features not loaded.")

    rows, cols = features["dem"].shape
    x = int(np.clip(x, 0, cols - 1))
    y = int(np.clip(y, 0, rows - 1))

    result = {}
    for key in TERRAIN_FEATURES:
        arr = features.get(key)
        if arr is not None:
            result[key] = round(float(arr[y, x]), 6)

    if model == "dynamic":
        for key in THERMAL_FEATURES:
            arr = features.get(key)
            if arr is not None:
                result[key] = round(float(arr[y, x]), 6)

    risk_map = AppState.get_risk_map(model)
    risk_val = None
    risk_label = None
    if risk_map is not None:
        rv = int(np.clip(risk_map[y, x], 0, 2))
        risk_val   = rv
        risk_label = LABEL_MAP[rv]

    return {
        "x": x,
        "y": y,
        "model": model,
        "features": result,
        "risk_class": risk_val,
        "risk_label": risk_label,
    }
