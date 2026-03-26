from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

 
ModelType  = Literal["static", "dynamic"]
RiskLabel  = Literal["safe", "moderate", "danger"]
 
 
class PixelRiskRequest(BaseModel):
    row:   int       = Field(..., ge=0, description="Pixel row (full-res grid)")
    col:   int       = Field(..., ge=0, description="Pixel column (full-res grid)")
    model: ModelType = Field("static", description="Which risk model to use")
 
 
class PixelRiskResponse(BaseModel):
    row:        int
    col:        int
    model:      ModelType
    risk_class: int        = Field(..., description="0=Safe, 1=Moderate, 2=Danger")
    risk_label: RiskLabel
    risk_norm:  float      = Field(..., description="Normalised score 0–1")
 
 
class RiskStatsResponse(BaseModel):
    model:         ModelType
    shape:         list[int]
    safe_pct:      float = Field(..., description="% pixels class 0")
    moderate_pct:  float = Field(..., description="% pixels class 1")
    danger_pct:    float = Field(..., description="% pixels class 2")
 
 
class PredictRequest(BaseModel):
    model:          ModelType = "static"
    elevation:      float
    slope:          float
    roughness:      float
    curvature:      float
    tpi:            float
    tri:            float
    
    temp_day:       Optional[float] = None
    temp_night:     Optional[float] = None
    temp_variation: Optional[float] = None
    temp_gradient:  Optional[float] = None
 
 
class PredictResponse(BaseModel):
    model:      ModelType
    risk_class: int
    risk_label: RiskLabel
    confidence: float = Field(..., description="Model probability for predicted class")
 
 
class TerrainRequest(BaseModel):
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
 
 
class TerrainResponse(BaseModel):
    row:       int
    col:       int
    elevation: float
    slope:     float
    roughness: float
    curvature: float
    tpi:       float
    tri:       float
 
 
