import os
import sys
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
from contextlib import asynccontextmanager
 
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from backend.routers import hazard
from backend.state import AppState
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    AppState.load()
    yield
 
app = FastAPI(
    title="Lunar Hazard API",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(hazard.router,  prefix="/hazard",  tags=["Hazard"])
 
 
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Lunar Hazard API",
        "version": "1.0.0",
        "assets_loaded": AppState.ready,
    }
 
 
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "static_map":  AppState.static_map  is not None,
        "dynamic_map": AppState.dynamic_map is not None,
        "features":    AppState.features    is not None,
    }
 
 
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
 
