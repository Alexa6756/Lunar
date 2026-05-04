import os
import sys
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.feature_extraction import extract_features

API_URL = os.getenv("LUNAR_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Lunar Dynamic Risk Prediction System", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0a0a14; color: #e0e0f0; }
  section[data-testid="stSidebar"] { background-color: #0f0f1e; }
  [data-testid="metric-container"] {
    background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 8px; padding: 8px 12px;
  }
  h1 { color: #c8a96e !important; }
  h2, h3 { color: #a0c4ff !important; }
</style>
""", unsafe_allow_html=True)

BADGE  = {0: "SAFE", 1: "MODERATE", 2: "DANGER"}
CSCALE = [[0, "#2ECC71"], [0.5, "#F1C40F"], [1.0, "#E74C3C"]]


def _get(path, params=None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params)
        return r.json()
    except:
        return None


def _post(path, body):
    try:
        r = requests.post(f"{API_URL}{path}", json=body)
        return r.json()
    except:
        return None


@st.cache_data(ttl=300)
def fetch_risk_map(model: str, downsample: int = 20):
    data = _get(f"/hazard/map/{model}", params={"downsample": downsample})
    return np.array(data["data"], dtype=np.uint8) if data else None


@st.cache_data(ttl=60)
def fetch_stats(model: str):
    return _get(f"/hazard/stats/{model}") or {}


@st.cache_data
def load_features():
    return extract_features("data/dem.tif")


@st.cache_data
def load_thermal_features():
    import rasterio
    with rasterio.open("data/temp_max.tif") as src:
        temp_day = src.read(1)
    with rasterio.open("data/temp_min.tif") as src:
        temp_night = src.read(1)
    temp_variation = temp_day - temp_night
    gy, gx = np.gradient(temp_day)
    temp_gradient = np.sqrt(gx**2 + gy**2)
    return {
        "temp_day":       temp_day,
        "temp_night":     temp_night,
        "temp_variation": temp_variation,
        "temp_gradient":  temp_gradient,
    }


features      = load_features()
dem           = features["dem"]
slope         = features["slope"]
roughness     = features["roughness"]
curvature     = features["curvature"]
tpi           = features["tpi"]
tri           = features["tri"]
thermal       = load_thermal_features()
temp_day      = thermal["temp_day"]
temp_night    = thermal["temp_night"]
temp_variation = thermal["temp_variation"]
temp_gradient  = thermal["temp_gradient"]

st.sidebar.markdown("# Lunar Dynamic Risk Prediction System")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio("**Risk Model**",
                                ["Terrain model", "Terrain-Thermal model"])
model_key = "static" if model_choice == "Terrain model" else "dynamic"

st.sidebar.markdown("---")

view_mode = st.sidebar.selectbox("**View**", [
    "Hazard Map",
    "Compare Models",
    "Predict from Features",
    "Rover Navigation",
])

DOWNSAMPLE = 20

with st.spinner("Loading risk map ..."):
    risk_arr = fetch_risk_map(model_key, DOWNSAMPLE)

if risk_arr is None:
    st.warning("Map not available")

if view_mode == "Hazard Map":
    st.markdown("## Lunar Hazard Map")
    st.caption(f"Model: **{model_choice}**")

    label_map = np.vectorize({0: "Safe", 1: "Moderate", 2: "Danger"}.get)(risk_arr)
    fig = go.Figure(go.Heatmap(
        z=risk_arr,
        customdata=label_map,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Risk: %{customdata}<extra></extra>",
        colorscale=CSCALE,
        zmin=0, zmax=2,
        colorbar=dict(title="Risk", tickvals=[0,1,2],
                      ticktext=["Safe","Moderate","Danger"]),
    ))
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor="#0a0a14", font=dict(color="#e0e0f0"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Inspect Location")
    c1, c2 = st.columns(2)
    with c1:
        x = st.number_input("X", 0, risk_arr.shape[1]-1, 0)
    with c2:
        y = st.number_input("Y", 0, risk_arr.shape[0]-1, 0)

    full_x = int(x * DOWNSAMPLE)
    full_y = int(y * DOWNSAMPLE)
    risk   = int(risk_arr[y, x])

    st.markdown("### Risk Level")
    if risk == 0:   st.success("SAFE ZONE")
    elif risk == 1: st.warning("MODERATE ZONE")
    else:           st.error("DANGER ZONE")

    st.markdown("### Terrain Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Elevation", f"{dem[full_y, full_x]:.2f}")
        st.metric("Slope",     f"{slope[full_y, full_x]:.2f}")
    with c2:
        st.metric("Roughness", f"{roughness[full_y, full_x]:.3f}")
        st.metric("Curvature", f"{curvature[full_y, full_x]:.3f}")
    with c3:
        st.metric("TPI", f"{tpi[full_y, full_x]:.3f}")
        st.metric("TRI", f"{tri[full_y, full_x]:.3f}")

    if model_key == "dynamic":
        st.markdown("### Thermal Features")
        ty = min(int(full_y * temp_day.shape[0] / dem.shape[0]), temp_day.shape[0]-1)
        tx = min(int(full_x * temp_day.shape[1] / dem.shape[1]), temp_day.shape[1]-1)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Temp Day",   f"{temp_day[ty, tx]:.2f}")
            st.metric("Temp Night", f"{temp_night[ty, tx]:.2f}")
        with c2:
            st.metric("Temp Variation", f"{temp_variation[ty, tx]:.2f}")
            st.metric("Temp Gradient",  f"{temp_gradient[ty, tx]:.4f}")

elif view_mode == "Compare Models":
    st.markdown("#### Terrain vs Terrain-Thermal")
    static_arr  = fetch_risk_map("static",  DOWNSAMPLE)
    dynamic_arr = fetch_risk_map("dynamic", DOWNSAMPLE)
    c1, c2 = st.columns(2)
    for col, arr, title in [(c1, static_arr, "Static"), (c2, dynamic_arr, "Dynamic")]:
        label_map = np.vectorize({0: "Safe", 1: "Moderate", 2: "Danger"}.get)(arr)
        fig = go.Figure(go.Heatmap(
            z=arr, customdata=label_map,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Risk: %{customdata}<extra></extra>",
            colorscale=CSCALE, zmin=0, zmax=2,
            colorbar=dict(title="Risk", tickvals=[0,1,2],
                          ticktext=["Safe","Moderate","Danger"]),
        ))
        col.plotly_chart(fig, use_container_width=True)
    disagree = np.mean(static_arr != dynamic_arr) * 100
    st.metric("Difference", f"{disagree:.1f}%")

elif view_mode == "Predict from Features":
    st.markdown("Predict with Features")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        elevation     = c1.number_input("Elevation",  value=1500.0)
        slope_val     = c1.number_input("Slope",      value=0.15)
        roughness_val = c2.number_input("Roughness",  value=0.1)
        curvature_val = c2.number_input("Curvature",  value=0.001)
        tpi_val       = c3.number_input("TPI",        value=0.5)
        tri_val       = c3.number_input("TRI",        value=0.08)
        if model_key == "dynamic":
            st.markdown("### Thermal Features")
            c4, c5 = st.columns(2)
            temp_day_val      = c4.number_input("Temp Day",      value=250.0)
            temp_night_val    = c4.number_input("Temp Night",    value=100.0)
            temp_variation_val = temp_day_val - temp_night_val
            temp_gradient_val = c5.number_input("Temp Gradient", value=0.5)
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "model": model_key, "elevation": elevation,
            "slope": slope_val, "roughness": roughness_val,
            "curvature": curvature_val, "tpi": tpi_val, "tri": tri_val,
        }
        if model_key == "dynamic":
            payload.update({
                "temp_day": temp_day_val, "temp_night": temp_night_val,
                "temp_variation": temp_variation_val,
                "temp_gradient": temp_gradient_val,
            })
        result = _post("/hazard/predict", payload)
        if result and "risk_class" in result:
            st.markdown(f"### Result: {BADGE.get(result['risk_class'])}")
        else:
            st.error(f"Invalid response: {result}")

elif view_mode == "Rover Navigation":
    st.markdown("## Rover Navigation")

    risk_arr = fetch_risk_map("dynamic", DOWNSAMPLE)
    if risk_arr is None:
        st.error("Dynamic map not available")
        st.stop()

    h_map, w_map = risk_arr.shape

    label_map = np.vectorize({0: "Safe", 1: "Moderate", 2: "Danger"}.get)(risk_arr)

    fig = go.Figure(go.Heatmap(
        z=risk_arr,
        customdata=label_map,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Risk: %{customdata}<extra></extra>",
        colorscale=CSCALE,
        zmin=0, zmax=2,
        colorbar=dict(title="Risk", tickvals=[0,1,2],
                      ticktext=["Safe","Moderate","Danger"]),
    ))

    if "path" not in st.session_state:
        st.session_state.path = None

    st.markdown("### Select Start & End")
    c1, c2 = st.columns(2)
    with c1:
        start_x = st.number_input("Start X", 0, w_map-1, 0)
        start_y = st.number_input("Start Y", 0, h_map-1, 0)
    with c2:
        end_x = st.number_input("End X", 0, w_map-1, w_map-1)
        end_y = st.number_input("End Y", 0, h_map-1, h_map-1)

    st.markdown("### Rover Configuration")
    c1, c2 = st.columns(2)
    max_slope    = c1.slider("Max Slope",    0,   45,  25)
    risk_weight  = c2.slider("Risk Weight",  0.0, 5.0, 2.0)
    

    if st.button("Find Path"):
        result = _post("/navigation/path", {
            "start": [int(start_x), int(start_y)],
            "end":   [int(end_x),   int(end_y)],
            "rover_config": {
                "max_slope":    max_slope,
                "risk_weight":  risk_weight,
                "slope_weight": 2.0,
            }
        })

        if result and result.get("path"):
            st.session_state.path = result["path"]
            
        elif result and result.get("message"):
            st.warning(result["message"])
        else:
            st.error(f"Navigation failed: {result}")

    if st.session_state.path:
        path = st.session_state.path
        px = []
        py = []
        for i in range(len(path)):
            px.append(path[i][0])
            py.append(path[i][1])
            if i < len(path) - 1:
                if abs(path[i+1][0] - path[i][0]) > w_map // 2:
                    px.append(None)
                    py.append(None)
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Path", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[start_x], y=[start_y],
            mode="markers",
            marker=dict(color="green", size=20, symbol="circle"),
            name="Start", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[end_x], y=[end_y],
            mode="markers",
            marker=dict(color="red", size=20, symbol="circle"),
            name="End", showlegend=False,
        ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0a0a14",
        font=dict(color="#e0e0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)
