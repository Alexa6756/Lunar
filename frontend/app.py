import os
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
 
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
 
BADGE = {
    0: "SAFE",
    1: "MODERATE",
    2: "DANGER",
}
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
 
 
@st.cache_data(show_spinner=False, ttl=300)
def fetch_risk_map(model: str, downsample: int = 20):
    data = _get(f"/hazard/map/{model}", params={"downsample": downsample})
    return np.array(data["data"], dtype=np.uint8) if data else None
 
 
@st.cache_data(show_spinner=False, ttl=60)
def fetch_stats(model: str):
    return _get(f"/hazard/stats/{model}") or {}
 

st.sidebar.markdown("# Lunar Dynamic Risk Prediction System ")
st.sidebar.markdown("---")
model_choice = st.sidebar.radio("**Risk Model**",
                                 ["Static (terrain)",
                                  "Dynamic (terrain + thermal)"])
model_key = "static" if "Static" in model_choice else "dynamic"
 
st.sidebar.markdown("---")
view_mode = st.sidebar.selectbox("**View**", [
    "Hazard Map",
    "Compare Models",
    "Predict from Features",
])
 
 
DOWNSAMPLE = 20
with st.spinner("Loading risk map ..."):
    risk_arr = fetch_risk_map(model_key, DOWNSAMPLE)
 
if risk_arr is None:
    st.warning("Map not available")

 
if view_mode == "Hazard Map":
    st.markdown("## Lunar Hazard Map")
    st.caption(f"Model: **{model_choice}**")
 
    fig = go.Figure(go.Heatmap(
        z=risk_arr, colorscale=CSCALE, zmin=0, zmax=2,
        colorbar=dict(title="Risk", tickvals=[0,1,2],
                      ticktext=["Safe","Moderate","Danger"]),
    ))
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor="#0a0a14", font=dict(color="#e0e0f0"))
    st.plotly_chart(fig, use_container_width=True)

 
 
elif view_mode == "Compare Models":
    st.markdown("#### Static vs Dynamic")
 
    with st.spinner("Loading maps"):
        static_arr  = fetch_risk_map("static",  DOWNSAMPLE)
        dynamic_arr = fetch_risk_map("dynamic", DOWNSAMPLE)
 
    if static_arr is None or dynamic_arr is None:
        st.error("Maps not loaded")
 
    c1, c2 = st.columns(2)
    for col, arr, title in [
        (c1, static_arr,  "Static (terrain)"),
        (c2, dynamic_arr, "Dynamic (terrain + thermal)"),
    ]:
        fig = go.Figure(go.Heatmap(
            z=arr, colorscale=CSCALE, zmin=0, zmax=2, showscale=False
        ))
        fig.update_layout(height=400, title=title,
                          paper_bgcolor="#0a0a14", font=dict(color="#e0e0f0"),
                          margin=dict(l=0,r=0,t=40,b=0))
        col.plotly_chart(fig, use_container_width=True)
 
    disagree = np.mean(static_arr != dynamic_arr) * 100
    st.metric("Difference between models", f"{disagree:.1f}%")
 
 
elif view_mode == "Predict from Features":
    st.markdown("Predict with Features")
 
    with st.form("predict_form"):
        st.markdown("#### Terrain Features")
        c1, c2, c3 = st.columns(3)
        elevation = c1.number_input("Elevation (m)",  value=1500.0)
        slope     = c1.number_input("Slope",          value=0.15,  format="%.4f")
        roughness = c2.number_input("Roughness",      value=0.10,  format="%.4f")
        curvature = c2.number_input("Curvature",      value=0.001, format="%.5f")
        tpi       = c3.number_input("TPI",            value=0.5,   format="%.4f")
        tri       = c3.number_input("TRI",            value=0.08,  format="%.4f")
 
        thermal = {}
        if model_key == "dynamic":
            st.markdown("#### Thermal Features")
            tc1, tc2 = st.columns(2)
            thermal["temp_day"]       = tc1.number_input("Temp Day (K)",   value=380.0)
            thermal["temp_night"]     = tc1.number_input("Temp Night (K)", value=100.0)
            thermal["temp_variation"] = tc2.number_input("Temp Variation", value=280.0)
            thermal["temp_gradient"]  = tc2.number_input("Temp Gradient",  value=1.5, format="%.4f")
 
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        body = {
            "model": model_key,
            "elevation": elevation, "slope": slope,
            "roughness": roughness, "curvature": curvature,
            "tpi": tpi, "tri": tri,
            **thermal,
        }
        with st.spinner("Predicting ..."):
            result = _post("/hazard/predict", body)
 
        if result:
            cls = result["risk_class"]
            st.markdown(f"### Result: {BADGE.get(cls, '?')}")
            c1, c2 = st.columns(2)
            c1.metric("Risk Class",  cls)
            c2.metric("Confidence",  f"{result['confidence']*100:.1f}%")
 
