import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import numpy as np
import plotly.express as px
from preprocessing.feature_extraction import extract_features

st.set_page_config(layout="wide")
st.title(" Lunar Hazard Map")

@st.cache_data
def load_risk():
    return np.load("outputs/static_risk_map.npy")
risk_map=load_risk()
@st.cache_data
def load_features():
    return extract_features("data/dem.tif")
features=load_features()

dem=features["dem"]
slope=features["slope"]
roughness=features["roughness"]
curvature=features["curvature"]
tpi=features["tpi"]
tri=features["tri"]


rows, cols=risk_map.shape
DOWNSAMPLE=8

risk_display=risk_map[::DOWNSAMPLE, ::DOWNSAMPLE]
dem_display=dem[::DOWNSAMPLE, ::DOWNSAMPLE]

fig=px.imshow(
    dem_display,
    color_continuous_scale="gray",
    aspect="auto"
)
fig.update_coloraxes(showscale=False)

fig.add_heatmap(
    z=risk_display,
    opacity=0.5,
    colorscale=[
        [0, "green"],
        [0.5, "yellow"],
        [1, "red"]
    ],
    showscale=True
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Inspect Location")

col1, col2=st.columns(2)

with col1:
    x=st.number_input("X coordinate", 0, risk_display.shape[1]-1, 0)

with col2:
    y=st.number_input("Y coordinate", 0, risk_display.shape[0]-1, 0)

full_x=int(x * DOWNSAMPLE)
full_y=int(y * DOWNSAMPLE)

risk=risk_map[full_y, full_x]
st.subheader(" Risk Level")
if risk == 0:
    st.success("SAFE ZONE ")
elif risk == 1:
    st.warning("MODERATE ZONE ")
else:
    st.error("RISKY ZONE ")

st.subheader(" Terrain Features")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Elevation", f"{dem[full_y,full_x]:.2f}")
    st.metric("Slope", f"{slope[full_y,full_x]:.2f}")

with col2:
    st.metric("Roughness", f"{roughness[full_y,full_x]:.3f}")
    st.metric("Curvature", f"{curvature[full_y,full_x]:.3f}")

with col3:
    st.metric("TPI", f"{tpi[full_y,full_x]:.3f}")
    st.metric("TRI", f"{tri[full_y,full_x]:.3f}")

