"""
Streamlit frontend for DeliverAI.

Run with: streamlit run frontend/app.py
(Requires the FastAPI backend running at http://localhost:8000 —
see README.md for the exact startup order.)
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="DeliverAI — ETA Predictor", page_icon="🛵", layout="centered")

st.title("🛵 DeliverAI")
st.caption("MLOps-based food delivery ETA prediction")

# --- Backend health check ---
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    api_up = True
except requests.exceptions.RequestException:
    api_up = False
    health = {}

if not api_up:
    st.error(
        "Can't reach the API at http://localhost:8000. "
        "Start it first: `uvicorn api.main:app --reload --port 8000`"
    )
else:
    model_name = health.get("model_info", {}).get("model_name", "unknown")
    st.success(f"API connected — serving model: **{model_name}**")

st.divider()

col1, col2 = st.columns(2)
with col1:
    distance_km = st.slider("Distance (km)", 0.5, 25.0, 4.0, 0.1)
    vehicle_type = st.selectbox("Vehicle type", ["bike", "scooter", "bicycle", "car"])
    weather = st.selectbox("Weather", ["clear", "rainy", "foggy", "stormy"])
with col2:
    traffic_level = st.selectbox("Traffic level", ["low", "medium", "high", "jam"])
    prep_time_min = st.slider("Kitchen prep time (min)", 0, 60, 15)
    courier_experience_yr = st.slider("Courier experience (years)", 0.0, 15.0, 2.0, 0.5)

if st.button("Predict ETA", type="primary", use_container_width=True, disabled=not api_up):
    payload = {
        "distance_km": distance_km,
        "vehicle_type": vehicle_type,
        "weather": weather,
        "traffic_level": traffic_level,
        "prep_time_min": prep_time_min,
        "courier_experience_yr": courier_experience_yr,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        st.metric("Predicted ETA", f"{result['predicted_eta_min']} min")
        st.caption(f"Model used: {result['model_used']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

st.divider()

with st.expander("Model comparison (from last training run)"):
    comparison_path = ROOT / "models" / "model_comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No comparison found yet. Run `python src/train.py`.")

with st.expander("Latest drift report"):
    drift_path = ROOT / "monitoring" / "drift_report.json"
    if drift_path.exists():
        st.json(json.loads(drift_path.read_text()))
    else:
        st.info("No drift report yet. Run `python monitoring/monitor_drift.py` after a few predictions.")
