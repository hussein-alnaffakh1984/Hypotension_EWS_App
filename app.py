# ============================================
# Hypotension EWS – Stable Streamlit App
# Uses single calibrated Logistic model (model.pkl)
# ============================================

import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --------------------------------------------
# Page config
# --------------------------------------------
st.set_page_config(
    page_title="Hypotension EWS (Cholecystectomy)",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
META_PATH = BASE / "meta.json"

# --------------------------------------------
# Safe loading
# --------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model.pkl not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


try:
    model, meta = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load model or metadata")
    st.code(str(e))
    st.stop()

# --------------------------------------------
# Feature setup
# --------------------------------------------
SIGNALS = ["MAP", "HR", "SpO2", "RR", "EtCO2"]

def extract_features(buffer, t_now, lookback_min=10):
    """
    15 features = mean, std, slope for each signal
    """
    w = buffer[buffer["time"] >= (t_now - lookback_min * 60)].copy()
    if len(w) < 5:
        return np.full((1, 15), np.nan)

    feats = []
    w = w.sort_values("time")
    t = (w["time"].values - w["time"].values[0]) / 60.0

    for s in SIGNALS:
        x = w[s].astype(float).values
        feats.append(np.nanmean(x))
        feats.append(np.nanstd(x))
        try:
            slope = np.polyfit(t, x, 1)[0]
        except Exception:
            slope = 0.0
        feats.append(slope)

    return np.array(feats).reshape(1, -1)

# --------------------------------------------
# Decision logic
# --------------------------------------------
def confidence_from_prob(p):
    # simple, monotonic confidence
    return float(np.clip(2 * abs(p - 0.5), 0.0, 1.0))


def decision(prob, conf, threshold, conf_min, yellow_band):
    if conf < conf_min:
        return "YELLOW – Low confidence", "yellow"

    if prob >= threshold:
        return "RED – High risk", "red"
    elif prob >= threshold - yellow_band:
        return "YELLOW – Borderline", "yellow"
    else:
        return "GREEN – Low risk", "green"


def classify_pattern(buffer, t_now, map_thr=65):
    """
    Simple A/B/C classification using recent MAP trend
    """
    w = buffer[buffer["time"] >= (t_now - 300)].copy()  # last 5 min
    if len(w) < 5:
        return "Unknown"

    x = w["MAP"].values
    t = (w["time"].values - w["time"].values[0]) / 60.0

    drop = np.max(x) - np.min(x)
    try:
        slope = np.polyfit(t, x, 1)[0]
    except Exception:
        slope = 0.0

    runs = np.sum(x < map_thr)

    if drop > 15 and t[-1] <= 3:
        return "Type A – Rapid drop"
    if slope < -2:
        return "Type B – Gradual decline"
    if runs >= 2:
        return "Type C – Intermittent"
    return "Stable"

# --------------------------------------------
# Session state
# --------------------------------------------
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=["time"] + SIGNALS)

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = -1e9

# --------------------------------------------
# Sidebar controls
# --------------------------------------------
st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Risk threshold", 0.01, 0.99,
    float(meta.get("threshold_mean_prob", 0.14)),
    0.01
)

conf_min = st.sidebar.slider(
    "Minimum confidence", 0.0, 1.0,
    float(meta.get("conf_min", 0.35)), 0.05
)

yellow_band = st.sidebar.slider(
    "Yellow band", 0.0, 0.30,
    float(meta.get("yellow_band", 0.10)), 0.01
)

refractory_min = st.sidebar.slider(
    "Refractory (minutes)", 0, 30,
    int(meta.get("refractory_min", 10)), 1
)

lookback_min = st.sidebar.slider(
    "Lookback window (minutes)", 5, 20,
    int(meta.get("lookback_min", 10)), 1
)

# --------------------------------------------
# Main UI
# --------------------------------------------
st.title("Early Warning System for Intraoperative Hypotension")
st.caption("Laparoscopic Cholecystectomy – Research Prototype")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Manual Data Entry")

    t = st.number_input("Time (seconds)", min_value=0.0, step=1.0)
    MAP = st.number_input("MAP", 0.0, 200.0, 75.0)
    HR = st.number_input("HR", 0.0, 250.0, 80.0)
    SpO2 = st.number_input("SpO2", 0.0, 100.0, 98.0)
    RR = st.number_input("RR", 0.0, 80.0, 14.0)
    EtCO2 = st.number_input("EtCO2", 0.0, 80.0, 36.0)

    if st.button("Add & Evaluate"):
        row = {
            "time": t,
            "MAP": MAP,
            "HR": HR,
            "SpO2": SpO2,
            "RR": RR,
            "EtCO2": EtCO2
        }
        st.session_state.buffer = pd.concat(
            [st.session_state.buffer, pd.DataFrame([row])],
            ignore_index=True
        )

        X = extract_features(st.session_state.buffer, t, lookback_min)
        prob = float(model.predict_proba(X)[0, 1])
        conf = confidence_from_prob(prob)

        label, color = decision(prob, conf, threshold, conf_min, yellow_band)
        pattern = classify_pattern(st.session_state.buffer, t)

        alarm = False
        if label.startswith("RED"):
            if (t - st.session_state.last_alarm_time) >= refractory_min * 60:
                alarm = True
                st.session_state.last_alarm_time = t

        st.subheader("Current Output")
        st.metric("Risk probability", f"{prob:.3f}")
        st.metric("Confidence", f"{conf:.3f}")
        st.write("Pattern:", pattern)
        st.markdown(f"### :{color}[{label}]")

        if alarm:
            st.error("🚨 Alarm triggered (refractory respected)")

with col2:
    st.subheader("Recent Data")
    st.dataframe(st.session_state.buffer.tail(20), use_container_width=True)

    st.info(
        """
        **Decision logic**
        - GREEN: low risk  
        - YELLOW: borderline or low confidence  
        - RED: high risk (alarm if refractory allows)

        **Patterns**
        - Type A: rapid MAP drop  
        - Type B: gradual decline  
        - Type C: intermittent hypotension
        """
    )
