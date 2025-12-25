import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Hypotension EWS (Chole)", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
META_PATH  = BASE / "meta.json"

def load_artifacts_no_cache():
    # show what files exist (debug)
    files_here = sorted([p.name for p in BASE.glob("*")])

    if not MODEL_PATH.exists():
        st.error("❌ model.pkl not found in app folder.")
        st.write("📁 Files found in this folder:")
        st.code("\n".join(files_here))
        st.stop()

    model = joblib.load(MODEL_PATH)

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return model, meta

model, meta = load_artifacts_no_cache()

SIGNALS = ["MAP", "HR", "SpO2", "RR", "EtCO2"]

def extract_features(buffer, t_now, lookback_min=10):
    w = buffer[buffer["time"] >= (t_now - lookback_min * 60)].copy()
    if len(w) < 5:
        return np.full((1, 15), np.nan)

    w = w.sort_values("time")
    t = (w["time"].values - w["time"].values[0]) / 60.0

    feats = []
    for s in SIGNALS:
        x = w[s].astype(float).values
        feats.append(np.nanmean(x))
        feats.append(np.nanstd(x))
        try:
            slope = np.polyfit(t, x, 1)[0]
        except Exception:
            slope = 0.0
        feats.append(slope)

    return np.array(feats, dtype=float).reshape(1, -1)

def confidence_from_prob(p):
    return float(np.clip(2 * abs(p - 0.5), 0.0, 1.0))

def decision(prob, conf, threshold, conf_min, yellow_band):
    if conf < conf_min:
        return "YELLOW – Low confidence", "yellow"
    if prob >= threshold:
        return "RED – High risk", "red"
    if prob >= threshold - yellow_band:
        return "YELLOW – Borderline", "yellow"
    return "GREEN – Low risk", "green"

def classify_pattern(buffer, t_now, map_thr=65):
    w = buffer[buffer["time"] >= (t_now - 300)].copy()  # 5 min
    if len(w) < 5:
        return "Unknown"
    w = w.sort_values("time")
    x = w["MAP"].values.astype(float)
    t = (w["time"].values - w["time"].values[0]) / 60.0
    drop = float(np.nanmax(x) - np.nanmin(x))
    try:
        slope = float(np.polyfit(t, x, 1)[0])
    except Exception:
        slope = 0.0
    runs = int(np.sum(x < map_thr))

    if drop > 15 and (t[-1] if len(t) else 999) <= 3:
        return "Type A – Rapid drop"
    if slope < -2:
        return "Type B – Gradual decline"
    if runs >= 2:
        return "Type C – Intermittent"
    return "Stable"

# ---------------- UI ----------------
st.title("Early Warning System for Intraoperative Hypotension")
st.caption("Laparoscopic cholecystectomy – Research prototype")

# Defaults from meta
thr0 = float(meta.get("threshold_mean_prob", 0.14))
conf_min0 = float(meta.get("conf_min", 0.35))
yellow0 = float(meta.get("yellow_band", 0.10))
refrac0 = int(meta.get("refractory_min", 10))
lookback0 = int(meta.get("lookback_min", 10))

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Risk threshold", 0.01, 0.99, thr0, 0.01)
conf_min = st.sidebar.slider("Minimum confidence", 0.0, 1.0, conf_min0, 0.05)
yellow_band = st.sidebar.slider("Yellow band", 0.0, 0.30, yellow0, 0.01)
refractory_min = st.sidebar.slider("Refractory (min)", 0, 30, refrac0, 1)
lookback_min = st.sidebar.slider("Lookback (min)", 5, 20, lookback0, 1)

if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=["time"] + SIGNALS)
if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = -1e9

c1, c2 = st.columns(2)

with c1:
    st.subheader("Manual input")
    t = st.number_input("time (sec)", min_value=0.0, step=1.0)
    MAP = st.number_input("MAP", 0.0, 200.0, 75.0)
    HR = st.number_input("HR", 0.0, 250.0, 80.0)
    SpO2 = st.number_input("SpO2", 0.0, 100.0, 98.0)
    RR = st.number_input("RR", 0.0, 80.0, 14.0)
    EtCO2 = st.number_input("EtCO2", 0.0, 80.0, 36.0)

    if st.button("Add & Evaluate"):
        row = {"time": t, "MAP": MAP, "HR": HR, "SpO2": SpO2, "RR": RR, "EtCO2": EtCO2}
        st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([row])], ignore_index=True)

        X = extract_features(st.session_state.buffer, t, lookback_min)
        prob = float(model.predict_proba(X)[0, 1])
        conf = confidence_from_prob(prob)
        label, color = decision(prob, conf, threshold, conf_min, yellow_band)
        pattern = classify_pattern(st.session_state.buffer, t)

        alarm = False
        if label.startswith("RED") and (t - st.session_state.last_alarm_time) >= refractory_min * 60:
            alarm = True
            st.session_state.last_alarm_time = t

        st.subheader("Output")
        st.metric("Risk probability", f"{prob:.3f}")
        st.metric("Confidence", f"{conf:.3f}")
        st.write("Pattern:", pattern)
        st.markdown(f"### :{color}[{label}]")
        if alarm:
            st.error("🚨 Alarm fired (refractory respected)")

with c2:
    st.subheader("Recent data")
    st.dataframe(st.session_state.buffer.tail(25), use_container_width=True)
