import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Hypotension EWS (Chole)", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
META_PATH = BASE / "meta.json"

SIGNALS = ["MAP", "HR", "SpO2", "RR", "EtCO2"]

# ----------------------------
# Load artifacts (no-cache, safer for cloud)
# ----------------------------
def load_artifacts():
    files_here = sorted([p.name for p in BASE.glob("*")])
    if not MODEL_PATH.exists():
        st.error("❌ model.pkl not found in app folder.")
        st.write("📁 Files found here:")
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

model, meta = load_artifacts()

# ----------------------------
# Helpers
# ----------------------------
def extract_features(buffer: pd.DataFrame, t_now: float, lookback_min: int = 10) -> np.ndarray:
    """15 features: mean/std/slope for each signal in lookback window."""
    w = buffer[buffer["time"] >= (t_now - lookback_min * 60)].copy()
    if len(w) < 5:
        return np.full((1, 15), np.nan, dtype=float)

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
        feats.append(float(slope))

    return np.array(feats, dtype=float).reshape(1, -1)

def prob_from_buffer(buffer: pd.DataFrame, t_now: float, lookback_min: int) -> float:
    X = extract_features(buffer, t_now, lookback_min)
    p = float(model.predict_proba(X)[0, 1])
    # منع التشبع 0/1 (حتى يكون العرض سريري)
    return float(np.clip(p, 1e-4, 1.0 - 1e-4))

def confidence_from_prob_and_data(prob: float, buffer: pd.DataFrame, t_now: float, lookback_min: int) -> float:
    """Confidence = prob-confidence * data-completeness factor."""
    base = float(np.clip(2 * abs(prob - 0.5), 0.0, 1.0))

    w = buffer[buffer["time"] >= (t_now - lookback_min * 60)]
    n = len(w)
    data_factor = float(np.clip(n / 30.0, 0.0, 1.0))  # يحتاج ~30 نقطة في lookback للثقة العالية

    return base * data_factor

def decision(prob: float, conf: float, threshold: float, conf_min: float, yellow_band: float):
    if conf < conf_min:
        return "YELLOW – Low confidence", "yellow"

    if prob >= threshold:
        return "RED – High risk", "red"
    elif prob >= threshold - yellow_band:
        return "YELLOW – Borderline", "yellow"
    else:
        return "GREEN – Low risk", "green"

def classify_pattern(buffer: pd.DataFrame, t_now: float, map_thr: float = 65.0) -> str:
    """Type A/B/C heuristic based on last 5 minutes of MAP."""
    w = buffer[buffer["time"] >= (t_now - 300)].copy()
    if len(w) < 5:
        return "Unknown"

    w = w.sort_values("time")
    x = w["MAP"].astype(float).values
    tt = (w["time"].astype(float).values - w["time"].astype(float).values[0]) / 60.0

    drop = float(np.nanmax(x) - np.nanmin(x))
    try:
        slope = float(np.polyfit(tt, x, 1)[0])
    except Exception:
        slope = 0.0

    runs = int(np.sum(x < map_thr))

    # Type A: rapid large drop
    if drop > 15 and (tt[-1] if len(tt) else 999) <= 3:
        return "Type A – Rapid drop"
    # Type B: gradual decline
    if slope < -2:
        return "Type B – Gradual decline"
    # Type C: intermittent
    if runs >= 2:
        return "Type C – Intermittent"
    return "Stable"

def compute_alarms_metrics(df: pd.DataFrame, threshold: float, refractory_min: int, k_required: int, m_window: int, map_thr: float = 65.0):
    """
    df: time-series for a single case (time sorted)
    Return alarms_df + alarms/hour + lead time to hypotension onset
    """
    g = df.sort_values("time").reset_index(drop=True)
    last_alarm_t = -1e18
    votes = deque(maxlen=m_window)
    alarms = []

    # hypotension onset (first time MAP < map_thr)
    hypo_times = g.loc[g["MAP"].astype(float) < map_thr, "time"]
    t_event = float(hypo_times.iloc[0]) if len(hypo_times) else None

    for i in range(len(g)):
        t = float(g.loc[i, "time"])
        # compute prob using buffer up to time t
        buf = g.loc[:i, ["time"] + SIGNALS].copy()

        p = prob_from_buffer(buf, t, st.session_state.lookback_min)
        high = (p >= threshold)

        votes.append(1 if high else 0)
        gated = (sum(votes) >= k_required)

        allow = (t - last_alarm_t) >= refractory_min * 60
        if high and gated and allow:
            alarms.append({"time": t, "prob": p})
            last_alarm_t = t

    alarms_df = pd.DataFrame(alarms)

    total_seconds = float(g["time"].max() - g["time"].min()) if len(g) else 0.0
    total_hours = total_seconds / 3600.0 if total_seconds > 0 else 0.0
    aph = (len(alarms_df) / total_hours) if total_hours > 0 else 0.0

    lead_min = None
    if t_event is not None and len(alarms_df):
        before = alarms_df[alarms_df["time"] <= t_event]
        if len(before):
            lead_min = (t_event - float(before["time"].iloc[0])) / 60.0

    return alarms_df, total_hours, aph, t_event, lead_min

# ----------------------------
# Session state
# ----------------------------
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=["time"] + SIGNALS)

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = -1e9

# store lookback in session for reuse
st.session_state.lookback_min = int(meta.get("lookback_min", 10))

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Risk threshold", 0.01, 0.99,
    float(meta.get("threshold_mean_prob", 0.14)), 0.01
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
st.session_state.lookback_min = lookback_min

st.sidebar.header("Episode gating")
m_window = st.sidebar.slider("Vote window size (m)", 3, 30, int(meta.get("m_window", 10)), 1)
k_required = st.sidebar.slider("Votes required (k)", 1, 20, int(meta.get("k_required", 3)), 1)

st.sidebar.header("Debug")
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ----------------------------
# Title
# ----------------------------
st.title("Early Warning System for Intraoperative Hypotension")
st.caption("Cholecystectomy (focus: Laparoscopic) – Research Prototype (Clinical-aware)")

# ----------------------------
# Demo scenarios
# ----------------------------
st.subheader("Quick Test Scenarios (1-click)")

cA, cB, cC, cD, cReset = st.columns(5)

def load_demo(kind: str):
    st.session_state.buffer = pd.DataFrame(columns=["time"] + SIGNALS)
    st.session_state.last_alarm_time = -1e9

    if kind == "stable":
        # 10 minutes stable
        for t in range(0, 10*60, 2):
            row = {"time": t, "MAP": 80.0, "HR": 80.0, "SpO2": 98.0, "RR": 14.0, "EtCO2": 36.0}
            st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([row])], ignore_index=True)

    elif kind == "typeA":
        # sudden drop after 5 min
        for t in range(0, 10*60, 2):
            map_v = 82.0 if t < 5*60 else 55.0
            row = {"time": t, "MAP": float(map_v), "HR": 100.0, "SpO2": 97.0, "RR": 16.0, "EtCO2": 33.0}
            st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([row])], ignore_index=True)

    elif kind == "typeB":
        # gradual decline 85 -> 55 over 10 minutes
        for t in range(0, 10*60, 2):
            map_v = 85.0 - (30.0 * (t/(10*60)))
            row = {"time": t, "MAP": float(map_v), "HR": 95.0, "SpO2": 98.0, "RR": 14.0, "EtCO2": 34.0}
            st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([row])], ignore_index=True)

    elif kind == "typeC":
        # intermittent dips
        for t in range(0, 10*60, 2):
            map_v = 75.0
            if (t // 60) % 2 == 1:  # every other minute dip
                map_v = 62.0
            row = {"time": t, "MAP": float(map_v), "HR": 90.0, "SpO2": 98.0, "RR": 14.0, "EtCO2": 35.0}
            st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([row])], ignore_index=True)

with cA:
    if st.button("Stable"):
        load_demo("stable")
with cB:
    if st.button("Type A"):
        load_demo("typeA")
with cC:
    if st.button("Type B"):
        load_demo("typeB")
with cD:
    if st.button("Type C"):
        load_demo("typeC")
with cReset:
    if st.button("Reset"):
        st.session_state.buffer = pd.DataFrame(columns=["time"] + SIGNALS)
        st.session_state.last_alarm_time = -1e9

# ----------------------------
# Main layout
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Manual Data Entry")

    t = st.number_input("Time (seconds)", min_value=0.0, step=1.0, value=float(st.session_state.buffer["time"].max()+2 if len(st.session_state.buffer) else 0))
    MAP = st.number_input("MAP", 0.0, 200.0, 75.0)
    HR = st.number_input("HR", 0.0, 250.0, 80.0)
    SpO2 = st.number_input("SpO2", 0.0, 100.0, 98.0)
    RR = st.number_input("RR", 0.0, 80.0, 14.0)
    EtCO2 = st.number_input("EtCO2", 0.0, 80.0, 36.0)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Add point"):
            row = {"time": t, "MAP": MAP, "HR": HR, "SpO2": SpO2, "RR": RR, "EtCO2": EtCO2}
            st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([row])], ignore_index=True)
            st.success("Point added.")
    with b2:
        if st.button("Evaluate now"):
            if len(st.session_state.buffer) < 3:
                st.warning("Add more points first (recommended ≥ 20).")
            else:
                buf = st.session_state.buffer[["time"] + SIGNALS].copy().sort_values("time")
                t_now = float(buf["time"].max())

                prob = prob_from_buffer(buf, t_now, lookback_min)
                conf = confidence_from_prob_and_data(prob, buf, t_now, lookback_min)
                label, color = decision(prob, conf, threshold, conf_min, yellow_band)
                pattern = classify_pattern(buf, t_now, map_thr=65.0)

                # Alarm logic
                alarm = False
                if label.startswith("RED"):
                    if (t_now - st.session_state.last_alarm_time) >= refractory_min * 60:
                        alarm = True
                        st.session_state.last_alarm_time = t_now

                st.subheader("Output")
                st.metric("Risk probability", f"{prob:.3f}")
                st.metric("Confidence", f"{conf:.3f}")
                st.write("Pattern:", pattern)
                st.markdown(f"### :{color}[{label}]")
                if alarm:
                    st.error("🚨 Alarm fired (refractory respected)")

                if show_debug:
                    X_dbg = extract_features(buf, t_now, lookback_min)
                    st.write("Debug: buffer points =", len(buf))
                    st.write("Debug: features nan count =", int(np.isnan(X_dbg).sum()))
                    st.write("Debug: threshold =", threshold, "conf_min =", conf_min, "yellow_band =", yellow_band)

    st.divider()
    st.subheader("Upload CSV (single case)")

    up = st.file_uploader("Upload CSV with columns: time,MAP,HR,SpO2,RR,EtCO2", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in (["time"] + SIGNALS) if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df = df[["time"] + SIGNALS].copy().sort_values("time")
            st.success(f"Loaded {len(df)} rows")

            alarms_df, hours, aph, t_event, lead_min = compute_alarms_metrics(
                df, threshold=threshold, refractory_min=refractory_min,
                k_required=k_required, m_window=m_window, map_thr=65.0
            )

            st.write("### CSV Evaluation Summary")
            st.write(f"- Monitoring hours: **{hours:.2f}**")
            st.write(f"- Total alarms: **{len(alarms_df)}**")
            st.write(f"- Alarms/hour: **{aph:.2f}**")

            if t_event is None:
                st.info("No hypotension onset detected (MAP<65) in this CSV.")
            else:
                st.write(f"- Hypotension onset time (first MAP<65): **{t_event:.0f} sec**")
                if lead_min is None:
                    st.warning("No alarm occurred before onset (lead time unavailable).")
                else:
                    st.write(f"- Lead time (min): **{lead_min:.2f}**")

            if len(alarms_df):
                st.write("Alarms preview:")
                st.dataframe(alarms_df.head(20), use_container_width=True)

with col2:
    st.subheader("Current Buffer (last 30 rows)")
    st.dataframe(st.session_state.buffer.tail(30), use_container_width=True)

    st.info(
        "**How to test quickly:**\n"
        "- Click a scenario (Stable / Type A/B/C)\n"
        "- Then press **Evaluate now**\n"
        "- Upload a CSV to compute alarms/hour + lead time\n\n"
        "**Decision:** GREEN / YELLOW (borderline or low confidence) / RED\n"
        "**Types:** A rapid, B gradual, C intermittent"
    )
