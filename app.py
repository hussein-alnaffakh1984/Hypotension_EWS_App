import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Chole Hypotension EWS", layout="wide")
BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "model_ens.pkl"
META_PATH  = BASE / "meta.json"

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    models = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        meta = {}
    return models, meta

models, meta = load_artifacts()

# -----------------------------
# Utilities
# -----------------------------
FEATURE_COLS = ["MAP", "HR", "SpO2", "RR", "EtCO2"]

def extract_features_from_buffer(buf: pd.DataFrame, t_now: float, lookback_min: int = 10) -> np.ndarray:
    """15 features: mean/std/slope لكل إشارة داخل آخر lookback_min."""
    past = buf[buf["time"] >= (t_now - lookback_min * 60)].copy()
    past = past.sort_values("time")
    if len(past) < 5:
        # return NaNs (pipeline has imputer)
        return np.full((1, 15), np.nan, dtype=float)

    feats = []
    t = past["time"].astype(float).values
    tmin = (t - t[0]) / 60.0
    if np.allclose(tmin, tmin[0]):
        tmin = np.arange(len(tmin), dtype=float) / 60.0

    for c in FEATURE_COLS:
        x = past[c].astype(float).values
        feats.append(np.nanmean(x))
        feats.append(np.nanstd(x))
        # slope
        try:
            slope = float(np.polyfit(tmin, x, 1)[0])
        except Exception:
            slope = 0.0
        feats.append(slope)

    return np.array(feats, dtype=float).reshape(1, -1)

def predict_with_uncertainty(models, feat: np.ndarray):
    """Ensemble mean prob + std -> confidence."""
    ps = []
    for m in models:
        p = float(m.predict_proba(feat)[0, 1])
        ps.append(p)
    ps = np.array(ps, dtype=float)

    p_mean = float(np.mean(ps))
    p_std = float(np.std(ps))

    # تحويل عدم اليقين إلى ثقة (0..1). max std عمليًا ~0.25
    conf = float(np.clip(1.0 - (p_std / 0.25), 0.0, 1.0))
    return p_mean, conf, p_std

def decide_action(prob: float, conf: float, thr: float, conf_min: float, yellow_band: float):
    """
    GREEN: low risk
    YELLOW: borderline or low confidence (review)
    RED: high risk with sufficient confidence
    """
    if conf < conf_min:
        return ("YELLOW (Review)", "yellow",
                "Low confidence (model disagreement): verify signal quality/artifacts and reassess trends.")

    if prob >= thr:
        return ("RED (High risk)", "red",
                "High-risk alert: review anesthetic depth/volume status/pneumoperitoneum effects and prepare preventive actions per clinician judgment.")
    elif prob >= (thr - yellow_band):
        return ("YELLOW (Watch)", "yellow",
                "Borderline risk: increase vigilance, reassess trends and context, and prepare for possible intervention.")
    else:
        return ("GREEN (Low risk)", "green",
                "Low risk: routine monitoring.")

def classify_hypotension_pattern(buf: pd.DataFrame, t_now: float,
                                map_col="MAP", time_col="time",
                                short_window_min=5,
                                drop_mmHg_A=18, drop_min_A=3,
                                slope_B=-2.0,
                                cv_C=0.08, min_runs_C=2, hypo_thr=65):
    """Type A/B/C based on recent MAP trend."""
    w = buf[buf[time_col] >= (t_now - short_window_min * 60)].copy().sort_values(time_col)
    if len(w) < 5:
        return "Unknown"

    t = w[time_col].astype(float).values
    x = w[map_col].astype(float).values
    msk = np.isfinite(t) & np.isfinite(x)
    t = t[msk]; x = x[msk]
    if len(x) < 5:
        return "Unknown"

    tmin = (t - t[0]) / 60.0
    dur_min = float(tmin[-1] - tmin[0]) if len(tmin) else 0.0

    # slope
    try:
        slope = float(np.polyfit(tmin, x, 1)[0])
    except Exception:
        slope = 0.0

    drop = float(np.nanmax(x) - np.nanmin(x))
    mean_x = float(np.nanmean(x))
    std_x = float(np.nanstd(x))
    cv = (std_x / mean_x) if mean_x != 0 else 0.0

    below = (x < hypo_thr).astype(int)
    runs = 0
    in_run = False
    for b in below:
        if b == 1 and not in_run:
            runs += 1
            in_run = True
        elif b == 0 and in_run:
            in_run = False

    if (drop >= drop_mmHg_A) and (dur_min <= drop_min_A):
        return "Type A (Rapid drop)"
    if (cv >= cv_C) and (runs >= min_runs_C):
        return "Type C (Intermittent)"
    if slope <= slope_B:
        return "Type B (Gradual drop)"
    return "Stable/Other"

def episode_gate(votes_deque: deque, is_high: bool, k_required: int, m_window: int):
    """k of last m windows must be high to trigger."""
    if votes_deque.maxlen != m_window:
        votes_deque = deque(votes_deque, maxlen=m_window)
    votes_deque.append(1 if is_high else 0)
    return (sum(votes_deque) >= k_required), votes_deque

def adaptive_threshold(recent_probs: deque, target_alarms_per_hour: float):
    """
    Alarm-rate controller via quantile threshold.
    Keep only top-q probs where q maps to target_alarms_per_hour.
    """
    if len(recent_probs) < 30:
        return None
    q = float(np.clip(target_alarms_per_hour / 10.0, 0.01, 0.50))
    arr = np.array(recent_probs, dtype=float)
    return float(np.quantile(arr, 1.0 - q))

# -----------------------------
# Session State
# -----------------------------
if "buf" not in st.session_state:
    st.session_state.buf = pd.DataFrame(columns=["time"] + FEATURE_COLS)
if "last_alarm_ts" not in st.session_state:
    st.session_state.last_alarm_ts = -1e18
if "recent_probs" not in st.session_state:
    st.session_state.recent_probs = deque(maxlen=600)
if "votes" not in st.session_state:
    st.session_state.votes = deque(maxlen=int(meta.get("episode_window", 10)))

# -----------------------------
# UI
# -----------------------------
st.title("Early Warning System for Hypotension (Laparoscopic Cholecystectomy)")
st.caption("Uncertainty-aware, decision-focused prototype for intraoperative monitoring (research use).")

# Defaults from meta
thr0 = float(meta.get("threshold_mean_prob", 0.35))
conf_min0 = float(meta.get("conf_min", 0.35))
yellow_band0 = float(meta.get("yellow_band", 0.10))
refractory0 = int(meta.get("refractory_min", 10))
lookback0 = int(meta.get("lookback_min", 10))
target_alarm0 = float(meta.get("adaptive_alarm_target_per_hr", 2.0))
k_req0 = int(meta.get("episode_votes", 3))
m_win0 = int(meta.get("episode_window", 10))

st.sidebar.header("Decision Settings")
use_adapt = st.sidebar.checkbox("Use adaptive threshold (alarm-rate control)", value=True)
target_alarms = st.sidebar.slider("Target alarms/hour", 0.5, 6.0, target_alarm0, 0.5)
threshold = st.sidebar.slider("Base threshold (if adaptive off)", 0.01, 0.99, thr0, 0.01)
conf_min = st.sidebar.slider("Minimum confidence for RED/GREEN", 0.0, 1.0, conf_min0, 0.05)
yellow_band = st.sidebar.slider("Yellow band below threshold", 0.0, 0.30, yellow_band0, 0.01)
refractory_min = st.sidebar.slider("Refractory (minutes)", 0, 30, refractory0, 1)
lookback_min = st.sidebar.slider("Lookback window (minutes)", 5, 20, lookback0, 1)
k_required = st.sidebar.slider("Episode votes required (k)", 1, 10, k_req0, 1)
m_window = st.sidebar.slider("Episode vote window (m)", 3, 20, m_win0, 1)

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input", ["Manual entry", "Upload CSV"])

colA, colB = st.columns([1, 1])

# -----------------------------
# CSV Upload Mode
# -----------------------------
def run_csv_simulation(df: pd.DataFrame):
    df = df.copy()
    for c in ["time"] + FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = df.sort_values("time").reset_index(drop=True)

    # reset session
    st.session_state.buf = pd.DataFrame(columns=["time"] + FEATURE_COLS)
    st.session_state.last_alarm_ts = -1e18
    st.session_state.recent_probs = deque(maxlen=600)
    st.session_state.votes = deque(maxlen=m_window)

    out_rows = []

    for i in range(len(df)):
        row = df.loc[i, ["time"] + FEATURE_COLS]
        st.session_state.buf = pd.concat([st.session_state.buf, row.to_frame().T], ignore_index=True)

        t_now = float(row["time"])
        feat = extract_features_from_buffer(st.session_state.buf, t_now, lookback_min=lookback_min)
        prob, conf, unc = predict_with_uncertainty(models, feat)

        st.session_state.recent_probs.append(prob)
        thr_eff = threshold
        if use_adapt:
            th_ad = adaptive_threshold(st.session_state.recent_probs, target_alarms_per_hour=target_alarms)
            if th_ad is not None:
                thr_eff = th_ad

        # episode voting gate
        is_high = (prob >= thr_eff)
        gated, st.session_state.votes = episode_gate(st.session_state.votes, is_high, k_required, m_window)

        # refractory
        allow_alarm = (t_now - st.session_state.last_alarm_ts) >= refractory_min * 60
        fired = False
        decision_label, decision_color, rec = decide_action(prob, conf, thr_eff, conf_min, yellow_band)
        pattern = classify_hypotension_pattern(st.session_state.buf, t_now)

        if decision_label.startswith("RED") and gated and allow_alarm:
            fired = True
            st.session_state.last_alarm_ts = t_now

        out_rows.append({
            "time": t_now,
            "prob": prob,
            "confidence": conf,
            "uncertainty": unc,
            "threshold_eff": thr_eff,
            "pattern": pattern,
            "decision": decision_label,
            "episode_gate": int(gated),
            "alarm_fired": int(fired)
        })

    return pd.DataFrame(out_rows)

# -----------------------------
# Manual Entry Mode
# -----------------------------
def manual_add_point(t_sec, map_v, hr_v, spo2_v, rr_v, etco2_v):
    row = {"time": float(t_sec), "MAP": map_v, "HR": hr_v, "SpO2": spo2_v, "RR": rr_v, "EtCO2": etco2_v}
    st.session_state.buf = pd.concat([st.session_state.buf, pd.DataFrame([row])], ignore_index=True)
    st.session_state.buf = st.session_state.buf.sort_values("time").reset_index(drop=True)

    t_now = float(t_sec)
    feat = extract_features_from_buffer(st.session_state.buf, t_now, lookback_min=lookback_min)
    prob, conf, unc = predict_with_uncertainty(models, feat)

    st.session_state.recent_probs.append(prob)

    thr_eff = threshold
    if use_adapt:
        th_ad = adaptive_threshold(st.session_state.recent_probs, target_alarms_per_hour=target_alarms)
        if th_ad is not None:
            thr_eff = th_ad

    is_high = (prob >= thr_eff)
    gated, st.session_state.votes = episode_gate(st.session_state.votes, is_high, k_required, m_window)

    allow_alarm = (t_now - st.session_state.last_alarm_ts) >= refractory_min * 60
    decision_label, decision_color, rec = decide_action(prob, conf, thr_eff, conf_min, yellow_band)
    pattern = classify_hypotension_pattern(st.session_state.buf, t_now)

    fired = False
    if decision_label.startswith("RED") and gated and allow_alarm:
        fired = True
        st.session_state.last_alarm_ts = t_now

    return prob, conf, unc, thr_eff, decision_label, decision_color, rec, pattern, gated, fired

# -----------------------------
# Main Layout
# -----------------------------
with colA:
    st.subheader("Data Input")

    if mode == "Upload CSV":
        up = st.file_uploader("Upload CSV with columns: time, MAP, HR, SpO2, RR, EtCO2", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            sim = run_csv_simulation(df)
            st.success("Simulation complete.")
            st.dataframe(sim.tail(30), use_container_width=True)

            total_hours = (sim["time"].max() - sim["time"].min()) / 3600.0 if len(sim) else 0.0
            alarms = int(sim["alarm_fired"].sum())
            st.metric("Total alarms", alarms)
            st.metric("Monitoring hours", round(total_hours, 2))
            st.metric("Alarms/hour", round(alarms / max(total_hours, 1e-9), 2))

            st.download_button(
                "Download simulation log CSV",
                sim.to_csv(index=False).encode("utf-8"),
                file_name="simulation_log.csv",
                mime="text/csv"
            )

    else:
        st.write("Enter one time-point. Time is in seconds from start (you control it).")
        c1, c2, c3 = st.columns(3)
        with c1:
            t_sec = st.number_input("time (sec)", min_value=0.0, value=float(st.session_state.buf["time"].max()+2 if len(st.session_state.buf) else 0.0), step=1.0)
            map_v = st.number_input("MAP", min_value=0.0, max_value=200.0, value=75.0, step=1.0)
        with c2:
            hr_v = st.number_input("HR", min_value=0.0, max_value=250.0, value=85.0, step=1.0)
            spo2_v = st.number_input("SpO2", min_value=0.0, max_value=100.0, value=98.0, step=1.0)
        with c3:
            rr_v = st.number_input("RR", min_value=0.0, max_value=80.0, value=14.0, step=1.0)
            etco2_v = st.number_input("EtCO2", min_value=0.0, max_value=80.0, value=36.0, step=1.0)

        if st.button("Add point & compute risk"):
            prob, conf, unc, thr_eff, dlab, dcol, rec, pattern, gated, fired = manual_add_point(
                t_sec, map_v, hr_v, spo2_v, rr_v, etco2_v
            )
            st.session_state.last_result = (prob, conf, unc, thr_eff, dlab, dcol, rec, pattern, gated, fired)

        if "last_result" in st.session_state:
            prob, conf, unc, thr_eff, dlab, dcol, rec, pattern, gated, fired = st.session_state.last_result
            st.subheader("Current Output")
            st.metric("Risk (mean prob)", f"{prob:.3f}")
            st.metric("Confidence", f"{conf:.3f}")
            st.metric("Uncertainty (std)", f"{unc:.3f}")
            st.metric("Effective threshold", f"{thr_eff:.3f}")
            st.write("Pattern type:", pattern)
            st.write("Episode gate:", "ON" if gated else "OFF")
            st.markdown(f"### Decision: :{dcol}[{dlab}]")
            st.info(rec)
            if fired:
                st.error("🚨 Alarm fired (refractory respected).")

        st.subheader("Recent buffer (tail)")
        st.dataframe(st.session_state.buf.tail(30), use_container_width=True)

with colB:
    st.subheader("Notes (Clinical-aware)")
    st.write(
        """
- **Decision layer**: GREEN/YELLOW/RED integrates probability + confidence.
- **Uncertainty-aware**: confidence is derived from ensemble disagreement (std across models).
- **Episode gating**: prevents single noisy window from triggering an alarm.
- **Adaptive threshold**: optional alarm-rate control to reduce alarm fatigue.
- **Pattern typing**: Type A/B/C provides interpretable MAP dynamics.
        """
    )

    st.subheader("Quick manual test values")
    st.code(
        "Example stable:\n"
        "time=0 MAP=80 HR=85 SpO2=98 RR=14 EtCO2=36\n"
        "time=2 MAP=79 HR=86 SpO2=98 RR=14 EtCO2=36\n\n"
        "Example gradual drop (Type B):\n"
        "time=0 MAP=80 ...\n"
        "time=60 MAP=75 ...\n"
        "time=120 MAP=70 ...\n"
        "time=180 MAP=66 ...\n"
        "time=240 MAP=63 ...\n",
        language="text"
    )
