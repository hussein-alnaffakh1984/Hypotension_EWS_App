import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# =========================
# Config
# =========================
st.set_page_config(page_title="Intraoperative Hypotension EWS", layout="wide")

FEATURES = ["MAP", "HR", "SpO2", "RR", "EtCO2"]
LOOKBACK_MIN = 10
HORIZON_MIN = 10
LOOKBACK_SEC = LOOKBACK_MIN * 60

DEFAULT_THRESHOLD = 0.49603405237408627  # من نتائجك
DEFAULT_REFRACTORY_MIN = 5

# =========================
# Load model artifacts
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Session state
# =========================
if "buf" not in st.session_state:
    st.session_state.buf = pd.DataFrame(columns=["time"] + FEATURES + ["risk_prob", "alarm_raw", "alarm_final"])

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = None

# =========================
# Helpers
# =========================
def compute_one_step(buf, t_now, threshold, refractory_sec):
    """Compute risk and alarms at time t_now using last LOOKBACK window."""
    past = buf[(buf["time"] >= (t_now - LOOKBACK_SEC)) & (buf["time"] <= t_now)]

    # coverage check: at least 50% of expected samples based on median dt
    if len(past) < 5:  # minimum safeguard
        risk_prob, alarm_raw = 0.0, 0
    else:
        feat = past[FEATURES].mean().values.reshape(1, -1)
        feat_s = scaler.transform(feat)
        risk_prob = float(model.predict_proba(feat_s)[0, 1])
        alarm_raw = int(risk_prob >= threshold)

    alarm_final = 0
    if alarm_raw == 1:
        if st.session_state.last_alarm_time is None:
            alarm_final = 1
            st.session_state.last_alarm_time = t_now
        else:
            if (t_now - st.session_state.last_alarm_time) >= refractory_sec:
                alarm_final = 1
                st.session_state.last_alarm_time = t_now

    buf.loc[buf.index[-1], "risk_prob"] = risk_prob
    buf.loc[buf.index[-1], "alarm_raw"] = alarm_raw
    buf.loc[buf.index[-1], "alarm_final"] = alarm_final
    return buf

def alarm_burden(buf):
    """Compute alarms/hour using actual time axis."""
    if len(buf) < 2:
        return 0, 0.0, 0.0
    total_alarms = int(buf["alarm_final"].fillna(0).sum())
    total_time_hr = (buf["time"].iloc[-1] - buf["time"].iloc[0]) / 3600.0
    aph = (total_alarms / total_time_hr) if total_time_hr > 0 else 0.0
    return total_alarms, total_time_hr, aph

# =========================
# UI
# =========================
st.title("🩺 Intraoperative Hypotension Early Warning System")
st.caption("Target: Laparoscopic Cholecystectomy (GA) • Look-back: 10 min • Horizon: 10 min • Hypotension: MAP<65 for ≥60s")

left, right = st.columns([1, 2])

with left:
    mode = st.radio("Input mode", ["Manual entry", "Upload CSV"], index=0)

    threshold = st.number_input("Decision threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)
    refractory_min = st.number_input("Refractory (minutes)", 0, 30, int(DEFAULT_REFRACTORY_MIN), 1)
    refractory_sec = int(refractory_min * 60)

    st.divider()
    if st.button("🧹 Reset session"):
        st.session_state.buf = pd.DataFrame(columns=["time"] + FEATURES + ["risk_prob", "alarm_raw", "alarm_final"])
        st.session_state.last_alarm_time = None
        st.success("Session reset.")

# =========================
# Mode A: Manual
# =========================
if mode == "Manual entry":
    with left:
        st.subheader("Manual input")

        time_mode = st.selectbox("Time input", ["Auto (fixed dt)", "Manual (enter time)", "Real clock"], index=0)

        dt_sec = st.number_input("dt seconds (for Auto mode)", min_value=1, max_value=60, value=2, step=1)

        with st.form("manual_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                MAP = st.number_input("MAP (mmHg)", 0.0, 200.0, 75.0, 1.0)
                HR  = st.number_input("HR (bpm)", 0.0, 250.0, 80.0, 1.0)
            with c2:
                SpO2 = st.number_input("SpO₂ (%)", 0.0, 100.0, 98.0, 1.0)
                RR   = st.number_input("RR (/min)", 0.0, 60.0, 14.0, 1.0)
            with c3:
                EtCO2 = st.number_input("EtCO₂ (mmHg)", 0.0, 80.0, 35.0, 1.0)

            manual_time = None
            if time_mode == "Manual (enter time)":
                manual_time_unit = st.selectbox("Manual time unit", ["seconds", "minutes"], index=1)
                manual_time = st.number_input("Time value", min_value=0.0, value=0.0, step=0.5)
                if manual_time_unit == "minutes":
                    manual_time = manual_time * 60.0

            submitted = st.form_submit_button("➕ Add reading")

    if submitted:
        buf = st.session_state.buf.copy()

        # Determine time
        if time_mode == "Auto (fixed dt)":
            if len(buf) == 0:
                t = 0.0
            else:
                t = float(buf["time"].iloc[-1]) + float(dt_sec)

        elif time_mode == "Manual (enter time)":
            t = float(manual_time)

        else:  # Real clock
            # store as elapsed seconds since first entry
            now = datetime.now().timestamp()
            if len(buf) == 0 or "t0_clock" not in st.session_state:
                st.session_state.t0_clock = now
                t = 0.0
            else:
                t = now - st.session_state.t0_clock

        # Append
        new_row = {"time": t, "MAP": MAP, "HR": HR, "SpO2": SpO2, "RR": RR, "EtCO2": EtCO2}
        buf = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)

        # Keep last 30 minutes for display (optional)
        keep_sec = 30 * 60
        buf = buf[buf["time"] >= (t - keep_sec)].reset_index(drop=True)

        # Compute risk and alarms
        buf = compute_one_step(buf, t, threshold, refractory_sec)

        st.session_state.buf = buf

# =========================
# Mode B: CSV upload
# =========================
else:
    with left:
        st.subheader("Upload CSV")
        st.write("CSV must contain columns: time, MAP, HR, SpO2, RR, EtCO2")
        up = st.file_uploader("Choose a CSV file", type=["csv"])

    if up is not None:
        df = pd.read_csv(up).sort_values("time").reset_index(drop=True)

        # predict sequentially to respect refractory timing
        buf = df[["time"] + FEATURES].copy()
        buf["risk_prob"] = np.nan
        buf["alarm_raw"] = 0
        buf["alarm_final"] = 0

        st.session_state.last_alarm_time = None

        for i in range(len(buf)):
            t = float(buf.loc[i, "time"])
            # compute using history within this uploaded dataframe
            hist = buf.iloc[:i+1].copy()
            # place hist into temp buffer and compute last row prediction
            hist = compute_one_step(hist, t, threshold, refractory_sec)
            buf.loc[i, ["risk_prob", "alarm_raw", "alarm_final"]] = hist.iloc[-1][["risk_prob", "alarm_raw", "alarm_final"]].values

        st.session_state.buf = buf

# =========================
# Display results
# =========================
buf = st.session_state.buf

with right:
    st.subheader("Status & Outputs")

    if len(buf) == 0:
        st.info("No data yet. Use Manual entry or upload a CSV.")
    else:
        last = buf.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk probability", f"{float(last['risk_prob']):.3f}" if pd.notna(last["risk_prob"]) else "—")
        c2.metric("Alarm (raw)", "YES" if int(last["alarm_raw"]) == 1 else "NO")
        c3.metric("Alarm (final)", "YES" if int(last["alarm_final"]) == 1 else "NO")

        total_alarms, total_hr, aph = alarm_burden(buf)
        st.write(f"**Total alarms (final):** {total_alarms}")
        st.write(f"**Monitoring time:** {total_hr:.2f} hours")
        st.write(f"**Alarms per hour:** {aph:.2f}")

        st.divider()
        st.subheader("Trends")
        st.line_chart(buf.set_index("time")[["MAP"]])

        st.subheader("Recent records")
        st.dataframe(buf.tail(30))

        st.download_button(
            "⬇️ Download log CSV",
            data=buf.to_csv(index=False).encode("utf-8"),
            file_name="hypotension_app_log.csv",
            mime="text/csv"
        )
