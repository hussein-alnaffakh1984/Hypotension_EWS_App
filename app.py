import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# App Config
# =========================
st.set_page_config(page_title="Hypotension EWS", layout="wide")

FEATURES = ["MAP", "HR", "SpO2", "RR", "EtCO2"]
LOOKBACK_MIN = 10
HORIZON_MIN = 10
LOOKBACK_SEC = LOOKBACK_MIN * 60

DEFAULT_THRESHOLD = 0.49603405237408627
DEFAULT_REFRACTORY_MIN = 5

# =========================
# Paths (Streamlit Cloud safe)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# =========================
# Header
# =========================
st.title("🩺 Intraoperative Hypotension Early Warning System")
st.caption(
    f"Procedure: Laparoscopic Cholecystectomy (GA) • Look-back: {LOOKBACK_MIN} min • "
    f"Horizon: {HORIZON_MIN} min • Hypotension: MAP < 65 mmHg sustained for ≥ 60 sec"
)

# =========================
# Load model (Pipeline) safely
# =========================
if not MODEL_PATH.exists():
    st.error("❌ Missing file: model.pkl. Upload it to the GitHub repo root (same folder as app.py).")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# Feature logic (auto-match to model.n_features_in_)
# =========================
def get_expected_n_features(pipeline_model) -> int:
    clf = getattr(pipeline_model, "named_steps", {}).get("clf", None)
    if clf is not None and hasattr(clf, "n_features_in_"):
        return int(clf.n_features_in_)
    # fallback
    return 5

def slope_per_signal(t, x):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    msk = np.isfinite(t) & np.isfinite(x)
    t = t[msk]
    x = x[msk]
    if len(x) < 2:
        return 0.0
    if np.allclose(t, t[0]):
        return 0.0
    try:
        return float(np.polyfit(t, x, 1)[0])
    except Exception:
        return 0.0

def build_feature_vector(past: pd.DataFrame, expected_n: int) -> np.ndarray:
    """
    expected_n:
      5  => mean per signal (MAP, HR, SpO2, RR, EtCO2)
      15 => mean+std+slope per signal (5*3)
    """
    past = past.sort_values("time")
    if expected_n == 5:
        return past[FEATURES].mean().values.reshape(1, -1)

    if expected_n == 15:
        feats = []
        t = past["time"].values.astype(float)
        for col in FEATURES:
            x = past[col].values.astype(float)
            feats.append(float(np.nanmean(x)))
            feats.append(float(np.nanstd(x)))
            feats.append(float(slope_per_signal(t, x)))
        return np.array(feats, dtype=float).reshape(1, -1)

    raise ValueError(f"Model expects {expected_n} features. App supports only 5 or 15.")

# =========================
# Session state
# =========================
if "buf" not in st.session_state:
    st.session_state.buf = pd.DataFrame(columns=["time"] + FEATURES + ["risk_prob", "alarm_raw", "alarm_final"])

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = None

if "t0_clock" not in st.session_state:
    st.session_state.t0_clock = None

# =========================
# Helpers
# =========================
def compute_risk_from_buffer(buf: pd.DataFrame, t_now: float) -> float:
    past = buf[(buf["time"] >= (t_now - LOOKBACK_SEC)) & (buf["time"] <= t_now)]
    if len(past) < 3:
        return 0.0

    expected_n = get_expected_n_features(model)
    feat = build_feature_vector(past, expected_n)

    prob = float(model.predict_proba(feat)[0, 1])
    if np.isnan(prob) or np.isinf(prob):
        return 0.0
    return prob

def apply_refractory(alarm_raw: int, t_now: float, refractory_sec: int) -> int:
    if alarm_raw != 1:
        return 0
    if st.session_state.last_alarm_time is None:
        st.session_state.last_alarm_time = t_now
        return 1
    if (t_now - st.session_state.last_alarm_time) >= refractory_sec:
        st.session_state.last_alarm_time = t_now
        return 1
    return 0

def alarm_burden(buf: pd.DataFrame):
    if len(buf) < 2:
        return 0, 0.0, 0.0
    total_alarms = int(buf["alarm_final"].fillna(0).sum())
    total_hours = (float(buf["time"].iloc[-1]) - float(buf["time"].iloc[0])) / 3600.0
    aph = (total_alarms / total_hours) if total_hours > 0 else 0.0
    return total_alarms, total_hours, aph

def validate_csv(df: pd.DataFrame):
    need = ["time"] + FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        return False, f"CSV is missing columns: {missing}"
    return True, ""

# =========================
# Diagnostics
# =========================
with st.expander("🔍 Diagnostics (deployment)", expanded=False):
    st.write("App folder:", str(BASE_DIR))
    try:
        st.write("Files in app folder:", os.listdir(BASE_DIR))
    except Exception as e:
        st.write("Could not list files:", e)
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("MODEL_EXISTS:", MODEL_PATH.exists())
    st.write("Expected n_features:", get_expected_n_features(model))
    st.write("Supported:", "5 (mean) or 15 (mean+std+slope)")

# =========================
# Sidebar
# =========================
st.sidebar.header("Controls")

mode = st.sidebar.radio("Input mode", ["Manual entry", "Upload CSV"], index=0)

threshold = st.sidebar.number_input(
    "Decision threshold",
    min_value=0.0, max_value=1.0,
    value=float(DEFAULT_THRESHOLD),
    step=0.01
)

refractory_min = st.sidebar.number_input(
    "Refractory (minutes)",
    min_value=0, max_value=30,
    value=int(DEFAULT_REFRACTORY_MIN),
    step=1
)
refractory_sec = int(refractory_min * 60)

st.sidebar.divider()
if st.sidebar.button("🧹 Reset session"):
    st.session_state.buf = pd.DataFrame(columns=["time"] + FEATURES + ["risk_prob", "alarm_raw", "alarm_final"])
    st.session_state.last_alarm_time = None
    st.session_state.t0_clock = None
    st.sidebar.success("Session reset.")

# =========================
# Layout
# =========================
left, right = st.columns([1, 2], gap="large")

# =========================
# Manual Mode
# =========================
if mode == "Manual entry":
    with left:
        st.subheader("Manual data entry")

        time_mode = st.selectbox(
            "Time input mode",
            ["Auto (fixed dt)", "Manual (enter time)", "Real clock (elapsed)"],
            index=0
        )

        dt_sec = st.number_input(
            "Auto mode dt (seconds)",
            min_value=1, max_value=60,
            value=2, step=1,
            help="Used only when time mode is Auto."
        )

        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            with c1:
                MAP = st.number_input("MAP (mmHg)", 0.0, 200.0, 75.0, 1.0)
                HR = st.number_input("HR (bpm)", 0.0, 250.0, 80.0, 1.0)
                SpO2 = st.number_input("SpO₂ (%)", 0.0, 100.0, 98.0, 1.0)
            with c2:
                RR = st.number_input("RR (/min)", 0.0, 60.0, 14.0, 1.0)
                EtCO2 = st.number_input("EtCO₂ (mmHg)", 0.0, 80.0, 35.0, 1.0)

            manual_time = None
            if time_mode == "Manual (enter time)":
                unit = st.selectbox("Manual time unit", ["minutes", "seconds"], index=0)
                manual_time = st.number_input("Time value", min_value=0.0, value=0.0, step=0.5)
                if unit == "minutes":
                    manual_time = manual_time * 60.0

            submitted = st.form_submit_button("➕ Add reading")

        if submitted:
            buf = st.session_state.buf.copy()

            # Determine time
            if time_mode == "Auto (fixed dt)":
                t = 0.0 if len(buf) == 0 else float(buf["time"].iloc[-1]) + float(dt_sec)
            elif time_mode == "Manual (enter time)":
                t = float(manual_time)
            else:
                now = datetime.now().timestamp()
                if st.session_state.t0_clock is None:
                    st.session_state.t0_clock = now
                    t = 0.0
                else:
                    t = float(now - st.session_state.t0_clock)

            # Append row
            new_row = {
                "time": t,
                "MAP": float(MAP),
                "HR": float(HR),
                "SpO2": float(SpO2),
                "RR": float(RR),
                "EtCO2": float(EtCO2),
                "risk_prob": np.nan,
                "alarm_raw": 0,
                "alarm_final": 0
            }
            buf = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)

            # Keep last 30 min for UI
            keep_sec = 30 * 60
            buf = buf[buf["time"] >= (t - keep_sec)].reset_index(drop=True)

            # Predict + alarms
            prob = compute_risk_from_buffer(buf, t)
            alarm_raw = int(prob >= threshold)
            alarm_final = apply_refractory(alarm_raw, t, refractory_sec)

            buf.loc[buf.index[-1], "risk_prob"] = prob
            buf.loc[buf.index[-1], "alarm_raw"] = alarm_raw
            buf.loc[buf.index[-1], "alarm_final"] = alarm_final

            st.session_state.buf = buf
            st.success("Reading added.")

# =========================
# CSV Mode
# =========================
else:
    with left:
        st.subheader("Upload CSV")
        st.write("CSV must contain: `time, MAP, HR, SpO2, RR, EtCO2`")
        up = st.file_uploader("Choose CSV file", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)
        ok, msg = validate_csv(df)
        if not ok:
            st.error(msg)
            st.stop()

        df = df.sort_values("time").reset_index(drop=True)

        # Reset refractory state for replay
        st.session_state.last_alarm_time = None
        st.session_state.t0_clock = None

        buf = df[["time"] + FEATURES].copy()
        buf["risk_prob"] = np.nan
        buf["alarm_raw"] = 0
        buf["alarm_final"] = 0

        for i in range(len(buf)):
            t = float(buf.loc[i, "time"])
            hist = buf.iloc[: i + 1].copy()

            prob = compute_risk_from_buffer(hist, t)
            alarm_raw = int(prob >= threshold)
            alarm_final = apply_refractory(alarm_raw, t, refractory_sec)

            buf.loc[i, "risk_prob"] = prob
            buf.loc[i, "alarm_raw"] = alarm_raw
            buf.loc[i, "alarm_final"] = alarm_final

        st.session_state.buf = buf
        st.success("CSV processed successfully.")

# =========================
# Outputs
# =========================
buf = st.session_state.buf

with right:
    st.subheader("Outputs")

    if len(buf) == 0:
        st.info("No data yet. Use Manual entry or Upload CSV.")
    else:
        last = buf.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk probability", f"{float(last['risk_prob']):.3f}" if pd.notna(last["risk_prob"]) else "—")
        c2.metric("Alarm (raw)", "YES" if int(last["alarm_raw"]) == 1 else "NO")
        c3.metric("Alarm (final)", "YES" if int(last["alarm_final"]) == 1 else "NO")

        total_alarms, total_hours, aph = alarm_burden(buf)
        c4.metric("Alarms/hour", f"{aph:.2f}")

        st.markdown(
            f"**Total alarms (final):** {total_alarms}  \n"
            f"**Monitoring time:** {total_hours:.2f} hours"
        )

        st.divider()
        st.subheader("Trend (MAP)")
        st.line_chart(buf.set_index("time")[["MAP"]])

        st.subheader("Recent rows")
        st.dataframe(buf.tail(30), use_container_width=True)

        st.download_button(
            "⬇️ Download session log CSV",
            data=buf.to_csv(index=False).encode("utf-8"),
            file_name="hypotension_ews_session_log.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Research prototype — for educational and validation use only (not for clinical decision-making).")
