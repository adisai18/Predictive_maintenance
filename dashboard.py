"""
BLDC Fault Detection Dashboard
Run: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="BLDC Fault Detection",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4f8; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 36px !important; max-width: 740px; }

/* ── Header ── */
.dash-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2540 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(15,37,64,0.18);
}
.dash-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #00c6ff, #0072ff, #7c3aed);
}
.dash-icon { font-size: 42px; margin-bottom: 10px; }
.dash-title {
    font-size: 26px; font-weight: 700;
    color: #ffffff; margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.dash-title span { color: #38bdf8; }
.dash-sub { font-size: 13px; color: #93c5fd; margin: 0; font-weight: 400; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
}
.card-title {
    font-size: 11px; font-weight: 700;
    color: #94a3b8; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: 22px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] input {
    background: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    color: #0f172a !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 10px 14px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #0072ff !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(0,114,255,0.10) !important;
}
label {
    color: #475569 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0072ff, #00c6ff) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 0 !important;
    width: 100% !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 16px rgba(0,114,255,0.30) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(0,114,255,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
.divider { height: 1px; background: #e2e8f0; margin: 4px 0 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model with 5-Fold CV...")
def train_model():
    df_raw = pd.read_excel("BLDC_4Fault_Dataset.xlsx")
    FEATURES    = ["Voltage", "Current", "Temperature", "Vibration", "Speed", "Torque"]
    FAULT_ORDER = ["Normal", "Bearing Fault", "Single Winding Fault", "Double Winding Fault"]
    NOISE_SCALE = 0.7
    SEED        = 42

    rng = np.random.default_rng(SEED)
    df  = df_raw.copy()
    for feat in FEATURES:
        df[feat] = df[feat] + rng.normal(0, df[feat].std() * NOISE_SCALE, size=len(df))

    le = LabelEncoder()
    le.fit(FAULT_ORDER)
    X = df[FEATURES].values
    y = le.transform(df["Condition"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        )),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cross_validate(pipeline, X_train, y_train, cv=cv, scoring=["accuracy"], return_train_score=True)
    pipeline.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test)) * 100
    return pipeline, le, FEATURES, round(test_acc, 2)

pipeline, le, FEATURES, test_acc = train_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dash-header">'
    '<div class="dash-icon">⚡</div>'
    '<p class="dash-title"><span>BLDC Motor</span> Fault Detection</p>'
    '<p class="dash-sub">AI-Based Predictive Maintenance &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; 5-Fold CV &nbsp;·&nbsp; Accuracy: ' + str(test_acc) + '%</p>'
    '</div>',
    unsafe_allow_html=True
)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">🔧 &nbsp;Enter Sensor Readings</div><div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    voltage     = st.number_input("⚡  Voltage (V)",         min_value=0.0,   max_value=100.0,  value=48.0,   step=0.5)
    temperature = st.number_input("🌡️  Temperature (°C)",    min_value=0.0,   max_value=200.0,  value=42.0,   step=1.0)
    speed       = st.number_input("🔄  Speed (RPM)",          min_value=0.0,   max_value=5000.0, value=3000.0, step=10.0)
with col2:
    current     = st.number_input("🔌  Current (A)",          min_value=0.0,   max_value=50.0,   value=8.0,    step=0.5)
    vibration   = st.number_input("📳  Vibration (g)",        min_value=0.0,   max_value=10.0,   value=0.3,    step=0.1)
    torque      = st.number_input("⚙️  Torque (Nm)",          min_value=0.0,   max_value=10.0,   value=4.0,    step=0.1)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Predict Fault Condition", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── Result ────────────────────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame(
        [[voltage, current, temperature, vibration, speed, torque]],
        columns=FEATURES
    )
    pred_enc   = pipeline.predict(input_df)[0]
    prediction = le.inverse_transform([pred_enc])[0]
    proba      = pipeline.predict_proba(input_df)[0]
    confidence = max(proba) * 100

    # Config per class: bg_light, bg_dark, border, text_color, bar_color, icon, status, message
    cfg = {
        "Normal": (
            "#f0fdf4", "#dcfce7", "#86efac", "#15803d", "#22c55e",
            "✅", "NORMAL OPERATION",
            "The motor is operating within normal parameters. No maintenance required."
        ),
        "Bearing Fault": (
            "#fffbeb", "#fef3c7", "#fcd34d", "#92400e", "#f59e0b",
            "⚠️", "BEARING FAULT DETECTED",
            "Bearing wear or damage detected. Schedule preventive maintenance soon."
        ),
        "Single Winding Fault": (
            "#fff7ed", "#ffedd5", "#fdba74", "#9a3412", "#f97316",
            "🔥", "SINGLE WINDING FAULT",
            "Winding insulation fault detected. Inspect and test motor windings."
        ),
        "Double Winding Fault": (
            "#fef2f2", "#fee2e2", "#fca5a5", "#991b1b", "#ef4444",
            "🚨", "DOUBLE WINDING FAULT",
            "Critical fault detected! Shut down motor immediately and inspect."
        ),
    }

    bg_l, bg_d, border, txt, bar, icon, status, msg = cfg[prediction]
    conf_w   = "{:.1f}".format(confidence)
    conf_str = "{:.1f}%".format(confidence)

    st.markdown(
        '<div style="background:' + bg_l + ';border:2px solid ' + border + ';'
        'border-radius:18px;padding:36px 40px;margin-top:4px;'
        'box-shadow:0 4px 24px rgba(0,0,0,0.08);">'

        # Icon + status badge
        '<div style="text-align:center;margin-bottom:20px;">'
        '<div style="font-size:56px;margin-bottom:12px;line-height:1;">' + icon + '</div>'
        '<div style="display:inline-block;background:' + bg_d + ';border:1.5px solid ' + border + ';'
        'border-radius:30px;padding:5px 18px;'
        'font-size:11px;font-weight:800;color:' + txt + ';letter-spacing:1.5px;">'
        + status +
        '</div>'
        '</div>'

        # Fault name
        '<div style="text-align:center;font-size:28px;font-weight:800;color:#0f172a;'
        'margin-bottom:10px;letter-spacing:-0.3px;">' + prediction + '</div>'

        # Message
        '<div style="text-align:center;font-size:14px;color:#475569;'
        'margin-bottom:28px;line-height:1.6;">' + msg + '</div>'

        # Divider
        '<div style="height:1px;background:' + border + ';opacity:0.4;margin-bottom:20px;"></div>'

        # Confidence row
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        '<span style="font-size:12px;font-weight:700;color:#64748b;letter-spacing:0.5px;">MODEL CONFIDENCE</span>'
        '<span style="font-family:JetBrains Mono,monospace;font-size:18px;font-weight:800;color:' + txt + ';">' + conf_str + '</span>'
        '</div>'
        '<div style="background:#e2e8f0;border-radius:20px;height:10px;overflow:hidden;">'
        '<div style="background:' + bar + ';height:10px;width:' + conf_w + '%;'
        'border-radius:20px;transition:width 0.6s ease;"></div>'
        '</div>'

        # Input summary
        '<div style="margin-top:24px;background:' + bg_d + ';border-radius:12px;padding:16px 20px;">'
        '<div style="font-size:11px;font-weight:700;color:' + txt + ';letter-spacing:1px;margin-bottom:12px;">INPUT READINGS</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">'
        '<div style="text-align:center;">'
        '<div style="font-size:11px;color:#64748b;font-weight:600;">VOLTAGE</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#0f172a;">' + "{:.1f}".format(voltage) + ' V</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="font-size:11px;color:#64748b;font-weight:600;">CURRENT</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#0f172a;">' + "{:.1f}".format(current) + ' A</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="font-size:11px;color:#64748b;font-weight:600;">TEMPERATURE</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#0f172a;">' + "{:.1f}".format(temperature) + ' °C</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="font-size:11px;color:#64748b;font-weight:600;">VIBRATION</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#0f172a;">' + "{:.2f}".format(vibration) + ' g</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="font-size:11px;color:#64748b;font-weight:600;">SPEED</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#0f172a;">' + "{:.0f}".format(speed) + ' RPM</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="font-size:11px;color:#64748b;font-weight:600;">TORQUE</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#0f172a;">' + "{:.1f}".format(torque) + ' Nm</div>'
        '</div>'
        '</div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True
    )