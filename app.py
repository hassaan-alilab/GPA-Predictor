# SCHOLASTIC STATS — Student Performance Analytics System
# Probability & Statistics Semester Project — Spring 2026
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from tab1_overview     import render_tab1
from tab2_visuals      import render_tab2
from tab3_stats        import render_tab3
from tab4_distribution import render_tab4
from tab5_model        import render_tab5

st.set_page_config(
    page_title="Student Performance Analytics(SCHOLASTIC STATS)",
    page_icon="📊",
    layout="wide"
)

# ── Premium dark UI styling ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
  }

  .hero-banner {
    background: linear-gradient(120deg, #1a1a2e, #16213e, #0f3460);
    border: 1px solid rgba(100,180,255,0.18);
    border-radius: 20px;
    padding: 2.8rem 2rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.45);
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -60%; left: -60%;
    width: 220%; height: 220%;
    background: radial-gradient(circle, rgba(100,180,255,0.07) 0%, transparent 60%);
    animation: pulse 6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { transform: scale(1);   opacity: 0.6; }
    50%      { transform: scale(1.1); opacity: 1;   }
  }
  .main-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #64b4ff, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; letter-spacing: -0.5px;
  }
  .sub-title {
    font-size: 1.05rem; color: rgba(200,220,255,0.72);
    margin-top: 0.5rem; font-weight: 300; letter-spacing: 0.5px;
  }
  .badge-row { margin-top: 1rem; display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; }
  .badge {
    background: rgba(100,180,255,0.12);
    border: 1px solid rgba(100,180,255,0.28);
    color: #93c5fd; padding: 0.25rem 0.85rem;
    border-radius: 999px; font-size: 0.78rem; font-weight: 500;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px; padding: 4px 6px; gap: 4px;
    border: 1px solid rgba(255,255,255,0.07);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px; color: rgba(200,220,255,0.65);
    font-weight: 500; font-size: 0.88rem;
    padding: 0.5rem 1.1rem; transition: all 0.2s;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(100,180,255,0.12);
  }
  [data-testid="stSidebar"] * { color: rgba(200,220,255,0.85) !important; }
  .sidebar-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(100,180,255,0.15);
    border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;
  }

  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(100,180,255,0.18);
    border-radius: 14px; padding: 1rem 1.2rem;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(37,99,235,0.25);
  }
  [data-testid="stMetricLabel"] { color: rgba(180,210,255,0.7) !important; font-size:0.82rem !important; }
  [data-testid="stMetricValue"] { color: #93c5fd !important; font-weight: 700 !important; font-size:1.6rem !important; }

  [data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(100,180,255,0.15);
  }

  .highlight {
    background: rgba(99,102,241,0.12);
    border-left: 4px solid #818cf8;
    border-radius: 8px; padding: 0.75rem 1rem;
    color: #c7d2fe; font-size: 0.9rem; margin-top: 0.6rem;
  }
  .section-head {
    font-size: 1.15rem; font-weight: 700; color: #93c5fd;
    border-bottom: 2px solid rgba(100,180,255,0.2);
    padding-bottom: 0.35rem; margin-bottom: 0.8rem;
  }

  .stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white; border: none; border-radius: 12px;
    font-weight: 600; font-size: 1rem;
    padding: 0.7rem 1.5rem; transition: all 0.25s;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35);
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(37,99,235,0.55);
  }

  .stAlert { border-radius: 12px; border: none; }
  hr { border-color: rgba(100,180,255,0.12) !important; }
  h3 { color: #bfdbfe !important; }
  h4 { color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="main-title">📊 SCHOLASTIC STATS</p>
  <p class="sub-title">Probability &amp; Statistics Semester Project — Spring 2026</p>
  <div class="badge-row">
    <span class="badge">500 Students</span>
    <span class="badge">7 Features</span>
    <span class="badge">GPA Prediction</span>
    <span class="badge">80 / 20 Split</span>
    <span class="badge">Linear Regression</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("### 📂 Data Source")

    data_source = st.radio(
        "Choose source",
        ["Use attached dataset (dataset.csv)", "Upload my own CSV"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    sep = ","

    if data_source == "Upload my own CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        sep = st.selectbox("Separator", [",", ";", "\t"], index=0)

    st.divider()
    st.markdown("### 🔧 Model Settings")

    ci_level    = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.markdown("""
<div class="sidebar-card">
  <b style="color:#93c5fd">📌 Split Strategy</b><br>
  <span style="font-size:0.85rem">Train: <b>80%</b> &nbsp;|&nbsp; Test: <b>20%</b></span>
</div>
""", unsafe_allow_html=True)

    test_size = 0.20

# ── Column definitions ───────────────────────────────────────
ALL_FEATURES  = ["Age", "StudyHoursPerWeek", "AttendanceRate", "SocialMediaUsage", "PhysicalActivity", "SleepHours", "MentalHealthScore"]
TARGET        = "GPA"
EXPECTED_COLS = ["StudentID"] + ALL_FEATURES + [TARGET]


# ── Data loading helpers ─────────────────────────────────────
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    """Load and clean the bundled dataset.csv file."""
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    present = [c for c in EXPECTED_COLS if c in df.columns]
    df.dropna(subset=present, inplace=True)
    return df


@st.cache_data
def load_uploaded(data: bytes, sep: str) -> pd.DataFrame:
    """Load and clean a user-uploaded CSV file."""
    import io
    df = pd.read_csv(io.BytesIO(data), sep=sep)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df


# ── Load data based on user selection ───────────────────────
if "attached dataset" in data_source:
    df = load_dataset("dataset.csv")
    st.info("ℹ️ Using **dataset.csv** — 500 students, 7 features → GPA.")
else:
    if uploaded_file is None:
        st.warning("⬅️ Please upload a CSV file from the sidebar to continue.")
        st.stop()
    df = load_uploaded(uploaded_file.read(), sep)
    st.success(f"✅ Dataset loaded — {len(df)} rows · {df.shape[1]} columns.")

# Only use features that are actually present in the loaded dataframe
FEATURES = [f for f in ALL_FEATURES if f in df.columns]

# ── Tab layout ───────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Data Overview",
    "📈  Visualizations",
    "📐  Statistics & CI",
    "🔔  Distributions",
    "🤖  Regression & Prediction"
])

with tab1:
    render_tab1(df, TARGET, FEATURES)

with tab2:
    render_tab2(df, TARGET, FEATURES)

with tab3:
    render_tab3(df, TARGET, FEATURES, ci_level)

with tab4:
    render_tab4(df, TARGET)

with tab5:
    render_tab5(df, TARGET, FEATURES, test_size, random_seed)

st.divider()
st.markdown(
    "<center><small style='color:rgba(180,210,255,0.4)'>"
    "SCHOLASTIC STATS &nbsp;|&nbsp; Probability &amp; Statistics — Spring 2026"
    "</small></center>",
    unsafe_allow_html=True
)