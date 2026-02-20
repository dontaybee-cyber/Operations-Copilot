import streamlit as st
import pandas as pd
import os
import json
import subprocess
import sys
from pathlib import Path

# Robust pathing for local + Streamlit Cloud (Linux)
PROJECT_ROOT = Path(__file__).resolve().parent
LOGO_PATH = PROJECT_ROOT / "logo.png"
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=200)

# --- Page Configuration (Branding Update) ---
st.set_page_config(
    page_title="DBAI Operations Copilot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- File Paths & Constants ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_FILE = os.path.join(DATA_DIR, 'master_ops_log.csv')
REPORT_FILE = os.path.join(DATA_DIR, 'anomalies_report.json')

# --- Gemini API Key (Streamlit Cloud secrets first, env fallback for local) ---
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Direct import from the modular_scripts package
from modular_scripts.analysis.anomaly_detector import get_insights_from_gemini


# --- Custom Styling (UI & Aesthetic Requirements) ---
def load_css():
    """Injects custom CSS for a high-contrast, modern UI."""
    css = """
    <style>
        /* Core Colors & Background */
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        /* Main header */
        h1 {
            color: #FFFFFF;
            border-bottom: 3px solid #A020F0; /* Neon Purple underline */
            padding-bottom: 10px;
        }

        /* Metric Cards */
        .metric-card {
            background-color: #1a1a1a;
            border: 1px solid #4B0082; /* Deep Violet border */
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .metric-card .label { font-size: 1.2rem; color: #FFFFFF; }
        .metric-card .value { font-size: 2.5rem; font-weight: bold; color: #A020F0; }

        /* AI Insight Box */
        .ai-insight {
            background-color: #1a1a1a;
            border-left: 5px solid #A020F0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .ai-insight h3 { color: #A020F0; }

        /* Custom Footer (Branding) */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #000000;
            color: #4B0082; /* Deep Violet text */
            text-align: center;
            padding: 10px;
            font-size: 0.9rem;
        }
        
        /* Hide default Streamlit footer */
        .st-emotion-cache-1cypcdb { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Data Loading & Processing ---
@st.cache_data(ttl=60)
def load_data():
    """Loads the master log and anomaly report from the data directory."""
    log_df = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame()
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, 'r') as f:
            anomalies = json.load(f)
        anomalies_df = pd.DataFrame(anomalies)
    else:
        anomalies_df = pd.DataFrame()
    return log_df, anomalies_df

def run_processing_pipeline():
    """Triggers the backend bulk processor and anomaly detector scripts."""
    with st.spinner('Analyzing new documents and detecting anomalies...'):
        try:
            # FIX: Using sys.executable to ensure the correct python env is used
            subprocess.run([sys.executable, "-m", "modular_scripts.bulk_processor"], check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, "-m", "modular_scripts.analysis.anomaly_detector"], check=True, capture_output=True, text=True)
            st.cache_data.clear()
            st.success("Data refreshed successfully!")
        except subprocess.CalledProcessError as e:
            st.error("An error occurred during data processing pipeline.")
            st.code(e.stderr)

# --- UI Rendering ---
load_css()

# Header
st.title("AI Operations Copilot")

if st.button("🔄 Refresh Data & Run Analysis"):
    run_processing_pipeline()

log_df, anomalies_df = load_data()

# --- Metrics Bar ---
st.markdown("---")
total_spend = log_df['total_amount'].sum() if 'total_amount' in log_df.columns else 0
detected_leaks = len(anomalies_df)
potential_savings = anomalies_df['total_amount'].sum() if 'total_amount' in anomalies_df.columns else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-card"><div class="label">Total Spend</div><div class="value">${total_spend:,.2f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="label">Detected Leaks</div><div class="value">{detected_leaks}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="label">Potential Savings</div><div class="value">${potential_savings:,.2f}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# --- Anomaly & Insight Section ---
if not anomalies_df.empty:
    st.subheader("🚨 Flagged Anomalies")
    
    with st.spinner('🤖 Generating AI summary for COO...'):
        ai_summary = get_insights_from_gemini(anomalies_df)
    
    st.markdown(f'<div class="ai-insight"><h3>AI-Generated Executive Summary</h3><p>{ai_summary}</p></div>', unsafe_allow_html=True)
    st.dataframe(anomalies_df, use_container_width=True)

elif not log_df.empty:
    st.success("✅ No anomalies detected in the current dataset.")
    st.subheader("Master Operations Log")
    st.dataframe(log_df, use_container_width=True)
    
else:
    st.info("📂 No data found. Please add PDF invoices to the '/data' folder and click 'Refresh Data'.")

# Branding Footer
st.markdown('<div class="footer">Powered by Dontay Beemon Automated Innovations (DBAI)</div>', unsafe_allow_html=True)
