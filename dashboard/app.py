"""
dashboard/app.py
=================
Main Streamlit application for the Intelligent NIDS Dashboard.
"""

import sys
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

st.set_page_config(
    page_title="Intelligent NIDS",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark cybersecurity theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    .stMetric { background-color: #1e2130; border-radius: 8px; padding: 12px; }
    .stSidebar { background-color: #12161f; }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 10px 15px;
    }
    .alert-high { color: #ff4b4b; font-weight: bold; }
    .alert-medium { color: #ffa500; font-weight: bold; }
    .alert-low { color: #2ecc71; font-weight: bold; }
    h1, h2, h3 { color: #00d4aa; }
    .stDataFrame { background-color: #1e2130; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# Sidebar
with st.sidebar:
    st.markdown("## NIDS Dashboard")
    st.markdown("---")

    # Load models
    from dashboard.components.model_loader import load_all_models, load_results
    models = load_all_models()
    st.session_state.models = models
    st.session_state.results = load_results()

    if models["ready"]:
        st.success("Models loaded")
        clf_name = models.get("classifier_name", "classifier")
        st.markdown(f"- **Classifier:** {clf_name}")
        st.markdown(f"- **Anomaly:** Isolation Forest")
        st.markdown(f"- **Features:** 42")
        st.markdown(f"- **Classes:** 10")
    else:
        st.warning("Some models not loaded")
        for m in models["missing"]:
            st.error(f"Missing: {m}")

    st.markdown("---")
    st.markdown("**Mode:** Dataset Batch")
    st.markdown("**Dataset:** UNSW-NB15")
    st.caption("Not live monitoring. Static predictions.")

# Main page - overview redirect
st.markdown("# Intelligent Network Intrusion Detection System")
st.markdown("### A Confidence-Aware Hybrid ML Framework")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("Use the **sidebar** to navigate between pages.")
with col2:
    st.info("**Upload a CSV** on the Data Input page to get predictions.")
with col3:
    st.info("**Research mode** results available in Model Performance page.")

st.markdown("---")
st.markdown("""
This dashboard demonstrates a hybrid NIDS combining:
- **Multiclass classifier** (XGBoost) for known attack identification
- **Isolation Forest** for anomaly detection (trained on normal traffic only)
- **Confidence-aware fusion** for open-set unknown attack flagging
- **SHAP explainability** for prediction transparency

> **Disclaimer:** This is a research system evaluated on the UNSW-NB15 benchmark.
> Model predictions reflect dataset patterns, not real-world network security guarantees.
""")
