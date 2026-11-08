"""dashboard/pages/6_Explainability.py"""
import sys, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
st.set_page_config(page_title="Explainability | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Explainability")
st.warning("Feature contributions describe model behaviour, not proof of malicious intent.")

from dashboard.components.charts import shap_importance_chart, local_shap_chart
from dashboard.components.model_loader import load_shap_artifacts

LABEL_NAMES = ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Normal","Reconnaissance","Shellcode","Worms"]

shap_data = load_shap_artifacts()
tab1, tab2 = st.tabs(["Global Feature Importance", "Local Explanation"])

with tab1:
    shap_png = Path(__file__).resolve().parent.parent.parent / "results" / "figures" / "shap_summary.png"
    imp_png  = Path(__file__).resolve().parent.parent.parent / "results" / "figures" / "shap_importance.png"
    if shap_data:
        sv = shap_data.get("shap_values", None)
        feature_names = shap_data.get("feature_names", [])
        if sv is not None and len(feature_names) > 0:
            if sv.ndim == 3:
                mean_abs = np.abs(sv).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(sv).mean(axis=0)
            st.plotly_chart(shap_importance_chart(feature_names, mean_abs, top_n=15), use_container_width=True)
            top5 = np.argsort(mean_abs)[::-1][:5]
            st.markdown("**Top 5 most influential features:**")
            for rank, i in enumerate(top5, 1):
                st.markdown(f"{rank}. **{feature_names[i]}** — mean |SHAP| = {float(mean_abs[i]):.4f}")
    elif shap_png.exists():
        st.image(str(shap_png), caption="SHAP Summary Plot (from experiment)", use_container_width=True)
    else:
        st.info("Run experiments/generate_shap.py to compute SHAP values.")

    if imp_png.exists():
        st.image(str(imp_png), caption="SHAP Feature Importance", use_container_width=True)

with tab2:
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("Run predictions first (Data Input page) to see local explanations.")
    else:
        df = pd.DataFrame(history)
        idx = st.slider("Select record index", 0, len(df)-1, 0)
        record = df.iloc[idx]
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Decision", record.get("prediction","N/A"))
        col2.metric("Confidence", f"{record.get('confidence',0):.4f}")
        col3.metric("Anomaly Score", f"{record.get('anomaly_score',0):.4f}")

        models = st.session_state.get("models", {})
        if shap_data and models.get("ready"):
            sv = shap_data.get("shap_values", None)
            fn = shap_data.get("feature_names", [])
            Xs = shap_data.get("X_sample", None)
            if sv is not None and Xs is not None and idx < len(sv):
                sv_single = sv[idx] if sv.ndim == 2 else sv[idx].sum(axis=1)
                st.plotly_chart(local_shap_chart(fn, sv_single, Xs[idx], top_n=10), use_container_width=True)
                st.caption("SHAP values computed on the first 500 test records. Local explanation available for those records only.")
            else:
                st.info("Local SHAP not available for this record index (SHAP computed on first 500 records).")
        else:
            st.info("SHAP artifacts or models not loaded.")
