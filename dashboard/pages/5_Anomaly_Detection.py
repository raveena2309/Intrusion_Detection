"""dashboard/pages/5_Anomaly_Detection.py"""
import sys, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
st.set_page_config(page_title="Anomaly Detection | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Anomaly Detection")

from dashboard.components.charts import anomaly_score_histogram
from dashboard.components.model_loader import load_results

results = load_results()
anom_results = results.get("anomaly_results", {})
fusion_results = results.get("fusion_results", {})

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUROC", f"{anom_results.get('auroc', 0):.4f}")
col2.metric("AUPRC", f"{anom_results.get('auprc', 0):.4f}")
col3.metric("Precision", f"{anom_results.get('precision', 0):.4f}")
col4.metric("FPR on Normal", f"{anom_results.get('fpr_on_normal', 0):.4f}")

st.markdown("---")
tab1, tab2 = st.tabs(["Score Distribution", "Open-Set Evaluation"])

with tab1:
    threshold = anom_results.get("threshold", 0.5)
    st.markdown(f"**Selected Threshold:** `{threshold:.4f}` (chosen at 5% target FPR on normal-only validation)")
    st.info("Score distribution from actual test set predictions.")
    history = st.session_state.get("prediction_history", [])
    if history:
        df = pd.DataFrame(history)
        scores = df["anomaly_score"].values
        st.plotly_chart(anomaly_score_histogram(scores, threshold), use_container_width=True)
        flagged = df[df["is_anomaly"] == True]
        st.markdown(f"**Flagged records:** {len(flagged):,} / {len(df):,}")
        if not flagged.empty:
            st.dataframe(flagged.head(20), use_container_width=True)
    else:
        st.info("Run predictions on the Data Input page to see score distributions.")
        st.markdown("### From Experiment Results (Test Set)")
        st.markdown(f"- **Threshold:** {threshold:.4f}")
        st.markdown(f"- **AUROC:** {anom_results.get('auroc', 'N/A')}")
        st.markdown(f"- **Detection Rate on Generic:** {anom_results.get('per_class_detection_rates', {}).get('Generic', 'N/A')}")
        st.markdown(f"- **Detection Rate on Worms:** {anom_results.get('per_class_detection_rates', {}).get('Worms', 'N/A')}")

with tab2:
    st.markdown("### Track B: Open-Set Evaluation (Worms Withheld)")
    openset = fusion_results.get("openset_worms", {})
    if openset:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Worms Recall", f"{openset.get('unknown_recall', 0):.4f}")
        col2.metric("False Positive Rate", f"{openset.get('false_positive_rate', 0):.4f}")
        col3.metric("Precision (Unknown)", f"{openset.get('precision_unknown', 0):.6f}")
        col4.metric("Withheld Samples", f"{openset.get('n_withheld_samples', 0)}")
        st.markdown("""
        > **Interpretation:** High unknown-attack recall at the cost of elevated FPR reflects the
        > fundamental trade-off in open-set detection. The classifier was trained without Worms examples;
        > the fusion system flags these via anomaly score + low confidence.
        """)
        st.markdown("### Fusion Comparison")
        simple = fusion_results.get("simple_fusion", {})
        tuned = fusion_results.get("tuned_fusion", {})
        comparison = pd.DataFrame({
            "Condition": ["Simple Fusion (default thresholds)", "Tuned Fusion (optimized thresholds)"],
            "Known Accuracy": [simple.get("known_accuracy", "-"), tuned.get("known_accuracy", "-")],
            "Unknown Recall": [simple.get("unknown_recall", "-"), tuned.get("unknown_recall", "-")],
            "FPR": [simple.get("false_positive_rate", "-"), tuned.get("false_positive_rate", "-")],
        })
        st.dataframe(comparison, use_container_width=True)
    else:
        st.info("Run experiments/train_fusion.py to generate open-set evaluation results.")
