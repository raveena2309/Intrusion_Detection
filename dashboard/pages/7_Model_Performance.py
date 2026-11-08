"""dashboard/pages/7_Model_Performance.py"""
import sys, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
st.set_page_config(page_title="Model Performance | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Model Performance")

from dashboard.components.charts import confusion_matrix_chart, metrics_table_chart
from dashboard.components.model_loader import load_results
import numpy as np

results = load_results()
clf = results.get("classifier_results", {})
anom = results.get("anomaly_results", {})
fusion = results.get("fusion_results", {})

tab1, tab2, tab3, tab4 = st.tabs(["Classifier", "Anomaly Detector", "Fusion", "Operational"])

with tab1:
    st.markdown("### Classifier Comparison (Official Test Set)")
    xgb = clf.get("xgboost", {})
    rf  = clf.get("random_forest", {})
    if xgb or rf:
        summary = pd.DataFrame({
            "Model": ["XGBoost", "Random Forest"],
            "Accuracy": [xgb.get("accuracy",0), rf.get("accuracy",0)],
            "Macro-F1": [xgb.get("macro_f1",0), rf.get("macro_f1",0)],
            "Weighted-F1": [xgb.get("weighted_f1",0), rf.get("weighted_f1",0)],
            "Train Time (s)": [xgb.get("training_time_s",0), rf.get("training_time_s",0)],
        })
        st.dataframe(summary.style.highlight_max(subset=["Accuracy","Macro-F1","Weighted-F1"],
                                                   color="#1a3a2a"), use_container_width=True)
        per_class = xgb.get("per_class_report", {})
        if per_class:
            st.markdown("### XGBoost Per-Class Metrics")
            st.plotly_chart(metrics_table_chart(per_class), use_container_width=True)
        cm_path = Path(__file__).resolve().parent.parent.parent / "results" / "confusion_matrices" / "xgboost_confusion_matrix.csv"
        if cm_path.exists():
            cm_df = pd.read_csv(cm_path, index_col=0)
            st.plotly_chart(confusion_matrix_chart(cm_df.values.astype(int), list(cm_df.columns),
                                                    "XGBoost Confusion Matrix (Test Set)"), use_container_width=True)
    else:
        st.info("Run experiments/train_classifiers.py to generate classifier results.")

with tab2:
    st.markdown("### Isolation Forest Anomaly Detector")
    if anom:
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("AUROC", f"{anom.get('auroc',0):.4f}")
        col2.metric("AUPRC", f"{anom.get('auprc',0):.4f}")
        col3.metric("Precision", f"{anom.get('precision',0):.4f}")
        col4.metric("Recall", f"{anom.get('recall',0):.4f}")
        st.markdown(f"**Selected Threshold:** `{anom.get('threshold',0):.4f}` (5% target FPR on normal-only val set)")
        st.markdown(f"**FPR on Normal Traffic:** `{anom.get('fpr_on_normal',0):.4f}`")
        per_class_det = anom.get("per_class_detection_rates", {})
        if per_class_det:
            det_df = pd.DataFrame(list(per_class_det.items()), columns=["Attack Category","Detection Rate"])
            st.dataframe(det_df, use_container_width=True)
    else:
        st.info("Run experiments/train_anomaly_detector.py to generate anomaly results.")

with tab3:
    st.markdown("### Decision Fusion Results")
    calib = fusion.get("calibration", {})
    if calib:
        col1,col2 = st.columns(2)
        col1.metric("ECE (Calibrated)", f"{calib.get('ece',0):.4f}")
        col2.metric("Brier Score", f"{calib.get('brier_score',0):.4f}")
    openset = fusion.get("openset_worms", {})
    if openset:
        st.markdown("### Open-Set Evaluation (Worms Withheld)")
        col1,col2,col3 = st.columns(3)
        col1.metric("Worms Recall", f"{openset.get('unknown_recall',0):.4f}")
        col2.metric("FPR", f"{openset.get('false_positive_rate',0):.4f}")
        col3.metric("Withheld Samples", f"{openset.get('n_withheld_samples',0)}")
    if fusion:
        fig_path = Path(__file__).resolve().parent.parent.parent / "results" / "figures" / "fusion_confusion_matrix.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Tuned Fusion Confusion Matrix", use_container_width=True)
    else:
        st.info("Run experiments/train_fusion.py to generate fusion results.")

with tab4:
    st.markdown("### Operational Metrics")
    op = fusion.get("operational", {})
    if op:
        col1,col2,col3 = st.columns(3)
        col1.metric("Throughput", f"{op.get('throughput_samples_per_sec',0):,.0f} flows/sec")
        col2.metric("P95 Latency/Record", f"{op.get('p95_latency_ms',0):.4f} ms")
        col3.metric("Mean Latency/Record", f"{op.get('mean_latency_ms',0):.4f} ms")
        st.caption("Measured as batch inference on the official test set (82,332 records).")
    models = st.session_state.get("models",{})
    clf_name = models.get("classifier_name","XGBoost")
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | Classifier | {clf_name} |
    | Anomaly Detector | Isolation Forest |
    | Fusion | Confidence-aware (tuned thresholds) |
    | Features | 42 (UNSW-NB15) |
    | Random Seed | 42 |
    | Dataset | UNSW-NB15 (official train/test split) |
    """)
