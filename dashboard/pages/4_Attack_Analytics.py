"""dashboard/pages/4_Attack_Analytics.py"""
import sys, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
st.set_page_config(page_title="Attack Analytics | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Attack Analytics")

from dashboard.components.charts import attack_distribution_chart, class_bar_chart, confusion_matrix_chart, metrics_table_chart
from dashboard.components.model_loader import load_results

results = load_results()
clf_results = results.get("classifier_results", {})
history = st.session_state.get("prediction_history", [])

tab1, tab2, tab3 = st.tabs(["Predictions", "Model Performance", "Confusion Matrix"])

with tab1:
    if not history:
        st.info("No predictions yet. Use Data Input to generate predictions.")
        # Show training distribution as reference
        st.markdown("### Reference: Training Set Distribution")
        ref = {"Normal":56000,"Generic":40000,"Exploits":33393,"Fuzzers":18184,
               "DoS":12264,"Reconnaissance":10491,"Analysis":2000,"Backdoor":1746,"Shellcode":1133,"Worms":130}
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(attack_distribution_chart(ref), use_container_width=True)
        with col2: st.plotly_chart(class_bar_chart(ref), use_container_width=True)
    else:
        df = pd.DataFrame(history)
        counts = df["prediction"].value_counts().to_dict()
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(attack_distribution_chart(counts, "Predicted Distribution"), use_container_width=True)
        with col2: st.plotly_chart(class_bar_chart(counts), use_container_width=True)

with tab2:
    xgb = clf_results.get("xgboost", {})
    rf  = clf_results.get("random_forest", {})
    if xgb:
        st.markdown("### XGBoost vs Random Forest Comparison")
        col1, col2, col3 = st.columns(3)
        col1.metric("XGBoost Accuracy", f"{xgb.get('accuracy',0):.4f}")
        col2.metric("XGBoost Macro-F1", f"{xgb.get('macro_f1',0):.4f}")
        col3.metric("XGBoost Weighted-F1", f"{xgb.get('weighted_f1',0):.4f}")
        col1, col2, col3 = st.columns(3)
        col1.metric("RF Accuracy", f"{rf.get('accuracy',0):.4f}")
        col2.metric("RF Macro-F1", f"{rf.get('macro_f1',0):.4f}")
        col3.metric("RF Weighted-F1", f"{rf.get('weighted_f1',0):.4f}")
        per_class = xgb.get("per_class_report", {})
        if per_class:
            st.markdown("### XGBoost Per-Class Metrics")
            st.plotly_chart(metrics_table_chart(per_class), use_container_width=True)
    else:
        st.info("No classifier results found. Run experiments/train_classifiers.py first.")

with tab3:
    import os
    cm_path = Path(__file__).resolve().parent.parent.parent / "results" / "confusion_matrices" / "xgboost_confusion_matrix.csv"
    if cm_path.exists():
        cm_df = pd.read_csv(cm_path, index_col=0)
        cm = cm_df.values.astype(int)
        label_names = list(cm_df.columns)
        st.plotly_chart(confusion_matrix_chart(cm, label_names, "XGBoost Confusion Matrix (Test Set)"), use_container_width=True)
    else:
        st.info("Confusion matrix not yet generated. Run experiments/train_classifiers.py.")
