"""
dashboard/pages/1_Overview.py
"""
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.set_page_config(page_title="Overview | NIDS", page_icon="shield", layout="wide")

st.markdown("""
<style>
.stApp{background-color:#0e1117}
div[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3250;border-radius:8px;padding:10px 15px}
h1,h2,h3{color:#00d4aa}
</style>""", unsafe_allow_html=True)

st.title("Overview")
st.caption("Mode: Dataset Batch | Dataset: UNSW-NB15")
st.markdown("---")

history = st.session_state.get("prediction_history", [])

if not history:
    st.info("No predictions yet. Go to **Data Input** to upload a CSV and run predictions.")
    st.markdown("### Dataset Statistics (Training Set)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records (Train)", "175,341")
    col2.metric("Normal Traffic", "56,000 (31.9%)")
    col3.metric("Attack Records", "119,341 (68.1%)")
    col4.metric("Attack Categories", "9")
    st.markdown("### Class Distribution (Training Set)")
    class_counts = {
        "Normal":56000,"Generic":40000,"Exploits":33393,"Fuzzers":18184,
        "DoS":12264,"Reconnaissance":10491,"Analysis":2000,
        "Backdoor":1746,"Shellcode":1133,"Worms":130
    }
    from dashboard.components.charts import attack_distribution_chart, class_bar_chart
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(attack_distribution_chart(class_counts, "Training Set Class Distribution"), use_container_width=True)
    with col2:
        st.plotly_chart(class_bar_chart(class_counts, "Class Counts"), use_container_width=True)
else:
    df = pd.DataFrame(history)
    total = len(df)
    normal_n = (df["prediction"] == "normal").sum()
    attack_n = (df["prediction"].isin(
        ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode","Worms"])).sum()
    unknown_n = (df["prediction"] == "unknown_suspicious").sum()
    uncertain_n = (df["prediction"] == "uncertain_review").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flows", f"{total:,}")
    col2.metric("Normal", f"{normal_n:,}", f"{100*normal_n/max(total,1):.1f}%")
    col3.metric("Known Attacks", f"{attack_n:,}", f"{100*attack_n/max(total,1):.1f}%")
    col4.metric("Unknown Suspicious", f"{unknown_n:,}", delta=f"{100*unknown_n/max(total,1):.1f}%", delta_color="inverse")

    pred_counts = df["prediction"].value_counts().to_dict()
    from dashboard.components.charts import attack_distribution_chart, class_bar_chart
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(attack_distribution_chart(pred_counts, "Predicted Distribution"), use_container_width=True)
    with col2:
        recent = df.tail(20)[["prediction","confidence","anomaly_score"]].copy()
        st.markdown("**Recent Predictions (last 20)**")
        st.dataframe(recent, use_container_width=True)
