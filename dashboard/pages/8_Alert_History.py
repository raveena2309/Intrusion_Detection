"""dashboard/pages/8_Alert_History.py"""
import sys, io, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
st.set_page_config(page_title="Alert History | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Alert History")

history = st.session_state.get("prediction_history", [])
if not history:
    st.info("No predictions yet. Upload a CSV on the Data Input page.")
    st.stop()

df = pd.DataFrame(history)

# Filters
col1, col2 = st.columns([1, 3])
with col1:
    all_cats = ["All"] + sorted(df["prediction"].unique().tolist())
    selected_cat = st.selectbox("Filter by category", all_cats)

if selected_cat != "All":
    df_view = df[df["prediction"] == selected_cat]
else:
    df_view = df.copy()

# Summary
col1,col2,col3,col4 = st.columns(4)
col1.metric("Total Records", len(df_view))
col2.metric("Attacks", (df_view["prediction"] != "normal").sum())
col3.metric("Unknown Suspicious", (df_view["prediction"] == "unknown_suspicious").sum())
col4.metric("Avg Confidence", f"{df_view['confidence'].mean():.4f}")

st.markdown("---")
st.markdown(f"### Showing {len(df_view):,} records")

display_cols = ["prediction","confidence","anomaly_score","is_anomaly","explanation_available"]
display_cols = [c for c in display_cols if c in df_view.columns]

def color_rows(row):
    if row.get("prediction") == "unknown_suspicious":
        return ["background-color: #3a1a1a"] * len(row)
    elif row.get("prediction") == "normal":
        return ["background-color: #1a2a1a"] * len(row)
    else:
        return ["background-color: #2a1f1a"] * len(row)

st.dataframe(df_view[display_cols].head(500), use_container_width=True)

# Export
csv_buf = io.StringIO()
df_view.to_csv(csv_buf, index=False)
st.download_button(
    label="Export to CSV",
    data=csv_buf.getvalue(),
    file_name="nids_alert_history.csv",
    mime="text/csv",
)

st.caption("Showing max 500 records. Export for full history.")
st.caption("Source/destination metadata not stored. Only predictions and scores are logged.")
