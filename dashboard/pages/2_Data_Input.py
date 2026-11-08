"""
dashboard/pages/2_Data_Input.py
"""
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.set_page_config(page_title="Data Input | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Data Input")
st.markdown("Upload a CSV file matching the UNSW-NB15 feature schema to run predictions.")

models = st.session_state.get("models", {})
if not models.get("ready"):
    st.warning("Models not fully loaded. Some features may be unavailable.")

tab1, tab2 = st.tabs(["Upload CSV", "Sample Dataset"])

with tab1:
    uploaded = st.file_uploader("Upload UNSW-NB15 format CSV", type=["csv"],
                                help="Must contain the 42 UNSW-NB15 feature columns.")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
            st.dataframe(df.head(5), use_container_width=True)

            # Schema validation
            from traffic.schema_mapping import validate_feature_schema, UNSWNB15_FEATURES
            val = validate_feature_schema(df)
            if val["valid"]:
                st.success("Schema valid - all required features present")
            else:
                st.error(f"Schema issues: {val['issues']}")
                if val["missing"]:
                    st.warning(f"Missing columns: {val['missing']}")

            if models.get("ready") and val["valid"]:
                if st.button("Run Predictions", type="primary"):
                    with st.spinner("Running inference..."):
                        engine = models["engine"]
                        try:
                            results, latency = engine.predict_with_latency(df)
                            result_dicts = [r.__dict__ for r in results]
                            result_df = pd.DataFrame(result_dicts)
                            st.session_state.prediction_history.extend(result_dicts)
                            st.success(f"Predicted {len(results):,} records | Avg latency: {latency:.3f} ms/record")
                            st.dataframe(result_df, use_container_width=True)
                            csv_buf = io.StringIO()
                            result_df.to_csv(csv_buf, index=False)
                            st.download_button("Download Predictions CSV", csv_buf.getvalue(),
                                               "nids_predictions.csv", "text/csv")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
        except Exception as e:
            st.error(f"Could not read file: {e}")

with tab2:
    sample_path = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "UNSW_NB15_sample.csv"
    if sample_path.exists():
        df_sample = pd.read_csv(sample_path)
        st.info(f"Sample dataset: {len(df_sample)} rows, 50 per attack category")
        st.dataframe(df_sample.head(10), use_container_width=True)
        if models.get("ready"):
            if st.button("Run Sample Predictions", type="primary"):
                with st.spinner("Running on sample..."):
                    engine = models["engine"]
                    try:
                        results, latency = engine.predict_with_latency(df_sample)
                        result_dicts = [r.__dict__ for r in results]
                        result_df = pd.DataFrame(result_dicts)
                        st.session_state.prediction_history.extend(result_dicts)
                        st.success(f"Predicted {len(results):,} records | Avg {latency:.3f} ms/record")
                        st.dataframe(result_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
    else:
        st.warning("Sample dataset not found in datasets/sample/")
