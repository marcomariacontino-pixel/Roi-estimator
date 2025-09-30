"""
app_streamlit.py
Simple Streamlit UI to upload CSV, train or load model, and show ROI predictions.
Run with: streamlit run app_streamlit.py
"""
import streamlit as st
import pandas as pd
import os
import subprocess
from model import predict, load_model, prepare_features

MODEL_PATH = "model.joblib"

st.set_page_config(page_title="ROI Estimator", layout="wide")

st.title("ROI Estimator - recovery version 🛠️")

st.sidebar.header("Model")
if st.sidebar.button("Train model on sample_data.csv"):
    with st.spinner("Training..."):
        # call train.py
        subprocess.run(["python", "train.py", "--input", "sample_data.csv", "--output", MODEL_PATH])
        st.success("Training finished and model saved. Refresh to load.")

uploaded = st.file_uploader("Upload CSV with columns: campaign_id, spend, impressions, clicks, conversions, avg_order_value", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if st.checkbox("Use sample data"):
        df = pd.read_csv("sample_data.csv")
    else:
        st.warning("Upload data or use sample data to proceed.")
        st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    result = predict(df, model)
    st.subheader("Predictions")
    st.dataframe(result[["campaign_id","spend","revenue","roi","predicted_revenue","predicted_roi"]])
    st.download_button("Download predictions CSV", result.to_csv(index=False), file_name="predictions.csv")
else:
    st.info("Model not found. Train it from the sidebar using the sample dataset or run train.py manually.")
