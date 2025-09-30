ROI Estimator - recovery project
Files:
- train.py: train model and save model.joblib
- model.py: helper functions for loading and predicting
- app_streamlit.py: Streamlit UI for uploading data and getting predictions
- sample_data.csv: small example dataset
- requirements.txt: python dependencies

Quickstart:
1. Create virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
2. Train model (optional, Streamlit can trigger training):
   python train.py --input sample_data.csv --output model.joblib
3. Run UI:
   streamlit run app_streamlit.py
CSV format expected columns:
campaign_id, spend, impressions, clicks, conversions, avg_order_value
