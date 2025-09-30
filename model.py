"""
model.py
Utilities to load the trained model and make predictions.
"""
import joblib
import pandas as pd

def load_model(path="model.joblib"):
    return joblib.load(path)

def prepare_features(df):
    df = df.copy()
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, 1)
    df["cvr"] = df["conversions"] / df["clicks"].replace(0, 1)
    features = ["spend", "impressions", "clicks", "avg_order_value", "ctr", "cvr"]
    return df[features]

def predict(df, model):
    X = prepare_features(df)
    preds = model.predict(X)
    df = df.copy()
    df["predicted_roi"] = preds
    df["predicted_revenue"] = df["predicted_roi"] * df["spend"]
    return df
