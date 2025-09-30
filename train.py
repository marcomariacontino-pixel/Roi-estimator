"""
train.py
Script to train a simple ROI estimator and save the model (joblib).
Usage: python train.py --input sample_data.csv --output model.joblib
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def load_data(path):
    df = pd.read_csv(path)
    # basic feature engineering
    df = df.copy()
    df["ctr"] = df["clicks"] / df["impressions"]
    df["cvr"] = df["conversions"] / (df["clicks"].replace(0, 1))
    df["revenue"] = df["conversions"] * df["avg_order_value"]
    df["roi"] = df["revenue"] / (df["spend"].replace(0, 1))
    features = ["spend", "impressions", "clicks", "avg_order_value", "ctr", "cvr"]
    X = df[features]
    y = df["roi"]
    return X, y

def build_pipeline():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    return pipe

def main(args):
    X, y = load_data(args.input)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    print("Train R2:", pipe.score(X_train, y_train))
    print("Test R2:", pipe.score(X_test, y_test))
    joblib.dump(pipe, args.output)
    print("Saved model to", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="model.joblib")
    args = parser.parse_args()
    main(args)
