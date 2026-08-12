"""ML-powered data quality monitoring using scikit-learn.

Trains an IsolationForest on historical ingestion pattern features
(row count, null rate, schema drift indicators, load duration, etc.)
to proactively flag upstream data quality issues before they cascade
downstream.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "row_count",
    "null_rate",
    "duplicate_rate",
    "load_duration_seconds",
    "schema_change_count",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train/score a data quality monitor")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--history", required=True, help="CSV of historical ingestion pattern features")
    train_p.add_argument("--model-out", default="artifacts/quality_monitor.joblib")
    train_p.add_argument("--contamination", type=float, default=0.05)

    score_p = sub.add_parser("score")
    score_p.add_argument("--model", required=True)
    score_p.add_argument("--batch", required=True, help="CSV of the current ingestion batch's features")

    return parser.parse_args()


def train(history_path, model_out, contamination):
    df = pd.read_csv(history_path)
    X = df[FEATURE_COLUMNS].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)

    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, model_out)
    print(f"Trained quality monitor on {len(df)} historical batches, saved to {model_out}.")


def score(model_path, batch_path):
    bundle = joblib.load(model_path)
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

    df = pd.read_csv(batch_path)
    X = df[features].values
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    results = df.copy()
    results["is_anomaly"] = predictions == -1
    results["anomaly_score"] = scores

    summary = {
        "num_batches": int(len(results)),
        "num_flagged": int(results["is_anomaly"].sum()),
        "flagged_rate": float(results["is_anomaly"].mean()),
    }
    print(json.dumps(summary, indent=2))
    return results


def main():
    args = parse_args()
    if args.command == "train":
        train(args.history, args.model_out, args.contamination)
    elif args.command == "score":
        score(args.model, args.batch)


if __name__ == "__main__":
    main()
