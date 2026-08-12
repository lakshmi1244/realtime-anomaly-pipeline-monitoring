"""Generate synthetic pipeline ingestion metrics with injected anomalies.

Useful for trying out prophet_forecaster.py, lstm_anomaly.py, and
quality_monitor.py without needing a real BigQuery-backed pipeline.

Usage:
    python scripts/generate_sample_metrics.py --output data/pipeline_metrics.csv --days 180
"""
import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic pipeline metrics")
    parser.add_argument("--output", type=str, default="data/pipeline_metrics.csv")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dates = pd.date_range("2024-01-01", periods=args.days, freq="D")
    n = args.days

    row_count = rng.normal(50000, 2000, size=n)
    null_rate = np.clip(rng.normal(0.01, 0.003, size=n), 0, 1)
    duplicate_rate = np.clip(rng.normal(0.005, 0.002, size=n), 0, 1)
    load_duration_seconds = rng.normal(300, 30, size=n)
    schema_change_count = rng.poisson(0.05, size=n)

    is_anomaly = rng.random(n) < args.anomaly_rate
    row_count[is_anomaly] *= rng.choice([0.3, 2.5], size=is_anomaly.sum())
    load_duration_seconds[is_anomaly] += rng.normal(600, 100, size=is_anomaly.sum())
    null_rate[is_anomaly] += rng.uniform(0.1, 0.3, size=is_anomaly.sum())

    df = pd.DataFrame({
        "timestamp": dates,
        "row_count": row_count.astype(int),
        "null_rate": null_rate,
        "duplicate_rate": duplicate_rate,
        "load_duration_seconds": load_duration_seconds,
        "schema_change_count": schema_change_count,
        "is_anomaly": is_anomaly.astype(int),
    })

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} ({is_anomaly.sum()} injected anomalies).")


if __name__ == "__main__":
    main()
