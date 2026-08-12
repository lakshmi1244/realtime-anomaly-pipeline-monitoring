"""End-to-end pipeline health monitoring orchestrator.

Pulls recent metrics from BigQuery, scores them with both the Prophet
forecaster and the LSTM sequence model, merges the two anomaly signals,
and sends a Slack alert for any timestamp flagged by either model.
Designed to be run on a schedule (e.g. via Cloud Scheduler / Airflow /
a cron-triggered container) to catch pipeline degradations quickly.

Usage:
    python src/pipeline_monitor.py --project my-gcp-project \
        --dataset pipeline_monitoring --table ingestion_metrics \
        --metric-col row_count --pipeline-name daily_orders_etl \
        --slack-channel "#data-pipeline-alerts"
"""
import argparse

import numpy as np
from google.cloud import bigquery

from bigquery_ingest import query_recent_metrics
from prophet_forecaster import fit_prophet_model, detect_anomalies, summarize_anomalies
from lstm_anomaly import SequenceAnomalyLSTM, make_windows, score_residuals, flag_anomalies
from slack_alerting import alert_on_anomalies


def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline health monitor")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--metric-col", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--slack-channel", required=True)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--lstm-window", type=int, default=14)
    return parser.parse_args()


def run_prophet_check(df, metric_col):
    prophet_df = df.rename(columns={"timestamp": "ds", metric_col: "y"})
    model = fit_prophet_model(prophet_df)
    scored = detect_anomalies(model, prophet_df)
    scored.attrs["metric_name"] = metric_col
    return scored, summarize_anomalies(scored)


def run_lstm_check(df, metric_col, window_size):
    values = df[[metric_col]].values.astype("float32")
    if len(values) <= window_size:
        return None
    X, targets = make_windows(values, window_size=window_size)
    model = SequenceAnomalyLSTM(num_metrics=1)
    _, z_scores = score_residuals(model, X, targets)
    flags = flag_anomalies(z_scores)
    return flags


def main():
    args = parse_args()
    client = bigquery.Client(project=args.project)
    table_ref = f"{args.project}.{args.dataset}.{args.table}"

    df = query_recent_metrics(client, table_ref, args.metric_col, days=args.history_days)
    if df.empty:
        print("No recent metrics found; nothing to check.")
        return

    prophet_scored, prophet_summary = run_prophet_check(df, args.metric_col)
    lstm_flags = run_lstm_check(df, args.metric_col, args.lstm_window)

    print("Prophet anomaly summary:", prophet_summary)
    if lstm_flags is not None:
        print(f"LSTM flagged {int(np.sum(lstm_flags))} of {len(lstm_flags)} recent windows.")

    sent = alert_on_anomalies(prophet_scored, args.pipeline_name, args.slack_channel)
    print(f"Sent {sent} Slack alert(s).")


if __name__ == "__main__":
    main()
