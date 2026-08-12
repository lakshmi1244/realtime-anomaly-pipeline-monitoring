"""Ingest pipeline metrics into BigQuery for monitoring and historical analysis.

Metrics (row counts, job duration, freshness lag, error rate, etc.) are
written to a BigQuery table so that Prophet/LSTM models can be trained
on historical ingestion patterns and dashboards can visualize trends.

Usage:
    python src/bigquery_ingest.py --project my-gcp-project \
        --dataset pipeline_monitoring --table ingestion_metrics \
        --input metrics.csv
"""
import argparse

import pandas as pd
from google.cloud import bigquery


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest pipeline metrics into BigQuery")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--input", required=True, help="CSV file with metrics to ingest")
    parser.add_argument("--write-disposition", default="WRITE_APPEND")
    return parser.parse_args()


def load_dataframe(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def ingest(client, df, table_ref, write_disposition):
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    return job


def main():
    args = parse_args()
    client = bigquery.Client(project=args.project)
    table_ref = f"{args.project}.{args.dataset}.{args.table}"

    df = load_dataframe(args.input)
    job = ingest(client, df, table_ref, args.write_disposition)

    print(f"Loaded {len(df)} rows into {table_ref} (job id: {job.job_id}).")


def query_recent_metrics(client, table_ref, metric_col, days=30):
    """Helper used by the pipeline monitor to pull recent history for
    training/scoring the Prophet and LSTM models.
    """
    query = f"""
        SELECT timestamp, {metric_col}
        FROM `{table_ref}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        ORDER BY timestamp
    """
    return client.query(query).to_dataframe()


if __name__ == "__main__":
    main()
