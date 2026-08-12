# Real-Time Anomaly Detection & Intelligent Pipeline Monitoring

Monitors data pipeline health using a combination of Prophet and LSTM forecasting models to flag metric anomalies (row counts, load duration, null/duplicate rates, etc.), stores historical metrics in BigQuery, sends real-time Slack alerts, and includes a scikit-learn IsolationForest quality monitor trained on historical ingestion patterns.

## Project structure

```
src/
  prophet_forecaster.py   # Prophet-based forecasting + prediction-interval anomaly flagging
  lstm_anomaly.py           # LSTM sequence model for residual-based anomaly detection
  bigquery_ingest.py        # Load and query pipeline metrics in BigQuery
  slack_alerting.py         # Format and send Slack alerts for flagged anomalies
  quality_monitor.py        # scikit-learn IsolationForest data quality monitor
  pipeline_monitor.py       # Orchestrator that ties everything together
scripts/
  generate_sample_metrics.py # Synthetic pipeline metrics generator
tests/
  test_quality_monitor.py    # Unit tests
```

## Getting started

```bash
pip install -r requirements.txt

# 1. Generate synthetic pipeline metrics (or use your own historical data)
python scripts/generate_sample_metrics.py --output data/pipeline_metrics.csv --days 180

# 2. Train the data quality monitor
python src/quality_monitor.py train --history data/pipeline_metrics.csv --model-out artifacts/quality_monitor.joblib

# 3. Score a new batch
python src/quality_monitor.py score --model artifacts/quality_monitor.joblib --batch data/pipeline_metrics.csv
```

## Production pipeline monitor

`src/pipeline_monitor.py` is designed to run on a schedule against a live BigQuery table: it pulls recent metrics, scores them with both Prophet and the LSTM model, and sends a Slack alert through `slack_alerting.py` for any timestamp flagged as anomalous. Set the `SLACK_BOT_TOKEN` environment variable before running it against a real Slack workspace.

```bash
python src/pipeline_monitor.py \
  --project my-gcp-project --dataset pipeline_monitoring --table ingestion_metrics \
  --metric-col row_count --pipeline-name daily_orders_etl --slack-channel "#data-pipeline-alerts"
```

## Note on results

This repository is a runnable reference implementation and demo/portfolio project. Detection latency (MTTD) and accuracy depend on your actual metric distributions, thresholds, and alerting cadence — run the scripts against your own pipeline data to measure real results rather than treating any numbers elsewhere as guaranteed outcomes of this exact codebase.

## License

MIT
