"""Slack alerting for detected pipeline anomalies.

Sends a formatted message to a Slack channel via a bot token whenever
the Prophet/LSTM ensemble in pipeline_monitor.py flags a degradation,
so on-call engineers are notified quickly (reducing mean time to
detection).
"""
import os

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def get_client(token=None):
    token = token or os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise ValueError("A Slack bot token must be provided via --token or SLACK_BOT_TOKEN env var.")
    return WebClient(token=token)


def format_anomaly_message(pipeline_name, metric_name, actual, expected_range, timestamp):
    low, high = expected_range
    return (
        f":rotating_light: *Anomaly detected in `{pipeline_name}`*\n"
        f"Metric: `{metric_name}`\n"
        f"Actual: `{actual:.2f}` (expected range: `{low:.2f}` to `{high:.2f}`)\n"
        f"Time: `{timestamp}`"
    )


def send_alert(client, channel, message):
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    except SlackApiError as e:
        print(f"Failed to send Slack alert: {e.response['error']}")
        return None


def alert_on_anomalies(scored_df, pipeline_name, channel, client=None):
    """Iterate over a scored DataFrame (with is_anomaly/actual/yhat_lower/
    yhat_upper columns, as produced by prophet_forecaster.detect_anomalies)
    and send one Slack alert per flagged anomaly.
    """
    client = client or get_client()
    anomalies = scored_df[scored_df["is_anomaly"]]

    sent = 0
    for _, row in anomalies.iterrows():
        message = format_anomaly_message(
            pipeline_name=pipeline_name,
            metric_name=scored_df.attrs.get("metric_name", "unknown_metric"),
            actual=row["actual"],
            expected_range=(row["yhat_lower"], row["yhat_upper"]),
            timestamp=row["ds"],
        )
        if send_alert(client, channel, message) is not None:
            sent += 1

    return sent
