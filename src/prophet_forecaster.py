"""Prophet-based forecaster for pipeline health metrics.

Fits a Prophet model to a historical metric series (e.g. row counts,
job duration, freshness lag) and computes a prediction interval used to
flag values that fall outside expected bounds.
"""
import pandas as pd
from prophet import Prophet


def fit_prophet_model(df, timestamp_col="ds", value_col="y", interval_width=0.95, weekly_seasonality=True):
    """Fit a Prophet model. Expects a DataFrame with timestamp_col/value_col,
    which will be renamed to Prophet's required 'ds'/'y' columns.
    """
    prophet_df = df[[timestamp_col, value_col]].rename(columns={timestamp_col: "ds", value_col: "y"})

    model = Prophet(
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        yearly_seasonality=False,
    )
    model.fit(prophet_df)
    return model


def detect_anomalies(model, df, timestamp_col="ds", value_col="y"):
    """Predict expected bounds for the given timestamps and flag any
    actual values that fall outside the prediction interval.
    """
    future = df[[timestamp_col]].rename(columns={timestamp_col: "ds"})
    forecast = model.predict(future)

    merged = df[[timestamp_col, value_col]].rename(columns={timestamp_col: "ds", value_col: "actual"})
    merged = merged.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")
    merged["is_anomaly"] = (merged["actual"] < merged["yhat_lower"]) | (merged["actual"] > merged["yhat_upper"])
    merged["deviation"] = merged["actual"] - merged["yhat"]
    return merged


def summarize_anomalies(scored_df):
    anomalies = scored_df[scored_df["is_anomaly"]]
    return {
        "total_points": int(len(scored_df)),
        "anomaly_count": int(len(anomalies)),
        "anomaly_rate": float(len(anomalies) / max(len(scored_df), 1)),
    }
