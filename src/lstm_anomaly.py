"""LSTM-based sequence anomaly detector for pipeline metrics.

Complements the Prophet forecaster by capturing more complex temporal
patterns (e.g. multi-metric correlations) that a univariate Prophet
model might miss. Both signals are combined by pipeline_monitor.py.
"""
import numpy as np
import torch
import torch.nn as nn


class SequenceAnomalyLSTM(nn.Module):
    """Predicts the next-step value(s) for one or more pipeline metrics
    from a rolling window of recent history. Large residuals indicate
    a likely pipeline degradation.
    """

    def __init__(self, num_metrics, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_metrics,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, num_metrics)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.output_layer(out[:, -1, :])


def make_windows(values, window_size=30):
    X, targets = [], []
    for start in range(len(values) - window_size):
        X.append(values[start:start + window_size])
        targets.append(values[start + window_size])
    return np.array(X, dtype="float32"), np.array(targets, dtype="float32")


def score_residuals(model, X, targets):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()
    residuals = np.abs(targets - preds)
    z_scores = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0) + 1e-8)
    return residuals, z_scores


def flag_anomalies(z_scores, threshold=3.0):
    return (np.abs(z_scores) > threshold).any(axis=1)
