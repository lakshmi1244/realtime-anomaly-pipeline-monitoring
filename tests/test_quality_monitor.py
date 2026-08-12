"""Unit tests for the quality monitor and LSTM anomaly utilities."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quality_monitor import train, score, FEATURE_COLUMNS  # noqa: E402
from lstm_anomaly import make_windows, SequenceAnomalyLSTM, score_residuals  # noqa: E402


@pytest.fixture
def synthetic_history(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "row_count": rng.normal(50000, 2000, size=n),
        "null_rate": rng.normal(0.01, 0.003, size=n),
        "duplicate_rate": rng.normal(0.005, 0.002, size=n),
        "load_duration_seconds": rng.normal(300, 30, size=n),
        "schema_change_count": rng.poisson(0.05, size=n),
    })
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    return path


def test_train_and_score(tmp_path, synthetic_history):
    model_path = tmp_path / "model.joblib"
    train(str(synthetic_history), str(model_path), contamination=0.05)
    assert model_path.exists()

    results = score(str(model_path), str(synthetic_history))
    assert "is_anomaly" in results.columns
    assert len(results) == 200


def test_make_windows_shapes():
    values = np.arange(50).reshape(-1, 1).astype("float32")
    X, targets = make_windows(values, window_size=10)
    assert X.shape == (40, 10, 1)
    assert targets.shape == (40, 1)


def test_lstm_anomaly_scoring_runs():
    values = np.random.randn(60, 2).astype("float32")
    X, targets = make_windows(values, window_size=15)
    model = SequenceAnomalyLSTM(num_metrics=2)
    residuals, z_scores = score_residuals(model, X, targets)
    assert residuals.shape == targets.shape
    assert z_scores.shape == targets.shape
