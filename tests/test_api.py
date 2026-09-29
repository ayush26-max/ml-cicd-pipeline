"""
Integration tests for the FastAPI inference server.
Uses a mock model so tests run without a live MLflow server.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mock MLflow and model loading so tests are self-contained ─────────────────
@pytest.fixture(autouse=True)
def mock_model_store():
    """Inject a fake trained model into the app's model_store."""
    from sklearn.ensemble import RandomForestClassifier

    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(50, 11)
    y = np.random.randint(0, 2, 50)
    mock_model.fit(X, y)

    from sklearn.preprocessing import StandardScaler
    mock_scaler = StandardScaler().fit(X)

    with patch("app.main.load_latest_model"):
        from app.main import model_store, app
        model_store["model"]  = mock_model
        model_store["scaler"] = mock_scaler
        model_store["run_id"] = "test-run-abc123"
        model_store["metrics"] = {
            "accuracy":  0.88,
            "f1_score":  0.87,
            "precision": 0.89,
            "recall":    0.86,
        }
        model_store["params"] = {
            "n_estimators": "100",
            "max_depth":    "10",
            "random_state": "42",
        }
        yield app


@pytest.fixture
def client(mock_model_store):
    from fastapi.testclient import TestClient
    return TestClient(mock_model_store)


# ── GET / ─────────────────────────────────────────────────────────────────────
def test_root_returns_ok(client):
    """Health check must return status ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_model_version(client):
    """Health check must include model_version field."""
    response = client.get("/")
    assert "model_version" in response.json()
    assert response.json()["model_version"] == "test-run-abc123"


def test_root_returns_accuracy(client):
    """Health check must include accuracy field."""
    response = client.get("/")
    assert "accuracy" in response.json()


# ── POST /predict ──────────────────────────────────────────────────────────────
SAMPLE_FEATURES = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]


def test_predict_returns_200(client):
    """Predict endpoint must return HTTP 200."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200


def test_predict_returns_prediction(client):
    """Predict response must contain prediction field (0 or 1)."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]


def test_predict_returns_confidence(client):
    """Confidence score must be between 0 and 1."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    data = response.json()
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_returns_label(client):
    """Prediction label must be a human-readable string."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    data = response.json()
    assert "prediction_label" in data
    assert data["prediction_label"] in ["good quality", "bad quality"]


def test_predict_returns_run_id(client):
    """Prediction response must include run_id for traceability."""
    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert "run_id" in response.json()


def test_predict_wrong_feature_count(client):
    """Sending wrong number of features should return HTTP 400."""
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 400


# ── GET /metrics ───────────────────────────────────────────────────────────────
def test_metrics_returns_200(client):
    """Metrics endpoint must return HTTP 200."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_contains_accuracy(client):
    """Metrics response must include accuracy."""
    response = client.get("/metrics")
    assert "accuracy" in response.json()


def test_metrics_contains_all_keys(client):
    """Metrics response must include all required performance metrics."""
    response = client.get("/metrics")
    data = response.json()
    for key in ["accuracy", "f1_score", "precision", "recall", "run_id", "hyperparameters"]:
        assert key in data, f"Missing key: {key}"


def test_metrics_values_are_floats(client):
    """Metric values must be numeric."""
    response = client.get("/metrics")
    data = response.json()
    for key in ["accuracy", "f1_score", "precision", "recall"]:
        assert isinstance(data[key], float), f"{key} should be float"


# ── Regression test: scaler must be applied before prediction ────────────────
def test_predict_applies_scaler_before_model(client):
    """
    The model is trained on scaled features, so /predict must scale the raw
    input first. This test fails if the scaler step is removed from the API.
    """
    from sklearn.preprocessing import StandardScaler
    from app.main import model_store

    # Scaler with a large mean/std so scaled != raw values
    raw_train = np.random.rand(50, 11) * 100 + 500
    scaler = StandardScaler().fit(raw_train)

    spy_model = MagicMock()
    spy_model.predict.return_value = np.array([1])
    spy_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    model_store["model"]  = spy_model
    model_store["scaler"] = scaler

    response = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200

    received = spy_model.predict.call_args[0][0]
    expected = scaler.transform(np.array(SAMPLE_FEATURES).reshape(1, -1))
    np.testing.assert_allclose(received, expected)
    assert not np.allclose(received, np.array(SAMPLE_FEATURES).reshape(1, -1))
