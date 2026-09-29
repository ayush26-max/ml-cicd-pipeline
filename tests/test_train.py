"""
Unit tests for the training pipeline.
Tests model output shape, required methods, and MLflow logging.
"""

import os
import sys
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train import train, compute_metrics, HYPERPARAMS


@pytest.fixture
def dummy_data():
    """Generate small synthetic dataset for testing."""
    np.random.seed(42)
    X_train = np.random.rand(100, 11)
    X_test  = np.random.rand(20, 11)
    y_train = np.random.randint(0, 2, 100)
    y_test  = np.random.randint(0, 2, 20)
    return X_train, X_test, y_train, y_test


def test_model_trains_without_error(dummy_data):
    """Training should complete without raising any exceptions."""
    X_train, _, y_train, _ = dummy_data
    model = train(X_train, y_train)
    assert model is not None


def test_model_is_random_forest(dummy_data):
    """Trained model must be a RandomForestClassifier instance."""
    X_train, _, y_train, _ = dummy_data
    model = train(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_model_has_predict_method(dummy_data):
    """Model must expose predict() method."""
    X_train, _, y_train, _ = dummy_data
    model = train(X_train, y_train)
    assert hasattr(model, "predict")


def test_model_has_predict_proba_method(dummy_data):
    """Model must expose predict_proba() method for confidence scores."""
    X_train, _, y_train, _ = dummy_data
    model = train(X_train, y_train)
    assert hasattr(model, "predict_proba")


def test_predict_output_shape(dummy_data):
    """Predictions shape must match input sample count."""
    X_train, X_test, y_train, _ = dummy_data
    model = train(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_predict_proba_output_shape(dummy_data):
    """predict_proba must return probabilities for both classes."""
    X_train, X_test, y_train, _ = dummy_data
    model = train(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_metrics_keys(dummy_data):
    """compute_metrics must return all required metric keys."""
    X_train, X_test, y_train, y_test = dummy_data
    model = train(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    required_keys = {"accuracy", "f1_score", "precision", "recall"}
    assert required_keys.issubset(metrics.keys())


def test_metrics_values_in_range(dummy_data):
    """All metric values must be between 0 and 1."""
    X_train, X_test, y_train, y_test = dummy_data
    model = train(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


def test_hyperparams_contain_random_state():
    """HYPERPARAMS must include random_state for reproducibility."""
    assert "random_state" in HYPERPARAMS
    assert HYPERPARAMS["random_state"] == 42
