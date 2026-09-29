"""
Evaluation gate:
  - Loads the model from the latest MLflow run
  - Evaluates against the held-out test set
  - Exits with code 1 if accuracy < ACCURACY_THRESHOLD
  - This is the CI gate — GitHub Actions will stop the pipeline on failure
"""

import os
import sys
import numpy as np
import mlflow.sklearn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlflow_config import setup_mlflow, TRACKING_URI

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ACCURACY_THRESHOLD = 0.75   # pipeline fails if model accuracy is below this


def load_run_id() -> str:
    """Read the run_id saved by train.py."""
    path = os.path.join(PROCESSED_DIR, "run_id.txt")
    if not os.path.exists(path):
        print("[evaluate] ERROR: run_id.txt not found. Run train.py first.")
        sys.exit(1)
    with open(path) as f:
        return f.read().strip()


def load_model(run_id: str):
    """Load the sklearn model artifact from the MLflow run."""
    model_uri = f"runs:/{run_id}/model"
    print(f"[evaluate] Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def load_test_splits():
    """Load held-out test data."""
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_test, y_test


def evaluate(model, X_test, y_test) -> dict:
    """Run prediction and return metrics dict."""
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
    }


if __name__ == "__main__":
    print("[evaluate] Setting up MLflow...")
    setup_mlflow()

    run_id = load_run_id()
    model  = load_model(run_id)

    print("[evaluate] Loading test data...")
    X_test, y_test = load_test_splits()

    print("[evaluate] Running evaluation...")
    metrics = evaluate(model, X_test, y_test)

    print("\n[evaluate] ── Evaluation Results ────────────────────")
    print(f"  Run ID    : {run_id}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print(f"  Threshold : {ACCURACY_THRESHOLD}")
    print("─────────────────────────────────────────────────\n")

    if metrics["accuracy"] < ACCURACY_THRESHOLD:
        print(f"[evaluate] ✗ EVALUATION FAILED — accuracy {metrics['accuracy']} < threshold {ACCURACY_THRESHOLD}")
        print("[evaluate] Pipeline halted. Model will NOT be deployed.")
        sys.exit(1)   # ← GitHub Actions sees this and stops the pipeline
    else:
        print(f"[evaluate] ✓ EVALUATION PASSED — accuracy {metrics['accuracy']} >= threshold {ACCURACY_THRESHOLD}")
        print("[evaluate] Proceeding to Docker build and deployment.")
        sys.exit(0)
