"""
Training pipeline:
  - Loads preprocessed splits from data/processed/
  - Trains a RandomForestClassifier
  - Logs hyperparameters, metrics, and model artifact to MLflow
  - Saves the MLflow run_id to data/processed/run_id.txt for downstream steps
"""

import os
import sys
import json
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlflow_config import setup_mlflow

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Hyperparameters — tweak these across commits to show different runs in MLflow
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
}


def load_splits():
    """Load preprocessed train/test splits from disk."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def train(X_train, y_train) -> RandomForestClassifier:
    """Train and return a RandomForestClassifier."""
    model = RandomForestClassifier(**HYPERPARAMS)
    model.fit(X_train, y_train)
    return model


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute and return evaluation metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
    }


def save_run_id(run_id: str):
    """Persist run_id so evaluate.py and the API can load the correct model."""
    path = os.path.join(PROCESSED_DIR, "run_id.txt")
    with open(path, "w") as f:
        f.write(run_id)
    print(f"[train] Run ID saved to {path}")


if __name__ == "__main__":
    print("[train] Setting up MLflow...")
    setup_mlflow()

    print("[train] Loading preprocessed splits...")
    X_train, X_test, y_train, y_test = load_splits()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[train] MLflow run started: {run_id}")

        # Log hyperparameters
        mlflow.log_params(HYPERPARAMS)

        # Train
        print("[train] Training RandomForestClassifier...")
        model = train(X_train, y_train)

        # Evaluate
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="wine-quality-rf",
        )

        # Save run_id for downstream steps
        save_run_id(run_id)

        print("\n[train] ── Training Summary ──────────────────────")
        print(f"  Run ID    : {run_id}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")
        print("─────────────────────────────────────────────────\n")

    print("[train] Done.")
