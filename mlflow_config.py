"""
Central MLflow configuration.
Set MLFLOW_TRACKING_URI environment variable to your DagsHub or local MLflow server.
Default falls back to local server at http://localhost:5000.
"""

import os
import mlflow

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "wine-quality-classifier"


def setup_mlflow() -> str:
    """Configure MLflow tracking URI and experiment. Returns experiment ID."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return experiment.experiment_id
