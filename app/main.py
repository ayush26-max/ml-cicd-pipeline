"""
FastAPI inference server.
Loads the latest validated MLflow model once at startup.

Endpoints:
  GET  /          → health check + model version
  POST /predict   → takes feature array, returns class + confidence
  GET  /metrics   → returns latest model metrics from MLflow
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlflow_config import setup_mlflow, EXPERIMENT_NAME

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ── Global model state ────────────────────────────────────────────────────────
model_store = {
    "model":      None,
    "scaler":     None,   # StandardScaler fitted during preprocessing
    "run_id":     None,
    "metrics":    {},
    "params":     {},
}


def load_latest_model():
    """Load model and metadata from the latest MLflow run at startup."""
    setup_mlflow()

    # Read run_id saved by train.py
    run_id_path = os.path.join(PROCESSED_DIR, "run_id.txt")
    if not os.path.exists(run_id_path):
        raise RuntimeError("run_id.txt not found. Run the training pipeline first.")

    with open(run_id_path) as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"
    model     = mlflow.sklearn.load_model(model_uri)

    # The model was trained on SCALED features, so the same scaler must be
    # applied to every incoming request (otherwise predictions are wrong).
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise RuntimeError("scaler.pkl not found. Run preprocess.py first.")
    scaler = joblib.load(scaler_path)

    # Fetch metrics and params from MLflow
    client  = mlflow.MlflowClient()
    run     = client.get_run(run_id)
    metrics = run.data.metrics
    params  = run.data.params

    model_store["model"]   = model
    model_store["scaler"]  = scaler
    model_store["run_id"]  = run_id
    model_store["metrics"] = metrics
    model_store["params"]  = params

    print(f"[app] Model loaded — run_id: {run_id}")
    print(f"[app] Accuracy: {metrics.get('accuracy', 'N/A')}")


# ── Lifespan (replaces @app.on_event deprecated pattern) ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_latest_model()
    yield


app = FastAPI(
    title="ML CI/CD Pipeline — Wine Quality Classifier",
    description="Production ML inference API with MLflow model tracking.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response Schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
            }
        }


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    run_id: str


class MetricsResponse(BaseModel):
    run_id: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    hyperparameters: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    """Returns API status and current model version."""
    return {
        "status":        "ok",
        "model_version": model_store["run_id"],
        "accuracy":      model_store["metrics"].get("accuracy"),
    }


@app.post("/predict", response_model=PredictResponse, summary="Run inference")
def predict(request: PredictRequest):
    """
    Accepts a flat list of 11 wine feature values and returns:
    - prediction (0 = bad quality, 1 = good quality)
    - confidence score
    - model run_id for traceability
    """
    if model_store["model"] is None or model_store["scaler"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        features = np.array(request.features).reshape(1, -1)
        # Apply the same scaling used during training
        features = model_store["scaler"].transform(features)
        prediction   = int(model_store["model"].predict(features)[0])
        probabilities = model_store["model"].predict_proba(features)[0]
        confidence   = round(float(probabilities[prediction]), 4)
        label        = "good quality" if prediction == 1 else "bad quality"

        return PredictResponse(
            prediction=prediction,
            prediction_label=label,
            confidence=confidence,
            run_id=model_store["run_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, summary="Model performance metrics")
def get_metrics():
    """Returns the current model's performance metrics and hyperparameters from MLflow."""
    if not model_store["metrics"]:
        raise HTTPException(status_code=503, detail="Metrics not available.")

    m = model_store["metrics"]
    p = model_store["params"]

    return MetricsResponse(
        run_id=model_store["run_id"],
        accuracy=m.get("accuracy", 0.0),
        f1_score=m.get("f1_score", 0.0),
        precision=m.get("precision", 0.0),
        recall=m.get("recall", 0.0),
        hyperparameters={
            "n_estimators":    p.get("n_estimators"),
            "max_depth":       p.get("max_depth"),
            "random_state":    p.get("random_state"),
        },
    )
