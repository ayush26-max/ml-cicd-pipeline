# ML Model Deployment with CI/CD Pipeline

An end-to-end MLOps project that automatically trains, evaluates, containerizes, and deploys a machine learning model on every Git push — with zero manual intervention.

---

## Pipeline Architecture

```
Git Push to main
      │
      ▼
┌─────────────────┐
│   GitHub Actions │
│   CI/CD Workflow │
└────────┬────────┘
         │
         ▼
  ┌─────────────┐     FAIL → pipeline stops
  │   1. TEST   │──────────────────────────────────────┐
  │  pytest 23  │                                      │
  └──────┬──────┘                                      │
         │ PASS                                         │
         ▼                                             │
  ┌─────────────┐                                      │
  │  2. TRAIN   │                                      │
  │ preprocess  │                                      │
  │ + RF model  │                                      │
  │ → MLflow    │                                      │
  └──────┬──────┘                                      │
         │                                             │
         ▼                                             │
  ┌─────────────┐     accuracy < 0.75                  │
  │ 3. EVALUATE │──────────────────────────────────────┘
  │  CI GATE    │  Model blocked from deployment
  └──────┬──────┘
         │ accuracy ≥ 0.75
         ▼
  ┌─────────────┐
  │  4. BUILD   │
  │  Docker img │
  │ → GHCR push │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  5. DEPLOY  │
  │ Render hook │
  │ / SSH pull  │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │        FastAPI Server           │
  │  GET  /          health check   │
  │  POST /predict   inference      │
  │  GET  /metrics   model stats    │
  └─────────────────────────────────┘
```

---

## About the Dataset

`data/dataset.csv` is a **synthetic dataset** (1,200 rows) that follows the schema of the
[UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
(11 physicochemical features + a `quality` score). Some feature values are slightly
negative because of how the data was generated, so it is **not real wine data**.

The focus of this project is the **MLOps pipeline** (automated testing, training,
evaluation gate, Docker build, deployment) and not the model's real-world accuracy.
To use real data, replace `data/dataset.csv` with the UCI file (same column names,
`;` separator converted to `,`) — the rest of the pipeline works unchanged.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | scikit-learn RandomForestClassifier |
| Experiment Tracking | MLflow (hosted on DagsHub) |
| API Server | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry (GHCR) |
| Deployment | Render.com (free tier) |

---

## Project Structure

```
ml-cicd-pipeline/
├── .github/workflows/ml_pipeline.yml  ← CI/CD pipeline definition
├── data/
│   ├── dataset.csv                    ← versioned training data
│   └── processed/                     ← generated splits (gitignored)
├── src/
│   ├── preprocess.py                  ← data cleaning + splitting
│   ├── train.py                       ← model training + MLflow logging
│   └── evaluate.py                    ← metric gate (blocks bad models)
├── app/
│   └── main.py                        ← FastAPI inference server
├── tests/
│   ├── test_train.py                  ← 9 unit tests
│   └── test_api.py                    ← 14 integration tests
├── mlflow_config.py                   ← central MLflow setup
├── Dockerfile                         ← multi-stage container build
├── requirements.txt
└── README.md
```

---

## Quickstart — Run Locally

### 1. Clone and install

```bash
git clone https://github.com/ayush26-max/ml-cicd-pipeline.git
cd ml-cicd-pipeline
pip install -r requirements.txt
```

### 2. Start MLflow (local)

```bash
mlflow server --host 0.0.0.0 --port 5000
# Visit http://localhost:5000 to see the dashboard
```

Or use DagsHub (free hosted MLflow — recommended):
- Create account at https://dagshub.com
- Create a new repo → go to Remote → MLflow
- Copy your tracking URI (looks like `https://dagshub.com/username/repo.mlflow`)

### 3. Set environment variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
# For DagsHub:
# export MLFLOW_TRACKING_URI=https://dagshub.com/USERNAME/REPO.mlflow
# export MLFLOW_TRACKING_USERNAME=your_dagshub_username
# export MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 4. Run the pipeline

```bash
# Step 1: Preprocess
python src/preprocess.py

# Step 2: Train (logs to MLflow)
python src/train.py

# Step 3: Evaluate (gates on accuracy >= 0.75)
python src/evaluate.py

# Step 4: Start API server
uvicorn app.main:app --reload --port 8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]}'

# Model metrics
curl http://localhost:8000/metrics
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Setting Up GitHub Actions (CI/CD)

### Required GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | Your DagsHub MLflow URI |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token |
| `RENDER_DEPLOY_HOOK_URL` | From Render dashboard → your service → Deploy Hook |

### Pipeline Triggers

The pipeline runs automatically on every push to `main`. It will:
1. Run all 23 tests — stops here if any fail
2. Preprocess data and train a new model
3. Evaluate accuracy — stops here if accuracy < 0.75 (model not deployed)
4. Build and push Docker image to GHCR
5. Trigger Render deployment

---

## Demo: Showing 3 Pipeline Runs

To showcase multiple runs in your MLflow dashboard (great for portfolio/viva):

```bash
# Commit 1: baseline (n_estimators=100)
git commit -m "feat: baseline model n_estimators=100"
git push origin main

# Edit src/train.py → change n_estimators to 150
# Commit 2: improved
git commit -m "feat: increase n_estimators to 150"
git push origin main

# Edit src/train.py → change max_depth to 20
# Commit 3: deeper trees
git commit -m "feat: increase max_depth to 20"
git push origin main
```

Each push triggers a full pipeline run. Your MLflow dashboard will show 3 runs with slightly different metrics — this is what makes the project feel production-grade.

---

## API Documentation

Once running, visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username | — |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub token | — |
| `PORT` | API server port | `8000` |
