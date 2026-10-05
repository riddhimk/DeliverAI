# DeliverAI — Food Delivery ETA Prediction (MLOps Lab Project)

DeliverAI predicts food delivery ETA (in minutes) based on order distance, traffic conditions, weather, vehicle type, kitchen preparation time, and courier experience.

This implementation covers the complete end-to-end MLOps pipeline as specified in the course architecture:
- **Data Versioning** with DVC & Remote Storage
- **DVC Preprocessing Pipeline Stage**
- **Experiment Tracking & Model Comparison** with MLflow
- **Best Model Selection & Registry Tracking**
- **FastAPI Model Serving** with runtime model loading
- **Streamlit Frontend** for interactive predictions
- **Data Drift Monitoring**
- **Docker Containerization** with restart policy
- **Automated CI/CD** with GitHub Actions

---

## 🏗️ Architecture & Pipeline Flow

The pipeline strictly reflects the MLOps lab specification:

```
[Raw Data]  -------------------> Versioned via DVC Remote (dvc push/pull)
    │
    ▼
[Preprocessing] ---------------> DVC pipeline stage (validates schema, cleans, splits train/test)
    │
    ▼
[Experiments] -----------------> Multiple models tracked in MLflow (Linear, Ridge, RF, Gradient Boosting)
    │
    ▼
[Best Model Selected] ---------> Compared via metrics (lowest RMSE selected)
    │
    ▼
[Model Registry] --------------> Pushed to DVC remote storage
    │
    ▼
[API + Docker Image] ----------> Serves predictions; fetches model at runtime if needed
    │
    ▼
[EC2 / Local Host] ------------> Hosts container + API (kept alive by restart policy)
    │
    ▼
[Client / Streamlit UI] -------> Sends requests, gets predictions, visualizes drift
```

### Component Breakdown
| Component | File / Directory | Description |
|---|---|---|
| **Data Generation** | `data/generate_data.py` | Generates realistic synthetic delivery dataset (`data/raw.csv`) |
| **Data Preprocessing** | `src/preprocess.py` | Standalone DVC stage: validates data and outputs `data/train.csv` & `data/test.csv` |
| **Shared Feature Engineering** | `src/preprocessing.py` | Unified ColumnTransformer pipeline ensuring zero train/serve skew |
| **Model Training & Tracking** | `src/train.py` | Trains 4 models, logs params/metrics to MLflow SQLite (`mlflow.db`), outputs best model |
| **Model Registry & Outputs** | `models/` | `best_model.joblib`, `best_model_info.json`, `model_comparison.csv` |
| **DVC Pipeline** | `dvc.yaml` & `dvc.lock` | Reproducible stages: `generate_data` ➔ `preprocess` ➔ `train` |
| **Inference API** | `api/main.py` | FastAPI service with Pydantic validation, `/health`, `/predict`, and query logging |
| **Frontend Application** | `frontend/app.py` | Streamlit interactive client with prediction input, model comparison, drift view |
| **Drift Monitoring** | `monitoring/monitor_drift.py` | Calculates statistical distribution drift between live traffic and baseline |
| **Containerization** | `Dockerfile`, `docker-compose.yml` | Multi-service local/cloud deployment with `restart: unless-stopped` |
| **CI/CD** | `.github/workflows/ci.yml` | Automated pytest suite, DVC pipeline validation, and model artifact upload |

---

## 🚀 Local Setup & Execution Guide

### 1. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# or: source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
pytest tests/ -v
```
Verifies data generation, preprocessing stage, model fitting, drift computation, and FastAPI `/health` and `/predict` endpoints.

### 3. Run the DVC Pipeline (Data ➔ Preprocess ➔ Train)
```bash
dvc repro
```
DVC checks hashes and runs only out-of-date stages. To force-rerun all stages:
```bash
dvc repro -f
```

### 4. Push / Pull Data & Model to DVC Remote
The project is configured with a local DVC storage remote (`storage`):
```bash
# Push tracked data and model artifacts to remote storage
dvc push

# Pull data and models from remote storage
dvc pull
```

### 5. Inspect Experiment Tracking (MLflow)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open [http://localhost:5000](http://localhost:5000) to view:
- All runs under experiment `deliverai-eta-prediction`
- Hyperparameters and comparison metrics: **MAE**, **RMSE**, **R²**, **MAPE**
- Scikit-learn model artifacts

### 6. Start the FastAPI Model Service
```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)
- Model prediction: `POST http://localhost:8000/predict`

### 7. Launch the Streamlit Frontend
In a new terminal (with venv activated):
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501). You can adjust distance, vehicle, weather, traffic, prep time, and courier experience to test live predictions.

### 8. Run Data Drift Monitoring
After making a few predictions from the frontend or API:
```bash
python monitoring/monitor_drift.py
```
This inspects `monitoring/prediction_log.csv` against `data/raw.csv` and writes `monitoring/drift_report.json`.

---

## 🐳 Running with Docker

To run the containerized API and Streamlit UI locally:
```bash
# Start both services
docker-compose up --build

# Run in background with restart policy enabled
docker-compose up -d
```
- API accessible at `http://localhost:8000`
- Streamlit UI accessible at `http://localhost:8501`

---

## ☁️ Cloud Deployment Architecture (Ready to Push)

When ready to push to AWS EC2 as outlined in the project specification:

```
[GitHub Repo]
     │ (git push)
     ▼
[GitHub Actions CI/CD] ──── Builds Docker Image ────► [Container Registry / ECR / Docker Hub]
                                                                    │
                                                                    │ docker pull
                                                                    ▼
                                                            [AWS EC2 Instance]
[DVC Remote (AWS S3)] ────── dvc pull best_model.joblib ───► [Running API Service]
                                                             (kept alive by restart policy)
                                                                    │
                                                                    ▲
                                                            [Client / Frontend]
```

### Steps to Switch DVC Remote to AWS S3
```bash
# 1. Configure S3 remote
dvc remote add -d s3remote s3://<your-bucket-name>/deliverai-dvc-storage
dvc remote modify s3remote region <your-aws-region>

# 2. Push data and models to S3
dvc push -r s3remote
```

### Steps to Deploy on AWS EC2
1. Launch an Ubuntu EC2 instance (`t3.medium` or `t2.medium`), open ports `8000` and `8501` in Security Group.
2. Install Docker and Docker Compose on the instance.
3. Clone repository and pull model:
   ```bash
   git clone <repo-url> deliverai
   cd deliverai
   # Pull model via DVC from S3 or use runtime volume mount
   docker compose up -d
   ```
4. The service is kept alive automatically via `restart: unless-stopped`.

---

## 📊 Rubric & Viva Alignment (CO1–CO4)

| Rubric Component | How DeliverAI Demonstrates It |
|---|---|
| **CO1: Cloud Services & MLOps Fundamentals** | Full local pipeline verified and containerized; DVC remote tracking implemented; Dockerfile and `docker-compose.yml` configured for EC2 host with automated restart policy; S3 remote switchable with one command. |
| **CO2: ML Pipeline Design & CI/CD** | 3-stage reproducible DVC pipeline (`generate_data` ➔ `preprocess` ➔ `train`); shared preprocessing pipeline eliminating train/serve skew; GitHub Actions CI workflow running tests and pipeline. |
| **CO3: Model Tracking, Monitoring & Optimization** | MLflow tracking with SQLite backend logging runs, hyperparameters, MAE, RMSE, R², MAPE; best model auto-selection; statistical drift detection in `monitor_drift.py`. |
| **CO4: Documentation & Deployment Strategy** | Comprehensive documentation, architecture diagrams, containerization strategy, clear runbooks for local execution and cloud migration. |
