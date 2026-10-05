# DeliverAI — Food Delivery ETA Prediction (MLOps Lab Project)

Predicts food delivery ETA (minutes) from distance, traffic, weather, vehicle type,
prep time, and courier experience. This build covers the full local MLOps loop —
data versioning, experiment tracking, model comparison, a served API, a frontend,
containerization, and basic monitoring — with the cloud deployment step left out
by request. See "What's not here" below before you submit.

## Architecture

```
data/generate_data.py   -> synthetic dataset (data/raw.csv)
src/preprocessing.py    -> shared feature pipeline (train + inference use the same one)
src/train.py            -> trains 4 regression models, logs to MLflow, saves the best
api/main.py             -> FastAPI service serving the best model
frontend/app.py         -> Streamlit UI calling the API
monitoring/monitor_drift.py -> compares live traffic to training distribution
dvc.yaml                -> pipeline definition (generate_data -> train)
.github/workflows/ci.yml -> runs tests + retrains on every push
Dockerfile / docker-compose.yml -> local containerization of API + frontend
```

## Setup

```bash
python -m venv venv && source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run the full pipeline

```bash
# 1. Generate data + train + compare models (or use dvc repro to run both as a pipeline)
dvc repro

# 2. Start the API (separate terminal)
uvicorn api.main:app --reload --port 8000

# 3. Start the frontend (separate terminal)
streamlit run frontend/app.py
```

Open http://localhost:8501. Make a few predictions, then:

```bash
python monitoring/monitor_drift.py
```

to see the drift check run against what you just sent it.

## Run with Docker instead

```bash
docker-compose up --build
```

API on :8000, frontend on :8501.

## Experiment tracking

Runs are logged to `mlflow.db` (SQLite backend). View them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Four models are compared per run: Linear Regression, Ridge, Random Forest, Gradient
Boosting. Metrics: MAE, RMSE, R², MAPE. The lowest-RMSE model is auto-selected and
saved to `models/best_model.joblib` for the API to serve.

## Data versioning

`dvc.yaml` defines two stages (`generate_data`, `train`); `dvc repro` re-runs only
what's changed. `dvc.lock` pins the exact inputs/outputs for reproducibility — that's
the artifact to show in the viva as evidence of versioning, since the raw CSV itself
is gitignored (regenerated deterministically instead of stored).

## Rubric mapping

| Rubric component | Where it's demonstrated |
|---|---|
| CO1 — Cloud Services & MLOps Fundamentals | **Not covered — see note below.** MLOps fundamentals (pipelines, tracking, versioning) are covered; cloud services are not, by request. |
| CO2 — ML Pipeline Design & CI/CD | `dvc.yaml` pipeline; shared `preprocessing.py` prevents train/serve skew; `.github/workflows/ci.yml` runs tests and retrains on every push (CI). No CD target since there's no cloud host to deploy to. |
| CO3 — Model Tracking, Monitoring, Optimization | MLflow logs every run's params/metrics/artifacts; `monitor_drift.py` flags feature drift; `model_comparison.csv` is the optimization evidence (which model/hyperparams won and why). |
| CO4 — Documentation, Deployment Strategy, Presentation | This README; `docker-compose.yml` as the deployment strategy (containerized, host-agnostic — swap in any cloud VM or PaaS without code changes). |
| Viva | Talking points below. |

## What's not here (and what to say about it in the viva)

I deliberately left out AWS EC2 deployment. That's a real gap against CO1's "Correct
Use of Cloud Services" (2 marks) — don't present this as if cloud were done. If asked:
say the system is containerized and cloud-ready (Dockerfile + compose file work
unmodified on any EC2/Render/Railway instance), and that the deployment step itself
wasn't executed. That's an honest answer that still shows you understand the
architecture, which is what CO1 is actually checking for the "MLOps Fundamentals"
half.

## Talking points for the viva (CO1–CO4)

- Why synthetic data: real ETA datasets (Swiggy/Zomato) aren't freely redistributable
  at the volume needed; the generator encodes the same causal structure (distance,
  traffic, weather → travel time; prep time added independently) so models have real
  signal, not noise, to learn from.
- Why 4 models, not 1: Linear/Ridge give an interpretable baseline; Random Forest and
  Gradient Boosting test whether nonlinear interactions (e.g., rain × traffic) matter.
  In this run, Gradient Boosting wins — check `models/model_comparison.csv` for your
  actual numbers, they'll vary slightly by random seed if you regenerate data.
- Why a shared preprocessing module: the single most common MLOps bug is
  train/serve skew — encoding categorical features differently in training vs.
  inference. Importing the same `build_pipeline()` in both `train.py` and `main.py`
  makes that class of bug structurally impossible here.
- Why drift monitoring is this simple: a full Evidently/Grafana stack is overkill for
  a lab project's scope; the point being demonstrated is the *concept* — compare
  live feature distributions to training distributions, flag deviation — not a
  production-grade implementation.
