"""
FastAPI service exposing the trained ETA model.

Run with: uvicorn api.main:app --reload --port 8000
"""

import csv
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
MODEL_INFO_PATH = ROOT / "models" / "best_model_info.json"
PREDICTION_LOG_PATH = ROOT / "monitoring" / "prediction_log.csv"

_model = None
_model_info = {}


class ETARequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=50, description="Delivery distance in km")
    vehicle_type: Literal["bike", "scooter", "bicycle", "car"]
    weather: Literal["clear", "rainy", "foggy", "stormy"]
    traffic_level: Literal["low", "medium", "high", "jam"]
    prep_time_min: float = Field(..., ge=0, le=120, description="Kitchen prep time in minutes")
    courier_experience_yr: float = Field(..., ge=0, le=30)


class ETAResponse(BaseModel):
    predicted_eta_min: float
    model_used: str


def _load_model():
    global _model, _model_info
    if not MODEL_PATH.exists():
        try:
            import subprocess
            print(f"Model not found at {MODEL_PATH}. Attempting to pull from DVC remote...")
            res = subprocess.run(
                ["dvc", "pull", str(MODEL_PATH)],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            if res.returncode == 0 and MODEL_PATH.exists():
                print("Successfully pulled model from DVC remote.")
            else:
                print(f"DVC pull output: {res.stdout} {res.stderr}")
        except Exception as e:
            print(f"Could not auto-pull model via DVC: {e}")

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Run `dvc repro` or `dvc pull` first."
        )
    _model = joblib.load(MODEL_PATH)
    if MODEL_INFO_PATH.exists():
        _model_info = json.loads(MODEL_INFO_PATH.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    PREDICTION_LOG_PATH.parent.mkdir(exist_ok=True)
    if not PREDICTION_LOG_PATH.exists():
        with open(PREDICTION_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "distance_km", "vehicle_type", "weather",
                 "traffic_level", "prep_time_min", "courier_experience_yr",
                 "predicted_eta_min"]
            )
    yield


app = FastAPI(
    title="DeliverAI ETA Prediction API",
    description="Predicts food delivery ETA (minutes) from order features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if _model is None:
        try:
            _load_model()
        except Exception:
            pass
    return {"status": "ok", "model_loaded": _model is not None, "model_info": _model_info}


@app.post("/predict", response_model=ETAResponse)
def predict(req: ETARequest):
    if _model is None:
        _load_model()
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    row = pd.DataFrame([req.model_dump()])
    try:
        pred = float(_model.predict(row)[0])
    except Exception as exc:  # keep the API contract simple for a lab project
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # log for the drift monitor
    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), req.distance_km, req.vehicle_type,
             req.weather, req.traffic_level, req.prep_time_min,
             req.courier_experience_yr, round(pred, 1)]
        )

    return ETAResponse(
        predicted_eta_min=round(pred, 1),
        model_used=_model_info.get("model_name", "unknown"),
    )
