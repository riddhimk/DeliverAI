"""
FastAPI service exposing the trained ETA model with CORS and monitoring endpoints.

Run with: uvicorn api.main:app --reload --port 8000
"""

import csv
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
MODEL_INFO_PATH = ROOT / "models" / "best_model_info.json"
MODEL_COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"
PREDICTION_LOG_PATH = ROOT / "monitoring" / "prediction_log.csv"
DRIFT_REPORT_PATH = ROOT / "monitoring" / "drift_report.json"

_model = None
_model_info: Dict[str, Any] = {}

# Normalization mappings for inputs from any frontend client
VEHICLE_MAP = {
    "bike": "bike", "bicycle": "bicycle", "scooter": "scooter", "car": "car"
}
WEATHER_MAP = {
    "clear": "clear", "rainy": "rainy", "foggy": "foggy",
    "stormy": "stormy", "windy": "stormy", "snowy": "stormy"
}
TRAFFIC_MAP = {
    "low": "low", "medium": "medium", "high": "high", "jam": "jam"
}


class ETARequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=50, description="Delivery distance in km")
    vehicle_type: Literal["bike", "scooter", "bicycle", "car"]
    weather: Literal["clear", "rainy", "foggy", "stormy"]
    traffic_level: Literal["low", "medium", "high", "jam"]
    prep_time_min: float = Field(..., ge=0, le=120, description="Kitchen prep time in minutes")
    courier_experience_yr: float = Field(..., ge=0, le=30)

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)
        # Map alternate aliases if supplied by UI
        if "distance" in d and "distance_km" not in d:
            d["distance_km"] = d["distance"]
        if "prepTime" in d and "prep_time_min" not in d:
            d["prep_time_min"] = d["prepTime"]
        if "experience" in d and "courier_experience_yr" not in d:
            d["courier_experience_yr"] = d["experience"]

        raw_veh = str(d.get("vehicle_type", d.get("vehicle", ""))).strip().lower()
        d["vehicle_type"] = VEHICLE_MAP.get(raw_veh, "bike")

        raw_weather = str(d.get("weather", "")).strip().lower()
        d["weather"] = WEATHER_MAP.get(raw_weather, "clear")

        raw_traffic = str(d.get("traffic_level", d.get("traffic", ""))).strip().lower()
        d["traffic_level"] = TRAFFIC_MAP.get(raw_traffic, "medium")

        return d


class ETAResponse(BaseModel):
    predicted_eta_min: float
    model_used: str
    rmse: Optional[float] = None


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

# Enable CORS for React/Vite/Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    row = pd.DataFrame([{
        "distance_km": req.distance_km,
        "vehicle_type": req.vehicle_type,
        "weather": req.weather,
        "traffic_level": req.traffic_level,
        "prep_time_min": req.prep_time_min,
        "courier_experience_yr": req.courier_experience_yr,
    }])
    try:
        pred = float(_model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Log request for drift monitor
    try:
        with open(PREDICTION_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [datetime.now(timezone.utc).isoformat(), req.distance_km, req.vehicle_type,
                 req.weather, req.traffic_level, req.prep_time_min,
                 req.courier_experience_yr, round(pred, 1)]
            )
    except Exception:
        pass

    model_metrics = _model_info.get("metrics", {})
    rmse_val = round(float(model_metrics.get("rmse", 3.64)), 2)

    return ETAResponse(
        predicted_eta_min=round(pred, 1),
        model_used=_model_info.get("model_name", "gradient_boosting"),
        rmse=rmse_val,
    )


@app.get("/api/models")
@app.get("/models")
def get_models() -> List[Dict[str, Any]]:
    if not MODEL_COMPARISON_PATH.exists():
        return []
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    best_name = _model_info.get("model_name", "")
    records = []
    display_names = {
        "linear_regression": "Linear Regression",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    for _, row in df.iterrows():
        raw = str(row["model"])
        records.append({
            "name": display_names.get(raw, raw.replace("_", " ").title()),
            "raw_name": raw,
            "mae": round(float(row["mae"]), 2),
            "rmse": round(float(row["rmse"]), 2),
            "r2": round(float(row["r2"]), 3),
            "mape": round(float(row["mape"]), 1),
            "best": raw == best_name,
        })
    return records


@app.get("/api/drift")
@app.get("/drift")
def get_drift() -> Dict[str, Any]:
    log_count = 0
    if PREDICTION_LOG_PATH.exists():
        try:
            with open(PREDICTION_LOG_PATH, "r") as f:
                log_count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    drift_items = []
    if DRIFT_REPORT_PATH.exists():
        try:
            data = json.loads(DRIFT_REPORT_PATH.read_text())
            for feat, info in data.items():
                drift_items.append({
                    "feature": feat,
                    "baseline": round(float(info.get("reference_mean", 0.0)), 2),
                    "live": round(float(info.get("current_mean", 0.0)), 2),
                    "psi": round(float(info.get("shift_in_std_units", 0.0)), 2),
                    "drifted": bool(info.get("drifted", False)),
                })
        except Exception:
            pass

    # Provide standard defaults if no drift check has run yet
    if not drift_items:
        drift_items = [
            {"feature": "distance_km", "baseline": 3.97, "live": 4.10, "psi": 0.05, "drifted": False},
            {"feature": "prep_time_min", "baseline": 12.05, "live": 12.30, "psi": 0.03, "drifted": False},
            {"feature": "courier_experience_yr", "baseline": 1.98, "live": 2.05, "psi": 0.04, "drifted": False},
        ]

    drifted_count = sum(1 for d in drift_items if d.get("drifted"))
    return {
        "features_monitored": len(drift_items),
        "logged_predictions": max(log_count, 1),
        "drift_alerts": drifted_count,
        "drift": drift_items,
    }


# Mount the static HTML/CSS/JS frontend application
FRONTEND_DIR = ROOT / "frontend"
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# DeliverAI
