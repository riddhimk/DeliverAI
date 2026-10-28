"""
Trains and compares multiple regression models for ETA prediction.

Covers rubric CO2 (pipeline correctness) and CO3 (model tracking):
- Every run is logged to MLflow (params, metrics, model artifact).
- MAE, RMSE, R^2, and MAPE are computed for each candidate.
- The best model (lowest RMSE) is saved to models/best_model.joblib
  and its metadata to models/best_model_info.json for the API to load.
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))
from preprocessing import ALL_FEATURES, TARGET, build_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("deliverai-eta-prediction")

CANDIDATES = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
    ),
}


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    best_rmse = float("inf")
    best_run = None

    for name, model in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(model)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            metrics = evaluate(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_params(
                {f"model__{k}": v for k, v in model.get_params().items()}
            )
            mlflow.log_metrics(metrics)
            try:
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
            except TypeError:
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                )

            results.append({"model": name, **metrics})
            print(f"[{name}] MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
                  f"R2={metrics['r2']:.3f} MAPE={metrics['mape']:.1f}%")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_run = {"name": name, "pipeline": pipeline, "metrics": metrics}

    # Persist comparison table for the report/README
    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (sorted by RMSE):")
    print(results_df.to_string(index=False))

    # Save the winning pipeline for the API to serve
    joblib.dump(best_run["pipeline"], MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "best_model_info.json", "w") as f:
        json.dump(
            {"model_name": best_run["name"], "metrics": best_run["metrics"]},
            f,
            indent=2,
        )
    print(f"\nBest model: {best_run['name']} -> saved to models/best_model.joblib")


if __name__ == "__main__":
    main()

