import sys
from pathlib import Path
import tempfile
import pandas as pd
from sklearn.linear_model import LinearRegression
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "data"))
sys.path.append(str(ROOT / "api"))
sys.path.append(str(ROOT / "monitoring"))

from preprocessing import build_pipeline, ALL_FEATURES, TARGET  # noqa: E402
from generate_data import generate  # noqa: E402
from preprocess import preprocess  # noqa: E402
from monitor_drift import compute_drift  # noqa: E402
from main import app  # noqa: E402


def test_generate_data_shape():
    df = generate(n_rows=200)
    assert len(df) == 200
    assert set(ALL_FEATURES + [TARGET]).issubset(df.columns)
    assert (df["eta_min"] > 0).all()


def test_preprocess_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_csv = tmp_path / "raw.csv"
        train_csv = tmp_path / "train.csv"
        test_csv = tmp_path / "test.csv"

        df = generate(n_rows=100)
        df.to_csv(raw_csv, index=False)

        preprocess(raw_path=raw_csv, train_path=train_csv, test_path=test_csv)

        assert train_csv.exists()
        assert test_csv.exists()

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        assert len(train_df) == 80
        assert len(test_df) == 20
        assert set(ALL_FEATURES + [TARGET]).issubset(train_df.columns)
        assert set(ALL_FEATURES + [TARGET]).issubset(test_df.columns)


def test_pipeline_fits_and_predicts():
    df = generate(n_rows=300)
    X, y = df[ALL_FEATURES], df[TARGET]
    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(5))
    assert len(preds) == 5
    assert all(p > 0 for p in preds)


def test_drift_computation():
    ref_df = generate(n_rows=100)
    cur_df = generate(n_rows=50)
    report = compute_drift(ref_df, cur_df)

    assert "distance_km" in report
    assert "prep_time_min" in report
    assert "courier_experience_yr" in report
    for metric in report.values():
        assert "reference_mean" in metric
        assert "current_mean" in metric
        assert "shift_in_std_units" in metric
        assert "drifted" in metric


def test_api_health_and_predict():
    model_path = ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        import joblib
        model_dir = ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        sample_df = generate(n_rows=100)
        pipe = build_pipeline(LinearRegression())
        pipe.fit(sample_df[ALL_FEATURES], sample_df[TARGET])
        joblib.dump(pipe, model_path)
        (model_dir / "best_model_info.json").write_text(
            '{"model_name": "linear_regression", "metrics": {"rmse": 3.8}}'
        )

    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True

        payload = {
            "distance_km": 5.2,
            "vehicle_type": "bike",
            "weather": "clear",
            "traffic_level": "medium",
            "prep_time_min": 15.0,
            "courier_experience_yr": 3.0,
        }
        pred_resp = client.post("/predict", json=payload)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert "predicted_eta_min" in pred_data
        assert pred_data["predicted_eta_min"] > 0
        assert "model_used" in pred_data
