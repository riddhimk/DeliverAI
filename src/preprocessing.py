"""
Shared preprocessing so training and inference can never drift apart.
Both api/main.py and src/train.py import build_preprocessor() from here.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["distance_km", "prep_time_min", "courier_experience_yr"]
CATEGORICAL_FEATURES = ["vehicle_type", "weather", "traffic_level"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "eta_min"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
