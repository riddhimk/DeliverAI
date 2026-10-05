"""
Generates a synthetic food-delivery dataset.

Real ETA datasets (Zomato/Swiggy) are not freely redistributable at scale,
so this script builds a synthetic dataset that follows the same causal
structure as a real one: ETA is a function of distance, traffic, weather,
vehicle type, prep time, and courier experience, plus noise. This keeps
the pipeline honest — the models actually have signal to find, not
random labels — without depending on an external download.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_ROWS = 6000

VEHICLE_TYPES = ["bike", "scooter", "bicycle", "car"]
WEATHER_TYPES = ["clear", "rainy", "foggy", "stormy"]
TRAFFIC_LEVELS = ["low", "medium", "high", "jam"]

VEHICLE_SPEED_FACTOR = {"bike": 1.0, "scooter": 0.95, "bicycle": 0.55, "car": 0.85}
WEATHER_DELAY_MIN = {"clear": 0, "rainy": 6, "foggy": 4, "stormy": 12}
TRAFFIC_DELAY_MIN = {"low": 0, "medium": 4, "high": 10, "jam": 18}


def generate(n_rows: int = N_ROWS) -> pd.DataFrame:
    distance_km = RNG.gamma(shape=2.2, scale=1.8, size=n_rows).clip(0.5, 25)
    vehicle_type = RNG.choice(VEHICLE_TYPES, size=n_rows, p=[0.45, 0.25, 0.15, 0.15])
    weather = RNG.choice(WEATHER_TYPES, size=n_rows, p=[0.55, 0.25, 0.12, 0.08])
    traffic = RNG.choice(TRAFFIC_LEVELS, size=n_rows, p=[0.3, 0.35, 0.25, 0.10])
    prep_time_min = RNG.gamma(shape=3.0, scale=4.0, size=n_rows).clip(2, 45)
    courier_experience_yr = RNG.exponential(scale=2.0, size=n_rows).clip(0, 12)

    base_speed_kmph = 22.0
    speed = base_speed_kmph * np.array([VEHICLE_SPEED_FACTOR[v] for v in vehicle_type])
    # experienced couriers navigate slightly faster, with diminishing returns
    speed *= (1 + 0.03 * np.log1p(courier_experience_yr))

    travel_time_min = (distance_km / speed) * 60
    weather_delay = np.array([WEATHER_DELAY_MIN[w] for w in weather])
    traffic_delay = np.array([TRAFFIC_DELAY_MIN[t] for t in traffic])

    noise = RNG.normal(loc=0, scale=3.5, size=n_rows)

    eta_min = (
        travel_time_min
        + prep_time_min
        + weather_delay
        + traffic_delay
        + noise
    ).clip(5, None)

    df = pd.DataFrame(
        {
            "distance_km": distance_km.round(2),
            "vehicle_type": vehicle_type,
            "weather": weather,
            "traffic_level": traffic,
            "prep_time_min": prep_time_min.round(1),
            "courier_experience_yr": courier_experience_yr.round(2),
            "eta_min": eta_min.round(1),
        }
    )
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "raw.csv"
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
