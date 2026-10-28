"""
Lightweight data-drift monitor.

This is the "Monitoring" third of CO3 (Model Tracking, Monitoring, and
Optimization). It's deliberately not a heavyweight tool like Evidently —
just enough to demonstrate the concept for a lab project: compare the
statistical profile of new incoming requests against the training
distribution and flag features that have drifted past a threshold.

Usage:
    python monitoring/monitor_drift.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "raw.csv"
LOG_PATH = ROOT / "monitoring" / "prediction_log.csv"
REPORT_PATH = ROOT / "monitoring" / "drift_report.json"

NUMERIC_FEATURES = ["distance_km", "prep_time_min", "courier_experience_yr"]
DRIFT_THRESHOLD_STD = 0.5  # flag if mean shifts more than 0.5 reference-std


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = {}
    for col in NUMERIC_FEATURES:
        ref_mean, ref_std = reference[col].mean(), reference[col].std()
        cur_mean = current[col].mean()
        shift_in_std = abs(cur_mean - ref_mean) / (ref_std + 1e-9)
        report[col] = {
            "reference_mean": round(float(ref_mean), 2),
            "current_mean": round(float(cur_mean), 2),
            "shift_in_std_units": round(float(shift_in_std), 2),
            "drifted": bool(shift_in_std > DRIFT_THRESHOLD_STD),
        }
    return report


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError("Run data/generate_data.py first.")
    if not LOG_PATH.exists():
        print(
            f"No prediction log at {LOG_PATH} yet. "
            "Make a few requests to the API first (it appends to this file)."
        )
        return

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(LOG_PATH)
    if len(current) < 5:
        print(f"Only {len(current)} logged predictions — need at least 5 for a meaningful check.")
        return

    report = compute_drift(reference, current)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    drifted = [k for k, v in report.items() if v["drifted"]]
    print(json.dumps(report, indent=2))
    if drifted:
        print(f"\nDRIFT DETECTED in: {', '.join(drifted)} -> consider retraining.")
    else:
        print("\nNo significant drift detected.")


if __name__ == "__main__":
    main()

