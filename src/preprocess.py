"""
Data preprocessing stage for DeliverAI.

Reads raw data, performs validation and cleaning, and splits
into training and testing datasets for DVC pipeline tracking.
"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))
from preprocessing import ALL_FEATURES, TARGET  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"


def preprocess(raw_path: Path = RAW_DATA_PATH, train_path: Path = TRAIN_DATA_PATH, test_path: Path = TEST_DATA_PATH):
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run data/generate_data.py first.")

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 1. Schema Validation
    required_cols = ALL_FEATURES + [TARGET]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    # 2. Clean missing/invalid values
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing values.")

    # 3. Basic sanity filtering
    df = df[df[TARGET] > 0]
    df = df[df["distance_km"] > 0]

    # 4. Train-Test Split (80% train, 20% test, fixed random seed for reproducibility)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    DATA_DIR.mkdir(exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Preprocessing completed successfully:")
    print(f"  Total samples: {len(df)}")
    print(f"  Training set saved to: {train_path} ({len(train_df)} rows)")
    print(f"  Test set saved to:     {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    preprocess()

