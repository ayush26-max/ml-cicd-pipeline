"""
Preprocessing pipeline:
  - Loads wine quality dataset
  - Drops missing values
  - Binarizes target (quality >= 6 → 1, else 0)
  - Scales features with StandardScaler
  - Saves train/test splits to data/processed/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and drop any rows with missing values."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[preprocess] Dropped {before - after} rows with missing values.")
    return df


def preprocess(df: pd.DataFrame):
    """
    Split features/target, binarize target, scale features.
    Returns X_train, X_test, y_train, y_test and the fitted scaler.
    """
    # Binarize: quality >= 6 is "good" (1), else "bad" (0)
    df = df.copy()
    df["target"] = (df["quality"] >= 6).astype(int)
    df = df.drop(columns=["quality"])

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def save_splits(X_train, X_test, y_train, y_test, scaler):
    """Save numpy splits and scaler to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
    print(f"[preprocess] Saved splits to {PROCESSED_DIR}")
    print(f"[preprocess] Train size: {len(X_train)} | Test size: {len(X_test)}")


if __name__ == "__main__":
    print("[preprocess] Loading data...")
    df = load_data(DATA_PATH)
    print(f"[preprocess] Dataset shape: {df.shape}")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    save_splits(X_train, X_test, y_train, y_test, scaler)
    print("[preprocess] Done.")
