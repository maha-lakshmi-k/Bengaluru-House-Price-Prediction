"""
train_model.py
--------------
Loads Bengaluru_House_Data.csv, cleans / engineers features,
trains a Random-Forest regression model, and saves artifacts
to the ./artifacts/ directory.

Artifacts produced
  artifacts/model.pkl      – trained sklearn model (pickle)
  artifacts/columns.json   – ordered list of feature column names
  artifacts/locations.json – sorted list of unique location strings
"""

import os
import re
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_PATH     = os.path.join(os.path.dirname(__file__), "Bengaluru_House_Data.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Clean
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns not useful for prediction
    df = df.drop(columns=["area_type", "availability", "society", "balcony"], errors="ignore")

    # Drop rows where critical fields are missing
    df = df.dropna(subset=["location", "size", "bath", "price"])

    # ── size → bhk (number of bedrooms) ──────────────────────────
    df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]) if pd.notnull(x) else np.nan)
    df = df.dropna(subset=["bhk"])
    df["bhk"] = df["bhk"].astype(int)

    # ── total_sqft: handle ranges like "2100 - 2850" ─────────────
    def convert_sqft(x):
        x = str(x).strip()
        if re.match(r"^\d+\.?\d*\s*-\s*\d+\.?\d*$", x):
            parts = x.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        try:
            return float(x)
        except ValueError:
            return np.nan

    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df = df.dropna(subset=["total_sqft"])

    # ── location: strip whitespace, group rare locations ─────────
    df["location"] = df["location"].apply(lambda x: x.strip())
    loc_counts = df["location"].value_counts()
    rare_locs   = loc_counts[loc_counts <= 10].index
    df["location"] = df["location"].apply(
        lambda x: "other" if x in rare_locs else x
    )

    # ── Remove outliers ──────────────────────────────────────────
    # price_per_sqft helps identify garbage entries
    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]

    # Per-location mean/std filter
    def remove_pps_outliers(df):
        out_df = pd.DataFrame()
        for loc, sub in df.groupby("location"):
            m, s = sub["price_per_sqft"].mean(), sub["price_per_sqft"].std()
            reduced = sub[(sub["price_per_sqft"] > (m - s)) & (sub["price_per_sqft"] <= (m + s))]
            out_df = pd.concat([out_df, reduced], ignore_index=True)
        return out_df

    df = remove_pps_outliers(df)

    # BHK anomaly: 2-BHK should not be costlier than 3-BHK in same location
    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for loc, loc_df in df.groupby("location"):
            bhk_stats = {}
            for bhk, bhk_df in loc_df.groupby("bhk"):
                bhk_stats[bhk] = {
                    "mean": bhk_df["price_per_sqft"].mean(),
                    "std":  bhk_df["price_per_sqft"].std(),
                    "count": bhk_df.shape[0],
                }
            for bhk, bhk_df in loc_df.groupby("bhk"):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats["count"] >= 5:
                    exclude_indices = np.append(
                        exclude_indices,
                        bhk_df[bhk_df["price_per_sqft"] < stats["mean"]].index.values,
                    )
        return df.drop(exclude_indices, axis=0)

    df = remove_bhk_outliers(df)

    # Drop helper column
    df = df.drop(columns=["size", "price_per_sqft"], errors="ignore")

    print(f"After cleaning: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 3. Feature engineering (one-hot locations)
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    locations = sorted(df["location"].unique().tolist())
    dummies   = pd.get_dummies(df["location"], drop_first=False)
    X = pd.concat(
        [df[["total_sqft", "bath", "bhk"]], dummies],
        axis=1,
    )
    y = df["price"]
    feature_columns = X.columns.tolist()
    return X, y, locations, feature_columns


# ─────────────────────────────────────────────
# 4. Train & evaluate
# ─────────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Lasso":            Lasso(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    best_name, best_model, best_score = None, None, -np.inf

    print("\nCross-validation scores:")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        mean_score = scores.mean()
        print(f"  {name:20s}  R² = {mean_score:.4f}  (±{scores.std():.4f})")
        if mean_score > best_score:
            best_score, best_name, best_model = mean_score, name, model

    print(f"\nBest model: {best_name} (R² = {best_score:.4f})")
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"Test R²:    {test_score:.4f}")
    return best_model


# ─────────────────────────────────────────────
# 5. Save artifacts
# ─────────────────────────────────────────────
def save_artifacts(model, feature_columns, locations):
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(ARTIFACTS_DIR, "columns.json"), "w") as f:
        json.dump({"data_columns": feature_columns}, f)

    with open(os.path.join(ARTIFACTS_DIR, "locations.json"), "w") as f:
        json.dump({"locations": locations}, f)

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df             = load_data(DATA_PATH)
    df             = clean_data(df)
    X, y, locs, cols = engineer_features(df)
    model          = train(X, y)
    save_artifacts(model, cols, locs)
    print("Done.")
