
from __future__ import annotations

import numpy as np
import pandas as pd


ORIGIN_DATE = pd.Timestamp("2025-01-01")

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "lane",
]

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "days_since_start",
    "day_of_year_sin",
    "day_of_year_cos",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

REQUIRED_INPUT_COLUMNS = {
    "pickup",
    "delivery",
    "equipment",
    "distance",
    "weight",
    "date",
}


def get_training_statistics(train_df: pd.DataFrame) -> dict[str, float]:
    """Fit imputation values from labeled training rows only."""
    if "weight" not in train_df.columns:
        raise ValueError("Input data is missing required column: weight")

    valid_weight = pd.to_numeric(train_df["weight"], errors="coerce").where(
        lambda values: values > 0
    )
    weight_median = float(valid_weight.median())
    if not np.isfinite(weight_median):
        raise ValueError("Training data does not contain a valid positive weight.")
    return {"weight_median": weight_median}


def prepare_common_features(
    df: pd.DataFrame,
    weight_median: float,
) -> pd.DataFrame:
    """Clean raw rows and build the inference-safe final feature set."""
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    if not np.isfinite(weight_median) or weight_median <= 0:
        raise ValueError("weight_median must be a positive finite number.")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")

    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    result.loc[result["weight"] <= 0, "weight"] = np.nan
    result["weight"] = result["weight"].fillna(weight_median)

    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_year"] = result["date"].dt.dayofyear
    result["week_of_year"] = result["date"].dt.isocalendar().week.astype(int)
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["days_since_start"] = (result["date"] - ORIGIN_DATE).dt.days
    result["day_of_year_sin"] = np.sin(
        2 * np.pi * result["day_of_year"] / 365.25
    )
    result["day_of_year_cos"] = np.cos(
        2 * np.pi * result["day_of_year"] / 365.25
    )

    result["lane"] = (
        result["pickup"].astype("string").fillna("Unknown")
        + "__"
        + result["delivery"].astype("string").fillna("Unknown")
    )
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("Unknown").astype(str)

    return result
