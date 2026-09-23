
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    get_training_statistics,
    prepare_common_features,
)


TRAIN_PATH = Path("data/train-test.csv")
OUTPUT_PATH = Path("outputs/final_temporal_metrics.csv")

TEMPORAL_FOLDS = [
    ("2025-08", "2025-08-01", "2025-09-01"),
    ("2025-09", "2025-09-01", "2025-10-01"),
    ("2025-10", "2025-10-01", "2025-11-01"),
]


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    results: list[dict[str, float | int | str]] = []

    for fold_name, start, end in TEMPORAL_FOLDS:
        fold_train_raw = train[train["date"] < start].copy()
        fold_validation_raw = train[
            (train["date"] >= start) & (train["date"] < end)
        ].copy()
        if fold_train_raw.empty or fold_validation_raw.empty:
            raise ValueError(f"Temporal fold {fold_name} is empty.")

        # Fit preprocessing only on dates before the validation month.
        statistics = get_training_statistics(fold_train_raw)
        fold_train = prepare_common_features(
            fold_train_raw,
            statistics["weight_median"],
        )
        fold_validation = prepare_common_features(
            fold_validation_raw,
            statistics["weight_median"],
        )

        model = CatBoostRegressor(
            iterations=1400,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            fold_train[MODEL_FEATURES],
            fold_train["posted_rate"],
            cat_features=CATEGORICAL_FEATURES,
            eval_set=(
                fold_validation[MODEL_FEATURES],
                fold_validation["posted_rate"],
            ),
            early_stopping_rounds=120,
            verbose=False,
        )

        predictions = np.maximum(
            model.predict(fold_validation[MODEL_FEATURES]),
            1.0,
        )
        actual = fold_validation["posted_rate"]
        results.append(
            {
                "fold": fold_name,
                "weight_median": statistics["weight_median"],
                "MAE": mean_absolute_error(actual, predictions),
                "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
                "R2": r2_score(actual, predictions),
                "best_iteration": model.get_best_iteration(),
            }
        )

    results_frame = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(OUTPUT_PATH, index=False)

    print(results_frame.to_string(index=False))
    print("\nMean metrics:")
    print(results_frame[["MAE", "RMSE", "R2"]].mean().to_string())
    print(f"\nSaved metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
