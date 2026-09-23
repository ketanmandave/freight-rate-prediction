
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features import MODEL_FEATURES, get_training_statistics, prepare_common_features


TRAIN_PATH = Path("data/train-test.csv")
DECEMBER_INPUT_PATH = Path("data/december-chart-inputs.csv")
MODEL_PATH = Path("models/freight_rate_final.cbm")
OUTPUT_PATH = Path("outputs/december_predictions.csv")
OUTPUT_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    december = pd.read_csv(DECEMBER_INPUT_PATH)
    if list(december.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "December input must contain the required seven columns in scorer order."
        )

    statistics = get_training_statistics(train)
    december_features = prepare_common_features(
        december,
        statistics["weight_median"],
    )

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    predictions = np.maximum(
        model.predict(december_features[MODEL_FEATURES]),
        1.0,
    )

    output = december.copy()
    output["predicted_rate"] = predictions
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output[OUTPUT_COLUMNS].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output)} December predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
