
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features import MODEL_FEATURES, get_training_statistics, prepare_common_features


TRAIN_PATH = Path("data/train-test.csv")
VALIDATION_PATH = Path("data/validation.csv")
TEMPLATE_PATH = Path("data/validation-predictions-template.csv")
MODEL_PATH = Path("models/freight_rate_final.cbm")
OUTPUT_PATH = Path("outputs/validation_predictions.csv")
EXPECTED_ROWS = 12_000


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)

    if len(validation) != EXPECTED_ROWS or len(template) != EXPECTED_ROWS:
        raise ValueError(f"Validation and template must each have {EXPECTED_ROWS:,} rows.")
    if not template["load_id"].equals(validation["load_id"]):
        raise ValueError("Validation and template load_id order does not match.")

    statistics = get_training_statistics(train)
    validation_features = prepare_common_features(
        validation,
        statistics["weight_median"],
    )

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    predictions = np.maximum(
        model.predict(validation_features[MODEL_FEATURES]),
        1.0,
    )

    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = predictions
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(submission):,} predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
