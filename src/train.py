
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from src.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    get_training_statistics,
    prepare_common_features,
)


TRAIN_PATH = Path("data/train-test.csv")
MODEL_PATH = Path("models/freight_rate_final.cbm")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    statistics = get_training_statistics(train)
    train_features = prepare_common_features(
        train,
        statistics["weight_median"],
    )

    model = CatBoostRegressor(
        iterations=1050,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        loss_function="MAE",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        train_features[MODEL_FEATURES],
        train_features["posted_rate"],
        cat_features=CATEGORICAL_FEATURES,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
