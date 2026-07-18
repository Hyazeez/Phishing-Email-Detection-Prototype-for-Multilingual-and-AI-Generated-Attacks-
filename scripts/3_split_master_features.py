from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = BASE_DIR / "data" / "features"
INPUT_PATH = FEATURE_DIR / "master_features.csv"

TRAIN_PATH = FEATURE_DIR / "train_features.csv"
VALIDATION_PATH = FEATURE_DIR / "validation_features.csv"
TEST_PATH = FEATURE_DIR / "test_features.csv"

TARGET_COLUMN = "category"
RANDOM_STATE = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    df = df[df[TARGET_COLUMN].notna()].copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    validation_df.to_csv(VALIDATION_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print("Saved feature dataset splits:")
    print(f"Train: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"Validation: {VALIDATION_PATH} ({len(validation_df)} rows)")
    print(f"Test: {TEST_PATH} ({len(test_df)} rows)")

    print("\nTrain class distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    print("\nValidation class distribution:")
    print(validation_df[TARGET_COLUMN].value_counts())

    print("\nTest class distribution:")
    print(test_df[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()
