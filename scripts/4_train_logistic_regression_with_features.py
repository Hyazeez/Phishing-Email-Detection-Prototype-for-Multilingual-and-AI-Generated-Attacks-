from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "features" / "train_features.csv"
VAL_PATH = BASE_DIR / "data" / "features" / "validation_features.csv"
TEST_PATH = BASE_DIR / "data" / "features" / "test_features.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TEXT_COLUMN = "clean_email_text"
TARGET_COLUMN = "category"

NUMERIC_FEATURES = [
    "url_count_extracted",
    "has_url",
    "has_defanged_url",
    "text_char_count",
    "text_word_count",
    "subject_char_count",
    "body_char_count",
    "suspicious_keyword_count",
    "email_entropy",
]

CATEGORICAL_FEATURES = [
    "detected_language",
]


def load_feature_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown")

    return df


def main() -> None:
    train_df = load_feature_data(TRAIN_PATH)
    val_df = load_feature_data(VAL_PATH)
    test_df = load_feature_data(TEST_PATH)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTraining class distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    X_train = train_df[[TEXT_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[[TEXT_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df[[TEXT_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
                TEXT_COLUMN,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining feature-based baseline model...")
    model.fit(X_train, y_train)

    print("\nValidation Results:")
    val_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, val_pred))
    print(classification_report(y_val, val_pred))

    print("\nTest Results:")
    test_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_pred))
    print(classification_report(y_test, test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred, labels=model.classes_))
    print("Class order:", list(model.classes_))

    model_path = MODEL_DIR / "baseline_tfidf_structured_features_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
