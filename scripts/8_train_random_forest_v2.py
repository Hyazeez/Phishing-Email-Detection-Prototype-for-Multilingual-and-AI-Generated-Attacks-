from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_V2_DIR = BASE_DIR / "data" / "features_v2"
TRAIN_PATH = FEATURE_V2_DIR / "train_features_v2.csv"
VAL_PATH = FEATURE_V2_DIR / "validation_features_v2.csv"
TEST_PATH = FEATURE_V2_DIR / "test_features_v2.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path, low_memory=False)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")

    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "Unknown"
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

    # Random Forest can be slow with very large TF-IDF vectors.
    # Therefore, max_features is reduced compared with Logistic Regression.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=15000,
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
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest benchmark model...")
    model.fit(X_train, y_train)

    print("\nValidation Results:")
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    print("Validation Accuracy:", val_accuracy)
    print("Validation Macro F1:", val_macro_f1)
    val_report = classification_report(y_val, val_pred)
    print(val_report)

    print("\nTest Results:")
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    print("Test Accuracy:", test_accuracy)
    print("Test Macro F1:", test_macro_f1)
    test_report = classification_report(y_test, test_pred)
    print(test_report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred, labels=model.classes_)
    print(cm)
    print("Class order:", list(model.classes_))

    model_path = MODEL_DIR / "random_forest_multilingual_features_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    results_path = RESULTS_DIR / "random_forest_v2_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Random Forest Benchmark - Multilingual V2 Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train shape: {train_df.shape}\n")
        f.write(f"Validation shape: {val_df.shape}\n")
        f.write(f"Test shape: {test_df.shape}\n\n")
        f.write("Training class distribution:\n")
        f.write(train_df[TARGET_COLUMN].value_counts().to_string())
        f.write("\n\nValidation Results:\n")
        f.write(f"Validation Accuracy: {val_accuracy}\n")
        f.write(f"Validation Macro F1: {val_macro_f1}\n")
        f.write(val_report)
        f.write("\n\nTest Results:\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Test Macro F1: {test_macro_f1}\n")
        f.write(test_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\nClass order:\n")
        f.write(str(list(model.classes_)))

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
