from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_V2_DIR = BASE_DIR / "data" / "features_v2"
HYBRID_DIR = BASE_DIR / "data" / "hybrid"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"

HYBRID_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = FEATURE_V2_DIR / "train_features_v2.csv"
VAL_PATH = FEATURE_V2_DIR / "validation_features_v2.csv"
TEST_PATH = FEATURE_V2_DIR / "test_features_v2.csv"

METADATA_PATH = HYBRID_DIR / "transformer_embedding_metadata.json"
LABEL_ENCODER_PATH = MODEL_DIR / "hybrid_label_encoder.pkl"

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

RANDOM_STATE = 42


SAMPLING_MODE = "simple"


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_PATH}\n"
            "Run Step 1 transformer embedding extraction first."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_rows_from_metadata(metadata: dict, split_name: str) -> int:
    key = f"{split_name}_shape"
    if key not in metadata:
        raise ValueError(f"Missing '{key}' in metadata file.")
    return int(metadata[key][0])


def stratified_sample(df: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    sampled_parts = []

    for _, group in df.groupby(TARGET_COLUMN):
        n = max(1, int(expected_rows * len(group) / len(df)))
        sampled_parts.append(
            group.sample(
                n=min(n, len(group)),
                random_state=RANDOM_STATE
            )
        )

    sampled = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled) > expected_rows:
        sampled = sampled.sample(
            n=expected_rows,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    elif len(sampled) < expected_rows:
        remaining = df.drop(sampled.index, errors="ignore")
        add_n = expected_rows - len(sampled)
        if add_n > 0 and len(remaining) > 0:
            extra = remaining.sample(
                n=min(add_n, len(remaining)),
                random_state=RANDOM_STATE
            )
            sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.reset_index(drop=True)


def load_split(path: Path, split_name: str, expected_rows: int, sample_limit) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("Unknown").astype(str)

    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "Unknown"
        df[column] = df[column].fillna("Unknown").astype(str)

    # IMPORTANT:
    # This must exactly match Step 1 sampling behavior.
    # Step 1 sampled every split when SAMPLE_LIMIT was not None.
    if sample_limit is not None:
        if SAMPLING_MODE == "simple":
            df = df.sample(
                n=min(sample_limit, len(df)),
                random_state=RANDOM_STATE
            ).reset_index(drop=True)

        elif SAMPLING_MODE == "stratified":
            df = stratified_sample(
                df,
                expected_rows
            ).reset_index(drop=True)

        else:
            raise ValueError("SAMPLING_MODE must be either 'simple' or 'stratified'.")

    # Safety check
    if len(df) != expected_rows:
        raise ValueError(
            f"{split_name} row count mismatch. "
            f"Expected {expected_rows}, got {len(df)}"
        )

    print(f"{split_name} structured data shape: {df.shape}")
    print(df[TARGET_COLUMN].value_counts())

    return df

def make_preprocessor():
    try:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", categorical_encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def validate_labels(split_name: str, generated_labels: np.ndarray, saved_label_path: Path):
    if not saved_label_path.exists():
        print(f"Warning: {saved_label_path} not found. Skipping label alignment check.")
        return

    saved_labels = np.load(saved_label_path)

    if len(saved_labels) != len(generated_labels):
        raise ValueError(
            f"{split_name} label length mismatch. "
            f"Saved labels: {len(saved_labels)}, generated labels: {len(generated_labels)}"
        )

    if not np.array_equal(saved_labels, generated_labels):
        raise ValueError(
            f"{split_name} labels do not match transformer labels.\n"
            "This means the structured sample order is not aligned with Step 1 embeddings.\n"
            "Fix: use the same SAMPLING_MODE and sample size used during Step 1."
        )

    print(f"{split_name} label alignment check passed.")


def main():
    print("Loading transformer embedding metadata...")
    metadata = load_metadata()

    expected_train_rows = expected_rows_from_metadata(metadata, "train")
    expected_val_rows = expected_rows_from_metadata(metadata, "validation")
    expected_test_rows = expected_rows_from_metadata(metadata, "test")

    print("Expected rows from Step 1:")
    print("Train:", expected_train_rows)
    print("Validation:", expected_val_rows)
    print("Test:", expected_test_rows)

    print("\nLoading structured feature splits...")
    sample_limit = metadata.get("sample_limit")

    train_df = load_split(TRAIN_PATH, "train", expected_train_rows, sample_limit)
    val_df = load_split(VAL_PATH, "validation", expected_val_rows, sample_limit)
    test_df = load_split(TEST_PATH, "test", expected_test_rows, sample_limit)

    if LABEL_ENCODER_PATH.exists():
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df[TARGET_COLUMN])
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    y_train = label_encoder.transform(train_df[TARGET_COLUMN]).astype("int32")
    y_val = label_encoder.transform(val_df[TARGET_COLUMN]).astype("int32")
    y_test = label_encoder.transform(test_df[TARGET_COLUMN]).astype("int32")

    class_names = list(label_encoder.classes_)
    print("\nClass order:")
    print(class_names)

    print("\nChecking label alignment with Step 1...")
    validate_labels("train", y_train, HYBRID_DIR / "train_labels.npy")
    validate_labels("validation", y_val, HYBRID_DIR / "validation_labels.npy")
    validate_labels("test", y_test, HYBRID_DIR / "test_labels.npy")

    X_train_raw = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_val_raw = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test_raw = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    print("\nPreparing structured features...")
    preprocessor = make_preprocessor()

    X_train_structured = preprocessor.fit_transform(X_train_raw).astype("float32")
    X_val_structured = preprocessor.transform(X_val_raw).astype("float32")
    X_test_structured = preprocessor.transform(X_test_raw).astype("float32")

    print("Train structured feature shape:", X_train_structured.shape)
    print("Validation structured feature shape:", X_val_structured.shape)
    print("Test structured feature shape:", X_test_structured.shape)

    np.save(HYBRID_DIR / "train_structured_features.npy", X_train_structured)
    np.save(HYBRID_DIR / "validation_structured_features.npy", X_val_structured)
    np.save(HYBRID_DIR / "test_structured_features.npy", X_test_structured)

    preprocessor_path = MODEL_DIR / "xgboost_structured_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print("Structured preprocessor saved to:", preprocessor_path)

    print("\nTraining XGBoost structured model...")
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    xgb_model.fit(
        X_train_structured,
        y_train,
        eval_set=[(X_val_structured, y_val)],
        verbose=False
    )

    print("\nGenerating XGBoost probability outputs...")
    train_probs = xgb_model.predict_proba(X_train_structured).astype("float32")
    val_probs = xgb_model.predict_proba(X_val_structured).astype("float32")
    test_probs = xgb_model.predict_proba(X_test_structured).astype("float32")

    np.save(HYBRID_DIR / "train_xgboost_probabilities.npy", train_probs)
    np.save(HYBRID_DIR / "validation_xgboost_probabilities.npy", val_probs)
    np.save(HYBRID_DIR / "test_xgboost_probabilities.npy", test_probs)

    print("Train XGBoost probability shape:", train_probs.shape)
    print("Validation XGBoost probability shape:", val_probs.shape)
    print("Test XGBoost probability shape:", test_probs.shape)

    print("\nValidation Results:")
    val_pred = np.argmax(val_probs, axis=1)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    val_report = classification_report(y_val, val_pred, target_names=class_names)
    print("Validation Accuracy:", val_accuracy)
    print("Validation Macro F1:", val_macro_f1)
    print(val_report)

    print("\nTest Results:")
    test_pred = np.argmax(test_probs, axis=1)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    test_report = classification_report(y_test, test_pred, target_names=class_names)
    print("Test Accuracy:", test_accuracy)
    print("Test Macro F1:", test_macro_f1)
    print(test_report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    model_path = MODEL_DIR / "xgboost_structured_v2_model.pkl"
    joblib.dump(xgb_model, model_path)
    print("\nXGBoost model saved to:", model_path)

    results_path = RESULTS_DIR / "xgboost_structured_v2_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("XGBoost Structured Model - Hybrid V2 Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write("Expected rows from Step 1:\n")
        f.write(f"Train: {expected_train_rows}\n")
        f.write(f"Validation: {expected_val_rows}\n")
        f.write(f"Test: {expected_test_rows}\n\n")
        f.write("Class order:\n")
        f.write(str(class_names))
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
        f.write("\n\nStructured feature shape:\n")
        f.write(f"Train: {X_train_structured.shape}\n")
        f.write(f"Validation: {X_val_structured.shape}\n")
        f.write(f"Test: {X_test_structured.shape}\n")

    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()
