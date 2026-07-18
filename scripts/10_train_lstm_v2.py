from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models

MAX_TEXT_CHARS = 5000
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


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
        df[column] = df[column].fillna("Unknown").astype(str)

    return df


def make_text_input(series):
    values = series.fillna("").astype(str).tolist()
    tensor = tf.constant(values, dtype=tf.string)
    return tf.reshape(tensor, (-1, 1))


def prepare_structured_features(train_df, val_df, test_df):
    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    X_val_num = scaler.transform(val_df[NUMERIC_FEATURES])
    X_test_num = scaler.transform(test_df[NUMERIC_FEATURES])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_train_cat = encoder.fit_transform(train_df[CATEGORICAL_FEATURES])
    X_val_cat = encoder.transform(val_df[CATEGORICAL_FEATURES])
    X_test_cat = encoder.transform(test_df[CATEGORICAL_FEATURES])

    X_train_structured = np.hstack([X_train_num, X_train_cat]).astype("float32")
    X_val_structured = np.hstack([X_val_num, X_val_cat]).astype("float32")
    X_test_structured = np.hstack([X_test_num, X_test_cat]).astype("float32")

    return X_train_structured, X_val_structured, X_test_structured, scaler, encoder


def build_lstm_model(num_classes, structured_feature_count):
    text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input")

    vectorizer = layers.TextVectorization(
        standardize="lower",
        split="character",
        max_tokens=3000,
        output_mode="int",
        output_sequence_length=800,
        name="character_vectorizer",
    )

    x = vectorizer(text_input)
    x = layers.Embedding(
        input_dim=3000,
        output_dim=32,
        mask_zero=True,
        name="char_embedding"
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            32,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=False
        ),
        name="bidirectional_lstm"
    )(x)

    x = layers.Dropout(0.4)(x)

    structured_input = layers.Input(
        shape=(structured_feature_count,),
        dtype=tf.float32,
        name="structured_input"
    )

    s = layers.Dense(64, activation="relu")(structured_input)
    s = layers.Dropout(0.2)(s)

    combined = layers.Concatenate()([x, s])
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.Dropout(0.4)(combined)

    output = layers.Dense(num_classes, activation="softmax", name="output")(combined)

    model = models.Model(
        inputs=[text_input, structured_input],
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, vectorizer


def main():
    train_df = load_feature_data(TRAIN_PATH)
    val_df = load_feature_data(VAL_PATH)
    test_df = load_feature_data(TEST_PATH)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTraining class distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    X_train_text = make_text_input(train_df[TEXT_COLUMN])
    X_val_text = make_text_input(val_df[TEXT_COLUMN])
    X_test_text = make_text_input(test_df[TEXT_COLUMN])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN]).astype("int32")
    y_val = label_encoder.transform(val_df[TARGET_COLUMN]).astype("int32")
    y_test = label_encoder.transform(test_df[TARGET_COLUMN]).astype("int32")

    class_names = list(label_encoder.classes_)
    print("\nClass order:", class_names)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(np.unique(y_train), class_weights_array)
    }
    print("\nClass weights:")
    print(class_weights)

    X_train_structured, X_val_structured, X_test_structured, scaler, onehot_encoder = (
        prepare_structured_features(train_df, val_df, test_df)
    )

    model, vectorizer = build_lstm_model(
        num_classes=len(class_names),
        structured_feature_count=X_train_structured.shape[1]
    )

    print("\nAdapting character vectorizer...")
    vectorizer.adapt(X_train_text)

    print("\nModel Summary:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    print("\nTraining LSTM benchmark model...")
    history = model.fit(
        x={
            "text_input": X_train_text,
            "structured_input": X_train_structured.astype("float32"),
        },
        y=y_train,
        validation_data=(
            {
                "text_input": X_val_text,
                "structured_input": X_val_structured.astype("float32"),
            },
            y_val
        ),
        epochs=5,
        batch_size=128,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print("\nValidation Results:")
    val_probs = model.predict(
        {
            "text_input": X_val_text,
            "structured_input": X_val_structured.astype("float32"),
        },
        batch_size=128
    )
    val_pred = np.argmax(val_probs, axis=1)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    print("Validation Accuracy:", val_accuracy)
    print("Validation Macro F1:", val_macro_f1)
    val_report = classification_report(y_val, val_pred, target_names=class_names)
    print(val_report)

    print("\nTest Results:")
    test_probs = model.predict(
        {
            "text_input": X_test_text,
            "structured_input": X_test_structured.astype("float32"),
        },
        batch_size=128
    )
    test_pred = np.argmax(test_probs, axis=1)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    print("Test Accuracy:", test_accuracy)
    print("Test Macro F1:", test_macro_f1)
    test_report = classification_report(y_test, test_pred, target_names=class_names)
    print(test_report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    weights_path = MODEL_DIR / "lstm_multilingual_features_model.weights.h5"
    model.save_weights(weights_path)
    print(f"\nLSTM model weights saved to: {weights_path}")

    vocab_path = MODEL_DIR / "lstm_character_vectorizer_vocabulary.txt"
    vocab_path.write_text(
        "\n".join(vectorizer.get_vocabulary()),
        encoding="utf-8"
    )
    print(f"Character vocabulary saved to: {vocab_path}")   

    preprocessing_path = MODEL_DIR / "lstm_preprocessing_objects.pkl"
    joblib.dump(
        {
            "label_encoder": label_encoder,
            "scaler": scaler,
            "onehot_encoder": onehot_encoder,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "text_column": TEXT_COLUMN,
            "class_names": class_names,
        },
        preprocessing_path
    )
    print(f"Preprocessing objects saved to: {preprocessing_path}")

    results_path = RESULTS_DIR / "lstm_v2_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("LSTM Benchmark - Multilingual V2 Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train shape: {train_df.shape}\n")
        f.write(f"Validation shape: {val_df.shape}\n")
        f.write(f"Test shape: {test_df.shape}\n\n")
        f.write("Training class distribution:\n")
        f.write(train_df[TARGET_COLUMN].value_counts().to_string())
        f.write("\n\nClass order:\n")
        f.write(str(class_names))
        f.write("\n\nClass weights:\n")
        f.write(str(class_weights))
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
        f.write("\n\nTraining History:\n")
        f.write(json.dumps(history.history, indent=2))

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
