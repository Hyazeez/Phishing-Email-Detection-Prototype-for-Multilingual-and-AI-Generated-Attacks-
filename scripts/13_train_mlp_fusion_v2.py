from pathlib import Path
import json
import random

import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


BASE_DIR = Path(__file__).resolve().parents[1]

HYBRID_DIR = BASE_DIR / "data" / "hybrid"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ENCODER_PATH = MODEL_DIR / "hybrid_label_encoder.pkl"
METADATA_PATH = HYBRID_DIR / "transformer_embedding_metadata.json"


def load_array(filename: str) -> np.ndarray:
    path = HYBRID_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Make sure Step 1 and Step 2 were completed successfully."
        )

    array = np.load(path)
    print(f"Loaded {filename}: {array.shape}")
    return array


def load_labels(filename: str) -> np.ndarray:
    labels = load_array(filename).astype("int32")
    return labels


def build_fusion_features(
    transformer_embeddings: np.ndarray,
    xgboost_probabilities: np.ndarray,
    structured_features: np.ndarray
) -> np.ndarray:
    if len(transformer_embeddings) != len(xgboost_probabilities):
        raise ValueError(
            "Row mismatch between transformer embeddings and XGBoost probabilities."
        )

    if len(transformer_embeddings) != len(structured_features):
        raise ValueError(
            "Row mismatch between transformer embeddings and structured features."
        )

    fusion_features = np.hstack([
        transformer_embeddings.astype("float32"),
        xgboost_probabilities.astype("float32"),
        structured_features.astype("float32"),
    ])

    return fusion_features.astype("float32")


def build_mlp_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="fusion_input"),

        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax", name="output")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("Loading Step 1 transformer embeddings...")
    train_embeddings = load_array("train_transformer_embeddings.npy")
    val_embeddings = load_array("validation_transformer_embeddings.npy")
    test_embeddings = load_array("test_transformer_embeddings.npy")

    print("\nLoading Step 2 XGBoost probabilities...")
    train_xgb_probs = load_array("train_xgboost_probabilities.npy")
    val_xgb_probs = load_array("validation_xgboost_probabilities.npy")
    test_xgb_probs = load_array("test_xgboost_probabilities.npy")

    print("\nLoading Step 2 structured features...")
    train_structured = load_array("train_structured_features.npy")
    val_structured = load_array("validation_structured_features.npy")
    test_structured = load_array("test_structured_features.npy")

    print("\nLoading labels...")
    y_train = load_labels("train_labels.npy")
    y_val = load_labels("validation_labels.npy")
    y_test = load_labels("test_labels.npy")

    print("\nBuilding fusion feature matrices...")
    X_train_fusion = build_fusion_features(
        train_embeddings,
        train_xgb_probs,
        train_structured
    )

    X_val_fusion = build_fusion_features(
        val_embeddings,
        val_xgb_probs,
        val_structured
    )

    X_test_fusion = build_fusion_features(
        test_embeddings,
        test_xgb_probs,
        test_structured
    )

    print("Train fusion shape:", X_train_fusion.shape)
    print("Validation fusion shape:", X_val_fusion.shape)
    print("Test fusion shape:", X_test_fusion.shape)

    print("\nScaling fusion features...")
    fusion_scaler = StandardScaler()

    X_train_fusion_scaled = fusion_scaler.fit_transform(X_train_fusion).astype("float32")
    X_val_fusion_scaled = fusion_scaler.transform(X_val_fusion).astype("float32")
    X_test_fusion_scaled = fusion_scaler.transform(X_test_fusion).astype("float32")

    scaler_path = MODEL_DIR / "mlp_fusion_scaler.pkl"
    joblib.dump(fusion_scaler, scaler_path)
    print("Fusion scaler saved to:", scaler_path)

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Missing label encoder: {LABEL_ENCODER_PATH}\n"
            "Run Step 1 transformer embedding extraction first."
        )

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    class_names = list(label_encoder.classes_)

    print("\nClass order:")
    print(class_names)

    num_classes = len(class_names)
    input_dim = X_train_fusion_scaled.shape[1]

    print("\nCalculating class weights...")
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(np.unique(y_train), class_weights_array)
    }

    print(class_weights)

    print("\nBuilding MLP fusion model...")
    model = build_mlp_model(
        input_dim=input_dim,
        num_classes=num_classes
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]

    print("\nTraining MLP fusion model...")
    history = model.fit(
        X_train_fusion_scaled,
        y_train,
        validation_data=(X_val_fusion_scaled, y_val),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print("\nValidation Results:")
    val_probs = model.predict(X_val_fusion_scaled, batch_size=128)
    val_pred = np.argmax(val_probs, axis=1)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    val_report = classification_report(
        y_val,
        val_pred,
        target_names=class_names
    )

    print("Validation Accuracy:", val_accuracy)
    print("Validation Macro F1:", val_macro_f1)
    print(val_report)

    print("\nTest Results:")
    test_probs = model.predict(X_test_fusion_scaled, batch_size=128)
    test_pred = np.argmax(test_probs, axis=1)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    test_report = classification_report(
        y_test,
        test_pred,
        target_names=class_names
    )

    print("Test Accuracy:", test_accuracy)
    print("Test Macro F1:", test_macro_f1)
    print(test_report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    print("\nSaving MLP fusion outputs...")
    np.save(HYBRID_DIR / "validation_mlp_fusion_probabilities.npy", val_probs.astype("float32"))
    np.save(HYBRID_DIR / "test_mlp_fusion_probabilities.npy", test_probs.astype("float32"))

    model_path = MODEL_DIR / "mlp_fusion_hybrid_v2_model.keras"
    model.save(model_path)
    print("MLP fusion model saved to:", model_path)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    results_path = RESULTS_DIR / "mlp_fusion_hybrid_v2_results.txt"

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("MLP Fusion Hybrid Model - Transformer + XGBoost + Structured Features\n")
        f.write("=" * 80 + "\n\n")

        f.write("Hybrid architecture:\n")
        f.write("Transformer embeddings + XGBoost probability outputs + structured features -> MLP classifier\n\n")

        f.write("Transformer metadata:\n")
        f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Fusion feature shapes:\n")
        f.write(f"Train: {X_train_fusion.shape}\n")
        f.write(f"Validation: {X_val_fusion.shape}\n")
        f.write(f"Test: {X_test_fusion.shape}\n\n")

        f.write("Class order:\n")
        f.write(str(class_names))
        f.write("\n\n")

        f.write("Class weights:\n")
        f.write(str(class_weights))
        f.write("\n\n")

        f.write("Validation Results:\n")
        f.write(f"Validation Accuracy: {val_accuracy}\n")
        f.write(f"Validation Macro F1: {val_macro_f1}\n")
        f.write(val_report)
        f.write("\n\n")

        f.write("Test Results:\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Test Macro F1: {test_macro_f1}\n")
        f.write(test_report)
        f.write("\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Training History:\n")
        f.write(json.dumps(history.history, indent=2))

    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()
