from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_V2_DIR = BASE_DIR / "data" / "features_v2"
HYBRID_DIR = BASE_DIR / "data" / "hybrid"
MODEL_DIR = BASE_DIR / "models"

HYBRID_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = FEATURE_V2_DIR / "train_features_v2.csv"
VAL_PATH = FEATURE_V2_DIR / "validation_features_v2.csv"
TEST_PATH = FEATURE_V2_DIR / "test_features_v2.csv"

TEXT_COLUMN = "clean_email_text"
TARGET_COLUMN = "category"

# Lighter than XLM-R and suitable for multilingual text.
MODEL_NAME = "distilbert-base-multilingual-cased"

# Increase to 256 if your laptop can handle it.
MAX_LENGTH = 128

# If you have GPU, try 32. If CPU is slow or memory fails, use 8.
BATCH_SIZE = 8

# Set this to a small number, such as 500, to quick-test the script.
# Keep it as None for full extraction.
SAMPLE_LIMIT = 10000


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing text column '{TEXT_COLUMN}' in {path}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {path}")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("Unknown").astype(str)

    if SAMPLE_LIMIT is not None:
        df = df.sample(n=min(SAMPLE_LIMIT, len(df)), random_state=42).reset_index(drop=True)

    print(f"{split_name} shape: {df.shape}")
    return df


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling gives one fixed-size embedding for each email.
    It averages token embeddings while ignoring padding tokens.
    """
    token_embeddings = last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    summed_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return summed_embeddings / summed_mask


def extract_embeddings(texts, tokenizer, model, device, split_name: str) -> np.ndarray:
    model.eval()
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    with torch.no_grad():
        for start_idx in tqdm(
            range(0, len(texts), BATCH_SIZE),
            total=total_batches,
            desc=f"Extracting {split_name} embeddings"
        ):
            batch_texts = texts[start_idx:start_idx + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            embeddings = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )
            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))
            del encoded, outputs, embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return np.vstack(all_embeddings).astype("float32")


def save_embeddings(name: str, embeddings: np.ndarray):
    output_path = HYBRID_DIR / f"{name}_transformer_embeddings.npy"
    np.save(output_path, embeddings)
    print(f"Saved {name} embeddings: {output_path}")
    print(f"{name} embedding shape: {embeddings.shape}")


def main():
    print("Loading V2 feature datasets...")
    train_df = load_split(TRAIN_PATH, "Train")
    val_df = load_split(VAL_PATH, "Validation")
    test_df = load_split(TEST_PATH, "Test")

    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN])
    y_val = label_encoder.transform(val_df[TARGET_COLUMN])
    y_test = label_encoder.transform(test_df[TARGET_COLUMN])

    np.save(HYBRID_DIR / "train_labels.npy", y_train.astype("int32"))
    np.save(HYBRID_DIR / "validation_labels.npy", y_val.astype("int32"))
    np.save(HYBRID_DIR / "test_labels.npy", y_test.astype("int32"))

    joblib.dump(label_encoder, MODEL_DIR / "hybrid_label_encoder.pkl")

    print("Class order:")
    print(list(label_encoder.classes_))

    print("\nLoading transformer model...")
    print("Model:", MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    print("\nExtracting train embeddings...")
    train_embeddings = extract_embeddings(
        train_df[TEXT_COLUMN].tolist(),
        tokenizer,
        model,
        device,
        "train"
    )
    save_embeddings("train", train_embeddings)

    print("\nExtracting validation embeddings...")
    val_embeddings = extract_embeddings(
        val_df[TEXT_COLUMN].tolist(),
        tokenizer,
        model,
        device,
        "validation"
    )
    save_embeddings("validation", val_embeddings)

    print("\nExtracting test embeddings...")
    test_embeddings = extract_embeddings(
        test_df[TEXT_COLUMN].tolist(),
        tokenizer,
        model,
        device,
        "test"
    )
    save_embeddings("test", test_embeddings)

    metadata = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "sample_limit": SAMPLE_LIMIT,
        "embedding_dimension": int(train_embeddings.shape[1]),
        "class_names": list(label_encoder.classes_),
        "train_shape": list(train_embeddings.shape),
        "validation_shape": list(val_embeddings.shape),
        "test_shape": list(test_embeddings.shape),
    }

    metadata_path = HYBRID_DIR / "transformer_embedding_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nMetadata saved to:", metadata_path)
    print("\nTransformer embedding extraction completed successfully.")

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
