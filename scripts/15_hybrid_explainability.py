from pathlib import Path
import json
import gc
import re

import joblib
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parents[1]

ROBUSTNESS_INPUT = BASE_DIR / "data" / "robustness" / "mlp_fusion_robustness_test_results.csv"

MLP_MODEL_PATH = BASE_DIR / "models" / "mlp_fusion_hybrid_v2_model.keras"
FUSION_SCALER_PATH = BASE_DIR / "models" / "mlp_fusion_scaler.pkl"

XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_structured_v2_model.pkl"
XGB_PREPROCESSOR_PATH = BASE_DIR / "models" / "xgboost_structured_preprocessor.pkl"

LABEL_ENCODER_PATH = BASE_DIR / "models" / "hybrid_label_encoder.pkl"
METADATA_PATH = BASE_DIR / "data" / "hybrid" / "transformer_embedding_metadata.json"

RESULTS_DIR = BASE_DIR / "reports" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "hybrid_explainability_results.csv"
OUTPUT_TXT = RESULTS_DIR / "hybrid_explainability_report.txt"

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

STRUCTURED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPLAIN_FIRST_N = 50
RUN_TEXT_OCCLUSION = True
MAX_OCCLUSION_WORDS = 20


def mean_pooling(last_hidden_state, attention_mask):
    token_embeddings = last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    summed_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return summed_embeddings / summed_mask


def extract_transformer_embeddings(
    texts,
    tokenizer,
    transformer_model,
    device,
    max_length=128,
    batch_size=8
):
    transformer_model.eval()
    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = transformer_model(**encoded)

            embeddings = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )

            all_embeddings.append(
                embeddings.cpu().numpy().astype("float32")
            )

            del encoded, outputs, embeddings

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.vstack(all_embeddings).astype("float32")


def get_top_xgboost_global_features(xgb_model, xgb_preprocessor, top_n=15):
    feature_names = xgb_preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return importance_df.head(top_n)


def build_rule_based_explanation(row, final_prediction, final_confidence, xgb_prediction, xgb_confidence):
    reasons = []

    reasons.append(
        f"Final hybrid model predicted '{final_prediction}' with {final_confidence * 100:.2f}% confidence."
    )

    reasons.append(
        f"The XGBoost structured branch predicted '{xgb_prediction}' with {xgb_confidence * 100:.2f}% confidence."
    )

    if int(row.get("has_url", 0)) == 1:
        reasons.append("The email contains one or more URLs.")

    if int(row.get("has_defanged_url", 0)) == 1:
        reasons.append("The email contains an obfuscated or defanged URL pattern such as hxxp or [.]")

    suspicious_count = int(row.get("suspicious_keyword_count", 0))
    if suspicious_count > 0:
        reasons.append(f"The email contains {suspicious_count} suspicious phishing-related keyword(s).")

    detected_language = str(row.get("detected_language", "Unknown"))
    if detected_language != "English":
        reasons.append(f"The email contains multilingual or code-mixed text: {detected_language}.")

    entropy_value = float(row.get("email_entropy", 0))
    if entropy_value > 4.5:
        reasons.append("The email has relatively high entropy, which may indicate unusual or obfuscated text.")

    if final_prediction in ["Traditional_Phishing", "AI_Generated_Phishing", "Business_Email_Compromise"]:
        reasons.append("The predicted class belongs to a high-risk phishing-related category.")

    if final_prediction == "Spam":
        reasons.append("The model identified the email as spam-like promotional or unwanted content.")

    if final_prediction == "Legitimate":
        reasons.append("The model did not find enough strong phishing indicators to classify it as phishing.")

    return " | ".join(reasons)


def get_candidate_words(text, max_words=20):
    tokens = re.findall(r"[A-Za-z0-9\u0D80-\u0DFF\u0B80-\u0BFF]+", str(text))

    cleaned_tokens = []
    stop_words = {
        "the", "and", "for", "your", "you", "this", "that", "with",
        "from", "please", "dear", "now", "today", "within", "will"
    }

    for token in tokens:
        token_clean = token.strip()
        if len(token_clean) < 3:
            continue
        if token_clean.lower() in stop_words:
            continue
        if token_clean not in cleaned_tokens:
            cleaned_tokens.append(token_clean)

    return cleaned_tokens[:max_words]


def text_occlusion_explanation(
    row,
    predicted_id,
    original_confidence,
    original_xgb_prob,
    original_structured_features,
    tokenizer,
    transformer_model,
    device,
    mlp_model,
    fusion_scaler,
    max_length
):
    text = str(row.get("clean_email_text", ""))
    candidate_words = get_candidate_words(text, MAX_OCCLUSION_WORDS)

    if len(candidate_words) == 0:
        return ""

    occlusion_results = []

    for word in candidate_words:
        modified_text = text.replace(word, "", 1)

        modified_embedding = extract_transformer_embeddings(
            [modified_text],
            tokenizer,
            transformer_model,
            device,
            max_length=max_length,
            batch_size=1
        )

        fusion_input = np.hstack([
            modified_embedding.astype("float32"),
            original_xgb_prob.reshape(1, -1).astype("float32"),
            original_structured_features.reshape(1, -1).astype("float32"),
        ]).astype("float32")

        fusion_input_scaled = fusion_scaler.transform(fusion_input).astype("float32")
        modified_probabilities = mlp_model.predict(fusion_input_scaled, verbose=0)

        modified_confidence = float(modified_probabilities[0][predicted_id])
        confidence_drop = float(original_confidence - modified_confidence)

        occlusion_results.append({
            "word_removed": word,
            "confidence_after_removal": modified_confidence,
            "confidence_drop": confidence_drop
        })

    occlusion_df = pd.DataFrame(occlusion_results)
    occlusion_df = occlusion_df.sort_values("confidence_drop", ascending=False)

    top_positive = occlusion_df[occlusion_df["confidence_drop"] > 0].head(5)

    if top_positive.empty:
        return "No individual word removal strongly reduced the prediction confidence."

    explanation_parts = []

    for _, item in top_positive.iterrows():
        explanation_parts.append(
            f"{item['word_removed']} (drop: {item['confidence_drop']:.4f})"
        )

    return "Important text tokens by occlusion: " + ", ".join(explanation_parts)


def main():
    if not ROBUSTNESS_INPUT.exists():
        raise FileNotFoundError(
            f"Missing robustness input file: {ROBUSTNESS_INPUT}\n"
            "Run scripts/14_mlp_fusion_robustness_testing.py first."
        )

    print("Loading robustness results...")
    df = pd.read_csv(ROBUSTNESS_INPUT, low_memory=False)

    df = df.head(EXPLAIN_FIRST_N).copy()

    print("Rows selected for explanation:", len(df))

    print("\nLoading hybrid model components...")
    mlp_model = load_model(MLP_MODEL_PATH)
    fusion_scaler = joblib.load(FUSION_SCALER_PATH)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_preprocessor = joblib.load(XGB_PREPROCESSOR_PATH)

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    class_names = list(label_encoder.classes_)

    print("Class order:", class_names)

    print("\nLoading transformer metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    transformer_model_name = metadata["model_name"]
    max_length = metadata.get("max_length", 128)

    print("Transformer:", transformer_model_name)
    print("Max length:", max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    transformer_model = AutoModel.from_pretrained(transformer_model_name)
    transformer_model.to(device)

    print("\nPreparing structured features...")
    X_structured_raw = df[STRUCTURED_COLUMNS]
    X_structured = xgb_preprocessor.transform(X_structured_raw).astype("float32")

    print("Structured shape:", X_structured.shape)

    print("\nGenerating XGBoost probabilities...")
    xgb_probabilities = xgb_model.predict_proba(X_structured).astype("float32")
    xgb_pred_ids = np.argmax(xgb_probabilities, axis=1)
    xgb_predictions = label_encoder.inverse_transform(xgb_pred_ids)
    xgb_confidences = xgb_probabilities.max(axis=1)

    print("XGBoost probabilities shape:", xgb_probabilities.shape)

    print("\nExtracting transformer embeddings...")
    transformer_embeddings = extract_transformer_embeddings(
        df["clean_email_text"].fillna("").astype(str).tolist(),
        tokenizer,
        transformer_model,
        device,
        max_length=max_length,
        batch_size=8
    )

    print("Transformer embedding shape:", transformer_embeddings.shape)

    print("\nBuilding fusion features...")
    X_fusion = np.hstack([
        transformer_embeddings,
        xgb_probabilities,
        X_structured
    ]).astype("float32")

    X_fusion_scaled = fusion_scaler.transform(X_fusion).astype("float32")

    print("Fusion feature shape:", X_fusion_scaled.shape)

    print("\nPredicting with MLP fusion model...")
    fusion_probabilities = mlp_model.predict(X_fusion_scaled, batch_size=16)

    predicted_ids = np.argmax(fusion_probabilities, axis=1)
    predictions = label_encoder.inverse_transform(predicted_ids)
    confidences = fusion_probabilities.max(axis=1)

    print("\nExtracting global XGBoost feature importance...")
    top_global_features = get_top_xgboost_global_features(
        xgb_model,
        xgb_preprocessor,
        top_n=15
    )

    top_global_features_path = RESULTS_DIR / "hybrid_xgboost_global_feature_importance.csv"
    top_global_features.to_csv(top_global_features_path, index=False, encoding="utf-8-sig")

    print("Saved global XGBoost feature importance to:", top_global_features_path)

    explanation_rows = []

    print("\nGenerating explanations...")
    for i, (_, row) in enumerate(df.iterrows()):
        final_prediction = predictions[i]
        final_confidence = float(confidences[i])

        xgb_prediction = xgb_predictions[i]
        xgb_confidence = float(xgb_confidences[i])

        rule_explanation = build_rule_based_explanation(
            row,
            final_prediction,
            final_confidence,
            xgb_prediction,
            xgb_confidence
        )

        if RUN_TEXT_OCCLUSION:
            text_explanation = text_occlusion_explanation(
                row=row,
                predicted_id=int(predicted_ids[i]),
                original_confidence=final_confidence,
                original_xgb_prob=xgb_probabilities[i],
                original_structured_features=X_structured[i],
                tokenizer=tokenizer,
                transformer_model=transformer_model,
                device=device,
                mlp_model=mlp_model,
                fusion_scaler=fusion_scaler,
                max_length=max_length
            )
        else:
            text_explanation = "Text occlusion explanation disabled."

        explanation_rows.append({
            "attack_type": row.get("attack_type", ""),
            "actual_category": row.get("category", ""),
            "final_prediction": final_prediction,
            "final_confidence": round(final_confidence, 4),
            "xgboost_prediction": xgb_prediction,
            "xgboost_confidence": round(xgb_confidence, 4),
            "detected_language": row.get("detected_language", ""),
            "url_count": row.get("url_count_extracted", 0),
            "has_defanged_url": row.get("has_defanged_url", 0),
            "suspicious_keyword_count": row.get("suspicious_keyword_count", 0),
            "email_entropy": row.get("email_entropy", 0),
            "rule_based_explanation": rule_explanation,
            "text_occlusion_explanation": text_explanation,
            "correct": row.get("category", "") == final_prediction,
        })

    explanation_df = pd.DataFrame(explanation_rows)
    explanation_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("Hybrid Model Explainability Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model architecture:\n")
        f.write("Transformer embeddings + XGBoost probability outputs + structured features -> MLP fusion classifier\n\n")

        f.write("Explainability methods used:\n")
        f.write("1. XGBoost global structured feature importance\n")
        f.write("2. Transformer text occlusion explanation\n")
        f.write("3. Rule-based Why Flagged explanation\n")
        f.write("4. Final prediction confidence and XGBoost branch confidence\n\n")

        f.write("Top global XGBoost structured features:\n")
        f.write(top_global_features.to_string(index=False))
        f.write("\n\n")

        f.write("Detailed local explanations:\n")
        f.write(explanation_df.to_string(index=False))

    print("\nSaved hybrid explainability CSV to:", OUTPUT_CSV)
    print("Saved hybrid explainability report to:", OUTPUT_TXT)

    del transformer_model, tokenizer
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nHybrid explainability integration completed successfully.")


if __name__ == "__main__":
    main()