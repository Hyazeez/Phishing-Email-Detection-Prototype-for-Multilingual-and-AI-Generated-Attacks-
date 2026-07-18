from pathlib import Path
import math
import re
import json
import gc

import joblib
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[1]

MLP_MODEL_PATH = BASE_DIR / "models" / "mlp_fusion_hybrid_v2_model.keras"
FUSION_SCALER_PATH = BASE_DIR / "models" / "mlp_fusion_scaler.pkl"

XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_structured_v2_model.pkl"
XGB_PREPROCESSOR_PATH = BASE_DIR / "models" / "xgboost_structured_preprocessor.pkl"

LABEL_ENCODER_PATH = BASE_DIR / "models" / "hybrid_label_encoder.pkl"
METADATA_PATH = BASE_DIR / "data" / "hybrid" / "transformer_embedding_metadata.json"

RESULTS_DIR = BASE_DIR / "reports" / "results"
DATA_DIR = BASE_DIR / "data" / "robustness"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


URL_PATTERN = re.compile(r"(?i)\b(?:https?://|hxxps?://|hxxp?://|www\.)[^\s<>\"']+")
EMAIL_PATTERN = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

SINHALA_PATTERN = re.compile(r"[\u0D80-\u0DFF]")
TAMIL_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

SUSPICIOUS_KEYWORDS = {
    "account", "verify", "verification", "login", "password",
    "urgent", "immediately", "suspend", "blocked", "update",
    "confirm", "bank", "payment", "invoice", "payroll", "otp",
    "wallet", "click", "security", "restore", "expire"
}

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


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_model(value):
    text = clean_text(value).lower()
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = URL_PATTERN.sub(" <URL> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text):
    has_sinhala = bool(SINHALA_PATTERN.search(text))
    has_tamil = bool(TAMIL_PATTERN.search(text))
    has_latin = bool(LATIN_PATTERN.search(text))

    if has_sinhala and has_latin:
        return "CodeMixed_Sinhala_English"
    if has_tamil and has_latin:
        return "CodeMixed_Tamil_English"
    if has_sinhala:
        return "Sinhala"
    if has_tamil:
        return "Tamil"
    if has_latin:
        return "English"
    return "Unknown"


def entropy(text):
    if not text:
        return 0.0

    probabilities = [text.count(char) / len(text) for char in set(text)]
    return -sum(prob * math.log2(prob) for prob in probabilities)


def extract_urls(text):
    return [
        url.strip(".,;:!?)]}\"'")
        for url in URL_PATTERN.findall(text or "")
    ]


def count_suspicious_keywords(text):
    lower_text = (text or "").lower()
    return sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lower_text)


def create_features(df):
    df = df.copy()

    df["subject"] = df["subject"].fillna("").apply(clean_text)
    df["body"] = df["body"].fillna("").apply(clean_text)

    df["email_text"] = (df["subject"] + " " + df["body"]).str.strip()
    df["clean_email_text"] = df["email_text"].apply(normalize_for_model)

    extracted_urls = df["email_text"].apply(extract_urls)

    df["extracted_urls"] = extracted_urls.apply(lambda urls: " | ".join(urls))
    df["url_count_extracted"] = extracted_urls.apply(len)
    df["has_url"] = (df["url_count_extracted"] > 0).astype(int)

    df["has_defanged_url"] = df["email_text"].str.contains(
        r"hxxp|\[\.\]|\(\.\)",
        case=False,
        regex=True,
        na=False
    ).astype(int)

    df["text_char_count"] = df["email_text"].str.len()
    df["text_word_count"] = df["email_text"].str.split().apply(len)
    df["subject_char_count"] = df["subject"].str.len()
    df["body_char_count"] = df["body"].str.len()

    df["suspicious_keyword_count"] = df["email_text"].apply(count_suspicious_keywords)
    df["email_entropy"] = df["clean_email_text"].apply(entropy).round(4)
    df["detected_language"] = df["email_text"].apply(detect_language)

    return df


def build_robustness_dataset():
    samples = []

    sinhala_phishing = [
        ("Account verify කරන්න", "ඔබගේ bank account එක suspend වෙන්න කලින් verify කරන්න. hxxps://secure-bank[.]test/login"),
        ("OTP confirm කරන්න", "ඔබගේ OTP number එක immediately confirm කරන්න account security සඳහා."),
        ("Bank update required", "ඔබගේ online banking details update කරන්න. hxxps://bank-update[.]test"),
        ("Payment issue එකක්", "ඔබගේ payment එක hold වෙලා ඇත. account verify කරන්න."),
        ("Security alert", "ඔබගේ account එක blocked වෙන්න පුළුවන්. login කරලා restore කරන්න."),
        ("Wallet verification", "ඔබගේ wallet access එක expire වෙලා. hxxps://wallet-secure[.]test"),
        ("Urgent login request", "ඔබගේ bank login details confirm කරන්න immediately."),
        ("Account blocked", "ඔබගේ account එක temporarily blocked. verify password now."),
        ("Card verification", "ඔබගේ card payment verify කරන්න hxxps://card-check[.]test"),
        ("Security update", "ඔබගේ security settings update කරන්න account protect කිරීමට."),
    ]

    for subject, body in sinhala_phishing:
        samples.append({
            "attack_type": "Sinhala-English code-mixed phishing",
            "subject": subject,
            "body": body,
            "category": "Traditional_Phishing",
        })

    tamil_phishing = [
        ("Account update செய்யவும்", "உங்கள் bank account security update செய்யவும். hxxps://bank-secure[.]test"),
        ("OTP verify தேவை", "உங்கள் OTP உடனே verify செய்யவும் account restore செய்ய."),
        ("Login problem", "உங்கள் online banking login blocked. உடனே confirm செய்யவும்."),
        ("Payment hold", "உங்கள் payment hold செய்யப்பட்டுள்ளது. account verify செய்யவும்."),
        ("Security confirmation", "உங்கள் security details update செய்யவும். hxxps://secure-login[.]test"),
        ("Password expire", "உங்கள் password இன்று expire ஆகும். reset செய்யவும்."),
        ("Wallet access", "உங்கள் wallet access restore செய்ய verification தேவை."),
        ("Bank alert", "உங்கள் bank profile update செய்யப்படவில்லை. login செய்யவும்."),
        ("Card blocked", "உங்கள் card temporarily blocked. verify details now."),
        ("Account restriction", "உங்கள் account restriction remove செய்ய immediate confirmation தேவை."),
    ]

    for subject, body in tamil_phishing:
        samples.append({
            "attack_type": "Tamil-English code-mixed phishing",
            "subject": subject,
            "body": body,
            "category": "Traditional_Phishing",
        })

    ai_phishing = [
        ("Important security review required", "Dear customer, we detected unusual activity in your account. Please complete verification within 24 hours."),
        ("Account protection notice", "For your safety, please review your account security settings using the secure verification portal."),
        ("Unusual sign-in detected", "A recent login attempt requires confirmation. Please verify your identity to avoid service interruption."),
        ("Payment verification required", "Your recent transaction requires additional confirmation before it can be processed."),
        ("Mailbox storage warning", "Your mailbox storage limit is almost reached. Please confirm your account to avoid suspension."),
        ("Payroll confirmation request", "Please review and confirm your payroll information before the next processing cycle."),
        ("Cloud access review", "Your cloud account requires security validation to continue uninterrupted access."),
        ("Document access verification", "A secure document has been shared with you. Please confirm your credentials to view it."),
        ("Banking security update", "To protect your account, complete the mandatory verification process today."),
        ("Subscription billing issue", "We could not validate your billing profile. Please update your payment details to continue service."),
    ]

    for subject, body in ai_phishing:
        samples.append({
            "attack_type": "AI-generated phishing",
            "subject": subject,
            "body": body,
            "category": "AI_Generated_Phishing",
        })

    obfuscated_phishing = [
        ("Urgent password reset", "Your password will expire today. Reset now at hxxps://login-security[.]test/reset"),
        ("Verify your account", "Click hxxp://account-verify[.]test/login to restore access."),
        ("Payment confirmation", "Confirm your payment at hxxps://paypaI-security[.]test where the domain uses a look-alike character."),
        ("Bank login warning", "Login required at hxxps://secure-bank-login[.]test to avoid suspension."),
        ("Account locked", "Your account is locked. Visit hxxps://unlock-account[.]test immediately."),
        ("Security token expired", "Your security token expired. Renew at hxxps://token-renew[.]test"),
        ("Invoice portal", "Download your invoice from hxxps://invoice-view[.]test/document"),
        ("Wallet recovery", "Recover wallet access at hxxps://wa11et-secure[.]test"),
        ("Email verification", "Verify mailbox at hxxps://mail-security[.]test/verify"),
        ("Customer portal", "Access customer portal at hxxps://cust0mer-login[.]test"),
    ]

    for subject, body in obfuscated_phishing:
        samples.append({
            "attack_type": "Obfuscated URL phishing",
            "subject": subject,
            "body": body,
            "category": "Traditional_Phishing",
        })

    mixed_samples = [
        {
            "attack_type": "Business Email Compromise",
            "subject": "Urgent invoice payment request",
            "body": "Please process this invoice immediately and confirm the wire transfer before end of day.",
            "category": "Business_Email_Compromise",
        },
        {
            "attack_type": "Business Email Compromise",
            "subject": "Updated bank details",
            "body": "Kindly use the updated bank account details for today’s supplier payment.",
            "category": "Business_Email_Compromise",
        },
        {
            "attack_type": "Business Email Compromise",
            "subject": "Confidential payment instruction",
            "body": "Please handle this request confidentially and complete the payment urgently.",
            "category": "Business_Email_Compromise",
        },
        {
            "attack_type": "Noisy spelling attack",
            "subject": "Ver1fy acc0unt immediatly",
            "body": "Your acc0unt has been bl0cked. Cl1ck here to ver1fy your l0gin details.",
            "category": "Traditional_Phishing",
        },
        {
            "attack_type": "Noisy spelling attack",
            "subject": "Passw0rd exp1red",
            "body": "Your passw0rd will exp1re. Upd4te your acc0unt now.",
            "category": "Traditional_Phishing",
        },
        {
            "attack_type": "Legitimate multilingual email",
            "subject": "Meeting reminder",
            "body": "Tomorrow meeting එක 10 AM තියෙනවා. Please bring the project notes.",
            "category": "Legitimate",
        },
        {
            "attack_type": "Legitimate Tamil-English email",
            "subject": "Project update",
            "body": "நாளை project discussion உள்ளது. Please prepare the progress summary.",
            "category": "Legitimate",
        },
        {
            "attack_type": "Legitimate English email",
            "subject": "Weekly team meeting",
            "body": "Dear team, this is a reminder for our weekly meeting tomorrow at 9 AM.",
            "category": "Legitimate",
        },
        {
            "attack_type": "Spam promotional email",
            "subject": "Special discount offer",
            "body": "Congratulations! You have been selected for a limited time offer. Click to claim your reward.",
            "category": "Spam",
        },
        {
            "attack_type": "Spam promotional email",
            "subject": "You won a prize",
            "body": "You are selected as a lucky winner. Claim your free reward today.",
            "category": "Spam",
        },
    ]

    samples.extend(mixed_samples)

    df = pd.DataFrame(samples)

    print("Robustness dataset created")
    print("Total samples:", len(df))
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nAttack type distribution:")
    print(df["attack_type"].value_counts())

    return df


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


def main():
    print("Loading MLP fusion components...")

    mlp_model = load_model(MLP_MODEL_PATH)
    fusion_scaler = joblib.load(FUSION_SCALER_PATH)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_preprocessor = joblib.load(XGB_PREPROCESSOR_PATH)

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    class_names = list(label_encoder.classes_)

    print("Class order:")
    print(class_names)

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

    print("\nCreating robustness dataset...")
    robustness_df = build_robustness_dataset()
    robustness_df = create_features(robustness_df)

    structured_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    print("\nCreating structured features...")
    X_structured_raw = robustness_df[structured_columns]

    X_structured = xgb_preprocessor.transform(X_structured_raw).astype("float32")
    print("Structured feature shape:", X_structured.shape)

    print("\nGenerating XGBoost probabilities...")
    xgb_probabilities = xgb_model.predict_proba(X_structured).astype("float32")
    print("XGBoost probability shape:", xgb_probabilities.shape)

    print("\nExtracting transformer embeddings...")
    transformer_embeddings = extract_transformer_embeddings(
        robustness_df["clean_email_text"].tolist(),
        tokenizer,
        transformer_model,
        device,
        max_length=max_length,
        batch_size=8
    )

    print("Transformer embedding shape:", transformer_embeddings.shape)

    print("\nBuilding MLP fusion input...")
    X_fusion = np.hstack([
        transformer_embeddings,
        xgb_probabilities,
        X_structured
    ]).astype("float32")

    print("Fusion feature shape:", X_fusion.shape)

    X_fusion_scaled = fusion_scaler.transform(X_fusion).astype("float32")

    print("\nPredicting with MLP fusion model...")
    probabilities = mlp_model.predict(X_fusion_scaled, batch_size=16)

    predicted_ids = np.argmax(probabilities, axis=1)
    y_pred = label_encoder.inverse_transform(predicted_ids)

    confidence = probabilities.max(axis=1)

    y_true = robustness_df["category"].astype(str)

    robustness_df["prediction"] = y_pred
    robustness_df["confidence"] = confidence.round(4)
    robustness_df["correct"] = robustness_df["category"] == robustness_df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)

    print("\nMLP Fusion Robustness Test Accuracy:", accuracy)

    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0
    )
    print(report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names
    )
    print(cm)

    output_csv = DATA_DIR / "mlp_fusion_robustness_test_results.csv"
    robustness_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    results_txt = RESULTS_DIR / "mlp_fusion_robustness_testing_results.txt"

    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("MLP Fusion Robustness Testing Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model: Transformer embeddings + XGBoost probabilities + structured features -> MLP fusion\n\n")
        f.write(f"Robustness Test Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nDetailed Results:\n")
        f.write(
            robustness_df[
                ["attack_type", "category", "prediction", "confidence", "correct"]
            ].to_string(index=False)
        )

    print("\nSaved MLP fusion robustness results to:", output_csv)
    print("Saved report to:", results_txt)

    del transformer_model, tokenizer
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()