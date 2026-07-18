from pathlib import Path
import math
import re
import pandas as pd


# Project root folder
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MULTILINGUAL_DIR = BASE_DIR / "data" / "multilingual"
FEATURE_DIR = BASE_DIR / "data" / "features"

INPUT_FILES = {
    "master_features.csv": PROCESSED_DIR / "master_dataset_cleaned.csv",
}

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


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_model(value: str) -> str:
    text = clean_text(value).lower()
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = URL_PATTERN.sub(" <URL> ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_language(text: str) -> str:
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


def entropy(text: str) -> float:
    if not text:
        return 0.0

    probabilities = [text.count(char) / len(text) for char in set(text)]
    return -sum(prob * math.log2(prob) for prob in probabilities)


def extract_urls(text: str) -> list[str]:
    return [
        url.strip(".,;:!?)]}\"'")
        for url in URL_PATTERN.findall(text or "")
    ]


def count_suspicious_keywords(text: str) -> int:
    lower_text = (text or "").lower()
    return sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lower_text)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Required text columns
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Optional metadata columns
    if "sender" in df.columns:
        df["sender"] = df["sender"].fillna("unknown_sender")

    if "receiver" in df.columns:
        df["receiver"] = df["receiver"].fillna("unknown_receiver")

    if "date" in df.columns:
        df["date"] = df["date"].fillna("unknown_date")

    if "urls" in df.columns:
        df["urls"] = pd.to_numeric(df["urls"], errors="coerce").fillna(0).astype(int)

    # Remove rows only if both subject and body are empty
    df = df[
        ~((df["subject"].str.strip() == "") & (df["body"].str.strip() == ""))
    ].copy()

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_missing_values(df)

    df["subject"] = df["subject"].apply(clean_text)
    df["body"] = df["body"].apply(clean_text)

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

    df["suspicious_keyword_count"] = df["email_text"].apply(
        count_suspicious_keywords
    )

    df["email_entropy"] = df["clean_email_text"].apply(entropy).round(4)
    df["detected_language"] = df["email_text"].apply(detect_language)

    return df


def process_file(output_name: str, input_path: Path) -> None:
    if not input_path.exists():
        print(f"Skipped missing file: {input_path}")
        return

    df = pd.read_csv(input_path, low_memory=False)
    featured = add_features(df)

    output_path = FEATURE_DIR / output_name
    featured.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved {output_path} ({len(featured)} rows)")


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    for output_name, input_path in INPUT_FILES.items():
        process_file(output_name, input_path)


if __name__ == "__main__":
    main()