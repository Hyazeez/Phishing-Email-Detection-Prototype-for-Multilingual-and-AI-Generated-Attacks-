import math
import re
from urllib.parse import urlparse

import pandas as pd


# =========================================================
# Feature column names
# =========================================================
TEXT_COLUMN = "clean_email_text"

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


# =========================================================
# Risk mapping
# =========================================================
RISK_MAP = {
    "Legitimate": "Safe",
    "Spam": "Medium",
    "Traditional_Phishing": "High",
    "Business_Email_Compromise": "Critical",
    "AI_Generated_Phishing": "Critical",
}


# =========================================================
# Regular-expression patterns
# =========================================================
URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|hxxps?://|www\.)[^\s<>\"']+"
)

EMAIL_PATTERN = re.compile(
    r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
)

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

SINHALA_PATTERN = re.compile(r"[\u0D80-\u0DFF]")
TAMIL_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


# =========================================================
# Suspicious phishing-related words
# =========================================================
SUSPICIOUS_KEYWORDS = {
    "account",
    "verify",
    "verification",
    "login",
    "password",
    "urgent",
    "immediately",
    "suspend",
    "suspended",
    "blocked",
    "update",
    "confirm",
    "bank",
    "payment",
    "invoice",
    "payroll",
    "otp",
    "wallet",
    "click",
    "security",
    "restore",
    "expire",
    "expired",
    "credentials",
    "transfer",
    "confidential",
    "winner",
    "reward",
    "prize",
}


# =========================================================
# Basic text-cleaning functions
# =========================================================
def clean_text(value: object) -> str:
    """
    Removes HTML tags, line breaks, and repeated spaces.
    """

    if value is None or pd.isna(value):
        return ""

    text = str(value)

    text = HTML_TAG_PATTERN.sub(" ", text)

    text = (
        text
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_model(value: str) -> str:
    """
    Converts text to lowercase and replaces email addresses and URLs
    with common placeholders.
    """

    text = clean_text(value).lower()

    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = URL_PATTERN.sub(" <URL> ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# Language detection
# =========================================================
def detect_language(text: str) -> str:
    """
    Detects English, Sinhala, Tamil, or code-mixed text using
    Unicode character ranges.
    """
    text = str(text or "")
    has_sinhala = bool(SINHALA_PATTERN.search(text))
    has_tamil = bool(TAMIL_PATTERN.search(text))
    has_latin = bool(LATIN_PATTERN.search(text))
    if has_sinhala and has_latin:
        return "CodeMixed_Sinhala_English"
    if has_tamil and has_latin:
        return "CodeMixed_Tamil_English"
    if has_sinhala and has_tamil and has_latin:
        return "CodeMixed_Multilingual"
    if has_sinhala:
        return "Sinhala"
    if has_tamil:
        return "Tamil"
    if has_latin:
        return "English"
    return "Unknown"


# =========================================================
# Text entropy
# =========================================================
def calculate_entropy(text: str) -> float:
    """
    Calculates Shannon entropy for the email text.
    Higher entropy may indicate unusual or obfuscated content.
    """

    text = str(text or "")

    if not text:
        return 0.0

    text_length = len(text)

    probabilities = [
        text.count(character) / text_length
        for character in set(text)
    ]

    entropy_value = -sum(
        probability * math.log2(probability)
        for probability in probabilities
    )

    return round(entropy_value, 4)


# Keep the shorter function name for compatibility
def entropy(text: str) -> float:
    return calculate_entropy(text)


# =========================================================
# URL extraction and analysis
# =========================================================
def extract_urls(text: str) -> list[str]:
    """
    Extracts normal and defanged URLs from email text.
    """

    urls = URL_PATTERN.findall(str(text or ""))

    return [
        url.strip(".,;:!?)]}\"'")
        for url in urls
    ]


def normalize_defanged_url(url: str) -> str:
    """
    Converts a defanged URL into a temporary normal form for
    domain parsing only.

    Example:
    hxxps://example[.]com -> https://example.com
    """

    normalized = str(url or "").strip()

    normalized = re.sub(
        r"(?i)^hxxps://",
        "https://",
        normalized
    )

    normalized = re.sub(
        r"(?i)^hxxp://",
        "http://",
        normalized
    )

    normalized = normalized.replace("[.]", ".")
    normalized = normalized.replace("(.)", ".")

    if normalized.lower().startswith("www."):
        normalized = "http://" + normalized

    return normalized


def extract_domain(url: str) -> str:
    """
    Extracts a domain from a normal or defanged URL.
    """

    try:
        normalized_url = normalize_defanged_url(url)
        parsed_url = urlparse(normalized_url)

        return parsed_url.netloc or "Unknown"
    except Exception:
        return "Unknown"


def analyze_single_url(url: str) -> dict:
    """
    Produces a simple rule-based URL risk report.
    """

    lower_url = str(url or "").lower()
    normalized_url = normalize_defanged_url(url)
    domain = extract_domain(url)

    score = 0
    reasons = []

    has_defanged_pattern = bool(
        re.search(
            r"(?i)hxxp|\[\.\]|\(\.\)",
            lower_url
        )
    )

    if has_defanged_pattern:
        score += 40
        reasons.append(
            "Defanged or obfuscated URL pattern detected."
        )

    suspicious_url_words = [
        "login",
        "verify",
        "verification",
        "secure",
        "security",
        "account",
        "bank",
        "payment",
        "invoice",
        "wallet",
        "reset",
        "update",
        "confirm",
        "unlock",
    ]

    matched_words = [
        word
        for word in suspicious_url_words
        if word in lower_url
    ]

    if matched_words:
        score += min(30, len(matched_words) * 10)

        reasons.append(
            "The URL contains suspicious words: "
            + ", ".join(matched_words)
        )

    lookalike_patterns = [
        "paypai",
        "micr0soft",
        "g00gle",
        "faceb00k",
        "amaz0n",
        "wa11et",
        "cust0mer",
        "acc0unt",
    ]

    if any(
        pattern in lower_url
        for pattern in lookalike_patterns
    ):
        score += 25
        reasons.append(
            "A possible look-alike domain pattern was detected."
        )

    if "xn--" in lower_url:
        score += 30
        reasons.append(
            "Punycode domain detected."
        )

    digit_count = sum(
        character.isdigit()
        for character in domain
    )

    if digit_count >= 3:
        score += 10
        reasons.append(
            "The domain contains several numeric characters."
        )

    hyphen_count = domain.count("-")

    if hyphen_count >= 3:
        score += 10
        reasons.append(
            "The domain contains an unusual number of hyphens."
        )

    if len(domain) > 40:
        score += 10
        reasons.append(
            "The domain name is unusually long."
        )

    if "@" in normalized_url:
        score += 20
        reasons.append(
            "The URL contains an @ symbol, which can hide the real destination."
        )

    if score >= 50:
        risk = "High"
    elif score >= 20:
        risk = "Suspicious"
    else:
        risk = "Low"

    if not reasons:
        reasons.append(
            "No major suspicious URL pattern was detected."
        )

    return {
        "raw_url": str(url),
        "normalized_url": normalized_url,
        "domain": domain,
        "risk": risk,
        "score": score,
        "reasons": reasons,
    }


def analyze_urls_simple(text: str) -> dict:
    """
    Analyzes every URL found in an email and returns an overall risk.
    """

    urls = extract_urls(text)

    if not urls:
        return {
            "url_count": 0,
            "highest_url_risk": "None",
            "urls": [],
        }

    analyzed_urls = [
        analyze_single_url(url)
        for url in urls
    ]

    risk_priority = {
        "None": 0,
        "Low": 1,
        "Suspicious": 2,
        "High": 3,
    }

    highest_url_risk = max(
        (
            item["risk"]
            for item in analyzed_urls
        ),
        key=lambda risk: risk_priority.get(risk, 0),
    )

    return {
        "url_count": len(analyzed_urls),
        "highest_url_risk": highest_url_risk,
        "urls": analyzed_urls,
    }


# =========================================================
# Suspicious keyword analysis
# =========================================================
def count_suspicious_keywords(text: str) -> int:
    """
    Counts how many suspicious keyword types appear in the email.
    """

    lower_text = str(text or "").lower()

    return sum(
        1
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_text
    )


def get_detected_suspicious_keywords(
    text: str
) -> list[str]:
    """
    Returns the suspicious words found in the email.
    """

    lower_text = str(text or "").lower()

    return sorted([
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_text
    ])


# =========================================================
# Feature-row creation
# =========================================================
def build_feature_row(
    subject: str,
    body: str
) -> pd.DataFrame:
    """
    Converts one email subject and body into the feature format
    expected by the trained XGBoost and hybrid models.
    """

    cleaned_subject = clean_text(subject)
    cleaned_body = clean_text(body)

    email_text = (
        f"{cleaned_subject} {cleaned_body}"
    ).strip()

    clean_email_text = normalize_for_model(
        email_text
    )

    extracted_urls = extract_urls(
        email_text
    )

    has_defanged_url = int(
        bool(
            re.search(
                r"(?i)hxxp|\[\.\]|\(\.\)",
                email_text
            )
        )
    )

    feature_row = {
        TEXT_COLUMN: clean_email_text,

        "url_count_extracted": len(
            extracted_urls
        ),

        "has_url": int(
            len(extracted_urls) > 0
        ),

        "has_defanged_url": (
            has_defanged_url
        ),

        "text_char_count": len(
            email_text
        ),

        "text_word_count": len(
            email_text.split()
        ),

        "subject_char_count": len(
            cleaned_subject
        ),

        "body_char_count": len(
            cleaned_body
        ),

        "suspicious_keyword_count": (
            count_suspicious_keywords(
                email_text
            )
        ),

        "email_entropy": calculate_entropy(
            clean_email_text
        ),

        "detected_language": detect_language(
            email_text
        ),
    }

    return pd.DataFrame(
        [feature_row]
    )


# =========================================================
# Explainability / Why Flagged
# =========================================================
def build_reasons(
    features: pd.DataFrame,
    url_report: dict,
    prediction: str,
    confidence: float,
    xgb_prediction: str,
    xgb_confidence: float,
) -> list[str]:
    """
    Creates human-readable reasons for the hybrid-model prediction.
    """

    if features.empty:
        return [
            "No feature information was available."
        ]

    row = features.iloc[0]

    reasons = [
        (
            "The final hybrid model predicted "
            f"'{str(prediction).replace('_', ' ')}' "
            f"with {confidence * 100:.2f}% confidence."
        ),
        (
            "The XGBoost structured branch predicted "
            f"'{str(xgb_prediction).replace('_', ' ')}' "
            f"with {xgb_confidence * 100:.2f}% confidence."
        ),
    ]

    suspicious_count = int(
        row.get(
            "suspicious_keyword_count",
            0
        )
    )

    if suspicious_count > 0:
        reasons.append(
            f"The email contains {suspicious_count} "
            "suspicious phishing-related keyword type(s)."
        )

    has_url = int(
        row.get(
            "has_url",
            0
        )
    )

    url_count = int(
        row.get(
            "url_count_extracted",
            0
        )
    )

    if has_url == 1:
        reasons.append(
            f"The email contains {url_count} URL(s)."
        )

    has_defanged_url = int(
        row.get(
            "has_defanged_url",
            0
        )
    )

    if has_defanged_url == 1:
        reasons.append(
            "The email contains an obfuscated or defanged URL "
            "such as hxxp or [.]"
        )

    highest_url_risk = url_report.get(
        "highest_url_risk",
        "None"
    )

    if highest_url_risk in {
        "Suspicious",
        "High",
    }:
        reasons.append(
            "The URL-analysis component identified "
            f"a {highest_url_risk} URL risk level."
        )

    detected_language = str(
        row.get(
            "detected_language",
            "Unknown"
        )
    )

    if detected_language not in {
        "English",
        "Unknown",
    }:
        reasons.append(
            "The email contains multilingual or code-mixed text: "
            f"{detected_language}."
        )

    entropy_value = float(
        row.get(
            "email_entropy",
            0.0
        )
    )

    if entropy_value > 4.5:
        reasons.append(
            "The email has relatively high text entropy, "
            "which may indicate unusual or obfuscated content."
        )

    if prediction == "Business_Email_Compromise":
        reasons.append(
            "The email follows a business, invoice, payment, "
            "or urgent-transfer risk pattern."
        )

    if prediction == "AI_Generated_Phishing":
        reasons.append(
            "The email text pattern is similar to polished or "
            "AI-generated phishing content learned by the model."
        )

    if prediction == "Traditional_Phishing":
        reasons.append(
            "The model detected patterns commonly associated "
            "with credential theft or deceptive verification requests."
        )

    if prediction == "Spam":
        reasons.append(
            "The model detected promotional, unsolicited, "
            "or reward-related email patterns."
        )

    if prediction == "Legitimate":
        reasons.append(
            "The combined text and structured-feature pattern "
            "was most similar to legitimate email samples."
        )

    return reasons