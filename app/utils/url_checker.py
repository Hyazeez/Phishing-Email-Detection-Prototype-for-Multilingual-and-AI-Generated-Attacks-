import math
import re
import ipaddress
from urllib.parse import urlparse


URL_PATTERN = re.compile(r"(?i)\b(?:https?://|hxxps?://|hxxp?://|www\.)[^\s<>\"']+")

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "cutt.ly",
    "rebrand.ly",
    "is.gd",
    "buff.ly",
    "ow.ly",
}

SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bank",
    "password",
    "otp",
    "wallet",
    "payment",
    "invoice",
    "payroll",
    "confirm",
    "signin",
    "reset",
    "unlock",
}


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    probabilities = [text.count(char) / len(text) for char in set(text)]
    return -sum(prob * math.log2(prob) for prob in probabilities)


def normalize_url(raw_url: str) -> str:
    url = raw_url.strip().strip(".,;:!?)]}\"'")
    url = url.replace("hxxps://", "https://")
    url = url.replace("hxxp://", "http://")
    url = url.replace("[.]", ".")
    url = url.replace("(.)", ".")

    if url.startswith("www."):
        url = "http://" + url

    return url


def is_ip_address(domain: str) -> bool:
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False


def analyze_single_url(raw_url: str) -> dict:
    normalized = normalize_url(raw_url)
    parsed = urlparse(normalized)
    domain = parsed.netloc.lower()

    if "@" in domain:
        domain = domain.split("@")[-1]

    if ":" in domain:
        domain = domain.split(":")[0]

    full_url = normalized.lower()
    score = 0
    reasons = []

    if "hxxp" in raw_url.lower() or "[.]" in raw_url or "(.)" in raw_url:
        score += 2
        reasons.append("Defanged or suspicious-looking URL format detected")

    if parsed.scheme == "http":
        score += 2
        reasons.append("URL uses HTTP instead of HTTPS")

    if "@" in normalized:
        score += 2
        reasons.append("URL contains '@' symbol, which can hide the real domain")

    if is_ip_address(domain):
        score += 3
        reasons.append("URL uses an IP address instead of a domain name")

    if "xn--" in domain:
        score += 2
        reasons.append("Punycode domain detected, possible homoglyph attack")

    if domain in SHORTENER_DOMAINS:
        score += 2
        reasons.append("URL shortener detected")

    keyword_hits = [keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in full_url]
    if keyword_hits:
        score += min(len(keyword_hits), 3)
        reasons.append("Suspicious URL keywords found: " + ", ".join(keyword_hits[:5]))

    if len(normalized) > 100:
        score += 1
        reasons.append("URL is unusually long")

    if domain.count(".") >= 3:
        score += 1
        reasons.append("URL contains many subdomains")

    if calculate_entropy(domain) > 3.8:
        score += 1
        reasons.append("Domain has high character randomness")

    if score == 0:
        risk = "Safe"
    elif score <= 2:
        risk = "Low"
    elif score <= 5:
        risk = "Suspicious"
    else:
        risk = "High"

    return {
        "raw_url": raw_url,
        "normalized_url": normalized,
        "domain": domain,
        "score": score,
        "risk": risk,
        "reasons": reasons or ["No major URL risk indicators detected"],
    }


def analyze_urls(text: str) -> dict:
    urls = URL_PATTERN.findall(text or "")
    results = [analyze_single_url(url) for url in urls]

    if not results:
        return {
            "url_count": 0,
            "highest_url_risk": "None",
            "highest_url_score": 0,
            "urls": [],
        }

    highest_score = max(item["score"] for item in results)
    if highest_score == 0:
        highest_risk = "Safe"
    elif highest_score <= 2:
        highest_risk = "Low"
    elif highest_score <= 5:
        highest_risk = "Suspicious"
    else:
        highest_risk = "High"

    return {
        "url_count": len(results),
        "highest_url_risk": highest_risk,
        "highest_url_score": highest_score,
        "urls": results,
    }
