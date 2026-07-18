from pathlib import Path
import csv
import random


OUTPUT_PATH = Path("/workspace/data/multilingual/code_mixed_samples.csv")
ROW_COUNT = 20_000
RANDOM_SEED = 42


SINHALA_SUBJECTS = [
    "Account verification required",
    "Urgent bank update",
    "University portal notice",
    "Salary confirmation needed",
    "Delivery address update",
    "Payment hold notification",
    "Security alert for your account",
    "Document approval pending",
]

TAMIL_SUBJECTS = [
    "Account verification required",
    "Urgent account update",
    "University portal notice",
    "Salary confirmation needed",
    "Delivery address update",
    "Payment hold notification",
    "Security alert for your account",
    "Document approval pending",
]

SINHALA_TEMPLATES = [
    "Dear customer, ඔබගේ account එක verify කරන්න. Please visit {url} before {timeframe}.",
    "Hello user, security update එක complete කරන්න. Login details confirm කරන්න using {url}.",
    "Dear student, university portal access එක temporary hold කර ඇත. Please update details here: {url}.",
    "Dear employee, salary details update කරන්න. Payroll confirmation required at {url}.",
    "Hello, ඔබගේ delivery address එක confirm කරන්න. Parcel release කිරීමට {url} visit කරන්න.",
    "Dear member, payment issue එකක් detect විය. Account restore කිරීමට {url} open කරන්න.",
    "Security team notice: ඔබගේ password එක expire වෙයි. New verification required: {url}.",
    "Dear applicant, document approval pending. ඔබගේ profile details confirm කරන්න: {url}.",
]

TAMIL_TEMPLATES = [
    "Dear customer, உங்கள் account உடனடியாக verify செய்ய வேண்டும். Please visit {url} before {timeframe}.",
    "Hello user, security update complete செய்யுங்கள். Login details confirm செய்ய {url} use செய்யவும்.",
    "Dear student, university portal access temporary hold செய்யப்பட்டுள்ளது. Details update செய்ய: {url}.",
    "Dear employee, salary details update செய்ய வேண்டும். Payroll confirmation required at {url}.",
    "Hello, உங்கள் delivery address confirm செய்யுங்கள். Parcel release செய்ய {url} visit செய்யவும்.",
    "Dear member, payment issue detect செய்யப்பட்டது. Account restore செய்ய {url} open செய்யவும்.",
    "Security team notice: உங்கள் password expire ஆகும். New verification required: {url}.",
    "Dear applicant, document approval pending. உங்கள் profile details confirm செய்யவும்: {url}.",
]

AI_STYLE_PHRASES = [
    "This automated message was generated after a recent security review.",
    "Our system detected unusual activity and requires immediate confirmation.",
    "To avoid interruption, complete the verification process as soon as possible.",
    "The request will be closed if confirmation is not completed today.",
    "This notice is part of the latest account protection process.",
]

TIMEFRAMES = ["today", "24 hours", "48 hours", "this evening", "the next login"]

FICTIONAL_SENDERS = [
    "security@sample-bank.test",
    "support@demo-pay.test",
    "notice@campus-portal.test",
    "hr@sample-company.test",
    "delivery@parcel-demo.test",
    "admin@secure-example.test",
]

FICTIONAL_RECEIVERS = [
    "user001@example.test",
    "student@example.test",
    "employee@example.test",
    "customer@example.test",
]

URLS = [
    "hxxps://example[.]test/login",
    "hxxps://sample-bank[.]test/verify",
    "hxxps://campus-portal[.]test/update",
    "hxxps://demo-pay[.]test/confirm",
    "hxxps://parcel-demo[.]test/release",
    "hxxps://secure-example[.]test/session",
]


def make_row(index: int, language: str) -> dict[str, str | int]:
    is_sinhala = language == "CodeMixed_Sinhala_English"
    is_ai_generated = index % 5 == 0

    subject = random.choice(SINHALA_SUBJECTS if is_sinhala else TAMIL_SUBJECTS)
    template = random.choice(SINHALA_TEMPLATES if is_sinhala else TAMIL_TEMPLATES)
    url = random.choice(URLS)
    timeframe = random.choice(TIMEFRAMES)
    body = template.format(url=url, timeframe=timeframe)

    if is_ai_generated:
        body = f"{random.choice(AI_STYLE_PHRASES)} {body}"
        category = "AI_Generated_Phishing"
        label = 4
    else:
        category = "Traditional_Phishing"
        label = 2

    email_text = f"{subject} {body}".strip()

    return {
        "email_id": f"CM_{index:06d}",
        "source_dataset": "Synthetic_CodeMixed",
        "sender": random.choice(FICTIONAL_SENDERS),
        "receiver": random.choice(FICTIONAL_RECEIVERS),
        "date": "",
        "subject": subject,
        "body": body,
        "email_text": email_text,
        "urls": 1,
        "original_label": "synthetic",
        "category": category,
        "label": label,
        "language": language,
    }


def main() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "email_id",
        "source_dataset",
        "sender",
        "receiver",
        "date",
        "subject",
        "body",
        "email_text",
        "urls",
        "original_label",
        "category",
        "label",
        "language",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(1, ROW_COUNT + 1):
            language = (
                "CodeMixed_Sinhala_English"
                if index <= ROW_COUNT // 2
                else "CodeMixed_Tamil_English"
            )
            writer.writerow(make_row(index, language))

    print(f"Created {OUTPUT_PATH}")
    print(f"Rows: {ROW_COUNT}")


if __name__ == "__main__":
    main()
