from datetime import datetime

from backend.database_service import (
    init_database,
    insert_feedback_record,
    get_all_feedback_records,
    get_feedback_count,
)


# Categories accepted by the feedback API
CLASS_LABELS = [
    "Legitimate",
    "Spam",
    "Traditional_Phishing",
    "Business_Email_Compromise",
    "AI_Generated_Phishing",
]


def save_feedback(
    subject: str,
    body: str,
    prediction_result: dict,
    feedback_status: str,
    correct_category: str,
    feedback_note: str = "",
) -> dict:
    """
    Prepare and save one feedback record into MySQL.
    """

    if feedback_status not in {"Correct", "Wrong"}:
        raise ValueError(
            "feedback_status must be either 'Correct' or 'Wrong'."
        )

    if correct_category not in CLASS_LABELS:
        raise ValueError(
            "Invalid correct category. Valid categories are: "
            + ", ".join(CLASS_LABELS)
        )

    # Ensure the feedback table exists
    init_database()

    features = prediction_result.get("features", {})

    feedback_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": subject,
        "body": body,
        "predicted_category": prediction_result.get(
            "prediction",
            "",
        ),
        "prediction_confidence": prediction_result.get(
            "confidence",
            0.0,
        ),
        "xgboost_prediction": prediction_result.get(
            "xgboost_prediction",
            "",
        ),
        "xgboost_confidence": prediction_result.get(
            "xgboost_confidence",
            0.0,
        ),
        "feedback_status": feedback_status,
        "correct_category": correct_category,
        "feedback_note": feedback_note,
        "detected_language": prediction_result.get(
            "detected_language",
            "Unknown",
        ),
        "url_count_extracted": features.get(
            "url_count_extracted",
            0,
        ),
        "has_url": features.get(
            "has_url",
            0,
        ),
        "has_defanged_url": features.get(
            "has_defanged_url",
            0,
        ),
        "suspicious_keyword_count": features.get(
            "suspicious_keyword_count",
            0,
        ),
        "email_entropy": features.get(
            "email_entropy",
            0.0,
        ),
        "clean_email_text": features.get(
            "clean_email_text",
            "",
        ),
    }

    feedback_id = insert_feedback_record(
        feedback_record
    )

    total_records = get_feedback_count()

    return {
        "message": "Feedback saved successfully.",
        "feedback_id": feedback_id,
        "database": "MySQL",
        "total_feedback_records": total_records,
    }


def read_feedback() -> dict:
    """
    Read all feedback records from MySQL.
    """

    init_database()

    records = get_all_feedback_records()

    return {
        "database": "MySQL",
        "total_feedback_records": len(records),
        "records": records,
    }