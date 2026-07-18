from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


# =========================
# Application Configuration
# =========================
st.set_page_config(
    page_title="Hybrid Phishing Email Detection",
    page_icon="🛡️",
    layout="wide",
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

OCR_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
OCR_SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]
EML_MAX_UPLOAD_BYTES = 15 * 1024 * 1024
EML_SUPPORTED_EXTENSIONS = ["eml"]

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "reports" / "results"

ROBUSTNESS_RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "robustness"
    / "mlp_fusion_robustness_test_results.csv"
)
EXPLAINABILITY_RESULTS_PATH = (
    RESULTS_DIR / "hybrid_explainability_results.csv"
)
XGB_IMPORTANCE_PATH = (
    RESULTS_DIR / "hybrid_xgboost_global_feature_importance.csv"
)

CLASS_LABELS = [
    "Legitimate",
    "Spam",
    "Traditional_Phishing",
    "Business_Email_Compromise",
    "AI_Generated_Phishing",
]

SAMPLES = {
    "Legitimate": {
        "subject": "Project meeting tomorrow",
        "body": (
            "Hi team, please attend the project meeting tomorrow at 10 AM. "
            "We will discuss the progress update and next tasks."
        ),
    },
    "Spam": {
        "subject": "Special discount offer",
        "body": (
            "Congratulations! You have been selected for a limited time offer. "
            "Click now to claim your discount coupon."
        ),
    },
    "Traditional Phishing": {
        "subject": "Account verification required",
        "body": (
            "Dear customer, your account will be suspended. Please verify your "
            "login details at hxxps://secure-example[.]test/login within 24 hours."
        ),
    },
    "Business Email Compromise": {
        "subject": "Urgent payment request",
        "body": (
            "Dear finance team, please process this urgent invoice payment today. "
            "Confirm once the transfer is completed."
        ),
    },
    "Sinhala-English": {
        "subject": "Account verification required",
        "body": (
            "Dear customer, ඔබගේ account එක verify කරන්න. Please visit "
            "hxxps://sample-bank[.]test/verify before 24 hours."
        ),
    },
    "Tamil-English": {
        "subject": "Bank account update",
        "body": (
            "Dear user, உங்கள் bank account உடனடியாக verify செய்ய வேண்டும். "
            "Please visit hxxps://demo-pay[.]test/confirm."
        ),
    },
    "AI-generated Phishing": {
        "subject": "Important security review required",
        "body": (
            "Dear customer, we detected unusual activity in your account. "
            "Please complete the secure verification process within 24 hours "
            "to avoid temporary restrictions."
        ),
    },
}


# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #d1d5db;
        margin-bottom: 20px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 15px;
        color: #f9fafb !important;
        line-height: 1.7;
    }

    .info-card * {
        color: #f9fafb !important;
    }

    .safe-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        text-align: center;
    }

    .medium-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #fef9c3;
        color: #854d0e;
        font-weight: 700;
        text-align: center;
    }

    .high-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        text-align: center;
    }

    .critical-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #fecaca;
        color: #7f1d1d;
        font-weight: 800;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# FastAPI Client Functions
# =========================
def call_health_api() -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def call_prediction_api(subject: str, body: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={
            "subject": subject,
            "body": body,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def call_feedback_api(
    subject: str,
    body: str,
    feedback_status: str,
    correct_category: str,
    feedback_note: str = "",
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/feedback",
        json={
            "subject": subject,
            "body": body,
            "feedback_status": feedback_status,
            "correct_category": correct_category,
            "feedback_note": feedback_note,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def call_get_feedback_api() -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/feedback",
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def call_explanation_api(
    subject: str,
    body: str,
    num_features: int = 10,
    num_samples: int = 100,
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/explain",
        json={
            "subject": subject,
            "body": body,
            "num_features": num_features,
            "num_samples": num_samples,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()

def call_screenshot_ocr_api(
    uploaded_file: Any,
) -> dict[str, Any]:
    uploaded_bytes = uploaded_file.getvalue()

    response = requests.post(
        f"{API_BASE_URL}/extract-email-screenshot",
        files={
            "file": (
                uploaded_file.name,
                uploaded_bytes,
                uploaded_file.type
                or "application/octet-stream",
            )
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def call_eml_extraction_api(
    uploaded_file: Any,
) -> dict[str, Any]:
    uploaded_bytes = uploaded_file.getvalue()

    response = requests.post(
        f"{API_BASE_URL}/extract-eml",
        files={
            "file": (
                uploaded_file.name,
                uploaded_bytes,
                uploaded_file.type
                or "message/rfc822",
            )
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()

# =========================
# UI Helper Functions
# =========================
def show_request_error(error: requests.exceptions.RequestException, action: str) -> None:
    if isinstance(error, requests.exceptions.ConnectionError):
        st.error(
            f"Cannot connect to the FastAPI backend while {action}. "
            f"Confirm that it is running at {API_BASE_URL}."
        )
        return

    if isinstance(error, requests.exceptions.Timeout):
        st.error(f"The request timed out while {action}.")
        return

    if isinstance(error, requests.exceptions.HTTPError):
        try:
            detail = error.response.json()
        except Exception:
            detail = error.response.text

        st.error(f"FastAPI error while {action}: {detail}")
        return

    st.error(f"Request failed while {action}: {error}")


def show_risk_badge(risk: str) -> None:
    if risk == "Safe":
        st.markdown(
            '<div class="safe-card">✅ SAFE EMAIL</div>',
            unsafe_allow_html=True,
        )
    elif risk == "Medium":
        st.markdown(
            '<div class="medium-card">⚠️ MEDIUM RISK</div>',
            unsafe_allow_html=True,
        )
    elif risk == "High":
        st.markdown(
            '<div class="high-card">🚨 HIGH RISK</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="critical-card">🚨 CRITICAL RISK</div>',
            unsafe_allow_html=True,
        )


def show_probability_table(
    class_names: list[str],
    probabilities: list[float],
) -> None:
    probability_df = pd.DataFrame(
        {
            "Class": [name.replace("_", " ") for name in class_names],
            "Probability": probabilities,
        }
    )
    probability_df["Probability (%)"] = (
        probability_df["Probability"] * 100
    ).round(2)
    probability_df = probability_df.sort_values(
        "Probability (%)",
        ascending=False,
    )

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(
        probability_df[["Class", "Probability (%)"]],
        use_container_width=True,
        hide_index=True,
    )


def get_model_benchmark_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "CNN",
                "LSTM",
                "XGBoost Structured Only",
                "Transformer + XGBoost + MLP Fusion",
            ],
            "Model Type": [
                "Classical ML Baseline",
                "Classical ML Baseline",
                "Deep Learning Baseline",
                "Deep Learning Baseline",
                "Structured Feature Model",
                "Hybrid Model",
            ],
            "Test Accuracy (%)": [
                98.01,
                98.28,
                97.77,
                87.82,
                80.08,
                95.82,
            ],
            "Macro F1": [
                0.980,
                0.984,
                0.979,
                0.886,
                0.819,
                0.948,
            ],
            "Remarks": [
                "Strong TF-IDF baseline",
                "Best benchmark model",
                "Strong deep-learning baseline",
                "Lower performance",
                "Structured features only",
                "Proposed hybrid architecture",
            ],
        }
    )


def render_xai_results(xai_result: dict[str, Any]) -> None:
    lime_result = xai_result.get("lime_text", {})
    shap_result = xai_result.get("shap_structured", {})

    if not lime_result and not shap_result:
        st.warning("The explanation response did not contain LIME or SHAP results.")
        return

    if lime_result:
        st.markdown("### LIME Word-Level Explanation")
        st.caption(
            "LIME perturbs the selected email text and repeatedly calls the complete "
            "Transformer–XGBoost–MLP prediction pipeline."
        )

        lime_col1, lime_col2, lime_col3, lime_col4 = st.columns(4)
        lime_col1.metric(
            "Explained Prediction",
            str(lime_result.get("predicted_class", "Unknown")).replace("_", " "),
        )
        lime_col2.metric(
            "Confidence",
            f"{float(lime_result.get('prediction_confidence', 0.0)):.2%}",
        )
        lime_col3.metric(
            "Local Fidelity R²",
            f"{float(lime_result.get('local_fidelity_r2', 0.0)):.4f}",
        )
        lime_col4.metric(
            "Perturbation Samples",
            int(lime_result.get("num_samples", 0)),
        )

        lime_terms = lime_result.get("terms", [])
        lime_df = pd.DataFrame(lime_terms)

        if lime_df.empty:
            st.info("No LIME terms were returned.")
        else:
            preferred_lime_columns = ["term", "weight", "effect"]
            available_lime_columns = [
                column
                for column in preferred_lime_columns
                if column in lime_df.columns
            ]

            st.dataframe(
                lime_df[available_lime_columns],
                use_container_width=True,
                hide_index=True,
            )

            if {"term", "weight"}.issubset(lime_df.columns):
                lime_chart_df = (
                    lime_df[["term", "weight"]]
                    .sort_values("weight")
                    .set_index("term")
                )
                st.bar_chart(lime_chart_df)

        st.info(
            "A positive LIME weight supports the displayed hybrid prediction. "
            "A negative weight opposes it."
        )

    if shap_result:
        st.markdown("### SHAP Structured-Feature Explanation")
        st.caption(
            "SHAP explains the XGBoost structured branch, including URL, language, "
            "length, keyword, and entropy-related inputs."
        )

        shap_col1, shap_col2, shap_col3 = st.columns(3)
        shap_col1.metric(
            "XGBoost Prediction",
            str(shap_result.get("predicted_class", "Unknown")).replace("_", " "),
        )
        shap_col2.metric(
            "XGBoost Confidence",
            f"{float(shap_result.get('prediction_confidence', 0.0)):.2%}",
        )
        shap_col3.metric(
            "Expected Value",
            f"{float(shap_result.get('expected_value', 0.0)):.4f}",
        )

        shap_features = shap_result.get("features", [])
        shap_df = pd.DataFrame(shap_features)

        if shap_df.empty:
            st.info("No SHAP feature contributions were returned.")
        else:
            preferred_shap_columns = [
                "feature",
                "processed_value",
                "shap_value",
                "effect",
            ]
            available_shap_columns = [
                column
                for column in preferred_shap_columns
                if column in shap_df.columns
            ]

            st.dataframe(
                shap_df[available_shap_columns],
                use_container_width=True,
                hide_index=True,
            )

            if {"feature", "shap_value"}.issubset(shap_df.columns):
                shap_chart_df = (
                    shap_df[["feature", "shap_value"]]
                    .sort_values("shap_value")
                    .set_index("feature")
                )
                st.bar_chart(shap_chart_df)

        st.info(
            "A positive SHAP value increases support for the XGBoost predicted "
            "class. A negative value reduces support for that class."
        )


def render_explanation_controls(
    subject: str,
    body: str,
    key_prefix: str,
) -> None:
    st.subheader("🧠 Explainable AI Analysis")
    st.caption(
        "LIME explains influential words in the complete hybrid prediction. "
        "SHAP explains the XGBoost structured-feature branch."
    )

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        num_features = st.slider(
            "Number of LIME terms",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key=f"{key_prefix}_num_features",
        )

    with control_col2:
        num_samples = st.slider(
            "LIME perturbation samples",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            key=f"{key_prefix}_num_samples",
            help=(
                "Higher values may improve explanation stability but increase CPU "
                "processing time."
            ),
        )

    generate_explanation = st.button(
        "Generate LIME and SHAP Explanation",
        use_container_width=True,
        key=f"{key_prefix}_generate_xai",
    )

    if generate_explanation:
        if not subject.strip() and not body.strip():
            st.warning("Analyse an email before generating an explanation.")
        else:
            try:
                with st.spinner(
                    "Generating explanations. LIME may take several minutes on CPU..."
                ):
                    explanation_result = call_explanation_api(
                        subject=subject,
                        body=body,
                        num_features=num_features,
                        num_samples=num_samples,
                    )

                st.session_state["last_xai_result"] = explanation_result
                st.session_state["last_xai_subject"] = subject
                st.session_state["last_xai_body"] = body
                st.success("LIME and SHAP explanations generated successfully.")

            except requests.exceptions.RequestException as error:
                show_request_error(error, "generating explanations")

    xai_result = st.session_state.get("last_xai_result")
    xai_subject = st.session_state.get("last_xai_subject", "")
    xai_body = st.session_state.get("last_xai_body", "")

    if xai_result:
        if xai_subject == subject and xai_body == body:
            render_xai_results(xai_result)
        else:
            st.info(
                "A previous explanation exists for a different email. Generate a new "
                "explanation for the current input."
            )


# =========================
# Pages
# =========================
def page_email_prediction() -> None:
    st.header("📧 Email Prediction Using Hybrid Model")

    try:
        health_result = call_health_api()
        model_status = health_result.get("model_loaded", True)
        if model_status:
            st.success("FastAPI backend and hybrid model are available.")
        else:
            st.warning("FastAPI is running, but the model may not be loaded.")
    except requests.exceptions.RequestException:
        st.warning(
            f"FastAPI status could not be confirmed at {API_BASE_URL}. "
            "Prediction will fail until the backend is running."
        )

    st.subheader("Sample Emails")
    sample_columns = st.columns(4)

    for index, (name, sample) in enumerate(SAMPLES.items()):
        if sample_columns[index % 4].button(
            name,
            key=f"sample_{index}",
            use_container_width=True,
        ):
            st.session_state["subject"] = sample["subject"]
            st.session_state["body"] = sample["body"]
            st.session_state.pop("last_analysis_result", None)
            st.session_state.pop("last_xai_result", None)
            st.session_state.pop("ocr_result", None)
            st.session_state.pop("eml_result", None)

    st.subheader("📷 Analyse an Email Screenshot")
    st.caption(
        "Upload a clear screenshot of an email. FastAPI will extract English, "
        "Sinhala, and Tamil text using Tesseract OCR. Review and edit the "
        "extracted subject and body before classification."
    )

    upload_col, extract_col = st.columns([3, 1])

    with upload_col:
        uploaded_screenshot = st.file_uploader(
            "Upload email screenshot",
            type=OCR_SUPPORTED_EXTENSIONS,
            accept_multiple_files=False,
            key="email_screenshot_uploader",
            help="Supported formats: PNG, JPG, JPEG, and WEBP. Maximum size: 10 MB.",
        )

    screenshot_too_large = bool(
        uploaded_screenshot is not None
        and int(getattr(uploaded_screenshot, "size", 0)) > OCR_MAX_UPLOAD_BYTES
    )

    with extract_col:
        st.write("")
        st.write("")
        extract_screenshot_button = st.button(
            "Extract Email Text",
            type="secondary",
            use_container_width=True,
            disabled=uploaded_screenshot is None or screenshot_too_large,
            key="extract_email_screenshot_button",
        )

    if uploaded_screenshot is not None:
        if screenshot_too_large:
            st.error("The screenshot exceeds the 10 MB upload limit.")
        else:
            st.image(
                uploaded_screenshot,
                caption=uploaded_screenshot.name,
                use_container_width=True,
            )

    if extract_screenshot_button and uploaded_screenshot is not None:
        try:
            with st.spinner(
                "Extracting text with Tesseract OCR. This may take a few seconds..."
            ):
                ocr_result = call_screenshot_ocr_api(uploaded_screenshot)

            extracted_subject = str(ocr_result.get("subject", "")).strip()
            extracted_body = str(ocr_result.get("body", "")).strip()
            complete_ocr_text = str(
                ocr_result.get("extracted_text", "")
            ).strip()

            if not extracted_body:
                extracted_body = complete_ocr_text

            st.session_state["ocr_result"] = ocr_result
            st.session_state["subject"] = extracted_subject
            st.session_state["body"] = extracted_body

            # Clear outputs created for a previously analysed email.
            st.session_state.pop("last_analysis_result", None)
            st.session_state.pop("last_xai_result", None)
            st.session_state.pop("last_xai_subject", None)
            st.session_state.pop("last_xai_body", None)

            st.success(
                "Screenshot text extracted successfully. Review the editable "
                "subject and body fields, then analyse the email."
            )
            st.rerun()

        except requests.exceptions.RequestException as error:
            show_request_error(error, "extracting text from the screenshot")

    ocr_result = st.session_state.get("ocr_result")
    if ocr_result:
        st.markdown("#### OCR Extraction Result")

        ocr_metric_col1, ocr_metric_col2, ocr_metric_col3 = st.columns(3)
        ocr_metric_col1.metric(
            "OCR Confidence",
            f"{float(ocr_result.get('ocr_confidence', 0.0)):.2%}",
        )
        ocr_metric_col2.metric(
            "Detected Language",
            str(ocr_result.get("detected_language", "Unknown")).replace("_", " "),
        )
        ocr_metric_col3.metric(
            "OCR Languages",
            ", ".join(
                str(language)
                for language in ocr_result.get("ocr_languages", [])
            )
            or "Unknown",
        )

        for warning in ocr_result.get("warnings", []):
            st.warning(str(warning))

        with st.expander("View Complete OCR Text", expanded=False):
            st.text_area(
                "Extracted screenshot text",
                value=str(ocr_result.get("extracted_text", "")),
                height=250,
                disabled=True,
                key="complete_ocr_text_preview",
            )

        if st.button(
            "Clear OCR Result",
            key="clear_ocr_result_button",
        ):
            st.session_state.pop("ocr_result", None)
            st.session_state.pop("last_analysis_result", None)
            st.session_state.pop("last_xai_result", None)
            st.rerun()

    st.subheader("📨 Analyse an Original .eml Email File")
    st.caption(
        "Upload an original .eml file to extract the subject, body, sender, "
        "authentication results, domain mismatches, and attachment metadata. "
        "Header indicators are displayed as a separate security report and are "
        "not yet part of the trained 785-feature hybrid model."
    )

    eml_upload_col, eml_extract_col = st.columns([3, 1])

    with eml_upload_col:
        uploaded_eml = st.file_uploader(
            "Upload original email file",
            type=EML_SUPPORTED_EXTENSIONS,
            accept_multiple_files=False,
            key="original_eml_uploader",
            help="Supported format: .eml. Maximum size: 15 MB.",
        )

    eml_too_large = bool(
        uploaded_eml is not None
        and int(getattr(uploaded_eml, "size", 0)) > EML_MAX_UPLOAD_BYTES
    )

    with eml_extract_col:
        st.write("")
        st.write("")
        extract_eml_button = st.button(
            "Extract .eml Details",
            type="secondary",
            use_container_width=True,
            disabled=uploaded_eml is None or eml_too_large,
            key="extract_original_eml_button",
        )

    if uploaded_eml is not None and eml_too_large:
        st.error("The .eml file exceeds the 15 MB upload limit.")

    if extract_eml_button and uploaded_eml is not None:
        try:
            with st.spinner(
                "Parsing the original email, authentication headers, and attachments..."
            ):
                eml_result = call_eml_extraction_api(uploaded_eml)

            extracted_subject = str(eml_result.get("subject", "")).strip()
            extracted_body = str(eml_result.get("body", "")).strip()

            st.session_state["eml_result"] = eml_result
            st.session_state["subject"] = extracted_subject
            st.session_state["body"] = extracted_body

            # The latest uploaded source replaces previous screenshot and model outputs.
            st.session_state.pop("ocr_result", None)
            st.session_state.pop("last_analysis_result", None)
            st.session_state.pop("last_xai_result", None)
            st.session_state.pop("last_xai_subject", None)
            st.session_state.pop("last_xai_body", None)

            st.success(
                "The .eml file was parsed successfully. Review the extracted email "
                "and header-security report, then run the hybrid prediction."
            )
            st.rerun()

        except requests.exceptions.RequestException as error:
            show_request_error(error, "extracting the original .eml email")

    eml_result = st.session_state.get("eml_result")
    if eml_result:
        st.markdown("#### Original Email and Header-Security Report")

        headers = eml_result.get("headers", {})
        authentication = eml_result.get("authentication", {})
        security_features = eml_result.get("security_features", {})
        attachments = eml_result.get("attachments", [])
        header_risk = str(eml_result.get("header_risk", "Low"))
        header_risk_score = int(eml_result.get("header_risk_score", 0))

        sender_col, risk_col, attachment_col = st.columns(3)
        sender_col.metric(
            "Sender",
            str(headers.get("from_address", "Unknown")) or "Unknown",
        )
        risk_col.metric(
            "Header Risk",
            f"{header_risk} ({header_risk_score})",
        )
        attachment_col.metric(
            "Attachments",
            int(security_features.get("attachment_count", len(attachments))),
        )

        auth_col1, auth_col2, auth_col3 = st.columns(3)
        auth_col1.metric("SPF", str(authentication.get("spf", "unknown")).upper())
        auth_col2.metric("DKIM", str(authentication.get("dkim", "unknown")).upper())
        auth_col3.metric("DMARC", str(authentication.get("dmarc", "unknown")).upper())

        if header_risk in {"High", "Critical"}:
            st.error(
                "The original email headers contain high-risk indicators. "
                "Verify the sender through a trusted channel."
            )
        elif header_risk == "Medium":
            st.warning(
                "The email contains header or attachment indicators that require review."
            )
        else:
            st.info(
                "No strong header anomaly was identified, but this does not prove that "
                "the email is safe."
            )

        for warning in eml_result.get("warnings", []):
            st.warning(str(warning))

        mismatch_df = pd.DataFrame(
            [
                {
                    "Security feature": "From / Reply-To mismatch",
                    "Detected": bool(
                        int(security_features.get("from_replyto_mismatch", 0))
                    ),
                },
                {
                    "Security feature": "From / Return-Path mismatch",
                    "Detected": bool(
                        int(security_features.get("from_returnpath_mismatch", 0))
                    ),
                },
                {
                    "Security feature": "Message-ID domain mismatch",
                    "Detected": bool(
                        int(security_features.get("message_id_domain_mismatch", 0))
                    ),
                },
                {
                    "Security feature": "Executable attachment",
                    "Detected": bool(
                        int(security_features.get("has_executable_attachment", 0))
                    ),
                },
                {
                    "Security feature": "Macro-enabled attachment",
                    "Detected": bool(
                        int(security_features.get("has_macro_attachment", 0))
                    ),
                },
            ]
        )
        st.dataframe(
            mismatch_df,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View Extracted Email Headers", expanded=False):
            header_rows = [
                {"Header": "From", "Value": headers.get("from", "")},
                {"Header": "Reply-To", "Value": headers.get("reply_to", "")},
                {"Header": "Return-Path", "Value": headers.get("return_path", "")},
                {"Header": "Date", "Value": headers.get("date", "")},
                {"Header": "Message-ID", "Value": headers.get("message_id", "")},
                {
                    "Header": "Received header count",
                    "Value": headers.get("received_header_count", 0),
                },
            ]
            st.dataframe(
                pd.DataFrame(header_rows),
                use_container_width=True,
                hide_index=True,
            )

            authentication_results = authentication.get(
                "authentication_results", []
            )
            if authentication_results:
                st.markdown("**Authentication-Results**")
                for value in authentication_results:
                    st.code(str(value), language=None)

            received_spf = str(authentication.get("received_spf", "")).strip()
            if received_spf:
                st.markdown("**Received-SPF**")
                st.code(received_spf, language=None)

        with st.expander("View Attachment Metadata", expanded=False):
            if attachments:
                attachment_df = pd.DataFrame(attachments)
                display_columns = [
                    column
                    for column in [
                        "filename",
                        "content_type",
                        "size_bytes",
                        "extension",
                        "risk",
                        "warnings",
                    ]
                    if column in attachment_df.columns
                ]
                st.dataframe(
                    attachment_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No attachments were detected in the .eml file.")

        with st.expander("View Parsed Email Body", expanded=False):
            st.write("Body source:", eml_result.get("body_source", "Unknown"))
            st.text_area(
                "Parsed original email body",
                value=str(eml_result.get("body", "")),
                height=250,
                disabled=True,
                key="parsed_eml_body_preview",
            )

        if st.button(
            "Clear .eml Result",
            key="clear_eml_result_button",
        ):
            st.session_state.pop("eml_result", None)
            st.session_state.pop("last_analysis_result", None)
            st.session_state.pop("last_xai_result", None)
            st.rerun()

    col_input, col_help = st.columns([2, 1])

    with col_input:
        subject = st.text_input(
            "Email Subject",
            key="subject",
        )

        body = st.text_area(
            "Email Body",
            height=240,
            key="body",
        )

        analyze_button = st.button(
            "Analyze Email with Hybrid Model",
            type="primary",
            use_container_width=True,
        )

    with col_help:
        st.markdown(
            """
            <div class="info-card">
            <b>Hybrid Architecture</b><br><br>
            1. Multilingual Transformer embeddings<br>
            2. XGBoost structured-feature probabilities<br>
            3. Structured phishing indicators<br>
            4. MLP fusion classifier<br><br>
            <b>Screenshot OCR</b><br><br>
            Extracts English, Sinhala, and Tamil email text from screenshots.<br><br>
            <b>Original .eml Analysis</b><br><br>
            Extracts sender headers, SPF, DKIM, DMARC, domain mismatches, and attachment metadata.<br><br>
            <b>Explainability</b><br><br>
            LIME word-level explanation and SHAP structured-feature explanation.<br><br>
            <b>Continual Learning</b><br><br>
            User feedback is stored in MySQL for controlled future retraining.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_button:
        if not subject.strip() and not body.strip():
            st.warning("Please enter an email subject or body.")
            return

        try:
            with st.spinner("Sending email to the FastAPI backend..."):
                result = call_prediction_api(subject, body)

        except requests.exceptions.RequestException as error:
            show_request_error(error, "analysing the email")
            return

        probability_dict = result.get("probabilities", {})
        class_names = list(probability_dict.keys())
        probabilities = [
            float(probability_dict[class_name])
            for class_name in class_names
        ]

        st.session_state.pop("last_xai_result", None)
        st.session_state.pop("last_xai_subject", None)
        st.session_state.pop("last_xai_body", None)

        st.session_state["last_analysis_result"] = {
            "subject": subject,
            "body": body,
            "features": pd.DataFrame([result.get("features", {})]),
            "url_report": result.get(
                "url_report",
                {
                    "url_count": 0,
                    "highest_url_risk": "None",
                    "urls": [],
                },
            ),
            "prediction": str(result.get("prediction", "Unknown")),
            "probabilities": probabilities,
            "class_names": class_names,
            "xgb_prediction": str(result.get("xgboost_prediction", "Unknown")),
            "xgb_confidence": float(result.get("xgboost_confidence", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
            "risk": str(result.get("risk", "High")),
            "reasons": result.get("reasons", []),
            "model": result.get("model", {}),
        }

    result_data = st.session_state.get("last_analysis_result")
    if not result_data:
        return

    subject = result_data["subject"]
    body = result_data["body"]
    features = result_data["features"]
    url_report = result_data["url_report"]
    prediction = result_data["prediction"]
    probabilities = result_data["probabilities"]
    class_names = result_data["class_names"]
    xgb_prediction = result_data["xgb_prediction"]
    xgb_confidence = result_data["xgb_confidence"]
    confidence = result_data["confidence"]
    risk = result_data["risk"]
    reasons = result_data["reasons"]

    st.subheader("Prediction Result")

    detected_language = "Unknown"
    if not features.empty and "detected_language" in features.columns:
        detected_language = str(features.iloc[0]["detected_language"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Hybrid Prediction", prediction.replace("_", " "))
    col2.metric("Final Confidence", f"{confidence:.2%}")
    col3.metric("Detected Language", detected_language)

    col4, col5 = st.columns(2)
    col4.metric(
        "XGBoost Branch Prediction",
        xgb_prediction.replace("_", " "),
    )
    col5.metric(
        "XGBoost Branch Confidence",
        f"{xgb_confidence:.2%}",
    )

    show_risk_badge(risk)

    if class_names and probabilities:
        show_probability_table(class_names, probabilities)
    else:
        st.warning("No class probabilities were returned by FastAPI.")

    st.subheader("🔍 Why Flagged?")
    if reasons:
        for reason in reasons:
            st.warning(str(reason))
    else:
        st.info("No rule-based explanation reasons were returned.")

    st.subheader("🔗 URL Analysis")
    if int(url_report.get("url_count", 0)) == 0:
        st.info("No URLs detected in this email.")
    else:
        st.write("URLs detected:", url_report.get("url_count", 0))
        st.write("Highest URL risk:", url_report.get("highest_url_risk", "None"))

        for item in url_report.get("urls", []):
            raw_url = str(item.get("raw_url", "Unknown URL"))
            with st.expander(raw_url):
                st.write("Domain:", item.get("domain", "Unknown"))
                st.write("Risk:", item.get("risk", "Unknown"))
                st.write("Score:", item.get("score", "Unknown"))
                st.write("Reasons:")
                for reason in item.get("reasons", []):
                    st.write(f"- {reason}")

    st.subheader("Recommended Action")
    if risk in {"High", "Critical"}:
        st.error("Do not click links, enter credentials, or reply. Report the email.")
    elif risk == "Medium":
        st.warning("Review the email carefully before clicking links or replying.")
    else:
        st.success("The email appears safe based on the current hybrid model.")

    with st.expander("Model Input Features"):
        st.dataframe(
            features,
            use_container_width=True,
            hide_index=True,
        )

    render_explanation_controls(
        subject=subject,
        body=body,
        key_prefix="prediction_page",
    )

    st.subheader("🔁 Continual Learning Feedback")
    st.markdown(
        """
        Confirm or correct the prediction. Feedback is sent through FastAPI and
        stored in MySQL for review and future model retraining.
        """
    )

    with st.form("continual_learning_feedback_form"):
        feedback_status = st.radio(
            "Was this prediction correct?",
            ["Correct", "Wrong"],
            horizontal=True,
        )

        if feedback_status == "Wrong":
            correct_category = st.selectbox(
                "Select the correct category",
                CLASS_LABELS,
            )
        else:
            correct_category = prediction

        feedback_note = st.text_area(
            "Optional feedback note",
            placeholder=(
                "Example: This should be Business Email Compromise because it "
                "requests an urgent payment."
            ),
            height=100,
        )

        submit_feedback = st.form_submit_button(
            "Save Feedback",
            type="primary",
            use_container_width=True,
        )

    if submit_feedback:
        try:
            with st.spinner("Sending feedback to the FastAPI backend..."):
                feedback_result = call_feedback_api(
                    subject=subject,
                    body=body,
                    feedback_status=feedback_status,
                    correct_category=correct_category,
                    feedback_note=feedback_note,
                )

            st.success(
                feedback_result.get(
                    "message",
                    "Feedback saved successfully.",
                )
            )

            feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
            feedback_col1.metric(
                "Feedback Record ID",
                feedback_result.get("feedback_id", "N/A"),
            )
            feedback_col2.metric(
                "Database",
                feedback_result.get("database", "MySQL"),
            )
            feedback_col3.metric(
                "Total Feedback Records",
                feedback_result.get("total_feedback_records", 0),
            )

        except requests.exceptions.RequestException as error:
            show_request_error(error, "saving feedback")


def page_hybrid_model_result() -> None:
    st.header("🔗 Hybrid Model Result")

    st.markdown(
        """
        The proposed architecture combines:

        **Transformer embeddings + XGBoost probability outputs + structured
        features → MLP fusion classifier**
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Transformer", "DistilBERT Multilingual")
    col2.metric("Fusion Input Size", "785 features")
    col3.metric("Hybrid Test Accuracy", "95.82%")

    hybrid_df = pd.DataFrame(
        {
            "Model": [
                "XGBoost Structured Only",
                "Transformer + XGBoost + MLP Fusion",
            ],
            "Test Accuracy (%)": [80.08, 95.82],
            "Macro F1": [0.819, 0.948],
            "Remarks": [
                "Uses only structured phishing indicators",
                "Combines semantic text embeddings and structured indicators",
            ],
        }
    )

    st.dataframe(
        hybrid_df,
        use_container_width=True,
        hide_index=True,
    )
    st.success(
        "The MLP fusion model improved substantially over the structured-only "
        "XGBoost model."
    )


def page_model_benchmarking() -> None:
    st.header("📊 Final Model Benchmarking Results")

    benchmark_df = get_model_benchmark_df()
    st.dataframe(
        benchmark_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Accuracy Comparison")
    st.bar_chart(
        benchmark_df.set_index("Model")[["Test Accuracy (%)"]]
    )

    st.info(
        "Random Forest achieved the highest benchmark accuracy, while the hybrid "
        "model demonstrates the proposed Transformer–XGBoost–MLP architecture."
    )


def page_robustness_testing() -> None:
    st.header("🧪 Robustness Testing Results")

    summary_df = pd.DataFrame(
        {
            "Model": ["MLP Fusion Hybrid"],
            "Samples": [50],
            "Robustness Accuracy (%)": [72.00],
            "Macro F1": [0.43],
            "Main Strength": [
                "Strong on Sinhala-English, Tamil-English, and obfuscated URLs"
            ],
            "Main Weakness": [
                "Weak on AI-generated phishing subtype and BEC classification"
            ],
        }
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The robustness test used difficult code-mixed, AI-generated, obfuscated "
        "URL, Business Email Compromise, spam, and legitimate multilingual samples."
    )

    if ROBUSTNESS_RESULTS_PATH.exists():
        st.subheader("Detailed Robustness Results")
        robustness_df = pd.read_csv(
            ROBUSTNESS_RESULTS_PATH,
            low_memory=False,
        )

        preferred_columns = [
            "attack_type",
            "category",
            "prediction",
            "confidence",
            "correct",
        ]
        available_columns = [
            column
            for column in preferred_columns
            if column in robustness_df.columns
        ]

        st.dataframe(
            robustness_df[available_columns],
            use_container_width=True,
            hide_index=True,
        )

        if "correct" in robustness_df.columns:
            st.subheader("Correct vs Incorrect")
            st.bar_chart(robustness_df["correct"].value_counts())
    else:
        st.warning(
            "Detailed robustness result file not found. Run the robustness testing "
            "script first."
        )


def page_explainability() -> None:
    st.header("🔍 Explainable AI")

    st.markdown(
        """
        This page provides two complementary local explanations:

        - **LIME:** word-level influences for the complete hybrid prediction.
        - **SHAP:** structured-feature contributions for the XGBoost branch.

        Analyse an email on the **Email Prediction** page first, then generate its
        explanation here or directly below the prediction result.
        """
    )

    result_data = st.session_state.get("last_analysis_result")

    if result_data:
        subject = str(result_data.get("subject", ""))
        body = str(result_data.get("body", ""))
        prediction = str(result_data.get("prediction", "Unknown"))
        confidence = float(result_data.get("confidence", 0.0))

        result_col1, result_col2 = st.columns(2)
        result_col1.metric(
            "Current Prediction",
            prediction.replace("_", " "),
        )
        result_col2.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

        with st.expander("Email being explained", expanded=False):
            st.write("**Subject:**", subject or "(empty)")
            st.write("**Body:**", body or "(empty)")

        render_explanation_controls(
            subject=subject,
            body=body,
            key_prefix="explainability_page",
        )
    else:
        st.info("Analyse an email before generating LIME and SHAP explanations.")

    with st.expander("Historical Explainability Outputs", expanded=False):
        if XGB_IMPORTANCE_PATH.exists():
            st.markdown("#### Existing XGBoost Global Feature Importance")
            importance_df = pd.read_csv(XGB_IMPORTANCE_PATH)
            st.dataframe(
                importance_df.head(15),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No saved XGBoost global feature-importance file was found.")

        if EXPLAINABILITY_RESULTS_PATH.exists():
            st.markdown("#### Existing Occlusion/Rule-Based Explanation Results")
            explanation_df = pd.read_csv(
                EXPLAINABILITY_RESULTS_PATH,
                low_memory=False,
            )
            st.dataframe(
                explanation_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No saved historical explainability-results file was found.")


def page_continual_learning_feedback() -> None:
    st.header("🔁 Continual Learning Feedback Management")

    st.markdown(
        """
        This page retrieves feedback records through **GET /feedback**.

        **Streamlit → FastAPI → SQLAlchemy → MySQL**
        """
    )

    refresh_col, status_col = st.columns([1, 3])
    with refresh_col:
        st.button(
            "🔄 Refresh Feedback",
            use_container_width=True,
        )

    try:
        with st.spinner("Loading feedback records from FastAPI..."):
            feedback_result = call_get_feedback_api()
    except requests.exceptions.RequestException as error:
        show_request_error(error, "retrieving feedback")
        return

    if isinstance(feedback_result, list):
        records = feedback_result
        total_records = len(records)
        database_name = "MySQL"
    else:
        records = feedback_result.get("records", [])
        total_records = int(
            feedback_result.get("total_feedback_records", len(records))
        )
        database_name = str(feedback_result.get("database", "MySQL"))

    with status_col:
        st.success(
            f"Connected to FastAPI and retrieved data from {database_name}."
        )

    if total_records == 0 or not records:
        st.info("No feedback records have been collected yet.")
        return

    feedback_df = pd.DataFrame(records).fillna("")

    correct_count = 0
    wrong_count = 0
    if "feedback_status" in feedback_df.columns:
        normalised_status = feedback_df["feedback_status"].astype(str).str.lower()
        correct_count = int((normalised_status == "correct").sum())
        wrong_count = int((normalised_status == "wrong").sum())

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total Feedback Records", total_records)
    metric_col2.metric("Correct Confirmations", correct_count)
    metric_col3.metric("Corrected Predictions", wrong_count)
    metric_col4.metric("Database", database_name)

    st.subheader("📋 Feedback Records")

    preferred_columns = [
        "id",
        "timestamp",
        "created_at",
        "subject",
        "predicted_category",
        "prediction_confidence",
        "feedback_status",
        "correct_category",
        "feedback_note",
        "detected_language",
        "url_count_extracted",
        "has_url",
        "has_defanged_url",
        "suspicious_keyword_count",
        "email_entropy",
    ]
    available_columns = [
        column
        for column in preferred_columns
        if column in feedback_df.columns
    ]

    display_df = feedback_df[available_columns] if available_columns else feedback_df
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("📊 Feedback Summary")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Feedback Status Distribution")
        if "feedback_status" in feedback_df.columns:
            feedback_status_counts = (
                feedback_df["feedback_status"]
                .value_counts()
                .rename_axis("Feedback Status")
                .to_frame("Count")
            )
            st.bar_chart(feedback_status_counts)
        else:
            st.info("Feedback-status information is unavailable.")

    with chart_col2:
        st.markdown("#### Correct Category Distribution")
        if "correct_category" in feedback_df.columns:
            category_counts = (
                feedback_df["correct_category"]
                .value_counts()
                .rename_axis("Correct Category")
                .to_frame("Count")
            )
            st.bar_chart(category_counts)
        else:
            st.info("Correct-category information is unavailable.")

    if "detected_language" in feedback_df.columns:
        st.subheader("🌐 Language Distribution")
        language_counts = (
            feedback_df["detected_language"]
            .value_counts()
            .rename_axis("Detected Language")
            .to_frame("Count")
        )
        st.bar_chart(language_counts)

    st.subheader("⬇️ Export Feedback")
    csv_data = feedback_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download Feedback as CSV",
        data=csv_data,
        file_name="mysql_feedback_records.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("View Complete Feedback Details"):
        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True,
        )


def page_about() -> None:
    st.header("ℹ️ About Project")

    st.markdown(
        """
        ## Explainable Multilingual Phishing Email Detection Prototype

        This research prototype detects multilingual, code-mixed, traditional,
        Business Email Compromise, spam, and AI-generated phishing emails.

        ### Supported Detection Categories

        - Legitimate
        - Spam
        - Traditional Phishing
        - Business Email Compromise
        - AI-Generated Phishing

        ### Main Architecture

        **DistilBERT multilingual embeddings + XGBoost probabilities + structured
        features + MLP fusion classifier**

        ### REST API

        - `GET /health`
        - `POST /predict`
        - `POST /extract-email-screenshot`
        - `POST /explain`
        - `POST /feedback`
        - `GET /feedback`

        ### Screenshot OCR

        - Upload PNG, JPG, JPEG, or WEBP email screenshots
        - Extract English, Sinhala, and Tamil text through FastAPI and Tesseract
        - Review and edit the extracted subject and body before prediction

        ### Email Input and Header Analysis

        - Manual subject and body entry
        - Screenshot OCR for English, Sinhala, and Tamil
        - Original `.eml` parsing with sender, SPF, DKIM, DMARC, mismatch, and attachment reports

        ### Explainability

        - LIME word-level local explanation for the complete hybrid prediction
        - SHAP structured-feature explanation for the XGBoost branch
        - Rule-based indicators and URL-risk analysis

        ### Feedback Loop

        User confirmations and corrected labels are sent through FastAPI and stored
        in MySQL for validation and controlled future retraining.

        ### Final Benchmark Result

        The hybrid model achieved **95.82% test accuracy** and a **0.948 macro
        F1-score**.
        """
    )


# =========================
# Main Application
# =========================
def main() -> None:
    st.markdown(
        """
        <div class="main-title">🛡️ Explainable Multilingual Phishing Email Detection Prototype</div>
        <div class="subtitle">
        Hybrid detection for AI-generated and code-mixed phishing attacks using
        screenshot OCR, Transformer, XGBoost, MLP fusion, LIME, and SHAP.
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Email Prediction",
            "Hybrid Model Result",
            "Model Benchmarking",
            "Robustness Testing",
            "Explainability",
            "Continual Learning Feedback",
            "About Project",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.success("Prediction model: Hybrid MLP Fusion")
    st.sidebar.info(f"FastAPI: {API_BASE_URL}")

    if page == "Email Prediction":
        page_email_prediction()
    elif page == "Hybrid Model Result":
        page_hybrid_model_result()
    elif page == "Model Benchmarking":
        page_model_benchmarking()
    elif page == "Robustness Testing":
        page_robustness_testing()
    elif page == "Explainability":
        page_explainability()
    elif page == "Continual Learning Feedback":
        page_continual_learning_feedback()
    elif page == "About Project":
        page_about()


if __name__ == "__main__":
    main()
