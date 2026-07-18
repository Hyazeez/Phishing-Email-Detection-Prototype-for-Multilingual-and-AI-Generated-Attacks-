from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.explainability_service import explainability_service

from backend.database_service import init_database
from backend.feedback_service import (
    CLASS_LABELS,
    read_feedback,
    save_feedback,
)
from pathlib import Path
from backend.model_service import model_service
from backend.eml_service import (
    MAX_EML_SIZE_BYTES,
    SUPPORTED_EML_CONTENT_TYPES,
    SUPPORTED_EML_EXTENSIONS,
    eml_service,
)
from backend.ocr_service import (
    MAX_SCREENSHOT_SIZE_BYTES,
    SUPPORTED_IMAGE_TYPES,
    ocr_service,
)

# =========================================================
# Create FastAPI application
# This must appear before @app.on_event and all endpoints.
# =========================================================
app = FastAPI(
    title="Explainable Multilingual Phishing Detection API",
    description=(
        "FastAPI backend for hybrid phishing email detection using "
        "Transformer embeddings, XGBoost, and MLP fusion."
    ),
    version="1.0.0",
)


# =========================================================
# CORS configuration
# Allows Streamlit to communicate with FastAPI.
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Request schemas
# =========================================================
class PredictionRequest(BaseModel):
    subject: str = Field(
        default="",
        description="Email subject",
    )

    body: str = Field(
        default="",
        description="Email body",
    )


class FeedbackRequest(BaseModel):
    subject: str = Field(
        default="",
        description="Email subject",
    )

    body: str = Field(
        default="",
        description="Email body",
    )

    feedback_status: str = Field(
        description="Correct or Wrong",
    )

    correct_category: str = Field(
        description="Correct email category",
    )

    feedback_note: Optional[str] = Field(
        default="",
        description="Optional user feedback note",
    )

class ExplanationRequest(BaseModel):
    subject: str = ""
    body: str = ""
    num_features: int = 10
    num_samples: int = 100

# =========================================================
# Startup event
# =========================================================
@app.on_event("startup")
def startup_event():
    print("Starting phishing detection backend...")

    try:
        init_database()
        print("MySQL database initialized successfully.")
    except Exception as error:
        print("Database initialization warning:")
        print(error)

    try:
        model_service.load()
        print("Hybrid model loaded successfully.")
    except Exception as error:
        print("Hybrid model loading warning:")
        print(error)


# =========================================================
# Root endpoint
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Explainable Multilingual Phishing Detection API",
        "status": "running",
        "documentation": "/docs",
        "health_endpoint": "/health",
    }


# =========================================================
# Health endpoint
# =========================================================
@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model_loaded": model_service.loaded,
        "model_architecture": (
            "Transformer + XGBoost + MLP Fusion"
        ),
        "database": "MySQL",
    }


# =========================================================
# Prediction endpoint
# =========================================================
@app.post("/predict")
def predict_email(request: PredictionRequest):
    if not request.subject.strip() and not request.body.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide an email subject or body.",
        )

    try:
        result = model_service.predict(
            subject=request.subject,
            body=request.body,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )


# =========================================================
# Feedback endpoint
# =========================================================
@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    valid_feedback_statuses = [
        "Correct",
        "Wrong",
    ]

    if request.feedback_status not in valid_feedback_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                "feedback_status must be either "
                "'Correct' or 'Wrong'."
            ),
        )

    if request.correct_category not in CLASS_LABELS:
        raise HTTPException(
            status_code=400,
            detail=(
                "correct_category must be one of: "
                + ", ".join(CLASS_LABELS)
            ),
        )

    if not request.subject.strip() and not request.body.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide an email subject or body.",
        )

    try:
        prediction_result = model_service.predict(
            subject=request.subject,
            body=request.body,
        )

        feedback_result = save_feedback(
            subject=request.subject,
            body=request.body,
            prediction_result=prediction_result,
            feedback_status=request.feedback_status,
            correct_category=request.correct_category,
            feedback_note=request.feedback_note or "",
        )

        return feedback_result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback saving failed: {str(error)}",
        )

@app.post("/extract-email-screenshot")
async def extract_email_screenshot(
    file: UploadFile = File(...),
):
    content_type = (
        file.content_type
        or "application/octet-stream"
    ).lower()

    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported screenshot type. "
                "Upload PNG, JPG, JPEG or WEBP."
            ),
        )

    try:
        file_bytes = await file.read(
            MAX_SCREENSHOT_SIZE_BYTES + 1
        )

    finally:
        await file.close()

    if len(file_bytes) > MAX_SCREENSHOT_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The screenshot exceeds the 10 MB limit.",
        )

    try:
        return ocr_service.extract_email_text(
            file_bytes=file_bytes,
            filename=file.filename or "email_screenshot",
            content_type=content_type,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        print("Screenshot OCR error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=(
                "The screenshot could not be processed."
            ),
        ) from error
    
@app.post("/extract-eml")
async def extract_eml(
    file: UploadFile = File(...),
):
    filename = file.filename or "uploaded_email.eml"
    extension = Path(filename).suffix.lower()
    content_type = (
        file.content_type
        or "application/octet-stream"
    ).lower()

    if extension not in SUPPORTED_EML_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload an original .eml email file.",
        )

    if content_type not in SUPPORTED_EML_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported .eml content type. Expected message/rfc822, "
                "text/plain, or application/octet-stream."
            ),
        )

    try:
        file_bytes = await file.read(MAX_EML_SIZE_BYTES + 1)
    finally:
        await file.close()

    if len(file_bytes) > MAX_EML_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The .eml file exceeds the 15 MB upload limit.",
        )

    try:
        return eml_service.parse_eml(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print("EML parsing error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="The original email file could not be processed.",
        ) from error

# =========================================================
# Read feedback endpoint
# =========================================================
@app.get("/feedback")
def get_feedback():
    try:
        return read_feedback()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Reading feedback failed: {str(error)}",
        )
    
@app.post("/explain")
def explain_email(request: ExplanationRequest):
    if not request.subject.strip() and not request.body.strip():
        raise HTTPException(
            status_code=400,
            detail="The subject and body cannot both be empty.",
        )

    try:
        return explainability_service.explain(
            subject=request.subject,
            body=request.body,
            num_features=request.num_features,
            num_samples=request.num_samples,
        )

    except Exception as error:
        print("Explainability error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=f"Could not generate explanation: {error}",
        ) from error