from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class FeedbackRecord(Base):
    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(String(50), nullable=False)

    subject = Column(Text)
    body = Column(Text)

    predicted_category = Column(String(100))
    prediction_confidence = Column(Float)

    xgboost_prediction = Column(String(100))
    xgboost_confidence = Column(Float)

    feedback_status = Column(String(50))
    correct_category = Column(String(100))
    feedback_note = Column(Text)

    detected_language = Column(String(100))

    url_count_extracted = Column(Integer)
    has_url = Column(Integer)
    has_defanged_url = Column(Integer)
    suspicious_keyword_count = Column(Integer)
    email_entropy = Column(Float)

    clean_email_text = Column(Text)


def init_database():
    Base.metadata.create_all(bind=engine)
    print("MySQL database initialized successfully.")


def insert_feedback_record(record: dict):
    db = SessionLocal()

    try:
        feedback = FeedbackRecord(
            timestamp=record.get("timestamp"),
            subject=record.get("subject"),
            body=record.get("body"),
            predicted_category=record.get("predicted_category"),
            prediction_confidence=record.get("prediction_confidence"),
            xgboost_prediction=record.get("xgboost_prediction"),
            xgboost_confidence=record.get("xgboost_confidence"),
            feedback_status=record.get("feedback_status"),
            correct_category=record.get("correct_category"),
            feedback_note=record.get("feedback_note"),
            detected_language=record.get("detected_language"),
            url_count_extracted=record.get("url_count_extracted"),
            has_url=record.get("has_url"),
            has_defanged_url=record.get("has_defanged_url"),
            suspicious_keyword_count=record.get("suspicious_keyword_count"),
            email_entropy=record.get("email_entropy"),
            clean_email_text=record.get("clean_email_text"),
        )

        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        return feedback.id

    finally:
        db.close()


def get_all_feedback_records():
    db = SessionLocal()

    try:
        records = (
            db.query(FeedbackRecord)
            .order_by(FeedbackRecord.id.desc())
            .all()
        )

        return [
            {
                "id": record.id,
                "timestamp": record.timestamp,
                "subject": record.subject,
                "body": record.body,
                "predicted_category": record.predicted_category,
                "prediction_confidence": record.prediction_confidence,
                "xgboost_prediction": record.xgboost_prediction,
                "xgboost_confidence": record.xgboost_confidence,
                "feedback_status": record.feedback_status,
                "correct_category": record.correct_category,
                "feedback_note": record.feedback_note,
                "detected_language": record.detected_language,
                "url_count_extracted": record.url_count_extracted,
                "has_url": record.has_url,
                "has_defanged_url": record.has_defanged_url,
                "suspicious_keyword_count": record.suspicious_keyword_count,
                "email_entropy": record.email_entropy,
                "clean_email_text": record.clean_email_text,
            }
            for record in records
        ]

    finally:
        db.close()


def get_feedback_count():
    db = SessionLocal()

    try:
        return db.query(FeedbackRecord).count()

    finally:
        db.close()