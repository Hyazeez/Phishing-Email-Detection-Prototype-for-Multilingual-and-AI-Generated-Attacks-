from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import os
import re

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    UnidentifiedImageError,
)
import pytesseract
from pytesseract import Output

from backend.feature_engineering import detect_language


MAX_SCREENSHOT_SIZE_BYTES = 10 * 1024 * 1024

SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


class OCRService:
    def __init__(self) -> None:
        self._configure_tesseract()

    @staticmethod
    def _configure_tesseract() -> None:
        """
        Configure the Tesseract executable.

        Priority:
        1. TESSERACT_CMD environment variable
        2. Default Windows installation path
        3. System PATH
        """
        configured_path = os.getenv("TESSERACT_CMD", "").strip()

        if configured_path:
            pytesseract.pytesseract.tesseract_cmd = configured_path
            return

        if os.name == "nt":
            default_windows_path = Path(
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )

            if default_windows_path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(
                    default_windows_path
                )

    @staticmethod
    def _preprocess_image(image: Image.Image) -> Image.Image:
        """
        Improve screenshot readability before OCR.
        """
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        width, height = image.size

        # Increase small screenshot resolution.
        if width < 1800:
            scale_factor = min(2.0, 1800 / max(width, 1))

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)

        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Sharpness(image).enhance(1.4)

        image = image.filter(ImageFilter.SHARPEN)

        return image

    @staticmethod
    def _get_ocr_languages() -> tuple[str, list[str]]:
        """
        Use English, Sinhala and Tamil when installed.

        Falls back to English when local-language models are unavailable.
        """
        try:
            installed_languages = set(
                pytesseract.get_languages(config="")
            )
        except Exception as error:
            raise RuntimeError(
                "Tesseract OCR could not be started. "
                "Check TESSERACT_CMD and the Tesseract installation."
            ) from error

        preferred_languages = [
            language
            for language in ["eng", "sin", "tam"]
            if language in installed_languages
        ]

        warnings: list[str] = []

        if not preferred_languages:
            raise RuntimeError(
                "No supported Tesseract language models were found. "
                "Install at least the English language model."
            )

        missing_languages = [
            language
            for language in ["eng", "sin", "tam"]
            if language not in installed_languages
        ]

        if missing_languages:
            warnings.append(
                "The following OCR language models are unavailable: "
                + ", ".join(missing_languages)
                + ". OCR will use the installed languages only."
            )

        return "+".join(preferred_languages), warnings

    @staticmethod
    def _normalise_extracted_text(text: str) -> str:
        cleaned_lines: list[str] = []

        for line in text.splitlines():
            cleaned_line = re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()

            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _split_subject_and_body(
        extracted_text: str,
    ) -> tuple[str, str, list[str]]:
        """
        Attempt to separate a subject from the email body.

        The extracted values are editable in Streamlit because OCR and
        screenshot layouts may vary.
        """
        warnings: list[str] = []

        lines = [
            line.strip()
            for line in extracted_text.splitlines()
            if line.strip()
        ]

        if not lines:
            return "", "", [
                "No readable text was found in the screenshot."
            ]

        # Prefer an explicit Subject: label.
        for index, line in enumerate(lines):
            subject_match = re.match(
                r"(?i)^(?:subject|sub)\s*[:\-]\s*(.+)$",
                line,
            )

            if subject_match:
                subject = subject_match.group(1).strip()
                body = "\n".join(
                    lines[index + 1:]
                ).strip()

                return subject, body, warnings

        metadata_prefixes = (
            "from:",
            "to:",
            "cc:",
            "bcc:",
            "reply-to:",
            "date:",
            "sent:",
            "received:",
        )

        subject_index: int | None = None

        for index, line in enumerate(lines):
            lower_line = line.lower()

            if lower_line.startswith(metadata_prefixes):
                continue

            if len(line) >= 3:
                subject_index = index
                break

        if subject_index is None:
            return "", extracted_text, [
                "A subject could not be identified automatically."
            ]

        subject = lines[subject_index]
        body = "\n".join(
            lines[subject_index + 1:]
        ).strip()

        warnings.append(
            "The email subject was estimated from the first meaningful "
            "text line. Review and edit it before prediction."
        )

        return subject, body, warnings

    @staticmethod
    def _average_confidence(
        ocr_data: dict[str, Any],
    ) -> float:
        valid_confidences: list[float] = []

        for raw_confidence in ocr_data.get("conf", []):
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue

            if confidence >= 0:
                valid_confidences.append(confidence)

        if not valid_confidences:
            return 0.0

        return round(
            sum(valid_confidences)
            / len(valid_confidences)
            / 100,
            4,
        )

    def extract_email_text(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        if not file_bytes:
            raise ValueError("The uploaded screenshot is empty.")

        if len(file_bytes) > MAX_SCREENSHOT_SIZE_BYTES:
            raise ValueError(
                "The screenshot exceeds the 10 MB upload limit."
            )

        try:
            image = Image.open(BytesIO(file_bytes))
            image.load()
        except UnidentifiedImageError as error:
            raise ValueError(
                "The uploaded file is not a valid supported image."
            ) from error
        except Exception as error:
            raise ValueError(
                "The image could not be opened."
            ) from error

        processed_image = self._preprocess_image(image)

        ocr_languages, warnings = self._get_ocr_languages()

        tesseract_config = "--oem 3 --psm 6"

        extracted_text = pytesseract.image_to_string(
            processed_image,
            lang=ocr_languages,
            config=tesseract_config,
        )

        ocr_data = pytesseract.image_to_data(
            processed_image,
            lang=ocr_languages,
            config=tesseract_config,
            output_type=Output.DICT,
        )

        extracted_text = self._normalise_extracted_text(
            extracted_text
        )

        subject, body, separation_warnings = (
            self._split_subject_and_body(
                extracted_text
            )
        )

        warnings.extend(separation_warnings)

        if not extracted_text:
            warnings.append(
                "OCR did not detect readable text. "
                "Try a clearer or higher-resolution screenshot."
            )

        detected_language = detect_language(
            extracted_text
        )

        return {
            "filename": filename,
            "content_type": content_type,
            "extraction_method": "Tesseract OCR",
            "ocr_languages": ocr_languages.split("+"),
            "ocr_confidence": self._average_confidence(
                ocr_data
            ),
            "detected_language": detected_language,
            "subject": subject,
            "body": body,
            "extracted_text": extracted_text,
            "warnings": warnings,
            "image_width": image.width,
            "image_height": image.height,
        }


ocr_service = OCRService()