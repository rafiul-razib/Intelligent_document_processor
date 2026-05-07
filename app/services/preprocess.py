from __future__ import annotations

from pathlib import Path

from app.ocr_quality import score_ocr_quality
from app.ocr_service import extract_text_from_pdf


# =========================
# HELPERS
# =========================

def _is_valid_text(text: str) -> bool:
    return len(text.strip()) > 30 and any(c.isalpha() for c in text)


# =========================
# MAIN PREPROCESS
# =========================

def preprocess_pdf(file_path: Path) -> dict:
    """
    OCR preprocessing wrapper.

    Flow:
    - PyMuPDF / EasyOCR extraction
    - Quality scoring
    - Validity check (for Gemini fallback decision)
    """

    text, method, lang_code = extract_text_from_pdf(str(file_path))
    quality = score_ocr_quality(text)

    is_valid = _is_valid_text(text)

    return {
        "status": "completed" if text else "failed",
        "result": {
            "file": file_path.name,
            "text": text,

            # 🔍 Extraction metadata
            "ocr_method": method,  # "native" | "easyocr"
            "ocr_used": method != "native",

            # 🧠 Quality signals
            "ocr_quality_score": quality.get("score", 0),
            "text_length": len(text),
            "is_valid_text": is_valid,

            # 🌍 Language info
            "source_language": lang_code,
            "translated_to_english": False,
            "translation_error": None,

            # 🚀 Downstream hint
            "needs_gemini_fallback": not is_valid,
        },
    }


# =========================
# LEGACY JOB HANDLER
# =========================

def get_preprocess_job(job_id: str) -> dict | None:
    """
    Deprecated: OCR jobs are now synchronous.
    """
    _ = job_id
    return None