from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import fitz  # PyMuPDF
import easyocr
from google import genai


# =========================
# LOGGING
# =========================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================
# GEMINI CLIENT (NEW SDK)
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# EASY OCR
# =========================

reader = easyocr.Reader(["en"], gpu=False)


# =========================
# PROMPT
# =========================

PROMPT = """
Extract structured information from the document.

Return ONLY valid JSON:

{
  "document_type": "",
  "main_bucket": "",
  "sub_bucket": "",
  "invoice_number": "",
  "date": "",
  "company": "",
  "total_amount": "",
  "confidence": 0.0
}

Rules:
- If unknown → empty string
- Confidence between 0 and 1
- No explanations
"""


# =========================
# TEXT EXTRACTION
# =========================

def extract_text_pymupdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:
        logger.warning("PyMuPDF failed: %s", e)
        return ""


def extract_text_easyocr(file_path: str) -> str:
    try:
        results = reader.readtext(file_path)
        return " ".join([r[1] for r in results]).strip()
    except Exception as e:
        logger.warning("EasyOCR failed: %s", e)
        return ""


def is_valid_text(text: str, min_len: int = 30) -> bool:
    return len(text) > min_len and any(c.isalpha() for c in text)


# =========================
# GEMINI CALLS (FIXED)
# =========================

def call_gemini_text(text: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[PROMPT, text[:4000]],
        )
        logger.info("Gemini text response received")
        return response.text or ""
    except Exception as e:
        logger.error("Gemini text call failed: %s", e)
        return ""


def call_gemini_file(file_path: str) -> str:
    try:
        uploaded_file = client.files.upload(file=file_path)

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[uploaded_file, PROMPT],
        )

        logger.info("Gemini file response received")
        return response.text or ""
    except Exception as e:
        logger.error("Gemini file call failed: %s", e)
        return ""


# =========================
# JSON CLEANING
# =========================

def clean_json(text: str) -> str:
    return (
        text.replace("```json", "")
        .replace("```", "")
        .strip()
    )


def parse_response(response_text: str) -> Dict[str, Any]:
    try:
        cleaned = clean_json(response_text)
        return json.loads(cleaned)
    except Exception as e:
        logger.warning("JSON parse failed: %s", e)
        return {}


# =========================
# MAIN PIPELINE
# =========================

def classify_document(file_path: str) -> Dict[str, Any]:
    file_path = str(file_path)

    # ---- Step 1: PyMuPDF ----
    text_pymupdf = extract_text_pymupdf(file_path)

    if is_valid_text(text_pymupdf):
        logger.info("Using PyMuPDF text")
        result = parse_response(call_gemini_text(text_pymupdf))
        result["source"] = "pymupdf"
        return result

    # ---- Step 2: EasyOCR ----
    text_ocr = extract_text_easyocr(file_path)

    if is_valid_text(text_ocr):
        logger.info("Using EasyOCR text")
        result = parse_response(call_gemini_text(text_ocr))
        result["source"] = "easyocr"
        return result

    # ---- Step 3: Gemini File ----
    logger.info("Using Gemini file fallback")
    result = parse_response(call_gemini_file(file_path))
    result["source"] = "gemini_file"
    return result


# =========================
# SAVE FILE
# =========================

def save_document(file_path: str, main_bucket: str, sub_bucket: str) -> str:
    base_dir = Path("storage") / main_bucket / sub_bucket
    base_dir.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)
    dest = base_dir / src.name

    src.rename(dest)

    logger.info("File saved to %s", dest)

    return str(dest)