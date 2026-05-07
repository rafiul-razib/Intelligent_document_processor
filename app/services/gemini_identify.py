from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from google import genai
try:
    from google.genai import types
except Exception:  # noqa: BLE001
    types = None

# =========================
# LOGGING
# =========================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =========================
# BUCKETS (MUST MATCH YOUR APP)
# =========================

ALLOWED_MAIN_BUCKETS = (
    "Transactional Documents",
    "Certifications",
)

TRANSACTIONAL_SUB_BUCKETS = (
    "Invoice",
    "Purchase Order",
    "Proof of Payment",
    "Production Records",
    "Transportation Records",
)

CERTIFICATION_SUB_BUCKETS = (
    "Social Audit Reports",
    "Security Audit Records",
    "Third party certification",
)

# =========================
# GEMINI CLIENT (NEW SDK)
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set")

MODEL = "gemini-2.0-flash"
_ENV_MODEL = os.getenv("GEMINI_MODEL", "").strip()
FALLBACK_MODELS = tuple(
    dict.fromkeys(
        m
        for m in (
            _ENV_MODEL or MODEL,
            MODEL,
            f"models/{_ENV_MODEL}" if _ENV_MODEL and not _ENV_MODEL.startswith("models/") else "",
            f"models/{MODEL}",
            "gemini-2.5-flash",
            "models/gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "models/gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "models/gemini-2.0-flash-lite",
        )
        if m
    )
)


# =========================
# PUBLIC FUNCTION (FIXED IMPORT NAME)
# =========================

def identify_document_with_gemini(
    *,
    extracted_text: str,
    pdf_path: Path | None,
    gemini_api_key: str,
    force_binary: bool = False,
) -> dict[str, Any]:

    if not gemini_api_key:
        return {"success": False, "error": "Missing GEMINI_API_KEY"}
    masked_runtime_key = (
        f"{gemini_api_key[:4]}...{gemini_api_key[-4:]}"
        if len(gemini_api_key) >= 8
        else "***"
    )
    logger.warning("Gemini runtime key in use: %s", masked_runtime_key)

    prompt = _build_prompt()

    # =========================
    # 1. TEXT MODE (PyMuPDF)
    # =========================
    last_error = ""

    if not force_binary and extracted_text.strip():
        result = _call_gemini(
            gemini_api_key,
            prompt,
            extracted_text[:20000],
            None,
        )
        if result.get("error"):
            last_error = result["error"]

        parsed = _parse(result.get("text"))
        if parsed:
            parsed["success"] = True
            parsed["source"] = "pymupdf"
            parsed["token_usage"] = result.get("token_usage", {})
            parsed["model_used"] = result.get("model_used", "")
            return parsed

    # =========================
    # 2. PDF MODE
    # =========================
    if pdf_path and pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()

        result = _call_gemini(
            gemini_api_key,
            prompt,
            None,
            pdf_bytes,
        )
        if result.get("error"):
            last_error = result["error"]

        parsed = _parse(result.get("text"))
        if parsed:
            parsed["success"] = True
            parsed["source"] = "pdf"
            parsed["token_usage"] = result.get("token_usage", {})
            parsed["model_used"] = result.get("model_used", "")
            return parsed

        # =========================
        # 3. IMAGE MODE (fallback)
        # =========================
        image_bytes = _pdf_to_image(pdf_path)

        result = _call_gemini(
            gemini_api_key,
            prompt,
            None,
            image_bytes,
            is_image=True,
        )
        if result.get("error"):
            last_error = result["error"]

        parsed = _parse(result.get("text"))
        if parsed:
            parsed["success"] = True
            parsed["source"] = "image"
            parsed["token_usage"] = result.get("token_usage", {})
            parsed["model_used"] = result.get("model_used", "")
            return parsed

    return {
        "success": False,
        "error": (
            f"Gemini failed to extract document. Last error: {last_error}"
            if last_error
            else "Gemini failed to extract document"
        ),
    }


# =========================
# PROMPT
# =========================

def _build_prompt() -> str:
    bucket_map = {
        "Transactional Documents": list(TRANSACTIONAL_SUB_BUCKETS),
        "Certifications": list(CERTIFICATION_SUB_BUCKETS),
    }
    return f"""
You are a document classification and key-information extraction AI.
Your task is to classify the document into the single best bucket and extract these fields:
- Doc ID
- Date
- Company name:
  1) From which Company the pdf is being received
  2) To which company the pdf is being given
- Quantity
- Product/Material Name

Return ONLY JSON:

{{
  "main_bucket": "",
  "sub_bucket": "",
  "entities": {{
    "document_id": "",
    "date": "",
    "from_company": "",
    "to_company": "",
    "quantity": "",
    "product_material_name": ""
  }}
}}

Rules:
- main_bucket: {list(ALLOWED_MAIN_BUCKETS)}
- sub_bucket must match the selected main_bucket using this map: {bucket_map}
- Choose the best suited bucket based on document purpose/content.
- If uncertain, still choose the closest valid bucket; do not invent new labels.
- If source document is not in English, translate extracted key information to English.
- Keep translated values concise and faithful to the source meaning.
- For non-English company names, provide an English-pronounceable transliteration
  (romanized form) in `from_company` and `to_company`.
- Map extracted values into `entities` with exact keys:
  - `document_id`
  - `date`
  - `from_company`
  - `to_company`
  - `quantity`
  - `product_material_name`
- If any field is missing/unclear, return empty string for that field.
- Do NOT add explanation
"""


# =========================
# GEMINI CALL
# =========================

def _call_gemini(
    api_key: str,
    prompt: str,
    text: str | None,
    binary: bytes | None,
    is_image: bool = False,
) -> dict[str, Any]:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize Gemini client")
        return {"text": "", "error": f"Gemini client init failed: {exc}", "token_usage": {}, "model_used": ""}

    contents: list[Any] = [prompt]
    if text:
        contents.append(text)

    if binary:
        mime_type = "image/png" if is_image else "application/pdf"
        if types and hasattr(types, "Part"):
            contents.append(types.Part.from_bytes(data=binary, mime_type=mime_type))
        else:
            contents.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(binary).decode("utf-8"),
                    }
                }
            )

    model_candidates = _discover_model_candidates(client)
    last_error = ""
    tried: list[str] = []
    for model_name in model_candidates:
        tried.append(model_name)
        try:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "contents": contents,
            }
            if types and hasattr(types, "GenerateContentConfig"):
                kwargs["config"] = types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            response = client.models.generate_content(**kwargs)
            raw_text = _extract_response_text(response)
            logger.info("Gemini response (%s): %s", model_name, raw_text)
            return {
                "text": raw_text,
                "error": "",
                "token_usage": _extract_usage_metadata(response),
                "model_used": model_name,
            }
        except Exception as e:
            last_error = str(e)
            logger.exception("Gemini API error with model %s", model_name)

    return {
        "text": "",
        "error": f"All Gemini models failed. Tried {tried}. Last error: {last_error}",
        "token_usage": {},
        "model_used": "",
    }


def _discover_model_candidates(client: genai.Client) -> list[str]:
    """Build an ordered candidate list from configured + discovered models."""
    preferred = list(dict.fromkeys([m for m in FALLBACK_MODELS if m]))
    discovered_exact: list[str] = []
    discovered_alias: list[str] = []
    try:
        models = client.models.list()
        for model in models:
            model_name = getattr(model, "name", "") or ""
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods and model_name:
                discovered_exact.append(model_name)
                if model_name.startswith("models/"):
                    discovered_alias.append(model_name.replace("models/", "", 1))
        logger.info(
            "Discovered Gemini models: exact=%s alias=%s",
            discovered_exact,
            discovered_alias,
        )
    except Exception:
        logger.exception("Could not list Gemini models")

    merged = list(dict.fromkeys(discovered_exact + discovered_alias + preferred))
    if not merged:
        logger.error("No Gemini model candidates found from discovery or fallback.")
    return merged or preferred


# =========================
# PARSER
# =========================

def _parse(text: str | None) -> dict | None:
    if not text:
        return None

    try:
        clean = _extract_json_block(text.strip().replace("```json", "").replace("```", ""))
        data = json.loads(clean)

        if not isinstance(data, dict):
            return None

        entities = data.get("entities", {}) if isinstance(data.get("entities", {}), dict) else {}
        normalized_entities = {
            "document_id": entities.get("document_id", entities.get("doc_id", "")),
            "date": entities.get("date", ""),
            "from_company": entities.get("from_company", entities.get("from", "")),
            "to_company": entities.get("to_company", entities.get("to", "")),
            "quantity": entities.get("quantity", ""),
            "product_material_name": entities.get(
                "product_material_name",
                entities.get("product_name", ""),
            ),
        }

        return {
            "main_bucket": data.get("main_bucket", ""),
            "sub_bucket": data.get("sub_bucket", ""),
            "entities": normalized_entities,
            "raw_response_text": text[:1000],
        }

    except Exception:
        return None


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text
    return ""


def _extract_json_block(raw: str) -> str:
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return match.group(0)
    return raw


def _extract_usage_metadata(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    usage_dict: dict[str, Any]
    if isinstance(usage, dict):
        usage_dict = usage
    else:
        usage_dict = {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
            "input_token_count": getattr(usage, "input_token_count", None),
            "output_token_count": getattr(usage, "output_token_count", None),
        }

    def _pick_int(*keys: str) -> int:
        for key in keys:
            value = usage_dict.get(key)
            if value is not None:
                try:
                    return int(value)
                except Exception:  # noqa: BLE001
                    continue
        return 0

    prompt_tokens = _pick_int("prompt_token_count", "input_token_count", "promptTokenCount", "inputTokenCount")
    completion_tokens = _pick_int(
        "candidates_token_count",
        "output_token_count",
        "completion_token_count",
        "candidatesTokenCount",
        "outputTokenCount",
        "completionTokenCount",
    )
    total_tokens = _pick_int("total_token_count", "totalTokenCount")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    logger.info("Gemini raw usage metadata: %s", usage_dict)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }




# =========================
# PDF → IMAGE
# =========================

def _pdf_to_image(pdf_path: Path) -> bytes:
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return b""
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
            return pix.tobytes("png")
    except Exception as e:
        logger.warning("PDF render failed: %s", e)
        return b""