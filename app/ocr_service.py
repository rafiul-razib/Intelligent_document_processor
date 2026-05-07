from __future__ import annotations

import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import fitz
import numpy as np
from langdetect import detect
from PIL import Image

if TYPE_CHECKING:
    import easyocr


logger = logging.getLogger(__name__)

_reader_cache: dict[tuple[str, ...], "easyocr.Reader"] = {}
_reader_cache_lock = threading.Lock()


def looks_like_garbled_native_text(text: str) -> bool:
    """Heuristically detect broken native PDF text extraction.

    Args:
        text: Native text extracted from PDF text layer.

    Returns:
        True when text appears corrupted/noisy and should trigger OCR fallback.

    Raises:
        None.
    """
    sample = (text or "").strip()[:5000]
    if not sample:
        return True
    total = len(sample)
    control_chars = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\r\t")
    replacement_chars = sample.count("\ufffd")
    language_chars = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff\u0600-\u06ff\u0980-\u09ff]", sample))
    odd_symbol_runs = len(re.findall(r"[^\w\s]{4,}", sample))

    if control_chars / total > 0.01:
        return True
    if replacement_chars > 0:
        return True
    if language_chars / total < 0.25:
        return True
    if odd_symbol_runs >= 6:
        return True
    return False


def inspect_native_text(pdf_path: str) -> dict[str, object]:
    """Inspect native PDF text layer to decide whether OCR fallback is needed.

    Args:
        pdf_path: Path to input PDF.

    Returns:
        Dict containing native text, quality flags, and language code.

    Raises:
        None.
    """
    path = Path(pdf_path)
    native_pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            native_pages.append(page.get_text("text") or "")
    page_count = max(1, len(native_pages))
    native_text = "\n".join(native_pages).strip()
    is_sufficient = len(native_text) >= 50 * page_count
    is_garbled = looks_like_garbled_native_text(native_text)
    cleaned_text = clean_ocr_output(native_text)
    lang_code = detect_language(cleaned_text or native_text)
    return {
        "native_text": native_text,
        "cleaned_text": cleaned_text,
        "page_count": page_count,
        "is_sufficient": is_sufficient,
        "is_garbled": is_garbled,
        "is_native_usable": bool(is_sufficient and not is_garbled),
        "lang_code": lang_code,
    }


def get_reader(langs: list[str]) -> "easyocr.Reader":
    """Return a cached EasyOCR Reader for the language set.

    Args:
        langs: Language codes supported by EasyOCR.

    Returns:
        A cached `easyocr.Reader` instance for the given language tuple.

    Raises:
        RuntimeError: If EasyOCR cannot initialize a reader.
    """
    key = tuple(sorted(langs))
    with _reader_cache_lock:
        cached = _reader_cache.get(key)
        if cached is not None:
            return cached
        logger.info("Initializing EasyOCR reader for languages: %s", list(key))
        try:
            import easyocr

            reader = easyocr.Reader(list(key), gpu=False, verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize EasyOCR reader for {key}") from exc
        _reader_cache[key] = reader
        return reader


def detect_language(text: str) -> str:
    """Detect language from sample text.

    Args:
        text: Raw text sample.

    Returns:
        A language code such as `en`, `ar`, or `zh-cn`.

    Raises:
        None.
    """
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        return detect(sample)
    except Exception:  # noqa: BLE001
        return "en"


def map_lang_to_easyocr(lang_code: str) -> list[str]:
    """Map a langdetect code into EasyOCR language list.

    Args:
        lang_code: Language code produced by langdetect.

    Returns:
        A language list compatible with EasyOCR.

    Raises:
        None.
    """
    normalized = (lang_code or "en").strip().lower()
    mapping = {
        "zh-cn": ["en", "ch_sim"],
        "zh-tw": ["en", "ch_tra"],
        "ar": ["en", "ar"],
        "bn": ["en", "bn"],
        "ko": ["en", "ko"],
        "ja": ["en", "ja"],
        "fr": ["en", "fr"],
        "de": ["en", "de"],
        "es": ["en", "es"],
        "tr": ["en", "tr"],
        "en": ["en"],
    }
    return mapping.get(normalized, ["en"])


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """Preprocess image for OCR with deskew, thresholding, and denoising.

    Args:
        pil_image: Input PIL image.

    Returns:
        Preprocessed PIL image.

    Raises:
        None.
    """
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary_inv > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(
                gray,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    return Image.fromarray(denoised)


def ocr_page(
    pil_image: Image.Image,
    reader: "easyocr.Reader",
    confidence_threshold: float = 0.4,
) -> str:
    """Run EasyOCR on one page and filter by confidence/token length.

    Args:
        pil_image: Preprocessed page image.
        reader: Initialized EasyOCR reader.
        confidence_threshold: Minimum confidence to keep a token.

    Returns:
        Newline-delimited OCR text for that page.

    Raises:
        None.
    """
    try:
        arr = np.array(pil_image)
        results = reader.readtext(arr, detail=1)
    except Exception as exc:  # noqa: BLE001
        logger.error("EasyOCR page read failed: %s", exc)
        return ""

    kept: list[str] = []
    for item in results:
        if len(item) < 3:
            continue
        _, text, confidence = item
        token = str(text or "").strip()
        if confidence >= confidence_threshold and len(token) >= 2:
            kept.append(token)
    logger.debug("OCR page kept %d/%d tokens", len(kept), len(results))
    return "\n".join(kept)


def clean_ocr_output(raw_text: str) -> str:
    """Normalize and denoise OCR output for classifier input.

    Args:
        raw_text: OCR output text before cleaning.

    Returns:
        Cleaned text truncated to 2500 characters.

    Raises:
        None.
    """
    text = unicodedata.normalize("NFKC", raw_text or "")
    text = re.sub(r"(?<!\w)[><|{}\[\]$*#+@\\~`](?!\w)", " ", text)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\u0600-\u06ff\u0980-\u09ff\s]{3,}", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total = len(line)
        if total == 0:
            continue
        alnum_count = sum(1 for ch in line if ch.isalnum())
        ratio = alnum_count / total
        if ratio < 0.30 and total > 8:
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    return cleaned[:2500]


def extract_text_from_pdf(pdf_path: str) -> tuple[str, str, str]:
    """Extract text using native layer first, then OCR fallback if needed.

    Args:
        pdf_path: Path to input PDF file.

    Returns:
        Tuple of `(text, method, lang_code)` where method is `native`, `ocr`, or
        `ocr_degraded`.

    Raises:
        None.
    """
    details = extract_text_from_pdf_details(pdf_path)
    return details["text"], details["method"], details["lang_code"]


def extract_text_from_pdf_details(pdf_path: str) -> dict[str, object]:
    """Extract and clean text without automatic translation.

    Args:
        pdf_path: Path to input PDF file.

    Returns:
        Dict with text, method, lang_code, translated_to_english, and translation_error.

    Raises:
        None.
    """
    path = Path(pdf_path)
    native_probe = inspect_native_text(pdf_path)
    # If PyMuPDF already extracts enough text, skip OCR to avoid unnecessary work.
    if native_probe["is_sufficient"]:
        raw_text = str(native_probe["cleaned_text"])
        lang_code = str(native_probe["lang_code"])
        logger.info(
            "Using native text extraction for %s (lang=%s, sufficient=%s, garbled=%s)",
            path.name,
            lang_code,
            native_probe["is_sufficient"],
            native_probe["is_garbled"],
        )
        return {
            "text": raw_text,
            "method": "native",
            "lang_code": lang_code,
            "translated_to_english": False,
            "translation_error": None,
        }
    if native_probe["is_garbled"]:
        logger.info("Native text appears garbled and insufficient for %s; forcing OCR fallback", path.name)

    lang_code = str(native_probe["lang_code"])
    langs = map_lang_to_easyocr(lang_code)
    reader = get_reader(langs)
    logger.info("Using OCR extraction for %s with langs=%s", path.name, langs)

    ocr_pages: list[str] = []
    page_images = _render_pdf_pages(path)
    for idx, pil_img in enumerate(page_images, start=1):
        try:
            prepared = preprocess_image(pil_img)
            page_text = ocr_page(prepared, reader)
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR preprocessing failed on page %d (%s): %s", idx, path.name, exc)
            page_text = ""
        ocr_pages.append(page_text)
        logger.debug("OCR page %d extracted chars=%d", idx, len(page_text))

    raw_ocr = "\n\n--- PAGE BREAK ---\n\n".join(ocr_pages).strip()
    cleaned = clean_ocr_output(raw_ocr)
    if len(cleaned) < 30:
        logger.warning("OCR degraded for %s; cleaned text too short (%d chars)", path.name, len(cleaned))
        degraded_text = raw_ocr[:2500]
        return {
            "text": degraded_text,
            "method": "ocr_degraded",
            "lang_code": lang_code,
            "translated_to_english": False,
            "translation_error": None,
        }
    return {
        "text": cleaned,
        "method": "ocr",
        "lang_code": lang_code,
        "translated_to_english": False,
        "translation_error": None,
    }


def _render_pdf_pages(pdf_path: Path) -> list[Image.Image]:
    """Render PDF pages to PIL images for OCR.

    Args:
        pdf_path: Input PDF path.

    Returns:
        List of page images rendered at OCR-friendly resolution.

    Raises:
        RuntimeError: If no renderer succeeds.
    """
    try:
        from pdf2image import convert_from_path

        # Primary path: keeps parity with existing OCR settings.
        return convert_from_path(str(pdf_path), dpi=300, fmt="png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdf2image rendering failed for %s (%s). Falling back to PyMuPDF.", pdf_path, exc)

    images: list[Image.Image] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                images.append(img.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to render PDF pages for OCR: {pdf_path}") from exc

    if not images:
        raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")
    return images
