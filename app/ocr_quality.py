from __future__ import annotations

import logging
import re
from typing import Any

from langdetect import DetectorFactory, detect_langs

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0


def score_ocr_quality(text: str) -> dict[str, Any]:
    """Score OCR text quality and produce a routing verdict.

    Args:
        text: OCR text to evaluate.

    Returns:
        Metrics dict including `score`, ratios, and verdict metadata.

    Raises:
        None.
    """
    payload = text or ""
    char_count = len(payload)
    non_empty_lines = [ln for ln in payload.splitlines() if ln.strip()]
    line_count = len(non_empty_lines)

    if char_count == 0:
        return {
            "score": 0.0,
            "char_count": 0,
            "alphanumeric_ratio": 0.0,
            "symbol_noise_ratio": 1.0,
            "line_count": 0,
            "language_confidence": "low",
            "verdict": "reject",
            "reason": "OCR returned empty text.",
        }

    alnum_count = sum(1 for ch in payload if ch.isalnum())
    noise_count = len(re.findall(r"[><|{}\[\]$*#+@\\~`]", payload))
    alphanumeric_ratio = alnum_count / char_count
    symbol_noise_ratio = noise_count / char_count

    try:
        lang_probs = detect_langs(payload[:1500])
        top_prob = lang_probs[0].prob if lang_probs else 0.0
    except Exception:  # noqa: BLE001
        top_prob = 0.0

    if top_prob >= 0.80:
        language_confidence = "high"
        lang_points = 1.0
    elif top_prob >= 0.50:
        language_confidence = "medium"
        lang_points = 0.7
    else:
        language_confidence = "low"
        lang_points = 0.35

    len_points = min(1.0, char_count / 800)
    line_points = min(1.0, line_count / 12) if line_count else 0.0
    score = (
        (0.45 * alphanumeric_ratio)
        + (0.20 * (1.0 - min(1.0, symbol_noise_ratio * 6)))
        + (0.20 * lang_points)
        + (0.10 * len_points)
        + (0.05 * line_points)
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.80:
        verdict = "clean"
        reason = "Text is well-formed with low noise."
    elif score >= 0.55:
        verdict = "acceptable"
        reason = "Text is usable for classification."
    elif score >= 0.35:
        verdict = "noisy"
        reason = "Text is noisy but still partially usable."
        logger.warning("OCR quality is noisy: score=%.3f", score)
    else:
        verdict = "reject"
        reason = "Text quality too poor for reliable classification."
        logger.warning("OCR quality rejected: score=%.3f", score)

    return {
        "score": round(score, 4),
        "char_count": char_count,
        "alphanumeric_ratio": round(alphanumeric_ratio, 4),
        "symbol_noise_ratio": round(symbol_noise_ratio, 4),
        "line_count": line_count,
        "language_confidence": language_confidence,
        "verdict": verdict,
        "reason": reason,
    }
