from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = BASE_DIR / "data" / "training"
TRAINING_JSONL = TRAINING_DIR / "extracted_classification_data.jsonl"
TRAIN_JSONL = TRAINING_DIR / "train.jsonl"
VALIDATION_JSONL = TRAINING_DIR / "validation.jsonl"
VALIDATION_PERCENT = 20

_KNOWN_RECORD_HASHES: set[str] | None = None

# Minimal English stopwords for keyword scoring (no extra deps).
_EN_STOPWORDS = frozenset(
    """
    a an the and or but if in on at to for of as is was are were be been being
    from with by than into over after before under again further then once here
    there when where why how all both each few more most other some such no nor
    not only own same so than too very can will just should now this that these
    those it its they them their what which who whom your our his her any
    about above below between through during per via also we you he she
    do does did doing done have has had having may might must shall could would
    """
    .split()
)

# Phrases and tokens that signal transactional vs certification document types
# (aligned with app.services.classify heuristics). Longer phrases first for detection.
_TYPE_SIGNAL_PHRASES: tuple[str, ...] = tuple(
    sorted(
        (
            "proof of payment",
            "payment proof",
            "purchase order",
            "transportation records",
            "transportation record",
            "production records",
            "production record",
            "third party certification",
            "social audit",
            "security audit",
            "invoice",
            "certification",
            "certificate",
            "shipment",
            "transport",
            "production",
            "payment",
            "audit",
        ),
        key=len,
        reverse=True,
    )
)


def extract_main_keywords(text: str, *, max_keywords: int = 25) -> list[str]:
    """
    Main keywords for training: type-indicating phrases plus frequent content words.
    Labels (main/sub bucket) should correlate with these terms for classifier training.
    """
    if not text or not str(text).strip():
        return []

    lowered = text.lower()
    raw_hits = [p for p in _TYPE_SIGNAL_PHRASES if p in lowered]
    # Drop shorter hits fully contained in a longer matched phrase (e.g. "production" inside
    # "production records").
    phrase_hits = [
        h
        for h in raw_hits
        if not any(h != other and h in other for other in raw_hits)
    ]

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", lowered)
    filtered = [w for w in words if w not in _EN_STOPWORDS]
    counts = Counter(filtered)
    frequent = [w for w, _ in counts.most_common(max_keywords * 3)]

    ordered: list[str] = []
    seen: set[str] = set()
    for term in phrase_hits + frequent:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
        if len(ordered) >= max_keywords:
            break
    return ordered


def _record_hash(document: dict) -> str:
    digest_source = "||".join(
        [
            document.get("file_name", ""),
            document.get("main_bucket", {}).get("label", ""),
            document.get("sub_bucket", {}).get("label", ""),
            document.get("preprocess", {}).get("extractor", ""),
            document.get("preprocess", {}).get("text", ""),
        ]
    )
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()


def _load_existing_hashes() -> set[str]:
    hashes: set[str] = set()
    if not TRAINING_JSONL.exists():
        return hashes
    with TRAINING_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                record_hash = payload.get("record_hash")
                if isinstance(record_hash, str) and record_hash:
                    hashes.add(record_hash)
            except json.JSONDecodeError:
                continue
    return hashes


def append_training_record(document: dict) -> dict:
    """
    Append extraction + classification data as JSONL for model training.
    Deduplicates by a stable content hash.
    """
    global _KNOWN_RECORD_HASHES
    if _KNOWN_RECORD_HASHES is None:
        _KNOWN_RECORD_HASHES = _load_existing_hashes()

    record_hash = _record_hash(document)
    if record_hash in _KNOWN_RECORD_HASHES:
        return {
            "status": "skipped_duplicate",
            "record_hash": record_hash,
            "path": str(TRAINING_JSONL),
        }

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    # Ensures validation.jsonl is visible next to train.jsonl even when no row has
    # landed in the validation split yet (20% deterministic by record hash).
    for split_path in (TRAIN_JSONL, VALIDATION_JSONL):
        if not split_path.exists():
            split_path.touch()
    body_text = document.get("preprocess", {}).get("text", "") or ""
    main_keywords = extract_main_keywords(body_text)
    record = {
        "record_hash": record_hash,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "file_name": document.get("file_name"),
        "extractor": document.get("preprocess", {}).get("extractor"),
        "text": body_text,
        "main_keywords": main_keywords,
        "main_bucket": document.get("main_bucket", {}).get("label"),
        "sub_bucket": document.get("sub_bucket", {}).get("label"),
        # Training-centric schema for document category classification
        "label_main": document.get("main_bucket", {}).get("label"),
        "label_sub": document.get("sub_bucket", {}).get("label"),
    }
    with TRAINING_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Deterministic split: same record always goes to same split.
    split_value = int(record_hash[:8], 16) % 100
    split = "validation" if split_value < VALIDATION_PERCENT else "train"
    target_file = VALIDATION_JSONL if split == "validation" else TRAIN_JSONL
    with target_file.open("a", encoding="utf-8") as split_handle:
        split_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _KNOWN_RECORD_HASHES.add(record_hash)

    return {
        "status": "appended",
        "record_hash": record_hash,
        "path": str(TRAINING_JSONL),
        "split": split,
        "train_path": str(TRAIN_JSONL),
        "validation_path": str(VALIDATION_JSONL),
        "validation_percent": VALIDATION_PERCENT,
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def get_training_data_stats() -> dict:
    return {
        "total_samples": _count_lines(TRAINING_JSONL),
        "train_samples": _count_lines(TRAIN_JSONL),
        "validation_samples": _count_lines(VALIDATION_JSONL),
        "validation_percent": VALIDATION_PERCENT,
        "paths": {
            "all": str(TRAINING_JSONL),
            "train": str(TRAIN_JSONL),
            "validation": str(VALIDATION_JSONL),
        },
    }
