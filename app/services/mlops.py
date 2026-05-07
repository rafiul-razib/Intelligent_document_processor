from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "corrections.jsonl"


def feedback_event(document_id: str, correction: str) -> dict:
    """
    Persist correction feedback for retraining loop.
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document_id,
        "correction": correction,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "queued_for_retrain",
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload
