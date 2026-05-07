from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone

_DOCUMENTS: dict[str, dict] = {}


# =========================
# SAVE
# =========================

def save_document(document: dict) -> dict:
    """
    Store or replace a processed document record keyed by file name.
    Adds timestamp if not present.
    """
    doc = deepcopy(document)

    if "created_at" not in doc:
        doc["created_at"] = datetime.utcnow().isoformat()

    _DOCUMENTS[doc["file_name"]] = doc
    return deepcopy(doc)


# =========================
# GETTERS
# =========================

def get_documents() -> list[dict]:
    docs = [deepcopy(doc) for doc in _DOCUMENTS.values()]
    docs.sort(key=lambda item: item["file_name"].lower(), reverse=True)
    return docs


def get_document(file_name: str) -> dict | None:
    doc = _DOCUMENTS.get(file_name)
    return deepcopy(doc) if doc else None


# =========================
# BUCKET VIEW
# =========================

def get_bucket_store() -> dict[str, dict[str, list[dict]]]:
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for doc in get_documents():
        main_bucket = doc.get("main_bucket", {}).get("label", "Unknown")
        sub_bucket = doc.get("sub_bucket", {}).get("label", "Unknown")

        buckets[main_bucket][sub_bucket].append(doc)

    return {
        main_bucket: dict(sub_buckets)
        for main_bucket, sub_buckets in buckets.items()
    }


# =========================
# SUMMARY
# =========================

def get_summary() -> dict:
    docs = get_documents()

    total_documents = len(docs)
    today_utc = datetime.now(timezone.utc).date()

    # Track sources instead of OCR
    source_counts = defaultdict(int)
    for doc in docs:
        source = doc.get("source", "unknown")
        source_counts[source] += 1

    # No real ML confidence → use fallback metric
    filled_main = sum(1 for doc in docs if doc.get("main_bucket", {}).get("label"))
    accuracy_estimate = round((filled_main / total_documents) * 100, 1) if total_documents else 0
    token_totals = [
        int(
            ((doc.get("ai_identification") or {}).get("token_usage") or {}).get(
                "total_tokens",
                0,
            )
            or 0
        )
        for doc in docs
    ]
    avg_tokens_per_document = (
        round(sum(token_totals) / len(token_totals), 1)
        if token_totals
        else 0
    )
    docs_today: list[dict] = []
    for doc in docs:
        created_at = str(doc.get("created_at") or "").strip()
        if not created_at:
            continue
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            if created_dt.astimezone(timezone.utc).date() == today_utc:
                docs_today.append(doc)
        except Exception:  # noqa: BLE001
            continue
    today_token_totals = [
        int(
            ((doc.get("ai_identification") or {}).get("token_usage") or {}).get(
                "total_tokens",
                0,
            )
            or 0
        )
        for doc in docs_today
    ]
    today_avg_tokens_per_document = (
        round(sum(today_token_totals) / len(today_token_totals), 1)
        if today_token_totals
        else 0
    )

    # Bucket breakdown
    breakdown: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for doc in docs:
        main = doc.get("main_bucket", {}).get("label", "Unknown")
        sub = doc.get("sub_bucket", {}).get("label", "Unknown")
        breakdown[main][sub] += 1

    return {
        "total_documents": total_documents,
        "processed_today": len(docs_today),
        "pending_ocr": avg_tokens_per_document,
        "avg_tokens_per_document": avg_tokens_per_document,
        "today_avg_tokens_per_document": today_avg_tokens_per_document,
        "model_accuracy": accuracy_estimate,
        "source_distribution": dict(source_counts),
        "data_completeness": accuracy_estimate,  # replaces fake confidence
        "breakdown": {
            main_bucket: dict(sub_buckets)
            for main_bucket, sub_buckets in breakdown.items()
        },
    }


# =========================
# COMPANY GROUPING
# =========================

def get_company_groups() -> dict[str, dict[str, list[dict]]]:
    """
    Group 01: From which company the PDF is received.
    Group 02: To which company the PDF is sent.
    """

    from_groups: dict[str, list[dict]] = defaultdict(list)
    to_groups: dict[str, list[dict]] = defaultdict(list)

    for doc in get_documents():
        entities = doc.get("entities", {})

        from_company = (entities.get("from_company") or "").strip() or "Unknown From Company"
        to_company = (entities.get("to_company") or "").strip() or "Unknown To Company"

        from_groups[from_company].append(doc)
        to_groups[to_company].append(doc)

    return {
        "group_01_from_company": dict(from_groups),
        "group_02_to_company": dict(to_groups),
    }