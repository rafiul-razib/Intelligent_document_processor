from collections import defaultdict
from typing import Dict, List


# =========================
# INTERNAL STORES
# =========================

_LOCAL_GROUP_STORE: Dict[str, set[str]] = defaultdict(set)
_BUCKET_STORE: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))


# =========================
# HELPERS
# =========================

def _normalize_key(value: str | None) -> str:
    """Normalize company/bucket names"""
    if not value:
        return "unknown"
    return value.strip().lower()


# =========================
# GROUPING
# =========================

def group_document(company: str, file_name: str) -> List[str]:
    """
    Group documents by company (from/to).
    Prevents duplicates and normalizes keys.
    """
    key = _normalize_key(company)

    _LOCAL_GROUP_STORE[key].add(file_name)

    return list(_LOCAL_GROUP_STORE[key])


# =========================
# BUCKET STORAGE
# =========================

def store_in_bucket(main_bucket: str, sub_bucket: str, file_name: str) -> List[str]:
    """
    Store file under main/sub bucket.
    Prevents duplicates.
    """
    main = _normalize_key(main_bucket)
    sub = _normalize_key(sub_bucket)

    _BUCKET_STORE[main][sub].add(file_name)

    return list(_BUCKET_STORE[main][sub])


# =========================
# UPDATE / MOVE DOCUMENT
# =========================

def move_document(
    file_name: str,
    old_main: str,
    old_sub: str,
    new_main: str,
    new_sub: str,
) -> None:
    """
    Move document between buckets (for manual correction).
    """

    old_main_n = _normalize_key(old_main)
    old_sub_n = _normalize_key(old_sub)

    new_main_n = _normalize_key(new_main)
    new_sub_n = _normalize_key(new_sub)

    # Remove from old bucket
    if file_name in _BUCKET_STORE[old_main_n][old_sub_n]:
        _BUCKET_STORE[old_main_n][old_sub_n].remove(file_name)

    # Add to new bucket
    _BUCKET_STORE[new_main_n][new_sub_n].add(file_name)


# =========================
# READ STORE
# =========================

def get_bucket_store() -> dict[str, dict[str, list[str]]]:
    """
    Read-only snapshot for UI.
    """
    return {
        main_bucket: {
            sub_bucket: sorted(list(files))
            for sub_bucket, files in sub_buckets.items()
        }
        for main_bucket, sub_buckets in _BUCKET_STORE.items()
    }


# =========================
# OPTIONAL: RESET (DEV ONLY)
# =========================

def reset_store() -> None:
    """Clear all in-memory data (useful for testing)."""
    _LOCAL_GROUP_STORE.clear()
    _BUCKET_STORE.clear()