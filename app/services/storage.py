from __future__ import annotations

import shutil
from pathlib import Path


# =========================
# HELPERS
# =========================

def _safe_folder_name(value: str) -> str:
    """
    Sanitize folder names for filesystem safety.
    """
    if not value:
        return "Uncategorized"

    cleaned = "".join(
        char if char.isalnum() or char in (" ", "-", "_") else "_"
        for char in value.strip()
    )

    cleaned = cleaned.replace(" ", "_")

    return cleaned[:100] or "Uncategorized"  # prevent overly long names


def _resolve_duplicate_path(path: Path) -> Path:
    """
    Prevent overwrite by renaming file if it already exists.
    example: file.pdf → file_1.pdf
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


# =========================
# SAVE FILE
# =========================

def save_category_copy(
    source_file: Path,
    root_dir: Path,
    main_bucket: str,
    sub_bucket: str,
) -> Path:
    """
    Copy uploaded file into structured folder:
    <root>/<main_bucket>/<sub_bucket>/<filename>

    Prevents overwrite by renaming duplicates.
    """

    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    main = _safe_folder_name(main_bucket)
    sub = _safe_folder_name(sub_bucket)

    target_dir = root_dir / main / sub
    target_dir.mkdir(parents=True, exist_ok=True)

    destination = target_dir / source_file.name
    destination = _resolve_duplicate_path(destination)

    shutil.copy2(source_file, destination)

    return destination


# =========================
# READ OPERATIONS
# =========================

def get_bucket_folder(root_dir: Path, main_bucket: str, sub_bucket: str) -> Path:
    return root_dir / _safe_folder_name(main_bucket) / _safe_folder_name(sub_bucket)


def list_bucket_pdfs(root_dir: Path, main_bucket: str, sub_bucket: str) -> list[Path]:
    folder = get_bucket_folder(root_dir, main_bucket, sub_bucket)

    if not folder.exists():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ],
        key=lambda p: p.name.lower(),
    )


def count_saved_pdfs(root_dir: Path) -> int:
    if not root_dir.exists():
        return 0

    return sum(
        1 for path in root_dir.rglob("*.pdf")
        if path.is_file()
    )