import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = "dev-key-change-me"
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB

    UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
    PROCESSED_FOLDER = BASE_DIR / "data" / "processed"
    _default_storage_root = (
        "C:/PDF_Classifier_Categories"
        if os.name == "nt"
        else "/tmp/PDF_Classifier_Categories"
    )
    CATEGORY_STORAGE_ROOT = Path(
        os.getenv("CATEGORY_STORAGE_ROOT", _default_storage_root)
    )
    ALLOWED_EXTENSIONS = {"pdf"}
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
