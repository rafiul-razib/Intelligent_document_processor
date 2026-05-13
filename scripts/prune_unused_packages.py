"""Uninstall packages removed from requirements.txt (HF / legacy Google / etc.).

Run with the same interpreter you use for the app:

  python scripts/prune_unused_packages.py

Safe to re-run; pip skips anything not installed.
"""
from __future__ import annotations

import subprocess
import sys

# Former pins — not used by the Flask app (imports + EasyOCR + google-genai only).
# Do not list setuptools, pip, wheel, or packages still required by torch/Flask/google-genai.
REMOVED = [
    "accelerate",
    "aiohappyeyeballs",
    "aiohttp",
    "aiosignal",
    "annotated-doc",
    "attrs",
    "beautifulsoup4",
    "datasets",
    "deep-translator",
    "dill",
    "frozenlist",
    "huggingface-hub",
    "hf-xet",
    "multiprocess",
    "pandas",
    "pyarrow",
    "tokenizers",
    "transformers",
    "safetensors",
    "google-generativeai",
    "google-api-python-client",
    "google-ai-generativelanguage",
    "google-api-core",
    "googleapis-common-protos",
    "grpcio",
    "grpcio-status",
    "proto-plus",
    "uritemplate",
    "httplib2",
    "google-auth-httplib2",
    "rich",
    "typer",
    "shellingham",
    "markdown-it-py",
    "mdurl",
    "Pygments",
    "regex",
    "tzdata",
    "wrapt",
    "xxhash",
    "yarl",
    "multidict",
    "propcache",
    "soupsieve",
    "smart-open",
    "psutil",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *REMOVED]
    print("Uninstalling", len(REMOVED), "optional/legacy packages...")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
