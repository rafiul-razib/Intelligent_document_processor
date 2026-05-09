from pathlib import Path
from io import BytesIO
import logging
import threading
import uuid
import zipfile
from datetime import datetime

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from app.services.gemini_identify import identify_document_with_gemini
from app.services.documents import (
    delete_document,
    get_bucket_store,
    get_company_groups,
    get_documents,
    get_summary,
    save_document,
)
from app.services.storage import (
    get_bucket_folder,
    list_bucket_pdfs,
    save_category_copy,
)
from app.ocr_quality import score_ocr_quality
from app.ocr_service import (
    clean_ocr_output,
    extract_text_from_pdf_details,
)

main_bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

_DOC_STATUS_LOCK = threading.Lock()
_DOC_STATUS: dict[str, dict] = {}


# =========================
# HELPERS
# =========================

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in current_app.config["ALLOWED_EXTENSIONS"]
    )


def _needs_ai_identification(entities: dict) -> bool:
    required_fields = (
        "document_id",
        "date",
        "from_company",
        "to_company",
        "quantity",
        "product_material_name",
    )
    return not any(str(entities.get(f, "")).strip() for f in required_fields)


def _looks_like_garbled_ocr_text(text: str) -> bool:
    sample = (text or "").strip()[:5000]
    if not sample:
        return True
    total = len(sample)
    if total == 0:
        return True
    alpha_num = sum(1 for ch in sample if ch.isalnum())
    weird = sum(
        1
        for ch in sample
        if (ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 0xFFFD
    )
    long_symbol_runs = 0
    run = 0
    for ch in sample:
        if not ch.isalnum() and not ch.isspace():
            run += 1
            if run >= 4:
                long_symbol_runs += 1
                run = 0
        else:
            run = 0

    alpha_num_ratio = alpha_num / total
    weird_ratio = weird / total
    return (
        alpha_num_ratio < 0.35
        or weird_ratio > 0.01
        or long_symbol_runs >= 4
    )


def _set_doc_status(doc_id: str, **updates) -> None:
    with _DOC_STATUS_LOCK:
        _DOC_STATUS.setdefault(doc_id, {}).update(updates)


def _get_doc_status(doc_id: str) -> dict | None:
    with _DOC_STATUS_LOCK:
        return _DOC_STATUS.get(doc_id)


# =========================
# GEMINI PIPELINE
# =========================

def _build_pipeline_result(upload_path: Path, preprocess_result: dict) -> dict:
    gemini_api_key = str(current_app.config.get("GEMINI_API_KEY", "")).strip()

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    ocr_method = str(preprocess_result.get("ocr_method", ""))
    extracted_text = str(preprocess_result.get("text", "")).strip()
    quality_score = float(preprocess_result.get("ocr_quality_score", 0.0))
    garbled_ocr = _looks_like_garbled_ocr_text(extracted_text)
    # If OCR/native extraction is too sparse or noisy, send original PDF to Gemini.
    force_binary = (
        ocr_method == "ocr_degraded"
        or not extracted_text
        or len(extracted_text) < 120
        or quality_score < 0.90
        or garbled_ocr
    )
    logger.info(
        "Original doc sent to Gemini: %s (method=%s, quality=%.3f, garbled=%s, text_len=%d)",
        force_binary,
        ocr_method,
        quality_score,
        garbled_ocr,
        len(extracted_text),
    )
    gemini = identify_document_with_gemini(
        extracted_text=extracted_text,
        pdf_path=upload_path if upload_path.exists() else None,
        gemini_api_key=gemini_api_key,
        force_binary=force_binary,
    )
    if not gemini.get("success"):
        raise RuntimeError(f"Gemini identification failed: {gemini.get('error', 'unknown error')}")

    entities = gemini.get("entities", {})
    logger.info(
        "Gemini token usage for %s: %s",
        upload_path.name,
        gemini.get("token_usage", {}),
    )
    needs_ai_identification = _needs_ai_identification(entities)
    ai_identified = not needs_ai_identification
    confidence = 0.9 if ai_identified else 0.55

    return {
        "file_name": upload_path.name,
        "file_url": f"/uploads/{upload_path.name}",
        "preview_url": f"/uploads/{upload_path.name}/preview.png",
        "preprocess": preprocess_result,

        "main_bucket": {
            "label": gemini.get("main_bucket", ""),
            "confidence": confidence,
        },
        "sub_bucket": {
            "label": gemini.get("sub_bucket", ""),
            "confidence": 0.88 if ai_identified else 0.5,
        },

        "entities": entities,
        "needs_ai_identification": needs_ai_identification,
        "ai_identified": ai_identified,

        "ai_identification": {
            "model": str(gemini.get("model_used") or f"gemini:{gemini.get('source', 'text')}"),
            "prompt": "Extract structured fields and classify document.",
            "gemini_response_preview": gemini.get("raw_response_text", ""),
            "token_usage": gemini.get("token_usage", {}),
            "sent_original_doc_to_gemini": force_binary,
        },

        "is_finalized": False,
    }


# =========================
# SAVE FINAL RESULT
# =========================

def _persist_result(result: dict) -> dict:
    finalized = dict(result)

    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / finalized["file_name"]

    category_copy_path = save_category_copy(
        source_file=upload_path,
        root_dir=Path(current_app.config["CATEGORY_STORAGE_ROOT"]),
        main_bucket=finalized["main_bucket"]["label"],
        sub_bucket=finalized["sub_bucket"]["label"],
    )

    finalized["category_storage_path"] = str(category_copy_path)
    finalized["is_finalized"] = True
    finalized["download_url"] = (
        f"/api/download/classified/{finalized['file_name']}"
        f"?main_bucket={finalized['main_bucket']['label']}"
        f"&sub_bucket={finalized['sub_bucket']['label']}"
    )

    return save_document(finalized)


# =========================
# ROUTES
# =========================

@main_bp.get("/")
def home():
    return render_template("index.html")


@main_bp.post("/upload")
def upload_pdf():
    file = request.files.get("pdf_file")

    if not file or not file.filename:
        flash("Please select a PDF file.", "error")
        return redirect(url_for("main.home"))

    if not _allowed_file(file.filename):
        flash("Only PDF files allowed.", "error")
        return redirect(url_for("main.home"))

    filename = secure_filename(file.filename)
    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    ocr = extract_text_from_pdf_details(str(upload_path))

    preprocess_result = {
        "file": upload_path.name,
        "text": clean_ocr_output(ocr["text"]),
        "ocr_method": ocr["method"],
        "ocr_quality_score": score_ocr_quality(ocr["text"])["score"],
    }

    result = _persist_result(
        _build_pipeline_result(upload_path, preprocess_result)
    )

    return render_template("result.html", result=result)


# =========================
# ASYNC API
# =========================

@main_bp.post("/api/upload")
def api_upload():
    file = request.files.get("pdf_file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    doc_id = str(uuid.uuid4())
    app_obj = current_app._get_current_object()

    _set_doc_status(doc_id, status="processing", file_name=filename)

    def worker():
        with app_obj.app_context():
            try:
                ocr = extract_text_from_pdf_details(str(upload_path))

                preprocess_result = {
                    "file": filename,
                    "text": clean_ocr_output(str(ocr["text"])),
                    "ocr_method": ocr["method"],
                    "ocr_quality_score": score_ocr_quality(ocr["text"])["score"],
                }

                result = _build_pipeline_result(upload_path, preprocess_result)

                _set_doc_status(doc_id, status="done", result=result)

            except Exception as e:
                logger.exception(e)
                _set_doc_status(doc_id, status="failed", error=str(e))

    threading.Thread(target=worker, daemon=True).start()

    return jsonify({"doc_id": doc_id, "status": "processing"})


@main_bp.get("/documents/<doc_id>/status")
def doc_status(doc_id):
    data = _get_doc_status(doc_id)
    if not data:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


# =========================
# FILE ACCESS
# =========================

@main_bp.get("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@main_bp.get("/uploads/<filename>/preview.png")
def preview_file(filename):
    import fitz

    file_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename

    with fitz.open(file_path) as doc:
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        return send_file(BytesIO(pix.tobytes("png")), mimetype="image/png")


@main_bp.delete("/api/documents/<path:file_name>")
def api_delete_document(file_name):
    safe_name = secure_filename(file_name)
    if not safe_name:
        return jsonify({"status": "error", "message": "Invalid file name"}), 400

    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / safe_name
    deleted_file = False
    if upload_path.exists() and upload_path.is_file():
        try:
            upload_path.unlink()
            deleted_file = True
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": str(exc)}), 500

    deleted_record = delete_document(safe_name)
    if not deleted_file and not deleted_record:
        return jsonify({"status": "error", "message": "Document not found"}), 404

    return jsonify({"status": "ok", "file_name": safe_name})


@main_bp.get("/api/download/classified/<path:file_name>")
def api_download_classified_file(file_name):
    safe_name = secure_filename(file_name)
    if not safe_name:
        return jsonify({"status": "error", "message": "Invalid file name"}), 400

    main_bucket = str(request.args.get("main_bucket", "")).strip()
    sub_bucket = str(request.args.get("sub_bucket", "")).strip()

    if main_bucket and sub_bucket:
        root_dir = Path(current_app.config["CATEGORY_STORAGE_ROOT"])
        bucket_folder = get_bucket_folder(
            root_dir=root_dir,
            main_bucket=main_bucket,
            sub_bucket=sub_bucket,
        )
        file_path = bucket_folder / safe_name
        if file_path.exists() and file_path.is_file():
            safe_main = "".join(c if c.isalnum() else "_" for c in main_bucket)
            safe_sub = "".join(c if c.isalnum() else "_" for c in sub_bucket)
            download_name = f"{safe_main}__{safe_sub}__{safe_name}"
            return send_file(file_path, as_attachment=True, download_name=download_name)

    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / safe_name
    if upload_path.exists() and upload_path.is_file():
        return send_file(upload_path, as_attachment=True, download_name=safe_name)

    return jsonify({"status": "error", "message": "File not found"}), 404


@main_bp.get("/bucket-files")
def bucket_files():
    main_bucket = str(request.args.get("main_bucket", "")).strip()
    sub_bucket = str(request.args.get("sub_bucket", "")).strip()
    if not main_bucket or not sub_bucket:
        return "main_bucket and sub_bucket are required", 400

    root_dir = Path(current_app.config["CATEGORY_STORAGE_ROOT"])
    folder = get_bucket_folder(
        root_dir=root_dir,
        main_bucket=main_bucket,
        sub_bucket=sub_bucket,
    )
    pdf_paths = list_bucket_pdfs(
        root_dir=root_dir,
        main_bucket=main_bucket,
        sub_bucket=sub_bucket,
    )
    return render_template(
        "bucket_files.html",
        main_bucket=main_bucket,
        sub_bucket=sub_bucket,
        folder_path=str(folder),
        pdf_files=[path.name for path in pdf_paths],
        export_bucket_url=(
            f"/api/export/bucket?main_bucket={main_bucket}&sub_bucket={sub_bucket}"
        ),
        export_all_url="/api/export/all",
    )


@main_bp.get("/api/export/bucket")
def api_export_bucket_zip():
    main_bucket = str(request.args.get("main_bucket", "")).strip()
    sub_bucket = str(request.args.get("sub_bucket", "")).strip()
    if not main_bucket or not sub_bucket:
        return jsonify({"status": "error", "message": "main_bucket and sub_bucket are required"}), 400

    root_dir = Path(current_app.config["CATEGORY_STORAGE_ROOT"])
    pdf_paths = list_bucket_pdfs(
        root_dir=root_dir,
        main_bucket=main_bucket,
        sub_bucket=sub_bucket,
    )
    if not pdf_paths:
        return jsonify({"status": "error", "message": "No PDFs found for this bucket"}), 404

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf_path in pdf_paths:
            if pdf_path.exists() and pdf_path.is_file():
                zf.write(pdf_path, arcname=pdf_path.name)
    buffer.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_main = "".join(c if c.isalnum() else "_" for c in main_bucket)
    safe_sub = "".join(c if c.isalnum() else "_" for c in sub_bucket)
    download_name = f"{safe_main}_{safe_sub}_{timestamp}.zip"
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@main_bp.get("/api/export/all")
def api_export_all_zip():
    root_dir = Path(current_app.config["CATEGORY_STORAGE_ROOT"])
    if not root_dir.exists():
        return jsonify({"status": "error", "message": "Classified storage folder not found"}), 404

    pdf_paths = [path for path in root_dir.rglob("*.pdf") if path.is_file()]
    if not pdf_paths:
        return jsonify({"status": "error", "message": "No classified PDFs found"}), 404

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf_path in pdf_paths:
            try:
                arcname = str(pdf_path.relative_to(root_dir))
            except ValueError:
                arcname = pdf_path.name
            zf.write(pdf_path, arcname=arcname)
    buffer.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    download_name = f"classified_pdfs_{timestamp}.zip"
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


# =========================
# STATE API
# =========================

@main_bp.get("/api/state")
def api_state():
    return {
        "documents": get_documents(),
        "bucket_store": get_bucket_store(),
        "company_groups": get_company_groups(),
        "summary": get_summary(),
    }


@main_bp.get("/api/gemini-health")
def api_gemini_health():
    gemini_api_key = str(current_app.config.get("GEMINI_API_KEY", "")).strip()
    if not gemini_api_key:
        return jsonify(
            {
                "status": "error",
                "connected": False,
                "message": "GEMINI_API_KEY is missing",
            }
        ), 400
    probe = identify_document_with_gemini(
        extracted_text="Invoice INV-1001 from ABC Textiles to XYZ Traders, quantity 120, date 2026-05-05.",
        pdf_path=None,
        gemini_api_key=gemini_api_key,
        force_binary=False,
    )
    connected = bool(probe.get("success"))
    return jsonify(
        {
            "status": "ok" if connected else "error",
            "connected": connected,
            "message": "Gemini connected" if connected else probe.get("error", "Gemini call failed"),
            "response": probe,
        }
    ), (200 if connected else 502)


@main_bp.post("/api/ai-identify")
def api_ai_identify():
    payload = request.get_json(silent=True) or {}
    result = payload.get("result")
    if not isinstance(result, dict):
        return jsonify({"status": "error", "message": "result payload is required"}), 400
    file_name = str(result.get("file_name") or "").strip()
    if not file_name:
        return jsonify({"status": "error", "message": "file_name is required"}), 400

    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / file_name
    if not upload_path.exists():
        return jsonify({"status": "error", "message": "Original file not found"}), 404

    preprocess_result = dict(result.get("preprocess") or {})
    preprocess_result.setdefault("text", "")
    preprocess_result.setdefault("ocr_method", "ocr")
    preprocess_result.setdefault("ocr_quality_score", 0.0)

    try:
        refreshed = _build_pipeline_result(upload_path, preprocess_result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500
    return jsonify({"status": "ok", "result": refreshed})


@main_bp.post("/api/finalize")
def api_finalize():
    payload = request.get_json(silent=True) or {}
    result = payload.get("result")
    correction = str(payload.get("correction") or "").strip()
    manual_entities = payload.get("manual_entities") or {}
    if not isinstance(result, dict):
        return jsonify({"status": "error", "message": "result payload is required"}), 400

    if correction:
        parts = [part.strip() for part in correction.split("/", 1)]
        if len(parts) == 2:
            result["main_bucket"] = {
                "label": parts[0],
                "confidence": result.get("main_bucket", {}).get("confidence", 0.9),
            }
            result["sub_bucket"] = {
                "label": parts[1],
                "confidence": result.get("sub_bucket", {}).get("confidence", 0.88),
            }

    if isinstance(manual_entities, dict) and manual_entities:
        entities = dict(result.get("entities") or {})
        for key in (
            "document_id",
            "date",
            "from_company",
            "to_company",
            "quantity",
            "product_material_name",
        ):
            value = manual_entities.get(key)
            if isinstance(value, str):
                entities[key] = value.strip()
        result["entities"] = entities

    entities_for_check = dict(result.get("entities") or {})
    required_keys = (
        "document_id",
        "date",
        "from_company",
        "to_company",
        "quantity",
        "product_material_name",
    )
    has_missing_key_info = any(
        not str(entities_for_check.get(key, "")).strip()
        for key in required_keys
    )
    if has_missing_key_info and not correction:
        return jsonify(
            {
                "status": "error",
                "message": "Need Manual Checking",
            }
        ), 400

    try:
        saved = _persist_result(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500
    return jsonify({"status": "ok", "result": saved})