from __future__ import annotations

import sys
import types

from app.ocr_quality import score_ocr_quality
from app.ocr_service import (
    clean_ocr_output,
    detect_language,
    get_reader,
    looks_like_garbled_native_text,
    map_lang_to_easyocr,
)


def test_clean_ocr_output_removes_symbols() -> None:
    raw = "A $+ B >>< C {print D | E"
    cleaned = clean_ocr_output(raw)
    for token in ("$+", ">><", "{print", "|"):
        assert token not in cleaned


def test_clean_ocr_output_preserves_chinese() -> None:
    raw = "发票 Invoice 编号 12345"
    cleaned = clean_ocr_output(raw)
    assert "发票" in cleaned
    assert "编号" in cleaned


def test_clean_ocr_output_preserves_arabic() -> None:
    raw = "فاتورة Invoice رقم 8877"
    cleaned = clean_ocr_output(raw)
    assert "فاتورة" in cleaned


def test_clean_ocr_output_removes_noisy_lines() -> None:
    noisy = "###%%$$@@@----!!!!!!\nInvoice No 123"
    cleaned = clean_ocr_output(noisy)
    assert "Invoice No 123" in cleaned
    assert "###%%$$@@@----!!!!!!" not in cleaned


def test_clean_ocr_output_keeps_short_legitimate_lines() -> None:
    raw = "No.\nQty:\n#####$"
    cleaned = clean_ocr_output(raw)
    assert "No." in cleaned
    assert "Qty:" in cleaned


def test_clean_ocr_output_truncates_at_2500() -> None:
    raw = "A" * 5000
    cleaned = clean_ocr_output(raw)
    assert len(cleaned) == 2500


def test_get_reader_caches(monkeypatch) -> None:  # noqa: ANN001
    class DummyReader:
        def __init__(self, langs, gpu=False, verbose=False):  # noqa: ANN001, FBT002
            self.langs = tuple(langs)
            self.gpu = gpu
            self.verbose = verbose

    fake_easyocr = types.SimpleNamespace(Reader=DummyReader)
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    r1 = get_reader(["en"])
    r2 = get_reader(["en"])
    assert r1 is r2


def test_map_lang_to_easyocr_chinese() -> None:
    mapped = map_lang_to_easyocr("zh-cn")
    assert "ch_sim" in mapped
    assert "en" in mapped


def test_map_lang_to_easyocr_unknown() -> None:
    assert map_lang_to_easyocr("xx-unknown") == ["en"]


def test_score_ocr_quality_clean() -> None:
    raw = """
    Invoice Number: INV-2026-00231
    Supplier: Jiangsu Shenghong Technology Trading Co., Ltd.
    Buyer: Wujiang Hengyang Textile Co., Ltd.
    Date: 2026-05-02
    Product: Recycled Polyester
    Quantity: 27072 KG
    Payment Terms: Bank Transfer
    """
    cleaned = clean_ocr_output(raw)
    quality = score_ocr_quality(cleaned)
    assert quality["verdict"] in {"clean", "acceptable"}


def test_score_ocr_quality_reject() -> None:
    quality = score_ocr_quality("$$$$$>>>>><<<<<#####~~~~~")
    assert quality["verdict"] == "reject"


def test_detect_language_short_input(monkeypatch) -> None:  # noqa: ANN001
    calls = {"count": 0}

    def fake_detect(_: str) -> str:
        calls["count"] += 1
        return "de"

    monkeypatch.setattr("app.ocr_service.detect", fake_detect)
    assert detect_language("1234567890") == "en"
    assert calls["count"] == 0


def test_looks_like_garbled_native_text_detects_corrupted_sample() -> None:
    bad = "q:, c:: r­ e fft!I • \x01 \x07 N2 19222995 1200222110 #### <<<< ****"
    assert looks_like_garbled_native_text(bad) is True
