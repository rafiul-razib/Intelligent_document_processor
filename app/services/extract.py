from __future__ import annotations

import re


# =========================
# CORE REGEX HELPER
# =========================

def _extract_with_regex(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


# =========================
# CLEANING HELPERS
# =========================

def _clean_line(value: str) -> str:
    """Remove excessive noise from extracted lines"""
    if not value:
        return ""

    # Remove long trailing junk
    value = value.split("  ")[0]  # double-space cutoff
    value = value.split(",")[0] if len(value) > 80 else value

    return value.strip()


def _extract_companies_fallback(text: str) -> list[str]:
    """
    Fallback: detect uppercase company-like lines
    """
    lines = text.splitlines()
    companies = []

    for line in lines:
        line = line.strip()

        # Heuristic: uppercase + length filter
        if (
            5 < len(line) < 80
            and line.isupper()
            and any(word in line for word in ["LTD", "LIMITED", "INC", "CORP", "COMPANY"])
        ):
            companies.append(line)

    return list(set(companies))


# =========================
# MAIN FUNCTION
# =========================

def extract_key_info(text: str) -> dict:
    """
    Extract key fields using regex + fallback heuristics
    """

    # ---- Document ID ----
    document_id = _extract_with_regex(
        r"(?:Doc(?:ument)?\s*ID|Invoice\s*No|Order\s*No|PO\s*No|Contract\s*No)[:\s#\-]*([A-Z0-9\-\/_]+)",
        text,
    )

    # ---- Date (expanded patterns) ----
    date = _extract_with_regex(
        r"(?:Date|Issue\s*Date|Invoice\s*Date|Signing\s*Date)[:\s\-]*("
        r"[0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2}|"
        r"[0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4}|"
        r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s*[0-9]{4}|"
        r"[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4}"
        r")",
        text,
    )

    # ---- Companies ----
    from_company = _extract_with_regex(
        r"(?:From|Supplier|Seller|Shipper|Exporter|Consignor)\s*(?:Company|Name)?[:\s\-]*([^\n\r]{3,80})",
        text,
    )

    to_company = _extract_with_regex(
        r"(?:To|Buyer|Receiver|Importer|Consignee)\s*(?:Company|Name)?[:\s\-]*([^\n\r]{3,80})",
        text,
    )

    # Clean them
    from_company = _clean_line(from_company)
    to_company = _clean_line(to_company)

    # ---- Quantity ----
    quantity = _extract_with_regex(
        r"(?:Qty|Quantity|Total\s*Qty|Total\s*Quantity)[:\s\-]*([0-9][0-9,.\s]{0,20}\s*[A-Za-z]{0,10})",
        text,
    )

    quantity = _clean_line(quantity)

    # ---- Product ----
    product_material_name = _extract_with_regex(
        r"(?:Product(?:/Material)?\s*Name|Material\s*Name|Product\s*Description|Item\s*Description)[:\s\-]*([^\n\r]{3,120})",
        text,
    )

    product_material_name = _clean_line(product_material_name)

    # ---- Fallback company detection ----
    companies_detected = _extract_companies_fallback(text)

    # If main fields missing → try fallback
    if not from_company and companies_detected:
        from_company = companies_detected[0]

    if not to_company and len(companies_detected) > 1:
        to_company = companies_detected[1]

    return {
        "document_id": document_id,
        "date": date,
        "from_company": from_company,
        "to_company": to_company,
        "quantity": quantity,
        "product_material_name": product_material_name,
        "companies_detected": companies_detected,
    }