# pdf_extraction_service.py
# Minimal PDF extraction fallback used by server.import_dataset
# - tries pypdf then PyPDF2 if available
# - performs a conservative line->row detection
# - returns (parsed_rows: list[dict], extraction_summary: str)

import io
import re
from typing import List, Tuple

async def extract_rows_from_pdf(dataset: str, filename: str, content: bytes) -> Tuple[List[dict], str]:
    """
    Minimal PDF extractor:
    - Tries to import pypdf or PyPDF2 and extract text.
    - Returns (rows, summary). Rows are simple dicts compatible with parse_dataset().
    """
    text = ""
    pdf_module = None
    # try modern pypdf first, then PyPDF2
    try:
        import pypdf as pdf_module  # type: ignore
    except Exception:
        try:
            import PyPDF2 as pdf_module  # type: ignore
        except Exception:
            return [], "Aucun parser PDF (pypdf/PyPDF2) installé — importez CSV/XLSX ou installez pypdf/PyPDF2."

    try:
        reader = pdf_module.PdfReader(io.BytesIO(content))
        pages_text = []
        for page in getattr(reader, "pages", []):
            t = getattr(page, "extract_text", lambda: None)()
            if t:
                pages_text.append(t)
        text = "\n".join(pages_text)
        if not text.strip():
            return [], "PDF lu mais aucun texte extrait (possible PDF scanné)."
    except Exception as exc:
        return [], f"Erreur lors de l'extraction PDF : {exc}"

    # Heuristic: detect csv-like lines or lines with an amount
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    amount_re = re.compile(r"[-+]?\d{1,3}(?:[ ,.\u202f]\d{3})*(?:[.,]\d{1,2})?")

    for ln in lines[:1000]:  # safety cap
        if "," in ln or ";" in ln:
            parts = re.split(r"[;,]\s*", ln)
            if len(parts) >= 2 and amount_re.search(parts[-1]):
                if dataset == "invoices":
                    candidates.append({"invoice_number": parts[0].strip(), "amount": parts[-1].strip()})
                else:
                    candidates.append({"label": parts[0].strip(), "amount": parts[-1].strip()})
                continue
        m = amount_re.search(ln)
        if m:
            if dataset == "invoices":
                candidates.append({"invoice_number": "", "amount": m.group(0)})
            else:
                candidates.append({"label": ln, "amount": m.group(0)})

    summary = f"Texte PDF extrait ({len(text)} caractères), {len(candidates)} lignes candidates identifiées."
    return candidates, summary
