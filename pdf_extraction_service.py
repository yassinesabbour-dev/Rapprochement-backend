# -*- coding: utf-8 -*-
import io
import json
import os
import re
import tempfile
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader

from reconciliation_engine import standardize_bank_entries, standardize_invoices


SYSTEM_MESSAGE = (
    "Tu extrais des donnÃ©es structurÃ©es depuis des PDF de factures et relevÃ©s bancaires en franÃ§ais. "
    "Tu rÃ©ponds uniquement en JSON strict, sans texte autour."
)
TEXT_MODEL = ("gemini", "gemini-2.5-flash")
PDF_MODEL = ("gemini", "gemini-2.5-flash")


def build_prompt(dataset: str, file_name: str) -> str:
    if dataset == "invoices":
        return f"""
Analyse ce PDF de facture nommÃ© {file_name}.

Retourne UNIQUEMENT un JSON valide de la forme :
{{
  "document_type": "invoice",
  "rows": [
    {{
      "invoice_number": "...",
      "customer_name": "...",
      "issue_date": "YYYY-MM-DD ou null",
      "due_date": "YYYY-MM-DD ou null",
      "amount": "nombre ou texte montant",
      "currency": "EUR ou MAD ou null",
      "confidence": 0.0,
      "extraction_notes": ["liste des champs incertains ou manquants"]
    }}
  ],
  "summary": "rÃ©sumÃ© court"
}}

Consignes :
- Extrais toutes les factures dÃ©tectables du PDF.
- Si le PDF est scannÃ©, lis-le visuellement.
- DÃ©tecte les montants TTC utiles pour le rapprochement.
- Si une valeur manque ou paraÃ®t ambiguÃ«, mets null si nÃ©cessaire et ajoute une note dans extraction_notes.
- Les devises attendues sont surtout EUR et MAD.
- N'invente jamais de donnÃ©e.
"""
    return f"""
Analyse ce PDF de relevÃ© bancaire nommÃ© {file_name}.

Retourne UNIQUEMENT un JSON valide de la forme :
{{
  "document_type": "bank_statement",
  "rows": [
    {{
      "booking_date": "YYYY-MM-DD ou null",
      "label": "...",
      "reference": "...",
      "amount": "nombre ou texte montant",
      "direction": "credit ou debit",
      "currency": "EUR ou MAD ou null",
      "confidence": 0.0,
      "extraction_notes": ["liste des champs incertains ou manquants"]
    }}
  ],
  "summary": "rÃ©sumÃ© court"
}}

Consignes :
- Extrais toutes les lignes utiles du relevÃ©.
- Garde chaque opÃ©ration bancaire sur une ligne distincte.
- Si le PDF est scannÃ©, lis-le visuellement.
- Le champ amount reprÃ©sente le montant de la ligne; direction indique crÃ©dit ou dÃ©bit.
- Si une valeur manque ou paraÃ®t ambiguÃ«, mets null si nÃ©cessaire et ajoute une note dans extraction_notes.
- Les devises attendues sont surtout EUR et MAD.
- N'invente jamais de donnÃ©e.
"""


def build_text_prompt(dataset: str, file_name: str, extracted_text: str) -> str:
    base_instructions = build_prompt(dataset, file_name)
    return f"""
{base_instructions}

Texte extrait du PDF :
---
{extracted_text[:18000]}
---
"""


def extract_json_payload(response_text: str):
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("RÃ©ponse OCR invalide : JSON introuvable.")
        return json.loads(match.group(0))


def extract_text_from_pdf(content: bytes):
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        page_text = (page.extract_text(extraction_mode="layout") or "").strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages).strip()


def first_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


DATE_PATTERN = r"(?:\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{2,4}|\d{2}\s+\d{2}\s+\d{4})"
AMOUNT_PATTERN = r"(?:-?\d{1,3}(?:[ .]\d{3})*|[-+]?\d+)[,\.]\d{2}(?:\s*(?:EUR|MAD|â¬|DH|DHS|DIRHAM))?"


def normalize_line(line: str):
    return re.sub(r"\s+", " ", (line or "").strip())


def infer_direction(text: str):
    lowered = normalize_line(text).lower()
    debit_markers = ["dÃ©bit", "debit", "prlv", "prÃ©lÃ¨vement", "prelevement", "carte", "frais", "commission", "retrait", "virement emis"]
    credit_markers = ["crÃ©dit", "credit", "virement", "versement", "vir recu", "vir reÃ§u", "encaissement", "remise", "depot", "dÃ©pÃ´t"]

    if any(marker in lowered for marker in debit_markers):
        return "debit"
    if any(marker in lowered for marker in credit_markers):
        return "credit"
    return "credit"


def extract_reference_from_text(text: str):
    explicit_ref = first_match(
        [
            r"(?:r[Ã©e]f[Ã©e]rence|ref|n[Â°o])\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-_/]{3,})",
        ],
        text,
    )
    if explicit_ref:
        return explicit_ref

    tokens = re.findall(r"\b[A-Z0-9][A-Z0-9\-_/]{5,}\b", normalize_line(text).upper())
    for token in tokens:
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
            return token
    return None


def build_bank_row_from_block(text: str, fallback_currency: str | None = None):
    normalized = normalize_line(text)
    if not normalized:
        return None

    dates = re.findall(DATE_PATTERN, normalized)
    amounts = re.findall(AMOUNT_PATTERN, normalized, flags=re.IGNORECASE)
    if not dates or not amounts:
        return None

    booking_date = dates[0]
    amount = amounts[-1]
    cleaned_label = normalized
    for date in dates[:2]:
        cleaned_label = cleaned_label.replace(date, " ", 1)
    cleaned_label = cleaned_label.replace(amount, " ", 1)
    cleaned_label = re.sub(r"\b(?:date|valeur|op[Ã©e]ration|d[Ã©e]bit|cr[Ã©e]dit|solde|page)\b", " ", cleaned_label, flags=re.IGNORECASE)
    cleaned_label = normalize_line(cleaned_label)
    if len(cleaned_label) < 3:
        return None

    return {
        "booking_date": booking_date,
        "label": cleaned_label,
        "reference": extract_reference_from_text(normalized),
        "amount": amount,
        "direction": infer_direction(normalized),
        "currency": normalize_currency(first_match([r"\b(EUR|MAD|DH|DHS|DIRHAM|â¬)\b"], amount) or fallback_currency),
        "confidence": 0.9,
        "extraction_notes": [],
    }


def dedupe_bank_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get("booking_date"), normalize_line(row.get("label", "")).upper(), row.get("amount"), row.get("direction"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def heuristic_invoice_rows(extracted_text: str):
    invoice_number = first_match(
        [
            r"(?:num[eé]ro\s*(?:de\s*)?facture|invoice\s*number|r[ée]f[ée]rence\s*facture)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-_/]+)",
            r"(?:facture|fact)\s*[:#n°N°]*\s*([A-Z0-9][A-Z0-9\-_/]*\d+[A-Z0-9\-_/]*)",
            r"N[°o]\s*([A-Z0-9][A-Z0-9\-_/]*\d+[A-Z0-9\-_/]*)",
            r"JRE\s*N[°o]\s*([A-Z0-9][A-Z0-9\-_/]*)",
        ],
        extracted_text,
    )
    customer_name = first_match(
        [
            r"(?:client|nom\s*client|soci[ée]t[ée]|company)\s*[:#-]?\s*([^\n]+)",
        ],
        extracted_text,
    )
    # If customer is SOBETRAC, find the supplier name instead
    if customer_name and "SOBETRAC" in customer_name.upper():
        supplier = first_match(
            [
                r"(?:STE|SOCIETE|SOCIÉTÉ)\s+([A-Z][A-Z\s]{3,30}(?:TRANS|SARL|SA|SARLAU|LOG|FRET))",
                r"^\s*((?:STE|SOCIETE)\s+[A-Z][A-Z\s]+)$",
            ],
            extracted_text,
        )
        if supplier and "SOBETRAC" not in supplier.upper():
            customer_name = supplier.strip()
        else:
            # Try extracting from filename-like patterns or header
            header_lines = extracted_text.split("\n")[:10]
            for hl in header_lines:
                hl = hl.strip()
                if len(hl) > 4 and "SOBETRAC" not in hl.upper() and any(kw in hl.upper() for kw in ["TRANS", "LOG", "FRET", "STE", "SARL"]):
                    customer_name = hl.strip()
                    break
    issue_date = first_match(
        [
            r"(?:date\s*facture|date\s*[ée]mission|date)\s*[:#-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
            r"(?:date\s*facture|date\s*[ée]mission|date)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2})",
            r"(?:LE|le)\s*[:#-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        ],
        extracted_text,
    )
    due_date = first_match(
        [r"(?:date\s*[ée]ch[ée]ance|[ée]ch[ée]ance|due\s*date)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})"],
        extracted_text,
    )
    amount_block = first_match(
        [
            r"(?:total\s*ttc|montant\s*(?:ttc|total)?|net\s*[aà]\s*payer)\s*[:#-]?\s*([0-9\s,\.]+\s*(?:EUR|MAD||DH|DHS|DIRHAM)?)",
        ],
        extracted_text,
    )
    if not amount_block:
        return []
    currency = first_match([r"\b(EUR|MAD|DH|DHS|DIRHAM|)\b"], amount_block) or first_match([r"\b(EUR|MAD|DH|DHS|DIRHAM|)\b"], extracted_text)
    row = {
        "invoice_number": invoice_number,
        "customer_name": customer_name,
        "issue_date": issue_date,
        "due_date": due_date,
        "amount": amount_block,
        "currency": normalize_currency(currency),
        "confidence": 0.95,
        "extraction_notes": [],
    }
    missing = [key for key in ["invoice_number", "issue_date", "amount"] if not row.get(key)]
    if missing:
        row["confidence"] = 0.78
        row["extraction_notes"] = [f"Champs à vérifier: {\', \'.join(missing)}"]
    return [row]

def parse_attijariwafa_line(line):
    """Parse a single Attijariwafa bank statement line in layout mode."""
    import re
    line = line.strip()
    if not line or len(line) < 20:
        return None
    # Pattern: CODE DD MM LABEL DD MM YYYY AMOUNT
    m = re.match(
        r'^(\w{4,8})'           # code
        r'(\d{2})\s+(\d{2})\s+' # booking day month
        r'(.+?)'                 # label
        r'(\d{2})\s+(\d{2})\s+(\d{4})\s+' # value date
        r'([\d\s]+[,\.]\d{2})',  # amount at end
        line
    )
    if not m:
        return None
    code = m.group(1)
    day, month = m.group(2), m.group(3)
    label = m.group(4).strip()
    val_day, val_month, val_year = m.group(5), m.group(6), m.group(7)
    amount_str = m.group(8).replace(' ', '')
    try:
        amount = float(amount_str.replace(',', '.'))
    except ValueError:
        return None
    booking_date = f"{val_year}-{val_month}-{val_day}"
    return {
        "booking_date": booking_date,
        "label": label,
        "reference": code,
        "amount": amount,
        "currency": "MAD",
        "confidence": 0.95,
        "extraction_notes": [],
    }

def parse_attijariwafa_statement(text):
    """Parse full Attijariwafa statement, detecting debit/credit by column position."""
    lines = text.splitlines()
    rows = []
    for line in lines:
        if any(kw in line.upper() for kw in ["SOLDE DEPART", "SOLDE FINAL", "TOTAL MOUVEMENTS", "PAGE", "CODE", "LIBELLE", "VALEUR"]):
            continue
        parsed = parse_attijariwafa_line(line)
        if parsed:
            # Detect direction by column position of amount
            stripped = line.rstrip()
            amount_end = len(stripped)
            # Credit amounts are typically at column 75+, debit at column 55-74
            amount_str = str(parsed["amount"]).replace(".", ",")
            pos = stripped.rfind(amount_str.split(",")[0])
            if pos > 70:
                parsed["direction"] = "credit"
            else:
                parsed["direction"] = "debit"
            rows.append(parsed)
    return rows

def normalize_currency(code):
    if code and code.upper() in ("DH", "DHS", "DIRHAM"):
        return "MAD"
    return code

def heuristic_bank_rows(extracted_text: str):
    # Try Attijariwafa-specific parser first
    if "attijariwafa" in extracted_text.lower() or "DIRHAM" in extracted_text:
        atw_rows = parse_attijariwafa_statement(extracted_text)
        if atw_rows:
            return atw_rows
    fallback_currency = normalize_currency(first_match([r"\b(EUR|MAD|DH|DHS|DIRHAM|â¬)\b"], extracted_text))

    lines = [normalize_line(line) for line in extracted_text.splitlines() if normalize_line(line)]
    block_rows = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(rf"^{DATE_PATTERN}\b", line):
            block_lines = [line]
            look_ahead = index + 1
            while look_ahead < len(lines) and len(block_lines) < 5 and not re.match(rf"^{DATE_PATTERN}\b", lines[look_ahead]):
                block_lines.append(lines[look_ahead])
                look_ahead += 1
            block_row = build_bank_row_from_block(" ".join(block_lines), fallback_currency=fallback_currency)
            if block_row:
                block_rows.append(block_row)
            index = look_ahead
            continue
        index += 1

    inline_rows = []
    for line in lines:
        if re.search(DATE_PATTERN, line) and re.search(AMOUNT_PATTERN, line, flags=re.IGNORECASE):
            inline_row = build_bank_row_from_block(line, fallback_currency=fallback_currency)
            if inline_row:
                inline_rows.append(inline_row)

    candidate_rows = dedupe_bank_rows(block_rows + inline_rows)
    candidate_rows = [r for r in candidate_rows if not any(kw in (r.get("label") or "").upper() for kw in ["SOLDE DEPART", "SOLDE FINAL", "DEPART AU", "FINAL AU", "TOTAL MOUVEMENTS"])]
    if candidate_rows:
        return candidate_rows

    booking_date = first_match(
        [rf"(?:date\s*op[Ã©e]ration|date\s*valeur|date)\s*[:#-]?\s*({DATE_PATTERN})"],
        extracted_text,
    )
    label = first_match([r"(?:libell[Ã©e]|motif|description)\s*[:#-]?\s*([^\n]+)"], extracted_text)
    reference = first_match([r"(?:r[Ã©e]f[Ã©e]rence|reference)\s*[:#-]?\s*([^\n]+)"], extracted_text)
    amount_block = first_match(
        [
            r"(?:montant\s*cr[Ã©e]dit|montant\s*d[Ã©e]bit|cr[Ã©e]dit|d[Ã©e]bit|montant)\s*[:#-]?\s*(?:.*?\s)?(" + AMOUNT_PATTERN + r")"
        ],
        extracted_text,
    )
    if not amount_block:
        return []

    direction = infer_direction(extracted_text)
    currency = first_match([r"\b(EUR|MAD|DH|DHS|DIRHAM|â¬)\b"], amount_block) or fallback_currency
    row = {
        "booking_date": booking_date,
        "label": label,
        "reference": reference,
        "amount": amount_block,
        "direction": direction,
        "currency": normalize_currency(currency),
        "confidence": 0.92,
        "extraction_notes": [],
    }
    missing = [key for key in ["booking_date", "label", "amount"] if not row.get(key)]
    if missing:
        row["confidence"] = 0.75
        row["extraction_notes"] = [f"Champs Ã  vÃ©rifier: {', '.join(missing)}"]
    return [row]


async def run_llm_extraction(prompt: str, api_key: str, session_id: str, file_path: str | None = None):
    client = genai.Client(api_key=api_key)
    model_name = PDF_MODEL[1] if file_path else TEXT_MODEL[1]

    contents = []
    if file_path:
        with open(file_path, "rb") as f:
            pdf_data = f.read()
        contents.append(types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"))
    contents.append(types.Part.from_text(text=prompt))

    response = await client.aio.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_MESSAGE,
            temperature=0.1,
        ),
    )
    return response.text


async def extract_rows_from_pdf(dataset: str, file_name: str, content: bytes):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("EMERGENT_LLM_KEY")
    if not api_key:
        raise ValueError("La clÃ© EMERGENT_LLM_KEY est manquante pour l'analyse PDF.")

    temp_path = None
    try:
        extracted_text = extract_text_from_pdf(content)
        if len(extracted_text) >= 80:
            heuristic_rows = heuristic_invoice_rows(extracted_text) if dataset == "invoices" else heuristic_bank_rows(extracted_text)
            if heuristic_rows:
                if dataset == "invoices":
                    return standardize_invoices(heuristic_rows), "Factures extraites depuis PDF texte standard."
                return standardize_bank_entries(heuristic_rows), "Lignes bancaires extraites depuis PDF texte standard."

            response = await run_llm_extraction(
                build_text_prompt(dataset, file_name, extracted_text),
                api_key,
                f"pdf-text-{dataset}-{Path(file_name).stem}",
            )
            payload = extract_json_payload(response)
            rows = payload.get("rows") or []
            if rows:
                if dataset == "invoices":
                    return standardize_invoices(rows), payload.get("summary") or "Factures extraites depuis PDF texte."
                return standardize_bank_entries(rows), payload.get("summary") or "Lignes bancaires extraites depuis PDF texte."

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix or ".pdf") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        response = await run_llm_extraction(
            build_prompt(dataset, file_name),
            api_key,
            f"pdf-vision-{dataset}-{Path(file_name).stem}",
            file_path=temp_path,
        )

        payload = extract_json_payload(response)
        rows = payload.get("rows") or []
        if not rows:
            raise ValueError("Aucune donnÃ©e exploitable n'a Ã©tÃ© dÃ©tectÃ©e dans le PDF.")

        if dataset == "invoices":
            return standardize_invoices(rows), payload.get("summary") or "Factures extraites depuis PDF."
        return standardize_bank_entries(rows), payload.get("summary") or "Lignes bancaires extraites depuis PDF."
    except Exception as exc:
        message = str(exc)
        if "budget" in message.lower() or "credit" in message.lower():
            raise ValueError(
                "Le PDF scannÃ© nÃ©cessite plus de crÃ©dit d'analyse que disponible actuellement. RÃ©essayez plus tard ou utilisez un PDF texte/export si possible."
            ) from exc
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)