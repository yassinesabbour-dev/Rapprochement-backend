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
    "Tu extrais des données structurées depuis des PDF de factures et relevés bancaires en français. "
    "Tu réponds uniquement en JSON strict, sans texte autour."
)
TEXT_MODEL = ("gemini", "gemini-2.5-flash")
PDF_MODEL = ("gemini", "gemini-2.5-flash")


def build_prompt(dataset: str, file_name: str) -> str:
    if dataset == "invoices":
        return f"""
Analyse ce PDF de facture nommé {file_name}.

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
  "summary": "résumé court"
}}

Consignes :
- Extrais toutes les factures détectables du PDF.
- Si le PDF est scanné, lis-le visuellement.
- Détecte les montants TTC utiles pour le rapprochement.
- Si une valeur manque ou paraît ambiguë, mets null si nécessaire et ajoute une note dans extraction_notes.
- Les devises attendues sont surtout EUR et MAD.
- N'invente jamais de donnée.
"""
    return f"""
Analyse ce PDF de relevé bancaire nommé {file_name}.

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
  "summary": "résumé court"
}}

Consignes :
- Extrais toutes les lignes utiles du relevé.
- Garde chaque opération bancaire sur une ligne distincte.
- Si le PDF est scanné, lis-le visuellement.
- Le champ amount représente le montant de la ligne; direction indique crédit ou débit.
- Si une valeur manque ou paraît ambiguë, mets null si nécessaire et ajoute une note dans extraction_notes.
- Les devises attendues sont surtout EUR et MAD.
- N'invente jamais de donnée.
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
            raise ValueError("Réponse OCR invalide : JSON introuvable.")
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
AMOUNT_PATTERN = r"(?:-?\d{1,3}(?:[ .]\d{3})*|[-+]?\d+)[,\.]\d{2}(?:\s*(?:EUR|MAD|€|DH|DHS))?"


def normalize_line(line: str):
    return re.sub(r"\s+", " ", (line or "").strip())


def infer_direction(text: str):
    lowered = normalize_line(text).lower()
    debit_markers = ["débit", "debit", "prlv", "prélèvement", "prelevement", "carte", "frais", "commission", "retrait", "virement emis"]
    credit_markers = ["crédit", "credit", "virement", "versement", "vir recu", "vir reçu", "encaissement", "remise", "depot", "dépôt"]

    if any(marker in lowered for marker in debit_markers):
        return "debit"
    if any(marker in lowered for marker in credit_markers):
        return "credit"
    return "credit"


def extract_reference_from_text(text: str):
    explicit_ref = first_match(
        [
            r"(?:r[ée]f[ée]rence|ref|n[°o])\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-_/]{3,})",
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
    cleaned_label = re.sub(r"\b(?:date|valeur|op[ée]ration|d[ée]bit|cr[ée]dit|solde|page)\b", " ", cleaned_label, flags=re.IGNORECASE)
    cleaned_label = normalize_line(cleaned_label)
    if len(cleaned_label) < 3:
        return None

    return {
        "booking_date": booking_date,
        "label": cleaned_label,
        "reference": extract_reference_from_text(normalized),
        "amount": amount,
        "direction": infer_direction(normalized),
        "currency": normalize_currency(first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], amount) or fallback_currency),
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
            r"^facture\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-_/]+)$",
        ],
        extracted_text,
    )
    customer_name = first_match(
        [
            r"(?:client|nom\s*client|soci[ée]t[ée]|company)\s*[:#-]?\s*([^\n]+)",
        ],
        extracted_text,
    )
    issue_date = first_match(
        [r"(?:date\s*facture|date\s*[ée]mission|date)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})"],
        extracted_text,
    )
    due_date = first_match(
        [r"(?:date\s*[ée]ch[ée]ance|[ée]ch[ée]ance|due\s*date)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})"],
        extracted_text,
    )
    amount_block = first_match(
        [
            r"(?:montant\s*(?:ttc|total)?|total\s*ttc|net\s*[aà]\s*payer)\s*[:#-]?\s*([0-9\s,\.]+\s*(?:EUR|MAD|€|DH|DHS)?)",
        ],
        extracted_text,
    )
    if not amount_block:
        return []

    currency = first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], amount_block) or first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], extracted_text)
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
        row["extraction_notes"] = [f"Champs à vérifier: {', '.join(missing)}"]
    return [row]


def normalize_currency(code):
    if code and code.upper() in ("DH", "DHS", "DIRHAM"):
        return "MAD"
    return code

def heuristic_bank_rows(extracted_text: str):
    fallback_currency = normalize_currency(first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], extracted_text))

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
        [rf"(?:date\s*op[ée]ration|date\s*valeur|date)\s*[:#-]?\s*({DATE_PATTERN})"],
        extracted_text,
    )
    label = first_match([r"(?:libell[ée]|motif|description)\s*[:#-]?\s*([^\n]+)"], extracted_text)
    reference = first_match([r"(?:r[ée]f[ée]rence|reference)\s*[:#-]?\s*([^\n]+)"], extracted_text)
    amount_block = first_match(
        [
            r"(?:montant\s*cr[ée]dit|montant\s*d[ée]bit|cr[ée]dit|d[ée]bit|montant)\s*[:#-]?\s*(?:.*?\s)?(" + AMOUNT_PATTERN + r")"
        ],
        extracted_text,
    )
    if not amount_block:
        return []

    direction = infer_direction(extracted_text)
    currency = first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], amount_block) or fallback_currency
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
        row["extraction_notes"] = [f"Champs à vérifier: {', '.join(missing)}"]
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
        raise ValueError("La clé EMERGENT_LLM_KEY est manquante pour l'analyse PDF.")

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
            raise ValueError("Aucune donnée exploitable n'a été détectée dans le PDF.")

        if dataset == "invoices":
            return standardize_invoices(rows), payload.get("summary") or "Factures extraites depuis PDF."
        return standardize_bank_entries(rows), payload.get("summary") or "Lignes bancaires extraites depuis PDF."
    except Exception as exc:
        message = str(exc)
        if "budget" in message.lower() or "credit" in message.lower():
            raise ValueError(
                "Le PDF scanné nécessite plus de crédit d'analyse que disponible actuellement. Réessayez plus tard ou utilisez un PDF texte/export si possible."
            ) from exc
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)