import io
import json
import os
import re
import tempfile
from pathlib import Path

try:n    from emergentintegrations.llm.chat import FileContentWithMimeType, LlmChat, UserMessagenexcept ImportError:n    LlmChat = Nonen    UserMessage = Nonen    FileContentWithMimeType = None
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
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages).strip()


def first_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


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
        "currency": currency,
        "confidence": 0.95,
        "extraction_notes": [],
    }
    missing = [key for key in ["invoice_number", "issue_date", "amount"] if not row.get(key)]
    if missing:
        row["confidence"] = 0.78
        row["extraction_notes"] = [f"Champs à vérifier: {', '.join(missing)}"]
    return [row]


def heuristic_bank_rows(extracted_text: str):
    booking_date = first_match(
        [r"(?:date\s*op[ée]ration|date\s*valeur|date)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})"],
        extracted_text,
    )
    label = first_match([r"(?:libell[ée]|motif|description)\s*[:#-]?\s*([^\n]+)"], extracted_text)
    reference = first_match([r"(?:r[ée]f[ée]rence|reference)\s*[:#-]?\s*([^\n]+)"], extracted_text)
    amount_block = first_match(
        [
            r"(?:montant\s*cr[ée]dit|montant\s*d[ée]bit|cr[ée]dit|d[ée]bit|montant)\s*[:#-]?\s*([0-9\s,\.]+\s*(?:EUR|MAD|€|DH|DHS)?)"
        ],
        extracted_text,
    )
    if not amount_block:
        return []

    lowered = extracted_text.lower()
    direction = "debit" if "debit" in lowered or "débit" in lowered else "credit"
    currency = first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], amount_block) or first_match([r"\b(EUR|MAD|DH|DHS|€)\b"], extracted_text)
    row = {
        "booking_date": booking_date,
        "label": label,
        "reference": reference,
        "amount": amount_block,
        "direction": direction,
        "currency": currency,
        "confidence": 0.92,
        "extraction_notes": [],
    }
    missing = [key for key in ["booking_date", "label", "amount"] if not row.get(key)]
    if missing:
        row["confidence"] = 0.75
        row["extraction_notes"] = [f"Champs à vérifier: {', '.join(missing)}"]
    return [row]


async def run_llm_extraction(prompt: str, api_key: str, session_id: str, file_path: str | None = None):
    chat = LlmChat(
        api_key=api_key,
        session_id=session_id,
        system_message=SYSTEM_MESSAGE,
    )

    if file_path:
        chat = chat.with_model(*PDF_MODEL)
        return await chat.send_message(
            UserMessage(
                text=prompt,
                file_contents=[FileContentWithMimeType(mime_type="application/pdf", file_path=file_path)],
            )
        )

    chat = chat.with_model(*TEXT_MODEL)
    return await chat.send_message(UserMessage(text=prompt))


async def extract_rows_from_pdf(dataset: str, file_name: str, content: bytes):
    api_key = os.environ.get("EMERGENT_LLM_KEY")
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