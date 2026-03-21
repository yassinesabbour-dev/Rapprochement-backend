import copy
import io
import json
import re
import uuid
from datetime import datetime, timezone
from itertools import combinations

import pandas as pd


INVOICE_FIELD_ALIASES = {
    "invoice_number": [
        "invoice_number",
        "invoice",
        "numero_facture",
        "num_facture",
        "facture",
        "reference",
        "ref",
        "invoice_ref",
    ],
    "customer_name": [
        "customer_name",
        "client",
        "nom_client",
        "customer",
        "societe",
        "company",
    ],
    "issue_date": ["issue_date", "date", "date_facture", "invoice_date", "date_emission"],
    "due_date": ["due_date", "date_echeance", "echeance", "due", "due_on"],
    "amount": ["amount", "montant", "total", "ttc", "amount_due", "solde"],
    "currency": ["currency", "devise", "currency_code", "monnaie"],
}

BANK_FIELD_ALIASES = {
    "booking_date": [
        "booking_date",
        "date",
        "date_operation",
        "date_valeur",
        "operation_date",
    ],
    "label": ["label", "libelle", "description", "motif", "wording"],
    "amount": ["amount", "montant", "credit", "incoming", "encaissement"],
    "credit": ["credit", "incoming", "encaissement"],
    "debit": ["debit", "sortie", "paiement"],
    "reference": ["reference", "ref", "bank_reference"],
    "currency": ["currency", "devise", "currency_code", "monnaie"],
}

CURRENCY_SYMBOLS = {
    "€": "EUR",
    "EUR": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "MAD": "MAD",
    "DH": "MAD",
    "DHS": "MAD",
    "DIRHAM": "MAD",
    "DIRHAMS": "MAD",
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str):
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_key(value):
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def compact_text(value):
    text = normalize_text(value).upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def parse_amount(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return round(float(value), 2)

    text = normalize_text(value)
    if not text:
        return None

    text = text.replace("€", "").replace(" ", "")
    text = re.sub(r"(?i)(EUR|EURO|EUROS|MAD|DH|DHS|DIRHAM|DIRHAMS)", "", text)
    if text.count(",") == 1 and text.count(".") > 1:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")

    try:
        return round(float(text), 2)
    except ValueError:
        return None


def normalize_currency(value, default="EUR"):
    text = normalize_text(value).upper()
    if not text:
        return default

    compact = re.sub(r"[^A-Z€]", "", text)
    for symbol, currency in CURRENCY_SYMBOLS.items():
        if symbol in compact:
            return currency
    return default


def detect_currency(*values, default="EUR"):
    for value in values:
        currency = normalize_currency(value, default=None)
        if currency:
            return currency
    return default


def parse_date(value):
    if value is None or normalize_text(value) == "":
        return None

    text_value = normalize_text(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text_value):
        try:
            return datetime.strptime(text_value, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            return None

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def date_distance(date_a, date_b):
    if not date_a or not date_b:
        return 99
    parsed_a = pd.to_datetime(date_a, errors="coerce")
    parsed_b = pd.to_datetime(date_b, errors="coerce")
    if pd.isna(parsed_a) or pd.isna(parsed_b):
        return 99
    return abs((parsed_a - parsed_b).days)


def money(value):
    return round(float(value or 0), 2)


def default_metrics():
    return {
        "total_invoices": 0,
        "total_bank_entries": 0,
        "matched_invoices": 0,
        "partially_matched_invoices": 0,
        "unmatched_invoices": 0,
        "to_review": 0,
        "matched_amount": 0.0,
        "outstanding_amount": 0.0,
        "matched_amounts_by_currency": {},
        "outstanding_amounts_by_currency": {},
    }


def build_empty_workspace():
    return {
        "id": "default-workspace",
        "workspace_name": "Atelier de rapprochement",
        "invoices_raw": [],
        "bank_entries_raw": [],
        "manual_links": [],
        "imported_sources": [],
        "activity": [],
        "invoices": [],
        "bank_entries": [],
        "matches": [],
        "metrics": default_metrics(),
        "last_run_at": None,
    }


def build_demo_workspace():
    workspace = build_empty_workspace()
    workspace["workspace_name"] = "Démonstration multi-factures"
    workspace["invoices_raw"] = [
        {
            "id": "inv-demo-001",
            "invoice_number": "FA-2025-001",
            "customer_name": "Studio Atlas",
            "issue_date": "2025-02-01",
            "due_date": "2025-02-15",
            "amount": 1200.0,
            "currency": "EUR",
            "source": "demo",
        },
        {
            "id": "inv-demo-002",
            "invoice_number": "FA-2025-002",
            "customer_name": "Studio Atlas",
            "issue_date": "2025-02-03",
            "due_date": "2025-02-17",
            "amount": 850.0,
            "currency": "EUR",
            "source": "demo",
        },
        {
            "id": "inv-demo-003",
            "invoice_number": "FA-2025-003",
            "customer_name": "Maison Rivage",
            "issue_date": "2025-02-04",
            "due_date": "2025-02-19",
            "amount": 640.0,
            "currency": "EUR",
            "source": "demo",
        },
        {
            "id": "inv-demo-004",
            "invoice_number": "FA-2025-004",
            "customer_name": "Atelier Lune",
            "issue_date": "2025-02-06",
            "due_date": "2025-02-21",
            "amount": 430.0,
            "currency": "EUR",
            "source": "demo",
        },
        {
            "id": "inv-demo-005",
            "invoice_number": "FA-2025-005",
            "customer_name": "Boreal Conseil",
            "issue_date": "2025-02-08",
            "due_date": "2025-02-24",
            "amount": 1290.0,
            "currency": "EUR",
            "source": "demo",
        },
        {
            "id": "inv-demo-006",
            "invoice_number": "FA-2025-006",
            "customer_name": "Atelier Nacre",
            "issue_date": "2025-02-10",
            "due_date": "2025-02-28",
            "amount": 510.0,
            "currency": "EUR",
            "source": "demo",
        },
    ]
    workspace["bank_entries_raw"] = [
        {
            "id": "bnk-demo-001",
            "booking_date": "2025-02-12",
            "label": "VIREMENT STUDIO ATLAS REGLEMENT FA-2025-001 FA-2025-002",
            "reference": "ATLAS-LOT-0212",
            "amount": 2050.0,
            "currency": "EUR",
            "direction": "credit",
            "source": "demo",
        },
        {
            "id": "bnk-demo-002",
            "booking_date": "2025-02-20",
            "label": "MAISON RIVAGE REGLEMENT FA-2025-003",
            "reference": "RIVAGE-0320",
            "amount": 640.0,
            "currency": "EUR",
            "direction": "credit",
            "source": "demo",
        },
        {
            "id": "bnk-demo-003",
            "booking_date": "2025-02-25",
            "label": "PAIEMENT REGROUPE FEVRIER",
            "reference": "LOT-940",
            "amount": 935.0,
            "currency": "EUR",
            "direction": "credit",
            "source": "demo",
        },
        {
            "id": "bnk-demo-004",
            "booking_date": "2025-02-26",
            "label": "BOREAL CONSEIL ACOMPTE FA-2025-005",
            "reference": "BOREAL-ACOMPTE",
            "amount": 800.0,
            "currency": "EUR",
            "direction": "credit",
            "source": "demo",
        },
        {
            "id": "bnk-demo-005",
            "booking_date": "2025-02-27",
            "label": "FRAIS BANCAIRES MENSUELS",
            "reference": "BANK-FEE",
            "amount": -24.0,
            "currency": "EUR",
            "direction": "debit",
            "source": "demo",
        },
    ]
    workspace["imported_sources"] = [
        {
            "id": make_id("src"),
            "dataset": "demo",
            "file_name": "jeu-de-demonstration",
            "rows_count": 11,
            "imported_at": utc_now_iso(),
        }
    ]
    workspace["activity"] = [
        {
            "id": make_id("act"),
            "kind": "demo",
            "message": "Jeu de démonstration chargé avec virements multi-factures.",
            "created_at": utc_now_iso(),
        }
    ]
    return recalculate_workspace(workspace)


def add_activity(workspace, kind: str, message: str):
    activity = workspace.get("activity", [])
    entry = {
        "id": make_id("act"),
        "kind": kind,
        "message": message,
        "created_at": utc_now_iso(),
    }
    workspace["activity"] = [entry, *activity][:12]


def pick_value(row, aliases):
    for alias in aliases:
        if alias in row and normalize_text(row.get(alias)) != "":
            return row.get(alias)
    return None


def load_dataframe(file_name: str, content: bytes):
    lowered = file_name.lower()
    if lowered.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if lowered.endswith(".xlsx") or lowered.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    if lowered.endswith(".json"):
        payload = json.loads(content.decode("utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows") or payload.get("items") or payload.get("data") or []
        return pd.DataFrame(payload)
    raise ValueError("Formats acceptés : CSV, XLSX, XLS, JSON.")


def parse_dataset(file_name: str, content: bytes, dataset: str):
    frame = load_dataframe(file_name, content)
    if frame.empty:
        raise ValueError("Le fichier est vide.")

    rows = []
    for raw_row in frame.fillna("").to_dict(orient="records"):
        rows.append({normalize_key(key): value for key, value in raw_row.items()})

    if dataset == "invoices":
        return standardize_invoices(rows)
    return standardize_bank_entries(rows)


def standardize_invoices(rows):
    invoices = []
    for index, row in enumerate(rows, start=1):
        amount = parse_amount(pick_value(row, INVOICE_FIELD_ALIASES["amount"]))
        if amount is None:
            continue

        invoice_number = normalize_text(pick_value(row, INVOICE_FIELD_ALIASES["invoice_number"]))
        issue_date = parse_date(pick_value(row, INVOICE_FIELD_ALIASES["issue_date"])) or utc_now_iso()[:10]
        due_date = parse_date(pick_value(row, INVOICE_FIELD_ALIASES["due_date"])) or issue_date
        customer_name = normalize_text(pick_value(row, INVOICE_FIELD_ALIASES["customer_name"])) or "Client sans nom"
        currency = detect_currency(
            pick_value(row, INVOICE_FIELD_ALIASES["currency"]),
            pick_value(row, INVOICE_FIELD_ALIASES["amount"]),
        )
        extraction_notes = [normalize_text(note) for note in row.get("extraction_notes", []) if normalize_text(note)]
        confidence = row.get("confidence")
        confidence = float(confidence) if confidence not in (None, "") else None
        review_required = bool(extraction_notes) or (confidence is not None and confidence < 0.85)

        invoices.append(
            {
                "id": make_id("inv"),
                "invoice_number": invoice_number or f"FA-IMPORT-{index:03d}",
                "customer_name": customer_name,
                "issue_date": issue_date,
                "due_date": due_date,
                "amount": abs(amount),
                "currency": currency,
                "source_review_required": review_required,
                "extraction_notes": extraction_notes,
                "source": "import",
            }
        )

    if not invoices:
        raise ValueError("Impossible d'extraire des factures. Vérifiez les colonnes montant, date et référence.")
    return invoices


def standardize_bank_entries(rows):
    entries = []
    for index, row in enumerate(rows, start=1):
        amount = parse_amount(pick_value(row, BANK_FIELD_ALIASES["amount"]))
        credit = parse_amount(pick_value(row, BANK_FIELD_ALIASES["credit"]))
        debit = parse_amount(pick_value(row, BANK_FIELD_ALIASES["debit"]))

        final_amount = amount
        if final_amount is None and credit is not None:
            final_amount = abs(credit)
        if final_amount is None and debit is not None:
            final_amount = -abs(debit)
        if final_amount is None:
            continue

        label = normalize_text(pick_value(row, BANK_FIELD_ALIASES["label"])) or f"Ligne bancaire {index}"
        reference = normalize_text(pick_value(row, BANK_FIELD_ALIASES["reference"]))
        booking_date = parse_date(pick_value(row, BANK_FIELD_ALIASES["booking_date"])) or utc_now_iso()[:10]
        direction = normalize_text(row.get("direction")).lower() or ("credit" if final_amount >= 0 else "debit")
        direction = "debit" if direction == "debit" else "credit"
        currency = detect_currency(
            pick_value(row, BANK_FIELD_ALIASES["currency"]),
            pick_value(row, BANK_FIELD_ALIASES["amount"]),
            label,
            reference,
        )
        extraction_notes = [normalize_text(note) for note in row.get("extraction_notes", []) if normalize_text(note)]
        confidence = row.get("confidence")
        confidence = float(confidence) if confidence not in (None, "") else None
        review_required = bool(extraction_notes) or (confidence is not None and confidence < 0.85)

        entries.append(
            {
                "id": make_id("bnk"),
                "booking_date": booking_date,
                "label": label,
                "reference": reference,
                "amount": abs(final_amount),
                "signed_amount": final_amount,
                "currency": currency,
                "direction": direction,
                "source_review_required": review_required,
                "extraction_notes": extraction_notes,
                "source": "import",
            }
        )

    if not entries:
        raise ValueError("Impossible d'extraire des lignes bancaires. Vérifiez les colonnes date, libellé et montant.")
    return entries


def similarity_score(bank_entry, invoice):
    customer_overlap = len(
        set(compact_text(bank_entry.get("label", "")).split())
        & set(compact_text(invoice.get("customer_name", "")).split())
    )
    date_score = max(0, 40 - date_distance(bank_entry.get("booking_date"), invoice.get("due_date")))
    return customer_overlap + date_score


def invoice_ref_in_label(label, invoice_number):
    return compact_text(invoice_number) in compact_text(label)


def initialize_runtime_records(workspace):
    invoices = copy.deepcopy(workspace.get("invoices_raw", []))
    bank_entries = copy.deepcopy(workspace.get("bank_entries_raw", []))

    for invoice in invoices:
        invoice["outstanding_amount"] = money(invoice.get("amount", 0))
        invoice["matched_amount"] = 0.0
        invoice["status"] = "non rapprochée"
        invoice["currency"] = normalize_currency(invoice.get("currency"), default="EUR")
        invoice["review_required"] = bool(invoice.get("source_review_required"))
        invoice["match_notes"] = list(invoice.get("extraction_notes", []))

    for bank_entry in bank_entries:
        signed_amount = bank_entry.get("signed_amount")
        if signed_amount is None:
            signed_amount = bank_entry.get("amount", 0) if bank_entry.get("direction") != "debit" else -bank_entry.get("amount", 0)
        bank_entry["signed_amount"] = signed_amount
        bank_entry["remaining_amount"] = money(bank_entry.get("amount", 0)) if bank_entry.get("direction") == "credit" else 0.0
        bank_entry["status"] = "hors scope" if bank_entry.get("direction") == "debit" else "non rapproché"
        bank_entry["currency"] = normalize_currency(bank_entry.get("currency"), default="EUR")
        bank_entry["review_required"] = bool(bank_entry.get("source_review_required"))

    return invoices, bank_entries


def currency_matches(bank_entry, invoice):
    return normalize_currency(bank_entry.get("currency"), default="EUR") == normalize_currency(
        invoice.get("currency"), default="EUR"
    )


def register_match(matches, bank_entry, invoices, match_basis, status, note, confidence):
    allocations = []
    starting_remaining = money(bank_entry.get("remaining_amount", 0))
    remaining = starting_remaining

    for invoice in invoices:
        if remaining <= 0 or invoice.get("outstanding_amount", 0) <= 0:
            continue
        applied_amount = min(invoice["outstanding_amount"], remaining)
        if applied_amount <= 0:
            continue

        invoice["outstanding_amount"] = money(invoice["outstanding_amount"] - applied_amount)
        invoice["matched_amount"] = money(invoice.get("matched_amount", 0) + applied_amount)
        invoice["review_required"] = invoice.get("review_required", False) or status == "à vérifier"
        invoice["match_notes"] = [*invoice.get("match_notes", []), note][:3]

        remaining = money(remaining - applied_amount)
        allocations.append(
            {
                "invoice_id": invoice["id"],
                "invoice_number": invoice["invoice_number"],
                "customer_name": invoice["customer_name"],
                "applied_amount": applied_amount,
                "currency": invoice.get("currency", "EUR"),
            }
        )

    consumed = money(starting_remaining - remaining)
    if consumed <= 0:
        return

    bank_entry["remaining_amount"] = remaining
    bank_entry["review_required"] = bank_entry.get("review_required", False) or status == "à vérifier"

    if remaining == 0:
        bank_entry["status"] = "rapproché"
    else:
        bank_entry["status"] = "partiel"

    matches.append(
        {
            "id": make_id("match"),
            "bank_entry_id": bank_entry["id"],
            "bank_label": bank_entry["label"],
            "bank_reference": bank_entry.get("reference", ""),
            "booking_date": bank_entry["booking_date"],
            "invoice_ids": [item["invoice_id"] for item in allocations],
            "invoice_numbers": [item["invoice_number"] for item in allocations],
            "allocations": allocations,
            "bank_amount": starting_remaining,
            "applied_amount": consumed,
            "difference": money(starting_remaining - consumed),
            "currency": bank_entry.get("currency", "EUR"),
            "match_basis": match_basis,
            "status": status,
            "confidence": confidence,
            "note": note,
        }
    )


def find_invoice(invoices, invoice_id):
    for invoice in invoices:
        if invoice["id"] == invoice_id:
            return invoice
    return None


def find_bank_entry(bank_entries, bank_entry_id):
    for bank_entry in bank_entries:
        if bank_entry["id"] == bank_entry_id:
            return bank_entry
    return None


def finalize_records(invoices, bank_entries):
    for invoice in invoices:
        if invoice["outstanding_amount"] <= 0:
            invoice["status"] = "payée"
        elif invoice["matched_amount"] > 0:
            invoice["status"] = "partielle"
        else:
            invoice["status"] = "non rapprochée"

        invoice["outstanding_amount"] = money(invoice["outstanding_amount"])
        invoice["matched_amount"] = money(invoice["matched_amount"])

    for bank_entry in bank_entries:
        if bank_entry.get("direction") == "debit":
            bank_entry["status"] = "hors scope"
            continue
        if bank_entry["remaining_amount"] <= 0:
            bank_entry["status"] = "rapproché"
        elif bank_entry["remaining_amount"] < bank_entry["amount"]:
            bank_entry["status"] = "partiel"
        else:
            bank_entry["status"] = "non rapproché"


def compute_metrics(invoices, bank_entries, matches):
    outstanding_amount = money(sum(invoice.get("outstanding_amount", 0) for invoice in invoices))
    matched_amount = money(sum(match.get("applied_amount", 0) for match in matches))
    outstanding_by_currency = {}
    for invoice in invoices:
        currency = normalize_currency(invoice.get("currency"), default="EUR")
        outstanding_by_currency[currency] = money(outstanding_by_currency.get(currency, 0) + invoice.get("outstanding_amount", 0))

    matched_by_currency = {}
    for match in matches:
        currency = normalize_currency(match.get("currency"), default="EUR")
        matched_by_currency[currency] = money(matched_by_currency.get(currency, 0) + match.get("applied_amount", 0))

    review_items = len([invoice for invoice in invoices if invoice.get("review_required")])
    review_items += len([entry for entry in bank_entries if entry.get("review_required")])
    review_items += len([match for match in matches if match.get("status") == "à vérifier"])

    return {
        "total_invoices": len(invoices),
        "total_bank_entries": len(bank_entries),
        "matched_invoices": len([invoice for invoice in invoices if invoice.get("status") == "payée"]),
        "partially_matched_invoices": len([invoice for invoice in invoices if invoice.get("status") == "partielle"]),
        "unmatched_invoices": len([invoice for invoice in invoices if invoice.get("status") == "non rapprochée"]),
        "to_review": review_items,
        "matched_amount": matched_amount,
        "outstanding_amount": outstanding_amount,
        "matched_amounts_by_currency": matched_by_currency,
        "outstanding_amounts_by_currency": outstanding_by_currency,
    }


def run_reference_matches(invoices, bank_entries, matches):
    for bank_entry in bank_entries:
        if bank_entry.get("direction") != "credit" or bank_entry.get("remaining_amount", 0) <= 0:
            continue

        direct_candidates = [
            invoice
            for invoice in invoices
            if invoice.get("outstanding_amount", 0) > 0
            and currency_matches(bank_entry, invoice)
            and invoice_ref_in_label(bank_entry.get("label", ""), invoice.get("invoice_number", ""))
        ]

        if not direct_candidates:
            continue

        exact_total = money(sum(invoice["outstanding_amount"] for invoice in direct_candidates))
        if abs(exact_total - bank_entry["remaining_amount"]) < 0.01:
            register_match(
                matches,
                bank_entry,
                sorted(direct_candidates, key=lambda item: item["invoice_number"]),
                "référence détectée",
                "confirmé",
                "Correspondance par référence trouvée dans le libellé bancaire.",
                0.98,
            )
            continue

        exact_single = [
            invoice
            for invoice in direct_candidates
            if abs(invoice.get("outstanding_amount", 0) - bank_entry.get("remaining_amount", 0)) < 0.01
        ]
        if exact_single:
            register_match(
                matches,
                bank_entry,
                [exact_single[0]],
                "référence + montant",
                "confirmé",
                "Le montant et la référence correspondent à la facture.",
                0.97,
            )
            continue

        if len(direct_candidates) == 1 and bank_entry["remaining_amount"] < direct_candidates[0]["outstanding_amount"]:
            register_match(
                matches,
                bank_entry,
                [direct_candidates[0]],
                "acompte référencé",
                "à vérifier",
                "Paiement partiel détecté sur une facture référencée.",
                0.74,
            )


def run_combination_matches(invoices, bank_entries, matches):
    for bank_entry in bank_entries:
        if bank_entry.get("direction") != "credit" or bank_entry.get("remaining_amount", 0) <= 0:
            continue

        open_invoices = [
            invoice
            for invoice in invoices
            if invoice.get("outstanding_amount", 0) > 0 and currency_matches(bank_entry, invoice)
        ]
        if len(open_invoices) < 2:
            continue

        candidate_pool = sorted(
            open_invoices,
            key=lambda invoice: (
                date_distance(bank_entry.get("booking_date"), invoice.get("due_date")),
                -invoice.get("amount", 0),
            ),
        )[:10]

        exact_match = None
        review_match = None
        max_size = min(4, len(candidate_pool))

        for size in range(2, max_size + 1):
            for combo in combinations(candidate_pool, size):
                total = money(sum(invoice.get("outstanding_amount", 0) for invoice in combo))
                difference = money(abs(total - bank_entry.get("remaining_amount", 0)))
                if difference < 0.01 and exact_match is None:
                    exact_match = combo
                elif difference <= 2 and review_match is None:
                    review_match = combo
            if exact_match:
                break

        if exact_match:
            register_match(
                matches,
                bank_entry,
                sorted(exact_match, key=lambda item: item["invoice_number"]),
                "somme de plusieurs factures",
                "confirmé",
                "Le virement couvre exactement plusieurs factures ouvertes.",
                0.88,
            )
            continue

        if review_match:
            register_match(
                matches,
                bank_entry,
                sorted(review_match, key=lambda item: item["invoice_number"]),
                "combinaison proche",
                "à vérifier",
                "Le montant est très proche d'un lot de factures, validation manuelle conseillée.",
                0.63,
            )


def run_single_amount_matches(invoices, bank_entries, matches):
    for bank_entry in bank_entries:
        if bank_entry.get("direction") != "credit" or bank_entry.get("remaining_amount", 0) <= 0:
            continue

        open_invoices = [
            invoice
            for invoice in invoices
            if invoice.get("outstanding_amount", 0) > 0 and currency_matches(bank_entry, invoice)
        ]
        exact_invoices = [
            invoice
            for invoice in open_invoices
            if abs(invoice.get("outstanding_amount", 0) - bank_entry.get("remaining_amount", 0)) < 0.01
        ]

        if len(exact_invoices) == 1:
            register_match(
                matches,
                bank_entry,
                [exact_invoices[0]],
                "montant exact",
                "confirmé",
                "Le montant du virement correspond exactement à une facture ouverte.",
                0.82,
            )
            continue

        if not exact_invoices:
            continue

        best_invoice = sorted(
            exact_invoices,
            key=lambda invoice: (
                date_distance(bank_entry.get("booking_date"), invoice.get("due_date")),
                -similarity_score(bank_entry, invoice),
            ),
        )[0]
        register_match(
            matches,
            bank_entry,
            [best_invoice],
            "montant exact à départager",
            "à vérifier",
            "Plusieurs factures ont le même montant, choisissez la bonne correspondance.",
            0.58,
        )


def apply_manual_links(workspace, invoices, bank_entries, matches):
    for link in workspace.get("manual_links", []):
        bank_entry = find_bank_entry(bank_entries, link.get("bank_entry_id"))
        if not bank_entry or bank_entry.get("remaining_amount", 0) <= 0:
            continue

        target_invoices = []
        for invoice_id in link.get("invoice_ids", []):
            invoice = find_invoice(invoices, invoice_id)
            if invoice and invoice.get("outstanding_amount", 0) > 0 and currency_matches(bank_entry, invoice):
                target_invoices.append(invoice)

        if not target_invoices:
            continue

        register_match(
            matches,
            bank_entry,
            target_invoices,
            "validation manuelle",
            "manuel",
            link.get("notes") or "Correspondance manuelle créée par l'utilisateur.",
            1.0,
        )


def recalculate_workspace(workspace):
    invoices, bank_entries = initialize_runtime_records(workspace)
    matches = []

    run_reference_matches(invoices, bank_entries, matches)
    run_combination_matches(invoices, bank_entries, matches)
    run_single_amount_matches(invoices, bank_entries, matches)
    apply_manual_links(workspace, invoices, bank_entries, matches)
    finalize_records(invoices, bank_entries)

    metrics = compute_metrics(invoices, bank_entries, matches)

    workspace["invoices"] = sorted(invoices, key=lambda item: (item["status"], item["due_date"], item["invoice_number"]))
    workspace["bank_entries"] = sorted(bank_entries, key=lambda item: (item["status"], item["booking_date"], item["label"]))
    workspace["matches"] = sorted(matches, key=lambda item: (item["status"], item["booking_date"]))
    workspace["metrics"] = metrics
    workspace["last_run_at"] = utc_now_iso()
    return workspace


def add_manual_link(workspace, bank_entry_id, invoice_ids, notes=None):
    current_links = [link for link in workspace.get("manual_links", []) if link.get("bank_entry_id") != bank_entry_id]
    current_links.append(
        {
            "id": make_id("manual"),
            "bank_entry_id": bank_entry_id,
            "invoice_ids": invoice_ids,
            "notes": notes or "Association manuelle créée.",
            "created_at": utc_now_iso(),
        }
    )
    workspace["manual_links"] = current_links
    return workspace


def build_export_rows(workspace):
    rows = []
    match_by_invoice = {}
    for match in workspace.get("matches", []):
        for allocation in match.get("allocations", []):
            match_by_invoice.setdefault(allocation["invoice_id"], []).append(match)

    for invoice in workspace.get("invoices", []):
        related_matches = match_by_invoice.get(invoice["id"], [])
        rows.append(
            {
                "facture": invoice["invoice_number"],
                "client": invoice["customer_name"],
                "montant_facture": invoice["amount"],
                "devise": invoice.get("currency", "EUR"),
                "montant_rapproche": invoice["matched_amount"],
                "reste_a_encaisser": invoice["outstanding_amount"],
                "statut": invoice["status"],
                "a_verifier": "oui" if invoice.get("review_required") else "non",
                "notes_extraction": " | ".join(invoice.get("extraction_notes", [])),
                "banque": " | ".join(match["bank_label"] for match in related_matches),
                "bases_de_rapprochement": " | ".join(match["match_basis"] for match in related_matches),
            }
        )
    return rows