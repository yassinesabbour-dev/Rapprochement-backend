"""API regression tests for reconciliation workspace, demo, run, manual-match and export."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import requests
from dotenv import load_dotenv
from fpdf import FPDF


# Ensure REACT_APP_BACKEND_URL is loaded from frontend env for public-endpoint testing.
load_dotenv(Path(__file__).resolve().parents[2] / "frontend" / ".env")

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL")
if not BASE_URL:
    raise RuntimeError("REACT_APP_BACKEND_URL is required to run API tests")

API_BASE = f"{BASE_URL.rstrip('/')}/api"


@pytest.fixture
def api_client():
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


def upload_csv(api_client, dataset: str, filename: str, content: str):
    files = {"file": (filename, content.encode("utf-8"), "text/csv")}
    response = api_client.post(f"{API_BASE}/reconciliation/import/{dataset}", files=files, timeout=20)
    return response


def upload_pdf(api_client, dataset: str, filename: str, text_lines: list[str]):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text_lines:
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")

    with NamedTemporaryFile(suffix=".pdf") as temp_file:
        pdf.output(temp_file.name)
        with open(temp_file.name, "rb") as handle:
            files = {"file": (filename, handle.read(), "application/pdf")}
        response = api_client.post(f"{API_BASE}/reconciliation/import/{dataset}", files=files, timeout=60)
    return response


def upload_pdf_fixture(api_client, dataset: str, fixture_path: Path):
    with fixture_path.open("rb") as handle:
        files = {"file": (fixture_path.name, handle.read(), "application/pdf")}
    return api_client.post(f"{API_BASE}/reconciliation/import/{dataset}", files=files, timeout=90)


# Core workspace and demo flow endpoints
def test_workspace_endpoint_returns_expected_structure(api_client):
    response = api_client.get(f"{API_BASE}/reconciliation/workspace", timeout=20)
    assert response.status_code == 200

    data = response.json()
    for key in ["id", "workspace_name", "invoices", "bank_entries", "matches", "metrics"]:
        assert key in data
    assert isinstance(data["invoices"], list)
    assert isinstance(data["bank_entries"], list)
    assert isinstance(data["matches"], list)
    assert isinstance(data["metrics"], dict)


# Demo and reconciliation run behaviors
def test_demo_endpoint_loads_expected_metrics(api_client):
    response = api_client.post(f"{API_BASE}/reconciliation/demo", timeout=20)
    assert response.status_code == 200

    data = response.json()
    metrics = data["metrics"]
    assert data["workspace_name"] == "Démonstration multi-factures"
    assert metrics["total_invoices"] == 6
    assert metrics["total_bank_entries"] == 5
    assert metrics["matched_invoices"] == 3
    assert metrics["partially_matched_invoices"] == 1
    assert metrics["unmatched_invoices"] == 2
    assert metrics["to_review"] == 3

    unresolved_credits = [
        entry for entry in data["bank_entries"] if entry["direction"] == "credit" and entry["remaining_amount"] > 0
    ]
    assert len(unresolved_credits) >= 1


def test_run_endpoint_updates_workspace_state(api_client):
    demo_response = api_client.post(f"{API_BASE}/reconciliation/demo", timeout=20)
    assert demo_response.status_code == 200

    run_response = api_client.post(f"{API_BASE}/reconciliation/run", timeout=20)
    assert run_response.status_code == 200

    data = run_response.json()
    assert data["last_run_at"]
    assert isinstance(data["metrics"]["matched_amount"], (int, float))


# Manual multi-invoice matching behavior with setup data via import endpoints
def test_manual_match_multi_invoice_persists_and_creates_manual_match(api_client):
    reset_response = api_client.post(f"{API_BASE}/reconciliation/reset", timeout=20)
    assert reset_response.status_code == 200

    invoices_csv = """invoice_number,customer_name,issue_date,due_date,amount
TEST-INV-001,Client Alpha,2025-02-01,2025-02-10,300
TEST-INV-002,Client Alpha,2025-02-02,2025-02-11,200
TEST-INV-003,Client Beta,2025-02-03,2025-02-12,150
"""
    bank_csv = """booking_date,label,amount,reference
2025-02-15,VIREMENT GROUPE CLIENT ALPHA,480,TEST-GRP-480
"""

    invoices_import = upload_csv(api_client, "invoices", "test_invoices.csv", invoices_csv)
    assert invoices_import.status_code == 200
    bank_import = upload_csv(api_client, "bank", "test_bank.csv", bank_csv)
    assert bank_import.status_code == 200

    workspace_response = api_client.get(f"{API_BASE}/reconciliation/workspace", timeout=20)
    assert workspace_response.status_code == 200
    workspace = workspace_response.json()

    target_bank = next((b for b in workspace["bank_entries"] if b["direction"] == "credit"), None)
    assert target_bank is not None
    open_invoices = [inv for inv in workspace["invoices"] if inv["outstanding_amount"] > 0]
    assert len(open_invoices) >= 2

    manual_payload = {
        "bank_entry_id": target_bank["id"],
        "invoice_ids": [open_invoices[0]["id"], open_invoices[1]["id"]],
        "notes": "TEST manual match multi-factures",
    }
    manual_response = api_client.post(f"{API_BASE}/reconciliation/manual-match", json=manual_payload, timeout=20)
    assert manual_response.status_code == 200
    manual_data = manual_response.json()

    manual_matches = [
        m
        for m in manual_data["matches"]
        if m["bank_entry_id"] == manual_payload["bank_entry_id"] and m["status"] == "manuel"
    ]
    assert len(manual_matches) >= 1
    assert len(manual_matches[0]["invoice_ids"]) >= 2
    assert "TEST manual match" in manual_matches[0]["note"]

    # GET verification for persistence
    persisted = api_client.get(f"{API_BASE}/reconciliation/workspace", timeout=20)
    assert persisted.status_code == 200
    persisted_data = persisted.json()
    persisted_manual_matches = [
        m
        for m in persisted_data["matches"]
        if m["bank_entry_id"] == manual_payload["bank_entry_id"] and m["status"] == "manuel"
    ]
    assert len(persisted_manual_matches) >= 1


def test_manual_match_rejects_invalid_ids(api_client):
    demo_response = api_client.post(f"{API_BASE}/reconciliation/demo", timeout=20)
    assert demo_response.status_code == 200

    manual_payload = {
        "bank_entry_id": "bank-invalid",
        "invoice_ids": ["invoice-invalid"],
        "notes": "invalid payload",
    }
    manual_response = api_client.post(f"{API_BASE}/reconciliation/manual-match", json=manual_payload, timeout=20)
    assert manual_response.status_code == 400
    assert "introuvable" in manual_response.json()["detail"]


def test_iso_dates_are_preserved_on_import(api_client):
    reset_response = api_client.post(f"{API_BASE}/reconciliation/reset", timeout=20)
    assert reset_response.status_code == 200

    invoices_csv = """invoice_number,customer_name,issue_date,due_date,amount
DATE-INV-001,Client Date,2025-02-03,2025-02-12,320
"""

    invoices_import = upload_csv(api_client, "invoices", "iso_dates.csv", invoices_csv)
    assert invoices_import.status_code == 200

    imported_invoice = next(
        invoice for invoice in invoices_import.json()["invoices"] if invoice["invoice_number"] == "DATE-INV-001"
    )
    assert imported_invoice["issue_date"] == "2025-02-03"
    assert imported_invoice["due_date"] == "2025-02-12"


def test_text_pdf_import_extracts_mad_invoice_and_bank_entry(api_client):
    reset_response = api_client.post(f"{API_BASE}/reconciliation/reset", timeout=20)
    assert reset_response.status_code == 200

    invoice_response = upload_pdf(
        api_client,
        "invoices",
        "facture_test_mad.pdf",
        [
            "FACTURE",
            "Numero facture: PDF-MAD-001",
            "Client: Rabat Services",
            "Date facture: 2025-05-01",
            "Date echeance: 2025-05-15",
            "Montant TTC: 3100 MAD",
        ],
    )
    assert invoice_response.status_code == 200

    bank_response = upload_pdf(
        api_client,
        "bank",
        "releve_test_mad.pdf",
        [
            "RELEVE BANCAIRE",
            "Date operation: 2025-05-10",
            "Libelle: VIREMENT RABAT SERVICES PDF-MAD-001",
            "Reference: BANK-MAD-001",
            "Montant credit: 3100 MAD",
        ],
    )
    assert bank_response.status_code == 200

    workspace = bank_response.json()
    assert workspace["invoices"][0]["currency"] == "MAD"
    assert workspace["bank_entries"][0]["currency"] == "MAD"
    assert workspace["matches"][0]["currency"] == "MAD"
    assert workspace["metrics"]["matched_amounts_by_currency"]["MAD"] == 3100.0


# CSV export endpoint validation
def test_export_csv_returns_downloadable_content(api_client):
    demo_response = api_client.post(f"{API_BASE}/reconciliation/demo", timeout=20)
    assert demo_response.status_code == 200

    export_response = api_client.get(f"{API_BASE}/reconciliation/export.csv", timeout=20)
    assert export_response.status_code == 200

    content_type = export_response.headers.get("content-type", "")
    disposition = export_response.headers.get("content-disposition", "")
    csv_text = export_response.text

    assert "text/csv" in content_type
    assert "attachment" in disposition
    assert "facture" in csv_text
    assert "statut" in csv_text
    assert "FA-2025-001" in csv_text


# Persistent fixture PDF flows (text + scanned behavior)
def test_text_pdf_fixtures_import_and_reconcile_mad(api_client):
    reset_response = api_client.post(f"{API_BASE}/reconciliation/reset", timeout=20)
    assert reset_response.status_code == 200

    fixtures_dir = Path(__file__).resolve().parents[2] / "tests" / "data"
    invoice_pdf = fixtures_dir / "facture_text_mad.pdf"
    bank_pdf = fixtures_dir / "releve_text_mad.pdf"

    invoice_response = upload_pdf_fixture(api_client, "invoices", invoice_pdf)
    assert invoice_response.status_code == 200
    bank_response = upload_pdf_fixture(api_client, "bank", bank_pdf)
    assert bank_response.status_code == 200

    workspace = bank_response.json()
    assert workspace["metrics"]["total_invoices"] >= 1
    assert workspace["metrics"]["total_bank_entries"] >= 1
    assert workspace["metrics"]["matched_amounts_by_currency"].get("MAD", 0) > 0

    invoice = next((item for item in workspace["invoices"] if item["currency"] == "MAD"), None)
    bank_entry = next((item for item in workspace["bank_entries"] if item["currency"] == "MAD"), None)
    assert invoice is not None
    assert bank_entry is not None
    assert invoice["invoice_number"]
    assert invoice["issue_date"]
    assert invoice["amount"] > 0


def test_scanned_invoice_pdf_returns_data_or_explicit_credit_error(api_client):
    reset_response = api_client.post(f"{API_BASE}/reconciliation/reset", timeout=20)
    assert reset_response.status_code == 200

    scan_pdf = Path(__file__).resolve().parents[2] / "tests" / "data" / "facture_scan_mad.pdf"
    response = upload_pdf_fixture(api_client, "invoices", scan_pdf)

    if response.status_code == 200:
        payload = response.json()
        imported = payload.get("imported_sources", [])
        assert imported, "Aucun historique d'import après succès API"
        assert imported[0]["rows_count"] > 0, "Succès silencieux détecté: 0 ligne importée"
    else:
        assert response.status_code == 400
        detail = (response.json().get("detail") or "").lower()
        assert (
            "crédit" in detail
            or "credit" in detail
            or "aucune donnée exploitable" in detail
            or "n'a pas pu être lu" in detail
        )
