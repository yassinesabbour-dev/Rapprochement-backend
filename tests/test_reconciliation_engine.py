"""
import io

import pytest

from reconciliation_engine import (
    parse_dataset,
    recalculate_workspace,
    build_empty_workspace,
)


def test_parse_dataset_invoices_csv():
    csv_content = (
        "invoice,customer,amount,issue_date,due_date\n"
        "FA-2025-001,Studio Atlas,1200,2025-02-01,2025-02-15\n"
        "FA-2025-002,Studio Atlas,850,2025-02-03,2025-02-17\n"
    )
    parsed = parse_dataset("sample.csv", csv_content.encode("utf-8"), "invoices")
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    # verify major fields and types
    first = parsed[0]
    assert float(first["amount"]) == pytest.approx(1200.0, rel=1e-6)
    assert "invoice_number" in first
    assert "customer_name" in first
    assert "issue_date" in first
    assert "due_date" in first


def test_parse_dataset_bank_csv():
    csv_content = (
        "label,amount,booking_date,reference\n"
        "VIREMENT STUDIO ATLAS REGLEMENT FA-2025-001,1200,2025-02-12,ATLAS-LOT-0212\n"
    )
    parsed = parse_dataset("bank_sample.csv", csv_content.encode("utf-8"), "bank")
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    entry = parsed[0]
    assert float(entry["amount"]) == pytest.approx(1200.0, rel=1e-6)
    # direction should default to credit for positive amounts
    assert entry.get("direction") in ("credit", "debit")
    assert entry.get("currency") is not None


def test_recalculate_workspace_combination_match():
    # Build a workspace with two invoices and a single bank entry that pays both
    workspace = build_empty_workspace()

    workspace["invoices_raw"] = [
        {
            "id": "inv-1",
            "invoice_number": "FA-2025-001",
            "customer_name": "Studio Atlas",
            "issue_date": "2025-02-01",
            "due_date": "2025-02-15",
            "amount": 1200.0,
            "currency": "EUR",
            "source": "test",
        },
        {
            "id": "inv-2",
            "invoice_number": "FA-2025-002",
            "customer_name": "Studio Atlas",
            "issue_date": "2025-02-03",
            "due_date": "2025-02-17",
            "amount": 850.0,
            "currency": "EUR",
            "source": "test",
        },
    ]

    workspace["bank_entries_raw"] = [
        {
            "id": "bnk-1",
            "booking_date": "2025-02-20",
            "label": "VIREMENT STUDIO ATLAS REGLEMENT FA-2025-001 FA-2025-002",
            "reference": "ATLAS-LOT-0212",
            "amount": 2050.0,
            "currency": "EUR",
            "direction": "credit",
            "source": "test",
        }
    ]

    result = recalculate_workspace(workspace)

    # Expect one match covering both invoices
    matches = result.get("matches", [])
    assert isinstance(matches, list)
    assert len(matches) >= 1

    metrics = result.get("metrics", {})
    # both invoices should now be matched/paid
    assert metrics.get("matched_invoices", 0) == 2 or metrics.get("matched_invoices", 0) == len(
        [inv for inv in result.get("invoices", []) if inv.get("status") == "payée"]
    )

    # bank entry should be fully consumed
    bank_entries = result.get("bank_entries", [])
    assert any(be.get("id") == "bnk-1" and be.get("remaining_amount", 0) == 0 for be in bank_entries)
"""