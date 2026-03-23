import csv
import io
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from reconciliation_engine import (
    add_activity,
    add_manual_link,
    build_demo_workspace,
    build_empty_workspace,
    build_export_rows,
    currency_matches,
    parse_dataset,
    recalculate_workspace,
    utc_now_iso,
)
from pdf_extraction_service import extract_rows_from_pdf


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


WORKSPACE_COLLECTION = "reconciliation_workspaces"


class ImportedSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    dataset: str
    file_name: str
    rows_count: int
    imported_at: str


class ActivityItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    kind: str
    message: str
    created_at: str


class InvoiceRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    invoice_number: str
    customer_name: str
    issue_date: str
    due_date: str
    amount: float
    matched_amount: float = 0.0
    outstanding_amount: float = 0.0
    currency: str = "EUR"
    status: str
    review_required: bool = False
    source: str = "import"
    match_notes: List[str] = Field(default_factory=list)
    extraction_notes: List[str] = Field(default_factory=list)


class BankEntryRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    booking_date: str
    label: str
    reference: str = ""
    amount: float
    signed_amount: float
    remaining_amount: float = 0.0
    currency: str = "EUR"
    direction: Literal["credit", "debit"]
    status: str
    review_required: bool = False
    source: str = "import"
    extraction_notes: List[str] = Field(default_factory=list)


class MatchAllocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    invoice_id: str
    invoice_number: str
    customer_name: str
    applied_amount: float
    currency: str = "EUR"


class ReconciliationMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    bank_entry_id: str
    bank_label: str
    bank_reference: str = ""
    booking_date: str
    invoice_ids: List[str]
    invoice_numbers: List[str]
    allocations: List[MatchAllocation]
    bank_amount: float
    applied_amount: float
    difference: float
    currency: str = "EUR"
    match_basis: str
    status: str
    confidence: float
    note: str


class WorkspaceMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_invoices: int
    total_bank_entries: int
    matched_invoices: int
    partially_matched_invoices: int
    unmatched_invoices: int
    to_review: int
    matched_amount: float
    outstanding_amount: float
    matched_amounts_by_currency: dict[str, float] = Field(default_factory=dict)
    outstanding_amounts_by_currency: dict[str, float] = Field(default_factory=dict)


class ReconciliationWorkspace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    workspace_name: str
    imported_sources: List[ImportedSource] = Field(default_factory=list)
    activity: List[ActivityItem] = Field(default_factory=list)
    invoices: List[InvoiceRecord] = Field(default_factory=list)
    bank_entries: List[BankEntryRecord] = Field(default_factory=list)
    matches: List[ReconciliationMatch] = Field(default_factory=list)
    metrics: WorkspaceMetrics
    last_run_at: Optional[str] = None


class ManualMatchInput(BaseModel):
    bank_entry_id: str
    invoice_ids: List[str]
    notes: Optional[str] = None


def find_matching_record(records, record_id):
    return next((record for record in records if record.get("id") == record_id), None)


async def get_workspace_document():
    existing = await db[WORKSPACE_COLLECTION].find_one({"id": "default-workspace"}, {"_id": 0})
    if existing:
        return existing

    workspace = recalculate_workspace(build_empty_workspace())
    await db[WORKSPACE_COLLECTION].replace_one({"id": workspace["id"]}, workspace, upsert=True)
    return workspace


async def save_workspace_document(workspace):
    workspace["updated_at"] = utc_now_iso()
    await db[WORKSPACE_COLLECTION].replace_one({"id": workspace["id"]}, workspace, upsert=True)
    stored = await db[WORKSPACE_COLLECTION].find_one({"id": workspace["id"]}, {"_id": 0})
    return stored

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "API de rapprochement factures / relevés disponible."}


@api_router.get("/reconciliation/workspace", response_model=ReconciliationWorkspace)
async def get_workspace():
    workspace = await get_workspace_document()
    return workspace


@api_router.post("/reconciliation/demo", response_model=ReconciliationWorkspace)
async def load_demo_workspace():
    workspace = build_demo_workspace()
    stored = await save_workspace_document(workspace)
    return stored


@api_router.post("/reconciliation/reset", response_model=ReconciliationWorkspace)
async def reset_workspace():
    workspace = recalculate_workspace(build_empty_workspace())
    add_activity(workspace, "reset", "Atelier vidé. Vous pouvez réimporter vos fichiers.")
    stored = await save_workspace_document(workspace)
    return stored



@api_router.delete("/reconciliation/source/{source_id}", response_model=ReconciliationWorkspace)
async def delete_source(source_id: str):
    workspace = await get_workspace_document()
    # Find the source to delete
    source = next((s for s in workspace.get("imported_sources", []) if s["id"] == source_id), None)
    if not source:
        raise HTTPException(status_code=404, detail="Source non trouv�e.")
    dataset = source["dataset"]
    # Remove the source
    workspace["imported_sources"] = [s for s in workspace["imported_sources"] if s["id"] != source_id]
    # Remove entries from this source
    if dataset == "bank":
        workspace["bank_entries"] = [e for e in workspace.get("bank_entries", []) if e.get("source_id") != source_id]
    elif dataset == "invoices":
        workspace["invoices"] = [i for i in workspace.get("invoices", []) if i.get("source_id") != source_id]
    # Recalculate and save
    workspace = recalculate_workspace(workspace)
    add_activity(workspace, "delete", f"Source {source['file_name']} supprim�e.")
    stored = await save_workspace_document(workspace)
    return stored

@api_router.post("/reconciliation/import/{dataset}", response_model=ReconciliationWorkspace)
async def import_dataset(dataset: Literal["invoices", "bank"], file: UploadFile = File(...)):
    workspace = await get_workspace_document()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Le fichier est vide.")

    try:
        filename = file.filename or "import.csv"
        if filename.lower().endswith(".pdf"):
            parsed_rows, extraction_summary = await extract_rows_from_pdf(dataset, filename, content)
        else:
            parsed_rows = parse_dataset(filename, content, dataset)
            extraction_summary = None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Import failure")
        raise HTTPException(status_code=400, detail="Le fichier n'a pas pu être lu.") from exc

    target_key = "invoices_raw" if dataset == "invoices" else "bank_entries_raw"
    workspace[target_key] = [*workspace.get(target_key, []), *parsed_rows]
    workspace["imported_sources"] = [
        {
            "id": f"src-{len(workspace.get('imported_sources', [])) + 1}-{dataset}",
            "dataset": dataset,
            "file_name": file.filename or "fichier",
            "rows_count": len(parsed_rows),
            "imported_at": utc_now_iso(),
        },
        *workspace.get("imported_sources", []),
    ][:10]
    add_activity(
        workspace,
        "import",
        (
            f"{len(parsed_rows)} lignes importées dans {('factures' if dataset == 'invoices' else 'relevé bancaire')} depuis {file.filename}."
            + (f" {extraction_summary}" if extraction_summary else "")
        ),
    )
    updated_workspace = recalculate_workspace(workspace)
    stored = await save_workspace_document(updated_workspace)
    return stored


@api_router.post("/reconciliation/run", response_model=ReconciliationWorkspace)
async def run_reconciliation():
    workspace = await get_workspace_document()
    updated_workspace = recalculate_workspace(workspace)
    metrics = updated_workspace.get("metrics", {})
    add_activity(
        updated_workspace,
        "run",
        f"Rapprochement exécuté : {metrics.get('matched_invoices', 0)} factures soldées, {metrics.get('to_review', 0)} correspondances à vérifier.",
    )
    stored = await save_workspace_document(updated_workspace)
    return stored


@api_router.post("/reconciliation/manual-match", response_model=ReconciliationWorkspace)
async def create_manual_match(payload: ManualMatchInput):
    if not payload.invoice_ids:
        raise HTTPException(status_code=400, detail="Sélectionnez au moins une facture.")

    workspace = await get_workspace_document()
    bank_entry = find_matching_record(workspace.get("bank_entries", []), payload.bank_entry_id)
    if not bank_entry:
        raise HTTPException(status_code=400, detail="Le virement sélectionné est introuvable.")
    if bank_entry.get("direction") != "credit" or bank_entry.get("remaining_amount", 0) <= 0:
        raise HTTPException(status_code=400, detail="Le virement sélectionné n'a plus de montant à affecter.")

    open_invoices = {
        invoice["id"]: invoice
        for invoice in workspace.get("invoices", [])
        if invoice.get("outstanding_amount", 0) > 0
    }
    unique_invoice_ids = list(dict.fromkeys(payload.invoice_ids))
    invalid_invoice_ids = [invoice_id for invoice_id in unique_invoice_ids if invoice_id not in open_invoices]
    if invalid_invoice_ids:
        raise HTTPException(
            status_code=400,
            detail="Certaines factures sélectionnées sont introuvables ou déjà soldées.",
        )

    if any(not currency_matches(bank_entry, open_invoices[invoice_id]) for invoice_id in unique_invoice_ids):
        raise HTTPException(
            status_code=400,
            detail="Le virement et les factures sélectionnées doivent être dans la même devise.",
        )

    workspace = add_manual_link(workspace, payload.bank_entry_id, unique_invoice_ids, payload.notes)
    add_activity(workspace, "manual", "Association manuelle ajoutée au rapprochement.")
    updated_workspace = recalculate_workspace(workspace)
    stored = await save_workspace_document(updated_workspace)
    return stored


@api_router.get("/reconciliation/export.csv")
async def export_reconciliation_csv():
    workspace = await get_workspace_document()
    rows = build_export_rows(workspace)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()) if rows else ["facture", "statut"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = 'attachment; filename="rapprochement.csv"'
    return response

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()