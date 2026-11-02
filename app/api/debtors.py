from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.auth import require_token
from app.debtors import CsvImportError, debtor_card, debtor_refs, import_debtors, parse_csv

router = APIRouter(dependencies=[Depends(require_token)])

MAX_CSV_BYTES = 512_000


@router.get("/debtors")
async def list_debtors(request: Request) -> list[str]:
    async with request.app.state.engine.connect() as conn:
        return await debtor_refs(conn)


@router.get("/debtors/{ref}")
async def get_debtor(ref: str, request: Request) -> dict[str, Any]:
    """Dev page context: who the tester is playing (synthetic data, token-protected)."""
    async with request.app.state.engine.connect() as conn:
        card = await debtor_card(conn, ref)
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown debtor")
    return card


@router.post("/debtors/import")
async def import_csv(request: Request) -> dict[str, list[str]]:
    """Body: the CSV file as text/csv. Adds new refs only; all rows valid or nothing added."""
    raw = await request.body()
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "CSV too large")
    try:
        rows = parse_csv(raw.decode("utf-8"), rules=request.app.state.rules)
    except UnicodeDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "CSV must be UTF-8") from exc
    except CsvImportError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, exc.errors) from exc
    async with request.app.state.engine.begin() as conn:
        return await import_debtors(conn, rows)
