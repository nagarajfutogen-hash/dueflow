"""Debtor card (dev page context) and CSV import of test debtors.

Import only adds new refs (existing refs are skipped, never changed). The whole file is
validated first: one bad row and nothing is inserted. Rows are never logged (PII).
"""

import csv
import io
from datetime import date
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.agent.actions import payments_on_file
from app.calls import account_snapshot, is_tester_debtor
from app.db.models import ContactPrefs, Debtor
from app.db.seed import SEED_DEBTORS, SeedDebtor, insert_debtor
from app.policy.config import PolicyRules

MAX_IMPORT_ROWS = 500
MAX_NOTES_CHARS = 500
CSV_COLUMNS = (
    "ref",
    "full_name",
    "dob",
    "balance_jpy",
    "timezone",
    "mobile",
    "home",
    "work",
    "stop_contact",
    "no_workplace_contact",
    "workplace_justification",
    "open_dispute",
    "notes",
    "min_monthly_jpy",
    "max_months",
)


async def debtor_refs(conn: AsyncConnection) -> list[str]:
    refs = (await conn.execute(select(Debtor.external_ref).order_by(Debtor.id))).scalars()
    return [r for r in refs if is_tester_debtor(r)]


async def debtor_card(conn: AsyncConnection, ref: str) -> dict[str, Any] | None:
    """What a tester needs to play this debtor: identity, account and contact facts."""
    orm = AsyncSession(bind=conn)
    debtor = await orm.scalar(select(Debtor).where(Debtor.external_ref == ref))
    if debtor is None:
        return None
    prefs = await orm.scalar(select(ContactPrefs).where(ContactPrefs.debtor_id == debtor.id))
    return {
        "ref": debtor.external_ref,
        # Seed debtors carry the case they were built to test; imported ones have none.
        "persona": next((d.persona for d in SEED_DEBTORS if d.ref == ref), None),
        "full_name": debtor.full_name,
        "dob": debtor.date_of_birth.isoformat(),
        "timezone": debtor.timezone,
        "phones": [
            kind
            for kind, number in (
                ("mobile", debtor.phone_mobile),
                ("home", debtor.phone_home),
                ("work", debtor.phone_work),
            )
            if number
        ],
        "stop_contact": bool(prefs and prefs.stop_contact),
        "no_workplace_contact": bool(prefs and prefs.no_workplace_contact),
        "notes": debtor.notes,
        "payments": [
            {
                "amount_jpy": p.amount_jpy,
                "paid_on": p.paid_on.isoformat(),
                "method": p.method,
                "reference": p.reference,
            }
            for p in await payments_on_file(conn, debtor.id, debtor.timezone)
        ],
        **await account_snapshot(orm, debtor),
    }


def _zone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {value!r}") from exc
    return value


def _blank_none(value: str | None) -> str | None:
    return (value.strip() or None) if value else None


class DebtorRow(BaseModel):
    ref: str = Field(pattern=r"^[A-Za-z0-9-]{1,32}$")
    full_name: str = Field(min_length=1, max_length=200)
    dob: date
    balance_jpy: int = Field(ge=0)
    timezone: Annotated[str, AfterValidator(_zone)] = "Asia/Tokyo"
    mobile: str | None = Field(default=None, max_length=32)
    home: str | None = Field(default=None, max_length=32)
    work: str | None = Field(default=None, max_length=32)
    stop_contact: bool = False
    no_workplace_contact: bool = False
    workplace_justification: str | None = None
    open_dispute: str | None = None
    notes: str | None = Field(default=None, max_length=MAX_NOTES_CHARS)
    # Agent v5 (rules-v8): this customer's own plan limits; empty = the global rule.
    min_monthly_jpy: int | None = Field(default=None, ge=1)
    max_months: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _has_number(self) -> "DebtorRow":
        if not (self.mobile or self.home or self.work):
            raise ValueError("needs at least one of mobile, home, work")
        return self


class CsvImportError(ValueError):
    """Bad CSV: messages say which row and field (no values, they may be PII)."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def parse_csv(text: str, *, rules: PolicyRules) -> list[DebtorRow]:
    """All rows valid, or CsvImportError listing every problem by row (never values).
    Customer plan limits may only be stricter than the global plan rules."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = [h.strip() for h in reader.fieldnames or []]
    errors: list[str] = []
    missing = [c for c in ("ref", "full_name", "dob", "balance_jpy") if c not in header]
    unknown = [h for h in header if h not in CSV_COLUMNS]
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown columns: {', '.join(unknown)}")
    if errors:
        raise CsvImportError(errors)
    rows: list[DebtorRow] = []
    refs: set[str] = set()
    for line, raw in enumerate(reader, start=2):
        if len(rows) >= MAX_IMPORT_ROWS:
            raise CsvImportError([f"more than {MAX_IMPORT_ROWS} rows"])
        values = {k.strip(): _blank_none(v) for k, v in raw.items() if k}
        try:
            row = DebtorRow.model_validate({k: v for k, v in values.items() if v is not None})
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, e["loc"])) or "row" for e in exc.errors())
            errors.append(f"row {line}: invalid {fields}")
            continue
        plan = rules.plan
        if row.max_months is not None and not plan.min_months <= row.max_months <= plan.max_months:
            errors.append(
                f"row {line}: max_months must be {plan.min_months}-{plan.max_months} "
                "(stricter than the global rule, and at least the shortest plan)"
            )
        if row.ref in refs:
            errors.append(f"row {line}: duplicate ref in file")
        refs.add(row.ref)
        rows.append(row)
    if errors:
        raise CsvImportError(errors)
    if not rows:
        raise CsvImportError(["no rows"])
    return rows


async def import_debtors(conn: AsyncConnection, rows: list[DebtorRow]) -> dict[str, list[str]]:
    existing = set((await conn.execute(select(Debtor.external_ref))).scalars())
    added: list[str] = []
    skipped: list[str] = []
    for r in rows:
        if r.ref in existing:
            skipped.append(r.ref)
            continue
        await insert_debtor(
            conn,
            SeedDebtor(
                ref=r.ref,
                persona="imported",
                full_name=r.full_name,
                dob=r.dob,
                balance_jpy=r.balance_jpy,
                timezone=r.timezone,
                mobile=r.mobile,
                home=r.home,
                work=r.work,
                stop_contact=r.stop_contact,
                no_workplace_contact=r.no_workplace_contact,
                workplace_justification=r.workplace_justification,
                open_dispute=r.open_dispute,
                notes=r.notes,
                min_monthly_jpy=r.min_monthly_jpy,
                max_months=r.max_months,
            ),
        )
        added.append(r.ref)
    return {"added": added, "skipped": skipped}
