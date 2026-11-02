"""Synthetic debtors for local development and scenarios. Names and numbers are fake.

Run: docker compose exec api python -m app.db.seed  (idempotent: skips existing refs)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import load_settings
from app.db.models import (
    ContactPrefs,
    Debtor,
    Dispute,
    DisputeStatus,
    Payment,
    Promise,
    PromiseStatus,
)
from app.db.session import make_engine


@dataclass(frozen=True)
class SeedPromise:
    months: int
    down_payment_jpy: int
    installment_jpy: int
    first_due_date: date
    status: PromiseStatus


@dataclass(frozen=True)
class SeedPayment:
    amount_jpy: int
    paid_on: date
    method: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class SeedDebtor:
    ref: str
    persona: str
    full_name: str
    dob: date
    balance_jpy: int
    timezone: str = "Asia/Tokyo"
    mobile: str | None = "090-0000-0000"
    home: str | None = None
    work: str | None = None
    stop_contact: bool = False
    no_workplace_contact: bool = False
    workplace_justification: str | None = None
    open_dispute: str | None = None
    promises: tuple[SeedPromise, ...] = ()
    payments_jpy: tuple[int, ...] = field(default=())
    notes: str | None = None
    payment_records: tuple[SeedPayment, ...] = ()
    min_monthly_jpy: int | None = None  # agent v5: customer plan limits (None = global rule)
    max_months: int | None = None


SEED_DEBTORS: tuple[SeedDebtor, ...] = (
    SeedDebtor(
        "D-001",
        "cooperative",
        "Taro Example",
        date(1985, 4, 12),
        120_000,
        notes="Usually pays on time; missed this month. Prefers quick, to-the-point calls.",
    ),
    SeedDebtor(
        "D-002",
        "hardship",
        "Hanako Sample",
        date(1979, 11, 3),
        300_000,
        notes="Has said money is tight since losing hours at work.",
    ),
    SeedDebtor(
        "D-003",
        "angry",
        "Jiro Testcase",
        date(1990, 1, 25),
        85_000,
        home="03-0000-0001",
        notes="Gets irritated by collection calls; has been short-tempered before.",
    ),
    SeedDebtor(
        "D-004",
        "evasive",
        "Yuki Placeholder",
        date(1995, 7, 30),
        60_000,
        notes="Often vague about dates and avoids committing.",
    ),
    SeedDebtor(
        "D-005",
        "returning_broken_promise",
        "Kenji Mockname",
        date(1982, 3, 18),
        200_000,
        promises=(SeedPromise(3, 20_000, 60_000, date(2026, 8, 1), PromiseStatus.BROKEN),),
        payments_jpy=(20_000,),
        notes="Agreed a plan before and stopped after the first payment.",
    ),
    SeedDebtor(
        "D-006",
        "returning_open_dispute",
        "Aiko Dummy",
        date(1988, 9, 9),
        150_000,
        open_dispute="Debtor says the loan was already repaid in 2025.",
        notes="Believes the loan was already repaid.",
    ),
    SeedDebtor(
        "D-007",
        "returning_stop_contact",
        "Hiro Fictional",
        date(1975, 12, 1),
        90_000,
        stop_contact=True,
        notes="Asked not to be contacted.",
    ),
    SeedDebtor(
        "D-008",
        "non_jst_no_workplace",
        "Mei Synthetic",
        date(1992, 5, 21),
        110_000,
        timezone="America/New_York",
        mobile="+1-555-0100",
        work="+1-555-0199",
        no_workplace_contact=True,
        notes="Lives in New York; no calls at work. Has asked about discounts before.",
    ),
    SeedDebtor(
        "D-009",
        "workplace_justified",
        "Sota Pretend",
        date(1987, 2, 14),
        70_000,
        mobile=None,
        work="03-0000-0009",
        workplace_justification="No personal number on file; home number disconnected.",
        notes="Only reachable at work; keep it discreet.",
    ),
    # Payment-claim tests (agent v4): a partial payment, a claim with nothing on file yet,
    # and two payments that do not cover the balance.
    SeedDebtor(
        "D-010",
        "payment_claim_partial",
        "Aya Receipt",
        date(1989, 6, 2),
        150_000,
        notes="Says they made a payment recently.",
        payment_records=(SeedPayment(30_000, date(2026, 10, 10), "bank_transfer", "TX-48213"),),
    ),
    SeedDebtor(
        "D-011",
        "payment_claim_pending",
        "Ken Pending",
        date(1993, 9, 14),
        90_000,
        notes="Says they paid at a convenience store last week.",
    ),
    SeedDebtor(
        "D-012",
        "payment_claim_full",
        "Mika Twopay",
        date(1981, 12, 5),
        60_000,
        notes="Often says the account is fully paid.",
        payment_records=(
            SeedPayment(40_000, date(2026, 9, 5), "card", "CARD-7781"),
            SeedPayment(20_000, date(2026, 10, 5), "convenience_store", "CVS-55120"),
        ),
    ),
    # V5-5: per-customer plan limits (agent v5 only): at most 4 months, at least ¥30,000 a month.
    SeedDebtor(
        "D-013",
        "customer_limits",
        "Rin Limitcase",
        date(1986, 8, 19),
        180_000,
        notes="Account has its own plan limits on file.",
        min_monthly_jpy=30_000,
        max_months=4,
    ),
    # Voice testing (owner, 2026-11-03): synthetic person, not real data.
    SeedDebtor(
        "D-014",
        "voice_test",
        "Arun Kumar",
        date(2005, 1, 1),
        80_000,
        notes="New account; first call.",
    ),
)

SEEDED_AT = datetime(2026, 1, 1, tzinfo=UTC)


async def insert_debtor(conn: AsyncConnection, d: SeedDebtor) -> int:
    """Insert one synthetic debtor with prefs, dispute, promises and payments."""
    debtor_id: int = (
        await conn.execute(
            insert(Debtor)
            .values(
                external_ref=d.ref,
                full_name=d.full_name,
                date_of_birth=d.dob,
                phone_home=d.home,
                phone_mobile=d.mobile,
                phone_work=d.work,
                timezone=d.timezone,
                balance_jpy=d.balance_jpy,
                notes=d.notes,
                min_monthly_jpy=d.min_monthly_jpy,
                max_months=d.max_months,
            )
            .returning(Debtor.id)
        )
    ).scalar_one()
    await conn.execute(
        insert(ContactPrefs).values(
            debtor_id=debtor_id,
            stop_contact=d.stop_contact,
            stop_contact_at=SEEDED_AT if d.stop_contact else None,
            no_workplace_contact=d.no_workplace_contact,
            workplace_justification=d.workplace_justification,
        )
    )
    if d.open_dispute:
        await conn.execute(
            insert(Dispute).values(
                debtor_id=debtor_id,
                status=DisputeStatus.OPEN,
                reason=d.open_dispute,
                opened_at=SEEDED_AT,
            )
        )
    for p in d.promises:
        await conn.execute(
            insert(Promise).values(
                debtor_id=debtor_id,
                months=p.months,
                down_payment_jpy=p.down_payment_jpy,
                installment_jpy=p.installment_jpy,
                first_due_date=p.first_due_date,
                status=p.status,
                status_changed_at=SEEDED_AT,
            )
        )
    for amount in d.payments_jpy:
        await conn.execute(
            insert(Payment).values(debtor_id=debtor_id, amount_jpy=amount, paid_at=SEEDED_AT)
        )
    zone = ZoneInfo(d.timezone)
    for pay in d.payment_records:
        await conn.execute(
            insert(Payment).values(
                debtor_id=debtor_id,
                amount_jpy=pay.amount_jpy,
                paid_at=datetime.combine(pay.paid_on, time(12), tzinfo=zone),
                method=pay.method,
                reference=pay.reference,
            )
        )
    return debtor_id


async def seed(conn: AsyncConnection) -> list[str]:
    """Insert missing seed debtors. Returns the refs that were inserted."""
    existing = set((await conn.execute(select(Debtor.external_ref))).scalars())
    inserted: list[str] = []
    for d in SEED_DEBTORS:
        if d.ref not in existing:
            await insert_debtor(conn, d)
            inserted.append(d.ref)
    return inserted


async def _main() -> None:
    engine = make_engine(load_settings().owner_url)
    async with engine.begin() as conn:
        inserted = await seed(conn)
    await engine.dispose()
    print(f"seeded {len(inserted)} debtor(s): {', '.join(inserted) or 'none (already present)'}")


if __name__ == "__main__":
    asyncio.run(_main())
