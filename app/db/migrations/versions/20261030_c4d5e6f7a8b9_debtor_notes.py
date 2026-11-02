"""debtor notes (agent v4 customer context)

Revision ID: c4d5e6f7a8b9
Revises: b7c8d9e0f1a2
Create Date: 2026-10-30 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: str | None = "b7c8d9e0f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Seed debtors already in a database get the same notes as app/db/seed.py.
SEED_NOTES = {
    "D-001": "Usually pays on time; missed this month. Prefers quick, to-the-point calls.",
    "D-002": "Has said money is tight since losing hours at work.",
    "D-003": "Gets irritated by collection calls; has been short-tempered before.",
    "D-004": "Often vague about dates and avoids committing.",
    "D-005": "Agreed a plan before and stopped after the first payment.",
    "D-006": "Believes the loan was already repaid.",
    "D-007": "Asked not to be contacted.",
    "D-008": "Lives in New York; no calls at work. Has asked about discounts before.",
    "D-009": "Only reachable at work; keep it discreet.",
}


def upgrade() -> None:
    op.add_column("debtors", sa.Column("notes", sa.Text(), nullable=True))
    debtors = sa.table("debtors", sa.column("external_ref", sa.String), sa.column("notes", sa.Text))
    for ref, notes in SEED_NOTES.items():
        op.execute(
            debtors.update()
            .where(debtors.c.external_ref == ref, debtors.c.notes.is_(None))
            .values(notes=notes)
        )


def downgrade() -> None:
    op.drop_column("debtors", "notes")
