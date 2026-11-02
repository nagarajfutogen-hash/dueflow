"""handoff reason below_account_floor (agent v5)

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-10-31 13:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f7a8b9c0d1e2"
down_revision: str | None = "e6f7a8b9c0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD = "reason IN ('hardship', 'dispute', 'unresolved', 'error')"
NEW = "reason IN ('hardship', 'dispute', 'unresolved', 'error', 'below_account_floor')"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_handoffs_handoffreason"), "handoffs", type_="check")
    op.create_check_constraint(op.f("ck_handoffs_handoffreason"), "handoffs", NEW)


def downgrade() -> None:
    op.drop_constraint(op.f("ck_handoffs_handoffreason"), "handoffs", type_="check")
    op.create_check_constraint(op.f("ck_handoffs_handoffreason"), "handoffs", OLD)
