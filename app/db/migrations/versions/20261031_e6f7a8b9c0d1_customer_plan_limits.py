"""customer plan limits (agent v5, rules-v8)

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-10-31 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: str | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Per-customer limits, stricter than the global plan rules; NULL = the global rule applies.
    op.add_column("debtors", sa.Column("min_monthly_jpy", sa.Integer(), nullable=True))
    op.add_column("debtors", sa.Column("max_months", sa.Integer(), nullable=True))
    op.create_check_constraint(
        op.f("ck_debtors_min_monthly_positive"), "debtors", "min_monthly_jpy > 0"
    )
    op.create_check_constraint(op.f("ck_debtors_max_months_positive"), "debtors", "max_months > 0")


def downgrade() -> None:
    op.drop_constraint(op.f("ck_debtors_max_months_positive"), "debtors", type_="check")
    op.drop_constraint(op.f("ck_debtors_min_monthly_positive"), "debtors", type_="check")
    op.drop_column("debtors", "max_months")
    op.drop_column("debtors", "min_monthly_jpy")
