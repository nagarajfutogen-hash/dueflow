"""payment method and reference (agent v4 payment checks)

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-10-30 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: str | None = "c4d5e6f7a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("payments", sa.Column("method", sa.String(length=32), nullable=True))
    op.add_column("payments", sa.Column("reference", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("payments", "reference")
    op.drop_column("payments", "method")
