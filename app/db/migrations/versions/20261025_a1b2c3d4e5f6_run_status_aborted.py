"""run status aborted + status reason

Revision ID: a1b2c3d4e5f6
Revises: 769cc0d5d332
Create Date: 2026-10-25 14:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "769cc0d5d332"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD = "status IN ('running', 'completed', 'failed')"
NEW = "status IN ('running', 'completed', 'failed', 'aborted')"


def upgrade() -> None:
    op.add_column("eval_runs", sa.Column("status_reason", sa.String(length=300), nullable=True))
    op.drop_constraint(op.f("ck_eval_runs_runstatus"), "eval_runs", type_="check")
    op.create_check_constraint("runstatus", "eval_runs", NEW)


def downgrade() -> None:
    op.drop_constraint(op.f("ck_eval_runs_runstatus"), "eval_runs", type_="check")
    op.create_check_constraint("runstatus", "eval_runs", OLD)
    op.drop_column("eval_runs", "status_reason")
