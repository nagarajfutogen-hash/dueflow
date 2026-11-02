"""replays (Failure Lab)

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-10-27 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b7c8d9e0f1a2"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The app role gets SELECT/INSERT/UPDATE through the default privileges (e80407688ac5).
    op.create_table(
        "replays",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("scenario_id", sa.String(length=128), nullable=False),
        sa.Column("version_a", sa.String(length=64), nullable=False),
        sa.Column("version_b", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "completed",
                "failed",
                name="replaystatus",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),  # fmt: skip
        sa.Column("run_a_id", sa.Integer(), nullable=True),
        sa.Column("run_b_id", sa.Integer(), nullable=True),
        sa.Column(
            "estimate",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("report", sa.Text(), nullable=True),
        sa.Column("error", sa.String(length=300), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_a_id"], ["eval_runs.id"], name=op.f("fk_replays_run_a_id_eval_runs")
        ),
        sa.ForeignKeyConstraint(
            ["run_b_id"], ["eval_runs.id"], name=op.f("fk_replays_run_b_id_eval_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_replays")),
    )
    op.create_index(
        "uq_replays_one_running",
        "replays",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_index("uq_replays_one_running", table_name="replays")
    op.drop_table("replays")
