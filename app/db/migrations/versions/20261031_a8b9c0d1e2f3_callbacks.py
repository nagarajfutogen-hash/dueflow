"""callbacks (agent v5, V5-3): customer-requested callbacks, dialed only once the owner approves

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-10-31 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8b9c0d1e2f3"
down_revision: str | None = "f7a8b9c0d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OUTCOMES = (
    "'promise', 'handoff', 'end_wrong_person', 'third_party_callback', 'no_verification', "
    "'stop_contact', 'refuse_discount', 'no_response', 'disconnected'"
)


def upgrade() -> None:
    op.create_table(
        "callbacks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("debtor_id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=False),
        sa.Column("reason", sa.String(length=40), nullable=False),
        sa.Column("customer_words", sa.String(length=200), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "requested",
                "approved",
                "rejected",
                "done",
                name="callbackstatus",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("done_conversation_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["debtor_id"], ["debtors.id"], name=op.f("fk_callbacks_debtor_id_debtors")
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_callbacks_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["done_conversation_id"],
            ["conversations.id"],
            name=op.f("fk_callbacks_done_conversation_id_conversations"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_callbacks")),
    )
    op.create_index(op.f("ix_callbacks_debtor_id"), "callbacks", ["debtor_id"], unique=False)
    op.drop_constraint(op.f("ck_conversations_outcome"), "conversations", type_="check")
    op.create_check_constraint(
        op.f("ck_conversations_outcome"),
        "conversations",
        f"outcome IS NULL OR outcome IN ({OUTCOMES}, 'callback_scheduled')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_conversations_outcome"), "conversations", type_="check")
    op.create_check_constraint(
        op.f("ck_conversations_outcome"),
        "conversations",
        f"outcome IS NULL OR outcome IN ({OUTCOMES})",
    )
    op.drop_index(op.f("ix_callbacks_debtor_id"), table_name="callbacks")
    op.drop_table("callbacks")
