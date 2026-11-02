"""events append only and app role

Two independent layers keep `events` append-only:
1. The app role only has SELECT/INSERT on events (and no DELETE anywhere).
2. A trigger rejects UPDATE/DELETE/TRUNCATE on events for every role, owner included.

Revision ID: e80407688ac5
Revises: e5e38b968b17
Create Date: 2026-10-24
"""

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.config import load_settings

revision: str = "e80407688ac5"
down_revision: str | None = "e5e38b968b17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ROLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _app_role() -> tuple[str, str]:
    settings = load_settings()
    if not _ROLE_NAME.match(settings.app_db_user):
        raise ValueError("APP_DB_USER must be a lowercase identifier")
    return settings.app_db_user, settings.app_db_password


def _sql_literal(value: str) -> str:
    # DDL can't take bind parameters, so quote the password literal by hand.
    return "'" + value.replace("'", "''") + "'"


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION events_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'events is append-only: % is not allowed', TG_OP
                USING ERRCODE = 'insufficient_privilege';
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER events_no_update_delete BEFORE UPDATE OR DELETE ON events "
        "FOR EACH ROW EXECUTE FUNCTION events_append_only()"
    )
    op.execute(
        "CREATE TRIGGER events_no_truncate BEFORE TRUNCATE ON events "
        "FOR EACH STATEMENT EXECUTE FUNCTION events_append_only()"
    )

    role, password = _app_role()
    exists = (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role})
        .scalar()
    )
    verb = "ALTER" if exists else "CREATE"
    op.execute(f'{verb} ROLE "{role}" LOGIN PASSWORD {_sql_literal(password)}')

    op.execute(
        f"""
        DO $$ BEGIN
            EXECUTE format('GRANT CONNECT ON DATABASE %I TO "{role}"', current_database());
        END $$
        """
    )
    op.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
    # Everything except events: read/insert/update, never delete.
    op.execute(f'GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO "{role}"')
    op.execute(f'REVOKE UPDATE ON events FROM "{role}"')
    op.execute(f'REVOKE UPDATE ON alembic_version FROM "{role}"')
    op.execute(f'REVOKE INSERT ON alembic_version FROM "{role}"')
    op.execute(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{role}"')
    # Future tables get the same default (no DELETE); restrict them in their migration if needed.
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f'GRANT SELECT, INSERT, UPDATE ON TABLES TO "{role}"'
    )
    op.execute(
        f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO "{role}"'
    )


def downgrade() -> None:
    # The role is cluster-wide (shared with the test database), so it is not dropped here.
    role, _ = _app_role()
    op.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON SEQUENCES FROM "{role}"')
    op.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM "{role}"')
    op.execute(f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM "{role}"')
    op.execute(f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "{role}"')
    op.execute(f'REVOKE USAGE ON SCHEMA public FROM "{role}"')
    op.execute(
        f"""
        DO $$ BEGIN
            EXECUTE format('REVOKE CONNECT ON DATABASE %I FROM "{role}"', current_database());
        END $$
        """
    )
    op.execute("DROP TRIGGER events_no_truncate ON events")
    op.execute("DROP TRIGGER events_no_update_delete ON events")
    op.execute("DROP FUNCTION events_append_only()")
