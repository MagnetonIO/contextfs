"""Add devices table for tracking user devices.

Revision ID: 010
Revises: 009
Create Date: 2025-01-10

Supports both SQLite and PostgreSQL.
"""

import sqlalchemy as sa
from alembic import context, op

# Revision identifiers
revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def get_dialect() -> str:
    """Get the database dialect name."""
    return context.get_context().dialect.name


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    dialect = get_dialect()
    if dialect == "postgresql":
        result = conn.execute(
            sa.text("SELECT 1 FROM information_schema.tables WHERE table_name = :name"),
            {"name": table_name},
        )
    else:  # sqlite
        result = conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table_name},
        )
    return result.fetchone() is not None


def get_existing_columns(conn, table_name: str) -> set[str]:
    """Get existing column names for a table."""
    dialect = get_dialect()
    if dialect == "postgresql":
        result = conn.execute(
            sa.text("SELECT column_name FROM information_schema.columns WHERE table_name = :name"),
            {"name": table_name},
        )
        return {row[0] for row in result.fetchall()}
    else:  # sqlite
        result = conn.execute(sa.text(f"PRAGMA table_info({table_name})"))
        return {row[1] for row in result.fetchall()}


def upgrade() -> None:
    """Create devices table or add auth columns to existing sync devices table."""
    dialect = get_dialect()
    conn = op.get_bind()

    # Default timestamp expression based on dialect
    default_now = "NOW()" if dialect == "postgresql" else "datetime('now')"

    # Check if devices table already exists (from sync server)
    if table_exists(conn, "devices"):
        existing_columns = get_existing_columns(conn, "devices")
        if "user_id" not in existing_columns:
            op.execute("ALTER TABLE devices ADD COLUMN user_id TEXT REFERENCES users(id)")
        return

    # Create devices table for fresh database
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            name TEXT NOT NULL,
            device_type TEXT,
            os TEXT,
            os_version TEXT,
            last_sync_at TEXT,
            created_at TEXT NOT NULL DEFAULT ({default_now}),
            UNIQUE(user_id, name)
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_devices_user ON devices(user_id)")


def downgrade() -> None:
    """Drop devices table."""
    op.execute("DROP TABLE IF EXISTS devices")
