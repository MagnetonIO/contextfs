"""Add session tracking fields for device_name and memories_created.

Revision ID: 008
Revises: 007
Create Date: 2026-01-19

Adds:
- device_name to sessions table
- memories_created (JSON array) to sessions table
"""

from alembic import context, op
from sqlalchemy import inspect

# Revision identifiers
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def get_dialect() -> str:
    """Get the database dialect name."""
    return context.get_context().dialect.name


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Add device_name and memories_created columns to sessions table."""
    dialect = get_dialect()

    # Add device_name column
    if not column_exists("sessions", "device_name"):
        op.execute("ALTER TABLE sessions ADD COLUMN device_name TEXT")

    # Add memories_created column (JSON array of memory IDs)
    if not column_exists("sessions", "memories_created"):
        if dialect == "postgresql":
            op.execute("ALTER TABLE sessions ADD COLUMN memories_created JSONB DEFAULT '[]'")
        else:  # sqlite
            op.execute("ALTER TABLE sessions ADD COLUMN memories_created TEXT DEFAULT '[]'")


def downgrade() -> None:
    """Remove columns (SQLite limitation - columns will just be ignored)."""
    pass
