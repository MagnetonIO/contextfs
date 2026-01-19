"""Add session tracking fields.

Revision ID: 007
Revises: 006
Create Date: 2026-01-19

Adds:
- device_name to sessions table
- memories_created (JSONB array) to sessions table
"""

from alembic import op
from sqlalchemy import inspect, text

# Revision identifiers
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Add device_name and memories_created columns to sessions table."""
    conn = op.get_bind()

    # Add device_name column
    if not column_exists("sessions", "device_name"):
        conn.execute(text("ALTER TABLE sessions ADD COLUMN device_name TEXT"))

    # Add memories_created column (JSONB array of memory IDs)
    if not column_exists("sessions", "memories_created"):
        conn.execute(text("ALTER TABLE sessions ADD COLUMN memories_created JSONB DEFAULT '[]'"))


def downgrade() -> None:
    """Remove columns."""
    conn = op.get_bind()
    conn.execute(text("ALTER TABLE sessions DROP COLUMN IF EXISTS device_name"))
    conn.execute(text("ALTER TABLE sessions DROP COLUMN IF EXISTS memories_created"))
