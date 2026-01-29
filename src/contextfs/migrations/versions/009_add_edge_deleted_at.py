"""Add deleted_at column to memory_edges table.

Enables soft-delete of edges when memories are soft-deleted,
aligning local SQLite schema with PostgreSQL SyncedEdgeModel.

Revision ID: 009
Revises: 008
Create Date: 2026-01-29
"""

from alembic import context, op
from sqlalchemy import inspect

# Revision identifiers
revision = "009"
down_revision = "008"
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
    """Add deleted_at column to memory_edges table."""
    if not column_exists("memory_edges", "deleted_at"):
        op.execute("ALTER TABLE memory_edges ADD COLUMN deleted_at TEXT")


def downgrade() -> None:
    """Remove deleted_at column (SQLite limitation - column will just be ignored)."""
    pass
