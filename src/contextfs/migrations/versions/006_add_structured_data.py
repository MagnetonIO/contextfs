"""Add structured_data column to memories table.

Supports typed memory with JSON schema validation per type.
The structured_data column stores optional structured content
that is validated against TYPE_SCHEMAS in schemas.py.

Revision ID: 006
Revises: 005

Supports both SQLite and PostgreSQL.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

# revision identifiers
revision: str = "006"
down_revision: str | None = "005"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def get_dialect() -> str:
    """Get the database dialect name."""
    return context.get_context().dialect.name


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
    """Add structured_data column to memories table."""
    conn = op.get_bind()

    # Check if column already exists
    existing_columns = get_existing_columns(conn, "memories")

    if "structured_data" not in existing_columns:
        op.add_column("memories", sa.Column("structured_data", sa.Text()))


def downgrade() -> None:
    """Remove structured_data column (not recommended - data loss)."""
    # SQLite doesn't support DROP COLUMN easily in older versions
    # For safety, we'll leave the column in place
    pass
