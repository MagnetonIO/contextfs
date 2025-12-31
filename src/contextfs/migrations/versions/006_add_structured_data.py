"""Add structured_data column to memories table.

Supports typed memory with JSON schema validation per type.
The structured_data column stores optional structured content
that is validated against TYPE_SCHEMAS in schemas.py.

Revision ID: 006
Revises: 005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "006"
down_revision: str | None = "005"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Add structured_data column to memories table."""
    conn = op.get_bind()

    # Check if column already exists
    result = conn.execute(sa.text("PRAGMA table_info(memories)"))
    existing_columns = {row[1] for row in result.fetchall()}

    if "structured_data" not in existing_columns:
        try:
            op.add_column("memories", sa.Column("structured_data", sa.Text()))
        except Exception:
            pass  # Column might already exist


def downgrade() -> None:
    """Remove structured_data column (not recommended - data loss)."""
    # SQLite doesn't support DROP COLUMN easily in older versions
    # For safety, we'll leave the column in place
    pass
