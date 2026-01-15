"""Ensure is_admin column exists on users table.

Revision ID: 006
Revises: 005
Create Date: 2024-01-06 00:00:00.000000

This migration ensures the is_admin column exists, even if the database
was stamped at revision 005 when the column was actually missing
(from failed old SQL migrations).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Check if is_admin column exists before adding
    conn = op.get_bind()
    result = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'users' AND column_name = 'is_admin'"
        )
    )
    if result.fetchone() is None:
        # Column doesn't exist, add it
        op.add_column(
            "users",
            sa.Column("is_admin", sa.Boolean(), server_default="false", nullable=False),
        )


def downgrade() -> None:
    # We don't remove is_admin in downgrade since it might have existed before
    pass
