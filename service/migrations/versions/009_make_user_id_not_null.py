"""Make user_id NOT NULL on memories and sessions.

Revision ID: 009
Revises: 008
Create Date: 2026-01-28

Backfills orphaned records (pre-multi-tenant with NULL user_id) to a
fallback admin user, then adds NOT NULL constraints to prevent future NULLs.
"""

from alembic import op
from sqlalchemy import text

# Revision identifiers
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None

# Fallback user for orphaned records
ADMIN_USER_ID = "admin-00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    """Backfill NULL user_id records and add NOT NULL constraint."""
    conn = op.get_bind()

    # Backfill orphaned memories
    conn.execute(
        text("UPDATE memories SET user_id = :uid WHERE user_id IS NULL"),
        {"uid": ADMIN_USER_ID},
    )

    # Backfill orphaned sessions
    conn.execute(
        text("UPDATE sessions SET user_id = :uid WHERE user_id IS NULL"),
        {"uid": ADMIN_USER_ID},
    )

    # Add NOT NULL constraint
    op.alter_column("memories", "user_id", nullable=False)
    op.alter_column("sessions", "user_id", nullable=False)


def downgrade() -> None:
    """Remove NOT NULL constraint (data stays backfilled)."""
    op.alter_column("memories", "user_id", nullable=True)
    op.alter_column("sessions", "user_id", nullable=True)
