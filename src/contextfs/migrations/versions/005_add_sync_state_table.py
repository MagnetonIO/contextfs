"""Add sync_state table for storing sync metadata.

Moves sync state from ~/.contextfs/sync_state.json into SQLite
for better reliability and ACID transactions.

Revision ID: 005
Revises: 004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "005"
down_revision: str | None = "004"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Create sync_state table."""
    conn = op.get_bind()

    # Check if table already exists
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='sync_state'")
    )
    if result.fetchone():
        return  # Table already exists

    # Create sync_state table
    op.create_table(
        "sync_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("device_id", sa.Text(), nullable=False, unique=True),
        sa.Column("server_url", sa.Text(), nullable=True),
        sa.Column("last_sync_at", sa.DateTime(), nullable=True),
        sa.Column("last_push_at", sa.DateTime(), nullable=True),
        sa.Column("last_pull_at", sa.DateTime(), nullable=True),
        sa.Column("device_tracker", sa.Text(), default="{}"),  # JSON: {device_id: timestamp}
        sa.Column("registered_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Create index on device_id
    op.create_index("idx_sync_state_device_id", "sync_state", ["device_id"])


def downgrade() -> None:
    """Drop sync_state table."""
    op.drop_index("idx_sync_state_device_id")
    op.drop_table("sync_state")
