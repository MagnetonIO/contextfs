"""Initial schema - baseline for existing databases.

Revision ID: 001
Revises: None
Create Date: 2024-12-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial schema if tables don't exist."""
    conn = op.get_bind()

    # Check if memories table exists
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
    )
    if result.fetchone() is not None:
        # Tables exist, this is an existing database
        return

    # Create memories table
    op.create_table(
        "memories",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("tags", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("namespace_id", sa.Text, nullable=False),
        sa.Column("source_file", sa.Text),
        sa.Column("source_repo", sa.Text),
        sa.Column("source_tool", sa.Text),
        sa.Column("project", sa.Text),
        sa.Column("session_id", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
    )

    # Create sessions table
    op.create_table(
        "sessions",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("label", sa.Text),
        sa.Column("namespace_id", sa.Text, nullable=False),
        sa.Column("tool", sa.Text, nullable=False),
        sa.Column("repo_path", sa.Text),
        sa.Column("branch", sa.Text),
        sa.Column("started_at", sa.Text, nullable=False),
        sa.Column("ended_at", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("metadata", sa.Text),
    )

    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("session_id", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("timestamp", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
    )

    # Create namespaces table
    op.create_table(
        "namespaces",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("parent_id", sa.Text),
        sa.Column("repo_path", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
    )

    # Create indexes
    op.create_index("idx_memories_namespace", "memories", ["namespace_id"])
    op.create_index("idx_memories_type", "memories", ["type"])
    op.create_index("idx_sessions_namespace", "sessions", ["namespace_id"])
    op.create_index("idx_sessions_label", "sessions", ["label"])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index("idx_sessions_label")
    op.drop_index("idx_sessions_namespace")
    op.drop_index("idx_memories_type")
    op.drop_index("idx_memories_namespace")
    op.drop_table("namespaces")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("memories")
