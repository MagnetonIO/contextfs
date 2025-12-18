"""Add memory_edges table for graph relationships.

Enables memory lineage tracking, relationship modeling,
and graph traversal without requiring external graph database.

Revision ID: 003
Revises: 002
Create Date: 2024-12-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create memory_edges table for graph relationships."""
    conn = op.get_bind()

    # Check if memory_edges table already exists
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_edges'")
    )
    if result.fetchone() is not None:
        return  # Table already exists

    # Create memory_edges table
    # Stores directed relationships between memories
    op.create_table(
        "memory_edges",
        # Composite primary key: from_id + to_id + relation
        sa.Column("from_id", sa.Text, nullable=False),
        sa.Column("to_id", sa.Text, nullable=False),
        sa.Column("relation", sa.Text, nullable=False),
        # Edge properties
        sa.Column("weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("created_by", sa.Text),  # Tool that created the edge
        sa.Column("metadata", sa.Text),  # JSON for additional properties
        # Primary key constraint
        sa.PrimaryKeyConstraint("from_id", "to_id", "relation"),
    )

    # Create indexes for efficient queries
    # Index for outgoing edges (from_id lookups)
    op.create_index("idx_edges_from_id", "memory_edges", ["from_id"])

    # Index for incoming edges (to_id lookups)
    op.create_index("idx_edges_to_id", "memory_edges", ["to_id"])

    # Index for relation type queries
    op.create_index("idx_edges_relation", "memory_edges", ["relation"])

    # Composite index for bidirectional lookups
    op.create_index("idx_edges_from_relation", "memory_edges", ["from_id", "relation"])

    op.create_index("idx_edges_to_relation", "memory_edges", ["to_id", "relation"])

    # Index for lineage queries (evolution relationships)
    # Covers: evolved_from, merged_from, split_from
    op.execute("""
        CREATE INDEX idx_edges_lineage ON memory_edges(from_id, to_id)
        WHERE relation IN ('evolved_from', 'merged_from', 'split_from',
                          'evolved_into', 'merged_into', 'split_into')
    """)


def downgrade() -> None:
    """Drop memory_edges table."""
    op.drop_index("idx_edges_lineage")
    op.drop_index("idx_edges_to_relation")
    op.drop_index("idx_edges_from_relation")
    op.drop_index("idx_edges_relation")
    op.drop_index("idx_edges_to_id")
    op.drop_index("idx_edges_from_id")
    op.drop_table("memory_edges")
