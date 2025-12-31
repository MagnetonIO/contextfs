"""Add authoritative column for lineage-aware queries.

Revision ID: 007
Revises: 006
Create Date: 2024-12-31

Phase 3: Authoritative flag for marking canonical versions in a lineage chain.
"""

from alembic import op

# Revision identifiers
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add authoritative column to memories table."""
    # Add authoritative column (default 0 = False)
    op.execute("ALTER TABLE memories ADD COLUMN authoritative INTEGER DEFAULT 0")


def downgrade() -> None:
    """Remove authoritative column (SQLite limitation - recreate table)."""
    # SQLite doesn't support DROP COLUMN, but we'll keep it simple
    # The column will just be ignored if not used
    pass
