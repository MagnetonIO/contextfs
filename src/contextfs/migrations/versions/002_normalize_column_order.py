"""Normalize memories column order.

Fixes column order inconsistency where source_tool and project
were added after session_id in some databases but before in the schema.

Revision ID: 002
Revises: 001
Create Date: 2024-12-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ensure_fts_table(conn) -> None:
    """Ensure FTS table exists, creating it if needed."""
    # Check if FTS table exists
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'")
    )
    if result.fetchone() is not None:
        return  # FTS table already exists

    # Create FTS table
    conn.execute(
        sa.text("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            id, content, summary, tags,
            content='memories',
            content_rowid='rowid'
        )
    """)
    )

    # Populate FTS index from existing memories
    conn.execute(
        sa.text("""
        INSERT INTO memories_fts (id, content, summary, tags)
        SELECT id, content, summary, tags FROM memories
    """)
    )


def upgrade() -> None:
    """
    Normalize column order by recreating the memories table.

    SQLite doesn't support column reordering, so we need to:
    1. Create new table with correct order
    2. Copy data
    3. Drop old table
    4. Rename new table
    """
    conn = op.get_bind()

    # Check current column order
    result = conn.execute(sa.text("PRAGMA table_info(memories)"))
    columns = {row[1]: row[0] for row in result.fetchall()}

    # If source_tool is already before session_id, no migration needed
    # Expected order: source_tool (8), project (9), session_id (10)
    # Old order: session_id (8), ..., source_tool (12), project (13)
    if columns.get("source_tool", 99) < columns.get("session_id", 0):
        # Column order is correct, but still need to ensure FTS table exists
        _ensure_fts_table(conn)
        return

    # Check if columns exist (might be old database)
    has_source_tool = "source_tool" in columns
    has_project = "project" in columns

    # Create new table with correct column order
    op.execute("""
        CREATE TABLE memories_new (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            tags TEXT,
            summary TEXT,
            namespace_id TEXT NOT NULL,
            source_file TEXT,
            source_repo TEXT,
            source_tool TEXT,
            project TEXT,
            session_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT
        )
    """)

    # Build column list for copy
    if has_source_tool and has_project:
        # All columns exist, just reorder
        op.execute("""
            INSERT INTO memories_new
            SELECT id, content, type, tags, summary, namespace_id,
                   source_file, source_repo, source_tool, project,
                   session_id, created_at, updated_at, metadata
            FROM memories
        """)
    else:
        # Old schema, add NULL for missing columns
        op.execute("""
            INSERT INTO memories_new
            SELECT id, content, type, tags, summary, namespace_id,
                   source_file, source_repo, NULL, NULL,
                   session_id, created_at, updated_at, metadata
            FROM memories
        """)

    # Drop old table and rename
    op.drop_index("idx_memories_namespace")
    op.drop_index("idx_memories_type")
    op.drop_table("memories")
    op.rename_table("memories_new", "memories")

    # Recreate indexes
    op.create_index("idx_memories_namespace", "memories", ["namespace_id"])
    op.create_index("idx_memories_type", "memories", ["type"])

    # Recreate FTS table
    op.execute("DROP TABLE IF EXISTS memories_fts")
    op.execute("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            id, content, summary, tags,
            content='memories',
            content_rowid='rowid'
        )
    """)

    # Rebuild FTS index
    op.execute("""
        INSERT INTO memories_fts (id, content, summary, tags)
        SELECT id, content, summary, tags FROM memories
    """)


def downgrade() -> None:
    """Revert to old column order (not recommended)."""
    # This is a data-preserving migration, downgrade not needed
    pass
