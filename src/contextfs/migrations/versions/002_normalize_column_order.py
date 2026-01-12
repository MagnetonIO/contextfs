"""Normalize memories column order.

Fixes column order inconsistency where source_tool and project
were added after session_id in some databases but before in the schema.

Revision ID: 002
Revises: 001
Create Date: 2024-12-15

Supports both SQLite and PostgreSQL.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def get_dialect() -> str:
    """Get the database dialect name."""
    return context.get_context().dialect.name


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    dialect = get_dialect()
    if dialect == "postgresql":
        result = conn.execute(
            sa.text("SELECT 1 FROM information_schema.tables WHERE table_name = :name"),
            {"name": table_name},
        )
    else:  # sqlite
        result = conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table_name},
        )
    return result.fetchone() is not None


def get_column_positions(conn) -> dict[str, int]:
    """Get column positions for the memories table."""
    dialect = get_dialect()
    if dialect == "postgresql":
        result = conn.execute(
            sa.text(
                "SELECT column_name, ordinal_position FROM information_schema.columns "
                "WHERE table_name = 'memories' ORDER BY ordinal_position"
            )
        )
        return {row[0]: row[1] for row in result.fetchall()}
    else:  # sqlite
        result = conn.execute(sa.text("PRAGMA table_info(memories)"))
        return {row[1]: row[0] for row in result.fetchall()}


def _ensure_fts_table(conn) -> None:
    """Ensure FTS table exists, creating it if needed. SQLite only."""
    dialect = get_dialect()
    if dialect == "postgresql":
        # PostgreSQL uses different full-text search (ts_vector)
        # Skip FTS5 for PostgreSQL - handled separately
        return

    # Check if FTS table exists (SQLite)
    if table_exists(conn, "memories_fts"):
        return  # FTS table already exists

    # Create FTS table with all required columns (SQLite only)
    conn.execute(
        sa.text("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            id UNINDEXED,
            content,
            summary,
            tags,
            type UNINDEXED,
            namespace_id UNINDEXED,
            source_repo UNINDEXED,
            source_tool UNINDEXED,
            project UNINDEXED,
            content='memories',
            content_rowid='rowid',
            tokenize='porter unicode61'
        )
    """)
    )

    # Populate FTS index from existing memories
    conn.execute(
        sa.text("""
        INSERT INTO memories_fts (id, content, summary, tags, type, namespace_id, source_repo, source_tool, project)
        SELECT id, content, summary, tags, type, namespace_id, source_repo, source_tool, project FROM memories
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

    PostgreSQL supports column reordering but this migration is primarily
    for fixing old SQLite databases, so we skip it for PostgreSQL.
    """
    conn = op.get_bind()
    dialect = get_dialect()

    # PostgreSQL: Just ensure schema is correct and skip column reordering
    if dialect == "postgresql":
        # PostgreSQL doesn't have column order issues from the initial migration
        return

    # SQLite: Check current column order
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

    # Recreate FTS table with all required columns
    # Searchable: content, summary, tags
    # Filterable (UNINDEXED): id, type, namespace_id, source_repo, source_tool, project
    op.execute("DROP TABLE IF EXISTS memories_fts")
    op.execute("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            id UNINDEXED,
            content,
            summary,
            tags,
            type UNINDEXED,
            namespace_id UNINDEXED,
            source_repo UNINDEXED,
            source_tool UNINDEXED,
            project UNINDEXED,
            content='memories',
            content_rowid='rowid',
            tokenize='porter unicode61'
        )
    """)

    # Rebuild FTS index
    op.execute("""
        INSERT INTO memories_fts (id, content, summary, tags, type, namespace_id, source_repo, source_tool, project)
        SELECT id, content, summary, tags, type, namespace_id, source_repo, source_tool, project FROM memories
    """)


def downgrade() -> None:
    """Revert to old column order (not recommended)."""
    # This is a data-preserving migration, downgrade not needed
    pass
