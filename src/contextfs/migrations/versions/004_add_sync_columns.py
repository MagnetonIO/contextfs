"""Add sync columns to memories and sessions tables.

Adds vector_clock, content_hash, deleted_at, and last_modified_by
columns to support multi-device synchronization.

Also adds portable path columns (repo_url, repo_name, relative_path)
for cross-machine path resolution.

Revision ID: 004
Revises: 003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "004"
down_revision: str | None = "003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Add sync columns to memories and sessions tables."""
    conn = op.get_bind()

    # Check which columns already exist in memories table
    result = conn.execute(sa.text("PRAGMA table_info(memories)"))
    existing_columns = {row[1] for row in result.fetchall()}

    # Add sync columns to memories table
    sync_columns = [
        ("vector_clock", "TEXT DEFAULT '{}'"),
        ("content_hash", "TEXT"),
        ("deleted_at", "TIMESTAMP"),
        ("last_modified_by", "TEXT"),
        # Portable path columns
        ("repo_url", "TEXT"),
        ("repo_name", "TEXT"),
        ("relative_path", "TEXT"),
    ]

    for col_name, col_def in sync_columns:
        if col_name not in existing_columns:
            try:
                op.add_column("memories", sa.Column(col_name, sa.Text()))
                # Set default for vector_clock
                if col_name == "vector_clock":
                    conn.execute(
                        sa.text(
                            "UPDATE memories SET vector_clock = '{}' WHERE vector_clock IS NULL"
                        )
                    )
            except Exception:
                pass  # Column might already exist

    # Check sessions table
    result = conn.execute(sa.text("PRAGMA table_info(sessions)"))
    existing_session_columns = {row[1] for row in result.fetchall()}

    # Add sync columns to sessions table
    session_sync_columns = [
        ("vector_clock", "TEXT DEFAULT '{}'"),
        ("content_hash", "TEXT"),
        ("deleted_at", "TIMESTAMP"),
        ("last_modified_by", "TEXT"),
        # Portable path columns
        ("repo_url", "TEXT"),
        ("repo_name", "TEXT"),
    ]

    for col_name, col_def in session_sync_columns:
        if col_name not in existing_session_columns:
            try:
                op.add_column("sessions", sa.Column(col_name, sa.Text()))
                if col_name == "vector_clock":
                    conn.execute(
                        sa.text(
                            "UPDATE sessions SET vector_clock = '{}' WHERE vector_clock IS NULL"
                        )
                    )
            except Exception:
                pass

    # Add sync columns to memory_edges table if it exists
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_edges'")
    )
    if result.fetchone():
        result = conn.execute(sa.text("PRAGMA table_info(memory_edges)"))
        existing_edge_columns = {row[1] for row in result.fetchall()}

        edge_sync_columns = [
            ("vector_clock", "TEXT DEFAULT '{}'"),
            ("deleted_at", "TIMESTAMP"),
            ("last_modified_by", "TEXT"),
        ]

        for col_name, col_def in edge_sync_columns:
            if col_name not in existing_edge_columns:
                try:
                    op.add_column("memory_edges", sa.Column(col_name, sa.Text()))
                    if col_name == "vector_clock":
                        conn.execute(
                            sa.text(
                                "UPDATE memory_edges SET vector_clock = '{}' WHERE vector_clock IS NULL"
                            )
                        )
                except Exception:
                    pass

    # Create indexes for sync queries
    try:
        op.create_index(
            "idx_memories_deleted_at",
            "memories",
            ["deleted_at"],
            if_not_exists=True,
        )
    except Exception:
        pass

    try:
        op.create_index(
            "idx_memories_repo_url",
            "memories",
            ["repo_url"],
            if_not_exists=True,
        )
    except Exception:
        pass

    try:
        op.create_index(
            "idx_sessions_deleted_at",
            "sessions",
            ["deleted_at"],
            if_not_exists=True,
        )
    except Exception:
        pass


def downgrade() -> None:
    """Remove sync columns (not recommended - data loss)."""
    # SQLite doesn't support DROP COLUMN easily
    # We'd need to recreate the tables, which is destructive
    # For safety, we'll leave the columns in place
    pass
