"""Backfill team_id on memories and sessions for existing team members.

Revision ID: 010
Revises: 009
Create Date: 2026-01-29

Memories created before a user joined a team (or before multi-tenant support)
have team_id = NULL. This migration backfills team_id and sets visibility to
'team_read' so that linked memories are accessible to teammates.
"""

from alembic import op
from sqlalchemy import text

# Revision identifiers
revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Backfill team_id on memories and sessions from team_members."""
    conn = op.get_bind()

    # Backfill memories: set team_id from team_members where NULL
    conn.execute(
        text("""
            UPDATE memories m
            SET team_id = tm.team_id, visibility = 'team_read'
            FROM team_members tm
            WHERE m.user_id = tm.user_id AND m.team_id IS NULL
        """)
    )

    # Backfill sessions: set team_id from team_members where NULL
    conn.execute(
        text("""
            UPDATE sessions s
            SET team_id = tm.team_id
            FROM team_members tm
            WHERE s.user_id = tm.user_id AND s.team_id IS NULL
        """)
    )


def downgrade() -> None:
    """No-op: backfilled data is harmless, but we clear it for reversibility."""
    conn = op.get_bind()

    # Clear team_id that was backfilled (only where visibility was set by us)
    # Note: This is best-effort; memories explicitly set to team_read by users
    # cannot be distinguished from backfilled ones.
    conn.execute(
        text("""
            UPDATE memories
            SET team_id = NULL, visibility = 'private'
            WHERE visibility = 'team_read' AND team_id IS NOT NULL
        """)
    )

    conn.execute(
        text("""
            UPDATE sessions
            SET team_id = NULL
            WHERE team_id IS NOT NULL
        """)
    )
