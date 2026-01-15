"""Teams feature for Team tier collaboration.

Revision ID: 003
Revises: 002
Create Date: 2024-01-03 00:00:00.000000

Creates:
- teams, team_members, team_invitations tables
- Adds team columns to memories, sessions, subscriptions
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
    # Teams table
    op.create_table(
        "teams",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "owner_id",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("description", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_teams_owner", "teams", ["owner_id"])

    # Team members table
    op.create_table(
        "team_members",
        sa.Column(
            "team_id",
            sa.Text(),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False, server_default="member"),
        sa.Column("invited_by", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("team_id", "user_id"),
    )
    op.create_index("idx_team_members_user", "team_members", ["user_id"])
    op.create_index("idx_team_members_team", "team_members", ["team_id"])

    # Team invitations table
    op.create_table(
        "team_invitations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "team_id",
            sa.Text(),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False, server_default="member"),
        sa.Column(
            "invited_by",
            sa.Text(),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_team_invitations_team", "team_invitations", ["team_id"])
    op.create_index("idx_team_invitations_email", "team_invitations", ["email"])

    # Add team columns to memories table
    op.add_column(
        "memories",
        sa.Column("owner_id", sa.Text(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column("team_id", sa.Text(), sa.ForeignKey("teams.id"), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column("visibility", sa.Text(), server_default="private"),
    )
    op.create_index("idx_memories_owner", "memories", ["owner_id"])
    op.create_index("idx_memories_team", "memories", ["team_id"])
    op.create_index("idx_memories_visibility", "memories", ["visibility"])

    # Add team columns to sessions table
    op.add_column(
        "sessions",
        sa.Column("owner_id", sa.Text(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("team_id", sa.Text(), sa.ForeignKey("teams.id"), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("visibility", sa.Text(), server_default="private"),
    )
    op.create_index("idx_sessions_owner", "sessions", ["owner_id"])
    op.create_index("idx_sessions_team", "sessions", ["team_id"])

    # Add team columns to subscriptions table
    op.add_column(
        "subscriptions",
        sa.Column("team_id", sa.Text(), sa.ForeignKey("teams.id"), nullable=True),
    )
    op.add_column(
        "subscriptions",
        sa.Column("seats_included", sa.Integer(), server_default="1"),
    )
    op.add_column(
        "subscriptions",
        sa.Column("seats_used", sa.Integer(), server_default="1"),
    )

    # Add updated_at trigger for teams table
    op.execute("DROP TRIGGER IF EXISTS teams_updated_at ON teams;")
    op.execute("""
        CREATE TRIGGER teams_updated_at
            BEFORE UPDATE ON teams
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    """)

    # Migrate existing data: Set owner_id from user_id where missing
    op.execute(
        "UPDATE memories SET owner_id = user_id " "WHERE owner_id IS NULL AND user_id IS NOT NULL;"
    )
    op.execute(
        "UPDATE sessions SET owner_id = user_id " "WHERE owner_id IS NULL AND user_id IS NOT NULL;"
    )


def downgrade() -> None:
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS teams_updated_at ON teams;")

    # Drop subscription columns
    op.drop_column("subscriptions", "seats_used")
    op.drop_column("subscriptions", "seats_included")
    op.drop_column("subscriptions", "team_id")

    # Drop session columns
    op.drop_index("idx_sessions_team", "sessions")
    op.drop_index("idx_sessions_owner", "sessions")
    op.drop_column("sessions", "visibility")
    op.drop_column("sessions", "team_id")
    op.drop_column("sessions", "owner_id")

    # Drop memory columns
    op.drop_index("idx_memories_visibility", "memories")
    op.drop_index("idx_memories_team", "memories")
    op.drop_index("idx_memories_owner", "memories")
    op.drop_column("memories", "visibility")
    op.drop_column("memories", "team_id")
    op.drop_column("memories", "owner_id")

    # Drop tables
    op.drop_table("team_invitations")
    op.drop_table("team_members")
    op.drop_table("teams")
