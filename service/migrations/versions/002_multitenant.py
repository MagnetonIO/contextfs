"""Multi-tenant isolation and auth tables.

Revision ID: 002
Revises: 001
Create Date: 2024-01-02 00:00:00.000000

Adds:
- user_id columns for multi-tenant isolation
- subscriptions, usage, password_reset_tokens tables
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add user_id to devices table
    op.add_column("devices", sa.Column("user_id", sa.Text(), nullable=True))
    op.create_index("idx_devices_user", "devices", ["user_id"])

    # Add user_id to memories table
    op.add_column("memories", sa.Column("user_id", sa.Text(), nullable=True))
    op.create_index("idx_memories_user", "memories", ["user_id"])

    # Add user_id to sessions table
    op.add_column("sessions", sa.Column("user_id", sa.Text(), nullable=True))
    op.create_index("idx_sessions_user", "sessions", ["user_id"])

    # Subscriptions table
    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("tier", sa.Text(), server_default="free"),
        sa.Column("stripe_customer_id", sa.Text()),
        sa.Column("stripe_subscription_id", sa.Text()),
        sa.Column("device_limit", sa.Integer(), server_default="3"),
        sa.Column("memory_limit", sa.Integer(), server_default="10000"),
        sa.Column("status", sa.Text(), server_default="active"),
        sa.Column("current_period_end", sa.DateTime(timezone=True)),
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
    op.create_index("idx_subscriptions_user", "subscriptions", ["user_id"])
    op.create_index("idx_subscriptions_stripe_customer", "subscriptions", ["stripe_customer_id"])

    # Usage table
    op.create_table(
        "usage",
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("device_count", sa.Integer(), server_default="0"),
        sa.Column("memory_count", sa.Integer(), server_default="0"),
        sa.Column("last_sync_at", sa.DateTime(timezone=True)),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Password reset tokens table
    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_password_reset_user", "password_reset_tokens", ["user_id"])

    # Add updated_at triggers for new tables
    for table in ["subscriptions", "usage"]:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_updated_at ON {table};")
        op.execute(f"""
            CREATE TRIGGER {table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)


def downgrade() -> None:
    # Drop triggers
    for table in ["subscriptions", "usage"]:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_updated_at ON {table};")

    # Drop tables
    op.drop_table("password_reset_tokens")
    op.drop_table("usage")
    op.drop_table("subscriptions")

    # Drop indexes and columns
    op.drop_index("idx_sessions_user", "sessions")
    op.drop_column("sessions", "user_id")

    op.drop_index("idx_memories_user", "memories")
    op.drop_column("memories", "user_id")

    op.drop_index("idx_devices_user", "devices")
    op.drop_column("devices", "user_id")
