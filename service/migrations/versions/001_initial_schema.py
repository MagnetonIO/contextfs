"""Initial schema for sync service.

Revision ID: 001
Revises: None
Create Date: 2024-01-01 00:00:00.000000

Creates all base tables:
- devices, sync_state
- memories, sessions, memory_edges, messages
- users, api_keys
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable required PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm";')
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')

    # Devices table
    op.create_table(
        "devices",
        sa.Column("device_id", sa.Text(), primary_key=True),
        sa.Column("device_name", sa.Text(), nullable=False),
        sa.Column("platform", sa.Text(), nullable=False),
        sa.Column("client_version", sa.Text(), nullable=False),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_sync_at", sa.DateTime(timezone=True)),
        sa.Column("sync_cursor", sa.DateTime(timezone=True)),
        sa.Column("metadata", JSONB, server_default="{}"),
    )

    # Sync state table
    op.create_table(
        "sync_state",
        sa.Column("device_id", sa.Text(), primary_key=True),
        sa.Column("last_push_at", sa.DateTime(timezone=True)),
        sa.Column("last_pull_at", sa.DateTime(timezone=True)),
        sa.Column("push_cursor", sa.DateTime(timezone=True)),
        sa.Column("pull_cursor", sa.DateTime(timezone=True)),
        sa.Column("total_pushed", sa.Integer(), server_default="0"),
        sa.Column("total_pulled", sa.Integer(), server_default="0"),
        sa.Column("total_conflicts", sa.Integer(), server_default="0"),
    )

    # Users table (created before memories/sessions for foreign keys)
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("email", sa.Text(), unique=True, nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column("provider", sa.Text(), nullable=False, server_default="system"),
        sa.Column("provider_id", sa.Text()),
        sa.Column("password_hash", sa.Text()),
        sa.Column("email_verified", sa.Boolean(), server_default="false"),
        sa.Column("verification_token", sa.Text()),
        sa.Column("verification_token_expires", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_login", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # Memories table
    op.create_table(
        "memories",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False, server_default="fact"),
        sa.Column("tags", sa.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("summary", sa.Text()),
        sa.Column("namespace_id", sa.Text(), nullable=False, server_default="global"),
        # Portable source reference
        sa.Column("repo_url", sa.Text()),
        sa.Column("repo_name", sa.Text()),
        sa.Column("relative_path", sa.Text()),
        # Legacy source fields
        sa.Column("source_file", sa.Text()),
        sa.Column("source_repo", sa.Text()),
        sa.Column("source_tool", sa.Text()),
        # Context
        sa.Column("project", sa.Text()),
        sa.Column("session_id", sa.Text()),
        # Timestamps
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
        # Sync fields
        sa.Column("vector_clock", JSONB, server_default="{}"),
        sa.Column("content_hash", sa.Text()),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("last_modified_by", sa.Text()),
        # Metadata
        sa.Column("metadata", JSONB, server_default="{}"),
    )
    # Memory indexes
    op.create_index("idx_memories_namespace", "memories", ["namespace_id"])
    op.create_index("idx_memories_type", "memories", ["type"])
    op.create_index("idx_memories_updated", "memories", ["updated_at"])
    op.create_index("idx_memories_deleted", "memories", ["deleted_at"])
    op.create_index("idx_memories_content_hash", "memories", ["content_hash"])
    op.create_index("idx_memories_repo_url", "memories", ["repo_url"])
    op.create_index("idx_memories_tags", "memories", ["tags"], postgresql_using="gin")

    # Add embedding column using raw SQL (pgvector type)
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding vector(384);")

    # Full-text search index
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON memories "
        "USING GIN(to_tsvector('english', content));"
    )

    # Sessions table
    op.create_table(
        "sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("label", sa.Text()),
        sa.Column("namespace_id", sa.Text(), nullable=False, server_default="global"),
        sa.Column("tool", sa.Text(), nullable=False, server_default="contextfs"),
        # Portable repo reference
        sa.Column("repo_url", sa.Text()),
        sa.Column("repo_name", sa.Text()),
        # Legacy field
        sa.Column("repo_path", sa.Text()),
        sa.Column("branch", sa.Text()),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("summary", sa.Text()),
        sa.Column("metadata", JSONB, server_default="{}"),
        # Timestamps
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
        # Sync fields
        sa.Column("vector_clock", JSONB, server_default="{}"),
        sa.Column("content_hash", sa.Text()),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("last_modified_by", sa.Text()),
    )
    op.create_index("idx_sessions_namespace", "sessions", ["namespace_id"])
    op.create_index("idx_sessions_updated", "sessions", ["updated_at"])
    op.create_index("idx_sessions_deleted", "sessions", ["deleted_at"])

    # Memory edges table
    op.create_table(
        "memory_edges",
        sa.Column("from_id", sa.Text(), nullable=False),
        sa.Column("to_id", sa.Text(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), server_default="1.0"),
        sa.Column("created_by", sa.Text()),
        sa.Column("metadata", JSONB, server_default="{}"),
        # Timestamps
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
        # Sync fields
        sa.Column("vector_clock", JSONB, server_default="{}"),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("last_modified_by", sa.Text()),
        sa.PrimaryKeyConstraint("from_id", "to_id", "relation"),
    )
    op.create_index("idx_edges_from", "memory_edges", ["from_id"])
    op.create_index("idx_edges_to", "memory_edges", ["to_id"])
    op.create_index("idx_edges_updated", "memory_edges", ["updated_at"])

    # Messages table
    op.create_table(
        "messages",
        sa.Column(
            "id",
            sa.Text(),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()::TEXT"),
        ),
        sa.Column(
            "session_id",
            sa.Text(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("metadata", JSONB, server_default="{}"),
        # Sync fields
        sa.Column("vector_clock", JSONB, server_default="{}"),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("last_modified_by", sa.Text()),
    )
    op.create_index("idx_messages_session", "messages", ["session_id"])
    op.create_index("idx_messages_timestamp", "messages", ["timestamp"])

    # API keys table
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("encryption_salt", sa.Text()),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_api_keys_user", "api_keys", ["user_id"])
    op.create_index("idx_api_keys_hash", "api_keys", ["key_hash"])

    # Create updated_at trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Apply triggers to tables
    for table in ["memories", "sessions", "memory_edges"]:
        op.execute(f"""
            CREATE TRIGGER {table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)


def downgrade() -> None:
    # Drop triggers
    for table in ["memories", "sessions", "memory_edges"]:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_updated_at ON {table};")

    op.execute("DROP FUNCTION IF EXISTS update_updated_at();")

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table("api_keys")
    op.drop_table("messages")
    op.drop_table("memory_edges")
    op.drop_table("sessions")
    op.drop_table("memories")
    op.drop_table("users")
    op.drop_table("sync_state")
    op.drop_table("devices")
