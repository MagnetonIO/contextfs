"""Add authentication and billing tables for commercial platform.

Revision ID: 008
Revises: 007
Create Date: 2025-01-09

Commercial Platform Phase 1: User authentication, API keys, subscriptions, and usage tracking.
Supports both SQLite and PostgreSQL.
"""

from alembic import context, op

# Revision identifiers
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def get_dialect() -> str:
    """Get the database dialect name."""
    return context.get_context().dialect.name


def upgrade() -> None:
    """Add auth and billing tables."""
    dialect = get_dialect()

    # Default timestamp expression based on dialect
    if dialect == "postgresql":
        default_now = "NOW()"
        bool_type = "BOOLEAN"
        bool_default = "true"
    else:  # sqlite
        default_now = "datetime('now')"
        bool_type = "INTEGER"
        bool_default = "1"

    # Users table
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            provider TEXT NOT NULL,
            provider_id TEXT,
            created_at TEXT DEFAULT ({default_now})
        )
    """)

    # API Keys table
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            encryption_salt TEXT,
            is_active {bool_type} DEFAULT {bool_default},
            created_at TEXT DEFAULT ({default_now}),
            last_used_at TEXT
        )
    """)

    # Subscriptions table
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id TEXT PRIMARY KEY,
            user_id TEXT UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            tier TEXT DEFAULT 'free',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            device_limit INTEGER DEFAULT 3,
            memory_limit INTEGER DEFAULT 10000,
            status TEXT DEFAULT 'active',
            current_period_end TEXT,
            created_at TEXT DEFAULT ({default_now}),
            updated_at TEXT DEFAULT ({default_now})
        )
    """)

    # Usage tracking table
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS usage (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            device_count INTEGER DEFAULT 0,
            memory_count INTEGER DEFAULT 0,
            last_sync_at TEXT,
            updated_at TEXT DEFAULT ({default_now})
        )
    """)

    # Create indexes for common queries
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe ON subscriptions(stripe_customer_id)"
    )


def downgrade() -> None:
    """Remove auth and billing tables."""
    op.execute("DROP TABLE IF EXISTS usage")
    op.execute("DROP TABLE IF EXISTS subscriptions")
    op.execute("DROP TABLE IF EXISTS api_keys")
    op.execute("DROP TABLE IF EXISTS users")
