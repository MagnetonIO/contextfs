"""Add password authentication fields to users table.

Revision ID: 009
Revises: 008
Create Date: 2025-01-09

Adds email/password authentication support alongside OAuth.
Supports both SQLite and PostgreSQL.
"""

from alembic import context, op

# Revision identifiers
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def get_dialect() -> str:
    """Get the database dialect name."""
    return context.get_context().dialect.name


def upgrade() -> None:
    """Add password auth fields to users table."""
    dialect = get_dialect()

    # Boolean type differs between databases
    if dialect == "postgresql":
        bool_type = "BOOLEAN"
        bool_false = "false"
        bool_true = "true"
    else:  # sqlite
        bool_type = "INTEGER"
        bool_false = "0"
        bool_true = "1"

    # Add password hash column
    op.execute("""
        ALTER TABLE users ADD COLUMN password_hash TEXT
    """)

    # Add email verification columns
    op.execute(f"""
        ALTER TABLE users ADD COLUMN email_verified {bool_type} DEFAULT {bool_false}
    """)

    op.execute("""
        ALTER TABLE users ADD COLUMN verification_token TEXT
    """)

    op.execute("""
        ALTER TABLE users ADD COLUMN verification_token_expires TEXT
    """)

    # Add password reset columns
    op.execute("""
        ALTER TABLE users ADD COLUMN reset_token TEXT
    """)

    op.execute("""
        ALTER TABLE users ADD COLUMN reset_token_expires TEXT
    """)

    # Add last login tracking
    op.execute("""
        ALTER TABLE users ADD COLUMN last_login TEXT
    """)

    # Add active status
    op.execute(f"""
        ALTER TABLE users ADD COLUMN is_active {bool_type} DEFAULT {bool_true}
    """)

    # Create index for verification token lookups
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_verification_token ON users(verification_token)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token)")


def downgrade() -> None:
    """Remove password auth fields - SQLite doesn't support DROP COLUMN easily."""
    # SQLite doesn't support DROP COLUMN in older versions
    # Would need to recreate the table, skipping for now
    pass
