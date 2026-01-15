"""
Alembic environment configuration for ContextFS Sync Service.

Uses PostgreSQL via CONTEXTFS_POSTGRES_URL environment variable.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Alembic Config object
config = context.config

# Set up logging if config file exists
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_url() -> str:
    """Get PostgreSQL database URL from config or environment."""
    # First try to get from alembic config (set by runner.py)
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url

    # Fallback to environment variable
    url = os.getenv("CONTEXTFS_POSTGRES_URL")
    if not url:
        raise ValueError("CONTEXTFS_POSTGRES_URL environment variable required")

    # Convert async URL to sync for Alembic (it uses sync connections)
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")

    return url


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL script without connecting to database.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Connects to database and runs migrations.
    """
    url = get_url()
    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
