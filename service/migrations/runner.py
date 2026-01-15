"""
Migration runner for ContextFS Sync Service.

Handles automatic migration on startup for PostgreSQL.
"""

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get PostgreSQL database URL from environment.

    Returns:
        SQLAlchemy database URL (sync version for Alembic)
    """
    url = os.getenv("CONTEXTFS_POSTGRES_URL")
    if not url:
        raise ValueError("CONTEXTFS_POSTGRES_URL environment variable required")

    # Convert async URL to sync for Alembic
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")

    return url


def get_alembic_config(url: str | None = None) -> Config:
    """Create Alembic config for the service database.

    Args:
        url: Direct database URL (uses env var if not provided)

    Returns:
        Alembic Config object
    """
    migrations_dir = Path(__file__).parent

    config = Config()
    config.set_main_option("script_location", str(migrations_dir))
    config.set_main_option("sqlalchemy.url", url or get_database_url())

    return config


def get_current_revision(url: str | None = None) -> str | None:
    """Get current database revision.

    Args:
        url: Direct database URL

    Returns:
        Current revision string or None
    """
    db_url = url or get_database_url()
    engine = create_engine(db_url)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        return context.get_current_revision()


def get_head_revision(config: Config) -> str | None:
    """Get head revision from migration scripts."""
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def _tables_exist(url: str | None = None) -> bool:
    """Check if core tables already exist in the database."""
    db_url = url or get_database_url()
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Check if the 'users' table exists (a core table from initial migration)
        result = conn.execute(
            text(
                "SELECT EXISTS (SELECT FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'users')"
            )
        )
        return result.scalar()


def run_migrations(url: str | None = None) -> bool:
    """
    Run pending migrations on the database.

    Handles the case where tables exist from old raw SQL migrations
    but Alembic version tracking doesn't exist - stamps at head instead
    of trying to create tables that already exist.

    Args:
        url: Direct database URL

    Returns:
        True if migrations were run, False if already up to date
    """
    config = get_alembic_config(url)

    try:
        current = get_current_revision(url)
    except Exception as e:
        # If alembic_version table doesn't exist, we need to stamp first
        logger.info(f"No migration history found, initializing: {e}")
        current = None

    head = get_head_revision(config)

    if current == head:
        logger.debug(f"Database at revision {current}, up to date")
        return False

    # If no Alembic version but tables exist, stamp at head
    # (migrated from old raw SQL migrations)
    if current is None and _tables_exist(url):
        logger.info("Tables exist from prior migrations, stamping at head")
        command.stamp(config, "head")
        return True

    logger.info(f"Running migrations: {current} -> {head}")
    command.upgrade(config, "head")

    return True


def stamp_database(revision: str = "head", url: str | None = None) -> None:
    """
    Stamp database with a revision without running migrations.

    Useful for marking existing databases as migrated.

    Args:
        revision: Revision to stamp (default: head)
        url: Direct database URL
    """
    config = get_alembic_config(url)
    command.stamp(config, revision)
    logger.info(f"Database stamped at revision: {revision}")
