"""
Migration runner for ContextFS.

Handles automatic migration on startup.
Supports both SQLite (local) and PostgreSQL (production).
"""

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def get_database_url(db_path: Path | None = None) -> str:
    """Get database URL based on configuration.

    Uses CONTEXTFS_BACKEND env var to determine backend:
    - sqlite (default): Uses SQLite at db_path or ~/.contextfs/context.db
    - postgres: Uses CONTEXTFS_POSTGRES_URL

    Args:
        db_path: Path to SQLite database (for sqlite backend)

    Returns:
        SQLAlchemy database URL
    """
    backend = os.getenv("CONTEXTFS_BACKEND", "sqlite")

    if backend in ("postgres", "postgres+falkordb"):
        url = os.getenv("CONTEXTFS_POSTGRES_URL")
        if not url:
            raise ValueError(
                "CONTEXTFS_POSTGRES_URL environment variable required for postgres backend"
            )
        return url
    else:
        if db_path is None:
            data_dir = Path(os.getenv("CONTEXTFS_DATA_DIR", "~/.contextfs")).expanduser()
            db_path = data_dir / "context.db"
        return f"sqlite:///{db_path}"


def get_alembic_config(db_path: Path | None = None, url: str | None = None) -> Config:
    """Create Alembic config for the given database.

    Args:
        db_path: Path to SQLite database (ignored if url provided)
        url: Direct database URL (overrides db_path)

    Returns:
        Alembic Config object
    """
    migrations_dir = Path(__file__).parent

    config = Config()
    config.set_main_option("script_location", str(migrations_dir))

    if url:
        config.set_main_option("sqlalchemy.url", url)
    else:
        config.set_main_option("sqlalchemy.url", get_database_url(db_path))

    return config


def get_current_revision(db_path: Path | None = None, url: str | None = None) -> str | None:
    """Get current database revision.

    Args:
        db_path: Path to SQLite database (ignored if url provided)
        url: Direct database URL

    Returns:
        Current revision string or None
    """
    db_url = url or get_database_url(db_path)
    engine = create_engine(db_url)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        return context.get_current_revision()


def get_head_revision(config: Config) -> str | None:
    """Get head revision from migration scripts."""
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def run_migrations(db_path: Path | None = None, url: str | None = None) -> bool:
    """
    Run pending migrations on the database.

    Args:
        db_path: Path to SQLite database (ignored if url provided)
        url: Direct database URL

    Returns:
        True if migrations were run, False if already up to date
    """
    config = get_alembic_config(db_path, url)

    current = get_current_revision(db_path, url)
    head = get_head_revision(config)

    if current == head:
        logger.debug(f"Database at revision {current}, up to date")
        return False

    db_url = url or get_database_url(db_path)
    backend_type = "postgres" if "postgresql" in db_url else "sqlite"
    logger.info(f"Running migrations on {backend_type}: {current} -> {head}")
    command.upgrade(config, "head")

    return True


def create_migration(message: str) -> str:
    """
    Create a new migration script.

    Args:
        message: Migration description

    Returns:
        Path to created migration file
    """
    # Use a temporary config pointing to a dummy db
    config = get_alembic_config(Path("/tmp/contextfs_migration.db"))

    return command.revision(config, message=message, autogenerate=False)


def stamp_database(
    db_path: Path | None = None, revision: str = "head", url: str | None = None
) -> None:
    """
    Stamp database with a revision without running migrations.

    Useful for marking existing databases as migrated.

    Args:
        db_path: Path to SQLite database (ignored if url provided)
        revision: Revision to stamp (default: head)
        url: Direct database URL
    """
    config = get_alembic_config(db_path, url)
    command.stamp(config, revision)
