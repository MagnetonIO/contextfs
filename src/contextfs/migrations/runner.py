"""
Migration runner for ContextFS.

Handles automatic migration on startup.
"""

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def get_alembic_config(db_path: Path) -> Config:
    """Create Alembic config for the given database."""
    migrations_dir = Path(__file__).parent

    config = Config()
    config.set_main_option("script_location", str(migrations_dir))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

    return config


def get_current_revision(db_path: Path) -> str | None:
    """Get current database revision."""
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        return context.get_current_revision()


def get_head_revision(config: Config) -> str | None:
    """Get head revision from migration scripts."""
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def run_migrations(db_path: Path) -> bool:
    """
    Run pending migrations on the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        True if migrations were run, False if already up to date
    """
    config = get_alembic_config(db_path)

    current = get_current_revision(db_path)
    head = get_head_revision(config)

    if current == head:
        logger.debug(f"Database at revision {current}, up to date")
        return False

    logger.info(f"Running migrations: {current} -> {head}")
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


def stamp_database(db_path: Path, revision: str = "head") -> None:
    """
    Stamp database with a revision without running migrations.

    Useful for marking existing databases as migrated.

    Args:
        db_path: Path to SQLite database
        revision: Revision to stamp (default: head)
    """
    config = get_alembic_config(db_path)
    command.stamp(config, revision)
