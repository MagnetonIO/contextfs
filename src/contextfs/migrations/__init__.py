"""
Database migrations for ContextFS.

Uses Alembic for SQLite schema management.
"""

from contextfs.migrations.runner import run_migrations

__all__ = ["run_migrations"]
