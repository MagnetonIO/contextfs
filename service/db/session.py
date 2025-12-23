"""Async database session management for sync service.

Provides async SQLAlchemy session factory for use with FastAPI.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from service.db.models import Base

# Module-level engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    """Get async database URL from environment.

    Converts postgresql:// to postgresql+asyncpg:// for async driver.
    """
    url = os.environ.get(
        "CONTEXTFS_POSTGRES_URL",
        "postgresql://contextfs:contextfs@localhost:5433/contextfs_sync",
    )

    # Convert to async driver URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    return url


async def init_db(url: str | None = None) -> AsyncEngine:
    """Initialize database engine and session factory.

    Args:
        url: Database URL (uses environment variable if not provided)

    Returns:
        Async database engine
    """
    global _engine, _session_factory

    if _engine is None:
        db_url = url or get_database_url()
        _engine = create_async_engine(
            db_url,
            echo=os.environ.get("CONTEXTFS_DB_ECHO", "").lower() == "true",
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _engine


async def create_tables() -> None:
    """Create all database tables."""
    global _engine

    if _engine is None:
        await init_db()

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """Drop all database tables (use with caution!)."""
    global _engine

    if _engine is None:
        await init_db()

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
            await session.commit()
    """
    global _session_factory

    if _session_factory is None:
        await init_db()

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session.

    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session_dependency)):
            ...
    """
    async with get_session() as session:
        yield session


async def close_db() -> None:
    """Close database engine and cleanup connections."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
