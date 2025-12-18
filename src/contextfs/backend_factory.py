"""
Backend Factory for ContextFS.

Automatically creates and configures the appropriate storage backends
based on the CONTEXTFS_BACKEND environment variable.

Supported backends:
- sqlite: SQLite + ChromaDB (default, local) - Best for local development
- postgres: PostgreSQL with pgvector (unified) - Best for hosted/production
- sqlite+falkordb: SQLite + ChromaDB + FalkorDB graph - Local with advanced graph
- postgres+falkordb: PostgreSQL + FalkorDB graph - Production with advanced graph

Architecture:
- SQLite backends use StorageRouter (SQLite + ChromaDB + optional FalkorDB)
- Postgres backends use PostgresUnifiedBackend directly (single database)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from contextfs.config import BackendType, Config, get_config
from contextfs.storage_protocol import GraphBackend

if TYPE_CHECKING:
    from contextfs.postgres_backend import PostgresUnifiedBackend
    from contextfs.storage_router import StorageRouter

logger = logging.getLogger(__name__)


def create_storage_backend(
    config: Config | None = None,
) -> StorageRouter | PostgresUnifiedBackend:
    """
    Create the appropriate storage backend based on configuration.

    Args:
        config: Optional Config object. Uses global config if not provided.

    Returns:
        StorageBackend instance (StorageRouter for SQLite, PostgresUnifiedBackend for Postgres)

    Raises:
        ImportError: If required dependencies not installed
        ConnectionError: If database connection fails
    """
    if config is None:
        config = get_config()

    backend_type = config.backend
    logger.info(f"Initializing storage backend: {backend_type.value}")

    if backend_type == BackendType.SQLITE:
        return _create_sqlite_backend(config)

    elif backend_type == BackendType.POSTGRES:
        return _create_postgres_backend(config)

    elif backend_type == BackendType.SQLITE_FALKORDB:
        return _create_sqlite_falkordb_backend(config)

    elif backend_type == BackendType.POSTGRES_FALKORDB:
        return _create_postgres_falkordb_backend(config)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def _create_sqlite_backend(config: Config) -> StorageRouter:
    """Create SQLite + ChromaDB backend."""
    from contextfs.rag import RAGBackend
    from contextfs.storage_router import StorageRouter

    # Ensure data directory exists
    data_dir = Path(config.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / config.sqlite_filename

    # Create RAG backend
    rag = RAGBackend(
        data_dir=data_dir,
        embedding_model=config.embedding_model,
        collection_name=config.chroma_collection,
    )

    # Create router without graph
    router = StorageRouter(
        db_path=db_path,
        rag_backend=rag,
        graph_backend=None,
    )

    logger.info(f"SQLite backend initialized: {db_path}")
    return router


def _create_postgres_backend(config: Config) -> PostgresUnifiedBackend:
    """
    Create PostgreSQL unified backend.

    Uses PostgreSQL with pgvector for ALL operations:
    - Memory storage
    - Semantic search (pgvector)
    - Graph traversal (recursive CTEs)
    - Full-text search (tsvector)

    No ChromaDB or other dependencies needed - single database solution.
    """
    from contextfs.postgres_backend import PostgresUnifiedBackend

    # Create unified Postgres backend - no other dependencies needed
    postgres = PostgresUnifiedBackend(
        connection_string=config.postgres_url,
        embedding_model=config.embedding_model,
    )

    logger.info(f"PostgreSQL unified backend initialized: {_mask_password(config.postgres_url)}")
    return postgres


def _mask_password(url: str) -> str:
    """Mask password in connection URL for logging."""
    if "@" in url:
        parts = url.split("@")
        creds = parts[0].rsplit(":", 1)
        if len(creds) > 1:
            return f"{creds[0]}:****@{parts[1]}"
    return url


def _create_sqlite_falkordb_backend(config: Config) -> StorageRouter:
    """Create SQLite + ChromaDB + FalkorDB backend."""
    from contextfs.graph_backend import FalkorDBBackend
    from contextfs.rag import RAGBackend
    from contextfs.storage_router import StorageRouter

    # Ensure data directory exists
    data_dir = Path(config.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / config.sqlite_filename

    # Create RAG backend
    rag = RAGBackend(
        data_dir=data_dir,
        embedding_model=config.embedding_model,
        collection_name=config.chroma_collection,
    )

    # Create FalkorDB graph backend
    graph: GraphBackend | None = None
    if config.falkordb_enabled:
        try:
            graph = FalkorDBBackend(
                host=config.falkordb_host,
                port=config.falkordb_port,
                password=config.falkordb_password,
                graph_name=config.falkordb_graph_name,
            )
            logger.info(f"FalkorDB connected: {config.falkordb_host}:{config.falkordb_port}")
        except Exception as e:
            logger.warning(f"FalkorDB connection failed, using SQLite fallback: {e}")
            graph = None

    # Create router with graph
    router = StorageRouter(
        db_path=db_path,
        rag_backend=rag,
        graph_backend=graph,
    )

    logger.info(f"SQLite+FalkorDB backend initialized: {db_path}")
    return router


def _create_postgres_falkordb_backend(config: Config) -> PostgresUnifiedBackend:
    """
    Create PostgreSQL + FalkorDB backend.

    Uses PostgreSQL for storage/search and FalkorDB for advanced graph operations.
    Falls back to PostgreSQL recursive CTEs if FalkorDB unavailable.
    """
    from contextfs.postgres_backend import PostgresUnifiedBackend

    # Create Postgres backend (handles storage, search, and graph fallback)
    postgres = PostgresUnifiedBackend(
        connection_string=config.postgres_url,
        embedding_model=config.embedding_model,
    )

    # Optionally connect FalkorDB for advanced graph features
    if config.falkordb_enabled:
        try:
            from contextfs.graph_backend import FalkorDBBackend

            graph = FalkorDBBackend(
                host=config.falkordb_host,
                port=config.falkordb_port,
                password=config.falkordb_password,
                graph_name=config.falkordb_graph_name,
            )
            # Attach FalkorDB as graph backend
            postgres._graph = graph
            logger.info(f"FalkorDB connected: {config.falkordb_host}:{config.falkordb_port}")
        except Exception as e:
            logger.warning(f"FalkorDB connection failed, using PostgreSQL graph (CTEs): {e}")

    logger.info(f"PostgreSQL+FalkorDB backend initialized: {_mask_password(config.postgres_url)}")
    return postgres


def get_backend_info(config: Config | None = None) -> dict:
    """
    Get information about the configured backend without initializing it.

    Returns:
        Dict with backend configuration details
    """
    if config is None:
        config = get_config()

    info = {
        "backend_type": config.backend.value,
        "data_dir": str(config.data_dir),
    }

    if config.backend in (BackendType.SQLITE, BackendType.SQLITE_FALKORDB):
        info["sqlite_file"] = str(Path(config.data_dir) / config.sqlite_filename)
        info["chroma_collection"] = config.chroma_collection

    if config.backend in (BackendType.POSTGRES, BackendType.POSTGRES_FALKORDB):
        # Mask password in URL
        url = config.postgres_url
        if "@" in url:
            parts = url.split("@")
            creds = parts[0].rsplit(":", 1)
            if len(creds) > 1:
                url = f"{creds[0]}:****@{parts[1]}"
        info["postgres_url"] = url

    if config.backend in (BackendType.SQLITE_FALKORDB, BackendType.POSTGRES_FALKORDB):
        info["falkordb_enabled"] = config.falkordb_enabled
        info["falkordb_host"] = config.falkordb_host
        info["falkordb_port"] = config.falkordb_port
        info["falkordb_graph"] = config.falkordb_graph_name

    info["embedding_model"] = config.embedding_model
    info["lineage_auto_track"] = config.lineage_auto_track
    info["lineage_merge_strategy"] = config.lineage_merge_strategy.value

    return info
