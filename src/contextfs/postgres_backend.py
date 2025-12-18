"""
PostgreSQL Unified Backend for ContextFS.

Provides a single-database solution using PostgreSQL with:
- pgvector for semantic search
- Recursive CTEs for graph traversal (lineage)
- Full-text search via tsvector

This is an alternative to the SQLite + ChromaDB + FalkorDB multi-backend setup.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from contextfs.schemas import Memory, MemoryType, SearchResult
from contextfs.storage_protocol import (
    EdgeRelation,
    GraphTraversalResult,
    MemoryEdge,
    StorageBackend,
    StorageCapabilities,
)

if TYPE_CHECKING:
    from psycopg import Connection
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Capabilities for Postgres unified backend
POSTGRES_UNIFIED_CAPABILITIES = StorageCapabilities(
    semantic_search=True,
    full_text_search=True,
    persistent=True,
    syncable=False,
    batch_operations=True,
    transactions=True,
    graph_traversal=True,
    memory_lineage=True,
    path_finding=False,  # Complex path finding not efficient in SQL
)


class PostgresUnifiedBackend(StorageBackend):
    """
    Unified PostgreSQL backend implementing StorageBackend protocol.

    Uses pgvector for semantic search and recursive CTEs for graph operations.
    All data in a single database for simpler operations.

    Attributes:
        capabilities: StorageCapabilities descriptor
    """

    capabilities: StorageCapabilities = POSTGRES_UNIFIED_CAPABILITIES

    def __init__(
        self,
        connection_string: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize PostgreSQL backend.

        Args:
            connection_string: PostgreSQL connection URL
            embedding_model: Sentence transformer model for embeddings
        """
        self._connection_string = connection_string
        self._embedding_model_name = embedding_model
        self._conn: Connection | None = None
        self._embedder: SentenceTransformer | None = None
        self._initialized = False
        # Optional FalkorDB graph backend for advanced graph operations
        # If set, graph operations use FalkorDB instead of PostgreSQL CTEs
        self._graph: Any | None = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize database connection and embedder."""
        if self._initialized:
            return

        try:
            import psycopg
            from sentence_transformers import SentenceTransformer

            # Connect to PostgreSQL
            self._conn = psycopg.connect(self._connection_string)
            self._conn.autocommit = False

            # Initialize embedding model
            self._embedder = SentenceTransformer(self._embedding_model_name)

            self._initialized = True
            logger.info("PostgreSQL backend initialized")

        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires: pip install psycopg[binary] sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL backend: {e}")
            raise

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        self._ensure_initialized()
        if not self._embedder:
            raise RuntimeError("Embedder not initialized")
        return self._embedder.encode(text).tolist()

    # =========================================================================
    # StorageBackend Protocol Implementation
    # =========================================================================

    def save(self, memory: Memory) -> Memory:
        """Save a memory to PostgreSQL."""
        self._ensure_initialized()
        if not self._conn:
            raise RuntimeError("Database not connected")

        # Generate embedding
        embedding = self._get_embedding(memory.content)

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    id, content, type, tags, summary, namespace_id,
                    source_file, source_repo, source_tool, project, session_id,
                    created_at, updated_at, metadata, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                )
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    type = EXCLUDED.type,
                    tags = EXCLUDED.tags,
                    summary = EXCLUDED.summary,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    memory.id,
                    memory.content,
                    memory.type.value,
                    memory.tags,
                    memory.summary,
                    memory.namespace_id,
                    memory.source_file,
                    memory.source_repo,
                    memory.source_tool,
                    memory.project,
                    memory.session_id,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    json.dumps(memory.metadata),
                    embedding,
                ),
            )
            self._conn.commit()

        return memory

    def save_batch(self, memories: list[Memory]) -> int:
        """Save multiple memories in batch."""
        self._ensure_initialized()
        if not self._conn or not memories:
            return 0

        # Generate all embeddings
        contents = [m.content for m in memories]
        embeddings = self._embedder.encode(contents).tolist() if self._embedder else []

        with self._conn.cursor() as cur:
            for i, memory in enumerate(memories):
                embedding = embeddings[i] if i < len(embeddings) else None
                cur.execute(
                    """
                    INSERT INTO memories (
                        id, content, type, tags, summary, namespace_id,
                        source_file, source_repo, source_tool, project, session_id,
                        created_at, updated_at, metadata, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        memory.id,
                        memory.content,
                        memory.type.value,
                        memory.tags,
                        memory.summary,
                        memory.namespace_id,
                        memory.source_file,
                        memory.source_repo,
                        memory.source_tool,
                        memory.project,
                        memory.session_id,
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat(),
                        json.dumps(memory.metadata),
                        embedding,
                    ),
                )
            self._conn.commit()

        return len(memories)

    def recall(self, memory_id: str) -> Memory | None:
        """Recall a memory by ID."""
        self._ensure_initialized()
        if not self._conn:
            return None

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, type, tags, summary, namespace_id,
                       source_file, source_repo, source_tool, project, session_id,
                       created_at, updated_at, metadata
                FROM memories
                WHERE id LIKE %s
                LIMIT 1
                """,
                (f"{memory_id}%",),
            )
            row = cur.fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    def search(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        namespace_id: str | None = None,
        project: str | None = None,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """Search memories using semantic similarity."""
        self._ensure_initialized()
        if not self._conn:
            return []

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        with self._conn.cursor() as cur:
            # Build dynamic query with filters
            filters = []
            params: list[Any] = [query_embedding, min_score]

            if namespace_id:
                filters.append("namespace_id = %s")
                params.append(namespace_id)
            if type:
                filters.append("type = %s")
                params.append(type.value)
            if project:
                filters.append("project = %s")
                params.append(project)
            if tags:
                filters.append("tags && %s")
                params.append(tags)

            where_clause = " AND ".join(filters) if filters else "TRUE"
            params.append(limit)

            cur.execute(
                f"""
                SELECT
                    id, content, type, tags, summary, namespace_id,
                    source_file, source_repo, source_tool, project, session_id,
                    created_at, updated_at, metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= %s
                  AND {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [query_embedding, query_embedding, min_score]
                + params[2:-1]
                + [query_embedding, limit],
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            memory = self._row_to_memory(row[:-1])
            similarity = row[-1]
            results.append(SearchResult(memory=memory, score=similarity, source="pgvector"))

        return results

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        self._ensure_initialized()
        if not self._conn:
            return False

        with self._conn.cursor() as cur:
            # First find full ID
            cur.execute("SELECT id FROM memories WHERE id LIKE %s LIMIT 1", (f"{memory_id}%",))
            row = cur.fetchone()
            if not row:
                return False

            full_id = row[0]
            cur.execute("DELETE FROM memories WHERE id = %s", (full_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_by_namespace(self, namespace_id: str) -> int:
        """Delete all memories in a namespace."""
        self._ensure_initialized()
        if not self._conn:
            return 0

        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE namespace_id = %s", (namespace_id,))
            count = cur.rowcount
            self._conn.commit()
            return count

    # =========================================================================
    # Graph Operations (via SQL)
    # =========================================================================

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Create an edge between two memories (syncs to FalkorDB if attached)."""
        self._ensure_initialized()
        if not self._conn:
            raise RuntimeError("Database not connected")

        # Also sync to FalkorDB if available
        if self._graph:
            try:
                self._graph.add_edge(from_id, to_id, relation, weight, metadata)
            except Exception:
                pass  # Continue with PostgreSQL

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_edges (from_id, to_id, relation, weight, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (from_id, to_id, relation) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    metadata = EXCLUDED.metadata
                """,
                (from_id, to_id, relation.value, weight, json.dumps(metadata or {})),
            )
            self._conn.commit()

        return MemoryEdge(
            from_id=from_id,
            to_id=to_id,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )

    def remove_edge(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation | None = None,
    ) -> bool:
        """Remove edge(s) between memories."""
        self._ensure_initialized()
        if not self._conn:
            return False

        with self._conn.cursor() as cur:
            if relation:
                cur.execute(
                    "DELETE FROM memory_edges WHERE from_id = %s AND to_id = %s AND relation = %s",
                    (from_id, to_id, relation.value),
                )
            else:
                cur.execute(
                    "DELETE FROM memory_edges WHERE from_id = %s AND to_id = %s",
                    (from_id, to_id),
                )
            self._conn.commit()
            return cur.rowcount > 0

    def get_edges(
        self,
        memory_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        relation: EdgeRelation | None = None,
    ) -> list[MemoryEdge]:
        """Get all edges connected to a memory."""
        self._ensure_initialized()
        if not self._conn:
            return []

        with self._conn.cursor() as cur:
            if direction == "outgoing":
                query = "SELECT from_id, to_id, relation, weight, created_at, metadata FROM memory_edges WHERE from_id = %s"
            elif direction == "incoming":
                query = "SELECT from_id, to_id, relation, weight, created_at, metadata FROM memory_edges WHERE to_id = %s"
            else:
                query = "SELECT from_id, to_id, relation, weight, created_at, metadata FROM memory_edges WHERE from_id = %s OR to_id = %s"

            params = [memory_id] if direction != "both" else [memory_id, memory_id]

            if relation:
                query += " AND relation = %s"
                params.append(relation.value)

            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            MemoryEdge(
                from_id=row[0],
                to_id=row[1],
                relation=EdgeRelation(row[2]),
                weight=row[3],
                created_at=datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4],
                metadata=json.loads(row[5]) if isinstance(row[5], str) else row[5],
            )
            for row in rows
        ]

    def get_related(
        self,
        memory_id: str,
        relation: EdgeRelation | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        max_depth: int = 1,
        min_weight: float = 0.0,
    ) -> list[GraphTraversalResult]:
        """Get related memories via graph traversal (or FalkorDB if attached)."""
        # Use FalkorDB if available
        if self._graph:
            try:
                return self._graph.get_related(
                    memory_id=memory_id,
                    relation=relation,
                    direction=direction,
                    max_depth=max_depth,
                    min_weight=min_weight,
                )
            except Exception:
                pass  # Fall back to PostgreSQL

        self._ensure_initialized()
        if not self._conn:
            return []

        # Use recursive CTE for traversal
        relation_filter = f"AND e.relation = '{relation.value}'" if relation else ""

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                WITH RECURSIVE traversal AS (
                    SELECT
                        e.to_id AS memory_id,
                        e.relation,
                        1 AS depth,
                        e.weight AS path_weight
                    FROM memory_edges e
                    WHERE e.from_id = %s
                      AND e.weight >= %s
                      {relation_filter}

                    UNION ALL

                    SELECT
                        e.to_id,
                        e.relation,
                        t.depth + 1,
                        t.path_weight * e.weight
                    FROM memory_edges e
                    JOIN traversal t ON e.from_id = t.memory_id
                    WHERE t.depth < %s
                      AND e.weight >= %s
                      {relation_filter}
                )
                SELECT DISTINCT ON (t.memory_id)
                    m.*, t.relation, t.depth, t.path_weight
                FROM traversal t
                JOIN memories m ON m.id = t.memory_id
                ORDER BY t.memory_id, t.depth
                """,
                (memory_id, min_weight, max_depth, min_weight),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            # Memory columns + relation, depth, path_weight
            memory = self._row_to_memory(row[:-3])
            results.append(
                GraphTraversalResult(
                    memory=memory,
                    relation=EdgeRelation(row[-3]),
                    depth=row[-2],
                    path_weight=row[-1],
                )
            )

        return results

    def has_graph(self) -> bool:
        """Check if graph operations are available."""
        return True  # Postgres always supports graph via CTEs

    def get_lineage(
        self,
        memory_id: str,
        direction: Literal["ancestors", "descendants", "both"] = "both",
    ) -> dict[str, Any]:
        """Get memory lineage using recursive CTEs (or FalkorDB if attached)."""
        # Use FalkorDB if available
        if self._graph:
            try:
                return self._graph.get_lineage(memory_id, direction)
            except Exception:
                pass  # Fall back to PostgreSQL

        self._ensure_initialized()
        if not self._conn:
            return {"root": memory_id, "ancestors": [], "descendants": []}

        result: dict[str, Any] = {
            "root": memory_id,
            "ancestors": [],
            "descendants": [],
        }

        with self._conn.cursor() as cur:
            # Get ancestors
            if direction in ("ancestors", "both"):
                cur.execute(
                    "SELECT * FROM get_memory_ancestors(%s, 10)",
                    (memory_id,),
                )
                for row in cur.fetchall():
                    result["ancestors"].append(
                        {
                            "memory_id": row[0],
                            "relation": row[1],
                            "depth": row[2],
                        }
                    )

            # Get descendants
            if direction in ("descendants", "both"):
                cur.execute(
                    "SELECT * FROM get_memory_descendants(%s, 10)",
                    (memory_id,),
                )
                for row in cur.fetchall():
                    result["descendants"].append(
                        {
                            "memory_id": row[0],
                            "relation": row[1],
                            "depth": row[2],
                        }
                    )

            # Find root
            if result["ancestors"]:
                result["root"] = result["ancestors"][-1]["memory_id"]

        return result

    def sync_node(self, memory: Memory) -> bool:
        """Sync a memory node (no-op for unified backend)."""
        # In unified backend, save() handles everything
        return True

    def delete_node(self, memory_id: str) -> bool:
        """Delete a memory and its edges."""
        return self.delete(memory_id)  # CASCADE handles edges

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_memory(self, row: tuple) -> Memory:
        """Convert a database row to Memory object."""
        return Memory(
            id=row[0],
            content=row[1],
            type=MemoryType(row[2]),
            tags=row[3] if isinstance(row[3], list) else [],
            summary=row[4],
            namespace_id=row[5],
            source_file=row[6],
            source_repo=row[7],
            source_tool=row[8],
            project=row[9],
            session_id=row[10],
            created_at=datetime.fromisoformat(row[11]) if isinstance(row[11], str) else row[11],
            updated_at=datetime.fromisoformat(row[12]) if isinstance(row[12], str) else row[12],
            metadata=json.loads(row[13]) if isinstance(row[13], str) else row[13],
        )

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        self._ensure_initialized()
        if not self._conn:
            return {"connected": False}

        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            memory_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memory_edges")
            edge_count = cur.fetchone()[0]

        return {
            "connected": True,
            "backend": "postgres",
            "memories": memory_count,
            "edges": edge_count,
            "embedding_model": self._embedding_model_name,
        }

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False
